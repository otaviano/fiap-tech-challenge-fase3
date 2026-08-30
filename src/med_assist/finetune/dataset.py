"""Preparação dos tensores de treino a partir do dataset de chat.

Dois cuidados que definem a qualidade do fine-tuning:

1. **Chat template do próprio modelo.** As mensagens são renderizadas com
   `tokenizer.apply_chat_template`, então o formato de treino é exatamente o que
   o servidor de inferência usa depois. Formato divergente entre treino e
   inferência é a causa mais comum de "treinei e piorou".

2. **Loss só na resposta.** Os tokens do system e do turno do usuário recebem
   label `-100`: o modelo não é premiado por reproduzir o enunciado da pergunta,
   apenas por produzir a resposta correta. Sem essa máscara, boa parte do sinal
   de treino é gasto decorando prompt.
"""

from __future__ import annotations

import json
from pathlib import Path

IGNORE_INDEX = -100


def load_jsonl(caminho: Path) -> list[dict]:
    return [json.loads(linha) for linha in
            Path(caminho).read_text(encoding="utf-8").splitlines() if linha.strip()]


def render(tokenizer, mensagens: list[dict]) -> tuple[str, str]:
    """Devolve (texto completo, texto só do prompt) segundo o chat template."""
    completo = tokenizer.apply_chat_template(mensagens, tokenize=False, add_generation_prompt=False)
    prompt = tokenizer.apply_chat_template(mensagens[:-1], tokenize=False, add_generation_prompt=True)
    return completo, prompt


def encode(tokenizer, exemplo: dict, max_len: int = 1024) -> dict | None:
    """Tokeniza um exemplo e mascara os labels do prompt."""
    completo, prompt = render(tokenizer, exemplo["messages"])
    ids_completo = tokenizer(completo, truncation=True, max_length=max_len,
                             add_special_tokens=False)["input_ids"]
    ids_prompt = tokenizer(prompt, truncation=True, max_length=max_len,
                           add_special_tokens=False)["input_ids"]

    n_prompt = min(len(ids_prompt), len(ids_completo))
    if len(ids_completo) - n_prompt < 8:
        return None  # resposta truncada quase inteira: não ensina nada

    labels = [IGNORE_INDEX] * n_prompt + ids_completo[n_prompt:]
    return {
        "input_ids": ids_completo,
        "labels": labels,
        "categoria": exemplo.get("categoria", "desconhecida"),
    }


def build_examples(tokenizer, registros: list[dict], max_len: int = 1024) -> list[dict]:
    exemplos = [e for r in registros if (e := encode(tokenizer, r, max_len)) is not None]
    return exemplos


def collate(lote: list[dict], pad_token_id: int):
    """Padding à direita, com máscara de atenção e labels alinhados."""
    import torch

    largura = max(len(e["input_ids"]) for e in lote)
    input_ids, labels, attention = [], [], []
    for exemplo in lote:
        faltam = largura - len(exemplo["input_ids"])
        input_ids.append(exemplo["input_ids"] + [pad_token_id] * faltam)
        labels.append(exemplo["labels"] + [IGNORE_INDEX] * faltam)
        attention.append([1] * len(exemplo["input_ids"]) + [0] * faltam)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
    }


def estatisticas(exemplos: list[dict]) -> dict:
    """Comprimentos e composição — vai para o relatório de treino."""
    from collections import Counter

    tamanhos = sorted(len(e["input_ids"]) for e in exemplos)
    n = len(tamanhos)
    return {
        "exemplos": n,
        "tokens_total": sum(tamanhos),
        "tokens_medio": round(sum(tamanhos) / n, 1) if n else 0,
        "tokens_p50": tamanhos[n // 2] if n else 0,
        "tokens_p95": tamanhos[int(n * 0.95)] if n else 0,
        "tokens_max": tamanhos[-1] if n else 0,
        "por_categoria": dict(sorted(Counter(e["categoria"] for e in exemplos).items())),
    }
