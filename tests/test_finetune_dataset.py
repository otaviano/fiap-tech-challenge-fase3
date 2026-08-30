"""Preparação dos tensores de treino — a máscara de labels é o ponto crítico."""

import pytest

from med_assist.config import TRAIN_FILE, VAL_FILE
from med_assist.finetune.dataset import IGNORE_INDEX, collate, encode, estatisticas, load_jsonl

transformers = pytest.importorskip("transformers", reason="stack de fine-tuning não instalada")


@pytest.fixture(scope="module")
def tokenizer():
    """Tokenizer leve com chat template — evita baixar o modelo inteiro."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception as exc:  # sem rede/cache
        pytest.skip(f"tokenizer indisponível: {exc}")


@pytest.fixture(scope="module")
def registros():
    return load_jsonl(TRAIN_FILE)


def test_dataset_esta_no_formato_de_chat(registros):
    assert len(registros) > 200
    for registro in registros[:20]:
        papeis = [m["role"] for m in registro["messages"]]
        assert papeis == ["system", "user", "assistant"]
        assert registro["categoria"]


def test_todas_as_categorias_do_projeto_estao_representadas(registros):
    categorias = {r["categoria"] for r in registros}
    assert {"protocolo_qa", "faq", "documento", "limite_de_atuacao"} <= categorias


def test_validacao_nao_vaza_para_o_treino(registros):
    perguntas_treino = {r["messages"][1]["content"] for r in registros}
    perguntas_val = {r["messages"][1]["content"] for r in load_jsonl(VAL_FILE)}
    assert perguntas_treino & perguntas_val == set()


def test_loss_so_incide_sobre_a_resposta(tokenizer, registros):
    exemplo = encode(tokenizer, registros[0], max_len=1024)
    assert exemplo is not None
    assert exemplo["labels"][0] == IGNORE_INDEX, "o prompt precisa estar mascarado"
    supervisionados = [t for t in exemplo["labels"] if t != IGNORE_INDEX]
    assert 0 < len(supervisionados) < len(exemplo["labels"])
    # Os tokens supervisionados correspondem exatamente ao fim da sequência (a resposta).
    corte = len(exemplo["labels"]) - len(supervisionados)
    assert exemplo["labels"][corte:] == exemplo["input_ids"][corte:]


def test_collate_alinha_padding_mascara_e_labels(tokenizer, registros):
    exemplos = [encode(tokenizer, r, max_len=512) for r in registros[:4]]
    exemplos = [e for e in exemplos if e]
    lote = collate(exemplos, tokenizer.pad_token_id or 0)
    assert lote["input_ids"].shape == lote["labels"].shape == lote["attention_mask"].shape
    for i, exemplo in enumerate(exemplos):
        n = len(exemplo["input_ids"])
        assert lote["attention_mask"][i][:n].all()
        assert not lote["attention_mask"][i][n:].any()
        assert (lote["labels"][i][n:] == IGNORE_INDEX).all()


def test_estatisticas_do_dataset(tokenizer, registros):
    exemplos = [e for r in registros[:30] if (e := encode(tokenizer, r, max_len=1024))]
    stats = estatisticas(exemplos)
    assert stats["exemplos"] == len(exemplos)
    assert stats["tokens_p50"] <= stats["tokens_p95"] <= stats["tokens_max"]
