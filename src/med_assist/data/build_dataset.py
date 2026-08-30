"""Pipeline de preparação do dataset de fine-tuning.

    data/raw/  ──anonimização──▶  curadoria  ──split──▶  data/processed/{train,val}.jsonl
                     │                                          └── data/processed/protocolos/  (base RAG)
                     └── results/eval/relatorio_anonimizacao.json
                         results/eval/relatorio_curadoria.json

O dataset final é uma lista de conversas no formato de chat (system/user/assistant),
que é o formato que o `chat template` do modelo consome no treino e o mesmo que a
API OpenAI-compatível recebe na inferência.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import (EVAL_DIR, KB_DIR, RAW_DIR, ROOT, SEED, TRAIN_FILE, VAL_FILE,
                      ensure_dirs)
from ..persona import SYSTEM_PROMPT, SYSTEM_PROMPT_EN
from . import contextualized, medquad
from .anonymize import AnonymizationReport, anonymize
from .curation import curate, split
from .protocol_source import PROTOCOLOS, Protocolo

# --------------------------------------------------------------------------- #
# Geração de pares a partir dos protocolos internos
# --------------------------------------------------------------------------- #

#: Cada entrada gera uma pergunta e a seção do protocolo que a responde. As
#: formulações variam de propósito: o modelo precisa reconhecer a intenção, não
#: decorar uma frase.
#:
#: Estas respostas encerram na citação da fonte, sem o aviso de validação. A
#: escolha foi medida: uma rodada de treino com o aviso em toda resposta interna
#: elevou a taxa de aviso do modelo (12,5% → 75%) e, no mesmo movimento, fez o
#: modelo recusar dúvidas clínicas legítimas ("posso colher hemocultura depois do
#: antibiótico?" → "não faço julgamento sobre ato médico"). Como o guardrail de
#: saída insere o aviso em 100% das respostas de qualquer forma, o treino estava
#: pagando utilidade por uma garantia que o sistema já dá. Ver relatório §6.5.
_PERGUNTAS_POR_SECAO = (
    ("indicacao", "1 — Indicação", (
        "Quando se aplica o protocolo {codigo}?",
        "Qual a indicação do protocolo de {titulo_curto}?",
        "Em que pacientes eu uso o {codigo}?",
    )),
    ("criterios", "2 — Critérios diagnósticos", (
        "Quais os critérios diagnósticos do {codigo}?",
        "Como eu confirmo o diagnóstico de {titulo_curto} pelo protocolo interno?",
        "Que critérios o protocolo de {titulo_curto} usa para caracterizar o quadro?",
    )),
    ("conduta", "3 — Conduta", (
        "Qual a conduta preconizada no {codigo}?",
        "Quais os passos do protocolo de {titulo_curto}?",
        "O que o protocolo interno manda fazer em {titulo_curto}?",
    )),
    ("exames_obrigatorios", "4 — Exames obrigatórios", (
        "Quais exames são obrigatórios no {codigo}?",
        "Que exames eu preciso ter antes de seguir com o protocolo de {titulo_curto}?",
        "Qual o rol mínimo de exames do protocolo de {titulo_curto}?",
    )),
    ("alertas", "5 — Alertas e não conformidades", (
        "Quais os alertas do {codigo}?",
        "O que gera não conformidade no protocolo de {titulo_curto}?",
        "Que armadilhas o protocolo de {titulo_curto} destaca?",
    )),
)


def _titulo_curto(p: Protocolo) -> str:
    return p.titulo.split(":")[0].split("—")[0].strip().lower()


def _fonte(p: Protocolo, secao: str) -> str:
    """Citação de referência: código, versão e seção — sem a data de revisão.

    Decisão tomada a partir de evidência: na rodada anterior o modelo acertava
    código e versão e **inventava a data** em 3 de 3 respostas clínicas. Metadado
    que o modelo não precisa reproduzir não deve estar na resposta de referência —
    cada campo factual a mais é superfície de alucinação. A revisão continua no
    documento do protocolo e no bloco de fontes que o sistema monta.
    """
    return f"Fonte: {p.codigo} v{p.versao}, seção {secao}."


def exemplos_de_protocolos(seed: int = SEED) -> list[dict]:
    """Transforma cada seção de cada protocolo em pares pergunta/resposta com fonte."""
    exemplos: list[dict] = []
    for p in PROTOCOLOS:
        for campo, secao, formulacoes in _PERGUNTAS_POR_SECAO:
            conteudo = getattr(p, campo)
            corpo = conteudo if isinstance(conteudo, str) else "\n".join(f"- {c}" for c in conteudo)
            for formulacao in formulacoes:
                pergunta = formulacao.format(codigo=p.codigo, titulo_curto=_titulo_curto(p))
                resposta = (
                    f"Segundo o protocolo {p.codigo} — {p.titulo} ({p.especialidade}):\n\n"
                    f"{corpo}\n\n{_fonte(p, secao)}"
                )
                exemplos.append({
                    "categoria": "protocolo_qa",
                    "pergunta": pergunta,
                    "resposta": resposta,
                    "fonte": p.codigo,
                    "lang": "pt",
                })
    return exemplos


def exemplos_de_faq(raw_dir: Path = RAW_DIR) -> list[dict]:
    """Carrega o FAQ bruto (com PII) — a anonimização acontece depois, em bloco."""
    caminho = raw_dir / "faq_medicos.jsonl"
    if not caminho.exists():
        return []
    exemplos: list[dict] = []
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        item = json.loads(linha)
        categoria = "limite_de_atuacao" if item["tipo"] == "limite_de_atuacao" else (
            "faq_contextualizada" if item["tipo"] == "faq_contextualizada" else "faq")
        exemplos.append({
            "categoria": categoria,
            "pergunta": item["pergunta"],
            "resposta": item["resposta"],
            "fonte": item.get("protocolo") or "política institucional",
            "lang": "pt",
        })
    return exemplos


_PEDIDOS_DOCUMENTO = {
    "laudo": "Monte o modelo de laudo institucional para um caso de {condicao} seguindo o {codigo}.",
    "receita": "Preciso do modelo de receituário institucional para um paciente em tratamento de {condicao} ({codigo}).",
    "procedimento": "Gere a descrição padrão de procedimento do {codigo} para um caso de {condicao}.",
}


def exemplos_de_documentos(raw_dir: Path = RAW_DIR) -> list[dict]:
    """Modelos de laudo, receita e procedimento viram exemplos de geração de documento."""
    pacientes_file = raw_dir / "pacientes.json"
    docs_dir = raw_dir / "documentos"
    if not pacientes_file.exists() or not docs_dir.is_dir():
        return []

    pacientes = {p["paciente_id"]: p for p in json.loads(pacientes_file.read_text(encoding="utf-8"))}
    exemplos: list[dict] = []
    vistos: set[tuple[str, str]] = set()

    for caminho in sorted(docs_dir.glob("*.md")):
        tipo, paciente_id = caminho.stem.split("_", 1)
        paciente = pacientes.get(paciente_id)
        if paciente is None:
            continue
        # Um exemplo por (tipo, protocolo): o modelo aprende o formato do documento,
        # não a repetir 24 variações do mesmo texto.
        chave = (tipo, paciente["protocolo"])
        if chave in vistos:
            continue
        vistos.add(chave)

        pedido = _PEDIDOS_DOCUMENTO[tipo].format(
            condicao=paciente["condicao"].lower(), codigo=paciente["protocolo"])
        resposta = (
            caminho.read_text(encoding="utf-8").strip()
            + f"\n\nFonte: modelo institucional de {tipo}, protocolo {paciente['protocolo']}. "
              "Documento requer conferência, assinatura e validação do médico responsável."
        )
        exemplos.append({
            "categoria": "documento",
            "pergunta": pedido,
            "resposta": resposta,
            "fonte": paciente["protocolo"],
            "lang": "pt",
        })
    return exemplos


def exemplos_de_medquad(n: int = 200, seed: int = SEED) -> list[dict]:
    """Conhecimento médico geral, em inglês, com a fonte original preservada."""
    if not medquad.is_available():
        return []
    exemplos = []
    for item in medquad.sample(n, seed=seed):
        resposta = item["resposta"]
        if item["url"]:
            resposta += f"\n\nSource: {item['fonte']} — {item['url']}"
        exemplos.append({
            "categoria": "conhecimento_geral",
            "pergunta": item["pergunta"],
            "resposta": resposta,
            "fonte": item["fonte"],
            "lang": "en",
        })
    return exemplos


# --------------------------------------------------------------------------- #
# Orquestração
# --------------------------------------------------------------------------- #


def _anonimizar_exemplos(exemplos: list[dict]) -> tuple[list[dict], AnonymizationReport]:
    consolidado = AnonymizationReport()
    saida: list[dict] = []
    for exemplo in exemplos:
        if exemplo.get("lang") == "en":
            # O MedQuAD é conteúdo público de referência, sem dado de paciente:
            # passa pelo detector, mas os padrões são de PII brasileira.
            saida.append(exemplo)
            continue
        pergunta, r1 = anonymize(exemplo["pergunta"])
        resposta, r2 = anonymize(exemplo["resposta"])
        consolidado.merge(r1)
        consolidado.merge(r2)
        saida.append({**exemplo, "pergunta": pergunta, "resposta": resposta})
    return saida, consolidado


def _to_chat(exemplo: dict) -> dict:
    system = SYSTEM_PROMPT_EN if exemplo.get("lang") == "en" else SYSTEM_PROMPT
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": exemplo["pergunta"]},
            {"role": "assistant", "content": exemplo["resposta"]},
        ],
        "categoria": exemplo["categoria"],
        "fonte": exemplo.get("fonte"),
        "lang": exemplo.get("lang", "pt"),
    }


def _relativo(caminho: Path) -> str:
    """Caminho relativo à raiz do projeto quando possível; absoluto fora dela."""
    try:
        return str(caminho.relative_to(ROOT))
    except ValueError:
        return str(caminho)


def _write_jsonl(path: Path, registros: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for registro in registros:
            fh.write(json.dumps(registro, ensure_ascii=False) + "\n")


def build_knowledge_base(kb_dir: Path = KB_DIR, raw_dir: Path = RAW_DIR) -> AnonymizationReport:
    """Publica os protocolos anonimizados — é a base que o RAG do assistente lê."""
    kb_dir.mkdir(parents=True, exist_ok=True)
    consolidado = AnonymizationReport()
    for caminho in sorted((raw_dir / "protocolos").glob("*.md")):
        limpo, report = anonymize(caminho.read_text(encoding="utf-8"))
        (kb_dir / caminho.name).write_text(limpo, encoding="utf-8")
        consolidado.merge(report)
    return consolidado


def build(n_medquad: int = 200, val_ratio: float = 0.15, seed: int = SEED) -> dict:
    """Executa o pipeline inteiro e devolve o resumo que vai para o relatório."""
    ensure_dirs()

    brutos = (exemplos_de_protocolos(seed)
              + exemplos_de_faq(RAW_DIR)
              + exemplos_de_documentos(RAW_DIR)
              + contextualized.gerar(seed=seed)
              + contextualized.gerar_consulta_geral(seed)
              + exemplos_de_medquad(n_medquad, seed))

    anonimizados, rel_anon = _anonimizar_exemplos(brutos)
    rel_kb = build_knowledge_base(KB_DIR, RAW_DIR)
    rel_anon.merge(rel_kb)

    aprovados, rel_cur = curate(anonimizados, seed=seed)
    treino, validacao = split(aprovados, val_ratio=val_ratio, seed=seed)

    _write_jsonl(TRAIN_FILE, [_to_chat(e) for e in treino])
    _write_jsonl(VAL_FILE, [_to_chat(e) for e in validacao])

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "relatorio_anonimizacao.json").write_text(
        json.dumps(rel_anon.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (EVAL_DIR / "relatorio_curadoria.json").write_text(
        json.dumps(rel_cur.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    resumo = {
        "exemplos_brutos": len(brutos),
        "anonimizacao": rel_anon.as_dict(),
        "curadoria": rel_cur.as_dict(),
        "treino": len(treino),
        "validacao": len(validacao),
        "arquivos": {
            "treino": _relativo(TRAIN_FILE),
            "validacao": _relativo(VAL_FILE),
            "base_conhecimento": _relativo(KB_DIR),
        },
    }
    (EVAL_DIR / "resumo_dataset.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    return resumo
