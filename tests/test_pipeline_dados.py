"""Pipeline de dados: geração sintética, ingestão externa e montagem do dataset."""

import json

import pytest

from med_assist.data import build_dataset, medquad, synth
from med_assist.data.anonymize import residual_pii
from med_assist.data.protocol_source import PROTOCOLOS


def test_corpus_bruto_e_deterministico(tmp_path):
    primeiro = synth.build_raw_corpus(tmp_path / "a", n_pacientes=6)
    segundo = synth.build_raw_corpus(tmp_path / "b", n_pacientes=6)
    assert primeiro["pacientes"] == segundo["pacientes"] == 6
    assert ((tmp_path / "a" / "pacientes.json").read_text(encoding="utf-8")
            == (tmp_path / "b" / "pacientes.json").read_text(encoding="utf-8"))


def test_corpus_bruto_contem_pii_de_proposito(tmp_path):
    """Sem PII na entrada não há como demonstrar a anonimização."""
    synth.build_raw_corpus(tmp_path, n_pacientes=4)
    laudo = next((tmp_path / "documentos").glob("laudo_*.md")).read_text(encoding="utf-8")
    achados = {a.split(":")[0] for a in residual_pii(laudo)}
    assert {"CPF", "NOME", "PRONTUARIO"} <= achados


def test_pacientes_referenciam_exames_do_proprio_protocolo():
    pacientes = synth.gerar_pacientes(8)
    for paciente in pacientes:
        protocolo = next(p for p in PROTOCOLOS if p.codigo == paciente["protocolo"])
        exames = {e["exame"] for e in paciente["exames"]}
        assert exames <= set(protocolo.exames_obrigatorios)
        assert any(e["status"] == "pendente" for e in paciente["exames"])


def test_faq_cobre_duvidas_contextualizadas_e_recusas():
    itens = synth.gerar_faq(synth.gerar_pacientes(12))
    tipos = {i["tipo"] for i in itens}
    assert {"faq", "faq_contextualizada", "limite_de_atuacao"} == tipos
    assert sum(1 for i in itens if i["tipo"] == "limite_de_atuacao") >= 10


def test_protocolo_renderizado_tem_todas_as_secoes():
    texto = synth.render_protocolo(PROTOCOLOS[0])
    for secao in ("## 1. Indicação", "## 2. Critérios diagnósticos", "## 3. Conduta",
                  "## 4. Exames obrigatórios", "## 5. Alertas e não conformidades"):
        assert secao in texto


@pytest.mark.skipif(not medquad.is_available(),
                    reason="MedQuAD não baixado (scripts/fetch_medquad.sh)")
def test_amostra_do_medquad_e_estratificada():
    amostra = medquad.sample(60)
    assert len(amostra) >= 45
    tipos = {item["qtype"] for item in amostra}
    assert len(tipos) >= 5
    assert all(item["resposta"] and item["fonte"].startswith("MedQuAD/") for item in amostra)


def test_dataset_completo_passa_pelos_dois_gates(tmp_path, monkeypatch):
    """Anonimização e curadoria rodando ponta a ponta, em diretório isolado."""
    raw = tmp_path / "raw"
    synth.build_raw_corpus(raw, n_pacientes=8)

    monkeypatch.setattr(build_dataset, "RAW_DIR", raw)
    monkeypatch.setattr(build_dataset, "KB_DIR", tmp_path / "kb")
    monkeypatch.setattr(build_dataset, "TRAIN_FILE", tmp_path / "train.jsonl")
    monkeypatch.setattr(build_dataset, "VAL_FILE", tmp_path / "val.jsonl")
    monkeypatch.setattr(build_dataset, "EVAL_DIR", tmp_path / "eval")

    resumo = build_dataset.build(n_medquad=0)

    assert resumo["treino"] > 50 and resumo["validacao"] > 5
    assert resumo["anonimizacao"]["identificadores_removidos"] > 50
    assert resumo["curadoria"]["motivos_de_descarte"].get("pii_residual", 0) == 0

    registros = [json.loads(linha) for linha in
                 (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    for registro in registros:
        texto = " ".join(m["content"] for m in registro["messages"])
        assert residual_pii(texto) == [], "exemplo de treino com PII residual"


def test_base_de_conhecimento_sai_anonimizada(tmp_path):
    raw = tmp_path / "raw"
    synth.build_raw_corpus(raw, n_pacientes=4)
    kb = tmp_path / "kb"
    relatorio = build_dataset.build_knowledge_base(kb, raw)

    assert len(list(kb.glob("*.md"))) == len(PROTOCOLOS)
    assert relatorio.counts["EMAIL"] >= len(PROTOCOLOS)
    for caminho in kb.glob("*.md"):
        conteudo = caminho.read_text(encoding="utf-8")
        assert residual_pii(conteudo) == []
        assert "santahelena.hosp.br" not in conteudo
        assert "## 3. Conduta" in conteudo, "a anonimização não pode destruir a estrutura"
