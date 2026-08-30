"""Base de conhecimento: recuperação de protocolo e prontuário estruturado."""

import pytest

from med_assist.knowledge import ehr
from med_assist.knowledge.retriever import get_retriever, tokenize


def test_indice_tem_todas_as_secoes_clinicas():
    retriever = get_retriever()
    assert len(retriever) >= 32
    protocolos = {t.protocolo for t in retriever.trechos}
    assert len(protocolos) == 8
    # Referências bibliográficas não são recuperáveis: não respondem pergunta clínica.
    assert not any("referência" in t.secao.lower() for t in retriever.trechos)


@pytest.mark.parametrize("consulta,esperado", [
    ("quanto tempo para antibiótico na sepse", "PROT-EME-002"),
    ("janela de trombólise no AVC", "PROT-NEU-005"),
    ("quando indico profilaxia de TEV", "PROT-HEM-007"),
    ("insulina na cetoacidose diabética", "PROT-END-004"),
    ("biópsia de nódulo mamário BI-RADS 4", "PROT-ONC-001"),
])
def test_recupera_o_protocolo_certo(consulta, esperado):
    resultados = get_retriever().search(consulta, k=3)
    assert resultados, f"nada recuperado para '{consulta}'"
    assert esperado in {t.protocolo for t, _ in resultados}


def test_tokenizacao_preserva_codigo_de_protocolo():
    assert "prot-eme-002" in tokenize("Qual a conduta do PROT-EME-002?")


def test_citacao_tem_granularidade_de_secao():
    trecho, score = get_retriever().search("conduta na sepse", k=1)[0]
    assert trecho.citacao.startswith("PROT-")
    assert "seção" in trecho.citacao
    assert score > 0


def test_contexto_clinico_nao_expoe_dados_identificaveis(db):
    paciente_id = ehr.list_patients()[0]["paciente_id"]
    contexto = ehr.clinical_context(paciente_id)
    for campo in ehr.CAMPOS_IDENTIFICAVEIS:
        assert campo not in contexto, f"{campo} vazou para o contexto do LLM"
    texto = ehr.format_context(contexto)
    completo = ehr.get_patient(paciente_id)
    for campo in ("cpf", "cns", "telefone", "email", "endereco", "nome"):
        assert str(completo[campo]) not in texto


def test_exames_pendentes_sao_a_diferenca_do_protocolo(db):
    paciente_id = ehr.list_patients()[1]["paciente_id"]
    pendencias = ehr.pending_exams(paciente_id)
    assert pendencias["encontrado"]
    obrigatorios = set(pendencias["exames_obrigatorios"])
    cobertos = set(pendencias["concluidos"]) | set(pendencias["pendentes"]) | set(pendencias["nao_solicitados"])
    assert obrigatorios <= cobertos
    assert pendencias["conforme"] == (not pendencias["pendentes"] and not pendencias["nao_solicitados"])


def test_busca_por_leito_ou_nome(db):
    paciente = ehr.get_patient(ehr.list_patients()[0]["paciente_id"])
    assert ehr.get_patient(paciente["leito"])["paciente_id"] == paciente["paciente_id"]
    assert ehr.get_patient("inexistente-999") is None
