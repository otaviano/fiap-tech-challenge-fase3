"""Fluxo do assistente: LangGraph ponta a ponta, com LLM falsa."""

import pytest

from med_assist.assistant.chain import MedicalAssistant
from med_assist.knowledge import ehr


@pytest.fixture()
def assistente(llm_fake, db, trilha):
    return MedicalAssistant(llm=llm_fake)


def test_consulta_com_paciente_percorre_o_fluxo_completo(assistente, db, trilha):
    paciente_id = ehr.list_patients()[1]["paciente_id"]  # caso de sepse
    resposta = assistente.ask("O que priorizar agora neste caso?",
                              paciente_id=paciente_id, usuario="CRM-SP 111.222")

    assert resposta.modo == "llm"
    assert not resposta.bloqueado
    assert resposta.pendencias["encontrado"]
    assert resposta.alertas, "o caso de sepse deveria gerar alertas"
    assert resposta.notificacoes, "alerta crítico deveria notificar a equipe"
    assert any(e["tipo"] == "prontuario" for e in resposta.evidencias)
    assert any(e["tipo"] == "protocolo" for e in resposta.evidencias)

    eventos = [e["evento"] for e in trilha.read(trace_id=resposta.trace_id)]
    assert eventos == ["interacao_iniciada", "triagem", "prontuario_consultado",
                       "exames_verificados", "alertas_avaliados", "protocolos_recuperados",
                       "resposta_gerada", "saida_validada", "alerta_notificado",
                       "alerta_notificado", "interacao_concluida"]


def test_consulta_sem_paciente_pula_os_nos_de_prontuario(assistente, trilha):
    resposta = assistente.ask("Qual a janela de trombólise no AVC isquêmico?")
    eventos = [e["evento"] for e in trilha.read(trace_id=resposta.trace_id)]
    assert "prontuario_consultado" not in eventos
    assert "protocolos_recuperados" in eventos
    assert resposta.evidencias
    assert resposta.alertas == []


def test_pedido_de_prescricao_nao_chega_ao_modelo(assistente, trilha):
    resposta = assistente.ask("Prescreva ceftriaxona 1g para o paciente do leito E-05.")
    assert resposta.bloqueado
    assert resposta.modo == "guardrail"
    eventos = [e["evento"] for e in trilha.read(trace_id=resposta.trace_id)]
    assert "resposta_gerada" not in eventos, "o LLM não deveria ter sido chamado"
    assert "não posso emitir prescrição" in resposta.resposta.lower()


def test_resposta_sempre_carrega_fonte_e_aviso(assistente, db):
    paciente_id = ehr.list_patients()[0]["paciente_id"]
    resposta = assistente.ask("Quais exames faltam?", paciente_id=paciente_id)
    assert "Fontes consultadas" in resposta.resposta
    assert "validação do médico assistente" in resposta.resposta


def test_prontuario_inexistente_nao_quebra_o_fluxo(assistente):
    resposta = assistente.ask("Qual a conduta?", paciente_id="PRT-0000-000000")
    assert resposta.resposta
    assert resposta.pendencias is None


def test_fallback_responde_sem_servidor_de_llm(db, trilha, monkeypatch):
    """Servidor fora do ar: a resposta fica mais seca, mas continua citada."""
    from med_assist.assistant import llm as llm_module

    monkeypatch.setattr(llm_module, "is_available", lambda *a, **kw: False)
    assistente = MedicalAssistant()
    resposta = assistente.ask("O que o protocolo diz sobre a sepse?",
                              paciente_id=ehr.list_patients()[1]["paciente_id"])
    assert resposta.modo == "fallback"
    assert "contingência" in resposta.resposta
    assert "PROT-EME-002" in resposta.resposta


def test_diagrama_do_fluxo_reflete_o_grafo(assistente):
    mermaid = assistente.mermaid()
    for no in ("triagem", "contexto_paciente", "exames_pendentes", "avaliar_alertas",
               "recuperar_protocolo", "redigir_resposta", "validar_saida", "notificar_equipe"):
        assert no in mermaid
