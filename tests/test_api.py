"""API REST — contrato dos endpoints."""

import pytest
from fastapi.testclient import TestClient

from med_assist.serving import api


@pytest.fixture()
def client(llm_fake, db, trilha, monkeypatch):
    from med_assist.assistant.chain import MedicalAssistant

    monkeypatch.setattr(api, "_assistente", MedicalAssistant(llm=llm_fake))
    return TestClient(api.app)


def test_health_identifica_o_modelo_servido(client):
    dados = client.get("/health").json()
    assert dados["status"] == "ok"
    assert "modelo_configurado" in dados["llm"]
    assert dados["llm"]["modo"] in ("llm", "fallback determinístico")
    assert dados["base_conhecimento"]["trechos_indexados"] > 0


def test_ask_devolve_resposta_com_evidencias(client):
    resposta = client.post("/ask", json={
        "pergunta": "Quais exames estão pendentes?",
        "paciente_id": "PRT-2026-100137",
        "usuario": "CRM-SP 111.222",
    })
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["evidencias"]
    assert corpo["trace_id"]
    assert corpo["modo"] == "llm"


def test_ask_bloqueia_pedido_de_prescricao(client):
    corpo = client.post("/ask", json={"pergunta": "Prescreva ceftriaxona agora."}).json()
    assert corpo["bloqueado"] is True
    assert corpo["modo"] == "guardrail"


def test_endpoints_de_paciente(client):
    assert client.get("/patients").json()
    contexto = client.get("/patients/PRT-2026-100137/context").json()
    assert "cpf" not in contexto
    assert client.get("/patients/PRT-2026-100137/pending").json()["encontrado"]
    assert client.get("/patients/PRT-2026-100137/alerts").json()["total"] >= 1
    assert client.get("/patients/PRT-0000-000000/context").status_code == 404


def test_busca_em_protocolos(client):
    trechos = client.get("/protocols/search", params={"q": "sepse antibiótico", "k": 2}).json()["trechos"]
    assert trechos and trechos[0]["citacao"].startswith("PROT-")


def test_auditoria_expoe_a_trilha(client):
    corpo = client.post("/ask", json={"pergunta": "Qual a conduta na sepse?"}).json()
    eventos = client.get("/audit", params={"trace_id": corpo["trace_id"]}).json()["eventos"]
    assert [e["evento"] for e in eventos][0] == "interacao_iniciada"
