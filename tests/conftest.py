"""Fixtures compartilhadas.

Os testes rodam sobre um banco e uma trilha de auditoria temporários e usam uma
LLM falsa: nada aqui depende de servidor no ar, de GPU ou de rede.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from med_assist.config import RAW_DIR
from med_assist.knowledge import ehr


@pytest.fixture(scope="session")
def pacientes() -> list[dict]:
    return json.loads((RAW_DIR / "pacientes.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def db():
    """Banco de prontuários recriado do seed sintético.

    Não é isolado por teste de propósito: o seed é determinístico e nenhum teste
    escreve no banco — todos os caminhos exercitados aqui são de leitura.
    """
    ehr.seed()
    return ehr.EHR_DB


@pytest.fixture()
def trilha(tmp_path, monkeypatch):
    """Trilha de auditoria em arquivo temporário."""
    from med_assist.safety import audit

    caminho = tmp_path / "auditoria.jsonl"
    nova = audit.AuditTrail(caminho)
    monkeypatch.setattr(audit, "_trail", nova)
    return nova


@pytest.fixture()
def llm_fake():
    """LLM determinística: devolve um texto plausível citando a fonte do contexto."""
    def _responder(prompt_value):
        texto = prompt_value.to_messages()[-1].content
        citacao = "PROT-EME-002"
        for marcador in ("PROT-",):
            pos = texto.find(marcador)
            if pos >= 0:
                citacao = texto[pos:pos + 12]
                break
        return AIMessage(content=(
            f"Resposta de teste fundamentada no {citacao}, seção 3 — Conduta. "
            "A conduta exige validação do médico assistente."))
    return RunnableLambda(_responder)
