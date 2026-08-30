"""API REST do assistente médico.

Superfície mínima, honesta com o que o sistema faz:

    GET  /health                     — status, modelo servido e tamanho da base
    POST /ask                        — pergunta ao assistente (fluxo completo)
    GET  /patients                   — pacientes da base estruturada
    GET  /patients/{id}/context      — contexto clínico minimizado
    GET  /patients/{id}/pending      — exames pendentes segundo o protocolo
    GET  /patients/{id}/alerts       — alertas clínicos ativos
    GET  /protocols/search           — busca nos protocolos internos
    GET  /audit                      — trilha de auditoria (rastreamento)

`/health` identifica **qual modelo está de fato servindo** — auditar isso
importa: a mesma pergunta responde diferente conforme o servidor esteja com o
modelo base ou com o fine-tunado do hospital.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from ..assistant import alerts as alert_engine
from ..assistant.chain import MedicalAssistant
from ..assistant.llm import is_available, served_model
from ..assistant.tools import buscar_protocolo
from ..config import HOSPITAL, LLMSettings
from ..knowledge import ehr
from ..knowledge.retriever import get_retriever
from ..safety.audit import get_audit_trail, setup_logging

setup_logging()

app = FastAPI(
    title="Assistente Clínico — Tech Challenge Fase 3",
    description=f"Assistente de apoio à decisão do {HOSPITAL}: LLM fine-tunada com dados "
                "internos, orquestrada por LangChain/LangGraph, com guardrails e auditoria.",
    version="1.0.0",
)

_assistente: MedicalAssistant | None = None


def get_assistente() -> MedicalAssistant:
    """Instância preguiçosa: o app sobe mesmo com o servidor da LLM fora do ar."""
    global _assistente
    if _assistente is None:
        _assistente = MedicalAssistant()
    return _assistente


class PerguntaRequest(BaseModel):
    pergunta: str = Field(..., description="Dúvida clínica do profissional")
    paciente_id: str | None = Field(None, description="Prontuário para contextualizar a resposta")
    usuario: str = Field("profissional não identificado",
                         description="Identificação do profissional (registrada na auditoria)")


class RespostaResponse(BaseModel):
    resposta: str
    trace_id: str
    modo: str
    bloqueado: bool
    evidencias: list[dict]
    alertas: list[dict]
    notificacoes: list[dict]
    pendencias: dict | None
    guardrail: dict


@app.get("/health")
def health() -> dict:
    settings = LLMSettings()
    disponivel = is_available(settings)
    return {
        "status": "ok",
        "hospital": HOSPITAL,
        "llm": {
            "base_url": settings.base_url,
            "modelo_configurado": settings.model,
            "modelo_servido": served_model(settings) if disponivel else None,
            "disponivel": disponivel,
            "modo": "llm" if disponivel else "fallback determinístico",
        },
        "base_conhecimento": {"trechos_indexados": len(get_retriever())},
        "prontuarios": len(ehr.list_patients()),
    }


@app.post("/ask", response_model=RespostaResponse)
def ask(req: PerguntaRequest) -> RespostaResponse:
    resposta = get_assistente().ask(req.pergunta, paciente_id=req.paciente_id, usuario=req.usuario)
    return RespostaResponse(**{k: v for k, v in resposta.as_dict().items()
                               if k in RespostaResponse.model_fields})


@app.get("/patients")
def patients(setor: str | None = None) -> list[dict]:
    return ehr.list_patients(setor=setor)


@app.get("/patients/{paciente_id}/context")
def patient_context(paciente_id: str) -> dict:
    contexto = ehr.clinical_context(paciente_id)
    if contexto is None:
        raise HTTPException(status_code=404, detail=f"Paciente {paciente_id} não encontrado")
    return contexto


@app.get("/patients/{paciente_id}/pending")
def patient_pending(paciente_id: str) -> dict:
    pendencias = ehr.pending_exams(paciente_id)
    if not pendencias.get("encontrado"):
        raise HTTPException(status_code=404, detail=f"Paciente {paciente_id} não encontrado")
    return pendencias


@app.get("/patients/{paciente_id}/alerts")
def patient_alerts(paciente_id: str) -> dict:
    contexto = ehr.clinical_context(paciente_id)
    if contexto is None:
        raise HTTPException(status_code=404, detail=f"Paciente {paciente_id} não encontrado")
    ativos = alert_engine.avaliar(contexto, ehr.pending_exams(paciente_id))
    return {"paciente_id": paciente_id, "total": len(ativos),
            "alertas": [a.as_dict() for a in ativos]}


@app.get("/protocols/search")
def protocols_search(q: str = Query(..., description="Termo ou pergunta clínica"),
                     k: int = 3, protocolo: str | None = None) -> dict:
    return {"consulta": q, "trechos": buscar_protocolo(q, k=k, protocolo=protocolo)}


@app.get("/audit")
def audit(trace_id: str | None = None, limite: int = 50) -> dict:
    registros = get_audit_trail().read(trace_id=trace_id, limite=limite)
    return {"total": len(registros), "eventos": registros}
