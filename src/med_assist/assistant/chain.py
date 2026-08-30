"""Fachada do assistente: uma chamada, uma interação auditada.

`MedicalAssistant.ask()` abre a trilha de auditoria, executa o fluxo LangGraph e
devolve um objeto com tudo o que a interação produziu — resposta, evidências,
alertas, notificações, pendências, modo de operação e o veredito dos guardrails.
É esta a superfície usada pela CLI, pela API e pelo notebook.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..config import LLMSettings
from ..safety.audit import get_audit_trail, get_logger
from .graph import build_graph

logger = get_logger("assistente")


@dataclass
class RespostaAssistente:
    """Resultado completo de uma consulta ao assistente."""

    pergunta: str
    resposta: str
    trace_id: str
    modo: str                                   # llm | fallback | guardrail
    bloqueado: bool = False
    evidencias: list[dict] = field(default_factory=list)
    alertas: list[dict] = field(default_factory=list)
    notificacoes: list[dict] = field(default_factory=list)
    pendencias: dict | None = None
    guardrail: dict = field(default_factory=dict)
    paciente_id: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        """Formato de terminal, usado pela CLI e pela demonstração."""
        partes = [self.resposta]
        if self.notificacoes:
            partes.append("\n**Notificações emitidas à equipe:**")
            partes += [f"- [{n['severidade'].upper()}] {n['codigo']} → {n['destinatario']}"
                       f" ({n.get('setor')}, leito {n.get('leito')})" for n in self.notificacoes]
        partes.append(f"\n_trace_id: {self.trace_id} | modo: {self.modo}_")
        return "\n".join(partes)


class MedicalAssistant:
    """Assistente clínico do hospital — LangChain + LangGraph sobre a LLM customizada."""

    def __init__(self, llm: Any | None = None, settings: LLMSettings | None = None):
        self.settings = settings or LLMSettings()
        self.grafo = build_graph(llm=llm)
        self.trilha = get_audit_trail()

    def ask(self, pergunta: str, paciente_id: str | None = None,
            usuario: str = "profissional não identificado") -> RespostaAssistente:
        """Executa uma interação completa, do guardrail de entrada à notificação."""
        with self.trilha.trace(usuario=usuario, pergunta=pergunta, paciente_id=paciente_id) as trace_id:
            estado_final = self.grafo.invoke({
                "pergunta": pergunta,
                "usuario": usuario,
                "paciente_id": paciente_id,
                "trace_id": trace_id,
            })

        return RespostaAssistente(
            pergunta=pergunta,
            resposta=estado_final.get("resposta", ""),
            trace_id=trace_id,
            modo=estado_final.get("modo_llm", "desconhecido"),
            bloqueado=bool(estado_final.get("bloqueado")),
            evidencias=estado_final.get("evidencias", []),
            alertas=estado_final.get("alertas", []),
            notificacoes=estado_final.get("notificacoes", []),
            pendencias=estado_final.get("pendencias"),
            guardrail=estado_final.get("guardrail_saida", {}),
            paciente_id=paciente_id,
        )

    def mermaid(self) -> str:
        """Diagrama do fluxo, gerado a partir do grafo compilado.

        Diagrama gerado do código não diverge da implementação — é o próprio
        grafo se desenhando.
        """
        return self.grafo.get_graph().draw_mermaid()
