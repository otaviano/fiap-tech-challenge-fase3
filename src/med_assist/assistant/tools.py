"""Ferramentas LangChain do assistente.

Cada capacidade do assistente é exposta como uma `@tool`: consultar protocolo,
consultar prontuário, verificar pendências de exame e avaliar alertas. O fluxo
do LangGraph chama as funções puras diretamente (determinismo e menos latência),
e as mesmas capacidades ficam disponíveis como ferramentas para um agente com
tool calling — é o mesmo comportamento pelos dois caminhos.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from ..knowledge import ehr
from ..knowledge.retriever import get_retriever
from . import alerts as alert_engine


def buscar_protocolo(consulta: str, k: int = 3, protocolo: str | None = None) -> list[dict]:
    """Busca seções de protocolo relevantes (função pura, usada pelo grafo)."""
    return [
        {"citacao": trecho.citacao, "protocolo": trecho.protocolo, "titulo": trecho.titulo,
         "secao": trecho.secao, "texto": trecho.texto, "score": score}
        for trecho, score in get_retriever().search(consulta, k=k, protocolo=protocolo)
    ]


@tool("consultar_protocolo")
def consultar_protocolo(consulta: str, protocolo: str | None = None) -> str:
    """Consulta os protocolos clínicos internos do hospital e devolve os trechos
    mais relevantes, com o código do protocolo, a seção e o score de relevância.
    Use para fundamentar qualquer afirmação sobre conduta, critério, exame ou alerta."""
    trechos = buscar_protocolo(consulta, protocolo=protocolo)
    if not trechos:
        return "Nenhum trecho de protocolo interno corresponde à consulta."
    return "\n\n".join(f"[{t['citacao']}] {t['titulo']}\n{t['texto']}" for t in trechos)


@tool("consultar_prontuario")
def consultar_prontuario(paciente_id: str) -> str:
    """Consulta o prontuário eletrônico e devolve o contexto clínico mínimo do
    paciente (idade, quadro, comorbidades, alergias, sinais vitais, escores,
    exames e medicações). Não retorna dados identificáveis."""
    contexto = ehr.clinical_context(paciente_id)
    if contexto is None:
        return f"Paciente {paciente_id} não encontrado no prontuário eletrônico."
    return ehr.format_context(contexto)


@tool("verificar_exames_pendentes")
def verificar_exames_pendentes(paciente_id: str) -> str:
    """Compara os exames obrigatórios do protocolo aplicável ao paciente com o
    que já tem resultado no prontuário e devolve o que está pendente ou nem foi
    solicitado."""
    pendencias = ehr.pending_exams(paciente_id)
    if not pendencias.get("encontrado"):
        return f"Paciente {paciente_id} não encontrado no prontuário eletrônico."
    return json.dumps(pendencias, ensure_ascii=False, indent=2)


@tool("avaliar_alertas")
def avaliar_alertas(paciente_id: str) -> str:
    """Aplica as regras clínicas determinísticas do hospital ao caso e devolve os
    alertas ativos, com severidade e o protocolo que os fundamenta."""
    contexto = ehr.clinical_context(paciente_id)
    if contexto is None:
        return f"Paciente {paciente_id} não encontrado no prontuário eletrônico."
    ativos = alert_engine.avaliar(contexto, ehr.pending_exams(paciente_id))
    if not ativos:
        return "Nenhum alerta ativo para o caso."
    return json.dumps([a.as_dict() for a in ativos], ensure_ascii=False, indent=2)


TOOLS = [consultar_protocolo, consultar_prontuario, verificar_exames_pendentes, avaliar_alertas]
