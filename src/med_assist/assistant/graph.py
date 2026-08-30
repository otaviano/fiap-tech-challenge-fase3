"""Fluxo de decisão automatizado do assistente (LangGraph).

    START
      │
      ▼
    triagem ──(pedido de ato médico)──────────────────────────────┐
      │                                                           │
      ├──(com prontuário)──▶ contexto_paciente ──▶ exames_pendentes ──▶ avaliar_alertas ──┐
      │                                                           │                       │
      └──(sem prontuário)─────────────────────────────────────────┼───────────────────────┤
                                                                  ▼                       ▼
                                                        recuperar_protocolo ──▶ redigir_resposta
                                                                                      │
                                                                                      ▼
                                                                              validar_saida
                                                                                      │
                                                          ┌──(alerta crítico)──▶ notificar_equipe ──▶ END
                                                          └──────────────────────────────────────────▶ END

Duas decisões de projeto que valem explicitar:

1. **O que é determinístico não passa pelo modelo.** Guardrail de entrada,
   consulta ao prontuário, cálculo de pendências e regras de alerta são código.
   O LLM entra em um único nó — `redigir_resposta` — e recebe tudo pronto.
2. **O caminho é auditável nó a nó.** Cada nó emite um evento na trilha com o
   mesmo `trace_id`, então dá para responder "por que o assistente disse isso"
   lendo o arquivo, sem reexecutar nada.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from ..knowledge import ehr
from ..safety import guardrails
from ..safety.audit import get_audit_trail, get_logger
from ..safety.explainability import Explicacao
from . import alerts as alert_engine
from .prompts import ASSISTANT_PROMPT, secao
from .tools import buscar_protocolo

logger = get_logger("graph")

SEVERIDADES_NOTIFICAVEIS = ("critico", "alto")


def _ultimo(_atual: Any, novo: Any) -> Any:
    """Reducer explícito: o último nó a escrever no campo vence."""
    return novo


class EstadoClinico(TypedDict, total=False):
    """Estado que trafega pelo fluxo. Tudo o que o auditor precisa ver está aqui."""

    # entrada
    pergunta: str
    usuario: str
    paciente_id: str | None
    trace_id: str

    # triagem
    bloqueado: bool
    motivos_bloqueio: list[str]

    # contexto clínico
    contexto: dict | None
    contexto_texto: str
    pendencias: dict | None
    alertas: list[dict]

    # recuperação e geração
    trechos: list[dict]
    resposta: str
    modo_llm: str

    # saída
    evidencias: list[dict]
    guardrail_saida: dict
    notificacoes: Annotated[list[dict], _ultimo]


def _trilha():
    return get_audit_trail()


# --------------------------------------------------------------------------- #
# Nós
# --------------------------------------------------------------------------- #


def no_triagem(estado: EstadoClinico) -> dict:
    """Aplica os limites de atuação antes de qualquer coisa custar dinheiro ou risco."""
    veredito = guardrails.check_input(estado["pergunta"])
    _trilha().event(estado["trace_id"], "triagem",
                    bloqueado=veredito.bloqueado, motivos=veredito.motivos,
                    paciente_id=estado.get("paciente_id"))
    if veredito.bloqueado:
        return {
            "bloqueado": True,
            "motivos_bloqueio": veredito.motivos,
            "resposta": veredito.resposta or "",
            "modo_llm": "guardrail",
            "evidencias": [{"tipo": "politica_institucional",
                            "referencia": "Limites de atuação do assistente",
                            "trecho": ", ".join(veredito.motivos), "score": None}],
        }
    return {"bloqueado": False, "motivos_bloqueio": []}


def no_contexto_paciente(estado: EstadoClinico) -> dict:
    """Consulta a base estruturada e traz só o mínimo necessário."""
    paciente_id = estado.get("paciente_id")
    contexto = ehr.clinical_context(paciente_id) if paciente_id else None
    if contexto is None:
        _trilha().event(estado["trace_id"], "prontuario_nao_encontrado", paciente_id=paciente_id)
        return {"contexto": None,
                "contexto_texto": f"Prontuário {paciente_id} não localizado na base estruturada."}

    _trilha().event(estado["trace_id"], "prontuario_consultado",
                    paciente_id=contexto["paciente_id"], protocolo=contexto["protocolo"],
                    campos_expostos=len(contexto), minimizado=True)
    return {"contexto": contexto, "contexto_texto": ehr.format_context(contexto)}


def no_exames_pendentes(estado: EstadoClinico) -> dict:
    """Diferença verificável entre o exigido pelo protocolo e o registrado."""
    contexto = estado.get("contexto")
    if not contexto:
        return {"pendencias": None}
    pendencias = ehr.pending_exams(contexto["paciente_id"])
    _trilha().event(estado["trace_id"], "exames_verificados",
                    paciente_id=contexto["paciente_id"],
                    pendentes=len(pendencias.get("pendentes", [])),
                    nao_solicitados=len(pendencias.get("nao_solicitados", [])),
                    conforme=pendencias.get("conforme"))
    return {"pendencias": pendencias}


def no_avaliar_alertas(estado: EstadoClinico) -> dict:
    """Regras clínicas sobre dados estruturados — nunca geração."""
    contexto = estado.get("contexto")
    if not contexto:
        return {"alertas": []}
    ativos = [a.as_dict() for a in alert_engine.avaliar(contexto, estado.get("pendencias"))]
    _trilha().event(estado["trace_id"], "alertas_avaliados",
                    paciente_id=contexto["paciente_id"],
                    quantidade=len(ativos), codigos=[a["codigo"] for a in ativos])
    return {"alertas": ativos}


def no_recuperar_protocolo(estado: EstadoClinico) -> dict:
    """Busca as seções que fundamentam a resposta, com score auditável."""
    contexto = estado.get("contexto") or {}
    consulta = estado["pergunta"]
    if contexto:
        consulta = f"{consulta} {contexto.get('condicao', '')}"
    trechos = buscar_protocolo(consulta, k=4, protocolo=contexto.get("protocolo"))
    if not trechos:  # sem protocolo do paciente, busca no corpus inteiro
        trechos = buscar_protocolo(consulta, k=4)
    _trilha().event(estado["trace_id"], "protocolos_recuperados",
                    quantidade=len(trechos),
                    citacoes=[t["citacao"] for t in trechos],
                    scores=[t["score"] for t in trechos])
    return {"trechos": trechos}


def no_redigir_resposta(estado: EstadoClinico, llm=None) -> dict:
    """Único nó em que o modelo atua — e ele recebe tudo já apurado."""
    from langchain_core.output_parsers import StrOutputParser

    from .llm import build_llm

    modelo, modo = (llm, "llm") if llm is not None else build_llm()

    alertas_texto = "\n".join(
        f"- [{a['severidade'].upper()}] {a['codigo']}: {a['mensagem']} (fonte: {a['protocolo']})"
        for a in estado.get("alertas", []))
    pendencias = estado.get("pendencias") or {}
    pendencias_texto = ""
    if pendencias.get("encontrado"):
        listadas = pendencias.get("pendentes", []) + pendencias.get("nao_solicitados", [])
        pendencias_texto = ("\n".join(f"- {e}" for e in listadas)
                            if listadas else "- Nenhuma pendência: rol de exames do protocolo completo.")
    protocolos_texto = "\n\n".join(f"[{t['citacao']}] {t['titulo']}\n{t['texto']}"
                                   for t in estado.get("trechos", []))

    entradas = {
        "contexto_paciente": secao("Contexto do paciente", estado.get("contexto_texto", "")),
        "pendencias": secao("Pendências", pendencias_texto),
        "alertas": secao("Alertas ativos", alertas_texto),
        "protocolos": secao("Trechos de protocolo recuperados", protocolos_texto),
        "pergunta": estado["pergunta"],
    }

    cadeia = ASSISTANT_PROMPT | modelo | StrOutputParser()
    resposta = cadeia.invoke(entradas)

    _trilha().event(estado["trace_id"], "resposta_gerada", modo=modo,
                    caracteres=len(resposta), trechos_no_prompt=len(estado.get("trechos", [])))
    return {"resposta": resposta, "modo_llm": modo}


def no_validar_saida(estado: EstadoClinico) -> dict:
    """Guardrail de saída + montagem do bloco de fontes."""
    explicacao = Explicacao()
    for trecho in estado.get("trechos", []):
        explicacao.add("protocolo", trecho["citacao"], trecho["texto"][:200], trecho["score"])
    if estado.get("contexto"):
        explicacao.add("prontuario", f"Prontuário {estado['contexto']['paciente_id']} "
                                     f"(dados mínimos)", estado.get("contexto_texto", "")[:200])
    for alerta in estado.get("alertas", []):
        explicacao.add("regra_clinica", f"{alerta['codigo']} — {alerta['protocolo']}",
                       alerta["mensagem"][:200])

    if estado.get("bloqueado"):
        # A recusa institucional não passa pelo reescritor de saída: ela já é o
        # texto aprovado, e anexar fonte de protocolo aqui seria enganoso.
        resposta_final = estado.get("resposta", "")
        veredito = guardrails.GuardrailVerdict(bloqueado=True,
                                               motivos=estado.get("motivos_bloqueio", []),
                                               severidade="bloqueio")
        evidencias = estado.get("evidencias", [])
    else:
        resposta_final, veredito = guardrails.check_output(estado.get("resposta", ""))
        # O modelo pode citar um protocolo que não foi recuperado, ou acertar o
        # código e errar a versão/revisão. A conferência é contra os metadados
        # reais; o bloco de fontes abaixo continua sendo a referência do sistema.
        recuperados = {t["protocolo"] for t in estado.get("trechos", [])}
        veredito.motivos.extend(guardrails.check_citations(resposta_final, recuperados))
        veredito.motivos.extend(guardrails.check_hallucinated_context(
            resposta_final, tem_paciente=bool(estado.get("contexto"))))
        resposta_final = resposta_final + "\n" + explicacao.render()
        evidencias = explicacao.as_list()

    _trilha().event(estado["trace_id"], "saida_validada",
                    motivos=veredito.motivos, severidade=veredito.severidade,
                    evidencias=len(evidencias))
    return {"resposta": resposta_final, "evidencias": evidencias,
            "guardrail_saida": veredito.as_dict()}


def no_notificar_equipe(estado: EstadoClinico) -> dict:
    """Emite a notificação dos alertas relevantes para a equipe assistencial.

    Em produção este nó publicaria em fila/pager; aqui ele registra na trilha,
    que é o comportamento observável e auditável da demonstração.
    """
    notificaveis = [a for a in estado.get("alertas", [])
                    if a["severidade"] in SEVERIDADES_NOTIFICAVEIS]
    contexto = estado.get("contexto") or {}
    notificacoes = [{
        "codigo": a["codigo"],
        "severidade": a["severidade"],
        "destinatario": a.get("destinatario", "equipe assistencial"),
        "setor": contexto.get("setor"),
        "leito": contexto.get("leito"),
        "paciente_id": contexto.get("paciente_id"),
        "mensagem": a["mensagem"],
        "protocolo": a["protocolo"],
    } for a in notificaveis]

    for notificacao in notificacoes:
        _trilha().event(estado["trace_id"], "alerta_notificado", **notificacao)
    logger.warning("%d alerta(s) notificado(s) para %s",
                   len(notificacoes), contexto.get("setor", "equipe assistencial"))
    return {"notificacoes": notificacoes}


# --------------------------------------------------------------------------- #
# Arestas condicionais
# --------------------------------------------------------------------------- #


def _apos_triagem(estado: EstadoClinico) -> Literal["bloqueado", "com_paciente", "sem_paciente"]:
    if estado.get("bloqueado"):
        return "bloqueado"
    return "com_paciente" if estado.get("paciente_id") else "sem_paciente"


def _apos_validacao(estado: EstadoClinico) -> Literal["notificar", "fim"]:
    tem_alerta = any(a["severidade"] in SEVERIDADES_NOTIFICAVEIS for a in estado.get("alertas", []))
    return "notificar" if tem_alerta else "fim"


def build_graph(llm=None):
    """Monta e compila o fluxo. `llm` injetável facilita teste e demonstração."""
    grafo = StateGraph(EstadoClinico)

    grafo.add_node("triagem", no_triagem)
    grafo.add_node("contexto_paciente", no_contexto_paciente)
    grafo.add_node("exames_pendentes", no_exames_pendentes)
    grafo.add_node("avaliar_alertas", no_avaliar_alertas)
    grafo.add_node("recuperar_protocolo", no_recuperar_protocolo)
    grafo.add_node("redigir_resposta", lambda estado: no_redigir_resposta(estado, llm=llm))
    grafo.add_node("validar_saida", no_validar_saida)
    grafo.add_node("notificar_equipe", no_notificar_equipe)

    grafo.add_edge(START, "triagem")
    grafo.add_conditional_edges("triagem", _apos_triagem, {
        "bloqueado": "validar_saida",
        "com_paciente": "contexto_paciente",
        "sem_paciente": "recuperar_protocolo",
    })
    grafo.add_edge("contexto_paciente", "exames_pendentes")
    grafo.add_edge("exames_pendentes", "avaliar_alertas")
    grafo.add_edge("avaliar_alertas", "recuperar_protocolo")
    grafo.add_edge("recuperar_protocolo", "redigir_resposta")
    grafo.add_edge("redigir_resposta", "validar_saida")
    grafo.add_conditional_edges("validar_saida", _apos_validacao, {
        "notificar": "notificar_equipe",
        "fim": END,
    })
    grafo.add_edge("notificar_equipe", END)

    return grafo.compile()
