"""Exemplos de treino no formato exato do prompt de inferência.

Motivação, descoberta na primeira rodada de treino: um dataset feito só de
"pergunta curta → resposta" ensina o modelo a responder perguntas curtas. Em
produção, porém, o nó `redigir_resposta` entrega um prompt com quatro seções —
contexto do paciente, pendências, alertas e trechos de protocolo. Diante desse
formato nunca visto, o modelo de 0,5B **ecoou as seções do prompt** em vez de
responder.

A correção é alinhar treino e inferência: estes exemplos são montados com o
**mesmo `ASSISTANT_PROMPT` do runtime**, preenchido com casos reais da base
estruturada, e a resposta de referência é redigida por template a partir dos
dados já apurados (alertas, pendências, conduta do protocolo).

A dependência deste módulo em `knowledge` e `assistant` é proposital: se o prompt
do runtime mudar, o dataset de treino acompanha automaticamente. Um gerador que
reimplementasse o template voltaria a divergir na primeira alteração.

Supervisão fraca, assumida: a resposta de referência é template, não redação
humana. O que ela ensina é **estrutura e ancoragem** — resumir o caso, começar
pelos alertas, listar pendências, citar fonte e fechar com o aviso de validação.
"""

from __future__ import annotations

import random

from ..assistant import alerts as alert_engine
from ..assistant.prompts import TEMPLATE_HUMANO, secao
from ..assistant.tools import buscar_protocolo
from ..config import SEED
from ..data.protocol_source import PROTOCOLOS_POR_CODIGO
from ..knowledge import ehr
from ..persona import DISCLAIMER

#: (pergunta do profissional, foco da resposta de referência)
PERGUNTAS = (
    ("O que devo priorizar agora neste caso?", "prioridades"),
    ("Quais são as pendências deste paciente segundo o protocolo?", "pendencias"),
    ("Há algo fora de conformidade neste caso?", "conformidade"),
    ("Resuma o caso e o que o protocolo interno determina.", "resumo"),
    ("Quais os próximos passos para este paciente?", "prioridades"),
)


def _resumo(contexto: dict, protocolo) -> str:
    return (f"Paciente de {contexto['idade']} anos, {contexto['sexo']}, no {contexto['setor']} "
            f"(leito {contexto['leito']}). Quadro: {contexto['condicao']}. "
            f"Protocolo aplicável: {protocolo.codigo} — {protocolo.titulo}.")


def _bloco_alertas(alertas: list) -> str:
    if not alertas:
        return "**Alertas ativos:** nenhum alerta disparado pelas regras institucionais para este caso."
    linhas = [f"**Alertas ativos ({len(alertas)}):**"]
    linhas += [f"- [{a['severidade'].upper()}] {a['mensagem']}" for a in alertas]
    return "\n".join(linhas)


def _bloco_pendencias(pendencias: dict) -> str:
    faltantes = pendencias.get("pendentes", []) + pendencias.get("nao_solicitados", [])
    if not faltantes:
        return ("**Pendências:** o rol de exames obrigatórios do protocolo está completo, "
                "com resultado registrado em prontuário.")
    linhas = ["**Pendências segundo o protocolo:**"]
    linhas += [f"- {exame}" for exame in faltantes]
    return "\n".join(linhas)


def _bloco_prioridades(alertas: list, pendencias: dict, protocolo) -> str:
    passos: list[str] = []
    for alerta in alertas[:2]:
        passos.append(alerta["mensagem"])
    faltantes = pendencias.get("pendentes", []) + pendencias.get("nao_solicitados", [])
    if faltantes:
        passos.append("Regularizar os exames pendentes do protocolo: " + "; ".join(faltantes) + ".")
    for conduta in protocolo.conduta[:2]:
        if len(passos) >= 4:
            break
        passos.append(conduta)
    return "**Prioridades sugeridas:**\n" + "\n".join(
        f"{i}. {passo}" for i, passo in enumerate(passos, 1))


def _fonte(protocolo, contexto: dict) -> str:
    # Sem a data de revisão: ver a justificativa em `build_dataset._fonte`.
    return (f"Fonte: {protocolo.codigo} v{protocolo.versao}, seções 3 — Conduta e "
            f"4 — Exames obrigatórios; prontuário {contexto['paciente_id']}.")


def redigir_referencia(foco: str, contexto: dict, pendencias: dict, alertas: list, protocolo) -> str:
    """Resposta de referência, montada a partir do que já foi apurado."""
    partes = [f"**Resumo do caso.** {_resumo(contexto, protocolo)}", ""]

    if foco in ("prioridades", "conformidade", "resumo"):
        partes += [_bloco_alertas(alertas), ""]
    if foco in ("prioridades", "resumo"):
        partes += [_bloco_prioridades(alertas, pendencias, protocolo), ""]
    if foco in ("pendencias", "prioridades", "conformidade"):
        partes += [_bloco_pendencias(pendencias), ""]
    if foco == "conformidade":
        conformidade = ("O caso está em conformidade com o rol de exames do protocolo."
                        if pendencias.get("conforme")
                        else "O caso apresenta desvios em relação ao protocolo — ver pendências e alertas acima.")
        partes += [f"**Conformidade.** {conformidade}", ""]
    if foco == "resumo":
        partes += ["**O que o protocolo determina:**",
                   "\n".join(f"- {c}" for c in protocolo.conduta[:3]), ""]

    partes += [_fonte(protocolo, contexto), "", f"> {DISCLAIMER}"]
    return "\n".join(partes).strip()


def montar_prompt(pergunta: str, contexto: dict, pendencias: dict, alertas: list,
                  trechos: list[dict]) -> str:
    """Reproduz exatamente o prompt humano que o nó `redigir_resposta` monta."""
    alertas_texto = "\n".join(
        f"- [{a['severidade'].upper()}] {a['codigo']}: {a['mensagem']} (fonte: {a['protocolo']})"
        for a in alertas)
    faltantes = pendencias.get("pendentes", []) + pendencias.get("nao_solicitados", [])
    pendencias_texto = ("\n".join(f"- {e}" for e in faltantes) if faltantes
                        else "- Nenhuma pendência: rol de exames do protocolo completo.")
    protocolos_texto = "\n\n".join(f"[{t['citacao']}] {t['titulo']}\n{t['texto']}" for t in trechos)

    return TEMPLATE_HUMANO.format(
        contexto_paciente=secao("Contexto do paciente", ehr.format_context(contexto)),
        pendencias=secao("Pendências", pendencias_texto),
        alertas=secao("Alertas ativos", alertas_texto),
        protocolos=secao("Trechos de protocolo recuperados", protocolos_texto),
        pergunta=pergunta,
    )


def gerar_consulta_geral(seed: int = SEED) -> list[dict]:
    """Exemplos do caminho **sem prontuário** do fluxo.

    Segunda lição da mesma família da primeira: não basta o formato do prompt
    principal, é preciso cobrir **todos os caminhos** do grafo. Treinado só com
    consultas contextualizadas, o modelo aprendeu a abrir toda resposta com
    "Resumo do caso" — e, numa pergunta geral sem paciente nenhum, **inventou um
    paciente** ("Paciente de 60 anos, homem"), com código de protocolo fabricado
    junto.

    Aqui o prompt tem só a seção de protocolos recuperados, como o nó
    `recuperar_protocolo` monta quando não há prontuário, e a resposta de
    referência é a do FAQ curado: objetiva, ancorada no protocolo, sem caso.
    """
    from .synth import _FAQ_TEMPLATES

    exemplos: list[dict] = []
    for codigo, pergunta, resposta in _FAQ_TEMPLATES:
        trechos = buscar_protocolo(pergunta, k=3, protocolo=codigo)
        if not trechos:
            continue
        protocolos_texto = "\n\n".join(
            f"[{t['citacao']}] {t['titulo']}\n{t['texto']}" for t in trechos)
        prompt = TEMPLATE_HUMANO.format(
            contexto_paciente="", pendencias="", alertas="",
            protocolos=secao("Trechos de protocolo recuperados", protocolos_texto),
            pergunta=pergunta,
        )
        exemplos.append({
            "categoria": "consulta_geral",
            "pergunta": prompt,
            "resposta": resposta.format(codigo=codigo),
            "fonte": codigo,
            "lang": "pt",
            "dedup_key": f"geral:{codigo}:{pergunta[:40]}",
        })
    return exemplos


def gerar(n_pacientes: int = 16, perguntas_por_paciente: int = 3, seed: int = SEED) -> list[dict]:
    """Gera exemplos contextualizados a partir da base estruturada."""
    ehr.ensure_seeded()
    rng = random.Random(seed + 7)
    pacientes = ehr.list_patients()[:n_pacientes]
    exemplos: list[dict] = []

    for paciente in pacientes:
        paciente_id = paciente["paciente_id"]
        contexto = ehr.clinical_context(paciente_id)
        if contexto is None:
            continue
        pendencias = ehr.pending_exams(paciente_id)
        alertas = [a.as_dict() for a in alert_engine.avaliar(contexto, pendencias)]
        protocolo = PROTOCOLOS_POR_CODIGO[contexto["protocolo"]]

        escolhidas = rng.sample(PERGUNTAS, k=min(perguntas_por_paciente, len(PERGUNTAS)))
        for pergunta, foco in escolhidas:
            trechos = buscar_protocolo(f"{pergunta} {contexto['condicao']}", k=3,
                                       protocolo=contexto["protocolo"])
            exemplos.append({
                "categoria": "consulta_contextualizada",
                "pergunta": montar_prompt(pergunta, contexto, pendencias, alertas, trechos),
                "resposta": redigir_referencia(foco, contexto, pendencias, alertas, protocolo),
                "fonte": protocolo.codigo,
                "lang": "pt",
                "dedup_key": f"contextualizada:{paciente_id}:{foco}",
            })

    return exemplos
