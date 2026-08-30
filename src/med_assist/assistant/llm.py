"""Conexão com a LLM customizada e o fallback determinístico.

O modelo fine-tunado é servido por um endpoint OpenAI-compatível (o mesmo
`llama-server` da Fase 2, agora carregando o GGUF do modelo treinado com os
dados do hospital). O LangChain fala com ele via `ChatOpenAI`, apontando
`base_url` para o servidor local — nenhum dado de paciente sai da rede.

Quando o servidor está fora do ar, o assistente **não quebra e não inventa**:
cai num redator determinístico que monta a resposta a partir do que os nós já
recuperaram — protocolo, pendências e alertas. A resposta fica mais seca, mas
continua correta, citada e auditável. Isso mantém a demonstração e a suíte de
testes executáveis em qualquer máquina.
"""

from __future__ import annotations

import requests
from langchain_core.messages import AIMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import Runnable, RunnableLambda

from ..config import LLMSettings
from ..safety.audit import get_logger

logger = get_logger("llm")


def is_available(settings: LLMSettings | None = None, timeout: float = 2.0) -> bool:
    """Ping no endpoint — evita esperar o timeout longo em cada chamada."""
    settings = settings or LLMSettings()
    try:
        resposta = requests.get(f"{settings.base_url.rstrip('/')}/models", timeout=timeout)
        return resposta.status_code == 200
    except requests.RequestException:
        return False


def served_model(settings: LLMSettings | None = None, timeout: float = 2.0) -> str | None:
    """Qual modelo está de fato no ar — auditar isso importa: a resposta clínica
    muda conforme o modelo servido seja o base ou o fine-tunado."""
    settings = settings or LLMSettings()
    try:
        resposta = requests.get(f"{settings.base_url.rstrip('/')}/models", timeout=timeout)
        dados = resposta.json()
        modelos = dados.get("data") or dados.get("models") or []
        if modelos:
            primeiro = modelos[0]
            return primeiro.get("id") or primeiro.get("name")
    except (requests.RequestException, ValueError, AttributeError, IndexError):
        return None
    return None


def _fallback_text(prompt_value: ChatPromptValue) -> AIMessage:
    """Redige a resposta a partir do próprio contexto montado pelo fluxo.

    Não é "o modelo pensando": é a apresentação organizada do que as etapas
    determinísticas já apuraram. Por isso nunca extrapola o contexto.
    """
    mensagens = prompt_value.to_messages()
    contexto = mensagens[-1].content if mensagens else ""

    def _secao(rotulo: str) -> str:
        if rotulo not in contexto:
            return ""
        corpo = contexto.split(rotulo, 1)[1]
        for proximo in ("### ", "\n\nPergunta do profissional:"):
            corpo = corpo.split(proximo, 1)[0]
        return corpo.strip()

    protocolo = _secao("### Trechos de protocolo recuperados")
    paciente = _secao("### Contexto do paciente")
    pendencias = _secao("### Pendências")
    alertas = _secao("### Alertas ativos")

    partes = ["**Resposta em modo contingência** (servidor da LLM indisponível — "
              "conteúdo montado diretamente das fontes recuperadas, sem geração).", ""]
    if paciente:
        partes += ["**Caso:**", paciente, ""]
    if protocolo:
        partes += ["**O que o protocolo determina:**", protocolo, ""]
    if pendencias:
        partes += ["**Pendências do caso:**", pendencias, ""]
    if alertas:
        partes += ["**Alertas ativos:**", alertas, ""]
    if not (protocolo or paciente):
        partes += ["Não há trecho de protocolo interno correspondente à consulta. "
                   "Reformule com o termo clínico ou informe o código do protocolo.", ""]

    return AIMessage(content="\n".join(partes).strip())


def build_llm(settings: LLMSettings | None = None, force_fallback: bool = False) -> tuple[Runnable, str]:
    """Devolve (runnable de chat, modo) — 'llm' ou 'fallback'."""
    settings = settings or LLMSettings()

    if force_fallback or not is_available(settings):
        logger.warning("LLM indisponível em %s — usando redator determinístico", settings.base_url)
        return RunnableLambda(_fallback_text), "fallback"

    from langchain_openai import ChatOpenAI

    modelo = ChatOpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        model=settings.model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout,
    )
    logger.info("LLM conectada: %s (%s)", settings.model, settings.base_url)
    return modelo, "llm"
