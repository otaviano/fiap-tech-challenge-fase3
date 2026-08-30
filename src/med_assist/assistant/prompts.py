"""Prompts do assistente — a mesma persona do fine-tuning, agora com contexto.

O prompt de sistema vem de `med_assist.persona`, o mesmo texto usado no dataset
de treino. O prompt humano é montado pelos nós do fluxo e tem seções fixas e
nomeadas: o modelo sempre sabe o que é protocolo, o que é dado do paciente e o
que é alerta determinístico — e o redator de contingência consegue recortar
essas mesmas seções quando a LLM está fora.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from ..persona import SYSTEM_PROMPT

TEMPLATE_HUMANO = """{contexto_paciente}{pendencias}{alertas}{protocolos}
Pergunta do profissional: {pergunta}

Instruções para esta resposta:
- Responda em português, de forma direta e objetiva, no registro clínico.
- Use APENAS as informações das seções acima. Se algo não estiver ali, diga que não consta.
- Cite explicitamente o código do protocolo e a seção que fundamentam cada afirmação.
- Se houver alertas ativos, mencione-os antes da conduta sugerida.
- Seja conciso: no máximo 300 palavras. Prefira lista numerada a texto corrido.
- Encerre lembrando que a conduta exige validação do médico assistente."""

ASSISTANT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", TEMPLATE_HUMANO),
])


def secao(titulo: str, conteudo: str) -> str:
    """Formata uma seção do prompt; vazio quando não há conteúdo."""
    if not conteudo:
        return ""
    return f"### {titulo}\n{conteudo}\n\n"
