"""Explicabilidade: toda resposta carrega de onde ela veio.

O requisito do desafio é "indicar a fonte da informação utilizada na resposta".
Aqui isso é estrutural, não cosmético: as evidências são coletadas pelos nós do
fluxo (trechos de protocolo recuperados com o score, prontuário consultado,
regras de alerta disparadas) e viajam junto com a resposta até a API, a trilha
de auditoria e o bloco de fontes exibido ao médico.

Consequência prática: se a resposta não tiver evidência, isso aparece — o
guardrail de saída marca `resposta_sem_fonte` e a API devolve a lista vazia.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Evidencia:
    """Um item de proveniência exibível ao médico."""

    tipo: str                 # protocolo | prontuario | regra_clinica | conhecimento_geral
    referencia: str           # "PROT-EME-002 v6.0, seção 3 — Conduta"
    trecho: str = ""          # texto efetivamente usado
    score: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Explicacao:
    """Conjunto de evidências de uma interação."""

    evidencias: list[Evidencia] = field(default_factory=list)

    def add(self, tipo: str, referencia: str, trecho: str = "", score: float | None = None) -> None:
        self.evidencias.append(Evidencia(tipo, referencia, trecho, score))

    def referencias(self) -> list[str]:
        return [e.referencia for e in self.evidencias]

    def as_list(self) -> list[dict]:
        return [e.as_dict() for e in self.evidencias]

    def render(self) -> str:
        """Bloco de fontes anexado ao final da resposta."""
        if not self.evidencias:
            return ""
        linhas = ["", "**Fontes consultadas:**"]
        for e in self.evidencias:
            score = f" (relevância {e.score:.2f})" if e.score is not None else ""
            linhas.append(f"- [{e.tipo}] {e.referencia}{score}")
        return "\n".join(linhas)
