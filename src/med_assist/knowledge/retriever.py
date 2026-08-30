"""Recuperação de trechos de protocolo (BM25 puro, sem dependência externa).

Por que BM25 e não embeddings: o corpus são 8 protocolos institucionais em
português, com vocabulário técnico fixo e códigos (`PROT-EME-002`) que precisam
casar literalmente. Busca léxica acerta esse caso, roda em milissegundos sem
modelo de embedding e — o que mais importa aqui — é **auditável**: dá para
mostrar ao médico por que aquele trecho foi recuperado, com o score e os termos
que casaram. Explainability começa na recuperação, não só na resposta.

Cada documento indexado é uma *seção* de protocolo, não o arquivo inteiro: é a
granularidade que a citação da fonte precisa ter ("PROT-EME-002, seção 3 —
Conduta").
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from ..config import KB_DIR

K1 = 1.5
B = 0.75

_STOPWORDS = {
    "a", "à", "às", "ao", "aos", "as", "com", "como", "da", "das", "de", "do", "dos", "e", "em",
    "entre", "essa", "esse", "esta", "este", "eu", "há", "isso", "já", "me", "mais", "mas", "na",
    "nas", "no", "nos", "não", "o", "os", "ou", "para", "pela", "pelo", "por", "posso", "pra",
    "que", "quando", "se", "sem", "ser", "só", "sobre", "sua", "suas", "seu", "seus", "também",
    "um", "uma", "uns", "umas", "é", "são", "ter", "tem", "the", "of", "and", "is", "in", "to",
}


def tokenize(text: str) -> list[str]:
    """Minúsculas, sem acento, preservando códigos de protocolo como um token."""
    text = text.lower()
    codigos = re.findall(r"prot-[a-z]{3}-\d{3}", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    tokens = [t for t in re.findall(r"[a-z0-9]+", text) if len(t) > 2 and t not in _STOPWORDS]
    return tokens + codigos


@dataclass(frozen=True)
class Trecho:
    """Uma seção de protocolo, com o suficiente para virar citação."""

    protocolo: str
    titulo: str
    especialidade: str
    versao: str
    revisao: str
    secao: str
    texto: str

    @property
    def citacao(self) -> str:
        return f"{self.protocolo} v{self.versao}, seção {self.secao}"

    def como_contexto(self) -> str:
        return f"[{self.citacao}] {self.titulo} — {self.especialidade}\n{self.texto}"


def _parse_protocolo(caminho: Path) -> list[Trecho]:
    conteudo = caminho.read_text(encoding="utf-8")
    cabecalho = conteudo.split("## ", 1)[0]

    def _campo(rotulo: str, padrao: str = "") -> str:
        m = re.search(rf"\*\*{rotulo}:\*\*\s*([^\n|]+)", cabecalho)
        return m.group(1).strip().rstrip("  ") if m else padrao

    titulo_linha = conteudo.splitlines()[0].lstrip("# ").strip()
    codigo, _, titulo = titulo_linha.partition("—")
    versao_linha = re.search(r"\*\*Versão:\*\*\s*([\d.]+)", cabecalho)
    revisao_linha = re.search(r"\*\*Última revisão:\*\*\s*([\d-]+)", cabecalho)

    trechos: list[Trecho] = []
    for bloco in re.split(r"\n## ", conteudo)[1:]:
        secao, _, corpo = bloco.partition("\n")
        corpo = corpo.split("---")[0].strip()
        if not corpo:
            continue
        # A seção de referências bibliográficas não responde pergunta clínica e,
        # por ser curta, o BM25 a favorece indevidamente na normalização por
        # tamanho. Fica fora do índice.
        if "referência" in secao.lower():
            continue
        trechos.append(Trecho(
            protocolo=codigo.strip(),
            titulo=titulo.strip(),
            especialidade=_campo("Especialidade"),
            versao=versao_linha.group(1) if versao_linha else "?",
            revisao=revisao_linha.group(1) if revisao_linha else "?",
            secao=secao.strip(),
            texto=corpo,
        ))
    return trechos


class ProtocolRetriever:
    """Índice BM25 sobre as seções dos protocolos anonimizados."""

    def __init__(self, kb_dir: Path = KB_DIR):
        self.kb_dir = kb_dir
        self.trechos: list[Trecho] = []
        for caminho in sorted(kb_dir.glob("*.md")):
            self.trechos.extend(_parse_protocolo(caminho))

        self._tokens = [tokenize(f"{t.protocolo} {t.titulo} {t.especialidade} {t.secao} {t.texto}")
                        for t in self.trechos]
        self._len = [len(t) for t in self._tokens]
        self._avg_len = (sum(self._len) / len(self._len)) if self._len else 0.0

        self._df: dict[str, int] = {}
        for tokens in self._tokens:
            for termo in set(tokens):
                self._df[termo] = self._df.get(termo, 0) + 1
        self._tf: list[dict[str, int]] = []
        for tokens in self._tokens:
            contagem: dict[str, int] = {}
            for termo in tokens:
                contagem[termo] = contagem.get(termo, 0) + 1
            self._tf.append(contagem)

    def __len__(self) -> int:
        return len(self.trechos)

    def _idf(self, termo: str) -> float:
        n = len(self.trechos)
        df = self._df.get(termo, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, consulta: str, k: int = 3, protocolo: str | None = None) -> list[tuple[Trecho, float]]:
        """Top-k seções mais relevantes, com o score BM25 que justifica a escolha."""
        termos = tokenize(consulta)
        if not termos or not self.trechos:
            return []

        pontuados: list[tuple[Trecho, float]] = []
        for i, trecho in enumerate(self.trechos):
            if protocolo and trecho.protocolo != protocolo:
                continue
            tf = self._tf[i]
            score = 0.0
            for termo in termos:
                f = tf.get(termo, 0)
                if not f:
                    continue
                denom = f + K1 * (1 - B + B * self._len[i] / (self._avg_len or 1))
                score += self._idf(termo) * (f * (K1 + 1)) / denom
            if score > 0:
                pontuados.append((trecho, round(score, 4)))

        pontuados.sort(key=lambda par: par[1], reverse=True)
        return pontuados[:k]

    def por_protocolo(self, codigo: str) -> list[Trecho]:
        return [t for t in self.trechos if t.protocolo == codigo]


_singleton: ProtocolRetriever | None = None


def get_retriever(kb_dir: Path = KB_DIR) -> ProtocolRetriever:
    """Índice compartilhado — construído uma vez por processo."""
    global _singleton
    if _singleton is None or _singleton.kb_dir != kb_dir:
        _singleton = ProtocolRetriever(kb_dir)
    return _singleton
