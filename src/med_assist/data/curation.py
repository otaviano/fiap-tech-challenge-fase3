"""Curadoria do dataset: deduplicação, filtros de qualidade e split.

A anonimização remove o que não pode entrar; a curadoria remove o que não
*deveria* entrar. São dois gates distintos e o segundo depende do primeiro:
nenhum exemplo com PII residual sobrevive aqui.
"""

from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from ..config import SEED
from .anonymize import residual_pii

MIN_PERGUNTA = 12
MIN_RESPOSTA = 60
MAX_RESPOSTA = 2000
NEAR_DUP_THRESHOLD = 0.85
SHINGLE = 5


@dataclass
class CurationReport:
    """Contabilidade do que entrou, do que saiu e por quê."""

    entrada: int = 0
    aprovados: int = 0
    descartes: Counter = field(default_factory=Counter)
    por_categoria: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict:
        return {
            "exemplos_de_entrada": self.entrada,
            "exemplos_aprovados": self.aprovados,
            "descartados": sum(self.descartes.values()),
            "motivos_de_descarte": dict(sorted(self.descartes.items(), key=lambda kv: -kv[1])),
            "aprovados_por_categoria": dict(sorted(self.por_categoria.items())),
        }


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", text).strip()


def _shingles(text: str, size: int = SHINGLE) -> set[str]:
    palavras = _normalize(text).split()
    if len(palavras) < size:
        return {" ".join(palavras)}
    return {" ".join(palavras[i:i + size]) for i in range(len(palavras) - size + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def curate(exemplos: list[dict], seed: int = SEED) -> tuple[list[dict], CurationReport]:
    """Aplica os filtros na ordem em que eles são mais baratos de avaliar."""
    report = CurationReport(entrada=len(exemplos))
    vistos_hash: set[str] = set()
    shingles_aprovados: list[tuple[str, set[str]]] = []
    aprovados: list[dict] = []

    for exemplo in exemplos:
        pergunta = exemplo["pergunta"].strip()
        resposta = exemplo["resposta"].strip()
        categoria = exemplo.get("categoria", "desconhecida")

        if len(pergunta) < MIN_PERGUNTA:
            report.descartes["pergunta_curta"] += 1
            continue
        if not (MIN_RESPOSTA <= len(resposta) <= MAX_RESPOSTA):
            report.descartes["resposta_fora_do_tamanho"] += 1
            continue

        # Gate de privacidade: PII que escapou da anonimização invalida o exemplo.
        residuo = residual_pii(f"{pergunta}\n{resposta}")
        if residuo:
            report.descartes["pii_residual"] += 1
            continue

        # Exemplos internos precisam carregar a fonte — é o que ensina explainability.
        if categoria in {"protocolo_qa", "faq", "faq_contextualizada", "documento"} \
                and not re.search(r"PROT-[A-Z]{3}-\d{3}", resposta):
            report.descartes["sem_citacao_de_fonte"] += 1
            continue

        chave = hashlib.sha1(_normalize(pergunta).encode()).hexdigest()
        if chave in vistos_hash:
            report.descartes["duplicata_exata"] += 1
            continue

        sh = _shingles(pergunta)
        if any(_jaccard(sh, outro) >= NEAR_DUP_THRESHOLD for _, outro in shingles_aprovados):
            report.descartes["quase_duplicata"] += 1
            continue

        vistos_hash.add(chave)
        shingles_aprovados.append((chave, sh))
        aprovados.append(exemplo)
        report.por_categoria[categoria] += 1

    report.aprovados = len(aprovados)
    return aprovados, report


def split(exemplos: list[dict], val_ratio: float = 0.15, seed: int = SEED) -> tuple[list[dict], list[dict]]:
    """Split estratificado por categoria — validação com a mesma composição do treino."""
    rng = random.Random(seed)
    por_categoria: dict[str, list[dict]] = {}
    for exemplo in exemplos:
        por_categoria.setdefault(exemplo.get("categoria", "desconhecida"), []).append(exemplo)

    treino: list[dict] = []
    validacao: list[dict] = []
    for categoria in sorted(por_categoria):
        grupo = por_categoria[categoria][:]
        rng.shuffle(grupo)
        n_val = max(1, round(len(grupo) * val_ratio)) if len(grupo) >= 4 else 0
        validacao.extend(grupo[:n_val])
        treino.extend(grupo[n_val:])

    rng.shuffle(treino)
    rng.shuffle(validacao)
    return treino, validacao
