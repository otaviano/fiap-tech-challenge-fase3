"""Ingestão do MedQuAD — conhecimento médico geral de apoio.

O MedQuAD (Ben Abacha & Demner-Fushman) reúne perguntas e respostas de fontes
oficiais dos NIH. Ele entra no dataset com um papel bem delimitado: ensinar ao
modelo o *registro* de resposta clínica geral (definição, sintomas, tratamento,
exames), enquanto o corpus interno ensina os protocolos, os documentos e os
limites de atuação do hospital.

Os pares estão em inglês e são mantidos assim, com `lang: "en"` e uma instrução
de sistema própria — traduzir por máquina introduziria erro terminológico em
domínio clínico, que é justamente onde ele é mais caro.

Download: `scripts/fetch_medquad.sh` (o corpus não é versionado neste repo).
"""

from __future__ import annotations

import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from ..config import EXTERNAL_DIR, SEED

MEDQUAD_DIR = EXTERNAL_DIR / "MedQuAD"

#: Subsets usados. Os conjuntos MedlinePlus (10 a 12) tiveram as respostas
#: removidas do repositório por direitos autorais — ficam de fora.
SUBSETS = (
    "1_CancerGov_QA",
    "2_GARD_QA",
    "3_GHR_QA",
    "5_NIDDK_QA",
    "6_NINDS_QA",
    "7_SeniorHealth_QA",
    "8_NHLBI_QA_XML",
    "9_CDC_QA",
)

#: Tipos de pergunta compatíveis com o uso clínico do assistente.
QTYPES = ("information", "symptoms", "treatment", "exams and tests", "causes",
          "prevention", "susceptibility", "stages", "outlook")

MAX_ANSWER_CHARS = 1200


def _clean(text: str) -> str:
    """Normaliza o espaçamento herdado do HTML das fontes originais."""
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"\s+-\s+", " - ", text)
    return text


def iter_pairs(root_dir: Path = MEDQUAD_DIR, subsets: tuple[str, ...] = SUBSETS):
    """Percorre os XMLs e devolve pares (pergunta, resposta) utilizáveis."""
    for subset in subsets:
        subset_dir = root_dir / subset
        if not subset_dir.is_dir():
            continue
        for xml_file in sorted(subset_dir.glob("*.xml")):
            try:
                tree = ET.parse(xml_file)
            except ET.ParseError:
                continue
            doc = tree.getroot()
            focus = _clean(doc.findtext("Focus", default=""))
            url = doc.get("url", "")
            source = doc.get("source", subset)
            for pair in doc.iterfind(".//QAPair"):
                question_el = pair.find("Question")
                answer = _clean(pair.findtext("Answer", default=""))
                if question_el is None or not answer:
                    continue
                qtype = (question_el.get("qtype") or "").lower()
                if qtype not in QTYPES:
                    continue
                question = _clean(question_el.text or "")
                if len(question) < 15 or len(answer) < 120:
                    continue
                yield {
                    "pergunta": question,
                    "resposta": answer[:MAX_ANSWER_CHARS].rsplit(" ", 1)[0] + ("…" if len(answer) > MAX_ANSWER_CHARS else ""),
                    "focus": focus,
                    "qtype": qtype,
                    "fonte": f"MedQuAD/{source}",
                    "url": url,
                }


def sample(n: int = 200, root_dir: Path = MEDQUAD_DIR, seed: int = SEED) -> list[dict]:
    """Amostra estratificada por tipo de pergunta — evita um dataset só de definições."""
    pairs = list(iter_pairs(root_dir))
    if not pairs:
        return []

    rng = random.Random(seed)
    por_tipo: dict[str, list[dict]] = {}
    for pair in pairs:
        por_tipo.setdefault(pair["qtype"], []).append(pair)

    cota = max(1, n // len(por_tipo))
    selecionados: list[dict] = []
    for tipo in sorted(por_tipo):
        grupo = por_tipo[tipo]
        rng.shuffle(grupo)
        selecionados.extend(grupo[:cota])

    rng.shuffle(selecionados)
    return selecionados[:n]


def is_available(root_dir: Path = MEDQUAD_DIR) -> bool:
    """O corpus externo é opcional: sem ele, o dataset é montado só com o interno."""
    return root_dir.is_dir() and any((root_dir / s).is_dir() for s in SUBSETS)
