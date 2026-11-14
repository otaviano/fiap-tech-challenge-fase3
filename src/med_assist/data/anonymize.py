"""Anonimização (pseudonimização) de dados médicos antes do fine-tuning.

Regra do projeto: **nenhum texto chega ao treino ou à base de conhecimento sem
passar por aqui**. O corpus bruto em `data/raw/` contém PII plantada de
propósito; o corpus processado em `data/processed/` é o único que alimenta o
modelo.

Estratégia: *pseudonimização determinística*. Cada valor identificado vira um
token estável derivado de HMAC-SHA256 com sal — o mesmo CPF sempre vira o mesmo
`[CPF_4F2A]`. Isso preserva a coerência referencial do texto (o modelo aprende
que um documento fala de uma única pessoa) sem manter o dado original, e não é
reversível sem o sal.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

#: Sal usado na derivação dos pseudônimos. Em produção viria de um cofre de
#: segredos; aqui o padrão é fixo para manter o dataset reprodutível.
SALT = os.getenv("ANON_SALT", "hsh-fase3-salt").encode()

_PRENOMES = (
    "Ana|Antônio|Anderson|Beatriz|Bruno|Carlos|Cláudia|Daniela|Eduardo|Fernando|Fernanda|"
    "Gabriel|Helena|Isabela|João|Joana|José|Juliana|Larissa|Lucas|Luciana|Marcelo|Marcos|"
    "Maria|Mariana|Marina|Paulo|Patrícia|Rafael|Renata|Ricardo|Roberto|Sandra|Sérgio|"
    "Tatiana|Thiago|Vanessa|Vinícius"
)

#: Ordem importa: padrões mais específicos primeiro (um CNS não pode ser comido
#: por um telefone, uma data de nascimento não pode virar data genérica).
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("CPF", re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b")),
    ("CNS", re.compile(r"\b\d{3}\s\d{4}\s\d{4}\s\d{4}\b")),
    ("RG", re.compile(r"\b\d{2}\.\d{3}\.\d{3}-[\dXx]\b")),
    ("PRONTUARIO", re.compile(r"\bPRT-\d{4}-\d{6}\b")),
    ("CRM", re.compile(r"\bCRM[-/ ]?[A-Z]{2}\s?\d{2,3}[.\s]?\d{3}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("TELEFONE", re.compile(r"\(\d{2}\)\s?\d{4,5}-\d{4}\b")),
    ("CEP", re.compile(r"\b\d{5}-\d{3}\b")),
    ("ENDERECO", re.compile(
        r"\b(?:Rua|Av\.|Avenida|Alameda|Travessa|Praça|Rodovia)\s+"
        r"[\wÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç.]+(?:\s+[\wÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç.]+){0,4}"
        r",?\s*(?:n[º°.]?|número)?\s*\d+\b"
    )),
    ("DATA", re.compile(r"\b\d{2}/\d{2}/\d{4}\b")),
    # Nome próprio: título clínico seguido de nome composto, ou prenome conhecido
    # seguido de pelo menos um sobrenome capitalizado.
    # `[ \t]+` em vez de `\s+`: o nome tem de estar na MESMA linha do título.
    # Com `\s+`, "paciente" no fim de um parágrafo casava com a primeira palavra
    # capitalizada do parágrafo seguinte e bloqueava resposta legítima.
    # O lookahead exclui termos clínicos: "Paciente Idoso com sepse" ou um título
    # "Paciente Crítico" não são nome — e, na checagem de saída, bloqueariam a
    # resposta inteira como PII.
    ("NOME", re.compile(
        r"\b(?:Dr\.|Dra\.|Sr\.|Sra\.|paciente|Paciente)[ \t]+"
        r"(?!(?:Idos[oa]|Jovem|Adult[oa]|Masculino|Feminino|Homem|Mulher|Gestante|Puérpera|"
        r"Pediátric[oa]|Neonat[oa]l?|Crític[oa]|Grave|Estável|Instável|Internad[oa]|"
        r"Hospitalizad[oa]|Admitid[oa]|Diabétic[oa]|Hipertens[oa]|Cardiopata|Renal|Oncológic[oa]|"
        r"Séptic[oa]|Obes[oa]|Alérgic[oa]|Portador[a]?|Imunossuprimid[oa]|Neutropênic[oa]|"
        r"Com|Sem|Em|No|Na|De|Do|Da|Que|Sob|Apresenta|Refere|Relata|Evolui|Mantém|Deve|"
        r"Precisa|Tem|Está|Segue|Aguarda)\b)"
        r"(?:[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç]+)"
        r"(?:[ \t]+(?:d[aeo]s?[ \t]+)?[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç]+){0,3}"
    )),
    ("NOME", re.compile(
        rf"\b(?:{_PRENOMES})"
        r"(?:[ \t]+(?:d[aeo]s?[ \t]+)?[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç]+){1,3}"
    )),
    # Regra safe-harbor: idades acima de 89 anos são reidentificáveis.
    ("IDADE_90+", re.compile(r"\b(?:9\d|1\d{2})\s*anos\b")),
)

#: Títulos que devem permanecer no texto (o pseudônimo substitui só o nome).
_TITULOS = ("Dr.", "Dra.", "Sr.", "Sra.", "paciente", "Paciente")


@dataclass
class AnonymizationReport:
    """Quantos identificadores foram removidos, por tipo — vira evidência em `results/`."""

    counts: Counter = field(default_factory=Counter)
    documents: int = 0

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def merge(self, other: "AnonymizationReport") -> None:
        self.counts.update(other.counts)
        self.documents += other.documents

    def as_dict(self) -> dict:
        return {
            "documentos_processados": self.documents,
            "identificadores_removidos": self.total,
            "por_tipo": dict(sorted(self.counts.items(), key=lambda kv: -kv[1])),
        }


def pseudonym(kind: str, value: str) -> str:
    """Token estável e não reversível para um valor identificado.

    O mesmo par (tipo, valor) produz sempre o mesmo token, o que mantém a
    coerência do texto: duas menções ao mesmo médico continuam apontando para a
    mesma pessoa depois da anonimização.
    """
    digest = hmac.new(SALT, f"{kind}:{value.strip().lower()}".encode(), hashlib.sha256)
    return f"[{kind}_{digest.hexdigest()[:4].upper()}]"


def anonymize(text: str) -> tuple[str, AnonymizationReport]:
    """Substitui toda PII reconhecida por pseudônimos estáveis."""
    report = AnonymizationReport(documents=1)
    result = text

    for kind, pattern in PATTERNS:
        def _replace(match: re.Match[str]) -> str:
            raw = match.group(0)
            prefix = ""
            if kind == "NOME":
                for titulo in _TITULOS:
                    if raw.startswith(titulo):
                        prefix, raw = f"{titulo} ", raw[len(titulo):].strip()
                        break
            if kind == "IDADE_90+":
                report.counts[kind] += 1
                return "[IDADE_90+]"
            report.counts[kind] += 1
            return prefix + pseudonym(kind, raw)

        result = pattern.sub(_replace, result)

    return result, report


def anonymize_all(texts: Iterable[str]) -> tuple[list[str], AnonymizationReport]:
    """Anonimiza uma coleção, consolidando um único relatório."""
    consolidated = AnonymizationReport()
    outputs: list[str] = []
    for text in texts:
        clean, report = anonymize(text)
        outputs.append(clean)
        consolidated.merge(report)
    return outputs, consolidated


#: Dado cadastral: nunca pode aparecer em texto de treino nem em resposta gerada.
TIPOS_CADASTRAIS = ("CPF", "CNS", "RG", "EMAIL", "TELEFONE", "ENDERECO", "CEP", "NOME")

#: Identificadores operacionais. O número do prontuário e as datas assistenciais
#: (admissão, solicitação de exame) são o que o profissional usa para saber de
#: qual caso se fala — bloqueá-los na resposta tornaria o assistente inútil.
#: Eles não entram no dataset de treino (a curadoria usa a lista completa), mas
#: circulam na resposta ao profissional, com o acesso registrado na auditoria.
TIPOS_OPERACIONAIS = ("PRONTUARIO", "CRM", "DATA", "IDADE_90+")


def residual_pii(text: str, tipos: tuple[str, ...] | None = None) -> list[str]:
    """PII que sobreviveu à anonimização — usada como *gate* na curadoria e na saída.

    Ignora os pseudônimos já inseridos: `[CPF_4F2A]` não é PII. Por padrão checa
    todos os tipos; passe `tipos=TIPOS_CADASTRAIS` para a checagem de resposta,
    que tolera prontuário e datas assistenciais.
    """
    sem_tokens = re.sub(r"\[[A-Z_+0-9]+_[0-9A-F]{4}\]|\[IDADE_90\+\]", "", text)
    achados: list[str] = []
    for kind, pattern in PATTERNS:
        if tipos is not None and kind not in tipos:
            continue
        achados.extend(f"{kind}:{m.group(0)}" for m in pattern.finditer(sem_tokens))
    return achados
