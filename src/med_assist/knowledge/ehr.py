"""Prontuário eletrônico (base estruturada) sobre SQLite.

É a "base de dados estruturada" que o assistente consulta para contextualizar as
respostas. O banco é gerado a partir de `data/raw/pacientes.json` e vive em
`data/structured/prontuarios.db` (não versionado — o seed é determinístico).

Decisão de privacidade que atravessa este módulo: **o banco guarda o cadastro
completo, mas o que sai daqui para o LLM é minimizado**. `clinical_context()`
devolve apenas o que sustenta a decisão clínica — idade, sexo, quadro, exames,
medicações, alergias, escores. CPF, CNS, RG, telefone, e-mail e endereço nunca
entram no prompt. Quem precisa desses dados usa o prontuário, não o assistente.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ..config import EHR_DB, RAW_DIR
from ..data.protocol_source import PROTOCOLOS_POR_CODIGO

SCHEMA = """
CREATE TABLE IF NOT EXISTS pacientes (
    paciente_id   TEXT PRIMARY KEY,
    nome          TEXT NOT NULL,
    cpf           TEXT,
    cns           TEXT,
    nascimento    TEXT,
    idade         INTEGER,
    sexo          TEXT,
    telefone      TEXT,
    email         TEXT,
    endereco      TEXT,
    convenio      TEXT,
    setor         TEXT,
    leito         TEXT,
    admissao      TEXT,
    queixa        TEXT,
    condicao      TEXT,
    protocolo     TEXT,
    alergias      TEXT,
    comorbidades  TEXT,
    sinais_vitais TEXT,
    escores       TEXT,
    medico_assistente TEXT
);

CREATE TABLE IF NOT EXISTS exames (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id   TEXT NOT NULL REFERENCES pacientes(paciente_id),
    exame         TEXT NOT NULL,
    status        TEXT NOT NULL,
    solicitado_em TEXT,
    resultado     TEXT
);

CREATE TABLE IF NOT EXISTS medicacoes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id  TEXT NOT NULL REFERENCES pacientes(paciente_id),
    medicamento  TEXT NOT NULL,
    dose         TEXT,
    via          TEXT,
    frequencia   TEXT
);

CREATE INDEX IF NOT EXISTS idx_exames_paciente ON exames(paciente_id);
CREATE INDEX IF NOT EXISTS idx_medicacoes_paciente ON medicacoes(paciente_id);
CREATE INDEX IF NOT EXISTS idx_pacientes_setor ON pacientes(setor);
"""


def connect(db_path: Path = EHR_DB) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def _sessao(db_path: Path = EHR_DB) -> Iterator[sqlite3.Connection]:
    """Conexão com commit/rollback **e** fechamento garantido.

    `with sqlite3.connect(...)` sozinho só controla a transação — a conexão
    continua aberta até o coletor de lixo passar (daí os `ResourceWarning`).
    """
    conn = connect(db_path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def seed(db_path: Path = EHR_DB, pacientes_json: Path | None = None) -> int:
    """(Re)cria o banco a partir do JSON de pacientes sintéticos."""
    pacientes_json = pacientes_json or (RAW_DIR / "pacientes.json")
    pacientes = json.loads(pacientes_json.read_text(encoding="utf-8"))

    with _sessao(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.execute("DELETE FROM exames")
        conn.execute("DELETE FROM medicacoes")
        conn.execute("DELETE FROM pacientes")
        for p in pacientes:
            conn.execute(
                """INSERT INTO pacientes VALUES
                   (:paciente_id,:nome,:cpf,:cns,:nascimento,:idade,:sexo,:telefone,:email,
                    :endereco,:convenio,:setor,:leito,:admissao,:queixa,:condicao,:protocolo,
                    :alergias,:comorbidades,:sinais_vitais,:escores,:medico_assistente)""",
                {**p,
                 "alergias": json.dumps(p["alergias"], ensure_ascii=False),
                 "comorbidades": json.dumps(p["comorbidades"], ensure_ascii=False),
                 "sinais_vitais": json.dumps(p["sinais_vitais"], ensure_ascii=False),
                 "escores": json.dumps(p["escores"], ensure_ascii=False)},
            )
            conn.executemany(
                "INSERT INTO exames (paciente_id, exame, status, solicitado_em, resultado) "
                "VALUES (?,?,?,?,?)",
                [(p["paciente_id"], e["exame"], e["status"], e["solicitado_em"], e["resultado"])
                 for e in p["exames"]],
            )
            conn.executemany(
                "INSERT INTO medicacoes (paciente_id, medicamento, dose, via, frequencia) VALUES (?,?,?,?,?)",
                [(p["paciente_id"], m["medicamento"], m["dose"], m["via"], m["frequencia"])
                 for m in p["medicacoes"]],
            )
    return len(pacientes)


def ensure_seeded(db_path: Path = EHR_DB) -> None:
    """Cria o banco na primeira consulta — a demo não exige passo manual."""
    if not db_path.exists() or db_path.stat().st_size == 0:
        seed(db_path)


def _row_to_patient(row: sqlite3.Row) -> dict:
    paciente = dict(row)
    for campo in ("alergias", "comorbidades", "sinais_vitais", "escores"):
        paciente[campo] = json.loads(paciente[campo]) if paciente[campo] else None
    return paciente


def list_patients(db_path: Path = EHR_DB, setor: str | None = None) -> list[dict]:
    ensure_seeded(db_path)
    with _sessao(db_path) as conn:
        if setor:
            rows = conn.execute(
                "SELECT paciente_id, nome, idade, sexo, setor, leito, condicao, protocolo "
                "FROM pacientes WHERE setor LIKE ? ORDER BY paciente_id", (f"%{setor}%",)).fetchall()
        else:
            rows = conn.execute(
                "SELECT paciente_id, nome, idade, sexo, setor, leito, condicao, protocolo "
                "FROM pacientes ORDER BY paciente_id").fetchall()
    return [dict(r) for r in rows]


def get_patient(paciente_id: str, db_path: Path = EHR_DB) -> dict | None:
    """Registro completo, incluindo exames e medicações. Uso interno.

    A identificação é **exata**: número do prontuário ou, como atalho, o leito —
    este só quando um único paciente o ocupa. Nunca há busca aproximada: um id
    digitado errado tem de dar "não encontrado", não o contexto, os alertas e as
    notificações de outro paciente. (A versão anterior caía num `nome LIKE`, e
    `"a"` devolvia o primeiro prontuário da base; leitos como E-24 aparecem em
    mais de um registro, e o primeiro vencia em silêncio.)
    """
    ensure_seeded(db_path)
    chave = (paciente_id or "").strip().upper()
    if not chave:
        return None
    with _sessao(db_path) as conn:
        row = conn.execute("SELECT * FROM pacientes WHERE paciente_id = ?", (chave,)).fetchone()
        if row is None:
            por_leito = conn.execute("SELECT * FROM pacientes WHERE leito = ?", (chave,)).fetchall()
            row = por_leito[0] if len(por_leito) == 1 else None
        if row is None:
            return None
        paciente = _row_to_patient(row)
        pid = paciente["paciente_id"]
        paciente["exames"] = [dict(r) for r in conn.execute(
            "SELECT exame, status, solicitado_em, resultado FROM exames WHERE paciente_id = ?", (pid,))]
        paciente["medicacoes"] = [dict(r) for r in conn.execute(
            "SELECT medicamento, dose, via, frequencia FROM medicacoes WHERE paciente_id = ?", (pid,))]
    return paciente


def pending_exams(paciente_id: str, db_path: Path = EHR_DB) -> dict:
    """Pendências do caso: o que o protocolo exige e ainda não tem resultado.

    A resposta é verificável por construção — é a diferença entre a lista de
    exames obrigatórios do protocolo e o que está registrado como concluído no
    prontuário, não uma opinião do modelo.
    """
    paciente = get_patient(paciente_id, db_path)
    if paciente is None:
        return {"encontrado": False, "paciente_id": paciente_id}

    concluidos = {e["exame"] for e in paciente["exames"] if e["status"] == "concluído"}
    pendentes_registrados = [e["exame"] for e in paciente["exames"] if e["status"] != "concluído"]

    protocolo = PROTOCOLOS_POR_CODIGO.get(paciente["protocolo"])
    exigidos = list(protocolo.exames_obrigatorios) if protocolo else []
    nao_solicitados = [e for e in exigidos
                       if e not in concluidos and e not in pendentes_registrados]

    return {
        "encontrado": True,
        "paciente_id": paciente["paciente_id"],
        "protocolo": paciente["protocolo"],
        "exames_obrigatorios": exigidos,
        "concluidos": sorted(concluidos),
        "pendentes": pendentes_registrados,
        "nao_solicitados": nao_solicitados,
        "conforme": not pendentes_registrados and not nao_solicitados,
    }


#: Campos que jamais entram no prompt do LLM (minimização de dados).
CAMPOS_IDENTIFICAVEIS = ("nome", "cpf", "cns", "nascimento", "telefone", "email",
                         "endereco", "convenio", "medico_assistente")


def clinical_context(paciente_id: str, db_path: Path = EHR_DB) -> dict | None:
    """Visão minimizada do paciente — é o único formato que chega ao modelo."""
    paciente = get_patient(paciente_id, db_path)
    if paciente is None:
        return None

    return {
        "paciente_id": paciente["paciente_id"],
        "idade": paciente["idade"],
        "sexo": paciente["sexo"],
        "setor": paciente["setor"],
        "leito": paciente["leito"],
        "admissao": paciente["admissao"],
        "queixa": paciente["queixa"],
        "condicao": paciente["condicao"],
        "protocolo": paciente["protocolo"],
        "alergias": paciente["alergias"],
        "comorbidades": paciente["comorbidades"],
        "sinais_vitais": paciente["sinais_vitais"],
        "escores": paciente["escores"],
        "exames": paciente["exames"],
        "medicacoes": paciente["medicacoes"],
    }


def format_context(contexto: dict) -> str:
    """Renderiza o contexto minimizado no texto que vai ao prompt."""
    exames = "\n".join(
        f"  - {e['exame']}: {e['status']}"
        + (f" — {e['resultado']}" if e.get("resultado") else "")
        for e in contexto["exames"])
    meds = "\n".join(
        f"  - {m['medicamento']} {m['dose']} {m['via']} {m['frequencia']}"
        for m in contexto["medicacoes"]) or "  - nenhuma prescrição ativa registrada"
    vitais = ", ".join(f"{k} {v}" for k, v in (contexto["sinais_vitais"] or {}).items())
    escores = ", ".join(f"{k} {v}" for k, v in (contexto["escores"] or {}).items()) or "não registrados"

    return (
        f"Prontuário {contexto['paciente_id']} (dados mínimos necessários):\n"
        f"- Idade/sexo: {contexto['idade']} anos, {contexto['sexo']}\n"
        f"- Local: {contexto['setor']}, leito {contexto['leito']} (admissão em {contexto['admissao']})\n"
        f"- Queixa: {contexto['queixa']}\n"
        f"- Hipótese/condição: {contexto['condicao']}\n"
        f"- Protocolo aplicável: {contexto['protocolo']}\n"
        f"- Comorbidades: {', '.join(contexto['comorbidades'])}\n"
        f"- Alergias: {', '.join(contexto['alergias'])}\n"
        f"- Sinais vitais: {vitais}\n"
        f"- Escores: {escores}\n"
        f"- Exames:\n{exames}\n"
        f"- Medicações ativas:\n{meds}"
    )
