"""Configuração central do assistente (caminhos e variáveis de ambiente)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
STRUCTURED_DIR = DATA_DIR / "structured"
EXTERNAL_DIR = DATA_DIR / "external"

KB_DIR = PROCESSED_DIR / "protocolos"          # base de conhecimento anonimizada (RAG)
TRAIN_FILE = PROCESSED_DIR / "train.jsonl"
VAL_FILE = PROCESSED_DIR / "val.jsonl"
EHR_DB = STRUCTURED_DIR / "prontuarios.db"

RESULTS_DIR = ROOT / "results"
LOGS_DIR = RESULTS_DIR / "logs"
EVAL_DIR = RESULTS_DIR / "eval"

AUDIT_FILE = LOGS_DIR / "auditoria.jsonl"
APP_LOG = LOGS_DIR / "med_assist.log"

HOSPITAL = "Hospital Santa Helena"
SEED = 42


@dataclass(frozen=True)
class LLMSettings:
    """Endpoint OpenAI-compatível (llama.cpp, Ollama, LM Studio, vLLM...).

    O padrão aponta para o mesmo `llama-server` usado na Fase 2 — o que muda na
    Fase 3 é o modelo servido: o Qwen fine-tunado com os dados do hospital.
    """

    base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
    model: str = os.getenv("LLM_MODEL", "med-assist")
    api_key: str = os.getenv("LLM_API_KEY", "sk-no-key-required")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "700"))
    timeout: float = float(os.getenv("LLM_TIMEOUT", "120"))


def ensure_dirs() -> None:
    """Cria as pastas de saída que o pipeline assume existirem."""
    for path in (RAW_DIR, PROCESSED_DIR, STRUCTURED_DIR, KB_DIR, RESULTS_DIR, LOGS_DIR, EVAL_DIR):
        path.mkdir(parents=True, exist_ok=True)
