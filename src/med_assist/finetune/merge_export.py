"""Merge do adapter LoRA e exportação para GGUF.

O caminho de produção do projeto: o adapter treinado é fundido nos pesos do
modelo base, o modelo resultante é convertido para GGUF e quantizado, e sobe no
mesmo `llama-server` que a Fase 2 já usava. O que muda entre as fases é o modelo
servido — a infraestrutura de inferência é a mesma.

    med-assist finetune merge
    med-assist finetune gguf --llama-cpp ~/llama.cpp --quant Q4_K_M
    llama-server -m models/med-assist-Q4_K_M.gguf --alias med-assist --port 8080

Por que fundir em vez de servir base + adapter: o llama.cpp carrega um GGUF
único com muito menos fricção, e o artefato fundido é o que se versiona,
assina e promove entre ambientes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ..config import ROOT
from ..safety.audit import get_logger
from .train import DEFAULT_BASE_MODEL, DEFAULT_OUTPUT

logger = get_logger("export")

DEFAULT_MERGED = ROOT / "models" / "med-assist-merged"


def merge(adapter_dir: Path = DEFAULT_OUTPUT, base_model: str | None = None,
          out_dir: Path = DEFAULT_MERGED) -> Path:
    """Funde o adapter no modelo base e salva o modelo completo em `out_dir`."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = Path(adapter_dir)
    if base_model is None:
        resumo = adapter_dir / "training_summary.json"
        base_model = (json.loads(resumo.read_text(encoding="utf-8"))["config"]["base_model"]
                      if resumo.exists() else DEFAULT_BASE_MODEL)

    logger.info("Fundindo %s em %s", adapter_dir, base_model)
    modelo = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32)
    modelo = PeftModel.from_pretrained(modelo, adapter_dir)
    modelo = modelo.merge_and_unload()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    modelo.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(out_dir)

    if (adapter_dir / "training_summary.json").exists():
        shutil.copy(adapter_dir / "training_summary.json", out_dir / "training_summary.json")

    logger.info("Modelo fundido salvo em %s", out_dir)
    return out_dir


def to_gguf(merged_dir: Path = DEFAULT_MERGED, llama_cpp: Path | None = None,
            quant: str = "Q4_K_M", out_file: Path | None = None) -> Path:
    """Converte o modelo fundido para GGUF e quantiza, usando o llama.cpp local."""
    merged_dir = Path(merged_dir)
    llama_cpp = Path(llama_cpp or os.getenv("LLAMA_CPP_DIR") or Path.home() / "llama.cpp")
    conversor = llama_cpp / "convert_hf_to_gguf.py"
    if not conversor.exists():
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py não encontrado em {llama_cpp}. "
            "Informe o caminho do llama.cpp com --llama-cpp.")

    modelos_dir = ROOT / "models"
    modelos_dir.mkdir(parents=True, exist_ok=True)
    f16 = modelos_dir / "med-assist-f16.gguf"
    out_file = Path(out_file or modelos_dir / f"med-assist-{quant}.gguf")

    logger.info("Convertendo para GGUF f16: %s", f16)
    subprocess.run([sys.executable, str(conversor), str(merged_dir),
                    "--outfile", str(f16), "--outtype", "f16"], check=True)

    quantizador = next((c for c in (llama_cpp / "build" / "bin" / "llama-quantize",
                                    llama_cpp / "llama-quantize") if c.exists()), None)
    if quantizador is None:
        logger.warning(
            "llama-quantize não encontrado em %s — ficando no f16 (%s). A quantização é "
            "opcional: para tê-la, compile o alvo com "
            "`cmake --build build --target llama-quantize` e rode este comando de novo.",
            llama_cpp, f16)
        return f16

    logger.info("Quantizando para %s: %s", quant, out_file)
    subprocess.run([str(quantizador), str(f16), str(out_file), quant], check=True)
    logger.info("GGUF pronto: %s (%.1f MB)", out_file, out_file.stat().st_size / 1e6)
    return out_file
