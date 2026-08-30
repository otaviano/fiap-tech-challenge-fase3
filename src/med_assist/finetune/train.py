"""Fine-tuning LoRA da LLM com os dados internos do hospital.

Por que LoRA e não fine-tuning completo: o objetivo é ensinar **formato, registro
e limites** — citar protocolo, recusar ato médico, responder no padrão do
hospital — não injetar conhecimento novo (esse vem por recuperação, em tempo de
consulta). LoRA entrega isso treinando ~1% dos parâmetros, cabe em CPU ou numa
T4 gratuita, e o adapter resultante tem alguns MB, o que torna versionamento e
rollback triviais.

O laço de treino é explícito (AdamW + warmup linear + cosine) em vez do `Trainer`:
são ~60 linhas, o comportamento fica visível para a banca e não há dependência de
mudanças de API entre versões da `transformers`.

    # CPU (validado nesta máquina, sem GPU):
    med-assist finetune train --base-model Qwen/Qwen2.5-0.5B-Instruct --epochs 2

    # GPU (Colab T4):
    med-assist finetune train --base-model Qwen/Qwen2.5-1.5B-Instruct --device cuda --epochs 3
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import EVAL_DIR, ROOT, SEED, TRAIN_FILE, VAL_FILE
from ..safety.audit import get_logger
from .dataset import build_examples, collate, estatisticas, load_jsonl

logger = get_logger("finetune")

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT = ROOT / "models" / "med-assist-lora"

#: Projeções de atenção e MLP — cobertura padrão para arquiteturas Llama/Qwen.
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass
class TrainConfig:
    base_model: str = DEFAULT_BASE_MODEL
    output_dir: Path = DEFAULT_OUTPUT
    train_file: Path = TRAIN_FILE
    val_file: Path = VAL_FILE
    epochs: int = 2
    lr: float = 2e-4
    batch_size: int = 1
    grad_accum: int = 8
    max_len: int = 1024
    warmup_ratio: float = 0.06
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    device: str = "auto"
    limit: int | None = None
    seed: int = SEED
    log_every: int = 10
    historico: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("historico", None)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}


def _resolve_device(escolha: str) -> str:
    import torch

    if escolha != "auto":
        return escolha
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _evaluate(modelo, exemplos: list[dict], pad_id: int, device: str, batch_size: int) -> dict:
    """Loss e perplexidade no conjunto de validação."""
    import torch

    modelo.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(exemplos), batch_size):
            lote = collate(exemplos[i:i + batch_size], pad_id)
            lote = {k: v.to(device) for k, v in lote.items()}
            saida = modelo(**lote)
            n_tokens = int((lote["labels"] != -100).sum())
            total_loss += float(saida.loss) * n_tokens
            total_tokens += n_tokens
    modelo.train()
    loss = total_loss / max(total_tokens, 1)
    return {"val_loss": round(loss, 4), "val_perplexity": round(math.exp(min(loss, 20)), 3)}


def train(config: TrainConfig) -> dict:
    """Executa o fine-tuning e devolve o resumo (também salvo em `results/eval/`)."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = _resolve_device(config.device)
    logger.info("Fine-tuning de %s em %s", config.base_model, device)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    treino_bruto = load_jsonl(config.train_file)
    validacao_bruta = load_jsonl(config.val_file)
    if config.limit:
        treino_bruto = treino_bruto[:config.limit]

    treino = build_examples(tokenizer, treino_bruto, config.max_len)
    validacao = build_examples(tokenizer, validacao_bruta, config.max_len)
    stats_treino = estatisticas(treino)
    logger.info("Treino: %d exemplos (%d tokens, média %.1f) | Validação: %d",
                stats_treino["exemplos"], stats_treino["tokens_total"],
                stats_treino["tokens_medio"], len(validacao))

    modelo = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        dtype=torch.float32 if device == "cpu" else torch.bfloat16,
    )
    modelo.config.use_cache = False

    lora = LoraConfig(
        r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=list(TARGET_MODULES),
    )
    modelo = get_peft_model(modelo, lora)
    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    total = sum(p.numel() for p in modelo.parameters())
    logger.info("Parâmetros treináveis: %s de %s (%.2f%%)",
                f"{treinaveis:,}", f"{total:,}", 100 * treinaveis / total)

    modelo.to(device)
    modelo.train()

    passos_por_epoca = math.ceil(len(treino) / (config.batch_size * config.grad_accum))
    total_passos = passos_por_epoca * config.epochs
    otimizador = torch.optim.AdamW(
        [p for p in modelo.parameters() if p.requires_grad], lr=config.lr, weight_decay=0.01)
    escalonador = get_cosine_schedule_with_warmup(
        otimizador, int(total_passos * config.warmup_ratio), total_passos)

    historico: list[dict] = []
    metricas_iniciais = _evaluate(modelo, validacao, tokenizer.pad_token_id, device, config.batch_size)
    logger.info("Antes do treino: %s", metricas_iniciais)
    historico.append({"epoca": 0, "passo": 0, **metricas_iniciais})

    inicio = time.perf_counter()
    passo_global = 0
    rng = random.Random(config.seed)

    for epoca in range(1, config.epochs + 1):
        ordem = list(range(len(treino)))
        rng.shuffle(ordem)
        acumulado, perdas_epoca = 0.0, []

        for i in range(0, len(ordem), config.batch_size):
            indices = ordem[i:i + config.batch_size]
            lote = collate([treino[j] for j in indices], tokenizer.pad_token_id)
            lote = {k: v.to(device) for k, v in lote.items()}

            saida = modelo(**lote)
            perda = saida.loss / config.grad_accum
            perda.backward()
            perda_valor = float(saida.loss.detach())
            acumulado += perda_valor
            perdas_epoca.append(perda_valor)

            if (i // config.batch_size + 1) % config.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in modelo.parameters() if p.requires_grad], 1.0)
                otimizador.step()
                escalonador.step()
                otimizador.zero_grad()
                passo_global += 1

                if passo_global % config.log_every == 0:
                    media = acumulado / (config.grad_accum * config.log_every)
                    logger.info("época %d | passo %d/%d | loss %.4f | lr %.2e",
                                epoca, passo_global, total_passos, media,
                                escalonador.get_last_lr()[0])
                    historico.append({"epoca": epoca, "passo": passo_global,
                                      "train_loss": round(media, 4),
                                      "lr": escalonador.get_last_lr()[0]})
                    acumulado = 0.0

        metricas = _evaluate(modelo, validacao, tokenizer.pad_token_id, device, config.batch_size)
        media_epoca = sum(perdas_epoca) / max(len(perdas_epoca), 1)
        logger.info("época %d concluída | train_loss %.4f | %s", epoca, media_epoca, metricas)
        historico.append({"epoca": epoca, "passo": passo_global,
                          "train_loss_epoca": round(media_epoca, 4), **metricas})

    duracao = time.perf_counter() - inicio
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    modelo.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    resumo = {
        "config": config.as_dict(),
        "device": device,
        "parametros_treinaveis": treinaveis,
        "parametros_totais": total,
        "percentual_treinavel": round(100 * treinaveis / total, 3),
        "dataset": stats_treino,
        "validacao": len(validacao),
        "passos": passo_global,
        "duracao_s": round(duracao, 1),
        "metricas_iniciais": metricas_iniciais,
        "metricas_finais": historico[-1],
        "historico": historico,
        "adapter": str(output_dir),
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "finetune_history.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "training_summary.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Fine-tuning concluído em %.1f min — adapter em %s", duracao / 60, output_dir)
    return resumo
