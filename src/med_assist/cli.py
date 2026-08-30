"""Interface de linha de comando do assistente médico.

    med-assist data synth                     # gera o corpus bruto sintético
    med-assist data build                     # anonimiza, cura e monta o dataset
    med-assist ehr seed | list | show | pending
    med-assist ask "..." --paciente PRT-2026-100137 --usuario "CRM-SP 111.222"
    med-assist alerts PRT-2026-100137
    med-assist audit --trace <id>
    med-assist graph mermaid
    med-assist finetune train | merge | gguf | evaluate
    med-assist serve --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .safety.audit import get_logger, setup_logging

logger = get_logger("cli")


# --------------------------------------------------------------------------- #
# Comandos
# --------------------------------------------------------------------------- #


def cmd_data(args: argparse.Namespace) -> int:
    if args.acao == "synth":
        from .data.synth import build_raw_corpus
        resumo = build_raw_corpus(n_pacientes=args.pacientes)
        print(json.dumps(resumo, ensure_ascii=False, indent=2))
    else:
        from .data.build_dataset import build
        resumo = build(n_medquad=args.medquad, val_ratio=args.val_ratio)
        print(json.dumps({k: v for k, v in resumo.items() if k != "anonimizacao"},
                         ensure_ascii=False, indent=2))
        print("\nAnonimização:", json.dumps(resumo["anonimizacao"], ensure_ascii=False, indent=2))
    return 0


def cmd_ehr(args: argparse.Namespace) -> int:
    from .knowledge import ehr

    if args.acao == "seed":
        print(f"{ehr.seed()} pacientes carregados em {ehr.EHR_DB}")
    elif args.acao == "list":
        for p in ehr.list_patients(setor=args.setor):
            print(f"{p['paciente_id']} | {p['setor']:<28} leito {p['leito']:<6} | "
                  f"{p['protocolo']} | {p['condicao']}")
    elif args.acao == "show":
        contexto = ehr.clinical_context(args.paciente)
        if contexto is None:
            print(f"Paciente {args.paciente} não encontrado.", file=sys.stderr)
            return 1
        print(ehr.format_context(contexto))
    elif args.acao == "pending":
        print(json.dumps(ehr.pending_exams(args.paciente), ensure_ascii=False, indent=2))
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from .assistant.chain import MedicalAssistant

    assistente = MedicalAssistant()
    resposta = assistente.ask(args.pergunta, paciente_id=args.paciente, usuario=args.usuario)
    if args.json:
        print(json.dumps(resposta.as_dict(), ensure_ascii=False, indent=2))
    else:
        print("\n" + resposta.render())
    return 0


def cmd_alerts(args: argparse.Namespace) -> int:
    from .assistant import alerts as alert_engine
    from .knowledge import ehr

    contexto = ehr.clinical_context(args.paciente)
    if contexto is None:
        print(f"Paciente {args.paciente} não encontrado.", file=sys.stderr)
        return 1
    ativos = alert_engine.avaliar(contexto, ehr.pending_exams(args.paciente))
    if not ativos:
        print("Nenhum alerta ativo.")
        return 0
    for alerta in ativos:
        print(f"[{alerta.severidade.upper():<10}] {alerta.codigo}\n"
              f"  {alerta.mensagem}\n  fonte: {alerta.protocolo}\n")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    from .safety.audit import get_audit_trail

    registros = get_audit_trail().read(trace_id=args.trace, limite=args.limite)
    if args.json:
        print(json.dumps(registros, ensure_ascii=False, indent=2))
        return 0
    for r in registros:
        extras = {k: v for k, v in r.items() if k not in ("timestamp", "trace_id", "evento")}
        print(f"{r['timestamp']} | {r['trace_id'][:8]} | {r['evento']:<24} | "
              + ", ".join(f"{k}={v}" for k, v in extras.items())[:160])
    return 0


def cmd_graph(args: argparse.Namespace) -> int:
    from .assistant.chain import MedicalAssistant

    mermaid = MedicalAssistant().mermaid()
    if args.saida:
        Path(args.saida).write_text(mermaid, encoding="utf-8")
        print(f"Diagrama salvo em {args.saida}")
    else:
        print(mermaid)
    return 0


def cmd_finetune(args: argparse.Namespace) -> int:
    if args.acao == "train":
        from .finetune.train import TrainConfig, train
        config = TrainConfig(base_model=args.base_model, epochs=args.epochs, lr=args.lr,
                             batch_size=args.batch_size, grad_accum=args.grad_accum,
                             max_len=args.max_len, device=args.device, limit=args.limit)
        resumo = train(config)
        print(json.dumps({k: v for k, v in resumo.items() if k != "historico"},
                         ensure_ascii=False, indent=2))
    elif args.acao == "merge":
        from .finetune.merge_export import merge
        print(f"Modelo fundido em {merge(base_model=args.base_model)}")
    elif args.acao == "gguf":
        from .finetune.merge_export import to_gguf
        print(f"GGUF gerado em {to_gguf(llama_cpp=args.llama_cpp, quant=args.quant)}")
    elif args.acao == "evaluate":
        from .finetune.evaluate import evaluate
        resultado = evaluate(base_model=args.base_model, limit=args.limit,
                             comparar_base=not args.sem_base)
        resumo = {"casos": resultado["casos"],
                  **{modelo: dados["metricas"] for modelo, dados in resultado["modelos"].items()}}
        if "delta" in resultado:
            resumo["delta"] = resultado["delta"]
        print(json.dumps(resumo, ensure_ascii=False, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("med_assist.serving.api:app", host=args.host, port=args.port, reload=False)
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    from .finetune.train import DEFAULT_BASE_MODEL

    parser = argparse.ArgumentParser(
        prog="med-assist",
        description="Assistente médico virtual — Tech Challenge Fase 3 (FIAP Postech)")
    sub = parser.add_subparsers(dest="comando", required=True)

    p_data = sub.add_parser("data", help="pipeline de dados (síntese, anonimização, curadoria)")
    p_data.add_argument("acao", choices=["synth", "build"])
    p_data.add_argument("--pacientes", type=int, default=24)
    p_data.add_argument("--medquad", type=int, default=200, help="pares do MedQuAD a amostrar")
    p_data.add_argument("--val-ratio", type=float, default=0.15)
    p_data.set_defaults(func=cmd_data)

    p_ehr = sub.add_parser("ehr", help="prontuário eletrônico (base estruturada)")
    p_ehr.add_argument("acao", choices=["seed", "list", "show", "pending"])
    p_ehr.add_argument("paciente", nargs="?", help="id do prontuário, leito ou nome")
    p_ehr.add_argument("--setor")
    p_ehr.set_defaults(func=cmd_ehr)

    p_ask = sub.add_parser("ask", help="consulta o assistente clínico")
    p_ask.add_argument("pergunta")
    p_ask.add_argument("--paciente", help="id do prontuário para contextualizar")
    p_ask.add_argument("--usuario", default="profissional não identificado")
    p_ask.add_argument("--json", action="store_true")
    p_ask.set_defaults(func=cmd_ask)

    p_alerts = sub.add_parser("alerts", help="alertas clínicos ativos de um paciente")
    p_alerts.add_argument("paciente")
    p_alerts.set_defaults(func=cmd_alerts)

    p_audit = sub.add_parser("audit", help="lê a trilha de auditoria")
    p_audit.add_argument("--trace", help="filtra por trace_id")
    p_audit.add_argument("--limite", type=int, default=30)
    p_audit.add_argument("--json", action="store_true")
    p_audit.set_defaults(func=cmd_audit)

    p_graph = sub.add_parser("graph", help="exporta o diagrama do fluxo")
    p_graph.add_argument("acao", choices=["mermaid"])
    p_graph.add_argument("--saida")
    p_graph.set_defaults(func=cmd_graph)

    p_ft = sub.add_parser("finetune", help="pipeline de fine-tuning")
    p_ft.add_argument("acao", choices=["train", "merge", "gguf", "evaluate"])
    p_ft.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p_ft.add_argument("--epochs", type=int, default=2)
    p_ft.add_argument("--lr", type=float, default=2e-4)
    p_ft.add_argument("--batch-size", type=int, default=1)
    p_ft.add_argument("--grad-accum", type=int, default=8)
    p_ft.add_argument("--max-len", type=int, default=1024)
    p_ft.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p_ft.add_argument("--limit", type=int, default=None, help="limita exemplos (train) ou casos (evaluate)")
    p_ft.add_argument("--llama-cpp", help="caminho do repositório llama.cpp (para gguf)")
    p_ft.add_argument("--quant", default="Q4_K_M")
    p_ft.add_argument("--sem-base", action="store_true", help="avalia só o fine-tunado")
    p_ft.set_defaults(func=cmd_finetune)

    p_serve = sub.add_parser("serve", help="sobe a API REST")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)
    if args.comando == "finetune" and args.acao == "evaluate" and args.limit is None:
        args.limit = 12
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
