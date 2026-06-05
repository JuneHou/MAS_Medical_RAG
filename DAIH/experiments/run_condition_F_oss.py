#!/usr/bin/env python3
"""
Condition F' — Qwen analysts + oss-120b retrieval + LOCAL Qwen integrator.

Mirrors KARE/gpt/src/run_condition_F.py byte-for-byte except the retrieval
bundle comes from Cond C' (oss-120b retrieval, copied from Cond A') instead of
Cond C (GPT-4o retrieval). The analyst outputs come from the pre-existing Qwen
RAG debate logs. The final integrator is local Qwen2.5-7B-Instruct via vLLM.

Requires Cond C' to have run first (we read its `gpt_query`/`gpt_docs` fields,
which it copies from Cond A').

Usage:
    cd /data/wang/junh/githubs/Debate
    python DAIH/experiments/run_condition_F_oss.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path("/data/wang/junh/githubs/Debate")
KARE_ROOT = REPO_ROOT / "KARE"
GPT_SRC = KARE_ROOT / "gpt" / "src"

sys.path.insert(0, str(GPT_SRC))
sys.path.insert(0, str(KARE_ROOT))

import run_condition_F as condF  # noqa: E402
from run_condition_F import QwenIntegrator, process_sample_condition_f  # noqa: E402


DEFAULT_FULL_DATA = KARE_ROOT / "gpt" / "manifests" / "selected_samples_full.parquet"
DEFAULT_COND_C_DIR = REPO_ROOT / "DAIH" / "results" / "condition_C_oss"
DEFAULT_QWEN_LOG = (
    "/data/wang/junh/githubs/Debate/KARE/results/"
    "rag_mor_Qwen_Qwen2.5_7B_Instruct_int__data_wang_junh_githubs_Debate_KARE_"
    "searchr1_checkpoints_searchr1_binary_single_agent_step100_MedCPT_8_8/debate_logs"
)
DEFAULT_OUTPUT = REPO_ROOT / "DAIH" / "results" / "condition_F_oss_qwen_int"
DEFAULT_QWEN = "Qwen/Qwen2.5-7B-Instruct"


def main():
    p = argparse.ArgumentParser(description="Condition F' — Qwen analysts + oss retrieval + Qwen integrator")
    p.add_argument("--full_data", type=str, default=str(DEFAULT_FULL_DATA))
    p.add_argument("--condition_c_dir", type=str, default=str(DEFAULT_COND_C_DIR),
                   help="Directory containing Cond C' logs (retrieval bundle reused from here)")
    p.add_argument("--qwen_log_dir", type=str, default=DEFAULT_QWEN_LOG)
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT))
    p.add_argument("--qwen_model", type=str, default=DEFAULT_QWEN)
    p.add_argument("--gpu_id", type=str, default="0")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    cond_c_dir = Path(args.condition_c_dir)
    cond_c_logs = cond_c_dir / "logs"
    if not cond_c_logs.is_dir():
        print(f"ERROR: Cond C' logs not found at {cond_c_logs}", file=sys.stderr)
        print("  Run DAIH/experiments/run_condition_C_oss.py first.", file=sys.stderr)
        sys.exit(1)

    qwen_log_dir = Path(args.qwen_log_dir)
    if not qwen_log_dir.is_dir():
        print(f"ERROR: Qwen RAG log dir not found at {qwen_log_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"CONDITION F' — Qwen analysts + oss retrieval (cached) + Qwen integrator ({args.qwen_model})")
    print(f"Cond C' cache: {cond_c_dir}")
    print(f"Qwen RAG log:  {qwen_log_dir}")
    print(f"Output:        {out_dir}")
    print("=" * 80, flush=True)

    samples_df = pd.read_parquet(args.full_data)
    pending = []
    for _, row in samples_df.iterrows():
        sid = row["sample_id"]
        if (logs_dir / f"{sid}.json").exists():
            continue
        if not (cond_c_logs / f"{sid}.json").exists():
            continue
        pending.append(row.to_dict())
    print(f"Pending: {len(pending)}")

    if args.limit is not None:
        pending = pending[: args.limit]
        print(f"--limit={args.limit}; will process {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    print(f"Initializing local Qwen integrator: {args.qwen_model} on cuda:{args.gpu_id}")
    condF._qwen_integrator = QwenIntegrator(model_name=args.qwen_model, gpu_id=args.gpu_id)

    summary = []
    for sample in tqdm(pending, desc="cond_F_oss"):
        sid = sample["sample_id"]
        try:
            result = process_sample_condition_f(
                sample, cond_c_dir, str(qwen_log_dir), args.qwen_model, args.gpu_id
            )
        except Exception as e:
            print(f"[FATAL] {sid}: {type(e).__name__}: {e}", flush=True)
            result = {
                "sample_id": sid,
                "label": int(sample["label"]),
                "error": f"{type(e).__name__}: {e}",
                "prediction": 1 - int(sample["label"]),
            }

        with open(logs_dir / f"{sid}.json", "w") as fh:
            json.dump(result, fh, indent=2, default=str)
        summary.append({
            "sample_id": sid,
            "label": result.get("label"),
            "prediction": result.get("prediction"),
            "mortality_probability": result.get("mortality_probability"),
            "called_retriever": result.get("called_retriever"),
            "error": result.get("error"),
        })

    with open(out_dir / "summary.json", "w") as fh:
        json.dump({
            "qwen_model": args.qwen_model,
            "condition_c_dir": str(cond_c_dir),
            "qwen_log_dir": str(qwen_log_dir),
            "n_samples": len(summary),
            "config": vars(args),
            "samples": summary,
        }, fh, indent=2, default=str)

    print(f"\n✓ Done. Wrote {len(summary)} per-sample logs to {logs_dir}/")


if __name__ == "__main__":
    main()
