#!/usr/bin/env python3
"""
Condition E' — oss-120b analysts + Qwen retrieval + LOCAL Qwen integrator.

Mirrors KARE/gpt/src/run_condition_E.py byte-for-byte except the analyst
outputs come from Cond A' (oss-120b) instead of Cond A (GPT-4o). The Qwen
retrieval bundle is loaded from the pre-existing Qwen RAG debate logs. The
final integrator is local Qwen2.5-7B-Instruct via vLLM (same loader as D').

Usage:
    cd /data/wang/junh/githubs/Debate
    python DAIH/experiments/run_condition_E_oss.py
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

import run_condition_E as condE  # noqa: E402
from run_condition_E import QwenIntegrator, process_sample_condition_e  # noqa: E402


DEFAULT_FULL_DATA = KARE_ROOT / "gpt" / "manifests" / "selected_samples_full.parquet"
DEFAULT_COND_A_DIR = REPO_ROOT / "DAIH" / "results" / "condition_A_oss"
DEFAULT_QWEN_LOG = (
    "/data/wang/junh/githubs/Debate/KARE/results/"
    "rag_mor_Qwen_Qwen2.5_7B_Instruct_int__data_wang_junh_githubs_Debate_KARE_"
    "searchr1_checkpoints_searchr1_binary_single_agent_step100_MedCPT_8_8/debate_logs"
)
DEFAULT_OUTPUT = REPO_ROOT / "DAIH" / "results" / "condition_E_oss_qwen_int"
DEFAULT_QWEN = "Qwen/Qwen2.5-7B-Instruct"


def main():
    p = argparse.ArgumentParser(description="Condition E' — oss analysts + Qwen retrieval + Qwen integrator")
    p.add_argument("--full_data", type=str, default=str(DEFAULT_FULL_DATA))
    p.add_argument("--condition_a_dir", type=str, default=str(DEFAULT_COND_A_DIR))
    p.add_argument("--qwen_log_dir", type=str, default=DEFAULT_QWEN_LOG)
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT))
    p.add_argument("--qwen_model", type=str, default=DEFAULT_QWEN)
    p.add_argument("--gpu_id", type=str, default="0")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    cond_a_dir = Path(args.condition_a_dir)
    cond_a_logs = cond_a_dir / "logs"
    if not cond_a_logs.is_dir():
        print(f"ERROR: Cond A' logs not found at {cond_a_logs}", file=sys.stderr)
        sys.exit(1)

    qwen_log_dir = Path(args.qwen_log_dir)
    if not qwen_log_dir.is_dir():
        print(f"ERROR: Qwen RAG log dir not found at {qwen_log_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"CONDITION E' — oss analysts (cached) + Qwen retrieval + Qwen integrator ({args.qwen_model})")
    print(f"Cond A' cache: {cond_a_dir}")
    print(f"Qwen RAG log:  {qwen_log_dir}")
    print(f"Output:        {out_dir}")
    print("=" * 80, flush=True)

    samples_df = pd.read_parquet(args.full_data)
    pending = []
    for _, row in samples_df.iterrows():
        sid = row["sample_id"]
        if (logs_dir / f"{sid}.json").exists():
            continue
        if not (cond_a_logs / f"{sid}.json").exists():
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
    condE._qwen_integrator = QwenIntegrator(model_name=args.qwen_model, gpu_id=args.gpu_id)

    summary = []
    for sample in tqdm(pending, desc="cond_E_oss"):
        sid = sample["sample_id"]
        try:
            result = process_sample_condition_e(
                sample, cond_a_dir, str(qwen_log_dir), args.qwen_model, args.gpu_id
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
            "condition_a_dir": str(cond_a_dir),
            "qwen_log_dir": str(qwen_log_dir),
            "n_samples": len(summary),
            "config": vars(args),
            "samples": summary,
        }, fh, indent=2, default=str)

    print(f"\n✓ Done. Wrote {len(summary)} per-sample logs to {logs_dir}/")


if __name__ == "__main__":
    main()
