#!/usr/bin/env python3
"""
Condition B' — oss-120b analysts + Qwen retrieval + oss-120b integrator.

Mirrors KARE/gpt/src/run_condition_B.py byte-for-byte EXCEPT the LLM client:
swaps GPTClient -> ARCClient. The analyst outputs are reused from Cond A'
(oss-120b) via the same cache-load pattern; the retrieval bundle is loaded
from the pre-existing Qwen RAG debate logs (the upstream Cond B/E source).
Cond B's `balanced_clinical_integrator_no_search` prompt already contains the
"Mortality is rare..." bias text, so no further prompt patching is needed
(retrieval is pre-supplied, not requested via <search>).

Usage:
    set -a; source /data/wang/junh/.cache/keys/arc_llm_api.sh; set +a
    export ARC_LLM_API_KEY="$API_KEY"
    cd /data/wang/junh/githubs/Debate
    python DAIH/experiments/run_condition_B_oss.py
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
sys.path.insert(0, str(Path(__file__).parent))  # arc_client

from arc_client import ARCClient, DEFAULT_MODEL  # noqa: E402
from run_condition_B import process_sample_condition_b  # noqa: E402


DEFAULT_FULL_DATA = KARE_ROOT / "gpt" / "manifests" / "selected_samples_full.parquet"
DEFAULT_COND_A_DIR = REPO_ROOT / "DAIH" / "results" / "condition_A_oss"
DEFAULT_QWEN_LOG = (
    "/data/wang/junh/githubs/Debate/KARE/results/"
    "rag_mor_Qwen_Qwen2.5_7B_Instruct_int__data_wang_junh_githubs_Debate_KARE_"
    "searchr1_checkpoints_searchr1_binary_single_agent_step100_MedCPT_8_8/debate_logs"
)
DEFAULT_OUTPUT = REPO_ROOT / "DAIH" / "results" / "condition_B_oss"


def main():
    p = argparse.ArgumentParser(description="Condition B' — oss analysts + Qwen retrieval + oss integrator")
    p.add_argument("--full_data", type=str, default=str(DEFAULT_FULL_DATA))
    p.add_argument("--condition_a_dir", type=str, default=str(DEFAULT_COND_A_DIR),
                   help="Directory containing Cond A' logs (analysts come from here)")
    p.add_argument("--qwen_log_dir", type=str, default=DEFAULT_QWEN_LOG)
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT))
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--rpm", type=int, default=30)
    p.add_argument("--rph", type=int, default=1000)
    p.add_argument("--rp3h", type=int, default=3000)
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
    print(f"CONDITION B' — oss analysts (cached) + Qwen retrieval + oss integrator")
    print(f"Model:         {args.model}")
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
            continue  # Cond A' hasn't produced this yet; skip
        pending.append(row.to_dict())
    print(f"Pending: {len(pending)}")

    if args.limit is not None:
        pending = pending[: args.limit]
        print(f"--limit={args.limit}; will process {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    arc_client = ARCClient(model=args.model, rpm=args.rpm, rph=args.rph, rp3h=args.rp3h)

    summary = []
    for sample in tqdm(pending, desc="cond_B_oss"):
        sid = sample["sample_id"]
        try:
            result = process_sample_condition_b(sample, arc_client, cond_a_dir, str(qwen_log_dir))
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
            "model": args.model,
            "condition_a_dir": str(cond_a_dir),
            "qwen_log_dir": str(qwen_log_dir),
            "n_samples": len(summary),
            "n_arc_calls": arc_client.n_calls,
            "config": vars(args),
            "samples": summary,
        }, fh, indent=2, default=str)

    print(f"\n✓ Done. Wrote {len(summary)} per-sample logs to {logs_dir}/")
    print(f"  Total ARC calls: {arc_client.n_calls}")


if __name__ == "__main__":
    main()
