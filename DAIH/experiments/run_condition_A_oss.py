#!/usr/bin/env python3
"""
Condition A' (oss-120b) — all-oss factorial cell for the DAIH transferability test.

Mirrors KARE/gpt/src/run_condition_A.py byte-for-byte EXCEPT the LLM client:
swaps GPTClient -> ARCClient (gpt-oss-120b via VT ARC). Same prompts, same
retrieval, same parsing, same fallback policy — so any difference vs the
existing GPT-4o Condition A results is attributable to the model substitution.

Per-patient JSON outputs land under DAIH/results/condition_A_oss/logs/ in the
same shape as KARE/gpt/results_bias/condition_A_gpt_4o/logs/ so Condition D'
can reuse them via the existing cache-loading pattern.

Usage:
    set -a; source /data/wang/junh/.cache/keys/arc_llm_api.sh; set +a
    export ARC_LLM_API_KEY="$API_KEY"
    cd /data/wang/junh/githubs/Debate

    # Smoke test (first 5 patients only)
    python DAIH/experiments/run_condition_A_oss.py --limit 5

    # Full n=100 stratified manifest
    python DAIH/experiments/run_condition_A_oss.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from tqdm import tqdm

# Repo paths
REPO_ROOT = Path("/data/wang/junh/githubs/Debate")
KARE_ROOT = REPO_ROOT / "KARE"
GPT_SRC = KARE_ROOT / "gpt" / "src"

# Make KARE-side modules importable (kare_data_adapter, gpt_utils, run_condition_A)
sys.path.insert(0, str(GPT_SRC))
sys.path.insert(0, str(KARE_ROOT))
sys.path.insert(0, str(Path(__file__).parent))  # arc_client

# Defer heavy imports until after path setup
import gpt_utils  # noqa: E402 — for AGENT_PROMPTS monkey-patch
from arc_client import ARCClient, DEFAULT_MODEL  # noqa: E402
from gpt_utils import initialize_medrag  # noqa: E402
from run_condition_A import process_sample_condition_a  # noqa: E402
import task_config  # noqa: E402 — per-task integrator prompts + naming


# The forced-<search> integrator prompts (per task) live in task_config.py.
# gpt-oss-120b emits the <search> tag only ~5% of the time vs GPT-4o ~86%
# (n=100 measurement), so the integrator prompt forces exactly one retrieval
# call to keep the same RAG pipeline the GPT-4o factorial exercised. For
# task="mortality" the prompt is byte-identical to the original.


def main():
    p = argparse.ArgumentParser(description="Condition A' — all gpt-oss-120b via ARC")
    p.add_argument("--task", type=str, default="mortality", choices=task_config.VALID_TASKS)
    p.add_argument("--dataset", type=str, default="mimic3", choices=task_config.VALID_DATASETS)
    p.add_argument("--full_data", type=str, default=None,
                   help="Manifest parquet (default: KARE/gpt/manifests/fullset_{task}_{dataset}.parquet)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output dir (default: DAIH/results/condition_A_oss_{task}_{dataset})")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL,
                   help="ARC model name (default: gpt-oss-120b)")
    p.add_argument("--k", type=int, default=8, help="MedRAG snippets per retrieval")
    p.add_argument("--corpus_name", type=str, default="MedCorp2")
    p.add_argument("--retriever_name", type=str, default="MedCPT")
    p.add_argument("--retriever_device", type=str, default="cuda:0")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N samples (smoke test). Skips already-done.")
    p.add_argument("--rpm", type=int, default=30)
    p.add_argument("--rph", type=int, default=1000)
    p.add_argument("--rp3h", type=int, default=3000)

    args = p.parse_args()

    # Fill task/dataset-derived defaults.
    if args.full_data is None:
        args.full_data = str(KARE_ROOT / "gpt" / "manifests"
                             / f"fullset_{args.task}_{args.dataset}.parquet")
    if args.output_dir is None:
        args.output_dir = str(REPO_ROOT / "DAIH" / "results"
                              / f"condition_A_oss_{args.task}_{args.dataset}")

    out_dir = Path(args.output_dir)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"CONDITION A' — all-oss factorial cell (model={args.model})")
    print(f"Task/dataset: {args.task} / {args.dataset}")
    print(f"Output:  {out_dir}")
    print(f"Manifest: {args.full_data}")
    print("=" * 80, flush=True)

    # Point the shared AGENT_PROMPTS integrator entries at this task's prompts.
    # Mutating the dict (not replacing it) flips the prompt at runtime without
    # touching the GPT-4o baseline code; for mortality the text is unchanged.
    task_config.apply_task_to_gpt_utils(args.task, gpt_utils)
    print(f"[oss patch] Forced-search integrator prompt activated for task={args.task}.", flush=True)

    # Load samples (same as run_condition_A.py)
    print(f"\nLoading samples from {args.full_data}...", flush=True)
    samples_df = pd.read_parquet(args.full_data)
    print(f"Loaded {len(samples_df)} samples")

    # Skip samples already done (resume support)
    pending = []
    for _, row in samples_df.iterrows():
        sid = row["sample_id"]
        if (logs_dir / f"{sid}.json").exists():
            continue
        pending.append(row.to_dict())
    print(f"Pending: {len(pending)} (skipping {len(samples_df) - len(pending)} already done)")

    if args.limit is not None:
        pending = pending[: args.limit]
        print(f"--limit={args.limit}; will process {len(pending)} samples")

    if not pending:
        print("Nothing to do.")
        return

    # Init ARC client + MedRAG
    arc_client = ARCClient(
        model=args.model, rpm=args.rpm, rph=args.rph, rp3h=args.rp3h,
    )
    medrag = initialize_medrag(
        corpus_name=args.corpus_name,
        retriever_name=args.retriever_name,
        retriever_device=args.retriever_device,
    )
    if medrag is None:
        print("ERROR: MedRAG initialization failed. Aborting.", file=sys.stderr)
        sys.exit(1)

    # Per-sample loop — write as we go
    summary = []
    for sample in tqdm(pending, desc="cond_A_oss"):
        sid = sample["sample_id"]
        try:
            result = process_sample_condition_a(sample, arc_client, medrag, k=args.k)
        except Exception as e:
            print(f"[FATAL] {sid}: {type(e).__name__}: {e}", flush=True)
            result = {
                "sample_id": sid,
                "label": int(sample["label"]),
                "error": f"{type(e).__name__}: {e}",
                "prediction": 1 - int(sample["label"]),  # match existing fallback
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

    # Save run-level summary
    with open(out_dir / "summary.json", "w") as fh:
        json.dump({
            "model": args.model,
            "n_samples": len(summary),
            "n_arc_calls": arc_client.n_calls,
            "config": vars(args),
            "samples": summary,
        }, fh, indent=2, default=str)

    print(f"\n✓ Done. Wrote {len(summary)} per-sample logs to {logs_dir}/")
    print(f"  Total ARC calls: {arc_client.n_calls}")
    print(f"  Next: python DAIH/experiments/run_condition_D_oss.py")


if __name__ == "__main__":
    main()
