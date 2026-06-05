#!/usr/bin/env python3
"""
Condition D' — oss-120b analysts + oss-120b retrieval + LOCAL Qwen integrator.

Reuses cached analyst & retrieval outputs from Condition A' (written by
run_condition_A_oss.py). Only the Qwen integrator step is new — and it runs
locally on a GPU, no API needed.

This tests whether the "integrator-swap is the lever" finding (which we
established with GPT-4o Cond A vs Cond D) ALSO holds when the analysts are
served by an open-weight model (gpt-oss-120b) instead of GPT-4o. If it does,
the role-attribution conclusion is model-family-invariant — that's the
transferability claim for the paper.

Usage:
    cd /data/wang/junh/githubs/Debate

    # Smoke test (whatever 5 patients Cond A' already finished)
    python DAIH/experiments/run_condition_D_oss.py --limit 5

    # Full
    python DAIH/experiments/run_condition_D_oss.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

# vLLM 0.11 (v1 engine) launches its engine core as a subprocess; the default
# start method is "fork", which crashes with "Cannot re-initialize CUDA in forked
# subprocess" once a CUDA context already exists in this parent process (vllm/torch
# touch CUDA during import/platform detection). Force "spawn" BEFORE importing vllm
# (pulled in by `import run_condition_D` below). See KARE/MEDRAG_GPU_SETUP_FIX.md.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path("/data/wang/junh/githubs/Debate")
KARE_ROOT = REPO_ROOT / "KARE"
GPT_SRC = KARE_ROOT / "gpt" / "src"

sys.path.insert(0, str(GPT_SRC))
sys.path.insert(0, str(KARE_ROOT))
sys.path.insert(0, str(Path(__file__).parent))  # task_config

# run_condition_D defines process_sample_condition_d() + module-level _qwen_integrator
# initializer pattern. We reuse both verbatim.
import gpt_utils  # noqa: E402 — for the integrator no-search prompt
import run_condition_D as condD  # noqa: E402
from run_condition_D import QwenIntegrator, process_sample_condition_d  # noqa: E402
import task_config  # noqa: E402 — per-task integrator prompts + naming


DEFAULT_QWEN = "Qwen/Qwen2.5-7B-Instruct"


def main():
    p = argparse.ArgumentParser(description="Condition D' — oss analysts/retrieval + Qwen integrator")
    p.add_argument("--task", type=str, default="mortality", choices=task_config.VALID_TASKS)
    p.add_argument("--dataset", type=str, default="mimic3", choices=task_config.VALID_DATASETS)
    p.add_argument("--full_data", type=str, default=None,
                   help="Manifest parquet (default: KARE/gpt/manifests/fullset_{task}_{dataset}.parquet)")
    p.add_argument("--condition_a_dir", type=str, default=None,
                   help="Cond A' logs dir (default: DAIH/results/condition_A_oss_{task}_{dataset})")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output dir (default: DAIH/results/condition_D_oss_qwen_int_{task}_{dataset})")
    p.add_argument("--qwen_model", type=str, default=DEFAULT_QWEN)
    p.add_argument("--gpu_id", type=str, default="0",
                   help="CUDA device for Qwen (single-GPU)")
    p.add_argument("--limit", type=int, default=None)

    args = p.parse_args()

    # Fill task/dataset-derived defaults.
    if args.full_data is None:
        args.full_data = str(KARE_ROOT / "gpt" / "manifests"
                             / f"fullset_{args.task}_{args.dataset}.parquet")
    if args.condition_a_dir is None:
        args.condition_a_dir = str(REPO_ROOT / "DAIH" / "results"
                                  / f"condition_A_oss_{args.task}_{args.dataset}")
    if args.output_dir is None:
        args.output_dir = str(REPO_ROOT / "DAIH" / "results"
                              / f"condition_D_oss_qwen_int_{args.task}_{args.dataset}")

    # Point the integrator (no-search) prompt at this task. run_qwen_integrator
    # reads AGENT_PROMPTS["balanced_clinical_integrator_no_search"] at call time.
    task_config.apply_task_to_gpt_utils(args.task, gpt_utils)
    print(f"[oss patch] Integrator prompt activated for task={args.task}.", flush=True)

    cond_a_dir = Path(args.condition_a_dir)
    cond_a_logs = cond_a_dir / "logs"
    if not cond_a_logs.is_dir():
        print(f"ERROR: Cond A' logs not found at {cond_a_logs}", file=sys.stderr)
        print("  Run DAIH/experiments/run_condition_A_oss.py first.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"CONDITION D' — oss analysts/retrieval (cached) + Qwen integrator ({args.qwen_model})")
    print(f"Task/dataset:  {args.task} / {args.dataset}")
    print(f"Cond A' cache: {cond_a_dir}")
    print(f"Output:        {out_dir}")
    print("=" * 80, flush=True)

    samples_df = pd.read_parquet(args.full_data)
    pending = []
    for _, row in samples_df.iterrows():
        sid = row["sample_id"]
        if (logs_dir / f"{sid}.json").exists():
            continue
        if not (cond_a_logs / f"{sid}.json").exists():
            # Cond A' hasn't finished this sample yet — skip; resumable later.
            continue
        pending.append(row.to_dict())
    print(f"Pending (Cond A' cache hit, not yet integrated): {len(pending)}")

    if args.limit is not None:
        pending = pending[: args.limit]
        print(f"--limit={args.limit}; will process {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    # Initialize the Qwen integrator into the run_condition_D module's global slot.
    # This matches the pattern used by run_condition_D.main().
    print(f"Initializing local Qwen integrator: {args.qwen_model} on cuda:{args.gpu_id}")
    condD._qwen_integrator = QwenIntegrator(model_name=args.qwen_model, gpu_id=args.gpu_id)

    summary = []
    for sample in tqdm(pending, desc="cond_D_oss"):
        sid = sample["sample_id"]
        try:
            result = process_sample_condition_d(sample, cond_a_dir)
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
            "error": result.get("error"),
        })

    with open(out_dir / "summary.json", "w") as fh:
        json.dump({
            "qwen_model": args.qwen_model,
            "condition_a_dir": str(cond_a_dir),
            "n_samples": len(summary),
            "config": vars(args),
            "samples": summary,
        }, fh, indent=2, default=str)

    print(f"\n✓ Done. Wrote {len(summary)} integrator outputs to {logs_dir}/")
    print(f"  Compare to: KARE/gpt/results_bias/condition_D_qwen/logs/")


if __name__ == "__main__":
    main()
