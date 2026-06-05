#!/usr/bin/env python3
"""
Condition G — all-Qwen, biased prompt.

Pipeline:
  - Analyst 1 (mortality risk) : Qwen2.5-7B   (reused from existing Qwen debate logs)
  - Analyst 2 (protective)     : Qwen2.5-7B   (reused from existing Qwen debate logs)
  - Retrieval                  : Qwen2.5-7B   (reused from existing Qwen debate logs)
  - Integrator                 : Qwen2.5-7B   (FRESH run with the biased prompt
                                              from gpt_utils_bias.AGENT_PROMPTS)

The pre-existing Qwen debate logs at rag_mor_Qwen_*/debate_logs/ used an UNBIASED
integrator prompt, so the integrator step is the only piece we re-run here to
produce a biased-regime number directly comparable to GPT-4o A-F and oss A'-F'.

We do NOT re-run the Qwen analysts or Qwen retrieval — those are bias-free by
construction (see DAIH/experiments/audit notes; mortality_debate_rag_binary.py
lines 162-200) so their cached outputs are valid under either prompt regime.

Usage:
    cd /data/wang/junh/githubs/Debate
    CUDA_VISIBLE_DEVICES=0 python DAIH/experiments/run_condition_G.py
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

# Cond F gives us: QwenIntegrator class, the biased-prompt integrator function,
# and the Qwen analyst-log loader. Cond E gives us the Qwen retrieval-log loader.
import run_condition_F as condF  # noqa: E402
from run_condition_F import (  # noqa: E402
    QwenIntegrator,
    run_qwen_integrator_final,
    load_qwen_analyst_outputs,
)
from run_condition_E import load_qwen_retrieval_bundle  # noqa: E402
from gpt_utils_bias import extract_probabilities  # noqa: E402


DEFAULT_FULL_DATA = KARE_ROOT / "gpt" / "manifests" / "selected_samples_full.parquet"
DEFAULT_QWEN_LOG = (
    "/data/wang/junh/githubs/Debate/KARE/results/"
    "rag_mor_Qwen_Qwen2.5_7B_Instruct_int__data_wang_junh_githubs_Debate_KARE_"
    "searchr1_checkpoints_searchr1_binary_single_agent_step100_MedCPT_8_8/debate_logs"
)
DEFAULT_OUTPUT = REPO_ROOT / "DAIH" / "results" / "condition_G_all_qwen"
DEFAULT_QWEN = "Qwen/Qwen2.5-7B-Instruct"


def process_sample_condition_g(sample, qwen_log_dir, qwen_model, gpu_id):
    sample_id = sample["sample_id"]
    target_context = sample["patient_context"]

    result = {
        "sample_id": sample_id,
        "label": int(sample["label"]),
        "qwen_analyst1": None,
        "qwen_analyst2": None,
        "qwen_query": None,
        "qwen_docs": None,
        "called_retriever": False,
        "qwen_integrator_final": None,
        "mortality_probability": None,
        "survival_probability": None,
        "prediction": None,
        "error": None,
    }

    try:
        analysts = load_qwen_analyst_outputs(sample_id, qwen_log_dir)
        if not analysts or not analysts["analyst1"] or not analysts["analyst2"]:
            raise ValueError(f"Qwen analyst outputs not found for {sample_id}")
        result["qwen_analyst1"] = analysts["analyst1"]
        result["qwen_analyst2"] = analysts["analyst2"]

        retrieval = load_qwen_retrieval_bundle(sample_id, qwen_log_dir)
        if retrieval and retrieval.get("called_retriever"):
            result["qwen_query"] = retrieval["query"]
            result["qwen_docs"] = retrieval["docs_text"]
            result["called_retriever"] = True
            docs_str = retrieval["docs_text"]
        else:
            docs_str = None

        integrator_final = run_qwen_integrator_final(
            target_context,
            result["qwen_analyst1"],
            result["qwen_analyst2"],
            docs_str,
            qwen_model,
            gpu_id,
        )
        result["qwen_integrator_final"] = integrator_final

        probs = extract_probabilities(integrator_final)
        result["mortality_probability"] = probs["mortality_probability"]
        result["survival_probability"] = probs["survival_probability"]
        result["prediction"] = probs["prediction"]

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    if result["prediction"] is None:
        # parse failure / exception fallback (matches Cond F convention)
        result["prediction"] = 1 - int(sample["label"])
        result["mortality_probability"] = 1.0 if result["prediction"] == 1 else 0.0
        result["survival_probability"] = 1.0 - result["mortality_probability"]

    return result


def main():
    p = argparse.ArgumentParser(description="Condition G — all-Qwen, biased prompt")
    p.add_argument("--full_data", type=str, default=str(DEFAULT_FULL_DATA))
    p.add_argument("--qwen_log_dir", type=str, default=DEFAULT_QWEN_LOG)
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT))
    p.add_argument("--qwen_model", type=str, default=DEFAULT_QWEN)
    p.add_argument("--gpu_id", type=str, default="0")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    qwen_log_dir = Path(args.qwen_log_dir)
    if not qwen_log_dir.is_dir():
        print(f"ERROR: Qwen log dir not found at {qwen_log_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"CONDITION G — all-Qwen + biased prompt ({args.qwen_model})")
    print(f"Qwen log dir: {qwen_log_dir}")
    print(f"Output:       {out_dir}")
    print("=" * 80, flush=True)

    samples_df = pd.read_parquet(args.full_data)
    pending = []
    for _, row in samples_df.iterrows():
        sid = row["sample_id"]
        if (logs_dir / f"{sid}.json").exists():
            continue
        # Require the Qwen debate log to exist (otherwise no cached analysts/retrieval)
        if not (qwen_log_dir / f"debate_responses_{sid}.log").exists():
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
    for sample in tqdm(pending, desc="cond_G"):
        sid = sample["sample_id"]
        try:
            result = process_sample_condition_g(
                sample, str(qwen_log_dir), args.qwen_model, args.gpu_id
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
            "qwen_log_dir": str(qwen_log_dir),
            "n_samples": len(summary),
            "config": vars(args),
            "samples": summary,
        }, fh, indent=2, default=str)

    print(f"\nDone. Wrote {len(summary)} per-sample logs to {logs_dir}/")


if __name__ == "__main__":
    main()
