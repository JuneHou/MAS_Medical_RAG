#!/usr/bin/env python3
"""
Condition H' — single-agent oss RAG (the DEBATE-ABLATION control).

H' keeps everything Cond A' has EXCEPT the two contrastive analysts: one
gpt-oss-120b call reads the Target patient + the SAME retrieved evidence A'
pulled, and emits the prediction directly. Same model, same retrieval, same
bias framing, same probability output + 0.5 threshold as A'. So a difference
between H' and A' is attributable to the *debate structure*, nothing else.

Why this is the control we need: the only single-agent RAG numbers we have
(Excel `KARE Ablation`) are Qwen-based AND ~87% format-fallback (868-877/996)
— a broken baseline. There is no model-matched, docs-matched single-agent RAG
result anywhere, so H' has to be run.

Cost/footprint: rides the A' cache — NO new FAISS index, NO retriever, NO GPU.
~1 oss API call per patient. Reuses A''s `gpt_docs` (the docs A''s integrator
retrieved via its forced <search>), so retrieval is held byte-identical to A'.

Usage:
    set -a; source /data/wang/junh/.cache/keys/arc_llm_api.sh; set +a
    export ARC_LLM_API_KEY="$API_KEY"
    cd /data/wang/junh/githubs/Debate

    # Smoke test (first 10 patients of one cell)
    python DAIH/experiments/run_condition_H_oss.py --task mortality --dataset mimic3 --limit 10

    # Full cell
    python DAIH/experiments/run_condition_H_oss.py --task mortality --dataset mimic3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path("/data/wang/junh/githubs/Debate")
KARE_ROOT = REPO_ROOT / "KARE"
GPT_SRC = KARE_ROOT / "gpt" / "src"

sys.path.insert(0, str(GPT_SRC))
sys.path.insert(0, str(KARE_ROOT))
sys.path.insert(0, str(Path(__file__).parent))  # arc_client, task_config

from arc_client import ARCClient, DEFAULT_MODEL  # noqa: E402
from gpt_utils import format_retrieved_docs, extract_probabilities  # noqa: E402
import task_config  # noqa: E402


def build_single_agent_prompt(
    system_prompt: str,
    target_context: str,
    retrieved_docs: Optional[str],
    pos_line: str,
    neg_line: str,
) -> str:
    """Single-agent prompt: patient + (optional) reused A' evidence, no analysts."""
    reminder = (
        f"{pos_line}: X.XX (0.00 to 1.00)\n"
        f"{neg_line}: X.XX (0.00 to 1.00)\n\n"
        "Note: The two probabilities MUST sum to exactly 1.00"
    )
    if retrieved_docs:
        return f"""{system_prompt}

## Target Patient:
{target_context}

<information>
{retrieved_docs}
</information>

Now provide your final assessment with:

{reminder}"""
    return f"""{system_prompt}

## Target Patient:
{target_context}

Provide your final assessment with:

{reminder}"""


def process_sample_condition_h(
    sample: Dict[str, Any],
    arc_client: ARCClient,
    cond_a_dir: Path,
    system_prompt: str,
    pos_line: str,
    neg_line: str,
) -> Dict[str, Any]:
    sample_id = sample["sample_id"]
    target_context = sample["patient_context"]

    result = {
        "sample_id": sample_id,
        "label": int(sample["label"]),
        "reused_docs_from": "condition_A_oss",
        "gpt_docs": None,
        "called_retriever": False,
        "single_agent_output": None,
        "mortality_probability": None,
        "survival_probability": None,
        "prediction": None,
        "error": None,
    }

    try:
        # Reuse A''s retrieved evidence (the integrator's forced-search docs).
        cond_a_file = cond_a_dir / "logs" / f"{sample_id}.json"
        if not cond_a_file.exists():
            raise FileNotFoundError(f"Cond A' log not found: {cond_a_file}")
        with open(cond_a_file) as f:
            cond_a = json.load(f)

        docs = cond_a.get("gpt_docs")
        docs_text = None
        if docs:
            docs_text = format_retrieved_docs(docs)
            result["gpt_docs"] = docs
            result["called_retriever"] = bool(cond_a.get("called_retriever", True))

        prompt = build_single_agent_prompt(
            system_prompt, target_context, docs_text, pos_line, neg_line
        )
        response = arc_client.generate(prompt, max_tokens=32768, temperature=0.7)
        result["single_agent_output"] = response

        probs = extract_probabilities(response)
        result["mortality_probability"] = probs["mortality_probability"]
        result["survival_probability"] = probs["survival_probability"]
        result["prediction"] = probs["prediction"]

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    # Same fallback policy as A'/B'/D': parse failure -> opposite of label.
    if result["prediction"] is None:
        result["prediction"] = 1 - int(sample["label"])
        result["mortality_probability"] = 1.0 if result["prediction"] == 1 else 0.0
        result["survival_probability"] = 1.0 - result["mortality_probability"]

    return result


def main():
    p = argparse.ArgumentParser(
        description="Condition H' — single-agent oss RAG (debate-ablation control)"
    )
    p.add_argument("--task", type=str, default="mortality", choices=task_config.VALID_TASKS)
    p.add_argument("--dataset", type=str, default="mimic3", choices=task_config.VALID_DATASETS)
    p.add_argument("--full_data", type=str, default=None,
                   help="Manifest parquet (default: KARE/gpt/manifests/fullset_{task}_{dataset}.parquet)")
    p.add_argument("--condition_a_dir", type=str, default=None,
                   help="Cond A' dir to reuse retrieved docs from "
                        "(default: DAIH/results/condition_A_oss_{task}_{dataset})")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output dir (default: DAIH/results/condition_H_oss_{task}_{dataset})")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--rpm", type=int, default=30)
    p.add_argument("--rph", type=int, default=1000)
    p.add_argument("--rp3h", type=int, default=3000)
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N samples (smoke test). Skips already-done.")
    args = p.parse_args()

    if args.full_data is None:
        args.full_data = str(KARE_ROOT / "gpt" / "manifests"
                             / f"fullset_{args.task}_{args.dataset}.parquet")
    if args.condition_a_dir is None:
        args.condition_a_dir = str(REPO_ROOT / "DAIH" / "results"
                                   / f"condition_A_oss_{args.task}_{args.dataset}")
    if args.output_dir is None:
        args.output_dir = str(REPO_ROOT / "DAIH" / "results"
                              / f"condition_H_oss_{args.task}_{args.dataset}")

    cfg = task_config.get_task(args.task)
    system_prompt = cfg["single_agent"]
    pos_line, neg_line = task_config.PROB_LINES[args.task]

    cond_a_dir = Path(args.condition_a_dir)
    cond_a_logs = cond_a_dir / "logs"
    if not cond_a_logs.is_dir():
        print(f"ERROR: Cond A' logs not found at {cond_a_logs}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CONDITION H' — single-agent oss RAG (debate-ablation control)")
    print(f"Model:         {args.model}")
    print(f"Task/dataset:  {args.task} / {args.dataset}")
    print(f"Cond A' cache: {cond_a_dir}  (reusing retrieved docs)")
    print(f"Output:        {out_dir}")
    print(f"Manifest:      {args.full_data}")
    print("=" * 80, flush=True)

    samples_df = pd.read_parquet(args.full_data)
    pending = []
    skipped_no_a = 0
    for _, row in samples_df.iterrows():
        sid = row["sample_id"]
        if (logs_dir / f"{sid}.json").exists():
            continue
        if not (cond_a_logs / f"{sid}.json").exists():
            skipped_no_a += 1  # A' hasn't produced this sample yet
            continue
        pending.append(row.to_dict())
    print(f"Pending: {len(pending)}  (skipped {skipped_no_a} with no Cond A' log)")

    if args.limit is not None:
        pending = pending[: args.limit]
        print(f"--limit={args.limit}; will process {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    arc_client = ARCClient(model=args.model, rpm=args.rpm, rph=args.rph, rp3h=args.rp3h)

    summary = []
    for sample in tqdm(pending, desc=f"cond_H_oss_{args.task}_{args.dataset}"):
        sid = sample["sample_id"]
        result = process_sample_condition_h(
            sample, arc_client, cond_a_dir, system_prompt, pos_line, neg_line
        )
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
            "condition": "H_prime",
            "model": args.model,
            "task": args.task,
            "dataset": args.dataset,
            "condition_a_dir": str(cond_a_dir),
            "n_samples": len(summary),
            "n_arc_calls": arc_client.n_calls,
            "config": vars(args),
            "samples": summary,
        }, fh, indent=2, default=str)

    n_err = sum(1 for s in summary if s["error"])
    print(f"\n✓ Done. Wrote {len(summary)} per-sample logs to {logs_dir}/")
    print(f"  Total ARC calls: {arc_client.n_calls}   errors: {n_err}")


if __name__ == "__main__":
    main()
