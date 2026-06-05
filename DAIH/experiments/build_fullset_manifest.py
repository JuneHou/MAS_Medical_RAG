#!/usr/bin/env python3
"""
Build a full-test-set manifest parquet for a (task, dataset) cell.

The oss Cond-A'/D' runners (run_condition_{A,D}_oss.py) consume a parquet whose
rows carry exactly the fields process_sample_condition_a/d read:
    sample_id, label, patient_context, positive_similars, negative_similars

The existing n=100 manifest (KARE/gpt/manifests/selected_samples_full.parquet) has
those columns for the mortality stratified subset. This script produces the same
schema for the FULL test set of any cell, by reusing KAREDataAdapter (which already
formats the temporal patient context and the positive/negative similar-patient
blocks) — so no context-formatting logic is duplicated here.

Usage:
    cd /data/wang/junh/githubs/Debate
    python DAIH/experiments/build_fullset_manifest.py --dataset mimic3 --task mortality
    python DAIH/experiments/build_fullset_manifest.py --dataset mimic4 --task readmission

Writes: KARE/gpt/manifests/fullset_{task}_{dataset}.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path("/data/wang/junh/githubs/Debate")
KARE_ROOT = REPO_ROOT / "KARE"

sys.path.insert(0, str(KARE_ROOT))
from kare_data_adapter import KAREDataAdapter  # noqa: E402

# task_config is co-located with this script
sys.path.insert(0, str(Path(__file__).parent))
import task_config  # noqa: E402


def build(dataset: str, task: str, base_path: Path, out_path: Path) -> int:
    adapter = KAREDataAdapter(base_path=str(base_path), split="test",
                              dataset=dataset, task=task)
    n = len(adapter.test_data)

    rows = []
    missing_similar = 0
    for i in tqdm(range(n), desc=f"{dataset}/{task}"):
        s = adapter.get_test_sample(i)
        pos = s["positive_similars"]
        neg = s["negative_similars"]
        if pos.startswith("No positive") and neg.startswith("No negative"):
            missing_similar += 1
        rows.append({
            "sample_id": s["patient_id"],          # KARE temporal id, e.g. "10117_0"
            "label": int(s["ground_truth"]),
            "patient_context": s["target_context"],
            "positive_similars": pos,
            "negative_similars": neg,
        })

    df = pd.DataFrame(rows, columns=[
        "sample_id", "label", "patient_context", "positive_similars", "negative_similars",
    ])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    pos_n = int((df["label"] == 1).sum())
    print(f"\n✓ Wrote {len(df)} rows -> {out_path}")
    print(f"  label distribution: positive(1)={pos_n}  negative(0)={len(df) - pos_n}  "
          f"({pos_n / len(df) * 100:.1f}% positive)")
    if missing_similar:
        print(f"  note: {missing_similar} patients had NO similar-patient context "
              f"(analysts will see 'No ... similar patients available').")
    return len(df)


def main():
    p = argparse.ArgumentParser(description="Build full-test-set manifest for a (task, dataset) cell")
    p.add_argument("--dataset", required=True, choices=task_config.VALID_DATASETS)
    p.add_argument("--task", required=True, choices=task_config.VALID_TASKS)
    p.add_argument("--base_path", default=str(KARE_ROOT / "data"),
                   help="KARE data root (default: KARE/data)")
    p.add_argument("--out", default=None,
                   help="Output parquet (default: KARE/gpt/manifests/fullset_{task}_{dataset}.parquet)")
    args = p.parse_args()

    out_path = Path(args.out) if args.out else (
        KARE_ROOT / "gpt" / "manifests" / f"fullset_{args.task}_{args.dataset}.parquet"
    )
    build(args.dataset, args.task, Path(args.base_path), out_path)


if __name__ == "__main__":
    main()
