#!/usr/bin/env python3
"""
Compute aggregate metrics from per-sample logs in a DAIH/results/<condition>/logs dir.

Reports accuracy / precision / recall / F1 / specificity overall and broken down
by the manifest's positive / hard-negative / easy-negative subgroups, matching
the columns in the Excel `GPT-4o` and `GPT-4o-Unbias` sheets so we can put oss
results side-by-side with the existing GPT-4o numbers.

Usage:
    python DAIH/experiments/compute_metrics.py \
        --logs_dir DAIH/results/condition_A_oss/logs

    # compare multiple at once:
    python DAIH/experiments/compute_metrics.py \
        --logs_dir DAIH/results/condition_A_oss/logs \
                   DAIH/results/condition_D_oss_qwen_int/logs \
                   KARE/gpt/results_bias/condition_A_gpt_4o/logs \
                   KARE/gpt/results_bias/condition_D_qwen/logs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path("/data/wang/junh/githubs/Debate")
MANIFEST_CSV = REPO_ROOT / "KARE" / "gpt" / "manifests" / "samples_swap_core.csv"


def _safe_div(a, b):
    return a / b if b else 0.0


def _metrics(labels: List[int], preds: List[int]) -> Dict[str, float]:
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    n = len(labels)
    acc = (tp + tn) / n if n else 0.0
    prec = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * prec * recall, prec + recall) if (prec + recall) else 0.0
    spec = _safe_div(tn, tn + fp)
    return {
        "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": acc, "precision": prec, "recall": recall,
        "f1": f1, "specificity": spec,
    }


def _load_manifest_subgroups() -> Dict[str, str]:
    """Return sample_id -> subgroup label.

    Reads samples_swap_core.csv if present and maps to {pos_all, neg_hard,
    neg_easy} matching the Excel breakdown convention. If the CSV doesn't
    annotate subgroups, falls back to just {pos, neg} by label.
    """
    sub = {}
    if not MANIFEST_CSV.exists():
        return sub
    import csv
    with open(MANIFEST_CSV) as fh:
        for row in csv.DictReader(fh):
            sid = row.get("sample_id")
            if not sid:
                continue
            # Common column variants seen in the gpt/ manifest pipeline
            cat = (
                row.get("subgroup")
                or row.get("category")
                or row.get("difficulty")
                or row.get("stratum")
            )
            if cat:
                sub[sid] = cat
    return sub


def analyze(logs_dir: Path, subgroups: Dict[str, str]) -> Dict:
    files = sorted(logs_dir.glob("*.json"))
    if not files:
        return {"logs_dir": str(logs_dir), "error": "no log files"}

    labels: List[int] = []
    preds: List[int] = []
    fallbacks = 0
    no_retr = 0
    by_sub: Dict[str, List[tuple]] = {}

    for fp in files:
        with open(fp) as fh:
            d = json.load(fh)
        sid = d.get("sample_id")
        label = d.get("label")
        pred = d.get("prediction")
        if label is None or pred is None:
            continue
        label = int(label); pred = int(pred)
        labels.append(label); preds.append(pred)

        # Heuristic: fallback occurred when mortality == 0.0 or 1.0 exactly AND prediction = 1 - label
        m = d.get("mortality_probability")
        if m in (0.0, 1.0, None) and pred == (1 - label):
            fallbacks += 1
        if d.get("called_retriever") is False:
            no_retr += 1

        # subgroup grouping (fall back to pos/neg by label if no manifest)
        sub = subgroups.get(sid)
        if not sub:
            sub = "pos" if label == 1 else "neg"
        by_sub.setdefault(sub, []).append((label, pred))

    overall = _metrics(labels, preds)
    overall["fallbacks"] = fallbacks
    overall["fallback_rate"] = _safe_div(fallbacks, len(labels))
    overall["no_retrieval_count"] = no_retr
    overall["no_retrieval_rate"] = _safe_div(no_retr, len(labels))

    sub_metrics = {}
    for sub, pairs in sorted(by_sub.items()):
        ys, ps = zip(*pairs) if pairs else ([], [])
        sub_metrics[sub] = _metrics(list(ys), list(ps))

    return {
        "logs_dir": str(logs_dir),
        "overall": overall,
        "by_subgroup": sub_metrics,
    }


def _fmt_pct(x: float) -> str:
    return f"{x * 100:5.1f}"


def print_report(r: Dict):
    if "error" in r:
        print(f"{r['logs_dir']}: {r['error']}")
        return
    o = r["overall"]
    print(f"\n=== {r['logs_dir']} ===")
    print(f"  n={o['n']:4d}  fallbacks={o['fallbacks']:3d}/{o['n']:3d} "
          f"({o['fallback_rate']*100:.1f}%)  no_retrieval={o['no_retrieval_count']:3d}/{o['n']:3d} "
          f"({o['no_retrieval_rate']*100:.1f}%)")
    print(f"  {'subset':<14} {'n':>4} {'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6} {'spec':>6}  "
          f"{'tp':>3} {'fp':>3} {'fn':>3} {'tn':>3}")
    print(f"  {'overall':<14} {o['n']:>4} "
          f"{_fmt_pct(o['accuracy']):>6} {_fmt_pct(o['precision']):>6} "
          f"{_fmt_pct(o['recall']):>6} {_fmt_pct(o['f1']):>6} {_fmt_pct(o['specificity']):>6}  "
          f"{o['tp']:>3} {o['fp']:>3} {o['fn']:>3} {o['tn']:>3}")
    for sub, m in r["by_subgroup"].items():
        print(f"  {sub:<14} {m['n']:>4} "
              f"{_fmt_pct(m['accuracy']):>6} {_fmt_pct(m['precision']):>6} "
              f"{_fmt_pct(m['recall']):>6} {_fmt_pct(m['f1']):>6} {_fmt_pct(m['specificity']):>6}  "
              f"{m['tp']:>3} {m['fp']:>3} {m['fn']:>3} {m['tn']:>3}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", required=True, nargs="+",
                    help="One or more per-sample logs/ directories")
    args = ap.parse_args()

    subgroups = _load_manifest_subgroups()
    if subgroups:
        print(f"(manifest subgroups loaded for {len(subgroups)} samples)")
    else:
        print("(no manifest subgroups found; reporting pos/neg by label only)")

    for d in args.logs_dir:
        print_report(analyze(Path(d), subgroups))


if __name__ == "__main__":
    main()
