#!/usr/bin/env python3
"""
Re-parse already-saved per-sample logs after a regex/parser update.

Reads each {logs_dir}/{sample_id}.json, runs the (now-updated)
extract_probabilities() on the saved integrator response, and writes the
corrected mortality_probability / survival_probability / prediction back to
the same file. Preserves all other fields. No API or model calls.

Use after editing KARE/gpt/src/gpt_utils.py:extract_probabilities() to avoid
re-running expensive LLM inference.

Usage:
    python DAIH/experiments/reparse_logs.py \
        --logs_dir DAIH/results/condition_A_oss/logs

    # or all DAIH condition dirs at once:
    for d in DAIH/results/*/logs; do
        python DAIH/experiments/reparse_logs.py --logs_dir "$d"
    done
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path("/data/wang/junh/githubs/Debate")
sys.path.insert(0, str(REPO_ROOT / "KARE" / "gpt" / "src"))

from gpt_utils import extract_probabilities  # noqa: E402


# Possible field names to re-parse, in priority order. We prefer the integrator's
# FINAL response (post-retrieval); if absent we fall back to the initial response
# or the qwen_integrator field used by Cond D'.
INTEGRATOR_FIELDS = [
    "gpt_integrator_final",
    "gpt_integrator_initial",
    "qwen_integrator",
]


def main():
    p = argparse.ArgumentParser(description="Re-parse per-sample logs with updated extract_probabilities()")
    p.add_argument("--logs_dir", required=True, type=str)
    p.add_argument("--dry_run", action="store_true",
                   help="Print what would change but don't write back")
    args = p.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.is_dir():
        sys.exit(f"ERROR: {logs_dir} not a directory")

    files = sorted(logs_dir.glob("*.json"))
    print(f"Found {len(files)} log files in {logs_dir}")

    n_changed = 0
    n_recovered = 0  # parse failure → success
    n_unchanged = 0
    n_no_text = 0

    for fp in files:
        with open(fp) as fh:
            d = json.load(fh)

        # Find the integrator text
        text = None
        used_field = None
        for f in INTEGRATOR_FIELDS:
            if d.get(f):
                text = d[f]
                used_field = f
                break

        if not text:
            n_no_text += 1
            continue

        new = extract_probabilities(text)
        old_pred = d.get("prediction")
        old_mort = d.get("mortality_probability")
        old_surv = d.get("survival_probability")

        # If new parse failed too, leave as is
        if new["mortality_probability"] is None and new["survival_probability"] is None:
            n_unchanged += 1
            continue

        # Did the new parse change anything?
        changed = (
            new["prediction"] != old_pred
            or new["mortality_probability"] != old_mort
            or new["survival_probability"] != old_surv
        )

        if not changed:
            n_unchanged += 1
            continue

        # Detect "recovery": old prediction came from fallback (mort=0.0 + label=1, or mort=1.0 + label=0)
        label = d.get("label")
        was_fallback = (
            old_mort in (0.0, 1.0)
            and label is not None
            and old_pred == (1 - int(label))
        )
        tag = "RECOVERED" if was_fallback else "updated  "
        if was_fallback:
            n_recovered += 1
        n_changed += 1

        print(
            f"  [{tag}] {d.get('sample_id')}: "
            f"label={label} | "
            f"pred {old_pred} -> {new['prediction']} | "
            f"mort {old_mort} -> {new['mortality_probability']} | "
            f"surv {old_surv} -> {new['survival_probability']} | "
            f"(from {used_field})"
        )

        if not args.dry_run:
            d["mortality_probability"] = new["mortality_probability"]
            d["survival_probability"] = new["survival_probability"]
            d["prediction"] = new["prediction"]
            d["_reparsed"] = True
            with open(fp, "w") as fh:
                json.dump(d, fh, indent=2, default=str)

    print()
    print(f"Summary: {n_changed} changed ({n_recovered} recovered from fallback), "
          f"{n_unchanged} unchanged, {n_no_text} no integrator text")
    if args.dry_run:
        print("(dry-run: no files written)")


if __name__ == "__main__":
    main()
