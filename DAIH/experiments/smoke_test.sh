#!/usr/bin/env bash
# Smoke test for the oss-120b factorial runners.
#
# Verifies:
#   1. ARC API key is loaded
#   2. ARCClient can hit the endpoint
#   3. Cond A' can complete 5 patients end-to-end (analysts + retrieval + integrator)
#   4. Cond D' can reuse those cached outputs and run the local Qwen integrator
#
# Run from repo root: bash DAIH/experiments/smoke_test.sh

set -euo pipefail

REPO_ROOT="/data/wang/junh/githubs/Debate"
cd "$REPO_ROOT"

if [[ -z "${ARC_LLM_API_KEY:-}" ]]; then
    echo "[smoke] ARC_LLM_API_KEY not set — sourcing key file..."
    if [[ -f /data/wang/junh/.cache/keys/arc_llm_api.sh ]]; then
        set -a; source /data/wang/junh/.cache/keys/arc_llm_api.sh; set +a
        export ARC_LLM_API_KEY="${API_KEY:-${ARC_LLM_API_KEY:-}}"
    else
        echo "[smoke] ERROR: key file not found at /data/wang/junh/.cache/keys/arc_llm_api.sh" >&2
        exit 1
    fi
fi

echo "=== [1/3] ARCClient self-test (1 API call) ==="
python DAIH/experiments/arc_client.py

echo
echo "=== [2/3] Cond A' on first 5 patients ==="
python DAIH/experiments/run_condition_A_oss.py --limit 5

echo
echo "=== [3/3] Cond D' on the 5 cached Cond A' samples ==="
python DAIH/experiments/run_condition_D_oss.py --limit 5

echo
echo "=== Smoke test complete ==="
echo "Cond A' logs: DAIH/results/condition_A_oss/logs/"
echo "Cond D' logs: DAIH/results/condition_D_oss_qwen_int/logs/"
echo
echo "Inspect a sample: head -50 DAIH/results/condition_A_oss/logs/*.json | head -100"
