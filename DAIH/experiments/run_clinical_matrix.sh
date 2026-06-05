#!/bin/bash
# Run the clinical generalization matrix: Cond D' (gpt-oss-120b analysts+retrieval
# via ARC + local Qwen integrator) over {mortality, readmission} x {mimic3, mimic4},
# on the FULL test set of each cell.
#
# Per cell: build manifest -> Cond A' (oss, fills analyst+retrieval cache)
#           -> Cond D' (Qwen integrator) -> metrics.
# All steps are resumable (runners skip already-done samples).
#
# Usage:
#   # all 4 cells, full test set:
#   bash DAIH/experiments/run_clinical_matrix.sh
#
#   # subset of cells (space-separated "dataset:task" tokens):
#   bash DAIH/experiments/run_clinical_matrix.sh mimic3:mortality mimic3:readmission
#
#   # smoke test (first N patients per cell):
#   LIMIT=5 bash DAIH/experiments/run_clinical_matrix.sh mimic3:mortality
#
# Env knobs:
#   LIMIT       process at most N patients per cell (default: all)
#   GPU_ID      CUDA device for the Qwen integrator (default: 0)
#   QWEN_MODEL  integrator model (default: Qwen/Qwen2.5-7B-Instruct)
#   ARC_KEY_FILE  path to the ARC API key script (default below)

set -euo pipefail

REPO_ROOT=/data/wang/junh/githubs/Debate
cd "$REPO_ROOT"

LIMIT="${LIMIT:-}"
GPU_ID="${GPU_ID:-0}"
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
ARC_KEY_FILE="${ARC_KEY_FILE:-/data/wang/junh/.cache/keys/arc_llm_api.sh}"

# vLLM 0.11 (v1 engine) forks its engine-core subprocess by default, which crashes
# with "Cannot re-initialize CUDA in forked subprocess" for the Qwen integrator.
# Force spawn (the start method the error message prescribes).
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

# Default cell list (all 4) unless cells are passed as args.
if [ "$#" -gt 0 ]; then
  CELLS=("$@")
else
  CELLS=(mimic3:mortality mimic3:readmission mimic4:mortality mimic4:readmission)
fi

# ARC credentials for Cond A' (gpt-oss-120b).
if [ -f "$ARC_KEY_FILE" ]; then
  set -a; source "$ARC_KEY_FILE"; set +a
  export ARC_LLM_API_KEY="${API_KEY:-${ARC_LLM_API_KEY:-}}"
else
  echo "WARNING: ARC key file not found at $ARC_KEY_FILE — Cond A' will fail to authenticate." >&2
fi

LIMIT_ARG=()
if [ -n "$LIMIT" ]; then
  LIMIT_ARG=(--limit "$LIMIT")
  echo ">>> SMOKE MODE: limit=$LIMIT patients per cell"
fi

for cell in "${CELLS[@]}"; do
  DATASET="${cell%%:*}"
  TASK="${cell##*:}"
  echo ""
  echo "############################################################"
  echo "# CELL: $DATASET / $TASK"
  echo "############################################################"

  MANIFEST="KARE/gpt/manifests/fullset_${TASK}_${DATASET}.parquet"

  # 1) Manifest (full test set)
  if [ ! -f "$MANIFEST" ]; then
    echo ">>> [1/3] Building manifest: $MANIFEST"
    python DAIH/experiments/build_fullset_manifest.py --dataset "$DATASET" --task "$TASK"
  else
    echo ">>> [1/3] Manifest exists: $MANIFEST"
  fi

  # 2) Cond A' — oss analysts + retrieval (caches per-sample logs for D')
  echo ">>> [2/3] Cond A' (gpt-oss-120b analysts + retrieval)"
  python DAIH/experiments/run_condition_A_oss.py \
    --task "$TASK" --dataset "$DATASET" "${LIMIT_ARG[@]}"

  # 3) Cond D' — Qwen integrator over the cached oss analyst/retrieval outputs
  echo ">>> [3/3] Cond D' (Qwen integrator: $QWEN_MODEL)"
  python DAIH/experiments/run_condition_D_oss.py \
    --task "$TASK" --dataset "$DATASET" \
    --qwen_model "$QWEN_MODEL" --gpu_id "$GPU_ID" "${LIMIT_ARG[@]}"

  # Metrics for this cell
  echo ">>> Metrics: $DATASET / $TASK"
  python DAIH/experiments/compute_metrics.py \
    --logs_dir "DAIH/results/condition_D_oss_qwen_int_${TASK}_${DATASET}/logs"
done

echo ""
echo "############################################################"
echo "# DONE. Cond D' logs per cell:"
for cell in "${CELLS[@]}"; do
  DATASET="${cell%%:*}"; TASK="${cell##*:}"
  echo "#   DAIH/results/condition_D_oss_qwen_int_${TASK}_${DATASET}/logs"
done
echo "############################################################"
