#!/bin/bash
# Submit all 8 MIRAGE gap-fill jobs to slurm in one shot.
# Each cell is an independent sbatch — slurm queues them in parallel as A100s free up.
#
# Usage:
#   cd /projects/slmreasoning/junh/Debate
#   bash DAIH/experiments/sbatch/submit_all.sh
#
# To submit only a subset, pass the numbers as args:
#   bash DAIH/experiments/sbatch/submit_all.sh 5 7 8     # only cells 5, 7, 8

set -euo pipefail

SBATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

declare -A CELLS=(
  [1]="mirage_01_cot_qwen3_8b_medmcqa.sbatch"
  [2]="mirage_02_cot_qwen25_7b_medmcqa.sbatch"
  [3]="mirage_03_cot_qwen3_4b_2507_medmcqa.sbatch"
  [4]="mirage_04_rag_qwen25_7b_medmcqa.sbatch"
  [5]="mirage_05_rag_qwen3_4b_2507_medqa.sbatch"
  [6]="mirage_06_rag_qwen3_4b_2507_medmcqa.sbatch"
  [7]="mirage_07_rag_qwen3_4b_2507_pubmedqa.sbatch"
  [8]="mirage_08_rag_qwen3_4b_2507_bioasq.sbatch"
)

if [ "$#" -eq 0 ]; then
  CELL_IDS=(1 2 3 4 5 6 7 8)
else
  CELL_IDS=("$@")
fi

for i in "${CELL_IDS[@]}"; do
  FILE="${CELLS[$i]:-}"
  if [ -z "$FILE" ]; then
    echo "Unknown cell ID: $i  (valid: 1..8)" >&2
    exit 1
  fi
  PATH_TO_SBATCH="$SBATCH_DIR/$FILE"
  if [ ! -f "$PATH_TO_SBATCH" ]; then
    echo "Missing sbatch file: $PATH_TO_SBATCH" >&2
    exit 1
  fi
  echo "Submitting cell $i  ($FILE) ..."
  sbatch "$PATH_TO_SBATCH"
done

echo
echo "Submitted ${#CELL_IDS[@]} job(s). Check progress with:  squeue -u \$USER"
