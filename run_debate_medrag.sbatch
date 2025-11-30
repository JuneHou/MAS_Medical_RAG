#!/bin/bash
#SBATCH --job-name=resume-qwen3-8B-CoT-equal-mmlu
#SBATCH --account=slmreasoning
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH -o slurm.%x.%j.out
#SBATCH -e slurm.%x.%j.err

set -euo pipefail

# --- Modules / env ---
module load Miniconda3
ENV_PATH="/projects/slmreasoning/junh/envs/medrag"

# Caches under /projects to avoid $HOME quota
export HF_HOME=/projects/slmreasoning/junh/.cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"
export XDG_CACHE_HOME=/projects/slmreasoning/junh/.cache
export SENTENCE_TRANSFORMERS_HOME=/projects/slmreasoning/junh/.cache/sentence-transformers
export TOKENIZERS_PARALLELISM=false

# (Optional) node-local scratch for temp files
export TMPDIR=/localscratch-nvme/$SLURM_JOB_ID
mkdir -p "$TMPDIR"

# --- Repo (adjust if your Debate repo lives elsewhere) ---
cd /projects/slmreasoning/junh/Debate

# --- Logging: mirror terminal output into a logfile too ---
mkdir -p logs
LOG_FILE="logs/run_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"

echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-N/A}" | tee -a "$LOG_FILE"
echo "Using conda: $(conda info --base 2>/dev/null || echo 'unknown')" | tee -a "$LOG_FILE"
conda run -p "$ENV_PATH" python -V | tee -a "$LOG_FILE"
conda run -p "$ENV_PATH" python -c "import sys; print('PY:', sys.executable)" | tee -a "$LOG_FILE"
conda run -p "$ENV_PATH" python -c "import torch; print('torch:', torch.__version__)" | tee -a "$LOG_FILE"
echo "START: $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$LOG_FILE"

# --- EXACT command (matching your local run; note: flag is --rounds) ---
# Local: python run_debate_medrag.py --mode cot --k 8 --dataset mmlu --round 2
# Here:  --rounds 2 (per your argparse)
stdbuf -oL -eL conda run -p "$ENV_PATH" --no-capture-output \
  python -u run_debate_medrag_cot_box.py \
    --dataset mmlu \
    --rounds 2 \
    --log_dir /projects/slmreasoning/junh/Debate/qwen3-8B-CoT-equal-mmlu_logs
  2>&1 | tee -a "$LOG_FILE"

EC=${PIPESTATUS[0]}
echo "END: $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$LOG_FILE"
echo "Exit code: $EC" | tee -a "$LOG_FILE"
tail -n 80 "$LOG_FILE" || true
exit $EC
