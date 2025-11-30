#!/bin/bash
#SBATCH --job-name=KARE-mor-cot-qwen2.5
#SBATCH --account=slmreasoning
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
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

# --- Repo (adjust if your KARE script lives elsewhere) ---
cd /projects/slmreasoning/junh/Debate

# Debug: Print working directory and permissions
echo "Working directory: $(pwd)"
echo "Directory contents: $(ls -la KARE/ | head -3)"
echo "Results directory: $(ls -ld KARE/results/ 2>/dev/null || echo 'Does not exist - will be created')"

# --- Logging: mirror terminal output into a logfile too ---
mkdir -p logs
LOG_FILE="logs/run_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
echo "Log file: $LOG_FILE"

echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-N/A}" | tee -a "$LOG_FILE"
echo "Using conda: $(conda info --base 2>/dev/null || echo 'unknown')" | tee -a "$LOG_FILE"
conda run -p "$ENV_PATH" python -V | tee -a "$LOG_FILE"
conda run -p "$ENV_PATH" python -c "import sys; print('PY:', sys.executable)" | tee -a "$LOG_FILE"
conda run -p "$ENV_PATH" python -c "import torch; print('torch:', torch.__version__)" | tee -a "$LOG_FILE"
echo "START: $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$LOG_FILE"

# Set Python to be completely unbuffered for SLURM
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# --- KARE mortality debate (COT mode, no RAG) ---
# Defaults: Qwen/Qwen3-4B-Instruct-2507, processes all test samples, auto output path
stdbuf -oL -eL conda run -p "$ENV_PATH" --no-capture-output \
  python -u KARE/run_kare_debate_mortality.py \
    --mode cot \
    --gpus 0 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --db_dir /projects/slmreasoning/junh/mirage_medrag/MedRAG/src/data/corpus \
    --output /projects/slmreasoning/junh/Debate/KARE/results/cot_mor_Qwen_Qwen2.5_7B_Instruct/kare_debate_mortality_results.json \
  2>&1 | tee -a "$LOG_FILE"

EC=${PIPESTATUS[0]}
echo "END: $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$LOG_FILE"
echo "Exit code: $EC" | tee -a "$LOG_FILE"
tail -n 80 "$LOG_FILE" || true
exit $EC
