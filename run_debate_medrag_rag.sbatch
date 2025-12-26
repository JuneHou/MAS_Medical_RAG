#!/bin/bash
#SBATCH --job-name=KARE-mor-rag-qwen2.5-qwen3-70b
#SBATCH --account=slmreasoning
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
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

# --- KARE mortality debate (RAG mode with MedRAG) ---
# Uses defaults: MedCorp2 / MedCPT / db_dir, round1_k=8, round3_k=16
# FAST MODE: Uses precomputed debate logs from agents 1-3, only runs integrator (agent 4)
PRECOMPUTED_LOGS="/projects/slmreasoning/junh/Debate/KARE/results/rag_mor_Qwen_Qwen2.5_7B_Instruct_MedCPT_8_8/debate_logs"

stdbuf -oL -eL conda run -p "$ENV_PATH" --no-capture-output \
  python -u KARE/run_kare_debate_mortality_fast.py \
    --mode rag \
    --gpus 0,1 \
    --integrator_model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --integrator_gpu 1 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --db_dir /projects/slmreasoning/junh/mirage_medrag/MedRAG/src/data/corpus \
    --precomputed_log_dir "$PRECOMPUTED_LOGS" \
    --output /projects/slmreasoning/junh/Debate/KARE/results/rag_mor_Qwen2.5_3-70b_8_16/kare_debate_mortality_results.json \
  2>&1 | tee -a "$LOG_FILE"

EC=${PIPESTATUS[0]}
echo "END: $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$LOG_FILE"
echo "Exit code: $EC" | tee -a "$LOG_FILE"
tail -n 80 "$LOG_FILE" || true
exit $EC
