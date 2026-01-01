#!/bin/bash
#SBATCH --job-name=grpo-pred-datagen
#SBATCH --account=slmreasoning
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
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

# --- Repo ---
cd /projects/slmreasoning/junh/Debate/KARE/verl/data_generation/prediction

# Debug: Print working directory
echo "Working directory: $(pwd)"
echo "Python script: $(ls -l generate_prediction_training_data.py)"

# --- Logging ---
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

# --- Generate GRPO Training Data ---
# This runs 3-round debate pipeline and captures integrator prompts for GRPO training
# Generates BOTH train.parquet and val.parquet

# Step 1: Generate training data (train split)
echo "=== Generating TRAINING data ===" | tee -a "$LOG_FILE"
stdbuf -oL -eL conda run -p "$ENV_PATH" --no-capture-output \
  python -u generate_prediction_training_data.py \
    --split train \
    --gpus 0,1 \
    --model Qwen/Qwen2.5-14B-Instruct \
    --integrator_model Qwen/Qwen2.5-14B-Instruct \
    --db_dir /projects/slmreasoning/junh/mirage_medrag/MedRAG/src/data/corpus \
    --corpus_name MedCorp2 \
    --retriever_name MedCPT \
    --output_dir /projects/slmreasoning/junh/Debate/KARE/verl/data_generation/prediction \
  2>&1 | tee -a "$LOG_FILE"

EC_TRAIN=${PIPESTATUS[0]}
echo "Training data generation exit code: $EC_TRAIN" | tee -a "$LOG_FILE"

# Step 2: Generate validation data (val split)
echo "=== Generating VALIDATION data ===" | tee -a "$LOG_FILE"
stdbuf -oL -eL conda run -p "$ENV_PATH" --no-capture-output \
  python -u generate_prediction_training_data.py \
    --split val \
    --gpus 0,1 \
    --model Qwen/Qwen2.5-14B-Instruct \
    --integrator_model Qwen/Qwen2.5-14B-Instruct \
    --db_dir /projects/slmreasoning/junh/mirage_medrag/MedRAG/src/data/corpus \
    --corpus_name MedCorp2 \
    --retriever_name MedCPT \
    --output_dir /projects/slmreasoning/junh/Debate/KARE/verl/data_generation/prediction \
  2>&1 | tee -a "$LOG_FILE"

EC_VAL=${PIPESTATUS[0]}
echo "Validation data generation exit code: $EC_VAL" | tee -a "$LOG_FILE"

# --- Summary ---
echo "END: $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$LOG_FILE"
echo "=== SUMMARY ===" | tee -a "$LOG_FILE"
echo "Training data exit code: $EC_TRAIN" | tee -a "$LOG_FILE"
echo "Validation data exit code: $EC_VAL" | tee -a "$LOG_FILE"

if [ $EC_TRAIN -eq 0 ] && [ $EC_VAL -eq 0 ]; then
    echo "✅ SUCCESS: Both train.parquet and val.parquet generated!" | tee -a "$LOG_FILE"
    echo "Output location: /projects/slmreasoning/junh/Debate/KARE/verl/data_generation/prediction/" | tee -a "$LOG_FILE"
    exit 0
else
    echo "❌ FAILED: Check logs above for errors" | tee -a "$LOG_FILE"
    exit 1
fi
