#!/bin/bash
#SBATCH --job-name=mirage-03-cot-qwen3-4b-2507-medmcqa
#SBATCH --account=slmreasoning
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=30:00:00
#SBATCH -o slurm.%x.%j.out
#SBATCH -e slurm.%x.%j.err

# Cell 3 of MIRAGE gap-fill: Debate+CoT, Qwen3-4B-Instruct-2507, MedMCQA (4,183 questions).
# Estimated wall-clock on 2 A100s: ~23 hr.
# Verify the HF model ID "Qwen/Qwen3-4B-Instruct-2507" matches your intended model.
# CoT mode = no retrieval, no corpus.

set -euo pipefail

module load Miniconda3/24.7.1-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/junh/envs/medrag
export PATH="/home/junh/envs/medrag/bin:$PATH"
export PYTHONNOUSERSITE=1
# fail fast if python is not 3.10 from the medrag env (avoids base python-3.13 fallback)
python -c "import sys; assert sys.executable.startswith('/home/junh/envs/medrag/'), sys.executable" || { echo 'WRONG PYTHON: medrag env not active' >&2; exit 1; }

export HF_HOME=/scratch/junh/hf_cache
export MIRAGE_MEDRAG_ROOT=/home/junh/repos/mirage_medrag
export TRANSFORMERS_CACHE="$HF_HOME"
export SENTENCE_TRANSFORMERS_HOME=/scratch/junh/hf_cache_st
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export VLLM_USE_FLASHINFER_SAMPLER=0   # force PyTorch-native top-k/top-p; node has no nvcc for FlashInfer JIT

mkdir -p "$HF_HOME"
cd /home/junh/repos/Debate

mkdir -p logs
LOG_FILE="logs/run_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
echo "START: $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee "$LOG_FILE"
echo "Host: $(hostname)" | tee -a "$LOG_FILE"

stdbuf -oL -eL python -u run_debate_medrag_inter.py \
  --mode cot \
  --dataset medmcqa \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --gpu_ids 0,1 \
  --log_dir ./debate_logs_mirage_gapfill \
  --corpus_name MedCorp \
  2>&1 | tee -a "$LOG_FILE"

EC=${PIPESTATUS[0]}
echo "END: $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$LOG_FILE"
echo "Exit code: $EC" | tee -a "$LOG_FILE"
exit $EC
