#!/bin/bash
#SBATCH --job-name=mirage-smoke-cot-qwen3-8b-medmcqa
#SBATCH --account=slmreasoning
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH -o slurm.%x.%j.out
#SBATCH -e slurm.%x.%j.err

# SMOKE TEST of the CoT pipeline (cell 1: Debate+CoT, Qwen3-8B, MedMCQA).
# Purpose: surface any remaining runtime errors FAST. Short walltime + 1 GPU + small mem
# all backfill quickly. CoT does no retrieval (tensor_parallel_size=1, no FAISS), so 1 GPU
# is enough. Writes to a SEPARATE log dir so it starts fresh and never touches the real
# gap-fill logs. It will run questions until the 20-min walltime kills it (TIMEOUT is the
# expected end state) — by then the log shows whether dataset load + the first debates work.

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
  --model Qwen/Qwen3-8B \
  --gpu_ids 0 \
  --log_dir ./debate_logs_smoke \
  --corpus_name MedCorp \
  2>&1 | tee -a "$LOG_FILE"

EC=${PIPESTATUS[0]}
echo "END: $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$LOG_FILE"
echo "Exit code: $EC" | tee -a "$LOG_FILE"
exit $EC
