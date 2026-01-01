#!/bin/bash
#SBATCH --job-name=grpo-pred-train
#SBATCH --account=slmreasoning
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
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

# --- Repo ---
cd /projects/slmreasoning/junh/Debate/KARE/verl

# Debug: Print working directory
echo "Working directory: $(pwd)"

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

# --- VERL GRPO Training Setup ---

# Data paths (adjust to server paths)
train_path=/projects/slmreasoning/junh/Debate/KARE/verl/data_generation/prediction/train.parquet
test_path=/projects/slmreasoning/junh/Debate/KARE/verl/data_generation/prediction/val.parquet

# Output paths (on server)
checkpoint_dir=/projects/slmreasoning/junh/Debate/KARE/verl/checkpoints/prediction
log_dir=/projects/slmreasoning/junh/Debate/KARE/verl/logs

# Create directories if they don't exist
mkdir -p $checkpoint_dir
mkdir -p $log_dir

# GPUs are set by SLURM via CUDA_VISIBLE_DEVICES automatically
# For 4 GPUs allocated, they will be visible as 0,1,2,3

# Enable vLLM V1 as required by VERL
export VLLM_USE_V1=1

# Add reward function to Python path
export PYTHONPATH=/projects/slmreasoning/junh/Debate/KARE/verl:$PYTHONPATH

# Set Ray temp directory to a short path to avoid Unix socket path length limit (107 bytes)
export RAY_TMPDIR=/projects/slmreasoning/junh/tmp/ray
mkdir -p $RAY_TMPDIR

# --- WandB Setup ---
# Check if WandB is logged in, if not use offline mode
if ! conda run -p "$ENV_PATH" wandb verify >/dev/null 2>&1; then
    echo "WARNING: WandB not logged in, running in offline mode" | tee -a "$LOG_FILE"
    export WANDB_MODE=offline
else
    echo "WandB logged in successfully" | tee -a "$LOG_FILE"
fi

# --- Run GRPO Training ---
echo "=== Starting GRPO Prediction Accuracy Training ===" | tee -a "$LOG_FILE"

stdbuf -oL -eL conda run -p "$ENV_PATH" --no-capture-output \
    python -u -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files="['$train_path']" \
        data.val_files="['$test_path']" \
        data.train_batch_size=8 \
        data.max_prompt_length=16384 \
        data.max_response_length=8192 \
        data.filter_overlong_prompts=False \
        data.truncation='error' \
        data.shuffle=False \
        actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=4 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.rollout.n=2 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=False \
        actor_rollout_ref.rollout.val_kwargs.temperature=0 \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        custom_reward_function.path=/projects/slmreasoning/junh/Debate/KARE/verl/reward_score/kare_prediction_reward.py \
        custom_reward_function.name=compute_score \
        trainer.critic_warmup=0 \
        trainer.logger='["console","wandb"]' \
        trainer.project_name='verl-kare-prediction' \
        trainer.experiment_name='grpo-qwen2.5-7b-prediction' \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.save_freq=50 \
        trainer.test_freq=50 \
        trainer.total_epochs=3 \
        trainer.default_hdfs_dir=null \
        trainer.default_local_dir=$checkpoint_dir \
        trainer.resume_mode=disable \
    2>&1 | tee -a "$LOG_FILE"

EC=${PIPESTATUS[0]}
echo "END: $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$LOG_FILE"
echo "Exit code: $EC" | tee -a "$LOG_FILE"
echo "Training completed! Checkpoints saved to: $checkpoint_dir" | tee -a "$LOG_FILE"
tail -n 100 "$LOG_FILE" || true
exit $EC
