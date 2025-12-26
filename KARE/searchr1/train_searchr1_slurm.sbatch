#!/bin/bash
#SBATCH --job-name=searchr1-kare
#SBATCH --output=logs/searchr1_kare_%j.out
#SBATCH --error=logs/searchr1_kare_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gputype=A40
#SBATCH --gpus-per-node=4
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# ==============================================================================
# SLURM Job for Search-R1 KARE Mortality Prediction Training
# ==============================================================================
# Runs entirely on allocated GPUs without external HTTP servers
# Uses 4 GPUs: GPU 0-2 for training, GPU 3 for MedRAG retrieval
# ==============================================================================

set -e

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "=========================================="

# Create log directory
mkdir -p logs

# Conda environment
source ~/.bashrc
conda activate searchr1

# Configuration
export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
export EXPERIMENT_NAME='searchr1-kare-mortality-single-agent'
export DATA_DIR='/data/wang/junh/githubs/Debate/KARE/searchr1/data/kare_mortality_single_agent'
export CHECKPOINT_DIR="/data/wang/junh/githubs/Debate/KARE/searchr1/checkpoints/${EXPERIMENT_NAME}"
export ROLLOUT_DATA_DIR="/data/wang/junh/githubs/Debate/KARE/searchr1/rollout_data/${EXPERIMENT_NAME}"

# GPU allocation
# IMPORTANT: Use GPUs 0,1,2 for training, GPU 3 for retrieval
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RETRIEVER_GPU=3  # Last GPU for retrieval
export NUM_TRAIN_GPUS=3  # First 3 GPUs for training

# Memory optimizations
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ray configuration (local mode, no head node needed)
export RAY_TMPDIR=/tmp/ray_${SLURM_JOB_ID}
mkdir -p $RAY_TMPDIR

# Create output directories
mkdir -p $CHECKPOINT_DIR
mkdir -p $ROLLOUT_DATA_DIR

# Pre-flight checks
echo ""
echo "Pre-flight checks..."
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "❌ ERROR: Training data not found at $DATA_DIR/train.parquet"
    exit 1
fi
echo "✓ Training data found"

if [ ! -f "$DATA_DIR/val.parquet" ]; then
    echo "❌ ERROR: Validation data not found at $DATA_DIR/val.parquet"
    exit 1
fi
echo "✓ Validation data found"

# Count samples
train_samples=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$DATA_DIR/train.parquet')))")
val_samples=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$DATA_DIR/val.parquet')))")
echo "  - Train samples: $train_samples"
echo "  - Val samples: $val_samples"

echo ""
echo "Training configuration:"
echo "  Model: $BASE_MODEL"
echo "  Training GPUs: 0,1,2 (3 GPUs)"
echo "  Retrieval GPU: 3"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo ""

# Change to Search-R1 directory
cd /data/wang/junh/githubs/Search-R1

# Suppress Ray worker logs
export RAY_LOG_TO_STDERR=0

echo "Starting Search-R1 training at $(date)..."
echo ""

# Run training with local retriever (NO HTTP server needed!)
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/val.parquet" \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=8 \
    data.val_batch_size=4 \
    data.max_prompt_length=18000 \
    data.max_response_length=3072 \
    data.max_start_length=6144 \
    data.max_obs_length=3072 \
    +data.truncation=left \
    data.shuffle_train_dataloader=False \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=22000 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.state_masking=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.no_think_rl=false \
    critic.model.path="$BASE_MODEL" \
    critic.model.enable_gradient_checkpointing=true \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.grad_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    critic.ppo_micro_batch_size=8 \
    trainer.logger='["console","wandb"]' \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.n_gpus_per_node=$NUM_TRAIN_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.project_name='searchr1-kare-mortality' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.total_epochs=5 \
    trainer.total_training_steps=null \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    +trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
    max_turns=2 \
    +retriever.local=true \
    +retriever.gpu_id=$RETRIEVER_GPU \
    +retriever.corpus_name=MedCorp \
    +retriever.retriever_name=MedCPT \
    retriever.topk=5

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
else
    echo "❌ Training failed with exit code $EXIT_CODE"
fi
echo "=========================================="
echo "Finished at: $(date)"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Logs: logs/searchr1_kare_${SLURM_JOB_ID}.out"
echo "=========================================="

# Cleanup Ray temp directory
rm -rf $RAY_TMPDIR

exit $EXIT_CODE
