#!/bin/bash
# Quick test run with 10 samples to verify Search-R1 setup works
# Usage: bash searchr1/test_searchr1_training.sh [GPU_IDS]
# Example: bash searchr1/test_searchr1_training.sh 3
# Example: bash searchr1/test_searchr1_training.sh 3,4,5,6

set -e  # Exit on error

# Parse arguments
GPU_IDS=${1:-0}  # Default to GPU 0 if not specified
NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)  # Count GPUs

echo "=========================================="
echo "Search-R1 Quick Test (10 samples)"
echo "=========================================="
echo "Using GPU(s): $GPU_IDS (Total: $NUM_GPUS GPUs)"

# Configuration
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
export EXPERIMENT_NAME='kare-test-searchr1'
export DATA_DIR='/data/wang/junh/githubs/Debate/KARE/searchr1/data/kare_mortality_single_agent'
export VLLM_ATTENTION_BACKEND=XFORMERS

# Output directories
export CHECKPOINT_DIR="/data/wang/junh/githubs/Debate/KARE/searchr1/checkpoints/${EXPERIMENT_NAME}"
export ROLLOUT_DATA_DIR="/data/wang/junh/githubs/Debate/KARE/searchr1/rollout_data/${EXPERIMENT_NAME}"
mkdir -p $CHECKPOINT_DIR
mkdir -p $ROLLOUT_DATA_DIR

# Ray tmp directory (use writable location)
export RAY_TMPDIR=/data/wang/junh/tmp/
mkdir -p $RAY_TMPDIR

# Paths - use Search-R1's veRL (has retrieval support)
SEARCH_R1_ROOT="/data/wang/junh/githubs/Search-R1"
RETRIEVER_URL="http://127.0.0.1:8000/retrieve"

# Check retriever server
echo ""
echo "[1/3] Checking MedRAG retriever server..."
if ! curl -s "${RETRIEVER_URL%/retrieve}/" > /dev/null; then
    echo "ERROR: MedRAG server not running on port 8000"
    echo "Start it in another terminal with:"
    echo "  cd /data/wang/junh/githubs/Debate/KARE"
    echo "  python searchr1/medrag_retrieval_server.py --port 8000"
    exit 1
fi
echo "✓ MedRAG server is running"

# Check data
echo ""
echo "[2/3] Checking training data..."
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "ERROR: Training data not found at $DATA_DIR/train.parquet"
    exit 1
fi
echo "✓ Training data found"
echo "  - Train: $DATA_DIR/train.parquet"
echo "  - Val: $DATA_DIR/val.parquet"

# Training
echo ""
echo "[3/3] Starting Search-R1 quick test..."
echo "  Model: $BASE_MODEL"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Retriever: $RETRIEVER_URL"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "  Rollout data: $ROLLOUT_DATA_DIR"
echo "  Using Search-R1's veRL (with retrieval support)"
echo ""

cd "$SEARCH_R1_ROOT"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/val.parquet" \
    data.train_data_num=8 \
    data.val_data_num=4 \
    data.train_batch_size=4 \
    data.val_batch_size=4 \
    data.max_prompt_length=16384 \
    data.max_response_length=4096 \
    data.max_start_length=8192 \
    data.max_obs_length=1024 \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.state_masking=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    critic.model.path="$BASE_MODEL" \
    critic.model.enable_gradient_checkpointing=true \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.grad_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    critic.ppo_micro_batch_size=4 \
    trainer.total_epochs=3 \
    trainer.default_hdfs_dir=null \
    trainer.experiment_name='searchr1-qwen2.5-7b-test' \
    trainer.logger='["wandb"]' \
    trainer.project_name='searchr1-kare-mortality' \
    +trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=2 \
    trainer.test_freq=1 \
    trainer.project_name='searchr1-kare-mortality' \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    max_turns=3 \
    retriever.url="$RETRIEVER_URL" \
    retriever.topk=5

echo ""
echo "=========================================="
echo "✓ Test complete!"
echo "=========================================="
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo "Rollout data saved to: $ROLLOUT_DATA_DIR"
