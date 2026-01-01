#!/bin/bash
# ==============================================================================
# Search-R1 Training for KARE Mortality Prediction - Probability-Based Reward
# ==============================================================================
# This script trains Qwen2.5-7B-Instruct using Search-R1 with probability-based
# calibration rewards (Option A: positive-only reward)
#
# Data: 200 train samples (100 survival, 100 mortality)
#       50 val samples (25 survival, 25 mortality)
# Hardware: 4x A40 GPUs (IDs: 1,3,4,5)
# Training: GRPO with MedRAG retrieval via HTTP server
#
# Key Differences from Binary Training:
# - Prompts request BOTH mortality and survival probabilities
# - Custom reward function: reward = mort_prob if GT=1, else (1-mort_prob)
# - Validates probabilities sum to 1.0 (±5% tolerance)
# - Range: [0.0, 1.0] smooth calibration signal
# ==============================================================================

set -e  # Exit on error

# Parse arguments
GPU_IDS=${1:-1,3,4,5}  # Default to GPUs 1,3,4,5
NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)  # Count GPUs

echo "=========================================="
echo "Search-R1 KARE Mortality Prediction Training"
echo "MODE: Probability-Based Calibration (Exp 2)"
echo "=========================================="
echo "Using GPU(s): $GPU_IDS (Total: $NUM_GPUS GPUs)"

# Configuration
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
export EXPERIMENT_NAME='searchr1-kare-mortality-prob'
export DATA_DIR='/data/wang/junh/githubs/Debate/KARE/searchr1/data/kare_mortality_prob'
export VLLM_ATTENTION_BACKEND=XFORMERS

# Custom reward function path
export REWARD_FUNCTION_PATH="/data/wang/junh/githubs/Debate/KARE/searchr1/reward_functions/kare_mortality_probability.py"

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Output directories
export CHECKPOINT_DIR="/data/wang/junh/githubs/Debate/KARE/searchr1/checkpoints/${EXPERIMENT_NAME}"
export ROLLOUT_DATA_DIR="/data/wang/junh/githubs/Debate/KARE/searchr1/rollout_data/${EXPERIMENT_NAME}"
mkdir -p $CHECKPOINT_DIR
mkdir -p $ROLLOUT_DATA_DIR

# Ray tmp directory
export RAY_TMPDIR=/data/wang/junh/tmp/
mkdir -p $RAY_TMPDIR

# Paths
SEARCH_R1_ROOT="/data/wang/junh/githubs/Search-R1"
RETRIEVER_URL="http://127.0.0.1:8000/retrieve"

# Pre-flight checks
echo ""
echo "[1/4] Pre-flight checks..."

# Check MedRAG server
if ! curl -s "${RETRIEVER_URL%/retrieve}/" > /dev/null; then
    echo "❌ ERROR: MedRAG server not running on port 8000"
    echo ""
    echo "Start it in another terminal with:"
    echo "  cd /data/wang/junh/githubs/Debate/KARE"
    echo "  nohup python searchr1/medrag_retrieval_server.py --port 8000 > medrag_server.log 2>&1 &"
    exit 1
fi
echo "✓ MedRAG server is running"

# Check custom reward function
if [ ! -f "$REWARD_FUNCTION_PATH" ]; then
    echo "❌ ERROR: Custom reward function not found at $REWARD_FUNCTION_PATH"
    exit 1
fi
echo "✓ Custom reward function found"

# Check training data
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "❌ ERROR: Training data not found at $DATA_DIR/train.parquet"
    echo ""
    echo "Generate probability-based data with:"
    echo "  cd /data/wang/junh/githubs/Debate/KARE"
    echo "  python searchr1/data_generation/prepare_searchr1_balanced_data.py \\"
    echo "    --balanced_json searchr1/data_generation/train_balanced_100pos_100neg.json \\"
    echo "    --split train \\"
    echo "    --output_dir searchr1/data/kare_mortality_prob"
    exit 1
fi
echo "✓ Training data found"

# Check validation data
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

# Verify reward function works
echo ""
echo "[2/4] Testing custom reward function..."
python3 -c "
import sys
sys.path.insert(0, '/data/wang/junh/githubs/Debate/KARE')
from searchr1.reward_functions.kare_mortality_probability import compute_score

# Quick test
test_output = '''<answer>
MORTALITY PROBABILITY: 0.85
SURVIVAL PROBABILITY: 0.15
</answer>'''
reward = compute_score(test_output, {'target': ['1']})
assert reward == 0.85, f'Expected 0.85, got {reward}'
print('✓ Reward function validated (test reward: {:.3f})'.format(reward))
"
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Reward function test failed"
    exit 1
fi

# Training configuration summary
echo ""
echo "[3/4] Training configuration..."
echo "  Model: $BASE_MODEL"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Reward: Probability-based (positive-only, range [0.0, 1.0])"
echo "  Retriever: $RETRIEVER_URL (topk=5)"
echo "  GPUs: $NUM_GPUS x A40 (IDs: $GPU_IDS)"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "  Rollout logs: $ROLLOUT_DATA_DIR"
echo ""

# Training
echo ""
echo "[4/4] Starting Search-R1 training..."
echo "  Time: $(date)"
echo ""

cd "$SEARCH_R1_ROOT"

# Add custom reward function to Python path
export PYTHONPATH="/data/wang/junh/githubs/Debate/KARE:$PYTHONPATH"

# Redirect verbose Ray worker output to log file
export RAY_LOG_TO_STDERR=0

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
    trainer.n_gpus_per_node=$NUM_GPUS \
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
    retriever.url="$RETRIEVER_URL" \
    retriever.topk=5 \
    +reward.custom_reward_module="searchr1.reward_functions.kare_mortality_probability" \
    +reward.custom_reward_function="compute_score"

echo ""
echo "=========================================="
echo "✓ Training complete!"
echo "=========================================="
echo "  Finished at: $(date)"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "  Rollout logs: $ROLLOUT_DATA_DIR"
echo ""
echo "To analyze results:"
echo "  - WandB dashboard: https://wandb.ai/your-username/searchr1-kare-mortality"
echo "  - Checkpoints: ls -lh $CHECKPOINT_DIR"
echo "  - Rollout logs: ls -lh $ROLLOUT_DATA_DIR"
echo "=========================================="
