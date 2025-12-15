#!/bin/bash
set -x

# ==============================================================================
# KARE Mortality Format Enforcement Training with GRPO
# ==============================================================================
# This script trains Qwen2.5-7B-Instruct to reliably output mortality 
# probabilities in the correct format: "MORTALITY PROBABILITY: X.XX"
#
# Hardware: 2x A40 40GB GPUs (IDs: 3,5)
# Training: GRPO with binary format reward (1 if valid format, 0 otherwise)
# ==============================================================================

# Data paths
train_path=/data/wang/junh/githubs/Debate/KARE/verl/data_generation/survival_grpo_data_hard/train.parquet
test_path=/data/wang/junh/githubs/Debate/KARE/verl/data_generation/survival_grpo_data_hard/test.parquet

# Output paths
checkpoint_dir=/data/wang/junh/githubs/Debate/KARE/verl/checkpoints
log_dir=/data/wang/junh/githubs/Debate/KARE/verl/logs

# Create directories if they don't exist
mkdir -p $checkpoint_dir
mkdir -p $log_dir

# Set GPUs - Using 4 GPUs with tensor parallelism for vLLM rollout
export CUDA_VISIBLE_DEVICES=3,4,5,6

# Enable vLLM V1 as required by VERL
export VLLM_USE_V1=1

# Add reward function to Python path
export PYTHONPATH=/data/wang/junh/githubs/Debate/KARE/verl:$PYTHONPATH

# Set Ray temp directory to a short path to avoid Unix socket path length limit (107 bytes)
export RAY_TMPDIR=/data/wang/junh/tmp/
mkdir -p $RAY_TMPDIR

# Training configuration
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="['$train_path']" \
    data.val_files="['$test_path']" \
    data.train_batch_size=8 \
    data.max_prompt_length=16384 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path=/data/wang/junh/githubs/Debate/KARE/verl/models/format_enforcer_7b_step57 \
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
    custom_reward_function.path=/data/wang/junh/githubs/Debate/KARE/verl/reward_score/kare_survival_format.py \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl-kare-mortality-format' \
    trainer.experiment_name='grpo-qwen2.5-7b-survival-format' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=3 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$checkpoint_dir \
    trainer.resume_mode=disable  # Starting new survival training from mortality checkpoint

echo "Training completed! Checkpoints saved to: $checkpoint_dir"
echo "Logs saved to: $log_dir"
