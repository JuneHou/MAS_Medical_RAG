#!/bin/bash
#SBATCH --job-name=searchr1-prob-900
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

# ==============================================================================
# SLURM Job for Search-R1 KARE Mortality Prediction Training
# Robust version: DO NOT rely on conda activate; use env python explicitly.
# ==============================================================================

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         ${SLURM_NODELIST}"
echo "GPUs:         ${SLURM_GPUS_ON_NODE:-unknown}"
echo "CPUs:         ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Memory(MB):   ${SLURM_MEM_PER_NODE:-unknown}"
echo "Start time:   $(date)"
echo "Workdir:      $(pwd)"
echo "=========================================="

# --- Modules / env ---
module load Miniconda3

ENV_PATH="/projects/slmreasoning/junh/envs/searchr1"
PY="${ENV_PATH}/bin/python"

# Neutralize common HPC Python pollution
unset PYTHONPATH || true
unset PYTHONHOME || true
export PYTHONNOUSERSITE=1

# --- Logging ---
mkdir -p logs
LOG_FILE="logs/run_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
echo "Log file: ${LOG_FILE}"

# --- Hard checks for env python ---
echo ""
echo "=== PYTHON DEBUG (must use env python) ==="
echo "ENV_PATH: ${ENV_PATH}"
if [[ ! -x "${PY}" ]]; then
  echo "❌ ERROR: Env python not found or not executable: ${PY}"
  echo "Contents of ENV_PATH:"
  ls -lah "${ENV_PATH}" || true
  exit 1
fi

echo "System python3: $(command -v python3 || echo 'not found')"
python3 -c "import sys; print('system sys.executable:', sys.executable)" 2>/dev/null || echo "system python3 not usable"

echo "Env python: ${PY}"
"${PY}" -c "import sys; print('env sys.executable:', sys.executable)"
"${PY}" -m pip -V
echo "Pandas in env?"
"${PY}" -m pip show pandas || true
"${PY}" -c "import pandas as pd; print('pandas version:', pd.__version__)"
echo "=== END PYTHON DEBUG ==="
echo ""

# ==============================================================================
# Configuration
# ==============================================================================
export BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
export EXPERIMENT_NAME="searchr1-kare-mortality-single-agent"

export DATA_DIR="/projects/slmreasoning/junh/Debate/KARE/searchr1/data/kare_mortality_single_agent_900"
export CHECKPOINT_DIR="/projects/slmreasoning/junh/Debate/KARE/searchr1/checkpoints/${EXPERIMENT_NAME}"
export ROLLOUT_DATA_DIR="/projects/slmreasoning/junh/Debate/KARE/searchr1/rollout_data/${EXPERIMENT_NAME}"

# GPU allocation: 0,1,2 training; 3 retrieval
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export RETRIEVER_GPU="3"
export NUM_TRAIN_GPUS="4"

# Memory / runtime knobs
export VLLM_ATTENTION_BACKEND="XFORMERS"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Ray configuration (local)
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"
mkdir -p "${RAY_TMPDIR}"

# Output dirs
mkdir -p "${CHECKPOINT_DIR}" "${ROLLOUT_DATA_DIR}"

# ==============================================================================
# Pre-flight checks (use env python)
# ==============================================================================
echo "Pre-flight checks..."

if [[ ! -f "${DATA_DIR}/train.parquet" ]]; then
  echo "❌ ERROR: Training data not found at ${DATA_DIR}/train.parquet"
  exit 1
fi
echo "✓ Training data found"

if [[ ! -f "${DATA_DIR}/val.parquet" ]]; then
  echo "❌ ERROR: Validation data not found at ${DATA_DIR}/val.parquet"
  exit 1
fi
echo "✓ Validation data found"

train_samples=$("${PY}" -c "import pandas as pd; print(len(pd.read_parquet('${DATA_DIR}/train.parquet')))")
val_samples=$("${PY}" -c "import pandas as pd; print(len(pd.read_parquet('${DATA_DIR}/val.parquet')))")
echo "  - Train samples: ${train_samples}"
echo "  - Val samples:   ${val_samples}"

echo ""
echo "Training configuration:"
echo "  Model:          ${BASE_MODEL}"
echo "  Checkpoints:    ${CHECKPOINT_DIR}"
echo "  Rollout data:   ${ROLLOUT_DATA_DIR}"
echo ""

# ==============================================================================
# Start MedRAG Retrieval Server on GPU 3
# ==============================================================================
echo "Starting MedRAG retrieval server on GPU ${RETRIEVER_GPU}..."

CUDA_VISIBLE_DEVICES="${RETRIEVER_GPU}" "${PY}" \
  /projects/slmreasoning/junh/Debate/KARE/searchr1/medrag_retrieval_server.py \
  --port 8000 \
  --corpus_name MedCorp \
  --retriever_name MedCPT \
  --db_dir /projects/slmreasoning/junh/mirage_medrag/MedRAG/src/data/corpus \
  --topk 5 \
  > "${LOG_FILE%.log}_retriever.log" 2>&1 &

RETRIEVER_PID=$!
echo "Retrieval server started with PID ${RETRIEVER_PID}"
echo "Waiting 30 seconds for server to initialize..."
sleep 30

# Test if server is responding
if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
  echo "✓ Retrieval server is ready"
else
  echo "❌ WARNING: Retrieval server may not be ready"
fi

# ==============================================================================
# Run Training
# ==============================================================================
cd /projects/slmreasoning/junh/Search-R1

# Suppress Ray worker logs (optional)
export RAY_LOG_TO_STDERR=0

echo "Starting Search-R1 training at $(date)..."
echo ""

set +e
PYTHONUNBUFFERED=1 "${PY}" -m verl.trainer.main_ppo \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/val.parquet" \
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
  actor_rollout_ref.model.path="${BASE_MODEL}" \
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
  actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.n_agent=1 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.no_think_rl=false \
  critic.model.path="${BASE_MODEL}" \
  critic.model.enable_gradient_checkpointing=true \
  critic.model.fsdp_config.param_offload=true \
  critic.model.fsdp_config.grad_offload=true \
  critic.model.fsdp_config.optimizer_offload=true \
  critic.ppo_micro_batch_size=8 \
  trainer.logger='["console","wandb"]' \
  +trainer.val_only=false \
  +trainer.val_before_train=false \
  trainer.n_gpus_per_node="${NUM_TRAIN_GPUS}" \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.test_freq=25 \
  trainer.project_name="searchr1-kare-mortality" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.total_epochs=5 \
  trainer.total_training_steps=null \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="${CHECKPOINT_DIR}" \
  +trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
  max_turns=2 \
  retriever.url=http://127.0.0.1:8000/retrieve \
  retriever.topk=5

EXIT_CODE=$?
set -e

# ==============================================================================
# Cleanup
# ==============================================================================
# Stop the retrieval server
if [[ -n "${RETRIEVER_PID}" ]] && kill -0 "${RETRIEVER_PID}" 2>/dev/null; then
  echo "Stopping retrieval server (PID ${RETRIEVER_PID})..."
  kill "${RETRIEVER_PID}" || true
  wait "${RETRIEVER_PID}" 2>/dev/null || true
fi

echo ""
echo "=========================================="
if [[ ${EXIT_CODE} -eq 0 ]]; then
  echo "✓ Training completed successfully!"
else
  echo "❌ Training failed with exit code ${EXIT_CODE}"
fi
echo "=========================================="
echo "Finished at: $(date)"
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo "Rollout dir: ${ROLLOUT_DATA_DIR}"
echo "Ray tmpdir:  ${RAY_TMPDIR}"
echo "=========================================="

# Cleanup Ray temp directory
rm -rf "${RAY_TMPDIR}"

exit "${EXIT_CODE}"
