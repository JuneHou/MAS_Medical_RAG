#!/bin/bash

# Convert VERL checkpoint to HuggingFace format
# Usage: ./convert_checkpoint.sh [checkpoint_step]

# Default to latest checkpoint if not specified
CHECKPOINT_STEP=${1:-150}

CHECKPOINT_DIR="/data/wang/junh/githubs/Debate/KARE/verl/checkpoints/prediction/global_step_${CHECKPOINT_STEP}/actor"
OUTPUT_DIR="/data/wang/junh/githubs/Debate/KARE/verl/models/prediction_brier_unlabel_7b_step${CHECKPOINT_STEP}"

echo "Converting checkpoint at step ${CHECKPOINT_STEP}..."
echo "  Source: ${CHECKPOINT_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo ""

python /data/wang/junh/githubs/Debate/KARE/verl/convert_fsdp_to_hf.py \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "================================================"
echo "Conversion complete!"
echo "================================================"
echo ""
echo "To use this model as integrator in your debate system:"
echo ""
echo "  python run_kare_debate_mortality.py \\"
echo "    --model Qwen/Qwen2.5-7B-Instruct \\"
echo "    --integrator_model ${OUTPUT_DIR} \\"
echo "    --gpus 3,4,5,6 \\"
echo "    --integrator_gpu 4 \\"
echo "    --mode rag"
echo ""
