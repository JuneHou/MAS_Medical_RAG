#!/bin/bash
#
# Shell script to run KARE single-agent experiments
# Tests Qwen2.5-7B-Instruct with CoT and RAG modes
#

# Default parameters
MODEL="Qwen/Qwen2.5-7B-Instruct"
COT_GPU="6"  # Single GPU for CoT
RAG_GPUS="6,7"  # Parallel GPUs for RAG (retriever + LLM)
NUM_SAMPLES=100  # Set to smaller number for testing, remove for full dataset

# Navigate to KARE directory
cd /data/wang/junh/githubs/Debate/KARE

echo "=================================================="
echo "KARE Single-Agent Experiments"
echo "Model: $MODEL"
echo "GPUs: $GPUS"
echo "=================================================="

# Experiment 1: Single-Agent CoT
echo ""
echo "Running Experiment 1: Single-Agent Chain-of-Thought"
echo "--------------------------------------------------"
python run_kare_single_agent_experiments.py \
    --model "$MODEL" \
    --gpus "$GPUS" \
    --mode cot \
    --num_samples $NUM_SAMPLES \
    --batch_size 10

# Experiment 2: Single-Agent RAG (MedCPT + MedCorp2)
echo ""
echo "Running Experiment 2: Single-Agent RAG (MedCPT + MedCorp2)"
echo "--------------------------------------------------"
python run_kare_single_agent_experiments.py \
    --model "$MODEL" \
    --gpus "$GPUS" \
    --mode rag \
    --corpus_name MedCorp2 \
    --retriever_name MedCPT \
    --num_samples $NUM_SAMPLES \
    --batch_size 10

echo ""
echo "=================================================="
echo "All experiments completed!"
echo "Results saved in: results/"
echo "=================================================="
