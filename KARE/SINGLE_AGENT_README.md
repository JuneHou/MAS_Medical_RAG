# KARE Single-Agent Experiments

This directory contains single-agent implementations for KARE mortality prediction, designed to test Qwen2.5-7B-Instruct performance following the KARE zero_shot_base setting.

## Overview

These implementations are minimal modifications of the multi-agent debate system, removing the debate mechanism and keeping only a single agent with KARE-style prompting.

## Files

### Core Implementations
- `mortality_single_agent_cot.py` - Single agent with Chain-of-Thought reasoning only
- `mortality_single_agent_rag.py` - Single agent with MedRAG retrieval
- `run_kare_single_agent_experiments.py` - Runner script for both modes
- `run_single_agent_experiments.sh` - Shell script to run all experiments

### Changes from Multi-Agent Debate

**Minimal changes made:**
1. Removed agents 1-3 (mortality_risk_assessor, protective_factor_analyst, target_patient_analyst)
2. Kept only the integrator agent, renamed as single agent
3. Modified system prompt to KARE zero_shot_base style (no debate history references)
4. For RAG mode: Kept MedRAG retrieval mechanism
5. For CoT mode: Removed all RAG components

## Usage

### Quick Start

```bash
# Run all experiments
bash run_single_agent_experiments.sh
```

### Individual Experiments

**Chain-of-Thought (CoT) mode (single GPU):**
```bash
python run_kare_single_agent_experiments.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --gpus 5 \
    --mode cot \
    --num_samples 100
```

**RAG mode with MedCPT + MedCorp2 (parallel GPUs for retriever + LLM):**
```bash
python run_kare_single_agent_experiments.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --gpus 5,6 \
    --mode rag \
    --corpus_name MedCorp2 \
    --retriever_name MedCPT \
    --num_samples 100
```

### Command-Line Arguments

- `--model`: HuggingFace model name (default: Qwen/Qwen2.5-7B-Instruct)
- `--gpus`: GPU IDs to use. Single GPU for CoT (e.g., `5`), comma-separated for RAG (e.g., `5,6` for retriever+LLM)
- `--mode`: Experiment mode - `cot` or `rag` (required)
- `--num_samples`: Number of samples to evaluate (default: all)
- `--start_idx`: Starting sample index (default: 0)
- `--batch_size`: Batch size for intermediate saves (default: 10)
- `--output`: Custom output path (auto-generated if not specified)

**RAG-specific arguments:**
- `--corpus_name`: MedRAG corpus name (default: MedCorp2)
- `--retriever_name`: MedRAG retriever name (default: MedCPT)
- `--db_dir`: MedRAG database directory

## Output Structure

Results are automatically saved to structured directories:

```
results/
├── single_cot_mor_Qwen_Qwen2.5_7B_Instruct/
│   ├── results.json
│   └── debate_logs/
│       └── single_agent_cot_*.log
└── single_rag_mor_Qwen_Qwen2.5_7B_Instruct_MedCPT/
    ├── results.json
    └── debate_logs/
        └── single_agent_rag_*.log
```

### Results Format

```json
{
  "metadata": {
    "timestamp": "2025-01-01 12:00:00",
    "mode": "cot",
    "total_samples": 100
  },
  "metrics": {
    "accuracy": 0.850,
    "precision": 0.750,
    "recall": 0.600,
    "f1_score": 0.667,
    "specificity": 0.900
  },
  "results": [
    {
      "patient_id": "patient_001",
      "ground_truth": 0,
      "prediction": 0,
      "mortality_probability": 0.25,
      "survival_probability": 0.75,
      "total_generation_time": 2.5
    }
  ]
}
```

## System Prompts

### CoT Mode
Direct adaptation of KARE zero_shot_base prompt for Chain-of-Thought reasoning without retrieval.

### RAG Mode
Same as CoT but includes `retrieve(query)` tool for medical evidence retrieval.

## Comparison with KARE

**Similarities:**
- Uses same dataset (KARE mortality prediction)
- Uses same input format (target patient + similar patients)
- Uses same zero_shot_base prompting style
- Binary classification task (0=survive, 1=mortality)

**Differences:**
- Implementation: VLLM-based (KARE uses Claude API)
- Model: Qwen2.5-7B-Instruct (KARE uses Claude 3.5 Sonnet)
- RAG: Uses MedRAG (KARE has custom retrieval)

## Performance Monitoring

The system tracks:
- Prediction accuracy, precision, recall, F1, specificity
- Confusion matrix (TP, FP, TN, FN)
- Generation time per sample
- Prediction and ground truth distributions

## Debugging

Log files contain:
- Input prompts with patient context
- Retrieved medical evidence (RAG mode only)
- Agent's full response
- Extracted probabilities and predictions

## Requirements

- Python 3.8+
- VLLM
- MedRAG (for RAG mode only)
- KARE data adapter
- CUDA-compatible GPUs:
  - **CoT mode**: 1 GPU (e.g., `--gpus 6`)
  - **RAG mode**: 2 GPUs recommended (e.g., `--gpus 5,6`) for parallel retrieval + generation

## Notes

1. **Mortality is rare:** The system is configured to require strong evidence for mortality predictions (conservative threshold)
2. **Probabilities must sum to 1.0:** The prompt explicitly requires mortality_prob + survival_prob = 1.0
3. **Resume capability:** The system can resume from partial results if interrupted
4. **Batch saves:** Results are saved every N samples to prevent data loss
