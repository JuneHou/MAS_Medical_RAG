# effGen-based Single-Agent System for KARE Mortality Prediction

This document describes the effGen implementation of the KARE single-agent mortality prediction system, enabling comparison with the VLLM-based version.

## Overview

The single-agent system provides a **simpler baseline** compared to the multi-agent debate system:
- **1 Agent**: Direct mortality prediction (vs 3 agents with debate)
- **Simple Output**: 1/0 prediction (vs mortality/survival probabilities)
- **KARE-Style Prompting**: Task-based format following KARE conventions
- **Two In-Context Modes**: Zero-shot and few-shot

## System Configurations

The system supports **4 configurations** (2 modes × 2 in-context settings):

| Mode | In-Context | Description | Similar Patients Used? |
|------|-----------|-------------|----------------------|
| CoT | zero-shot | Pure reasoning | ❌ No |
| CoT | few-shot | Reasoning with examples | ✅ Yes |
| RAG | zero-shot | MedRAG retrieval | ❌ No |
| RAG | few-shot | MedRAG + examples | ✅ Yes |

## Key Features

### Identical to VLLM Version

- ✅ **Same Model**: Qwen2.5-7B-Instruct (full precision)
- ✅ **Same Hyperparameters**: temp=0.5, max_tokens=32768, top_p=0.9, rep_penalty=1.2
- ✅ **Same Prompts**: KARE task-based format
- ✅ **Same Data**: KARE test set
- ✅ **Same Output**: results.json + logs/

### KARE-Style Prompting

The system uses KARE's structured format:

```
# Task #
[Mortality prediction task description]

# Patient Context # (or # Patient EHR Context #)
[Patient clinical data]

# Similar Patients # (few-shot only)
Similar Patients Who Died:
[Positive examples]

Similar Patients Who Survived:
[Negative examples]

# Reasoning #
[Agent reasoning]

# Prediction #
[1/0]
```

## Installation

See main `README.md` for effGen installation.

## Usage

### Quick Test (5 samples)

```bash
cd /data/wang/junh/githubs/Debate/KARE/effgen

# CoT zero-shot
python run_kare_single_agent_effgen.py \
    --mode cot \
    --in_context zero-shot \
    --model Qwen/Qwen2.5-7B-Instruct \
    --gpus 0 \
    --num_samples 5

# CoT few-shot
python run_kare_single_agent_effgen.py \
    --mode cot \
    --in_context few-shot \
    --model Qwen/Qwen2.5-7B-Instruct \
    --gpus 0 \
    --num_samples 5

# RAG zero-shot
python run_kare_single_agent_effgen.py \
    --mode rag \
    --in_context zero-shot \
    --model Qwen/Qwen2.5-7B-Instruct \
    --gpus 0,1 \
    --num_samples 5

# RAG few-shot
python run_kare_single_agent_effgen.py \
    --mode rag \
    --in_context few-shot \
    --model Qwen/Qwen2.5-7B-Instruct \
    --gpus 0,1 \
    --num_samples 5
```

### Full Evaluation

```bash
# CoT zero-shot (entire test set)
python run_kare_single_agent_effgen.py \
    --mode cot \
    --in_context zero-shot \
    --model Qwen/Qwen2.5-7B-Instruct \
    --gpus 0

# CoT few-shot (entire test set)
python run_kare_single_agent_effgen.py \
    --mode cot \
    --in_context few-shot \
    --model Qwen/Qwen2.5-7B-Instruct \
    --gpus 0

# RAG zero-shot (entire test set)
python run_kare_single_agent_effgen.py \
    --mode rag \
    --in_context zero-shot \
    --model Qwen/Qwen2.5-7B-Instruct \
    --gpus 0,1

# RAG few-shot (entire test set)
python run_kare_single_agent_effgen.py \
    --mode rag \
    --in_context few-shot \
    --model Qwen/Qwen2.5-7B-Instruct \
    --gpus 0,1
```

## Output Structure

```
effgen/results/
├── single_effgen_cot_Qwen_Qwen2.5_7B_Instruct_zero_shot/
│   ├── results.json
│   └── debate_logs_zero_shot/
│       └── single_agent_cot_*.log
├── single_effgen_cot_Qwen_Qwen2.5_7B_Instruct_few_shot/
│   ├── results.json
│   └── debate_logs_few_shot/
│       └── single_agent_cot_*.log
├── single_effgen_rag_Qwen_Qwen2.5_7B_Instruct_MedCPT_zero_shot/
│   ├── results.json
│   └── debate_logs_zero_shot/
│       ├── single_agent_rag_*.log
│       └── retrieve_*.json
└── single_effgen_rag_Qwen_Qwen2.5_7B_Instruct_MedCPT_few_shot/
    ├── results.json
    └── debate_logs_few_shot/
        ├── single_agent_rag_*.log
        └── retrieve_*.json
```

## Output Format

### results.json

```json
{
  "metadata": {
    "timestamp": "2026-02-04 15:00:00",
    "mode": "cot",
    "in_context": "zero-shot",
    "framework": "effgen",
    "total_samples": 1500
  },
  "metrics": {
    "accuracy": 0.834,
    "precision": 0.689,
    "recall": 0.645,
    "f1_score": 0.666,
    "macro_f1": 0.821,
    "fallback_predictions": 12,
    "fallback_patient_ids": ["10188_1", ...]
  },
  "results": [
    {
      "patient_id": "10188_1",
      "visit_id": "visit_001",
      "ground_truth": 0,
      "prediction": 0,
      "is_fallback": false,
      "total_generation_time": 3.45
    }
  ]
}
```

## Hyperparameters

All configurations use **identical hyperparameters**:

```python
TEMPERATURE = 0.5
MAX_TOKENS = 32768
TOP_P = 0.9
REPETITION_PENALTY = 1.2
STOP = ["<|im_end|>", "</s>"]
```

## Comparison: Zero-Shot vs Few-Shot

| Aspect | Zero-Shot | Few-Shot |
|--------|-----------|----------|
| Similar Patients | Not included | Included in prompt |
| Prompt Length | Shorter (~500-1000 tokens) | Longer (~2000-4000 tokens) |
| Context | Target patient only | Target + positive/negative examples |
| Expected Performance | Lower (less context) | Higher (more examples) |
| Speed | Faster (shorter prompt) | Slower (longer prompt) |

## Comparison: CoT vs RAG

| Aspect | CoT | RAG |
|--------|-----|-----|
| Retrieval | None | MedRAG from MedCorp2 |
| GPU Usage | Single GPU | Multi-GPU (tensor parallelism) |
| Steps | 1 (direct prediction) | 1 (effGen handles retrieval internally) |
| External Knowledge | None | Medical literature + UMLS |
| Expected Performance | Lower (no evidence) | Higher (with evidence) |
| Speed | Faster | Slower (retrieval overhead) |

## Comparison: Single-Agent vs Multi-Agent Debate

| Aspect | Single-Agent | Multi-Agent Debate |
|--------|--------------|-------------------|
| Agents | 1 | 3 |
| Rounds | 1 | 2 |
| Output | 1/0 prediction | Mortality/survival probabilities |
| Reasoning | Direct | Contrastive + synthesis |
| Complexity | Low | High |
| Expected Performance | Lower | Higher |

## Testing

### Unit Tests

```bash
# Test CoT
python mortality_single_agent_effgen_cot.py

# Test RAG
python mortality_single_agent_effgen_rag.py
```

### Integration Tests

```bash
# Test all 4 configurations (5 samples each)
for mode in cot rag; do
    for context in zero-shot few-shot; do
        echo "Testing $mode $context..."
        python run_kare_single_agent_effgen.py \
            --mode $mode \
            --in_context $context \
            --num_samples 5 \
            --gpus 0
    done
done
```

## Troubleshooting

### Temperature Setting

If you need to adjust temperature:
- Edit `AgentConfig(temperature=0.5)` in the implementation files
- Default is 0.5 (matches VLLM version)

### In-Context Mode

- **Zero-shot**: Faster, less context, may have lower accuracy
- **Few-shot**: Slower, more context, typically better accuracy
- Choose based on your speed/accuracy tradeoff

### Fallback Predictions

If fallback rate is high (>10%):
1. Check model is generating proper KARE format
2. Verify "# Prediction #" appears in responses
3. Review log files to see response format
4. Consider adjusting extraction patterns

## Performance Expectations

Based on KARE benchmarks:

| Configuration | Expected Accuracy Range |
|--------------|------------------------|
| CoT zero-shot | 75-80% |
| CoT few-shot | 78-83% |
| RAG zero-shot | 80-85% |
| RAG few-shot | 82-87% |

Multi-agent debate typically adds +2-5% over single-agent few-shot.

## Comparison Commands

```bash
# Compare effGen vs VLLM for CoT zero-shot
python ../compare_results.py \
    --vllm ../results/single_cot_mor_*_zero_shot/results.json \
    --effgen ./results/single_effgen_cot_*_zero_shot/results.json

# Compare all 4 configurations
for mode in cot rag; do
    for context in zero_shot few_shot; do
        echo "Comparing $mode $context..."
        python ../compare_results.py \
            --vllm ../results/single_${mode}_*_${context}/results.json \
            --effgen ./results/single_effgen_${mode}_*_${context}/results.json
    done
done
```

## License

MIT License (same as effGen and KARE project)
