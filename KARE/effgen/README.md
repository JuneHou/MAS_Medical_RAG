# effGen-based Multi-Agent Debate System for KARE Mortality Prediction

This directory contains the effGen implementation of the KARE multi-agent mortality prediction system, enabling direct comparison with the VLLM-based implementation.

## Overview

The effGen implementation maintains **identical functionality** to the VLLM version while using the effGen framework for agent execution. This allows for fair performance comparison between the two frameworks.

### Key Features

- **Same Model**: Qwen2.5-7B-Instruct (full precision, no quantization)
- **Same Architecture**: 3 specialized agents (mortality risk assessor, protective factor analyst, balanced clinical integrator)
- **Same Debate Structure**: 2-round debate (similar patient analysis → integration & consensus)
- **Same Hyperparameters**: Temperature, max_tokens, top_p, repetition_penalty all matched
- **Same Data**: KARE test set with identical preprocessing

### Two Modes

1. **CoT Mode** (`mortality_debate_effgen_cot.py`): Pure chain-of-thought reasoning without external retrieval
2. **RAG Mode** (`mortality_debate_effgen_rag.py`): Enhanced with MedRAG retrieval from MedCorp2 corpus

## Installation

### Prerequisites

```bash
# Install effgen
pip install effgen

# Or with vLLM support for faster inference
pip install effgen[vllm]

# Install from source (if needed)
cd /path/to/effGen
pip install -e .
```

### Dependencies

All dependencies from the parent KARE directory are required:
- KARE data adapter
- MedRAG (for RAG mode)
- Standard libraries (json, time, logging, etc.)

## File Structure

```
effgen/
├── README.md                           # This file
├── IMPLEMENTATION_PLAN.md              # Detailed implementation plan
├── mortality_debate_effgen_cot.py      # CoT mode implementation
├── mortality_debate_effgen_rag.py      # RAG mode implementation
├── effgen_medrag_tool.py               # Custom MedRAG tool for effgen
├── run_kare_debate_mortality_effgen.py # Main runner script
└── results/                            # Output directory
    ├── effgen_cot_MODEL_NAME/          # CoT results
    │   ├── results.json                # Predictions and metrics
    │   └── logs/                       # Per-patient debate logs
    └── effgen_rag_MODEL_NAME/          # RAG results
        ├── results.json                # Predictions and metrics
        └── logs/                       # Per-patient debate logs
```

## Usage

### Quick Start

```bash
# CoT Mode (5 samples for testing)
python run_kare_debate_mortality_effgen.py \
    --mode cot \
    --model Qwen/Qwen2.5-7B-Instruct \
    --gpus 0 \
    --num_samples 5

# RAG Mode (5 samples for testing)
python run_kare_debate_mortality_effgen.py \
    --mode rag \
    --model Qwen/Qwen2.5-7B-Instruct \
    --gpus 0 \
    --num_samples 5
```

### Full Evaluation

```bash
# CoT Mode (entire test set)
python run_kare_debate_mortality_effgen.py \
    --mode cot \
    --model Qwen/Qwen2.5-7B-Instruct \
    --gpus 0

# RAG Mode (entire test set)
python run_kare_debate_mortality_effgen.py \
    --mode rag \
    --model Qwen/Qwen2.5-7B-Instruct \
    --gpus 0 \
    --corpus_name MedCorp2 \
    --retriever_name MedCPT
```

### Advanced Options

```bash
python run_kare_debate_mortality_effgen.py \
    --mode rag \
    --model Qwen/Qwen2.5-7B-Instruct \
    --gpus 0,1 \
    --start_idx 0 \
    --num_samples 100 \
    --batch_size 10 \
    --include_history \
    --output ./custom_output/results.json \
    --corpus_name MedCorp2 \
    --retriever_name MedCPT \
    --db_dir /path/to/medrag/corpus
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | `cot` | Debate mode: `cot` or `rag` |
| `--model` | str | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace model name |
| `--gpus` | str | `0` | GPU IDs (comma-separated) |
| `--start_idx` | int | `0` | Starting sample index |
| `--num_samples` | int | `None` | Number of samples (None = all) |
| `--batch_size` | int | `10` | Batch size for intermediate saves |
| `--include_history` | flag | `False` | Include full debate history in output |
| `--output` | str | `None` | Output file path (auto-generated if None) |
| `--corpus_name` | str | `MedCorp2` | MedRAG corpus name (RAG mode only) |
| `--retriever_name` | str | `MedCPT` | MedRAG retriever name (RAG mode only) |
| `--db_dir` | str | `/path/to/corpus` | MedRAG database directory (RAG mode only) |

## Output Format

### Results File (`results.json`)

```json
{
  "metadata": {
    "timestamp": "2026-02-04 12:00:00",
    "total_samples": 1500,
    "include_debate_history": false,
    "framework": "effgen"
  },
  "metrics": {
    "accuracy": 0.856,
    "precision": 0.723,
    "recall": 0.689,
    "f1_score": 0.705,
    "macro_f1": 0.843,
    "specificity": 0.912,
    "total_samples": 1500,
    "tp": 245,
    "fp": 94,
    "fn": 111,
    "tn": 1050
  },
  "results": [
    {
      "patient_id": "10188_1",
      "base_patient_id": "10188",
      "visit_id": "visit_001",
      "visit_index": 1,
      "ground_truth": 0,
      "prediction": 0,
      "rounds_completed": 2,
      "total_generation_time": 12.34
    }
  ]
}
```

### Log Files (`logs/debate_responses_PATIENT_ID.log`)

Per-patient debate logs containing:
- Raw responses from each agent
- Extracted probabilities
- Intermediate reasoning steps
- Tool calls (RAG mode only)

## Comparison with VLLM Implementation

### Similarities

| Aspect | VLLM | effGen |
|--------|------|--------|
| Model | Qwen2.5-7B-Instruct | Qwen2.5-7B-Instruct |
| Quantization | None (full precision) | None (full precision) |
| Architecture | 3 agents, 2 rounds | 3 agents, 2 rounds |
| Prompts | Identical | Identical |
| Temperature | 0.3 / 0.5 | 0.3 / 0.5 |
| max_tokens | 32768 | 32768 |
| top_p | 0.9 | 0.9 |
| repetition_penalty | 1.15 / 1.2 | 1.15 / 1.2 |
| Data | KARE test set | KARE test set |
| Metrics | Accuracy, F1, etc. | Accuracy, F1, etc. |

### Differences

| Aspect | VLLM | effGen |
|--------|------|--------|
| Framework | VLLM (direct) | effGen |
| Agent Management | Manual | effGen AgentConfig |
| Tool Integration | Custom | effGen BaseTool |
| Code Complexity | Higher | Lower (abstracted) |

## Testing

### Unit Tests

```bash
# Test CoT mode
python mortality_debate_effgen_cot.py

# Test RAG mode
python mortality_debate_effgen_rag.py

# Test MedRAG tool
python effgen_medrag_tool.py
```

### Integration Test

```bash
# Run on a small sample
python run_kare_debate_mortality_effgen.py \
    --mode cot \
    --num_samples 5 \
    --gpus 0
```

## Troubleshooting

### CUDA Initialization Errors

If you encounter:
```
RuntimeError: Cannot re-initialize CUDA in forked subprocess
```

**Solution**: The code follows `MEDRAG_GPU_SETUP_FIX.md` - CUDA_VISIBLE_DEVICES is set once in `__init__`. If error persists, check for external CUDA environment variables.

### Memory Errors

If GPU runs out of memory:
1. Use fewer GPUs (single GPU mode)
2. Reduce batch size
3. Check for memory leaks from previous runs

### MedRAG Import Errors (RAG mode)

If MedRAG import fails:
1. Verify MedRAG path: `/data/wang/junh/githubs/mirage_medrag/MedRAG`
2. Check corpus data: `/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus`
3. Ensure MedCPT retriever is initialized

### Model Loading Errors

If model fails to load:
1. Check cache directory: `/data/wang/junh/.cache/huggingface/models--Qwen--Qwen2.5-7B-Instruct`
2. Verify model name: `Qwen/Qwen2.5-7B-Instruct`
3. Ensure sufficient disk space

## Performance Comparison

To compare effGen vs VLLM:

```bash
# Run both versions on same data
# VLLM CoT
python ../run_kare_debate_mortality.py --mode cot --num_samples 100

# effGen CoT
python run_kare_debate_mortality_effgen.py --mode cot --num_samples 100

# Compare results
python compare_results.py \
    --vllm ../results/cot_*/results.json \
    --effgen ./results/effgen_cot_*/results.json
```

## Citation

If you use this code, please cite both the KARE paper and effGen:

```bibtex
@software{effgen2026,
  title={effGen: Enabling Small Language Models as Capable Autonomous Agents},
  author={Srivastava, Gaurav and others},
  year={2026},
  url={https://github.com/ctrl-gaurav/effGen}
}
```

## Contact

For issues specific to:
- **effGen framework**: https://github.com/ctrl-gaurav/effGen/issues
- **KARE dataset**: See parent directory README
- **MedRAG integration**: Check `effgen_medrag_tool.py`

## License

MIT License (same as effGen and KARE project)
