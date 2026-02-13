# effGen RAG Implementation - Migration from vllm

This document describes the new effGen-based single-agent RAG implementation for KARE mortality prediction, migrated from the original vllm implementation with **exact prompt and hyperparameter parity**.

## Overview

The new implementation (`mortality_single_agent_effgen_rag.py`) maintains complete compatibility with the original vllm version while leveraging effGen's model loading and generation capabilities.

## Key Requirements Met

### 1. Exact Prompts ✓

All prompts are **byte-for-byte identical** to the original vllm implementation:

- `task_description`: Identical KARE mortality prediction task description
- `retrieval_instruction`: Identical retrieval instruction
- `retrieval_prompt`: Identical first-turn prompt (zero-shot and few-shot versions)
- `final_prompt`: Identical second-turn prompt with supplementary information

### 2. Exact Hyperparameters ✓

```python
temperature = 0.7
top_p = 0.9
max_new_tokens = 32768
repetition_penalty = 1.2
```

These are **hardcoded** in the generation calls to match the original exactly.

### 3. Exact Retrieval Logic ✓

The `MedRAGTool` class replicates the exact retrieval logic from the original:

- **MedCorp2 splitting**: Split k between medcorp and UMLS sources (k_medcorp = k // 2 + k % 2, k_umls = k // 2)
- **Query truncation**: Maximum 200 characters
- **Document formatting**: Identical format with document numbers and scores
- **RRF parameters**: Uses rrf_k=60 as in original

### 4. Re-prompting Strategy ✓

**Critical difference from standard effGen patterns:**

The original vllm code uses a **two-turn manual orchestration** where:

1. **First turn**: Agent receives `retrieval_prompt` and should request retrieval
2. **Parse tool call**: Manually extract retrieve() call from response
3. **Execute retrieval**: Call MedRAG tool directly
4. **Second turn**: Agent receives **fresh `final_prompt`** with retrieved docs (NOT appended to chat history)

This is **NOT** effGen's default ReAct loop. The implementation manually orchestrates these steps using direct model.generate() calls.

### 5. Exact Prediction Parsing ✓

The `_extract_prediction_and_probabilities` method uses the **exact same regex patterns** as the original:

```python
prediction_patterns = [
    r'#\s*Prediction\s*#[\s:]*([01])',
    r'Prediction[:\s]+([01])',
    r'\*\*Prediction\*\*[:\s]+([01])',
    r'prediction[:\s]+([01])',
]
```

Fallback logic is also identical: opposite of ground truth, or 0 if no ground truth.

## Architecture Comparison

### Original vllm Implementation

```
Initialize:
  - MedRAG (retriever on cuda:0)
  - VLLMWrapper (LLM with tensor parallelism)

Prediction Flow:
  1. Format retrieval_prompt
  2. Generate with vllm.generate(sampling_params)
  3. Parse retrieve() call from response
  4. Execute retrieval via MedRAG
  5. Format final_prompt with retrieved docs
  6. Generate with vllm.generate(sampling_params)
  7. Extract prediction with regex
```

### New effGen Implementation

```
Initialize:
  - MedRAG (retriever on cuda:0)
  - load_model() via effGen (model on cuda:0)

Prediction Flow:
  1. Format retrieval_prompt (EXACT same)
  2. Generate with model.generate(exact hyperparameters)
  3. Parse retrieve() call from response (EXACT same regex)
  4. Execute retrieval via MedRAGTool (EXACT same logic)
  5. Format final_prompt with retrieved docs (EXACT same)
  6. Generate with model.generate(exact hyperparameters)
  7. Extract prediction with regex (EXACT same patterns)
```

## Implementation Details

### GPU Usage

**Original vllm:**
- Retriever: cuda:0
- LLM: Tensor parallelism across all GPUs (e.g., cuda:0 and cuda:1)

**New effGen:**
- Retriever: cuda:0
- LLM: cuda:0 (single GPU to avoid tensor parallelism complexity)

**Rationale:** effGen's load_model() is simpler with single GPU. If multi-GPU is needed, use the original vllm version or implement tensor parallelism separately.

### MedRAG Tool

The `MedRAGTool` class wraps MedRAG retrieval with:

```python
class MedRAGTool(BaseTool):
    def __init__(self, medrag_instance, k=8):
        self.name = "retrieve"
        self.medrag = medrag_instance
        self.k = k
    
    def execute(self, query: str) -> str:
        # EXACT logic from original _create_retrieval_tool
        # Including MedCorp2 source splitting
        # Returns formatted docs string
```

### Manual Generation

The implementation uses **direct transformers generation** instead of effGen's Agent.run():

```python
# Format prompt
if "qwen" in self.model_name.lower():
    formatted_prompt = f"<|im_start|>user\n{retrieval_prompt}<|im_end|>\n<|im_start|>assistant\n"

# Tokenize
tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)

# Generate with EXACT parameters
with torch.no_grad():
    outputs = self.model.generate(
        **inputs,
        temperature=0.7,  # EXACT
        top_p=0.9,  # EXACT
        max_new_tokens=32768,  # EXACT
        repetition_penalty=1.2,  # EXACT
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
    )

# Decode
response_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
```

This ensures **exact hyperparameter control** rather than relying on effGen's AgentConfig.

## Usage

### Basic Usage

```python
from mortality_single_agent_effgen_rag import MortilitySingleAgentEffGenRAG

# Initialize system
system = MortilitySingleAgentEffGenRAG(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    gpu_ids="6,7",  # Will use first GPU only
    corpus_name="MedCorp2",
    retriever_name="MedCPT",
    db_dir="/path/to/medrag/corpus",
    in_context="zero-shot"  # or "few-shot"
)

# Make prediction
result = system.predict_mortality(
    patient_context=patient_context,
    positive_similars=positive_similars,
    negative_similars=negative_similars,
    patient_id="12345",
    output_dir="./results",
    ground_truth=1
)

print(f"Prediction: {result['final_prediction']}")
print(f"Is Fallback: {result['is_fallback']}")
print(f"Time: {result['total_generation_time']:.2f}s")
```

### With Runner Script

```bash
# Zero-shot RAG mode
python run_kare_single_agent_effgen.py \
    --mode rag \
    --in_context zero-shot \
    --gpus 6,7 \
    --num_samples 100 \
    --corpus_name MedCorp2 \
    --retriever_name MedCPT

# Few-shot RAG mode
python run_kare_single_agent_effgen.py \
    --mode rag \
    --in_context few-shot \
    --gpus 6,7 \
    --num_samples 100 \
    --corpus_name MedCorp2 \
    --retriever_name MedCPT
```

## Differences from Old Implementation

The old implementation in `agentic_retrieve/mortality_single_agent_effgen_rag.py` had several issues:

### Problem 1: Wrong Hyperparameters ❌

```python
# Old implementation
temperature=0.5  # Should be 0.7
# Missing top_p, max_new_tokens, repetition_penalty control
```

### Problem 2: Used effGen's ReAct Loop ❌

The old implementation used:
```python
self.agent = Agent(
    config=AgentConfig(
        tools=[self.retrieval_tool],
        system_prompt=system_prompt,
        max_iterations=5
    )
)
result = self.agent.run(prompt)  # ReAct loop
```

This uses effGen's automatic tool calling, which:
- Maintains chat history across turns (not fresh prompt)
- May not parse tool calls the same way
- Doesn't guarantee exact hyperparameters

### Problem 3: No Explicit Re-prompting ❌

The old implementation didn't construct separate `retrieval_prompt` and `final_prompt`. It relied on effGen's ReAct loop to handle everything, which doesn't match the original vllm pattern.

### New Implementation Fixes ✓

1. **Exact hyperparameters** via direct model.generate() calls
2. **Manual orchestration** of two-turn process
3. **Explicit re-prompting** with fresh final_prompt
4. **Identical parsing** using original regex patterns

## Testing

Test the implementation:

```bash
cd /data/wang/junh/githubs/Debate/KARE/effgen
python mortality_single_agent_effgen_rag.py
```

Expected output:
```
Testing Single-Agent effGen RAG System...
Set CUDA_VISIBLE_DEVICES=6
In-context mode: zero-shot
Initializing MedRAG with MedCorp2 corpus and MedCPT retriever...
[MedRAG Tool] Initialized with k=8
Loading model with effGen...
[AGENT] Generating initial response...
[RETRIEVE] Query: ...
[AGENT] Generating final response with evidence...
Prediction: 1
Time: XX.XXs
```

## File Structure

```
KARE/effgen/
├── mortality_single_agent_effgen_rag.py  # NEW implementation (this file)
├── mortality_single_agent_effgen_cot.py  # CoT-only version
├── run_kare_single_agent_effgen.py       # Runner script
├── EFFGEN_RAG_MIGRATION.md               # This documentation
└── agentic_retrieve/
    └── mortality_single_agent_effgen_rag.py  # OLD implementation (deprecated)
```

## Validation Checklist

- [x] Exact task_description string
- [x] Exact retrieval_instruction string
- [x] Exact retrieval_prompt format (zero-shot and few-shot)
- [x] Exact final_prompt format (zero-shot and few-shot)
- [x] temperature=0.7
- [x] top_p=0.9
- [x] max_new_tokens=32768
- [x] repetition_penalty=1.2
- [x] MedCorp2 source splitting logic
- [x] Query truncation to 200 chars
- [x] RRF k=60
- [x] Document formatting with scores
- [x] Tool call parsing regex patterns
- [x] Prediction extraction regex patterns
- [x] Fallback logic
- [x] Two-turn manual orchestration
- [x] Fresh final_prompt (not chat history)

## Known Limitations

1. **Single GPU only**: Uses first GPU for both retriever and model (not tensor parallelism)
2. **Memory requirements**: Full model on single GPU requires sufficient VRAM
3. **Performance**: May be slower than vllm's optimized inference

## Future Improvements

1. Add tensor parallelism support if needed
2. Optimize memory usage with quantization (if exact precision not required)
3. Add batch processing for multiple patients
4. Add vllm backend option for faster inference

## Contact

For issues or questions about this implementation, refer to:
- Original vllm implementation: `KARE/mortality_single_agent_rag.py`
- This implementation: `KARE/effgen/mortality_single_agent_effgen_rag.py`
- Runner script: `KARE/effgen/run_kare_single_agent_effgen.py`
