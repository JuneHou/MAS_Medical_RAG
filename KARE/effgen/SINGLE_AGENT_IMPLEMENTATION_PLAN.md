# effGen Implementation Plan for KARE Single-Agent Mortality Prediction System

## Overview

This document outlines the plan to create an effGen-based implementation of the KARE single-agent mortality prediction system, enabling comparison with the VLLM-based version.

## Current VLLM System Analysis

### Architecture

**Single-Agent System** (vs multi-agent debate):
- **1 Agent**: Direct mortality prediction without debate
- **Simple Output**: Predicts 1/0 (mortality/survival) - no probability scores
- **KARE-Style Prompting**: Task-based format with "# Task #", "# Patient Context #", etc.

### Two Operation Modes

1. **CoT Mode** (`mortality_single_agent_cot.py`):
   - Pure chain-of-thought reasoning
   - No external retrieval
   - Single GPU (gpu_id parameter)
   - Uses VLLM LLM directly

2. **RAG Mode** (`mortality_single_agent_rag.py`):
   - Two-step process: retrieval request → final prediction
   - MedRAG retrieval from MedCorp2
   - Multi-GPU with tensor parallelism (gpu_ids parameter)
   - Uses VLLMWrapper

### In-Context Modes

Both CoT and RAG support two in-context modes:

1. **Zero-Shot**: No similar patients in prompt
   - Only target patient context
   - Simpler, faster

2. **Few-Shot**: With similar patients
   - Includes positive similars (died)
   - Includes negative similars (survived)
   - More contextual information

### Key Hyperparameters (IDENTICAL for both CoT and RAG)

```python
AGENT_PARAMS = {
    "temperature": 0.5,  # Changed from 0.7 to 0.5
    "max_tokens": 32768,
    "top_p": 0.9,
    "repetition_penalty": 1.2,
    "stop": ["<|im_end|>", "</s>"]
}

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_CACHE = "/data/wang/junh/.cache/huggingface/models--Qwen--Qwen2.5-7B-Instruct"
QUANTIZATION = None  # Full precision
```

### KARE-Style Prompt Format

**Zero-Shot CoT:**
```
Given the following task description and patient context, please make a prediction with reasoning...

# Task #
[Task description]
========================================

# Patient Context #
[Patient EHR data]
========================================

Give the prediction and reasoning in the following format:
# Reasoning #
[Your reasoning here]

# Prediction #
[Your prediction here (1/0)]

Output:
```

**Few-Shot CoT:**
```
Given the following task description, patient EHR context, similar patients, please make a prediction...

# Task #
[Task description]

# Patient EHR Context #
[Target patient]

# Similar Patients #
Similar Patients Who Died:
[Positive similars]

Similar Patients Who Survived:
[Negative similars]

Give the prediction and reasoning in the following format:
# Reasoning #
[Your reasoning here]

# Prediction #
[Your prediction here (1/0)]

Output:
```

**RAG Mode** (two-step):
- Step 1: Request retrieval with task + patient context
- Step 2: Final prediction with task + patient + supplementary information

### Prediction Extraction

Simple 1/0 format (no probabilities):
```python
# Patterns to match:
r'#\s*Prediction\s*#[\s:]*([01])'
r'Prediction[:\s]+([01])'
r'\*\*Prediction\*\*[:\s]+([01])'
r'prediction[:\s]+([01])'
```

Fallback: If no prediction found, use opposite of ground truth (or 0 if no GT)

## effGen Implementation Strategy

### File Structure

```
effgen/
├── mortality_single_agent_effgen_cot.py    # CoT mode
├── mortality_single_agent_effgen_rag.py    # RAG mode  
├── run_kare_single_agent_effgen.py         # Runner script
├── SINGLE_AGENT_IMPLEMENTATION_PLAN.md     # This file
└── SINGLE_AGENT_README.md                  # Usage docs
```

### Phase 1: effGen CoT Mode

**File**: `mortality_single_agent_effgen_cot.py`

**Class**: `MortilitySingleAgentEffGenCoT`

**Key Features**:
1. Single effGen agent with KARE-style prompts
2. Supports both zero-shot and few-shot modes
3. Simple 1/0 prediction extraction
4. Identical hyperparameters to VLLM
5. Single GPU allocation

**Implementation Details**:
```python
class MortilitySingleAgentEffGenCoT:
    def __init__(self, 
                 model_name: str,
                 gpu_id: str,
                 in_context: str = "zero-shot"):
        # Load model with effgen (no quantization)
        self.model = load_model(model_name)
        
        # Create agent with KARE-style system prompt
        self.agent = Agent(
            config=AgentConfig(
                name="mortality_predictor",
                model=self.model,
                tools=[],  # No tools for CoT
                system_prompt="",  # No system prompt (KARE uses task in user prompt)
                max_iterations=1,
                temperature=0.5,
                enable_thinking=True
            )
        )
    
    def predict_mortality(self, 
                         patient_context: str,
                         positive_similars: str,
                         negative_similars: str,
                         ...):
        # Build KARE-style prompt based on in_context mode
        if self.in_context == "zero-shot":
            prompt = build_zero_shot_prompt(patient_context)
        else:
            prompt = build_few_shot_prompt(patient_context, positive_similars, negative_similars)
        
        # Run agent
        result = self.agent.run(prompt)
        
        # Extract 1/0 prediction
        prediction = extract_prediction(result.output)
        
        return {'final_prediction': prediction, ...}
```

### Phase 2: effGen RAG Mode

**File**: `mortality_single_agent_effgen_rag.py`

**Class**: `MortilitySingleAgentEffGenRAG`

**Key Features**:
1. Single effGen agent with MedRAG tool
2. Two-step process (like VLLM):
   - Step 1: Agent requests retrieval
   - Step 2: Agent makes prediction with supplementary info
3. Supports zero-shot and few-shot modes
4. Reuses `MedRAGRetrievalTool` from debate system

**Implementation Details**:
```python
class MortilitySingleAgentEffGenRAG:
    def __init__(self,
                 model_name: str,
                 gpu_ids: str,
                 in_context: str = "zero-shot",
                 ...):
        # Initialize MedRAG first
        self.medrag = MedRAG(...)
        
        # Load model
        self.model = load_model(model_name)
        
        # Create retrieval tool
        self.retrieval_tool = MedRAGRetrievalTool(self.medrag, k=8)
        
        # Create agent WITH tool
        self.agent = Agent(
            config=AgentConfig(
                name="mortality_predictor_rag",
                model=self.model,
                tools=[self.retrieval_tool],  # Include retrieval tool
                system_prompt="",
                max_iterations=3,  # Allow for retrieval + reasoning
                temperature=0.5
            )
        )
    
    def predict_mortality(self, ...):
        # Step 1: Initial prompt requesting retrieval
        retrieval_prompt = build_retrieval_request_prompt(...)
        result = self.agent.run(retrieval_prompt)
        
        # Step 2: Parse tool call and execute retrieval
        # (effgen should handle this automatically via tool)
        
        # Step 3: Final prediction with supplementary info
        # (continuation of agent run or new call with retrieved docs)
        
        return {'final_prediction': prediction, ...}
```

**Challenges**:
1. effGen agent tool integration may differ from manual two-step process
2. May need to adapt to effGen's tool-calling mechanism
3. Query truncation (200 chars in VLLM RAG) needs implementation

### Phase 3: Runner Script

**File**: `run_kare_single_agent_effgen.py`

**Features**:
- Command-line interface matching VLLM version
- Mode selection: `--mode cot/rag`
- In-context mode: `--in_context zero-shot/few-shot`
- GPU allocation
- Auto-generated output paths
- Metrics calculation (reuse from multi-agent)
- Results format: `results.json` + `logs/` subdirectory

**Output Structure**:
```
effgen/results/
├── single_effgen_cot_MODEL_zero_shot/
│   ├── results.json
│   └── debate_logs_zero_shot/
│       └── single_agent_cot_*.log
├── single_effgen_cot_MODEL_few_shot/
│   ├── results.json
│   └── debate_logs_few_shot/
├── single_effgen_rag_MODEL_zero_shot/
│   ├── results.json
│   └── debate_logs_zero_shot/
└── single_effgen_rag_MODEL_few_shot/
    ├── results.json
    └── debate_logs_few_shot/
```

## Key Differences from Multi-Agent System

| Aspect | Multi-Agent Debate | Single-Agent |
|--------|-------------------|--------------|
| Agents | 3 (risk, protective, integrator) | 1 (predictor) |
| Rounds | 2 (similar analysis → integration) | 1 (direct prediction) |
| Output | Mortality/survival probabilities | 1/0 prediction |
| Prompts | Clinical assistant style | KARE task-based style |
| In-Context | Always uses similar patients | Optional (zero-shot/few-shot) |
| Complexity | Higher (debate flow) | Lower (single call) |

## Hyperparameter Mapping

### VLLM → effGen (Single-Agent)

| VLLM Parameter | Value | effGen Equivalent |
|----------------|-------|-------------------|
| `temperature` | 0.5 | `temperature=0.5` in AgentConfig |
| `max_tokens` | 32768 | Set via sampling params or config |
| `top_p` | 0.9 | `top_p=0.9` in sampling params |
| `repetition_penalty` | 1.2 | `repetition_penalty=1.2` |
| `stop` | `["<|im_end|>", "</s>"]` | `stop` in sampling params |

### GPU Allocation

**CoT Mode**:
- VLLM: Single GPU (`gpu_id="6"`)
- effGen: Set `CUDA_VISIBLE_DEVICES` once, single GPU

**RAG Mode**:
- VLLM: Multi-GPU with tensor parallelism (`gpu_ids="6,7"`)
- effGen: Set `CUDA_VISIBLE_DEVICES` to all GPUs, retriever uses cuda:0

## MedRAG Integration (RAG Mode Only)

### Reuse from Multi-Agent System

```python
# Reuse MedRAGRetrievalTool from effgen_medrag_tool.py
from effgen_medrag_tool import MedRAGRetrievalTool

# Create tool with query truncation (200 chars vs 2048 tokens)
retrieval_tool = MedRAGRetrievalTool(
    medrag_instance=self.medrag,
    k=8,
    max_query_tokens=50,  # ~200 chars (more conservative for single-agent)
    log_dir=log_dir
)
```

### Key Differences

| Feature | Multi-Agent RAG | Single-Agent RAG |
|---------|----------------|------------------|
| Query Truncation | 2048 tokens (~8192 chars) | 200 chars (~50 tokens) |
| Tool Usage | Integrator only | Main agent |
| Dual-Query | Supported (MedCorp + UMLS) | Not used |
| Retrieval Step | Part of integrator reasoning | Separate step before prediction |

## Testing Strategy

### Unit Tests

1. **Test CoT zero-shot** (5 samples)
2. **Test CoT few-shot** (5 samples)
3. **Test RAG zero-shot** (5 samples)
4. **Test RAG few-shot** (5 samples)

### Comparison Tests

Compare all 4 configurations:
- CoT zero-shot: effGen vs VLLM
- CoT few-shot: effGen vs VLLM
- RAG zero-shot: effGen vs VLLM
- RAG few-shot: effGen vs VLLM

### Validation Criteria

For each configuration:
1. ✅ Predictions extracted successfully (not all fallbacks)
2. ✅ Accuracy within ±5% of VLLM baseline
3. ✅ Output format matches (results.json + logs/)
4. ✅ Hyperparameters verified identical

## Implementation Checklist

### Phase 1: CoT Mode ✅ (to be implemented)
- [ ] Create `mortality_single_agent_effgen_cot.py`
- [ ] Implement zero-shot mode
- [ ] Implement few-shot mode
- [ ] Test with 5 samples each
- [ ] Verify prediction extraction
- [ ] Match VLLM hyperparameters

### Phase 2: RAG Mode ✅ (to be implemented)
- [ ] Create `mortality_single_agent_effgen_rag.py`
- [ ] Initialize MedRAG before model
- [ ] Integrate retrieval tool
- [ ] Implement two-step process
- [ ] Test zero-shot and few-shot
- [ ] Verify query truncation (200 chars)

### Phase 3: Integration ✅ (to be implemented)
- [ ] Create `run_kare_single_agent_effgen.py`
- [ ] Support all 4 configurations
- [ ] Auto-generate output paths
- [ ] Calculate metrics
- [ ] Create README with usage instructions

## Expected Command-Line Interface

```bash
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

## Success Criteria

### CoT Mode
- [ ] Zero-shot works correctly
- [ ] Few-shot works correctly
- [ ] Predictions match format (1/0)
- [ ] Fallback rate < 10%
- [ ] Accuracy within ±5% of VLLM

### RAG Mode
- [ ] MedRAG initializes correctly
- [ ] Retrieval tool works
- [ ] Two-step process executes
- [ ] Zero-shot and few-shot both work
- [ ] Accuracy within ±5% of VLLM

### Integration
- [ ] All 4 configurations supported
- [ ] Output directory structure matches VLLM
- [ ] Metrics calculation correct
- [ ] Resume support works

## Summary

The single-agent effGen implementation is **simpler** than the multi-agent debate system:
- Only 1 agent (vs 3)
- Simple 1/0 predictions (vs probabilities)
- KARE task-based prompts (vs clinical assistant)
- Two in-context modes add flexibility

**Estimated Implementation**: ~3 files, ~60KB code

**Key Challenges**:
1. Adapting KARE-style prompting to effGen
2. Two-step RAG process with effGen tools
3. Query truncation differences (200 chars for single-agent)
4. Supporting 4 configurations (2 modes × 2 in-context)

**Reuse from Multi-Agent**:
- MedRAG tool class
- Metrics calculation
- Output directory structure
- Logging format

Ready to proceed with implementation upon approval! 🚀
