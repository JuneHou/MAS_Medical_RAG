# effGen Implementation Plan for KARE Multi-Agent Debate System

## Overview
This document outlines the plan to migrate the KARE multi-agent mortality prediction system from VLLM to the effgen framework while maintaining identical functionality for fair comparison.

## Current System Analysis

### Architecture
- **3 Specialized Agents**:
  1. `mortality_risk_assessor` - Analyzes positive similar patients (mortality=1)
  2. `protective_factor_analyst` - Analyzes negative similar patients (mortality=0)
  3. `balanced_clinical_integrator` - Synthesizes analyses and makes final prediction

- **2 Debate Rounds**:
  1. Round 1: Both analysts generate label-blind clinical pattern comparisons
  2. Round 2: Integrator synthesizes analyses and predicts mortality/survival probabilities

### Two Modes
1. **CoT Mode** (`mortality_debate_cot.py`):
   - Pure chain-of-thought reasoning
   - No external retrieval
   - Uses VLLM directly

2. **RAG Mode** (`mortality_debate_rag.py`):
   - Enhanced with MedRAG retrieval
   - Retrieves from MedCorp2 corpus using MedCPT retriever
   - Uses VLLM through VLLMWrapper
   - Integrator can call retrieval tools during reasoning

### Key Hyperparameters (MUST MATCH EXACTLY)
```python
# Analyst agents (Round 1)
temperature: 0.3
max_tokens: 32768  # Same for both RAG and CoT
top_p: 0.9
repetition_penalty: 1.15 (RAG) / 1.2 (CoT)
stop_sequences: ["<|im_end|>", "</s>", "End of response.", "---"] (RAG)
                 ["<|im_end|>", "</s>"] (CoT)

# Integrator agent (Round 2)
temperature: 0.5
max_tokens: 32768  # Same for both RAG and CoT
top_p: 0.9
repetition_penalty: 1.15 (RAG) / 1.2 (CoT)

# Model Configuration
model_path: "/data/wang/junh/.cache/huggingface/models--Qwen--Qwen2.5-7B-Instruct"
quantization: None  # VLLM uses full precision (no quantization)
```

### Critical Components to Preserve
1. **Data Flow**:
   - Uses `KAREDataAdapter` for loading patient data
   - Patient context formatting (temporal rolling visits)
   - Similar patient retrieval (positive/negative separation)

2. **Agent Prompts**:
   - Identical system prompts for each agent
   - Label-blind analysis for Round 1 agents
   - Probability extraction patterns for integrator

3. **Logging**:
   - Patient-specific log files in `debate_logs/` directory
   - Structured output with predictions and probabilities

4. **Metrics**:
   - Accuracy, Precision, Recall, F1, Macro-F1, Specificity
   - Confusion matrix tracking

## effGen Framework Analysis

### Key Components
1. **Model Loading**:
   ```python
   from effgen import Agent, load_model
   # Use full precision (no quantization) to match VLLM
   model = load_model("Qwen/Qwen2.5-7B-Instruct")
   # Or use cached model path directly
   model = load_model("/data/wang/junh/.cache/huggingface/models--Qwen--Qwen2.5-7B-Instruct")
   ```

2. **Agent Configuration**:
   ```python
   from effgen.core.agent import AgentConfig
   config = AgentConfig(
       name="agent_name",
       model=model,
       tools=[],  # Optional tools
       system_prompt="...",
       max_iterations=1,  # Single-turn (not 10)
       temperature=0.3,   # Match VLLM settings
       max_tokens=32768   # Match VLLM settings
   )
   ```

3. **Agent Execution**:
   ```python
   agent = Agent(config=config)
   result = agent.run("Your task here")
   print(result.output)
   ```

4. **Built-in Tools**:
   - `Calculator` - Math operations
   - `Retrieval` - RAG-based retrieval
   - `AgenticSearch` - Grep-based search
   - Custom tools can be created

### Configuration (from config.yaml - MODIFIED for our use case)
```yaml
model:
  default_provider: "transformers"
  default_model: "Qwen/Qwen2.5-7B-Instruct"
  cache_dir: "/data/wang/junh/.cache/huggingface"
  device: "auto"
  quantization: None  # NO quantization (match VLLM full precision)

agent:
  max_iterations: 1  # Single-turn execution (not 10)
  temperature: 0.3-0.5  # Varies by agent
  enable_sub_agents: false
  enable_memory: false
```

## Implementation Strategy

### Phase 1: effGen CoT Mode
**File**: `mortality_debate_effgen_cot.py`

**Approach**:
1. Create a custom model loader that uses Qwen2.5-7B-Instruct (matching VLLM version)
2. Define 3 agent configs with identical prompts from original system
3. Execute debate in sequence:
   - Agent 1 (mortality_risk_assessor) analyzes target + positive similar
   - Agent 2 (protective_factor_analyst) analyzes target + negative similar
   - Agent 3 (balanced_clinical_integrator) synthesizes and predicts
4. Extract probabilities using same regex patterns
5. Match all hyperparameters exactly

**Key Challenges**:
- effGen default max_iterations=10, but we need single-turn execution
- Need to ensure temperature and other params match VLLM exactly
- Must handle GPU allocation (avoid CUDA re-initialization errors)

**Solutions**:
- Set `max_iterations=1` for single-turn execution
- Override effGen defaults with explicit hyperparameters
- Follow MEDRAG_GPU_SETUP_FIX.md: Set CUDA_VISIBLE_DEVICES once in __init__

### Phase 2: effGen RAG Mode
**File**: `mortality_debate_effgen_rag.py`

**Approach**:
1. Integrate MedRAG retriever as a custom effgen tool
2. Create custom `MedRAGRetrieval` tool class that wraps MedRAG system
3. Connect to MedCorp2 corpus using MedCPT retriever (same as VLLM version)
4. Integrator agent gets access to retrieval tool
5. Support dual-query retrieval (separate MedCorp and UMLS queries)

**Key Challenges**:
- effGen's built-in Retrieval tool may not match MedRAG's interface
- Need to maintain same retrieval parameters (k=8, rrf_k=60)
- Must avoid MedRAG initialization errors (see MEDRAG_GPU_SETUP_FIX.md)

**Solutions**:
- Create custom tool class extending effgen's tool base
- Initialize MedRAG once in debate system __init__ (not in tool)
- Tool delegates to pre-initialized MedRAG instance
- Match retrieval query truncation limits (2048 tokens for integrator)

### Phase 3: Integration Scripts
**Files**:
- `run_kare_debate_mortality_effgen.py` - Main runner supporting both modes
- Reuses existing `kare_data_adapter.py` (no changes needed)

**Features**:
- Command-line args for mode selection (--mode cot/rag)
- Model selection (--model)
- GPU allocation (--gpus)
- Output path management
- Metrics calculation (reuse existing functions)

## File Structure

```
/data/wang/junh/githubs/Debate/KARE/effgen/
├── IMPLEMENTATION_PLAN.md                  # This file
├── mortality_debate_effgen_cot.py          # CoT mode using effgen
├── mortality_debate_effgen_rag.py          # RAG mode using effgen
├── run_kare_debate_mortality_effgen.py     # Main runner script
├── effgen_medrag_tool.py                   # Custom MedRAG tool for effgen
└── README.md                               # Usage instructions
```

## Hyperparameter Mapping

### VLLM → effGen Parameter Mapping

| VLLM Parameter | effGen Equivalent | Notes |
|----------------|-------------------|-------|
| `max_tokens` | `max_tokens` in sampling_params or agent config | Same |
| `temperature` | `temperature` in AgentConfig | Same |
| `top_p` | `top_p` in sampling_params | Need to verify effgen supports |
| `repetition_penalty` | `repetition_penalty` in sampling_params | Need to verify effgen supports |
| `stop_sequences` | `stop` in sampling_params | Same concept |
| `gpu_memory_utilization` | Set via model loading | May differ |
| `tensor_parallel_size` | Set via model loading | May differ |

### Critical Settings to Match
```python
# Model path
MODEL_PATH = "/data/wang/junh/.cache/huggingface/models--Qwen--Qwen2.5-7B-Instruct"
# OR use model name and let it auto-load from cache
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# For all agents (BOTH CoT and RAG use same max_tokens now)
AGENT_PARAMS = {
    "analyst1": {
        "temperature": 0.3,
        "max_tokens": 32768,  # SAME for both RAG and CoT
        "top_p": 0.9,
        "repetition_penalty": 1.15,  # RAG: 1.15, CoT: 1.2
    },
    "analyst2": {
        "temperature": 0.3,
        "max_tokens": 32768,  # SAME for both RAG and CoT
        "top_p": 0.9,
        "repetition_penalty": 1.15,  # RAG: 1.15, CoT: 1.2
    },
    "integrator": {
        "temperature": 0.5,
        "max_tokens": 32768,  # SAME for both RAG and CoT
        "top_p": 0.9,
        "repetition_penalty": 1.15,  # RAG: 1.15, CoT: 1.2
    }
}
```

## MedRAG Integration (RAG Mode Only)

### Current MedRAG Initialization (VLLM)
```python
from medrag import MedRAG
medrag = MedRAG(
    llm_name=model_name,
    rag=True,
    retriever_name="MedCPT",
    corpus_name="MedCorp2",
    db_dir="/path/to/corpus",
    corpus_cache=True,
    HNSW=True,
    retriever_device=f"cuda:{gpu_id}"
)
```

### effGen Custom Tool Approach
```python
from effgen.tools.base import BaseTool

class MedRAGRetrieval(BaseTool):
    """Custom tool wrapping MedRAG retrieval system"""
    
    def __init__(self, medrag_instance, k=8):
        super().__init__(
            name="retrieve_medical_evidence",
            description="Retrieve medical evidence from MedCorp2 and UMLS"
        )
        self.medrag = medrag_instance  # Pre-initialized
        self.k = k
    
    def execute(self, query: str) -> str:
        """Execute retrieval and return formatted results"""
        # Use direct retrieval bypass (avoid LLM generation issues)
        snippets, scores = self.medrag.retrieval_system.retrieve(query, k=self.k)
        return self._format_results(snippets, scores)
```

### Key Differences from VLLM Version
1. Tool wraps pre-initialized MedRAG (not created fresh each time)
2. Returns formatted string (not dict) for effgen compatibility
3. Handles dual-query format (medcorp + umls) via custom parsing

## Testing Strategy

### Validation Criteria
Both effgen and VLLM versions must produce:
1. **Same model**: Qwen2.5-7B-Instruct
2. **Same hyperparameters**: Temperature, max_tokens, etc.
3. **Same debate structure**: 3 agents, 2 rounds
4. **Same prompts**: Identical system prompts for each agent
5. **Same data**: KARE test set with same preprocessing

### Comparison Tests
1. **Single Sample Test**:
   - Run same patient through both systems
   - Compare:
     - Response lengths
     - Probability outputs
     - Prediction consistency

2. **Small Batch Test** (10 samples):
   - Compare metrics (accuracy, F1)
   - Compare generation times
   - Identify any systematic differences

3. **Full Evaluation** (entire test set):
   - Compare final metrics
   - Statistical significance testing
   - Runtime comparison

## Known Risks and Mitigations

### Risk 1: Hyperparameter Mismatch
**Risk**: effgen may not support exact same hyperparameters as VLLM
**Mitigation**: Verify effgen's vLLM backend supports all params; use custom SamplingParams if needed

### Risk 2: GPU Memory Management
**Risk**: Different GPU memory handling may affect capacity
**Mitigation**: Monitor memory usage; adjust gpu_memory_utilization if needed

### Risk 3: MedRAG Integration
**Risk**: Tool interface may not support complex retrieval patterns
**Mitigation**: Create custom tool with full control over MedRAG calls

### Risk 4: Response Format Differences
**Risk**: effgen may wrap responses differently
**Mitigation**: Test extraction patterns; adjust if needed

### Risk 5: CUDA Initialization
**Risk**: Same CUDA re-initialization errors as VLLM
**Mitigation**: Follow MEDRAG_GPU_SETUP_FIX.md strictly; set CUDA_VISIBLE_DEVICES once

## Success Criteria

### Phase 1 (CoT Mode)
- [ ] effgen CoT mode runs successfully on test samples
- [ ] Hyperparameters match VLLM exactly
- [ ] Output format matches (probabilities extractable)
- [ ] Metrics within ±2% of VLLM baseline

### Phase 2 (RAG Mode)
- [ ] MedRAG retrieval integrates successfully
- [ ] Retrieval parameters match (k=8, etc.)
- [ ] Dual-query retrieval works
- [ ] Metrics within ±2% of VLLM baseline

### Phase 3 (Integration)
- [ ] Runner script supports both modes
- [ ] Command-line interface matches original
- [ ] Output directory structure identical
- [ ] Logging format consistent

## Next Steps

1. **Review this plan** - Get approval before implementation
2. **Implement CoT mode** - Start with simpler non-RAG version
3. **Test and validate** - Run small batch tests
4. **Implement RAG mode** - Add MedRAG integration
5. **Full evaluation** - Run on complete test set
6. **Comparison analysis** - Document any differences

## References

- effGen GitHub: https://github.com/ctrl-gaurav/effGen
- effGen Paper: https://arxiv.org/abs/2602.00887
- Original VLLM CoT: `/data/wang/junh/githubs/Debate/KARE/mortality_debate_cot.py`
- Original VLLM RAG: `/data/wang/junh/githubs/Debate/KARE/mortality_debate_rag.py`
- MedRAG Setup Fix: `/data/wang/junh/githubs/Debate/KARE/MEDRAG_GPU_SETUP_FIX.md`
