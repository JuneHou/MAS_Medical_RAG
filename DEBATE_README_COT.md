# Multi-Agent CoT Debate System for MedQA

This implements a **two-agent Chain-of-Thought (CoT) debate system** for medical question answering:
- **Pure reasoning-based debate** without retrieval
- **Multi-round structured dialogue** with summarization
- **Intelligent consensus verification** with fallback mechanisms

## System Architecture

### CoT Debate Process Flow
```
┌─────────────────────────────────────────────────────────────┐
│                        Question                              │
└────────────────┬────────────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
   ┌────▼─────┐     ┌────▼─────┐
   │ Agent1   │     │ Agent2   │
   │ (Round 1)│     │ (Round 1)│
   │ Direct   │     │ Direct   │
   │ Reasoning│     │ Reasoning│
   └────┬─────┘     └────┬─────┘
        │                │
        │ Answer A       │ Answer B
        │                │
   ┌────▼─────┐     ┌────▼─────┐
   │ Summarize│     │ Summarize│
   │ Round 1  │     │ Round 1  │
   │ (~500    │     │ (~500    │
   │ tokens)  │     │ tokens)  │
   └────┬─────┘     └────┬─────┘
        │                │
        │ Summary1       │ Summary2
        │                │
   ┌────▼─────┐     ┌────▼─────┐
   │ Agent1   │     │ Agent2   │
   │ (Round 2)│     │ (Round 2)│
   │+Sum1+Sum2│     │+Sum2+Sum1│
   │ Context  │     │ Context  │
   └────┬─────┘     └────┬─────┘
        │                │
        │ Final A        │ Final B
        │                │
   ┌────▼─────┐     ┌────▼─────┐
   │ Summarize│     │ Summarize│
   │ Round 2  │     │ Round 2  │
   │ (~500    │     │ (~500    │
   │ tokens)  │     │ tokens)  │
   └────┬─────┘     └────┬─────┘
        │                │
        │ Summary3       │ Summary4
        │                │
        └────────┬────────┘
                 │
          ┌──────▼──────┐
          │ Verifier    │
          │ (Judge)     │
          │ Uses ALL 4  │
          │ Summaries   │
          │ (~2000 tok) │
          └──────┬──────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            ▼
Consensus    Disagreement   Error
(Direct)     (Reasoning)   (Fallback)
    │            │            │
    └────────────┼────────────┘
                 │
          ┌──────▼──────┐
          │ Final Answer│
          └─────────────┘
```

## Detailed CoT Debate Process

### Round 1: Independent Reasoning
1. **Agent1 Turn**: 
   - Receives question + options only
   - Uses system prompt: "Write at most 2 sentences of clinical reasoning and state your answer choice clearly"
   - Generates reasoning and answer independently (no prior context)
   - Response gets **automatically summarized** to ~500 tokens for efficiency

2. **Agent2 Turn**:
   - Receives question + options only (INDEPENDENT reasoning)
   - Uses system prompt: "Write at most 2 sentences of clinical reasoning and state your answer choice clearly"
   - Generates reasoning and answer independently (no knowledge of Agent1's position)
   - Response gets **automatically summarized** to ~500 tokens for efficiency

### Round 2: Debate with Cross-Summaries
3. **Agent1 Turn (Round 2)**:
   - Receives question + options + **both Round 1 summaries** (own + Agent2's, ~1000 tokens)
   - Uses debate prompt: "Review the previous conclusion and say whether you agree (add one new reason) or disagree (give one counterpoint)"
   - Generates new reasoning considering both its own and Agent2's summarized positions (temperature=0.7)
   - Response gets **automatically summarized** to ~500 tokens for verification

4. **Agent2 Turn (Round 2)**:
   - Receives question + options + **both Round 1 summaries** (own + Agent1's, ~1000 tokens)  
   - Uses same debate prompt as Agent1
   - Generates final reasoning considering both its own and Agent1's summarized positions (temperature=0.3)
   - Response gets **automatically summarized** to ~500 tokens for verification

### Verification: Comprehensive Consensus Decision
5. **Verifier (Judge)**:
   - **If consensus**: Agent1 and Agent2 have same final answer → Return that answer immediately (no LLM call)
   - **If disagreement**: 
     - Uses **all 4 summaries**: Agent1 Round 1, Agent2 Round 1, Agent1 Round 2, Agent2 Round 2 (~2000 tokens total)
     - Prompt: "Review this complete medical debate and determine the best answer. Consider evolution of arguments, quality of evidence, and final positions."
     - Generates reasoning based on complete debate progression and makes final decision
   - **Fallback**: If summaries unavailable, uses current responses; if verifier fails, returns Agent1's answer

### Key Features

**Efficiency Optimizations**:
- Both Round 1 and Round 2 responses automatically summarized to ~500 tokens after generation
- Round 2 agents receive both Round 1 summaries (own + other's) instead of full Round 1 reasoning (saves tokens)
- Verifier uses all 4 summaries (~2000 tokens) for comprehensive disagreement resolution

**Progressive Context Building**:
- Round 1: Independent reasoning → no context sharing (true independence)
- Round 2: Summary-based debate → focused on key points
- Verification: Complete debate summary → informed final decision based on full progression

**No Retrieval**:
- Pure reasoning-based approach
- No tool calling or document retrieval
- Relies entirely on model's medical knowledge

**Temperature Settings**:
- Agent1: temperature=0.7 for diverse reasoning
- Agent2: temperature=0.3 for more focused counterarguments
- Allows for varied reasoning approaches between agents

## Core Features

### 1. **Pure CoT Reasoning**
- **No Retrieval**: Relies entirely on model's built-in medical knowledge
- **Direct Reasoning**: Agents generate medical analysis without external documents
- **Focused Debate**: Clean, structured medical argumentation

### 2. **Intelligent Summarization System**
- **Round 1 Auto-Summarization**: Full reasoning responses automatically compressed to ~500 tokens
- **Round 2 Efficiency**: Agents receive summaries instead of full Round 1 reasoning  
- **Verifier Optimization**: Uses Round 1 summaries for consensus checking
- **Fallback Handling**: Truncation-based backup if LLM summarization fails

### 3. **Multi-Round Debate**
- Default: 2 rounds of Agent1 → Agent2 exchanges
- Configurable via `--rounds` parameter

### 4. **Adaptive Consensus Verification**
- **Quick Consensus**: If agents agree, returns answer immediately (no additional LLM call)
- **Disagreement Resolution**: Verifier analyzes Round 1 summaries to make final decision
- **Fallback Strategy**: Uses Agent1's answer if verification fails
- **Medical Focus**: Specialized medical expert system prompts for clinical decision-making

## Files

- **`run_debate_medrag_cot.py`**: Main CoT debate system implementation
  - Agent prompts for medical reasoning
  - Summarization system
  - Debate orchestration
  - Benchmark runner

- **`run_debate_medrag.py`**: Original RAG version (kept for reference)

## Usage

### Full Benchmark

```bash
cd /data/wang/junh/githubs/Debate

# Run CoT debate on different datasets
python run_debate_medrag_cot.py --dataset mmlu --rounds 2
python run_debate_medrag_cot.py --dataset medqa --rounds 2
python run_debate_medrag_cot.py --dataset medmcqa --rounds 2

# Run all datasets
for dataset in mmlu medqa medmcqa pubmedqa bioasq; do
    python run_debate_medrag_cot.py --dataset $dataset --rounds 2
done
```

### Parameters

```bash
--dataset    Dataset name: mmlu|medqa|medmcqa|pubmedqa|bioasq
--rounds     Number of debate rounds (default: 2)
--log_dir    Directory for logs (default: ./debate_logs)
--split      Dataset split (default: test, auto-switched to dev for medmcqa)
```

## Output Files

For each question `{qid}`, the system saves:

1. **`{qid}__complete_debate.json`**: Complete debate log
   - Round-by-round responses from both agents
   - Timing information
   - Final consensus decision
   - Debug information

2. **`{qid}__round{N}__{agent}__summary.json`**: Summarized responses
   - Agent1/Agent2 summaries for each round
   - Used for efficient Round 2 context

3. **`{qid}__verification.json`**: Verification process log
   - Verifier reasoning when agents disagree
   - Consensus type (agreement/disagreement/fallback)

4. **`{dataset}_results.json`**: Aggregated results
   - Questions, gold answers, predictions
   - Accuracy statistics
   - Timing summaries

5. **`summarization_logs.jsonl`**: Summarization tracking
   - Compression ratios
   - Summary usage patterns

## Example Debate Log

```json
{
  "question": "A 55-year-old man with type 2 diabetes presents with poor glycemic control despite maximum metformin. BMI is 32. What is the next best step?",
  "round_1": {
    "agent1_full": "Given this patient's obesity (BMI 32) and inadequate glycemic control on metformin, GLP-1 receptor agonists would be ideal as they provide glucose-lowering effects and significant weight loss benefits. The dual benefit addresses both hyperglycemia and obesity, key factors in his diabetes management.\n\nAnswer: A",
    "agent1_summary": "GLP-1 agonists recommended for obese T2DM patient failing metformin due to dual glucose-lowering and weight loss benefits. Answer: A",
    "agent2_full": "I agree with GLP-1 agonist therapy. This patient's elevated BMI makes him an excellent candidate since these agents promote weight loss while improving glycemic control. The cardiovascular benefits are an added advantage for diabetic patients.\n\nAnswer: A",
    "agent2_summary": "Agrees with GLP-1 agonist for obese diabetic patient due to weight loss, glucose control, and cardiovascular benefits. Answer: A"
  },
  "consensus": "agreement",
  "final_answer": "A",
  "verification": null
}
```

## Differences from Original RAG Version

| Feature | RAG Version | CoT Version |
|---------|-------------|-------------|
| **Information Source** | External document retrieval | Model's built-in knowledge |
| **Reasoning Type** | Evidence-based with citations | Pure chain-of-thought reasoning |
| **Context Management** | Document snippets | Automatic summarization |
| **Agent Input** | Question + retrieved docs | Question + previous summaries |
| **Efficiency** | Retrieval latency + LLM | LLM only with smart summarization |
| **Scalability** | Limited by corpus size | Limited by model knowledge cutoff |
| **Backend** | MedRAG + vLLM | Direct vLLM integration |

## Configuration

Edit `run_debate_medrag_cot.py` to change:

```python
# GPU selection
os.environ['CUDA_VISIBLE_DEVICES'] = '3,5,6'  # Your GPUs

# Model
LLM_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# Temperature settings
AGENT1_TEMP = 0.7   # More diverse reasoning
AGENT2_TEMP = 0.3   # More focused responses

# Debate parameters
MAX_ROUNDS = 2      # Number of debate rounds
SUMMARY_TARGET = 500 # Target tokens for summaries
```

## Evaluation

After running predictions, evaluate with MIRAGE's evaluator:

```bash
cd /data/wang/junh/githubs/mirage_medrag/MIRAGE

python src/evaluate.py \
    --results_dir ../Debate/debate_logs_cot \
    --llm_name meta-llama/Meta-Llama-3-8B-Instruct \
    --cot  # CoT evaluation mode
```

## Troubleshooting

### Import Errors
The lint warnings about unresolved imports are expected - the modules exist in the MedRAG directory which is added to `sys.path` at runtime.

### Memory Issues
If OOM occurs:
1. Reduce `max_model_len` in VLLMWrapper
2. Disable summarization temporarily: comment out `summarize_agent_response()` calls
3. Reduce batch size for processing multiple questions

### Slow Performance
1. Use GPU acceleration for LLM inference when available
2. Enable summarization to reduce context lengths in later rounds
3. Adjust `SUMMARY_TARGET` to balance efficiency vs information retention

## Next Steps

1. **Test on single question**: `python test_debate_single_cot.py` (if available)
2. **Run small benchmark**: `python run_debate_medrag_cot.py --dataset mmlu --rounds 2`
3. **Scale up**: Run on full datasets with 2 rounds
4. **Tune parameters**: Adjust rounds, temperature settings, summary target based on results
5. **Analyze logs**: Review `debate_logs_cot/*.json` to understand agent reasoning patterns

## Credits

Built on top of:
- **MIRAGE**: Medical QA benchmark and evaluation framework
- **vLLM**: Efficient LLM inference and batching
- **MedRAG**: Dataset loading utilities (parsing components only)
