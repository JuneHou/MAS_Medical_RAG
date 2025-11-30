# RAG Multi-Agent Debate System

## Overview

This system implements a **Retrieval-Augmented Generation (RAG) multi-agent debate** approach for medical question answering. Two agents (Agent1 and Agent2) engage in a structured debate with access to medical evidence retrieval, culminating in a verification step to determine the final answer.

## Key Features

- **Equal Agent Architecture**: Both agents have the same role and capabilities, differentiated only by temperature settings
- **RAG Tool Calling**: Agents retrieve medical evidence using `retrieve()` tool before reasoning
- **Multi-Round Debate**: 2-round structured debate with summarization between rounds
- **Evidence-Based Reasoning**: All reasoning must be supported by retrieved medical literature
- **Consensus Verification**: LLM verifier resolves disagreements with fallback mechanisms

## Architecture

### Agent Configuration
- **Agent1**: Temperature 0.7 (more exploratory reasoning)
- **Agent2**: Temperature 0.3 (more focused/conservative reasoning)
- **Both agents**: Equal roles with access to RAG retrieval tools

### Debate Flow
```
Round 1: Independent Evidence Gathering & Reasoning
├── Agent1: retrieve() → analyze evidence → \\boxed{A}
├── Agent2: retrieve() → analyze evidence → \\boxed{B}
└── Summarize responses (500 tokens each)

Round 2: Cross-Informed Evidence Gathering
├── Agent1: receives Agent2's R1 summary + retrieve() → \\boxed{A}
├── Agent2: receives Agent1's R1 summary + retrieve() → \\boxed{C}
└── Summarize responses

Verification: Consensus Resolution
├── Check direct agreement → if agree: return answer
├── LLM Verifier: analyze full debate → \\boxed{final}
└── Fallback: majority vote across rounds
```

## System Prompts

### Round 1 (Independent Reasoning)
```
You are a medical AI assistant participating in a clinical discussion with access to evidence retrieval.

Available tools:
- retrieve(query): Retrieve medical evidence for your query

Instructions:
1) Call retrieve("search query") to gather relevant medical evidence
2) Read the question and options carefully, then analyze the retrieved evidence
3) Write at most 2 sentences of clinical reasoning based on the evidence
4) State your answer choice clearly using \boxed{letter} format (for example \boxed{A})
```

### Round 2+ (Cross-Informed Debate)
```
You are a medical AI assistant participating in a clinical discussion with access to evidence retrieval.

Available tools:
- retrieve(query): Retrieve medical evidence for your query

Instructions:
1) Call retrieve("search query") to gather additional relevant evidence
2) Review the previous conclusion, then analyze the retrieved evidence
3) Write at most 2 sentences: briefly say whether you agree (add one new reason) or disagree (give one counterpoint) based on evidence
4) State your answer choice clearly using \boxed{letter} format (for example \boxed{A})

Consider the previous discussion when forming your response, but gather your own evidence.
```

## Configuration Parameters

### Model Settings
- **Model**: `Qwen/Qwen2.5-7B-Instruct`
- **GPU Assignment**: 
  - GPU 6: FAISS index + embeddings
  - GPU 7: VLLM inference
- **Max Length**: 4096 tokens per generation
- **Repetition Penalty**: 1.2

### RAG Settings
- **Corpus**: MedCorp (single-source optimized)
- **Retriever**: MedCPT
- **Default k**: 32 documents per agent
- **RRF k**: 60 (Reciprocal Rank Fusion parameter)
- **Database**: HNSW index with corpus caching

### Debate Settings
- **Max Rounds**: 2
- **Summarization**: 500 tokens (~2 sentences) per round
- **Summary Temperature**: 0.1 (focused summarization)
- **Verification Temperature**: 0.7 (balanced reasoning)

## Information Flow

### Round 1 → Round 2
Agent2 in Round 2 receives:
- Agent1's Round 1 summary (500 tokens)
- Own retrieval capability for new evidence

Agent1 in Round 2 receives:
- Agent2's Round 1 summary (500 tokens)  
- Own retrieval capability for new evidence

### Round 2 → Verification
Verifier receives:
- Agent1's Round 2 full response (cleaned, truncated)
- Agent2's Round 2 full response (cleaned, truncated)
- Original question and options

## Response Parsing

### Answer Extraction Priority
1. **\\boxed{} format**: `\boxed{A}`, `$\boxed{B}$`
2. **Answer choice patterns**: `Answer Choice: **C**`
3. **Explicit statements**: `The correct answer is D`
4. **Final answer patterns**: `Final answer: A`
5. **Other patterns**: `Therefore, B`

### Error Handling
- **Parse failures**: Return "PARSE_FAILED"
- **Empty responses**: Return "ERROR"
- **Invalid choices**: Trigger fallback mechanisms

## Verification Process

### Consensus Checking
1. **Direct Agreement**: If both agents give same answer → return immediately
2. **LLM Verification**: Full debate analysis with `\boxed{}` format requirement
3. **Majority Vote**: Across all rounds if verification fails
4. **Final Fallback**: Use Agent1's answer

### Verification Prompt
```
Review this complete medical debate and determine the best answer based on medical evidence:

Question: {question}
Options: {A: ..., B: ..., C: ..., D: ...}

Complete Debate Summary:
Agent1: {reasoning} Answer: {choice}
Agent2: {reasoning} Answer: {choice}

Based on the full debate progression across all rounds, which answer is most appropriate?
Provide your reasoning and final answer using \boxed{letter} format:
```

## Usage

### Basic Usage
```bash
python run_debate_medrag_rag.py \
    --dataset mmlu \
    --k 32 \
    --log_dir ./debate_logs \
    --rounds 2 \
    --split test
```

### Advanced Usage
```bash
python run_debate_medrag_rag.py \
    --dataset medqa \
    --k 64 \
    --log_dir ./experiments/medqa_rag \
    --rounds 2 \
    --split dev \
    --corpus_name MedCorp
```

### Supported Datasets
- **mmlu**: Medical subset of MMLU
- **medqa**: USMLE-style questions
- **medmcqa**: Indian medical entrance exam
- **pubmedqa**: PubMed research questions  
- **bioasq**: Biomedical semantic indexing

## Output Structure

### Timing Metrics
```json
{
  "total_debate_time": 45.2,
  "agent1_times": [{"round": 1, "generation_time": 8.1, "retrieval_time": 2.3}],
  "agent2_times": [{"round": 1, "generation_time": 7.8, "retrieval_time": 2.1}],
  "verification_time": 5.4
}
```

### Debate Log
```json
{
  "qid": "test_001",
  "question": "Which medication...",
  "options": {"A": "Metformin", "B": "Insulin"},
  "mode": "rag",
  "rounds": 2,
  "history": [
    {
      "role": "agent1",
      "round": 1,
      "answer": {
        "step_by_step_thinking": "Retrieved evidence shows...",
        "answer_choice": "A",
        "retrieval_time": 2.3,
        "generation_time": 8.1
      }
    }
  ],
  "final_answer": {
    "answer_choice": "A",
    "consensus_type": "verification_consensus"
  }
}
```

## Performance Optimization

### Memory Management
- **Response Cleaning**: Remove repetitive patterns, limit to 2000 chars
- **Summary Caching**: Round 1 summaries saved to disk for Round 2
- **Truncation**: Debate summaries limited to 3500 chars for verification

### GPU Optimization
- **FAISS GPU**: Dedicated GPU for vector similarity search
- **VLLM GPU**: Separate GPU for language model inference
- **Corpus Caching**: Pre-loaded medical corpus for faster retrieval

### Stop Sequences
- **Generation**: `["<|im_end|>", "</s>", "###", "\n\n\n\n"]`
- **Summarization**: `["<|im_end|>", "</s>", "\n\n\n", "Agent3", "Round 3", "###"]`
- **Verification**: `["<|im_end|>", "</s>", "###", "\n\n\n\n", "\n---", "---\n"]`

## Comparison with CoT Debate

| Feature | RAG Debate | CoT Debate |
|---------|------------|------------|
| **Evidence** | Retrieved medical literature | Internal model knowledge |
| **Tools** | `retrieve()` function | None |
| **Reasoning** | Evidence-based analysis | Pure logical reasoning |
| **Agent Roles** | Agent1 (0.7) + Agent2 (0.3) | Agent1 (0.7) + Agent2 (0.3) |
| **Prompt Length** | Longer (includes evidence) | Shorter (question only) |
| **Retrieval Time** | ~2-3s per agent per round | 0s |
| **Total Time** | ~45-60s per question | ~15-30s per question |

## Error Handling

### Common Issues
1. **Retrieval Failures**: Fall back to direct reasoning without evidence
2. **Tool Call Parsing**: Use regex patterns to extract retrieve() calls
3. **Response Truncation**: Clean repetitive patterns, limit length
4. **Verification Failures**: Multiple fallback levels (majority vote → Agent1)

### Debugging
- **Detailed Logs**: Complete debate history with raw responses
- **Timing Breakdown**: Per-step timing for performance analysis
- **Summarization Logs**: JSONL format for summarization quality analysis
- **Retrieval Logs**: Document retrieval results and relevance scores

## File Structure

```
debate_logs/
├── {dataset}/
│   ├── MedCorp/
│   │   ├── rag/
│   │   │   ├── {qid}__complete_debate.json
│   │   │   ├── {qid}__round1__{role}__summary.json
│   │   │   ├── {qid}__round2__{role}__response.json
│   │   │   ├── {qid}__verification.json
│   │   │   └── summarization_logs.jsonl
│   │   └── {dataset}_results.json
```

## Future Improvements

1. **Multi-Source RAG**: Support for multiple medical corpora
2. **Dynamic k**: Adaptive retrieval based on question complexity
3. **Source Attribution**: Track which documents influenced final answers
4. **Confidence Scoring**: Quantify agent certainty in retrieved evidence
5. **Real-time Retrieval**: Update corpus with latest medical literature

## References

- **Base System**: Adapted from CoT debate architecture
- **RAG Framework**: MedRAG with MedCPT retriever
- **Medical Corpus**: MedCorp collection
- **Evaluation**: MIRAGE benchmark suite