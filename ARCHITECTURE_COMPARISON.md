# Architecture Comparison: Original vs New Debate System

## Original System (two_agents.ipynb)

```
User Question
     |
     v
[MedCorpRetriever]
     |
     | BM25-style TF-IDF
     | 10,000 docs indexed
     |
     v
[Generic Retrieval]
     |
     +------------------+
     |                  |
     v                  v
[Analyst]         [Skeptic]
     |                  |
     | Manual          | Manual
     | TOOL: retrieve()| TOOL: retrieve()
     |                  |
     v                  v
[Same Evidence]   [Same Evidence]
     |                  |
     v                  v
[Generic Debate]
     |
     v
[Judge]
     |
     v
Final Answer
```

**Characteristics:**
- ✓ Simple BM25 retrieval
- ✓ Small corpus (10K docs)
- ✓ Generic agent roles
- ✓ Manual tool calls via TOOL: syntax
- ✓ Same evidence pool for both agents
- ✓ Callback-based interception
- ✗ No subquery expansion
- ✗ No role-specific evidence
- ✗ No structured logging
- ✗ No benchmarking infrastructure

---

## New System (run_debate_medrag.py)

```
User Question
     |
     +----------------------------------+
     |                                  |
     v                                  v
[Analyst Subquery Gen]         [Skeptic Subquery Gen]
     |                                  |
     | Q1: "clinical guidelines"        | Q1: "contraindications"
     | Q2: "standard treatment"         | Q2: "differential diagnosis"
     | Q3: "supporting evidence"        | Q3: "adverse effects"
     |                                  |
     v                                  v
[MedRAG Retrieval]             [MedRAG Retrieval]
     |                                  |
     | MedCPT Encoder                   | MedCPT Encoder
     | MedCorp Database                 | MedCorp Database
     | HNSW Index                       | HNSW Index
     |                                  |
     v                                  v
[3 Result Sets]                [3 Result Sets]
     |                                  |
     | RRF Fusion                       | RRF Fusion
     | (within role)                    | (within role)
     |                                  |
     v                                  v
[Role-Specific Evidence]       [Role-Specific Evidence]
     |                                  |
     | - Supportive docs                | - Critical docs
     | - Guideline-based                | - Safety concerns
     | - Standard practice              | - Edge cases
     |                                  |
     v                                  v
[Analyst Answer]               [Skeptic Counter]
     |                                  |
     | + Doc citations                  | + Doc citations
     | + Clinical reasoning             | + Critical analysis
     | + Proposed answer                | + Alternative view
     |                                  |
     +------------------+---------------+
                        |
                        v
                   [Judge]
                        |
                        | Reviews both sides
                        | Weighs evidence
                        | Outputs JSON
                        |
                        v
                 Final Answer
                        |
                        v
              [Structured Logging]
                        |
                        +-- analyst__snippets.json
                        +-- skeptic__snippets.json
                        +-- debate.jsonl
                        +-- results.json
```

**Characteristics:**
- ✓ Full MedRAG pipeline (MedCPT + MedCorp)
- ✓ Large corpus (full MedCorp)
- ✓ Role-specific subqueries
- ✓ Automatic LLM-based query expansion
- ✓ RRF fusion per role
- ✓ Distinct evidence pools for each agent
- ✓ Direct vLLM integration
- ✓ Medical-domain prompts
- ✓ Comprehensive logging
- ✓ Benchmarking infrastructure
- ✓ Batch processing support

---

## Key Technical Differences

### 1. Retrieval Strategy

| Aspect | Original | New |
|--------|----------|-----|
| **Algorithm** | BM25 (TF-IDF) | MedCPT (dense retrieval) |
| **Corpus Size** | 10,000 docs | Full MedCorp (~millions) |
| **Index Type** | In-memory dict | FAISS HNSW |
| **Query Expansion** | None | LLM-generated subqueries |
| **Fusion** | None | RRF across subqueries |
| **Role Specificity** | Same for all agents | Role-specific bias |

### 2. Agent Architecture

| Aspect | Original | New |
|--------|----------|-----|
| **System Prompts** | Generic debate | Medical-domain specific |
| **Tool Interface** | Manual TOOL: syntax | Direct retrieval calls |
| **Evidence Access** | Shared pool | Role-specific pools |
| **Subqueries** | None | 3 per role |
| **Output Format** | Free text | Structured (dict/JSON) |

### 3. Conversation Flow

**Original:**
```
User → Analyst → [Tool] → Skeptic → [Tool] → Analyst → ... → Judge
```
- Callback-based tool interception
- Round-robin speaker selection
- Tool results injected as messages
- No explicit rounds

**New:**
```
Round 1:
  User → Analyst.retrieve() → Analyst.answer()
       → Skeptic.retrieve() → Skeptic.answer()
Round 2:
  Analyst.retrieve() → Analyst.answer()
  Skeptic.retrieve() → Skeptic.answer()
Judge:
  Review → Decide → JSON output
```
- Direct function calls
- Explicit multi-round structure
- Separate retrieval and reasoning steps
- Structured output format

### 4. Logging & Outputs

| Aspect | Original | New |
|--------|----------|-----|
| **Conversation Log** | groupchat.messages | {qid}__debate.jsonl |
| **Retrieval Results** | Not saved | {qid}__{role}__snippets.json |
| **Final Answer** | Free text | JSON with reasoning |
| **Aggregated Results** | None | {dataset}_results.json |
| **Accuracy Tracking** | Manual | Automatic |
| **Reproducibility** | Limited | Full audit trail |

---

## Code Mapping

### Original: Manual Tool Call
```python
# In agent message:
"TOOL: retrieve(\"diabetes treatment metformin\")"

# Callback intercepts:
def intercept_and_reply(recipient, messages, sender, config, **kwargs):
    tool_out = tool_router(content)
    return True, f"Tool result:\n{tool_out}"
```

### New: Direct Retrieval
```python
# Subquery generation:
subqueries = expand_subqueries(llm, "analyst", question, options)
# ["clinical guidelines diabetes type 2",
#  "metformin failure next step",
#  "glycemic control HbA1c targets"]

# Retrieve for each:
for subq in subqueries:
    snippets, scores = medrag.retrieval_system.retrieve(subq, k=32)
    runs.append(snippets)

# Fuse with RRF:
fused = rrf_fuse(runs, K=60)
```

---

## Performance Expectations

### Original System
- **Speed**: ⚡⚡⚡ Fast (BM25 is very fast)
- **Quality**: ⭐⭐⭐ Moderate (limited corpus, no subqueries)
- **Coverage**: ⚠️ Limited (only 10K docs)
- **Evidence Diversity**: ⚠️ Low (same retrieval for both agents)

### New System
- **Speed**: ⚡⚡ Moderate (dense retrieval + subqueries)
- **Quality**: ⭐⭐⭐⭐ High (medical-domain encoder, role-specific)
- **Coverage**: ✓ Comprehensive (full MedCorp)
- **Evidence Diversity**: ✓ High (role-specific subqueries + RRF)

---

## Migration Path

### Step 1: Test Equivalence
```bash
# Test with original notebook
cd /data/wang/junh/githubs/Debate
jupyter notebook two_agents.ipynb
# Run Cell 11 (medical_task debate)

# Test with new system
python test_debate_single.py
# Compare outputs
```

### Step 2: Compare Evidence Quality
```bash
# Original: Check groupchat.messages for "Tool result:"
# New: Check test_debate_logs/test_001__analyst__snippets.json

# Look for:
# - Number of documents
# - Document relevance
# - Citation quality
```

### Step 3: Scale to Benchmark
```bash
# Start small
python run_debate_medrag.py --dataset mmlu --k 10

# Check logs
cat debate_logs/mmlu_results.json | jq '.[] | {correct: .correct}'

# Scale up
python run_debate_medrag.py --dataset mmlu --k 32 --rounds 2
```

### Step 4: Full Evaluation
```bash
# Run all datasets
for dataset in mmlu medqa medmcqa pubmedqa bioasq; do
    python run_debate_medrag.py --dataset $dataset --k 32 --rounds 2
done

# Aggregate results
python analyze_debate_results.py --log_dir debate_logs
```

---

## When to Use Each System

### Use Original (two_agents.ipynb) for:
- ✓ Quick prototyping
- ✓ Testing conversation flows
- ✓ Understanding AutoGen mechanics
- ✓ Interactive experimentation
- ✓ Small-scale testing

### Use New (run_debate_medrag.py) for:
- ✓ Production benchmarking
- ✓ Large-scale evaluation
- ✓ Medical QA tasks
- ✓ Role-specific retrieval needs
- ✓ Reproducible research
- ✓ Evidence audit trails

---

## Summary

The new system implements your full specification:

1. ✅ **Same MedRAG backend** (MedCPT + MedCorp)
2. ✅ **Two roles with different "info needs"**
3. ✅ **Agent-specific retrieval** via subqueries + RRF
4. ✅ **Judge makes final decision**
5. ✅ **Role-specific prompts** (Analyst/Skeptic/Judge)
6. ✅ **Pseudocode implemented** in `run_debate_medrag.py`
7. ✅ **Benchmarking integration** with existing datasets
8. ✅ **Comprehensive logging** for every question

You can now:
- Run `test_debate_single.py` to test quickly
- Run `run_debate_medrag.py` for full benchmarks
- Use the last notebook cell for interactive testing
- Read `DEBATE_README.md` for complete documentation
