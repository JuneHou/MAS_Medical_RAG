# Module Summary: KARE/effgen — EffGen Experimental Track (Feb 2026)

**Directory:** `/data/wang/junh/githubs/Debate/KARE/effgen/`
**Date range of results:** 2026-02-07 to 2026-02-08

---

## 1. What Is "EffGen"?

"EffGen" is **not a model and not a retrieval algorithm**. It is a third-party Python framework for building agentic LLM workflows. The full name is stylized as `effGen` (pip package: `effgen`). It provides:

- `load_model(model_name, ...)` — a thin wrapper around HuggingFace `transformers` / vLLM that returns a `TransformersEngine` or `VLLMEngine` object
- `Agent` + `AgentConfig` — agent primitives with a built-in ReAct loop (tool use + reasoning)
- `BaseTool` — base class for LLM-callable tools

The KARE/effgen track re-implements the existing KARE multi-agent debate system using effGen as the **inference backend**, replacing the direct vLLM usage in the parent KARE directory. The stated goal is a fair framework-level performance comparison: same model, same hyperparameters, same prompts, different inference stack.

Key point from `IMPLEMENTATION_PLAN.md:1`: *"migrate the KARE multi-agent mortality prediction system from VLLM to the effgen framework while maintaining identical functionality for fair comparison."*

The framework is referenced as an open-source project at `https://github.com/ctrl-gaurav/effGen` (cited in `TOOL_IMPLEMENTATION_FIX.md:9`).

---

## 2. Integration With the KARE Debate Architecture

The effgen track **preserves the exact 3-agent, 2-round debate structure** from the parent KARE directory.

### Agents (unchanged from parent)

| Role | Round | Tools (CoT) | Tools (RAG) | Temp |
|------|-------|-------------|-------------|------|
| `mortality_risk_assessor` | 1 | none | none | 0.3 |
| `protective_factor_analyst` | 1 | none | none | 0.3 |
| `balanced_clinical_integrator` | 2 | none | `MedRAGRetrievalTool` | 0.5 |

### Round 1 (both analysts, in parallel)
Each analyst receives the target patient + one similar patient (positive/negative respectively) and does contrastive clinical pattern analysis. They are explicitly instructed **not to speculate about outcomes or mortality** — purely structural comparison. (`mortality_debate_effgen_cot.py:143`, `mortality_debate_effgen_cot.py:163`)

### Round 2 (integrator)
The integrator receives the target patient plus both round-1 analyses. In RAG mode it also has access to the retrieval tool and is instructed to call it before making a prediction. Output format: `MORTALITY PROBABILITY: X.XX` + `SURVIVAL PROBABILITY: X.XX` (must sum to 1.00). (`agentic_retrieve/mortality_debate_effgen_rag.py:223-237`)

### What effGen adds / changes
- Model is loaded via `effgen.load_model()` rather than `vllm.LLM()`
- Agents are defined as `effgen.Agent(config=AgentConfig(...))` objects
- The integrator uses `agent.run(prompt)` in default (ReAct) mode so effGen internally manages the tool-use loop; other agents use `agent.run(prompt, mode=AgentMode.SINGLE)` (`agentic_retrieve/mortality_debate_effgen_rag.py:455-461`)
- Data loading, similar-patient retrieval, and result serialization are fully reused from the parent KARE directory via `KAREDataAdapter`

### Known limitation discovered during development
`AgentConfig` in effGen only accepts `temperature` as a sampling parameter — it does **not** expose `top_p`, `max_new_tokens`, or `repetition_penalty`. This is noted in `EFFGEN_MODEL_ACCESS.md:34`. For the single-agent RAG variant, the solution was to drop below the `Agent` abstraction and call `model.generate()` directly on the underlying HuggingFace model (accessed as `model_wrapper.model`).

---

## 3. The `agentic_retrieve/` Subfolder and Its Variant

`agentic_retrieve/` contains a single implementation file:
- `mortality_debate_effgen_rag.py` (~30 KB) — the multi-agent RAG debate using effGen's native ReAct loop

This is the **corrected and final version** of the RAG debate, while the top-level `mortality_debate_effgen_rag.py` (referenced in `run_kare_debate_mortality_effgen.py:29` but commented out) appears to be an earlier, broken draft.

### What "agentic retrieve" means vs. prior RAG

In the parent KARE directory (vLLM-based RAG), retrieval is **manually orchestrated**: the script calls MedRAG directly, formats the retrieved text, and injects it into the prompt as a second turn. The agent never calls a tool itself.

In the `agentic_retrieve/` variant, the **integrator agent autonomously decides when and what to retrieve** using effGen's ReAct loop:

1. The integrator's system prompt explicitly mandates tool use before prediction (`agentic_retrieve/mortality_debate_effgen_rag.py:223-237`)
2. effGen internally manages the loop: agent generates → sees `retrieve_medical_evidence` tool available → calls it → receives result → reasons → generates final output
3. `max_iterations=5` allows for multiple tool-call/reasoning cycles (`agentic_retrieve/mortality_debate_effgen_rag.py:297`)

The integrator agent is created **per patient** (not once at startup) so that each debate gets its own `log_dir`-aware retrieval tool instance (`agentic_retrieve/mortality_debate_effgen_rag.py:278-305`).

**The round-1 analysts never have retrieval tools** in any variant. Only the integrator (round 2) uses agentic retrieval.

---

## 4. `effgen_medrag_tool.py` — The LLM-Callable Tool

File: `effgen_medrag_tool.py`

Two tool classes are defined:

### `MedRAGRetrievalTool` (primary tool)
Inherits from `effgen.tools.base.BaseTool`. Tool identity fields set as instance attributes after `super().__init__()`:

```python
self.name = "retrieve_medical_evidence"
self.description = "Retrieve medical evidence from MedCorp2 corpus and UMLS knowledge base. Use this tool to find clinical evidence, prognosis information, and medical terminology."
```
(`effgen_medrag_tool.py:50-51`)

Parameters exposed to the LLM:
- `input` (string, required=False) — the medical query; uses this name because effGen's ReAct loop always passes the argument as `input` (a bug fix documented in `agentic_retrieve/RAG_IMPLEMENTATION_FIXES.md:87-97`)
- `query` (string, required=False) — alternative name for backward compatibility
- `qid` (string, required=False) — query ID for logging

The `execute()` method truncates queries exceeding `max_query_chars = max_query_tokens * 4` characters, then calls `_retrieve_direct()`. For MedCorp2, it splits k=8 documents equally between the two sub-corpora: k_medcorp = k//2 + k%2 (= 5), k_umls = k//2 (= 4). Uses RRF parameter rrf_k=60. Returns a formatted string of up to 8 documents, each capped at 1000 characters of content (`effgen_medrag_tool.py:185-205`).

### `DualQueryMedRAGTool` (experimental)
Extends `MedRAGRetrievalTool` with a different tool identity:

```python
self.name = "retrieve_dual_medical_evidence"
self.description = "Retrieve medical evidence using separate queries for MedCorp (clinical literature) and UMLS (terminology). Provide queries as JSON: {\"medcorp\": \"query1\", \"umls\": \"query2\"}"
```
(`effgen_medrag_tool.py:250-251`)

Accepts a JSON string with `medcorp` and `umls` keys, or falls back to using the same string for both. Retrieves k=4 from each source (hardcoded at `effgen_medrag_tool.py:329, 343`).

### Three bugs documented and fixed in tool development
1. `BaseTool.__init__()` does not accept `name`/`description` arguments — must set as attributes (`agentic_retrieve/RAG_IMPLEMENTATION_FIXES.md:1-73`)
2. effGen's ReAct loop always passes the query argument as `input`, not `query` — parameter schema and `execute()` signature must use `input` (`agentic_retrieve/RAG_IMPLEMENTATION_FIXES.md:75-147`)
3. Empty system prompt prevents ReAct loop from triggering tool use — integrator must have a non-empty system prompt that instructs tool use (`agentic_retrieve/RAG_IMPLEMENTATION_FIXES.md:~150+`)

---

## 5. Variant Matrix

The effgen track implements 6 distinct configurations across two agent counts and three reasoning modes:

```
                        RETRIEVAL
                   None (CoT)    MedRAG (RAG)
                  ┌────────────┬─────────────────────┐
MULTI-AGENT  few  │ debate_cot │ agentic_retrieve/   │
(3 agents,   shot │            │ debate_rag          │
 2 rounds)        │            │                     │
                  ├────────────┼─────────────────────┤
SINGLE-AGENT zero │ single_cot │ single_rag          │
(1 agent,    shot │            │                     │
 1 round)    ─────┤            ├─────────────────────┤
             few  │ single_cot │ single_rag          │
             shot │            │                     │
                  └────────────┴─────────────────────┘
```

### Multi-agent debate
- Always "few-shot" in the sense that it always uses positive + negative similar patients
- CoT integrator: no tools, outputs MORTALITY PROBABILITY + SURVIVAL PROBABILITY
- RAG integrator: MedRAGRetrievalTool, effGen ReAct loop, max_iterations=5

### Single-agent
- Zero-shot: target patient only, no similar patients
- Few-shot: includes positive and negative similar patients in prompt
- CoT: single effGen Agent, no tools, predicts `# Prediction # 1/0`
- RAG: retrieval first turn → parse retrieve() call → execute MedRAG → final prediction (two-turn manual orchestration, NOT effGen's ReAct loop; `EFFGEN_RAG_MIGRATION.md:44-53`)

### Output format difference
- Multi-agent: floating-point probabilities (threshold at 0.5)
- Single-agent: binary prediction `1` or `0` extracted by regex

---

## 6. CLI Arguments

### `run_kare_debate_mortality_effgen.py` (multi-agent runner)

| Argument | Default | Description |
|----------|---------|-------------|
| `--start_idx` | 0 | Starting sample index |
| `--num_samples` | None (all) | Number of samples |
| `--output` | auto-generated | Output JSON file path |
| `--model` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace model name |
| `--gpus` | `0` | GPU IDs (comma-separated) |
| `--include_history` | False | Include full debate history in output |
| `--batch_size` | 10 | Intermediate save frequency |
| `--mode` | `cot` | `cot` or `rag` |
| `--corpus_name` | `MedCorp2` | MedRAG corpus (RAG only) |
| `--retriever_name` | `MedCPT` | MedRAG retriever (RAG only) |
| `--db_dir` | `/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus` | MedRAG database |

Auto-generated output path pattern:
- CoT: `results/effgen_cot_{model}/{results.json}`
- RAG: `results/effgen_rag_{model}_{retriever}/{results.json}`

(`run_kare_debate_mortality_effgen.py:407-448`)

### `run_kare_single_agent_effgen.py` (single-agent runner)

Same arguments as above, plus:

| Argument | Default | Description |
|----------|---------|-------------|
| `--in_context` | `zero-shot` | `zero-shot` or `few-shot` |

Auto-generated output path pattern:
- CoT: `results/single_effgen_cot_{model}_{zero/few}_shot/results.json`
- RAG: `results/single_effgen_rag_{model}_{retriever}_{zero/few}_shot/results.json`

(`run_kare_single_agent_effgen.py:357-394`)

Both runners support **resume**: they detect an existing `results.json` on startup and skip already-processed patient IDs. They also save intermediate results every `batch_size` samples and do a final save in a `finally` block even on Ctrl+C.

---

## 7. Verification and Comparison Scripts

### `compare_results.py`
Utility to diff a VLLM-run `results.json` against an effGen-run `results.json`. Takes `--vllm` and `--effgen` paths. Checks:
- **Metrics comparison**: accuracy, precision, recall, F1, macro-F1, specificity (numeric difference column)
- **Confusion matrix**: TP, FP, FN, TN counts
- **Per-patient prediction agreement rate**: counts patients where both frameworks predict the same label
- **Disagreement breakdown**: how many cases where VLLM correct but effGen wrong, effGen correct but VLLM wrong, or both wrong
- **Prediction bias**: whether one framework systematically predicts mortality more often
- **Runtime comparison**: average and total generation time per framework

(`compare_results.py:19-196`)

### `verify_exact_parity.py`
Static analysis tool that reads source code as text and checks implementation equivalence between `KARE/mortality_single_agent_rag.py` (vLLM) and `KARE/effgen/mortality_single_agent_effgen_rag.py` (effGen). Six invariant checks:

1. **Task description**: exact string match of `self.task_description` variable
2. **Retrieval instruction**: exact string match of `self.retrieval_instruction`
3. **Hyperparameters**: checks `temperature=0.7`, `top_p=0.9`, `max_tokens=32768`, `repetition_penalty=1.2` are all present (note: `max_tokens` vs `max_new_tokens` mapping)
4. **Retrieval logic**: regex checks for MedCorp2 k-splitting formula, UMLS splitting, `query[:200]` truncation, `rrf_k=60`, score formatting
5. **Prediction parsing**: presence of same regex patterns for extracting `# Prediction #` etc.
6. **Fallback logic**: `1 - ground_truth`, `prediction = 0`, `is_fallback = True` all present in both

Exits with code 0 on full pass, 1 on any mismatch. (`verify_exact_parity.py:95-274`)

### `test_configuration.py`
Smoke test for the effGen installation itself (no model weights needed for most checks):
1. Import `effgen`, `Agent`, `load_model`, `AgentConfig`
2. Create a valid `AgentConfig` with `enable_sub_agents=False, enable_memory=False` — prints out the accepted parameter set
3. Instantiate an `Agent` with `require_model=False`
4. Confirm that `enable_thinking=True` is **rejected** (invalid parameter)
5. Check that `/data/wang/junh/.cache/huggingface` and the Qwen2.5-7B-Instruct model cache exist
6. Confirm `HF_HOME` can be set

(`test_configuration.py`)

### `test_effgen_rag_implementation.py`
End-to-end integration test requiring live GPU:
1. Import `MortilitySingleAgentEffGenRAG`
2. Initialize with `gpu_ids="6"`, MedCorp2/MedCPT, `in_context="zero-shot"`
3. Assert `system.model`, `system.medrag`, `system.medrag_tool` are all non-None
4. Verify hardcoded hyperparameters are present
5. Check `"Mortality Prediction Task:"` is in `task_description`
6. Check `"retrieve relevant medical evidence using retrieve(query)"` is in `retrieval_instruction`
7. Optionally runs a live prediction on a mock patient record

(`test_effgen_rag_implementation.py`)

---

## 8. Hard-Coded Paths and Models

All paths below appear verbatim in source files; changing the machine/environment would require editing each file.

| Path / Value | Where | Purpose |
|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | all py files | HuggingFace model ID (default `--model`) |
| `/data/wang/junh/.cache/huggingface` | all py files | HF_HOME / model cache (`model_cache_dir` default) |
| `/data/wang/junh/.cache/huggingface/models--Qwen--Qwen2.5-7B-Instruct` | `IMPLEMENTATION_PLAN.md:47`, `test_configuration.py:97` | Local model snapshot path |
| `/data/wang/junh/githubs/mirage_medrag/MedRAG` | `effgen_medrag_tool.py:15`, all RAG py files | MedRAG root (added to `sys.path`) |
| `/data/wang/junh/githubs/mirage_medrag/MedRAG/src` | all RAG py files | MedRAG src (added to `sys.path`) |
| `/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus` | all RAG py files, default `--db_dir` | MedRAG corpus database directory |
| `/data/wang/junh/githubs/mirage_medrag/MIRAGE/src` | `agentic_retrieve/mortality_debate_effgen_rag.py:21`, `mortality_single_agent_effgen_rag.py:26` | MIRAGE evaluation src |

### MedRAG configuration (hard-coded defaults)
```python
MedRAG(
    llm_name=model_name,
    rag=True,
    retriever_name="MedCPT",
    corpus_name="MedCorp2",
    db_dir="/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus",
    corpus_cache=True,
    HNSW=True,
    retriever_device=f"cuda:{main_gpu}"
)
```
(`mortality_single_agent_effgen_rag.py:104-115`)

### Generation hyperparameters (hard-coded, not CLI-configurable)
```python
# Single-agent RAG (must match vLLM original exactly):
temperature=0.7, top_p=0.9, max_new_tokens=32768, repetition_penalty=1.2

# Debate system (all modes):
# Analysts: temperature=0.3, max_tokens=32768, top_p=0.9
# Integrator: temperature=0.5, max_tokens=32768, top_p=0.9
# repetition_penalty: 1.15 (RAG), 1.2 (CoT)
```
(`IMPLEMENTATION_PLAN.md:32-47`)

---

## 9. Existing Results (as of Feb 2026)

Five result directories exist under `effgen/results/`:

| Directory | n | Accuracy | Macro-F1 | Notes |
|---|---|---|---|---|
| `effgen_cot_Qwen_Qwen2.5_7B_Instruct` | 645 | 0.000 | 0.000 | All predicted 1, 0 TN — collapsed to always-mortality |
| `single_effgen_cot_Qwen_Qwen2.5_7B_Instruct_zero_shot` | 996 | 0.073 | 0.073 | 37 fallbacks, very high false-positive rate |
| `single_effgen_cot_Qwen_Qwen2.5_7B_Instruct_few_shot` | 996 | 0.062 | 0.061 | 113 fallbacks, performance worse than zero-shot |
| `single_effgen_rag_Qwen_Qwen2.5_7B_Instruct_MedCPT_zero_shot` | 190 | 0.005 | 0.005 | 188/190 fallbacks — tool not working |
| `single_effgen_rag_Qwen_Qwen2.5_7B_Instruct_MedCPT_few_shot` | 72 | 0.000 | 0.000 | 72/72 fallbacks — complete tool failure |

The RAG results are clearly broken: near-100% fallback rate indicates the `retrieve()` call was never successfully executed. This is consistent with the three tool bugs documented in `agentic_retrieve/RAG_IMPLEMENTATION_FIXES.md` — the fixes were applied to `agentic_retrieve/mortality_debate_effgen_rag.py` but the top-level single-agent RAG file (`mortality_single_agent_effgen_rag.py`) may not have been fully fixed before those runs. There are **no result files for the agentic_retrieve RAG debate variant** (the presumably correct version), suggesting it was fixed but not yet run to completion.

---

## 10. Key Implementation Decisions and Design Tensions

1. **Why effGen at all?** It was explored as a potential replacement for direct vLLM usage, offering agent primitives. However, the `AgentConfig` interface turned out to be too restrictive (`top_p`, `repetition_penalty` not supported), so for exact parity the code bypasses `Agent.run()` and calls the underlying HuggingFace model directly. This largely defeats the purpose of using the framework.

2. **Two-turn vs. ReAct for RAG**: The parent KARE vLLM RAG uses a manual two-turn orchestration; the `agentic_retrieve/` variant switches to true ReAct (agent calls tool autonomously). These are architecturally different and not directly comparable.

3. **Device mapping**: effGen's `load_model()` uses `device_map={"": 0}` (all layers on device 0). Physical GPU selection is handled by setting `CUDA_VISIBLE_DEVICES` before any CUDA initialization. MedRAG retriever goes to `cuda:{main_gpu}`. The comment at `mortality_debate_effgen_cot.py:93` explains the remapping logic.

4. **attn_implementation="eager"**: Used in all `load_model()` calls to avoid flash attention compatibility issues. (`mortality_debate_effgen_cot.py:99`)
