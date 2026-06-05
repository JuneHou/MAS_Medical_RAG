# Module 02 — KARE Mortality-Debate Core Scripts

**Location:** `/data/wang/junh/githubs/Debate/KARE/`
**Scope:** Multi-agent debate system for MIMIC-III/IV in-hospital mortality prediction, built on top of KARE's similar-patient retrieval and MedRAG evidence retrieval.

---

## 1. Four-Agent Architecture

The canonical system (`mortality_debate_rag.py`) exposes three *named* agent roles that collectively implement what the code comments call a "4-agent" debate. In practice two agents run in parallel in Round 1, so the logical count is:

| Logical # | Role key | Round | Purpose |
|-----------|----------|-------|---------|
| Agent 1 | `mortality_risk_assessor` | 1 (batch) | Contrastive analysis vs. a **positive** similar patient (mortality=1 case) |
| Agent 2 | `protective_factor_analyst` | 1 (batch) | Contrastive analysis vs. a **negative** similar patient (survival=0 case) |
| Agent 3/4 | `balanced_clinical_integrator` | 2 | Reads both analyses, performs dual MedCorp + UMLS retrieval, outputs mortality + survival probabilities |

`self.agent_roles` (`mortality_debate_rag.py:148-152`):
```python
self.agent_roles = [
    "mortality_risk_assessor",
    "protective_factor_analyst",
    "balanced_clinical_integrator"
]
self.max_rounds = 2
```

### System Prompts (canonical `_rag.py` version)

**mortality_risk_assessor** (`mortality_debate_rag.py:162-180`):
> "You are a medical AI that analyzes clinical patterns between patients. ... **IMPORTANT:** Do NOT speculate about outcomes or mortality. Focus solely on clinical pattern analysis."

The prompt instructs: (1) list shared conditions/procedures/medications across target + positive-similar, (2) list similar-patient-specific features, (3) analyze temporal progression. Outcome labels are intentionally withheld — analysts are label-blind.

**protective_factor_analyst** (`mortality_debate_rag.py:182-200`):
Identical prompt text to `mortality_risk_assessor`. The differentiation is solely in which similar-patient block is injected (positive vs. negative). The two agents run together in a VLLM batch call.

**balanced_clinical_integrator** (`mortality_debate_rag.py:202-224`):
```
You are a medical AI Clinical Assistant analyzing mortality and survival probabilities
for the NEXT hospital visit.

IMPORTANT: Mortality is rare. Only assign a high mortality probability when the patient
appears at extremely high risk of death with strong evidence. The Target patient is the
source of truth. Do not treat Similar-only items as present in the Target.

Available tools:
- <search_umls>query</search_umls>: Retrieve concept/synonym/abbreviation help from UMLS.
- <search_medcorp>query</search_medcorp>: Retrieve clinical/prognosis evidence from MedCorp.
...
MORTALITY PROBABILITY: X.XX (0.00 to 1.00)
SURVIVAL PROBABILITY: X.XX (0.00 to 1.00)
Note: The two probabilities MUST sum to exactly 1.00
```

The integrator is explicitly survival-leaning ("mortality is rare") — this is the conservative integration design intent noted in the original project brief.

---

## 2. Variant Comparison

All five live variants share the same class name `MortalityDebateSystem` and similar `__init__` signatures; the deltas are described below.

### 2a. `mortality_debate_rag.py` — Canonical (1926 LOC)

- **Retrieval:** Dual-query to MedCorp and UMLS via separate `<search_medcorp>` / `<search_umls>` XML tags (`_parse_tool_call`, line ~617). Each corpus returns k=4 docs (8 total).
- **GPU management:** Preserves externally set `CUDA_VISIBLE_DEVICES` if present (`original_cuda_visible_devices` check, line 64). This was a fix over the `_fast` variant.
- **Integrator prompt:** Explicitly names both `<search_umls>` and `<search_medcorp>` tools.
- **Fallback on parse failure:** Retries integrator up to 2 times; if still no probabilities, re-runs Round 1 agents in batch (`debate_mortality_prediction` line ~1770).

### 2b. `mortality_debate_rag_fast.py` — Speed-Optimized (1932 LOC)

- **Key addition:** `--precomputed_log_dir` flag and `_load_debate_history_from_logs()` method (`line 698`). When this flag is set, Agents 1–2 are skipped entirely; the integrator is fed debate history parsed from a prior run's `.log` files. This cuts wall time by ~2/3 when experimenting with the integrator alone.
- **GPU management:** Unconditionally sets `CUDA_VISIBLE_DEVICES = self.main_gpu` (line 104). No external-env preservation — this is the original pattern before the canonical version added the guard.
- **Integrator prompt:** Uses single `<search>query</search>` tag (not dual-query). `_parse_tool_call` validates query keywords against a hardcoded list (`['mortality', 'survival', 'risk', ...]` line ~641).
- **`_extract_prediction_and_probabilities`:** Adds binary `PREDICTION: [0|1]` pattern extraction and `\boxed{0}` / `\boxed{1}` parsing (lines 498-608). This makes it tolerant of models that output a boxed answer rather than probabilities.
- **Driver:** `run_kare_debate_mortality_fast.py` — adds `--precomputed_log_dir` arg and writes results to `KARE/results/` (not `results_unbiased/`).

### 2c. `mortality_debate_rag_binary.py` — Binary Outcome Variant (1783 LOC)

- **Integrator prompt** (`line 202`): Drops probability output, requests:
  ```
  PREDICTION: [0 or 1]
  Where:
  - 0 = Survival (patient will survive the next visit)
  - 1 = Mortality (patient will die during the next visit)
  Note: Output ONLY a binary prediction (0 or 1), not probabilities.
  ```
- **`_extract_prediction_and_probabilities`:** Same extended parser as `_fast` with `\boxed{}` patterns. Still attempts probability extraction but falls through to binary.
- **`_summarize_round_response` stop sequences:** Extended to include conversational loop patterns (`"Good night"`, `"Feel free"`, etc.) not present in canonical.
- **Design intent:** Ablation — removes the probability head to test whether the forced binary output changes calibration.

### 2d. `mortality_debate_rag_unbiased.py` — Bias-Mitigation Variant (1712 LOC)

- **Integrator prompt** (`line 202`): Same single `<search>` tag as `_fast`, but "Mortality is rare" language is retained. Key change: removes the explicit "Be conservative" qualifier from the workflow steps, simplifying the instruction to just "analyze BOTH risky factors AND survival factors."
- **GPU management:** Same guard as canonical (preserves external `CUDA_VISIBLE_DEVICES`).
- **Driver:** `run_kare_debate_mortality.py` — outputs to `results_unbiased/` directory (note the canonical driver also writes to `results_unbiased/` by default, line 537).
- **Design intent:** Attempts to reduce the system's known survival-prediction bias by softening the conservative framing.

### 2e. `mortality_debate_cot.py` — Chain-of-Thought, No RAG (969 LOC)

- **No MedRAG:** Imports only `vllm.LLM` and `SamplingParams`. MedRAG paths are not added to `sys.path`.
- **Optional Fireworks API:** Constructor accepts `use_fireworks: bool` and `fireworks_api_key`. When enabled, agents call `self.fireworks_client.chat.completions.create()` against `https://api.fireworks.ai/inference/v1` (`_call_fireworks_api`, line 192).
- **Tensor parallelism:** All GPUs in `gpu_ids` are given to a single `LLM()` instance (all agents share the same model). No separate integrator GPU.
- **Integrator prompt:** No tool-calling instructions. Just:
  ```
  MORTALITY PROBABILITY: X.XX (0.00 to 1.00)
  SURVIVAL PROBABILITY: X.XX (0.00 to 1.00)
  Note: The two probabilities MUST sum to exactly 1.00
  ```
- **Design intent:** Pure reasoning baseline — no external knowledge, tests whether debate structure alone improves over single-agent.

### 2f. `old_mortality_debate_rag.py` — Legacy Version (1673 LOC)

Structurally nearly identical to the canonical. Differences versus canonical:
- Same single `<search>` tag in integrator prompt (no dual-query).
- Simpler stop sequences in `_summarize_round_response` (no conversational-loop guards).
- Predates the GPU-env guard (uses unconditional `CUDA_VISIBLE_DEVICES` assignment like `_fast`).
- No `_execute_dual_retrieval` method — retrieval is always single-query.

This is a snapshot from before the dual-retrieval upgrade and GPU-env fix were merged. Lower priority for further inspection.

---

## 3. Pipeline: Patient Flow

### 3a. Data Loading

`KAREDataAdapter` (`kare_data_adapter.py`) is the sole entry point for patient data.

Key data files it loads:
- **EHR test set:** `KARE/data/ehr_data/mimic3_mortality_samples_test.json` — MIMIC-III rolling-visit records; fields `patient_id`, `visit_id`, `conditions[][]`, `procedures[][]`, `drugs[][]`, `label`.
- **Similar-patient contexts (preferred):** `KARE/data/patient_context/similar_patient_debate/patient_to_top_1_patient_contexts_mimic3_mortality_improved.json` — precomputed positive/negative similar-patient text blocks keyed by `{patient_id}_{visit_index}`.
- **Fallback similar-patient file:** `KARE/data/patient_context/similar_patient_qwen/patient_to_top_1_patient_contexts_mimic3_mortality.json`

`format_patient_context()` (`kare_data_adapter.py:104`) converts raw arrays into KARE's rolling-visit text format (Visit 0 / Visit 1 / ... with `(new)` and `(continued from previous visit)` annotations).

`get_test_sample()` (`kare_data_adapter.py:187`) returns:
```python
{
  'patient_id':      f"{base_patient_id}_{visit_index}",   # e.g. "10188_1"
  'base_patient_id': ...,
  'visit_id':        ...,
  'visit_index':     ...,
  'target_context':  <formatted EHR text>,
  'positive_similars': <text block of mortality=1 similar patients>,
  'negative_similars': <text block of survival=0 similar patients>,
  'ground_truth':    0 or 1
}
```

### 3b. Contrastive Preprocessing

Before agents run, `preprocess_for_debate()` from `kare_contrastive_preprocessing.py` is called (`debate_mortality_prediction` line ~1704). It:
1. Parses target + positive-similar + negative-similar contexts into per-visit ICD/Procedure/Medication sets.
2. Computes shared vs. unique concept sets for each domain.
3. Returns `analyst1_input` (target + positive-similar formatted with shared/unique markers) and `analyst2_input` (target + negative-similar).

This ensures both analyst agents are label-blind — they see clinical codes, not outcome labels.

### 3c. Round 1 — Parallel Batch Analysis

`_agent_turn_batch(roles=["mortality_risk_assessor", "protective_factor_analyst"], ...)` (`mortality_debate_rag.py:1177`):
- Builds prompts for both agents simultaneously.
- Calls `self.llm.llm.generate(prompts, sampling_params)` — a single VLLM batch call with `temperature=0.3`, `max_tokens=32768`.
- Falls back to sequential `_agent_turn()` if batch fails.
- Returns two `dict` entries with `role`, `message`, `generation_time`, `prompt_length`, `response_length`.

Note: Agents 1 and 2 do **not** use retrieval in Round 1. Retrieval is reserved for the integrator.

### 3d. Round 2 — Integrator with Dual Retrieval

`_agent_turn(role="balanced_clinical_integrator", ...)` dispatches to `_integrator_single_step_prediction()` (`line ~834`), which:

1. Calls `_prepare_integrator_history()` — summarizes analyst outputs if > 24,000 chars (~6,000 tokens), labeling them "Similar Case with Mortality=1 (positive class) Analysis" and "Similar Case with Survival=0 (negative class) Analysis".
2. Builds initial prompt: system prompt + target EHR + labeled history.
3. Generates with `integrator_llm`, stopping only at `<|im_end|>` / `</s>` to allow both `<search_medcorp>` and `<search_umls>` tags to appear.
4. Truncates response after the last closing search tag.
5. `_parse_tool_call()` extracts `medcorp_query` and `umls_query` from XML tags.
6. `_execute_dual_retrieval()` (`line 701`) retrieves:
   - 4 docs from `medcorp` sub-retriever of MedCorp2
   - 4 docs from `umls` sub-retriever of MedCorp2
   - Injects as `<information>...</information>` block.
7. Resumes generation (`max_tokens=32768`, `temperature=0.5`) to produce final analysis + probabilities.
8. `_extract_prediction_and_probabilities()` (`line 487`) regex-extracts:
   ```
   MORTALITY PROBABILITY: X.XX
   SURVIVAL PROBABILITY: X.XX
   ```
   Prediction = 1 if `mortality_prob > survival_prob`, else 0. Conservative default = 0 on parse failure.

Retry logic: up to 2 integrator attempts. If still no probabilities, re-runs Round 1 agents and uses their probabilities as fallback. Last-resort fallback when `ground_truth` is available: predict `1 - ground_truth` (intentional oracle-free deterioration to avoid silent 0s).

### 3e. Output

`debate_mortality_prediction()` returns:
```python
{
  'final_prediction': 0 or 1,
  'final_mortality_probability': float or None,
  'final_survival_probability': float or None,
  'final_confidence': str or None,
  'debate_history': [...],
  'rounds_completed': 2,
  'total_generation_time': float,
  'integrator_prediction': ...,
  ...
}
```

---

## 4. Key Classes and Functions

| Name | File:Line | Role |
|------|-----------|------|
| `MortalityDebateSystem.__init__` | `mortality_debate_rag.py:40` | Initializes MedRAG, dual VLLM instances (main + integrator), retrieval tools |
| `_initialize_agent_prompts` | `mortality_debate_rag.py:158` | Returns dict of 3 system prompt strings |
| `_create_retrieval_tool` | `mortality_debate_rag.py:227` | Factory: returns `{"name":"retrieve","func":retrieve_tool}` closure |
| `_execute_dual_retrieval` | `mortality_debate_rag.py:701` | Routes separate MedCorp + UMLS queries, returns combined doc list |
| `_parse_tool_call` | `mortality_debate_rag.py:606` | Regex parser for `<search_medcorp>`, `<search_umls>`, and legacy `<search>` tags |
| `_execute_integrator_attempt` | `mortality_debate_rag.py:885` | Search-R1 style: generate → inject → continue |
| `_integrator_single_step_prediction` | `mortality_debate_rag.py:834` | Retry wrapper (up to 2 attempts) around `_execute_integrator_attempt` |
| `_agent_turn_batch` | `mortality_debate_rag.py:1177` | Parallel VLLM batch call for Agents 1 & 2 |
| `_agent_turn` | `mortality_debate_rag.py:1355` | Sequential single-agent call; dispatches integrator to its own method |
| `_prepare_integrator_history` | `mortality_debate_rag.py:425` | Summarizes + labels prior-round outputs |
| `_summarize_round_response` | `mortality_debate_rag.py:326` | LLM-based compression if response > 24k chars |
| `_extract_prediction_and_probabilities` | `mortality_debate_rag.py:487` | Regex extraction of `MORTALITY PROBABILITY:` and `SURVIVAL PROBABILITY:` |
| `debate_mortality_prediction` | `mortality_debate_rag.py:1595` | Top-level entry point per patient; orchestrates all rounds + fallback logic |
| `KAREDataAdapter` | `kare_data_adapter.py:12` | Loads MIMIC-III/IV data + similar-patient lookup |
| `KAREDataAdapter.get_test_sample` | `kare_data_adapter.py:187` | Per-index sample formatter |
| `preprocess_for_debate` | `kare_contrastive_preprocessing.py` | Shared/unique concept extraction for label-blind analyst inputs |
| `format_integrator_history_with_labels` | `kare_contrastive_preprocessing.py` | (imported but `_prepare_integrator_history` is used instead in practice) |
| `MortilitySingleAgentRAG` | `mortality_single_agent_rag.py:29` | Single-agent RAG baseline |
| `MortilitySingleAgentCoT` | `mortality_single_agent_cot.py:20` | Single-agent CoT baseline |
| `run_kare_debate_evaluation` | `run_kare_debate_mortality.py:229` | Main loop: iterates test set, calls `debate_system.debate_mortality_prediction()`, saves JSON |
| `calculate_metrics` | `run_kare_debate_mortality.py:24` | Computes accuracy, precision, recall, F1, macro-F1, specificity |

---

## 5. CLI Arguments

### `run_kare_debate_mortality.py` (`main()` line 477)

| Argument | Default | Notes |
|----------|---------|-------|
| `--start_idx` | 0 | Starting sample index |
| `--num_samples` | None (all) | Max samples to process |
| `--output` | auto-generated | If None, auto-builds from model/mode/retriever params |
| `--model` | `Qwen/Qwen3-4B-Instruct-2507` | VLLM model for Agents 1–3 |
| `--integrator_model` | None (= `--model`) | Separate integrator model |
| `--gpus` | `6,7` | Comma-separated GPU IDs |
| `--integrator_gpu` | None (= 2nd in `--gpus`) | GPU(s) for integrator; multiple triggers tensor parallelism |
| `--include_history` | False | Include full debate history in JSON output |
| `--batch_size` | 10 | Intermediate save frequency |
| `--mode` | `rag` | `rag` or `cot` |
| `--use_fireworks` | False | CoT mode only: use Fireworks API |
| `--fireworks_api_key` | None | Falls back to `FIREWORKS_API_KEY` env var |
| `--round1_k` | 8 | Number of docs retrieved in Round 1 (used only for output path naming) |
| `--round3_k` | 8 | Number of docs retrieved in Round 3 (path naming only) |
| `--corpus_name` | `MedCorp2` | MedRAG corpus |
| `--retriever_name` | `MedCPT` | MedRAG retriever |
| `--db_dir` | `/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus` | MedRAG corpus directory |

### `run_kare_debate_mortality_fast.py` — additional arg

| Argument | Default | Notes |
|----------|---------|-------|
| `--precomputed_log_dir` | None | Path to `debate_logs/` from a prior run; triggers fast mode (integrator only) |

Note: The fast driver writes to `results/` while the canonical driver writes to `results_unbiased/` (see auto-path logic at `run_kare_debate_mortality.py:537`).

---

## 6. External Dependencies

### Hard-Coded Paths

```python
# mortality_debate_rag.py:19-21 (identical in all _rag variants)
medrag_root = "/data/wang/junh/githubs/mirage_medrag/MedRAG"
mirage_src  = "/data/wang/junh/githubs/mirage_medrag/MIRAGE/src"

# Default db_dir in __init__ and CLI defaults:
db_dir = "/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus"
```

These paths are hard-coded in every RAG variant. The `_cot.py` variant does not import MedRAG at all.

### Models

Default model: `Qwen/Qwen3-4B-Instruct-2507` (set at `__init__` default arg, line 41).
All VLLM wrappers call `VLLMWrapper(model_name=model_name, enable_thinking=True)` — thinking mode (chain-of-thought tokens in `<think>` tags) is always on.

Earlier runs (visible in `results/` directory names) used `Qwen/Qwen2.5-7B-Instruct`.

Optional: any Fireworks-hosted model via `--use_fireworks` in CoT mode.

### KARE Data Files Under `KARE/data/`

| File | Purpose |
|------|---------|
| `data/ehr_data/mimic3_mortality_samples_test.json` | Primary test set |
| `data/ehr_data/mimic3_mortality_samples_train.json` | Training split (loaded on demand) |
| `data/ehr_data/mimic3_mortality_samples_val.json` | Validation split |
| `data/ehr_data/mimic4_mortality_samples_test.json` | MIMIC-IV test (loaded on demand) |
| `data/patient_context/similar_patient_debate/patient_to_top_1_patient_contexts_mimic3_mortality_improved.json` | Preferred similar-patient lookup (positive/negative split) |
| `data/patient_context/similar_patient_debate/patient_to_top_2_patient_contexts_mimic3_mortality_improved.json` | Top-2 variant |
| `data/patient_context/similar_patient_qwen/patient_to_top_1_patient_contexts_mimic3_mortality.json` | Fallback similar-patient lookup |
| `data/base_context_qwen/patient_embeddings_mimic3_mortality.pkl` | Precomputed Qwen embeddings (used upstream by KARE for similar-patient retrieval) |
| `data/base_context_qwen/patient_contexts_mimic3_mortality.json` | Text contexts for KARE similar-patient search |

---

## 7. Output Artifact Layout

Results land under `KARE/results/` or `KARE/results_unbiased/`. The auto-generated directory name encodes the full experiment configuration:

```
results/
  {mode}_mor_{clean_model_name}[_int_{clean_integrator_name}]_{retriever}_{round1_k}_{round3_k}/
    kare_debate_mortality_results.json
    debate_logs/
      debate_responses_{patient_id}.log
      retrieve_integrator_medcorp_{patient_id}.json
      retrieve_integrator_umls_{patient_id}.json
      retrieve_{role}_{patient_id}.json

results_unbiased/
  {mode}_mor_{clean_model_name}_{retriever}_{round1_k}_{round3_k}/
    kare_debate_mortality_results.json
    debate_logs/
      ...
```

**Naming convention examples (on disk):**
- `rag_mor_Qwen_Qwen2.5_7B_Instruct_MedCPT_8_8` — RAG mode, Qwen2.5-7B, MedCPT retriever, k=8 both rounds
- `cot_mor_Qwen_Qwen2.5_7B_Instruct` — CoT mode, no retrieval params
- `rag_mor_Qwen_Qwen2.5_7B_Instruct_int__data_wang_junh_githubs_Debate_KARE_searchr1_checkpoints_searchr1_binary_single_agent_step100_MedCPT_8_8` — separate integrator model (local fine-tuned checkpoint)
- `single_rag_mor_Qwen_Qwen2.5_7B_Instruct_MedCPT_zero_shot` — single-agent RAG baseline

The `results_unbiased/` root is used when `run_kare_debate_mortality.py` (canonical driver) auto-generates the path (`line 537`). The `results/` root is used by `run_kare_debate_mortality_fast.py` (`line 548`).

`kare_debate_mortality_results.json` structure:
```json
{
  "metadata": {"timestamp": ..., "total_samples": ..., "include_debate_history": ...},
  "metrics": {"accuracy": ..., "precision": ..., "recall": ..., "f1_score": ...,
               "macro_f1": ..., "f1_mortality": ..., "f1_survival": ...,
               "specificity": ..., "tp": ..., "fp": ..., "fn": ..., "tn": ...,
               "pred_distribution": {"0": ..., "1": ...}, "gt_distribution": ...},
  "results": [{"patient_id": ..., "visit_id": ..., "ground_truth": ...,
                "prediction": ..., "rounds_completed": 2, "total_generation_time": ...,
                "error": null}]
}
```

---

## 8. Notable Quirks, Dead Code, and Single-Agent Baselines

### Quirks

- **`format_integrator_history_with_labels`** is imported from `kare_contrastive_preprocessing.py` in all RAG variants but never called — `_prepare_integrator_history()` is used instead. The import is dead code.

- **Fallback: predict opposite of ground truth.** When the integrator fails to produce any probability after all retries (`debate_mortality_prediction` line ~1820), the code sets `final_prediction = 1 - ground_truth`. This guarantees a wrong prediction and was clearly a debugging scaffold that was never removed. It means the metrics will look systematically degraded for any patient where the integrator silently fails.

- **`round1_k` / `round3_k` CLI args** do not actually configure retrieval depth — both are hard-coded to k=8 inside `MortalityDebateSystem.__init__` (`self.retrieval_tools = {"round1": self._create_retrieval_tool(k=8), ...}`). The CLI params are only used to name the output directory.

- **`_fast.py` overwrites `CUDA_VISIBLE_DEVICES` unconditionally.** If the caller set `CUDA_VISIBLE_DEVICES=0,1` and passes `--gpus 6,7`, the wrapper sets logical GPUs 0 and 1 but the env var is overwritten to `6,7`, which may conflict with external GPU mapping. The canonical version added `original_cuda_visible_devices` guard to fix this.

- **Analysts don't use retrieval.** Despite `self.retrieval_tools` being initialized with three round keys (`round1`, `round2`, `round3`), only the integrator actually calls retrieval. Round 1 and 2 keys are unused.

- **`_binary.py` `_parse_tool_call` still checks for UMLS/MedCorp tags** but the integrator prompt only advertises the generic `<search>` tag. The dual-query detection code is unreachable for this variant.

- **GPU logic in `_cot.py`** is unusually verbose (steps 1–12 with nvidia-smi subprocess calls). This was clearly debugging scaffolding that was never stripped.

### Single-Agent Baselines

Two single-agent baselines exist for comparison against the debate system:

- **`mortality_single_agent_rag.py`** (`MortilitySingleAgentRAG`): Single agent with MedRAG retrieval. Supports `zero-shot` (no similar patients in context) and `few-shot` (includes similar patients). Uses a 3-step flow: initial query → retrieve (k=8) → final prediction. Shares the same MedRAG initialization pattern as the debate system.

- **`mortality_single_agent_cot.py`** (`MortilitySingleAgentCoT`): Single agent, no retrieval. CoT only. Uses KARE's original `task_description` prompt (the one from the KARE repo that focuses on conditions, procedures, medications severity) rather than the debate-system prompts.

Both baselines are driven by `run_kare_single_agent_experiments.py` and `run_single_agent_experiments.sh`. Their output goes to `results/single_rag_mor_*/` and `results/single_cot_mor_*/` directories. They exist as ablation points: single-agent-RAG isolates the value of retrieval, single-agent-CoT isolates the value of debate structure, and the full debate system combines both.
