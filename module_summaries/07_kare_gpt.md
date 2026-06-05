# Module Summary: KARE/gpt — GPT-Based Condition Experiments + Bias Analysis

**Module location:** `/data/wang/junh/githubs/Debate/KARE/gpt/`
**Experiment date:** January 2026
**Author's research question:** Are failures of the multi-agent debate framework due to
**(1) weak reasoning** in Qwen 2.5-7B-Instruct, or **(2) poor retrieval quality** from MedCPT/MedRAG?

---

## 1. What Are "Conditions A–F"? — The Ablation Matrix

The experiment is a factorial swap design with three pipeline positions:
`Analysts | Retrieval | Integrator`

Each slot is filled by either **GPT** (OpenAI gpt-4o) or **Qwen** (Qwen2.5-7B-Instruct via vLLM).
Condition A is the all-GPT ceiling; every other condition replaces exactly one or two slots with Qwen.

### 1.1 Condition Descriptions

| Cond | Label | Analysts | Retrieval | Integrator | Key Question |
|------|-------|----------|-----------|------------|--------------|
| **Baseline** | Qwen+Qwen+Qwen | Qwen | Qwen | Qwen | Existing KARE result |
| **A** | GPT+GPT+GPT | GPT | GPT | GPT | Ceiling — all GPT |
| **B** | GPT+Qwen+GPT | GPT | Qwen | GPT | Is Qwen retrieval the bottleneck? |
| **C** | Qwen+GPT+GPT | Qwen | GPT | GPT | Are Qwen analysts the bottleneck? |
| **D** | GPT+GPT+Qwen | GPT | GPT | Qwen | Is Qwen integrator the bottleneck? |
| **E** | GPT+Qwen+Qwen | GPT | Qwen | Qwen | GPT analysts + Qwen back-end |
| **F** | Qwen+GPT+Qwen | Qwen | GPT | Qwen | GPT retrieval + Qwen front/back |

Reference: `README.md:43-59` (4-condition table) and `analyze_conditions.py:181-241` (extended to 6 conditions).

### 1.2 Per-Condition Delta Table

| Cond | Script | Source of analysts | Source of retrieval | Source of integrator | Notes |
|------|--------|--------------------|---------------------|----------------------|-------|
| A | `run_condition_A.py` | GPT (live call) | GPT (live MedRAG call) | GPT (2-turn: initial search + final) | Analysts also get MedRAG injection; integrator does dynamic `<search>` |
| B | `run_condition_B.py` | Reused from A logs | Qwen log bundle (query + docs) | GPT (no-search variant) | Qwen retrieval loaded from `debate_responses_{id}.log`; integrator has no `<search>` capability |
| C | `run_condition_C.py` | Qwen log bundle | Reused from A logs | GPT (no-search variant) | Qwen analyst outputs parsed from existing debate log files |
| D | `run_condition_D.py` | Reused from A logs | Reused from A logs | Qwen via vLLM | Only new inference is Qwen integrator; vLLM loaded with `gpu_memory_utilization=0.85` |
| E | `run_condition_E.py` | Reused from A logs | Qwen log bundle | Qwen via vLLM | Imports from `gpt_utils_bias` (bias-corrected prompts) |
| F | `run_condition_F.py` | Qwen log bundle | Reused from C logs | Qwen via vLLM | Imports from `gpt_utils_bias`; integrator prompt strips the `<search>` tool description |

**Key implementation details that differ:**
- Condition A integrator runs two turns: an "initial" turn where it may emit `<search>query</search>`, then a "final" turn with `<information>retrieved docs</information>` injected. (`run_condition_A.py:297-334`)
- Conditions B, C, D, E, F integrators run a single "no-search" turn with pre-populated retrieved text. (`gpt_utils.py:510-586`)
- Condition A analysts receive up to 8 retrieved snippets (400-char truncated) injected into their prompt. (`run_condition_A.py:68-103`)
- Conditions B–F analysts reuse cached outputs — no new GPT analyst inference for conditions after A.
- Conditions E and F import from `gpt_utils_bias` instead of `gpt_utils` — this changes the integrator system prompt (see Section 3).

### 1.3 Reuse Dependencies

```
Condition A (live)
    |-- Analysts + retrieval cached in results_bias/condition_A_gpt_4o/logs/{id}.json
    |
    +-- Condition B (reuses A analysts, adds Qwen retrieval)
    +-- Condition C (reuses A retrieval, adds Qwen analysts)
    |       |
    |       +-- Condition F (reuses C retrieval, adds Qwen integrator)
    +-- Condition D (reuses A analysts + retrieval, adds Qwen integrator)
    +-- Condition E (reuses A analysts, adds Qwen retrieval + Qwen integrator)
```

---

## 2. GPT Model(s) Used

### 2.1 Model Names

From `gpt_utils.py:52-58` and `gpt_utils_bias.py:52-58`:

```python
MAX_COMPLETION_TOKENS = {
    "gpt-4-turbo-preview": 4096,
    "gpt-4": 8192,
    "gpt-4o": 16384,
    "gpt-4o-mini": 16384,
    "o3-mini": 100000,
    "gpt-3.5-turbo": 4096,
}
NEW_API_MODELS = ["o3-mini", "o1-preview", "o1-mini"]
```

The **default model** across all condition scripts is `gpt-4o` (`run_condition_A.py:369`, `run_condition_B.py:188`). The `README.md:74` cites `gpt-4-turbo-preview` as the test-run model. Actual results directories are named `condition_A_gpt_4o`, confirming gpt-4o was the final run model.

### 2.2 API Setup

- Client: `openai.OpenAI` (Python `openai` library). (`gpt_utils.py:84`)
- API key: read from `OPENAI_API_KEY` environment variable, or via `--api_key` CLI argument. (`gpt_utils.py:75-77`)
- Max tokens for generation: 32768 (analyst and integrator calls). (`run_condition_A.py:106`)
- Temperature: 0.7 for all calls. (`gpt_utils.py:87`)
- Rate limiting: 1-second sleep between samples. (`run_condition_A.py:441`)
- For `o3-mini`/`o1` models: uses `max_completion_tokens` instead of `max_tokens`. (`gpt_utils.py:105-113`)

### 2.3 MedRAG Placeholder

MedRAG is initialized with `llm_name="OpenAI/gpt-3.5-turbo-16k"` as a placeholder — this field is not actually used for retrieval; only the retriever component (MedCPT, HNSW index) is active. (`gpt_utils.py:238-240`)

---

## 3. The Bias Track

### 3.1 What "Bias" Means Here

The bias being studied is **survival bias** — the tendency of LLM-based integrators to always predict survival (label=0) and avoid predicting mortality (label=1). This was identified as the primary failure mode in both the Qwen baseline and the initial GPT experiments.

Evidence: Qwen baseline on 100 samples: recall=22.2%, 42 false negatives out of 54 positive cases (`results_bias/ablation_analysis.json:8-15`). Condition A (GPT+GPT+GPT with standard prompts): recall dropped further to 3.7% (`ablation_analysis.json:22-26`).

### 3.2 The Two Prompt Variants

`gpt_utils.py` (standard / "unbiased" track) integrator IMPORTANT clause (`gpt_utils.py:175`):
```
Mortality is rare. Only assign a high mortality probability when the patient appears at
extremely high risk of death with strong evidence.
```

`gpt_utils_bias.py` (bias-corrected track) integrator IMPORTANT clause (`gpt_utils_bias.py:175`):
```
Mortality is rare - only predict mortality probability > 0.5 if evidence STRONGLY supports it.
When uncertain, predict survival probability > 0.5.
```

Additional difference in step 3 of the workflow (`gpt_utils_bias.py:183`):
```
Be conservative: mortality is rare, so strong evidence is needed for high mortality probability.
```

This is an **explicit survival-bias reinforcement** prompt: the bias-corrected version makes the model's default-to-survival behavior explicit as an instruction, with the paradoxical goal of measuring whether framing this explicitly changes behavior (or makes it worse). The `no_search` variant receives the same change.

The four differences are purely in the `AGENT_PROMPTS` dictionary values — the Python API call logic, retrieval logic, and parsing logic are identical across both files.

### 3.3 Results Track Separation

- `results/` — runs using `gpt_utils.py` (standard prompts): only `condition_A_gpt_4o/` exists, plus `biased_qwen/` and `unbiased_qwen/` reference directories.
- `results_bias/` — runs using `gpt_utils_bias.py` (bias-corrected prompts): all six conditions A–F plus test runs and the `ablation_analysis.json`.

The main analysis (`analyze_conditions.py`) defaults to `results_bias/` directories (`analyze_conditions.py:479-492`).

### 3.4 Qwen Reference Conditions

Under `results/`:
- `biased_qwen/` — Qwen baseline with the original (biased) prompts
- `unbiased_qwen/` — Qwen baseline with unbiased/modified prompts (reference for comparison)

Under `results_bias/`:
- `analyst_no_rag/` — analysts run without retrieval injection
- `tool_call/` — integrator tool-call behavior analysis

---

## 4. Single-Condition Pipeline

### 4.1 Pipeline Overview

```
Step 0: extract_metadata.py     → cache/candidate_table.parquet
Step 1: sample_select.py        → manifests/samples_swap_core.csv
                                → manifests/selected_samples_full.parquet
                                → manifests/samples_swap_core_metadata.json
Step 2: inspect_samples.py      → manifests/gpt_experiment_samples.json
Step 3: run_condition_X.py      → results_bias/condition_X_*/logs/{id}.json
                                → results_bias/condition_X_*/results.json
Step 4: analyze_conditions.py   → results/ablation_analysis.json
```

### 4.2 Step 0: `extract_metadata.py`

**Inputs:**
- KARE test dataset via `KAREDataAdapter` (`/data/wang/junh/githubs/Debate/KARE/kare_data_adapter.py`) — 996 samples
- Three result JSONs: `results/cot_mor_Qwen_*/kare_debate_mortality_results.json`, `results/rag_mor_Qwen_*_MedCPT_8_8/kare_debate_mortality_results.json`, `results/rag_mor_Qwen_*_searchr1_*/kare_debate_mortality_results.json`
- Debate log directories (for retrieval detection): `debate_responses_{id}.log` files

**Processing** (`extract_metadata.py:118-267`):
- Per sample: extracts predictions from CoT, RAG-Qwen, RAG-R1 runs
- Computes `wrong_multi_cot`, `wrong_multi_rag_qwen`, `wrong_multi_rag_r1` boolean flags
- Computes `wrong_count_3` (0–3: how many of the three runs were wrong)
- Detects retrieval: checks for `<search>` + `<information>` tags in log files, or existence of `retrieve_integrator_combined_{id}.json`
- Computes `prompt_len_tokens_target` (len(text)//4), `rag_called_both`

**Output:** `cache/candidate_table.parquet` — one row per sample, 996 rows.

### 4.3 Step 1: `sample_select.py`

**Input:** `cache/candidate_table.parquet`

**Selection algorithm** (`sample_select.py:51-143`):
- **54 positives** (label=1): all of them, `split_tag='pos_all'`
- **46 negatives** (label=0): balanced difficulty
  - 23 "hard" (`split_tag='neg_hard'`): all N3 (wrong in all 3 runs, typically 1 sample) + top N2 (wrong in 2 runs) sorted by prompt length descending
  - 23 "easy" (`split_tag='neg_easy'`): top N0 (correct in all 3 runs) sorted by `rag_called_both` then `prompt_len_tokens_target` descending
- Selection is **fully deterministic** (sorted by stable keys)

**Outputs:**
- `manifests/samples_swap_core.csv` — minimal: `[sample_id, label, split_tag]`
- `manifests/samples_swap_core_metadata.json` — pool sizes and selection rules
- `manifests/selected_samples_full.parquet` — full row for each selected sample (includes `patient_context`, `positive_similars`, `negative_similars`, all flag columns)

### 4.4 Step 2: `inspect_samples.py`

Analyzes and exports the 100-sample slice to `manifests/gpt_experiment_samples.json` (ready-to-use JSON for GPT experiments). Also prints summary statistics about retrieval patterns and difficulty distribution.

### 4.5 Step 3: `run_condition_A.py` (most complex — illustrative)

**Input:** `manifests/selected_samples_full.parquet`

**Per-sample flow:**
1. Skip if `results_bias/condition_A_gpt_4o/logs/{sample_id}.json` already exists (incremental)
2. Run GPT Analyst 1 (`mortality_risk_assessor`) — target vs positive-similar patient, with MedRAG retrieval (8 docs, 400-char truncated)
3. Run GPT Analyst 2 (`protective_factor_analyst`) — target vs negative-similar patient, with MedRAG retrieval
4. Run GPT Integrator (initial turn) — given analyst outputs; may emit `<search>query</search>`
5. If search query found: retrieve 8 documents via MedRAG with GPT's query
6. Run GPT Integrator (final turn) — full context including `<information>docs</information>`
7. Parse `MORTALITY PROBABILITY: X.XX` and `SURVIVAL PROBABILITY: X.XX` from response
8. Predict 1 if mortality_prob > survival_prob, else 0
9. Save per-sample JSON to `logs/` subdirectory; save compact summary to `results.json`

**Fallback** (`run_condition_A.py:347-351`): if prediction parse fails, set prediction = 1 - label (i.e., always wrong), to surface errors cleanly.

**Output per sample** (`logs/{sample_id}.json` keys): `sample_id`, `label`, `gpt_analyst1`, `gpt_analyst1_docs`, `gpt_analyst2`, `gpt_analyst2_docs`, `gpt_integrator_initial`, `gpt_query`, `gpt_docs`, `called_retriever`, `gpt_integrator_final`, `mortality_probability`, `survival_probability`, `prediction`, `error`

### 4.6 MedRAG Retrieval Details

Corpus: `MedCorp2` (dual-source: `medcorp` general literature + `umls` terminology). Retriever: `MedCPT`. HNSW index enabled. For dual-source, splits k=8 as k//2+k%2 from medcorp + k//2 from umls. (`gpt_utils.py:276-298`)

Document path on disk: `/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus`
MedRAG source: `/data/wang/junh/githubs/mirage_medrag/MedRAG/`

### 4.7 Qwen Integrator Setup (Conditions D, E, F)

Uses vLLM directly (`run_condition_D.py:62-69`):
```python
LLM(model=model_name, tensor_parallel_size=1, trust_remote_code=True,
    gpu_memory_utilization=0.85, enforce_eager=True)
```
Sampling: `temperature=0.7, top_p=0.9, max_tokens=4096, stop=["<|im_end|>", "</s>"], repetition_penalty=1.2`
Prompt format: `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`

---

## 5. `analyze_conditions.py` — Metrics and Research Questions

### 5.1 Metrics Computed

For each condition, `calculate_metrics()` (`analyze_conditions.py:66-112`) computes:
- `accuracy` = correct / total valid
- `precision` = TP / (TP + FP) — for mortality class
- `recall` = TP / (TP + FN) — for mortality class
- `f1` = harmonic mean of precision and recall
- Confusion matrix: `tp`, `fp`, `fn`, `tn`
- `total_samples`, `valid_predictions`, `errors`

Metrics are computed at **two levels**:
1. **Overall** (all 100 samples)
2. **By split_tag** (`pos_all`, `neg_hard`, `neg_easy`) — 54 / 23 / 23 samples respectively

### 5.2 Research Questions Answered (`analyze_conditions.py:244-390`)

Seven questions are evaluated with automated interpretation (threshold: >5% gap = "significant"):

| Q# | Question | Comparison |
|----|----------|------------|
| Q1 | Does GPT-4 improve over Qwen baseline? | Cond A vs Baseline |
| Q2 | Is Qwen retrieval limiting performance? | Cond A vs Cond B |
| Q3 | Is Qwen analyst reasoning limiting? | Cond A vs Cond C |
| Q4 | Is Qwen integrator reasoning limiting? | Cond A vs Cond D |
| Q5 | Does GPT retrieval help when Qwen integrates? | Cond D vs Cond E |
| Q6 | Do GPT analysts help when Qwen integrates? | Cond D vs Cond F |
| Q7 | Does GPT integrator help with GPT upstream? | Cond A vs Cond D |

### 5.3 Retrieval Usage Analysis (`analyze_conditions.py:393-468`)

Compares whether Condition A GPT integrator requested retrieval more/less than Qwen baseline:
- Baseline: checks for `retrieve_integrator_combined_{id}.json` or `<search>`+`<information>` in log
- Condition A: checks `gpt_query` field in `logs/{id}.json`

### 5.4 Actual Results (from `results_bias/ablation_analysis.json`)

Overall accuracy on 100 samples:

| Condition | Accuracy | Precision | Recall | F1 |
|-----------|----------|-----------|--------|-----|
| Qwen Baseline | 38% | 37.5% | 22.2% | 27.9% |
| A: GPT+GPT+GPT | 45% | 40.0% | 3.7% | 6.8% |
| B: GPT+Qwen+GPT | 46% | 0% | 0% | 0% |
| C: Qwen+GPT+GPT | 47% | 100% | 1.9% | 3.6% |
| D: GPT+GPT+Qwen | **53%** | 56.4% | **57.4%** | **56.9%** |
| E: GPT+Qwen+Qwen | 47% | 51.5% | 31.5% | 39.1% |
| F: Qwen+GPT+Qwen | 43% | 46.3% | 35.2% | 40.0% |

**Key finding (Q7):** Condition D (Qwen integrator) outperforms Condition A (GPT integrator) by 8 percentage points — the GPT integrator has stronger survival bias and essentially never predicts mortality (recall 3.7%). The Qwen integrator, despite being smaller, produces more calibrated predictions. GPT retrieval helps significantly over Qwen retrieval when Qwen integrates (Q5: D vs E, +6%). GPT analysts help over Qwen analysts under Qwen integrator (Q6: D vs F, +10%).

Retrieval rate: Qwen baseline called retrieval 58/100 times; GPT integrator called retrieval 86/100 times.

---

## 6. Result Folder Naming Conventions

### 6.1 Under `results/` (standard / non-bias-corrected prompts)

```
results/
├── condition_A_gpt_4o/       # gpt-4o, standard prompts
│   ├── logs/                 # {sample_id}.json per sample
│   └── results.json          # compact summary
├── biased_qwen/              # Qwen baseline reference (original biased prompts)
└── unbiased_qwen/            # Qwen baseline reference (modified prompts)
```

Pattern: `condition_{LETTER}_{model_safe_name}` where model safe name replaces `-` and `.` with `_`.
Generated at runtime: `args.output_dir = f"results/condition_{X}_{model_name.replace('-','_')}"` (`run_condition_A.py:386`).

### 6.2 Under `results_bias/` (bias-corrected prompts — main analysis track)

```
results_bias/
├── condition_A_gpt_4o/            # GPT+GPT+GPT, bias-corrected
├── condition_B_gpt_4o/            # GPT+Qwen+GPT, bias-corrected
├── condition_C_gpt_4o/            # Qwen+GPT+GPT, bias-corrected
├── condition_D_qwen/              # GPT+GPT+Qwen, bias-corrected
├── condition_E_gpt_qwen_qwen/     # GPT+Qwen+Qwen, bias-corrected
├── condition_F_qwen_gpt_qwen/     # Qwen+GPT+Qwen, bias-corrected
├── test_condition_A_gpt_4o/       # Test run (3 samples)
├── test_condition_A_gpt_4o_mini/  # Test run with gpt-4o-mini
├── test_condition_B/              # Test run
├── test_condition_C/              # Test run
├── analyst_no_rag/                # Analysts without RAG injection
├── tool_call/                     # Tool-call behavior analysis
├── filtered_patient_ids/          # Patient-level filtering analysis
└── ablation_analysis.json         # Master comparison output
```

Each condition directory contains:
- `logs/{sample_id}.json` — full per-sample result including raw model outputs
- `results.json` — compact summary with predictions + aggregate metrics

---

## 7. Relationship to the Rest of the Debate Repo

### 7.1 This Module is an Independent GPT-Based Comparison

The GPT conditions do **not** call into KARE debate code at runtime. They replicate the debate architecture independently using GPT API calls and cached Qwen log files.

**Imports from repo** (runtime):
- `from kare_data_adapter import KAREDataAdapter` — used in `extract_metadata.py` and `run_condition_*.py` headers; this is `KARE/kare_data_adapter.py` which loads the KARE test dataset (EHR patient data, similar patient indices).
- `from gpt_utils import ...` / `from gpt_utils_bias import ...` — local to this module.

**No runtime imports from:**
- `mortality_debate_rag.py` (the main Qwen debate script)
- `mortality_single_agent_cot.py`
- Any MedRAG LLM inference (only the retriever is used)

**Data dependencies on existing Qwen runs** (read-only):
- Conditions B, C, E, F read Qwen debate logs: `KARE/results/rag_mor_Qwen_*/debate_logs/debate_responses_{id}.log` — to extract Qwen analyst outputs and retrieval bundles.
- `extract_metadata.py` reads `KARE/results/*/kare_debate_mortality_results.json`.

### 7.2 Prompt Fidelity

System prompts (`AGENT_PROMPTS` dict) are described as "EXACT SAME PROMPTS AS QWEN SYSTEM (from mortality_debate_rag.py)" (`gpt_utils.py:131`). The integrator template structure and document formatting format (`[Document {rank}, Score: {score:.3f}]\n{content}`) are kept identical to the Qwen system for fair comparison (`README.md:143-160`).

---

## 8. Dependencies

### 8.1 Python Libraries

```
openai         # GPT API (OpenAI Python SDK v1+)
vllm           # Qwen inference (Conditions D, E, F only)
pandas         # Data manipulation
pyarrow        # Parquet I/O
tqdm           # Progress bars
```

### 8.2 Environment Variables

- `OPENAI_API_KEY` — required for all GPT conditions (A, B, C). Read at `gpt_utils.py:76`.
- `CUDA_VISIBLE_DEVICES` — set by Qwen integrator scripts if not already set externally. (`run_condition_D.py:55-62`)

### 8.3 External Data Paths (Hardcoded)

| Path | Purpose |
|------|---------|
| `/data/wang/junh/githubs/mirage_medrag/MedRAG/` | MedRAG source root |
| `/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus` | MedCPT corpus / HNSW index |
| `KARE/results/rag_mor_Qwen_*/debate_logs/` | Qwen debate logs (for B, C, E, F) |

### 8.4 No Anthropic API

This module uses **only the OpenAI API**. No Anthropic/Claude SDK is imported anywhere. The `gpt_utils.py` comment about `gpt-3.5-turbo-16k` in MedRAG is a non-functional placeholder (`gpt_utils.py:238`).

---

## 9. File Reference Index

| File | Purpose | Key line ranges |
|------|---------|-----------------|
| `src/gpt_utils.py` | GPT client, AGENT_PROMPTS, MedRAG helpers, probability parser | L48-128 (client), L131-204 (prompts), L207-256 (MedRAG init), L373-437 (probability extraction) |
| `src/gpt_utils_bias.py` | Identical to gpt_utils.py except 4 lines in AGENT_PROMPTS are bias-corrected | L175, L183, L193, L197 |
| `src/run_condition_A.py` | All-GPT ceiling condition, live MedRAG retrieval | L41-111 (analyst), L114-231 (integrator 2-turn), L234-353 (sample loop) |
| `src/run_condition_B.py` | Swaps retrieval to Qwen logs | L40-64 (load Qwen bundle), L67-167 (sample loop) |
| `src/run_condition_C.py` | Swaps analysts to Qwen logs | L39-61 (load Qwen analysts), L64-184 (sample loop) |
| `src/run_condition_D.py` | Swaps integrator to Qwen vLLM | L45-92 (QwenIntegrator class), L99-188 (run function) |
| `src/run_condition_E.py` | GPT analysts + Qwen retrieval + Qwen integrator (bias track) | L37 (imports gpt_utils_bias) |
| `src/run_condition_F.py` | Qwen analysts + GPT retrieval + Qwen integrator (bias track) | L126-128 (modified system prompt) |
| `src/extract_metadata.py` | Build candidate_table.parquet from 996 test samples | L118-267 |
| `src/sample_select.py` | Select 100-sample diagnostic slice | L22-221 |
| `src/analyze_conditions.py` | Metrics, 7 research questions, retrieval analysis | L66-112 (metrics), L181-241 (compare all 7 conditions), L244-390 (Q1-Q7), L393-468 (retrieval analysis) |
| `src/inspect_samples.py` | Export selected samples to JSON | |
| `src/test_run.py` | 3-sample smoke test + cost estimation | |
| `manifests/selected_samples_full.parquet` | 100 selected samples with full EHR context | |
| `cache/candidate_table.parquet` | 996-sample metadata table | |
| `results_bias/ablation_analysis.json` | Master comparison output (all 7 conditions x 3 split types) | |
| `README.md` | Main documentation | |
| `QUICKSTART.md` | Usage guide with cost estimates | |
