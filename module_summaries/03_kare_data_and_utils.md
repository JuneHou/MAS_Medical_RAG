# Module 03: KARE Data Adaptation, Preprocessing, Retrieval Utilities, and Analysis Scripts

**Module path**: `/data/wang/junh/githubs/Debate/KARE/`

---

## 1. `kare_data_adapter.py` — EHR Loading and Context Formatting

### Purpose

`KAREDataAdapter` is the single gateway through which all experiment scripts load patient records and their precomputed similar-patient contexts.  It bridges two raw formats—KARE's flat-JSON EHR representation and the KARE temporal-rolling-visit convention—into the dict structure consumed by the debate and single-agent scripts.

### Key class: `KAREDataAdapter` (file:12)

**Constructor** (`file:18`):

```python
def __init__(self, base_path: str = "./data", split: str = "test")
```

Loads three things on construction:

1. The EHR split file (`mimic3_mortality_samples_{train,val,test}.json`) into `self.data` / `self.test_data`.
2. The precomputed similar-patient lookup into `self.similar_patients` via `_load_similar_patients()` (file:83).  The preferred file is:
   `data/patient_context/similar_patient_debate/patient_to_top_1_patient_contexts_mimic3_mortality_improved.json`
   with a fallback to the original `similar_patient_qwen/` file.
3. Prints load counts for both.

### KARE patient-ID convention (file:202–214)

Each EHR JSON record carries a `patient_id` (integer, e.g. `10004`) and a `visit_id`.  KARE generates one *prediction instance* per visit using a rolling-window context.  The temporal ID is:

```
kare_patient_id = f"{patient_id}_{visit_index}"   # e.g. "10004_2"
```

where `visit_index = (number_of_visits_in_record) - 1`.  This is the key used to look up similar-patient entries in `self.similar_patients`.

### `format_patient_context(patient_data)` (file:104)

Converts a raw EHR dict into the human-readable rolling-visit string that all LLM prompts use.  Conditions/procedures/drugs are annotated with `"(new)"` or `"(continued from previous visit)"` for visits after visit 0.  Output format:

```
Patient ID: <kare_patient_id>

Visit 0:
Conditions:
1. <concept>
...
Procedures:
...
Medications:
...

Visit 1:
...
```

### `get_test_sample(index)` (file:187) — primary consumer API

Returns a dict consumed by all experiment entry-points:

```python
{
    'patient_id':       str,   # e.g. "10004_2"
    'base_patient_id':  str,   # e.g. "10004"
    'visit_id':         str,
    'visit_index':      int,
    'target_context':   str,   # output of format_patient_context()
    'positive_similars':str,   # formatted context(s) of similar patients who died
    'negative_similars':str,   # formatted context(s) of similar patients who survived
    'ground_truth':     int,   # 0 = survival, 1 = mortality
    'original_data':    dict   # raw EHR record
}
```

`positive_similars` and `negative_similars` are plain strings assembled by joining pre-rendered context blobs from the similar-patient JSON, separated by `"\n\n"`.  If the lookup returns `"None"` for either polarity, the field is set to `"No positive/negative similar patients available."`.

### Other public methods

| Method | Purpose |
|---|---|
| `get_batch_samples(start, batch_size)` (file:256) | Batched wrapper around `get_test_sample` |
| `get_task_description()` (file:270) | Returns the mortality-prediction system prompt shared across all baselines |
| `format_as_retrieval_query(context, ground_truth)` (file:298) | Converts numeric labels to text (`0 → "survive"`, `1 → "mortality"`) for use as retrieval queries |
| `get_statistics()` (file:331) | Returns total sample count, label distribution, and similar-context coverage fraction |

### How the rest of the codebase consumes it

`mortality_debate_rag.py`, `mortality_single_agent_rag.py`, `mortality_single_agent_cot.py`, and the preprocessing script all import `KAREDataAdapter`, call `get_test_sample(idx)`, and pass `target_context`, `positive_similars`, and `negative_similars` downstream.

---

## 2. `kare_contrastive_preprocessing.py` — Label-Blind Contrastive Formatting

### What "contrastive" means here

In the debate architecture, two **analyst agents** each examine the target patient alongside one of the similar patients (one who died, one who survived).  A naive presentation would include the outcome label directly, letting the analyst simply copy it.  This module instead reformats both patient records into a **shared / unique** split so each analyst sees *clinical patterns* without seeing the outcome label.  Labels are re-injected only when the integrator receives the analysts' conclusions (`format_integrator_history_with_labels`, file:289).

### Processing pipeline

```
target_context (str)   positive_similar (str)   negative_similar (str)
        │                      │                        │
        └──────────────────────┼────────────────────────┘
                               ▼
              parse_patient_context()  ×3  (file:35)
              → {visit_num: {ICD:[...], Procedure:[...], Medication:[...]}}
                               │
                  ┌────────────┴─────────────┐
                  ▼                           ▼
   build_contrastive_visit_view(             build_contrastive_visit_view(
     target_visits, positive_visits)           target_visits, negative_visits)
   (file:126)                               (file:126)
   → (target_pos_view, positive_view)       → (target_neg_view, negative_view)
                  │                                      │
                  └───────────────┬──────────────────────┘
                                  ▼
             build_label_blind_analyst_inputs()  (file:225)
             → analyst1_input, analyst2_input
```

### Key functions

**`normalize_concept(line)`** (file:11): Strips numbering prefix (`"1. "`) and trailing annotations (`"(new)"`, `"(continued ...)"`) to obtain a bare concept string for set comparison.

**`parse_patient_context(text)`** (file:35): Parses the formatted rolling-visit string back into `{visit_num: {"ICD": [...], "Procedure": [...], "Medication": [...]}}`.  Recognizes `"Visit N:"`, `"Conditions:"`, `"Procedures:"`, `"Medications:"` headers.

**`union_order(target_list, similar_list)`** (file:94): Creates a stable union ordering (target items first, then similar-only items) so that the same concept always appears at the same index in both the target and similar views.

**`build_contrastive_visit_view(target_visits, similar_visits)`** (file:126): Iterates visits up to the maximum across both patients.  For each visit and sector, classifies each concept as *shared* (in both), *unique to target*, or *unique to similar*.  Returns two parallel text strings — the target view and the similar view — using this structure:

```
Visit N:
Conditions:
  Shared with Similar:
    1. <concept>
  Unique to Target:
    1. <concept>
```

**`build_label_blind_analyst_inputs(target, positive, negative)`** (file:225): Calls `build_contrastive_visit_view` twice and wraps the output in identical analyst instructions that explicitly forbid outcome speculation.

**`preprocess_for_debate(target_context, positive_similars, negative_similars)`** (file:320): Top-level function.  Splits multiple similar patients (separated by `"\n\n\n"`), takes the first from each polarity, runs the full pipeline, and returns:

```python
{
    'analyst1_input':        str,  # label-blind, analyzing mortality=1 case
    'analyst2_input':        str,  # label-blind, analyzing survival=0 case
    'original_target':       str,
    'positive_similar_raw':  str,
    'negative_similar_raw':  str
}
```

**`format_integrator_history_with_labels(a1, a2)`** (file:289): Called *after* analyst responses are collected; adds outcome headings (`"Similar Case with Mortality=1"` / `"Similar Case with Survival=0"`) so the integrator can reason about which pattern correlates with which outcome.

---

## 3. `improved_faiss_retrieval.py` — FAISS Similar-Patient Index Builder

### What it improves over the default path

The default KARE pipeline (`sim_patient_ret_faiss.py`, in the upstream KARE repo) searched only a small neighbourhood (k=100) and did not enforce a balanced positive/negative split, leaving many patients with no negative example (especially when the dataset is heavily skewed toward survival).  This script raises `MIN_SEARCH_NEIGHBORS` to 1000 and explicitly partitions retrieved candidates by label before selecting `MAX_K=1` from each polarity.

### Pipeline (`main()`, file:140)

1. **Load** `patient_contexts_mimic3_mortality.json` (rendered context strings), `pateint_mimic3_mortality.json` (label lookup), and `patient_embeddings_mimic3_mortality.pkl` (Qwen-encoded float arrays).
2. **Build FAISS index** (`build_faiss_index`, file:47): stacks embeddings into a float32 matrix, L2-normalises (so inner-product = cosine similarity), and loads into `faiss.IndexFlatIP`.
3. **For each patient** (`find_similar_patients`, file:66):
   - Searches up to `min(N-1, 1000)` nearest neighbours.
   - Excludes self and other temporal instances of the same base patient.
   - Filters out contexts longer than 30 000 characters.
   - Splits candidates by `label == target_label` (positive) vs. not (negative).
   - Takes top-1 from each sorted list; appends `"\n\nLabel:\n{label}\n\n"` to each context string.
   - Returns `{'positive': [str | "None"], 'negative': [str | "None"]}`.
4. **Saves** to `data/patient_context/similar_patient_debate/patient_to_top_1_patient_contexts_mimic3_mortality_improved.json` (the file that `KAREDataAdapter._load_similar_patients()` prefers).

### Configuration constants (file:15–18)

| Constant | Value | Effect |
|---|---|---|
| `MAX_K` | 1 | Similar patients per polarity |
| `MAX_CONTEXT_LENGTH` | 30 000 chars | Context length filter |
| `MIN_SEARCH_NEIGHBORS` | 1 000 | Neighbourhood size |

### When/where invoked

Run once as a preprocessing step before any experiment.  Not imported by any experiment script; its output JSON is consumed by `KAREDataAdapter`.

---

## 4. Analysis and Utility Scripts

### 4.1 `analyze_retrieve_rate.py` (224 LOC)

**Input**: A completed experiment result directory containing `kare_debate_mortality_results.json` (the aggregated results file) and a `debate_logs/` subdirectory with per-patient `.log` files and per-patient retrieve JSON files.

**What it computes**: For each patient in the results JSON it (a) looks for the pattern `PARSED TOOL CALL: tool='...', query='...'` in the `.log` file to determine whether the integrator generated a retrieval query, and (b) reads retrieval score fields from either a combined `retrieve_integrator_combined_<pid>.json` or from separate `retrieve_integrator_medcorp_balanced_...` / `retrieve_integrator_umls_balanced_...` files, collecting up to 8 document scores.  It reports the query-generation rate (percentage of patients for which a query was actually issued) to stdout.

**Output**: A CSV file (one row per patient) with columns `patient_id`, `ground_truth`, `prediction`, `query`, `retrieve_score_1` … `retrieve_score_8`.  The `main()` function (file:197) is hardcoded to the `results_unbiased/` directory and analyses two named subdirectories: one for the SearchR1 checkpoint and one for the Qwen 2.5-7B baseline.  Per the accompanying `retrieval_analysis.md`, observed retrieval rates were ~21% for SearchR1 and ~10.5% for Qwen2.5-7B.

---

### 4.2 `analyze_fallback_predictions.py` (416 LOC)

**Input**: A `debate_logs/` directory with `debate_responses_<patient_id>.log` files, and optionally a `kare_debate_mortality_results.json` for final predictions.

**What it computes**: Reads each log file and extracts, using regex:
- `EXTRACTED MORTALITY PROBABILITY: <float>` (last occurrence, after any retry)
- `EXTRACTED SURVIVAL PROBABILITY: <float>` (last occurrence)
- `MANUAL FINAL PREDICTION: <0|1>` (last occurrence)

A prediction is classified as **real** if both probability fields were successfully parsed (suggesting the integrator output a properly structured response), or **fallback** if either is missing (suggesting the model did not generate a parseable probability block, and the system fell back to a heuristic).  The script then computes and prints:

- Summary counts: real vs. fallback, further broken down by which probability was absent.
- Confusion matrix and Precision/Recall/Specificity/F1 for real predictions only.
- An "adjusted" confusion matrix that penalises fallback predictions as errors (fallback TPs counted as FNs, fallback TNs counted as FPs).

**Output**: `../prediction_analysis.csv` (relative to `log_dir`) with per-patient fields `patient_id`, `ground_truth`, `manual_prediction`, `mortality_prob`, `survival_prob`, `has_valid_mortality`, `has_valid_survival`, `is_fallback`, `correct`.

---

### 4.3 `fix_ground_truth_labels.py` (295 LOC)

**Problem it solves**: Early experiments saved result files in which the `ground_truth` field was derived from an intermediate representation that used a different label encoding or patient-ID mapping, causing mismatches with the canonical MIMIC-III labels.

**Input**: `data/ehr_data/mimic3_mortality_samples_test.json` (source of truth for labels) and one or more `kare_debate_mortality_results.json` files found by walking a results directory.

**What it does**:
1. `load_correct_labels(test_data_path)` (file:14): Reconstructs the `kare_patient_id → label` mapping using the same `f"{patient_id}_{visit_index}"` formula as `KAREDataAdapter`, reading the `"label"` field (not `"labels"`).
2. `fix_result_file(result_file_path, correct_labels)` (file:60): Creates a `.backup` copy, iterates `result_data["results"]`, overwrites `ground_truth` wherever it differs from the canonical label, then calls `calculate_metrics()` (file:138) to recompute and store updated metric fields.  Records `metadata.ground_truth_fixed = True` and `metadata.changes_made`.

**Output**: Overwrites the result JSON in place (with backup).

**CLI flags** (file:247): `--test_data`, `--result_file`, `--results_dir`, `--no_backup`.

---

### 4.4 `update_results_files.py` (91 LOC)

**Purpose**: A one-time migration script to normalise single-agent results files written by earlier code to the canonical KARE result schema.

**What it does** (file:11): Reads each `results.json`, strips probability fields from individual result records (keeping only `patient_id`, `visit_id`, `ground_truth`, `prediction`, `is_fallback`, `total_generation_time`), computes `pred_distribution` and `gt_distribution` (count of 0s and 1s in predictions and ground-truths), and appends them to the `metrics` block.  Overwrites in place.

**Hardcoded targets** (file:67–76): Four single-agent result files under `results/` for CoT and RAG variants in zero-shot and few-shot modes.

---

## 5. Single-Agent Baselines

### 5.1 `mortality_single_agent_rag.py` — RAG Baseline (499 LOC)

**Class**: `MortilitySingleAgentRAG` (file:29).

**Pipeline**: Initialises MedRAG (MedCPT encoder + MedCorp2 corpus) for retrieval and a VLLM engine for generation.  For each patient:

1. Constructs a KARE-style prompt (zero-shot: patient context only; few-shot: adds `positive_similars` / `negative_similars` blocks) instructing the model to call `retrieve(query)` first (file:301–337).
2. Generates a first-pass response using VLLM (`temperature=0.7`, `max_tokens=32768`).
3. If a tool call is detected via `_parse_tool_call()` (file:192) — patterns: `retrieve('...')`, `Tool Call: retrieve(...)`, `RETRIEVE(...)` — executes retrieval via `_execute_tool_call()` (file:209), which splits k=8 documents equally between MedCorp and UMLS sub-retrievers.
4. Formats retrieved documents as `[Document i, Score: x.xxx]\n<content>` (file:215) and generates a final response with the evidence appended.
5. Extracts the binary prediction (`# Prediction # 1/0` pattern) via `_extract_prediction_and_probabilities()` (file:226).  Falls back to `0` (survival) if no valid prediction is found.

**Return dict** (file:454):
```python
{'final_prediction': int, 'is_fallback': bool, 'total_generation_time': float, 'response': str}
```

**What is missing vs. the multi-agent debate**: There is no debate loop, no separate risk-assessor or protective-factor-analyst agents, no contrastive preprocessing, no integrator probability output (`mortality_probability` / `survival_probability` fields are absent in the actual return dict, despite appearing in the test block at file:491–492 — those fields are only populated by `mortality_debate_rag.py`).  This is the simplest retrieval-augmented baseline.

---

### 5.2 `mortality_single_agent_cot.py` — CoT Baseline (288 LOC)

**Class**: `MortilitySingleAgentCoT` (file:20).

**Pipeline**: Initialises VLLM directly (no MedRAG).  Single-pass generation with the same KARE task description and prompt structure, with `temperature=0.7`, `max_tokens=32768`, `repetition_penalty=1.2`.  Extracts prediction via the same regex patterns as the RAG baseline.  Supports the same zero-shot / few-shot modes.

**What is missing vs. the RAG baseline**: No retrieval whatsoever.  What is missing vs. the debate: same as above plus no retrieval.  Serves as the pure chain-of-thought control.

**Return dict** (file:244): same schema as the RAG baseline.

---

## 6. `test_dual_query.py` — Dual-Query Retrieval Integration Test (176 LOC)

### What it tests

Exercises the dual-query extension added to `mortality_debate_rag.py` (documented in `DUAL_RETRIEVAL_IMPLEMENTATION.md`).  The extension allows the integrator to emit two specialised search tags instead of one:

```xml
<search_medcorp>query for clinical literature</search_medcorp>
<search_umls>query for medical terminology</search_umls>
```

Each tag routes to the corresponding sub-retriever inside MedCorp2 (4 docs per source = 8 total).

### Test procedure (file:17)

1. Instantiates `MortalityDebateSystem` with `Qwen/Qwen2.5-7B-Instruct`, RAG enabled, MedCPT retriever.
2. Constructs a fixed-text test patient (multiple myeloma + sepsis scenario).
3. Calls `system._integrator_single_step_prediction()` directly (bypasses the full debate loop).
4. Verifies that the return dict contains `mortality_probability`, `survival_probability`, `prediction`, and `confidence`.
5. Inspects `result['query']`: if it is a `dict` with keys `medcorp_query` / `umls_query`, the dual-query format was used; if a plain string, the legacy single-query path was taken.
6. Scans the `test_dual_query_logs/` directory for `retrieve_integrator_*.json` log files, reporting MedCorp / UMLS / combined document counts.  Flags a warning if the combined count is not exactly 8.

### Dependency

Directly imports `from mortality_debate_rag import MortalityDebateSystem` (file:16), so it is tightly coupled to the main debate module and serves as a smoke test after any modification to the retrieval path.

---

## 7. JSON Data Schemas (sampled from on-disk files)

### 7.1 EHR split files — `ehr_data/mimic3_mortality_samples_{train,val,test}.json`

Array of record objects.  Each record is one *temporal prediction instance*:

```json
[
  {
    "patient_id": 10004,
    "visit_id": "...",
    "conditions": [
      ["concept_a", "concept_b"],
      ["concept_c"]
    ],
    "procedures": [[], ["proc_a"]],
    "drugs": [["drug_a"], ["drug_b", "drug_c"]],
    "label": 0
  },
  ...
]
```

- `conditions`, `procedures`, `drugs` are **lists of lists** — outer index = visit number, inner = concepts for that visit.  The rolling-window interpretation means record index `i` covers visits `0..i`.
- `label`: `0` = survival, `1` = mortality at next visit.
- These files power `KAREDataAdapter._load_test_data()` etc.

### 7.2 Patient context lookup — `data/base_context_qwen/patient_contexts_mimic3_mortality.json`

Dict mapping KARE patient IDs to pre-rendered context strings (identical format to `format_patient_context()` output):

```json
{
  "10004_0": "Patient ID: 10004_0\n\nVisit 0:\nConditions:\n1. other fractures\n...",
  "10004_1": "...",
  ...
}
```

Consumed by `improved_faiss_retrieval.py` to populate the similar-patient contexts that are stored in the output JSON.

### 7.3 Patient embeddings — `data/base_context_qwen/patient_embeddings_mimic3_mortality.pkl`

Pickle of `Dict[str, np.ndarray]`, mapping KARE patient IDs to float embedding vectors (Qwen-encoded).  Loaded by `improved_faiss_retrieval.py` to build the FAISS index.

### 7.4 `pateint_mimic3_mortality.json` — label lookup for FAISS retrieval

(Note the filename typo `pateint` — it's the canonical KARE label file, not a split-specific file.)

Dict mapping KARE patient IDs to a dict with at least a `"label"` field plus per-visit data:

```json
{
  "10004_0": {
    "label": 0,
    "visit 0": {
      "conditions": ["other fractures", "intracranial injury", ...],
      "procedures": ["spinal fusion", ...],
      "drugs": ["i.v. solution additives", ...]
    }
  },
  "1004_0": {
    "label": 1,
    "visit 0": { ... }
  },
  ...
}
```

Note: unlike the split JSONs, conditions/procedures/drugs here are flat lists (not lists-of-lists) within a `"visit N"` sub-key.  This file is used by `improved_faiss_retrieval.py` (`load_data()`, file:38) only to look up `patient_data[pid]['label']` for the positive/negative split.

### 7.5 Similar-patient debate file — `data/patient_context/similar_patient_debate/patient_to_top_1_patient_contexts_mimic3_mortality_improved.json`

Dict mapping KARE patient IDs to `{"positive": [str, ...], "negative": [str, ...]}`:

```json
{
  "25368_0": {
    "positive": [
      "Patient ID: 28926_0\n\nVisit 0:\nConditions:\n1. septicemia ...\n\nLabel:\n0\n\n"
    ],
    "negative": [
      "Patient ID: 18557_0\n\nVisit 0:\nConditions:\n1. septicemia ...\n\nLabel:\n1\n\n"
    ]
  },
  ...
}
```

Each list entry is a fully-rendered patient context string with a `"\n\nLabel:\n{0|1}\n\n"` suffix.  The naming convention `positive` / `negative` is relative to the target patient's label (same-label neighbor = positive, opposite-label = negative), NOT to mortality=1.  This is loaded by `KAREDataAdapter._load_similar_patients()` (file:83) and split into `positive_similars` / `negative_similars` in `get_test_sample()` (file:223–239).

### 7.6 `similar_patient_qwen/patient_to_top_1_patient_contexts_mimic3_mortality.json`

Same schema as 7.5 but produced by the original KARE pipeline with smaller search neighbourhood and without the improved class-balance logic.  Used as fallback when the `_improved` file is absent.

---

## 8. Dependencies Between Utilities and `mortality_debate_rag.py`

```
improved_faiss_retrieval.py
    writes → similar_patient_debate/patient_to_top_1_...improved.json
                 ↑ read by
kare_data_adapter.py  (KAREDataAdapter._load_similar_patients)
    consumed by → mortality_debate_rag.py  (MortalityDebateSystem.run_debate_on_dataset)
    consumed by → mortality_single_agent_rag.py  (MortilitySingleAgentRAG)
    consumed by → mortality_single_agent_cot.py  (MortilitySingleAgentCoT)
    consumed by → kare_contrastive_preprocessing.py (demo / __main__ only)

kare_contrastive_preprocessing.py
    consumed by → mortality_debate_rag.py  (preprocess_for_debate called before analyst prompts)

test_dual_query.py
    imports     → mortality_debate_rag.py  (MortalityDebateSystem)
    tests       → _integrator_single_step_prediction()

analyze_retrieve_rate.py
    reads       → results/<dir>/kare_debate_mortality_results.json
    reads       → results/<dir>/debate_logs/debate_responses_<pid>.log
    reads       → results/<dir>/debate_logs/retrieve_integrator_*.json
    writes      → results/<dir>_retrieve_analysis.csv

analyze_fallback_predictions.py
    reads       → results/<dir>/debate_logs/debate_responses_<pid>.log
    reads       → results/<dir>/kare_debate_mortality_results.json
    reads       → data/ehr_data/mimic3_mortality_samples_test.json
    writes      → results/<dir>/prediction_analysis.csv

fix_ground_truth_labels.py
    reads       → data/ehr_data/mimic3_mortality_samples_test.json
    reads/writes→ results/**/*results.json  (in place, with .backup)

update_results_files.py
    reads/writes→ results/single_{cot,rag}_mor_*/results.json  (in place)
```

### Execution order for a fresh experiment

1. Run `improved_faiss_retrieval.py` once to generate the improved similar-patient JSON.
2. Run the main experiment (`mortality_debate_rag.py` or the single-agent variants) which consumes `KAREDataAdapter` and optionally `kare_contrastive_preprocessing.preprocess_for_debate`.
3. If result files were produced with incorrect ground-truth labels, run `fix_ground_truth_labels.py`.
4. If result file schema needs updating to current format, run `update_results_files.py`.
5. Analyse retrieval behaviour with `analyze_retrieve_rate.py`.
6. Diagnose fallback vs. real predictions with `analyze_fallback_predictions.py`.
7. For testing dual-query changes to `mortality_debate_rag.py`, run `test_dual_query.py`.
