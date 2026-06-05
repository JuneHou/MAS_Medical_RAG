# Module Summary: MIRAGE Benchmark + Prediction Result Directories

**Repo:** `/data/wang/junh/githubs/mirage_medrag/`
**Module root:** `MIRAGE/` (benchmark code) + `prediction*/` (result stores)
**Paper:** Xiong et al., ACL Findings 2024 — "Benchmarking Retrieval-Augmented Generation for Medicine"

---

## 1. What is MIRAGE?

MIRAGE (Medical Information Retrieval-Augmented Generation Evaluation) is a benchmark that evaluates RAG pipelines for medical question answering. It aggregates **7,663 questions** across five existing medical QA datasets, all cast as **zero-shot, multiple-choice** problems. Crucially, answer options are withheld during retrieval (Question-Only Retrieval), simulating realistic clinical scenarios where the retriever cannot see the answer choices.

The benchmark enforces four constraints:
- **Zero-Shot Learning (ZSL)** — no in-context few-shot examples permitted.
- **Multi-Choice Evaluation (MCE)** — all answers are letter choices (A/B/C/D or Yes/No/Maybe).
- **Retrieval-Augmented Generation (RAG)** — systems must retrieve external evidence before answering.
- **Question-Only Retrieval (QOR)** — answer options are excluded from the retrieval query.

### Dataset Subsets

| Dataset | Size | Options | Avg. tokens/Q | Source type |
|---|---|---|---|---|
| MMLU-Med | 1,089 | 4 (A–D) | 63 | Examination |
| MedQA-US | 1,273 | 4 (A–D) | 177 | Examination |
| MedMCQA | 4,183 | 4 (A–D) | 26 | Examination |
| PubMedQA* | 500 | 3 (Yes/No/Maybe) | 24 | Literature |
| BioASQ-Y/N | 618 | 2 (Yes/No) | 17 | Literature |

**MMLU-Med** (1,089 Q): Six biomedical subcategories of the MMLU benchmark — anatomy, clinical knowledge, professional medicine, human genetics, college medicine, and college biology. Standard four-option format.

**MedQA-US** (1,273 Q): English subset of MedQA sourced from the US Medical Licensing Examination (USMLE). Four options; questions are long, clinical vignette style (avg. 177 tokens). Split used: test.

**MedMCQA** (4,183 Q): The dev set of MedMCQA, drawn from Indian medical entrance exams (AIIMS/NEET PG). Short, factual questions. Four options. Split used: dev (not test, since no ground-truth labels for the test split exist publicly).

**PubMedQA\*** (500 Q): Expert-annotated questions from PubMedQA with the provided supporting abstracts removed, forcing the system to retrieve evidence on its own. Answers are yes / no / maybe, reflecting scientific conclusions.

**BioASQ-Y/N** (618 Q): Yes/No questions from BioASQ Task B, years 2019–2023. Ground-truth supporting snippets removed. Binary classification of a factual biomedical claim.

Raw data lives in `MIRAGE/rawdata/{bioasq,medmcqa,medqa,mmlu,pubmedqa}/data_clean/` (source CSV/JSON files, not modified by the benchmark code).

---

## 2. `benchmark.json` Schema

**File:** `MIRAGE/benchmark.json` (4.3 MB, ~7,663 records total)

The file is a single JSON object keyed by dataset name. Each dataset value is itself a dict keyed by a zero-padded four-digit question ID string.

```
{
  "medqa": {
    "0000": {
      "question": "<question text>",
      "options": {
        "A": "<option text>",
        "B": "<option text>",
        "C": "<option text>",
        "D": "<option text>"
      },
      "answer": "B"
    },
    "0001": { ... },
    ...
  },
  "mmlu": { ... },
  "medmcqa": { ... },
  "pubmedqa": { ... },
  "bioasq": { ... }
}
```

Key fields per record:
- `"question"` — plain text question string (no answer choices embedded).
- `"options"` — dict mapping letter keys (`"A"`, `"B"`, `"C"`, `"D"`, or `"Yes"`, `"No"`, `"Maybe"` for PubMedQA; `"Yes"`, `"No"` for BioASQ) to their text.
- `"answer"` — single letter string giving the correct option key.

The top-level keys are lowercase dataset identifiers (`"medqa"`, `"mmlu"`, `"medmcqa"`, `"pubmedqa"`, `"bioasq"`). The inner question IDs are strings (not integers), which the `QADataset` class sorts lexicographically to establish a stable iteration order (see `src/utils.py:13`).

---

## 3. `run_benchmark_vllm.py` — Main Benchmark Runner

**File:** `MIRAGE/run_benchmark_vllm.py` (322 lines)

This script was written for this project (not the upstream MIRAGE repo) to drive the benchmark using a locally hosted VLLM-backed LLM instead of the OpenAI API. It bridges `MedRAG` (the retrieval+generation engine) and MIRAGE's dataset/evaluation infrastructure.

### GPU Assignment

```python
# run_benchmark_vllm.py:31-35
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '7,5')
RETRIEVER_DEVICE = "cuda:0"   # embedding model (GPU 7)
LLM_DEVICE = "cuda:1"         # VLLM (GPU 5)
```

### `argparse` CLI Block

```python
# run_benchmark_vllm.py:224-293
parser = argparse.ArgumentParser(description="Run MIRAGE benchmark with VLLM")

parser.add_argument('--dataset', type=str, nargs='+', default=['all'],
    choices=['all', 'mmlu', 'medqa', 'medmcqa', 'pubmedqa', 'bioasq'],
    help='Dataset(s) to run (default: all)')

parser.add_argument('--mode', type=str, default='rag', choices=['cot', 'rag'],
    help='Mode: cot (no retrieval) or rag (with retrieval) (default: rag)')

parser.add_argument('--k', type=int, default=32,
    help='Number of snippets to retrieve (default: 32)')

parser.add_argument('--llm_name', type=str, default='Qwen/Qwen3-8B',
    help='LLM model name (default: Qwen/Qwen3-8B)')

parser.add_argument('--retriever_name', type=str, default='MedCPT',
    choices=['BM25', 'Contriever', 'SPECTER', 'MedCPT', 'RRF-2', 'RRF-4'],
    help='Retriever to use (default: MedCPT)')

parser.add_argument('--corpus_name', type=str, default='MedCorp',
    choices=['PubMed', 'Textbooks', 'StatPearls', 'Wikipedia',
             'MedText', 'MedCorp', 'UMLS', 'MedCorp2'],
    help='Corpus to use (default: MedCorp)')

parser.add_argument('--results_dir', type=str, default='./prediction_by_source_new',
    help='Directory to save results (default: ./prediction)')

parser.add_argument('--no-resume', action='store_true',
    help='Reprocess all questions (default: skip already processed)')

parser.add_argument('--max_questions', type=int, default=None,
    help='Maximum number of questions to process (default: all questions)')
```

Note: the `--results_dir` default is `'./prediction_by_source_new'` (run_benchmark_vllm.py:276), reflecting that this script was last configured for the `prediction_by_source_new` experiments.

### Iteration Logic (`run_benchmark` function, lines 74–220)

1. Calls `patch_medrag_for_vllm()` to monkey-patch the MedRAG LLM backend to use VLLM instead of the OpenAI API.
2. Instantiates a single `MedRAG` object (lines 113–122) with corpus cache and HNSW index enabled.
3. Expands `'all'` into `['mmlu', 'medqa', 'medmcqa', 'pubmedqa', 'bioasq']` (line 125).
4. For each dataset:
   - Loads `QADataset(dataset_name, dir="MIRAGE")` (line 136).
   - Sets split to `"dev"` for medmcqa, `"test"` for all others (lines 138–139).
   - Constructs the output directory path (lines 142–157):
     - **RAG mode:** `{results_dir}/{dataset}/rag_{k}/{llm_name}/{corpus_name}/{retriever_name}/`
     - **CoT mode:** `{results_dir}/{dataset}/cot/{llm_name}/`
   - Iterates questions with `tqdm`; skips existing files when `resume=True`.
   - Calls `vllm_medrag_answer(medrag, question, options, k, ...)` for each question.
   - On success: saves a JSON file via `save_prediction()`.
   - On exception: saves a fallback answer (`"A"`) and continues.

### Per-Question Output Format (`save_prediction`, lines 61–71)

Each answer is saved as a single-element JSON array:

```json
[
  {
    "step_by_step_thinking": "<chain-of-thought reasoning>",
    "answer_choice": "A"
  }
]
```

The file is named `{split}_{question_id}.json` (e.g., `test_0000.json`, `dev_0000.json`).

---

## 4. `src/evaluate.py` — Metrics and Scoring

**File:** `MIRAGE/src/evaluate.py` (105 lines)

### Main Evaluation Function

```python
# src/evaluate.py:10-54
def evaluate(dataset, save_dir, split="test", locate_fun=locate_answer):

    flag = False
    pred = []
    empty_count = 0
    na_count = 0
    error_count = 0
    answer_list = ["A", "B", "C", "D"]
    answer2idx = {ans:i for i, ans in enumerate(answer_list)}
    
    total_len = len(dataset)

    for q_idx in range(len(dataset)):
        fpath = os.path.join(save_dir, split + "_" + dataset.index[q_idx] + ".json")
        answers = []
        for it in json.load(open(fpath))[:1]:
            if isinstance(it, dict):
                # New format: JSON object
                if "error" in it:
                    error_count += 1
                answer_choice = it.get("answer_choice")
                answers.append(locate_fun(answer_choice))
            else:
                # Old format: JSON string
                answers.append(locate_fun(it.split('"answer_choice": "')[-1].strip()))
        answers = [ans for ans in answers if ans != "NA"]
        if len(answers) == 0:
            pred.append(-1)
            continue
        ans = statistics.mode(answers)
        if ans in answer_list:
            pred.append(answer_list.index(ans))
        else:
            pred.append(-1)
    
    truth = [answer2idx[item['answer']] for item in dataset]
    if len(pred) < len(truth):
        truth = truth[:len(pred)]
        flag = True
    
    acc = (np.array(truth) == np.array(pred)).mean()
    std = np.sqrt(acc * (1-acc) / len(truth))
    return acc, std, flag, error_count
```

Key behaviors:
- **Metric:** Simple accuracy (fraction of correct letter choices). No partial credit. Returned as a float in [0,1] along with proportion standard deviation `sqrt(p*(1-p)/n)`.
- **Reads only the first element** of each prediction JSON list (`[:1]`), so self-consistency sampling (multiple completions) is not used by default, although the list format would support it via `statistics.mode`.
- **Answer parsing:** Delegates to `locate_fun`. The default `locate_answer` uses a priority-ordered chain of regex patterns (see §5). If no match, returns `"A"` as fallback.
- **Special model handling:** For `pmc_llama` models, uses `locate_answer4pub_llama` instead (strips `"Answer:"` prefix first).
- **Incomplete run detection:** If `len(pred) < len(truth)`, the `flag` return value is set to `True` and evaluation proceeds on the truncated set.
- **-1 predictions:** Questions where answer parsing yields only `"NA"` values get prediction index -1, which always scores as incorrect.

### CLI arguments for `evaluate.py`

```
--llm_name        e.g., "OpenAI/gpt-35-turbo-16k"
--rag             flag; if absent, evaluates CoT results
--k               int, number of retrieved snippets (default 32)
--corpus_name     lowercase corpus name (default "medcorp")
--retriever_name  retriever name (default "RRF-4")
--results_dir     root prediction directory (default "./prediction")
```

The script iterates all five datasets and prints per-dataset accuracy, then prints the macro-average over completed datasets.

---

## 5. `src/utils.py` — Dataset Loader and Answer Parsers

**File:** `MIRAGE/src/utils.py` (92 lines)

### `QADataset` Class (lines 5–23)

Loads `benchmark.json` at instantiation (hardcoded absolute path: `/data/wang/junh/githubs/mirage_medrag/MIRAGE/benchmark.json`, line 9). Stores sorted question ID keys as `self.index` for stable ordering. Supports integer indexing and slicing.

```python
class QADataset:
    def __init__(self, data, dir="."):
        self.data = data.lower().split("_")[0]   # normalizes e.g. "mmlu_med" -> "mmlu"
        benchmark = json.load(open("/data/wang/junh/githubs/mirage_medrag/MIRAGE/benchmark.json"))
        ...
        self.index = sorted(self.dataset.keys())
```

The `dir` parameter is accepted but unused (the path is hardcoded).

### `locate_answer(sentence)` (lines 27–69)

Priority-ordered regex chain to extract a letter answer from the model's `answer_choice` field:

1. Bare letter: `^\s*(A|B|C|D)$`
2. Letter + "or": `^\s*(A|B|C|D) or`
3. Letter + "and": `^\s*(A|B|C|D) and`
4. Letter + "/": `^\s*(A|B|C|D)/`
5. Letter + ",": `^\s*(A|B|C|D),`
6. "Option X": `[Oo]ption (A|B|C|D)`
7. Colon + letter: `:\s*(A|B|C|D)`
8. Letter + ".": `^\s*(A|B|C|D)\.`
9. Letter + `"`: `^\s*(A|B|C|D)"`
10. Letter + ":": `^\s*(A|B|C|D):`
11. **Default fallback:** returns `"A"` (not `"NA"`) — meaning any unparseable answer defaults to A.

### `locate_answer4pub_llama(sentence)` (lines 71–92)

Variant for PMC-LLaMA models. Splits on `"Answer:"`, then applies a shorter regex chain (`"Option X"`, `"OPTION X"`, letter-quote, letter-colon patterns). Returns `"A"` on failure.

---

## 6. `test_setup.py` — Sanity Check Script

**File:** `MIRAGE/test_setup.py` (139 lines)

A five-step smoke test that verifies the end-to-end VLLM + MedRAG + MIRAGE pipeline is functional before running the full benchmark. Not a unit test framework; just a sequential script:

1. **VLLM patch** — calls `patch_medrag_for_vllm()` and confirms no exception.
2. **MedRAG init** — instantiates `MedRAG` with `Meta-Llama-3-8B-Instruct`, `MedCPT` retriever, `Textbooks` corpus (small corpus chosen for speed), HNSW enabled.
3. **Dataset load** — loads MMLU (1,089 questions) via `QADataset("mmlu", dir="MIRAGE")`.
4. **Single prediction** — runs `vllm_medrag_answer` on question index 0 with `k=5` snippets. Prints predicted vs. actual answer and correct/incorrect.
5. **Save/load round-trip** — writes the prediction JSON to `/tmp/mirage_test/test_question.json` and re-reads it to confirm file I/O works.

Exits with code 0 on success, 1 on any failure. Intended to be run before long benchmark runs.

---

## 7. Prediction Result-Directory Naming Convention

All three result trees share the same hierarchical path structure. The leaf directory contains individual per-question JSON files.

### Path Template

**RAG mode:**
```
{results_root}/{dataset}/rag_{k}/{llm_org}/{llm_model}/{corpus_name}/{retriever_name}/
```

**CoT mode (no retrieval):**
```
{results_root}/{dataset}/cot/{llm_org}/{llm_model}/
```

### Concrete Examples (from `prediction/medqa/`)

```
prediction/medqa/
  cot/
    Qwen/Qwen3-8B/
      test_0000.json          # per-question answer
      test_0000_raw_response.txt
      response.json
      snippets.json
  rag_32/
    Qwen/Qwen3-8B/
      medcorp/MedCPT/
        test_0000.json
        ...
      medcorp2/MedCPT/
        test_0000.json
        ...
      umls/MedCPT/
        test_0000.json
        ...
    meta-llama/Meta-Llama-3-8B-Instruct/
      medcorp/MedCPT/
        test_0000.json
        ...
      medcorp/BM25/
        test_0000.json
        ...
      medcorp2/MedCPT/
        test_0000.json
        ...
    google/medgemma-27b-text-it/
      medcorp/MedCPT/
        test_0000.json
        ...
```

### Per-Question File Naming

- Format: `{split}_{question_id}.json`
- `split` is `"test"` for MedQA, MMLU, PubMedQA, BioASQ; `"dev"` for MedMCQA.
- `question_id` is the zero-padded 4-digit key from `benchmark.json` (e.g., `test_0000.json`).
- For `prediction_by_source_new/mmlu/`, MMLU questions use their subcategory prefix instead of a numeric ID (e.g., `test_anatomy-000.json`, `test_clinical_knowledge-042.json`). The six prefixes are: `anatomy`, `clinical_knowledge`, `college_biology`, `college_medicine`, `medical_genetics`, `professional_medicine`.

### File Contents (leaf JSON)

Every per-question file contains a single-element list regardless of mode:

```json
[
  {
    "step_by_step_thinking": "<reasoning chain>",
    "answer_choice": "A"
  }
]
```

Alongside each `.json`, CoT runs and some RAG runs also emit a `_raw_response.txt` with the unprocessed model output. Some directories contain `response.json`, `snippets.json`, or `source_contexts.json` aggregate files (see §9).

### Models Observed Across Runs

| Directory | Models |
|---|---|
| `prediction/` | `Qwen/Qwen3-8B`, `meta-llama/Meta-Llama-3-8B-Instruct`, `google/medgemma-27b-text-it` |
| `prediction_by_source/` | `Qwen/Qwen3-8B`, `Qwen/Qwen2.5-7B-Instruct` |
| `prediction_by_source_new/` | `Qwen/Qwen2.5-7B-Instruct` |

---

## 8. `prediction/` vs `prediction_by_source/` — What the Difference Is

Both directories follow the same five-dataset structure (bioasq, medmcqa, medqa, mmlu, pubmedqa).

**`prediction/`** contains runs against fused corpora (MedCorp, MedCorp2, UMLS) with multiple models and retrievers. This matches the standard MIRAGE evaluation protocol where the system retrieves from the full pooled corpus and generates a single answer.

**`prediction_by_source/`** contains runs exclusively under `medcorp2/MedCPT/`, but the leaf directories additionally contain three extra files that do not appear in `prediction/`:

- `response_by_source.json` — stores the raw model response for each corpus source separately.
- `snippets_by_source.json` — stores retrieved snippets broken down by contributing corpus source.
- `source_contexts.json` — stores the full formatted context string that was fed to the LLM, with a separate entry per corpus source (e.g., `"Source: MEDCORP\nQuery: ...\nRetrieved Documents:\n..."` and `"Source: UMLS\nQuery: ...\nRetrieved Documents:\n..."`).

Inspection of `source_contexts.json` (e.g., at `prediction_by_source/medqa/rag_32/Qwen/Qwen3-8B/medcorp2/MedCPT/source_contexts.json`) confirms the intent: each entry is a separately-formatted retrieval context from a different constituent corpus within MedCorp2. This allows downstream analysis of which individual source (StatPearls, UMLS, PubMed, etc.) contributed the evidence that led to the final answer, rather than treating MedCorp2 as a single opaque pool. The final per-question `test_XXXX.json` still contains a single merged answer.

In summary:
- `prediction/` = standard fused-corpus evaluation, multiple models.
- `prediction_by_source/` = evaluation with per-source context logging enabled, restricted to Qwen models + MedCorp2 + MedCPT.

---

## 9. `prediction_by_source_new/` — What's Different

**Scope:** Only `mmlu/` is present (no other datasets).

**Model:** Only `Qwen/Qwen2.5-7B-Instruct` (vs. `Qwen3-8B` in `prediction_by_source/`).

**Retrievers:** Both `BM25` and `MedCPT` are present (vs. only `MedCPT` in `prediction_by_source/`).

**Corpus:** Only `medcorp2`.

**Key structural difference — question ID format for MMLU:** In `prediction_by_source_new/mmlu/`, question files are named by subcategory + index (e.g., `test_anatomy-000.json`, `test_clinical_knowledge-042.json`) rather than by sequential zero-padded integer (`test_0000.json`). This finer-grained naming preserves the MMLU task identity throughout the file system, making it possible to slice results by subject area without parsing the benchmark JSON.

The six MMLU subcategory prefixes present are:
- `test_anatomy-*.json`
- `test_clinical_knowledge-*.json`
- `test_college_biology-*.json`
- `test_college_medicine-*.json`
- `test_medical_genetics-*.json`
- `test_professional_medicine-*.json`

Each leaf directory (`BM25/` or `MedCPT/`) contains ~1,089 answer files (2 × 1,089 = 2,178 total per-question files across both retrievers) plus the aggregate `response_by_source.json`, `snippets_by_source.json`, and `source_contexts.json`.

In summary: `prediction_by_source_new/` is an incremental experiment that (a) switched to a smaller/older model (`Qwen2.5-7B` vs `Qwen3-8B`), (b) added BM25 as a second retriever for comparison, and (c) adopted subcategory-level naming for MMLU files. It appears to be an ablation focused exclusively on MMLU with the intent to compare BM25 vs MedCPT retrieval on individual MMLU subject areas.

---

## 10. Key File Reference Map

| Item | Path |
|---|---|
| Benchmark data | `MIRAGE/benchmark.json` |
| Benchmark runner (VLLM) | `MIRAGE/run_benchmark_vllm.py` |
| Evaluation script | `MIRAGE/src/evaluate.py` |
| Dataset loader + answer parsers | `MIRAGE/src/utils.py` |
| Setup smoke test | `MIRAGE/test_setup.py` |
| Raw source data | `MIRAGE/rawdata/{dataset}/data_clean/` |
| Figures | `MIRAGE/figs/MIRAGE.png`, `result_llm.png`, `result_corpus_retriever.png` |
| Static docs site | `MIRAGE/docs/index.html` |
| Standard predictions | `prediction/{dataset}/{mode}/{llm}/{corpus}/{retriever}/` |
| Per-source predictions | `prediction_by_source/{dataset}/rag_32/{llm}/medcorp2/MedCPT/` |
| New per-source (MMLU only) | `prediction_by_source_new/mmlu/rag_32/Qwen/Qwen2.5-7B-Instruct/medcorp2/{BM25,MedCPT}/` |
