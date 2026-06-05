# mirage_medrag — Top-Level Module Summary
## Orchestration, Debugging, and Consistency-Testing Layer

---

## 1. What is mirage_medrag Overall

`mirage_medrag` is a **combined fork** of two upstream open-source projects by Teddy-XiongGZ:

- **MedRAG** (`github.com/Teddy-XiongGZ/MedRAG`) — the retrieval-augmented generation core: corpus management, embedding retrieval via MedCPT, and LLM inference pipelines.
- **MIRAGE** (`github.com/Teddy-XiongGZ/MIRAGE`) — the benchmark harness: 5 medical QA datasets (MMLU, MedQA, MedMCQA, PubMedQA, BioASQ), evaluation scripts, and dataset utilities.

The repo merges these two subprojects under a single root and adds a **custom VLLM integration layer** that monkey-patches the original `transformers.pipeline` call in `MedRAG/src/medrag.py` to transparently route through VLLM. The key custom files are:

| File | Role | Original? |
|------|------|-----------|
| `MedRAG/run_medrag_vllm.py` | VLLM wrapper + answer parsing | **Custom (not in upstream)** |
| `MIRAGE/run_benchmark_vllm.py` | Main benchmark runner with `--resume`, `--max_questions` | **Custom (not in upstream)** |
| `MIRAGE/src/evaluate.py` | Modified to handle both dict and string answer formats | **Modified** |
| `MedRAG/src/medrag.py` | Core MedRAG class | **Mostly original, minor patches** |
| `MIRAGE/src/utils.py` | `locate_answer()` + `locate_answer4pub_llama()` | **Preserved from original** |

### Original vs. Custom Delta (README:182–188)

Key modifications from upstream:

1. VLLM integration via `MedRAG/run_medrag_vllm.py` (`patch_medrag_for_vllm()` monkey-patches `transformers.pipeline`)
2. Llama-3-8B-Instruct as primary target model
3. GPU selection hardcoded to `cuda:4` (configurable), `gpu_memory_utilization=0.5`, `max_model_len=4096`
4. Path and import fixes for the merged repo layout
5. Memory-optimization knobs for consumer GPUs

**Upstream citation:** Both MIRAGE and MedRAG are authored by Teddy-XiongGZ. This repo does not introduce new benchmark datasets or new retrieval algorithms; it is an engineering integration that enables VLLM-accelerated inference and adds robustness/debugging scaffolding.

---

## 2. Conda Environment

**Env name/path:** `/data/wang/junh/envs/medrag` (`environment.yml:1`)

**Python:** `3.10.18` (`environment.yml:32`)

Key pip dependencies (selected from `environment.yml:49–246`):

```
torch==2.8.0
torchaudio==2.8.0
torchvision==0.23.0
vllm==0.11.0
xformers==0.0.32.post1
triton==3.4.0
transformers==4.57.1
tokenizers==0.22.1
faiss-cpu==1.7.4
sentence-transformers==5.1.2
accelerate==1.10.1
numpy==1.26.4           # pinned to <2.0 for numba compat
numba==0.61.2
cupy-cuda12x==13.6.0
ray==2.50.1
pyserini==0.22.1        # BM25 retrieval (Lucene-based)
nmslib==2.1.1           # HNSW index for fast ANN search
langchain==0.0.345      # (very old pinned version)
openai==2.6.1
medrag==1.0.0           # local package install of MedRAG/src
datasets==4.3.0
scikit-learn==1.7.2
```

Notable points:
- `vllm==0.11.0` is a relatively recent release; the log shows it initializing a **V1 LLM engine**.
- `faiss-cpu==1.7.4` is used (CPU FAISS, not GPU), alongside `nmslib` for HNSW approximate nearest-neighbor.
- `numpy` is pinned to `1.26.4` (not 2.x) due to `numba` compatibility requirements.
- `langchain==0.0.345` is an extremely early pin, suggesting the LangChain dependency was locked and never updated.
- `cupy-cuda12x==13.6.0` is present for any CUDA kernel work, coordinated with CUDA 12.x drivers.

---

## 3. `test_pmc_llama_consistency.py` (361 LOC)

### Purpose

This script is a **five-test debugging harness** written to diagnose PMC-LLaMA integration failures when switching from the original `transformers.pipeline` path to the VLLM path. It does **not** compare two independent runs head-to-head in a statistical sense; instead it verifies that the local implementation is *consistent with* the original MedRAG repository's methodology for PMC-LLaMA.

### The Five Tests (`test_pmc_llama_consistency.py:24–357`)

| Test | Function | What it checks |
|------|----------|----------------|
| 1 | `test_pmc_llama_detection()` | Whether PMC-LLaMA model name strings match the `supported_models` list in `run_medrag_vllm.py` |
| 2 | `test_pmc_llama_config()` | Whether `MedRAG.__init__()` correctly sets `max_length=2048`, `context_length=1024` for PMC-LLaMA and loads the jinja template |
| 3 | `test_vllm_wrapper_init()` | Whether `VLLMWrapper("axiong/PMC_LLaMA_13B")` can load under VLLM (may fail if model is unavailable or GPU OOM) |
| 4 | `test_template_loading()` | Whether `MedRAG/templates/pmc_llama.jinja` exists and is non-empty |
| 5 | `test_response_parsing()` | Whether `parse_response_standard()` correctly extracts `answer_choice` from two canonical output formats (perfect JSON and free-text) |

### Why PMC-LLaMA Needs a Consistency Test

Context from `PMC_LLAMA_CONSISTENCY_NOTES.md`:

The original MedRAG repo uses a **unified JSON format** for all models:
```json
{"step_by_step_thinking": "...", "answer_choice": "A"}
```

During local development, a **custom PMC-LLaMA-specific parser** was added to `run_medrag_vllm.py` (lines 136–149) that looked for an array-format response:
```python
# run_medrag_vllm.py:136–149 (now removed per consistency notes)
if model_name and "pmc" in model_name.lower():
    array_match = re.search(
        r'\[\s*\{[^}]*"(?:answer_choice|answer)"\s*:\s*"([ABCD])"[^}]*\}\s*\]',
        response, re.DOTALL
    )
```

This diverged from the original, causing accuracy to drop to ~25% (random-chance level). The consistency notes document the decision to **remove the custom parser** and rely on the same `parse_response_standard()` for all models — with PMC-LLaMA expected to produce standard JSON through the system prompt. The consistency test suite (`test_pmc_llama_consistency.py`) was written to verify this restoration was correct and complete.

The system prompt that all models receive (`PMC_LLAMA_CONSISTENCY_NOTES.md:70`):
```python
general_medrag_system = '''You are a helpful medical expert, and your task is to answer a
multi-choice medical question using the relevant documents. Please first think step-by-step
and then choose the answer from the provided options. Organize your output in a json formatted
as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your
responses will be used for research purposes only, so please have a definite answer.'''
```

---

## 4. `test_prewarmed_medcorp2.py` (136 LOC)

### What "Pre-warmed MedCorp2" Means

"MedCorp2" is an extended version of the standard "MedCorp" corpus (Textbooks + PubMed + StatPearls + Wikipedia), adding a fifth source. "Pre-warming" refers to **loading all per-source retrieval indices into memory at initialization time**, rather than lazily loading them on each query. This makes subsequent queries faster by amortizing the index-load cost.

The test validates the method `medrag.medrag_answer_by_source_prewarmed()` which:
1. Checks that `medrag.source_retrievers` dict is populated at init (`test_prewarmed_medcorp2.py:61–65`)
2. Issues a query and distributes the `k` retrieved documents across sources
3. Times initialization vs. query to demonstrate the warmup benefit
4. Runs a second query to show reduced latency on warm indices

### Key Constants (from `test_prewarmed_medcorp2.py`)

```python
# test_prewarmed_medcorp2.py:27–36
medrag = MedRAG(
    llm_name="Qwen/Qwen3-8B",        # Local model to avoid OpenAI key requirement
    rag=True,
    retriever_name="MedCPT",
    corpus_name="MedCorp2",
    db_dir="/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus",
    corpus_cache=True,
    HNSW=True                         # Approximate NN via HNSW index
)

# test_prewarmed_medcorp2.py:67–71
answer, snippets, scores = medrag.medrag_answer_by_source_prewarmed(
    question=question,
    options=options,
    k=20,           # Total documents, distributed across 5 sources (4 each)
    save_dir="./test_prewarmed_results"
)
```

The script prints per-source document counts and up to three sample "tailored queries" (source-specific rephrased questions generated for each retrieval source), confirming that the pre-warming path correctly separates retrieval by corpus sub-source.

---

## 5. `ANSWER_PARSING_ANALYSIS.md` — The Parsing Problem and Fix

### The Problem

Answer parsing is split across two pipeline stages:

```
Model generates response
        |
run_medrag_vllm.py::parse_response_standard()   [Stage 1 — at inference time]
        |
Saved to prediction/*.json
        |
MIRAGE/src/evaluate.py::evaluate()              [Stage 2 — at eval time]
```

**Stage 1 problem:** The custom `parse_response_standard()` function has a chain of fallback strategies (PMC-LLaMA array regex → JSON extraction → free-text regex → default "A"). When the model name detection (`if "pmc" in model_name.lower()`) fires but the array regex doesn't match, parsing silently falls through to the default answer "A", inflating wrong answers. (`ANSWER_PARSING_ANALYSIS.md:139–166`)

**Stage 2 problem:** The original MIRAGE `evaluate.py` only handled string-format predictions:
```python
# Original (evaluate.py — upstream version)
answers.append(locate_fun(it.split('"answer_choice": "')[-1].strip()))
```
But the custom pipeline saves predictions as Python dicts. The modified version adds:
```python
# Modified (evaluate.py:27–34)
if isinstance(it, dict):
    answer_choice = it.get("answer_choice", "A")
    answers.append(locate_fun(answer_choice))
else:
    answers.append(locate_fun(it.split('"answer_choice": "')[-1].strip()))
```

### Recommended Fix (`ANSWER_PARSING_ANALYSIS.md:173–178`)

Remove the PMC-LLaMA-specific array-format parser entirely. Use `parse_response_standard()` uniformly for all models. The system prompt already instructs all models to produce standard JSON — PMC-LLaMA should comply. This is the fix also documented in `PMC_LLAMA_CONSISTENCY_NOTES.md`.

---

## 6. `DEBUG_GUIDE.md` — Failure Modes Synopsis

`DEBUG_GUIDE.md` documents five classes of failure encountered during development:

1. **Wrong conda environment** — The debugger was initially launched inside `kare_env` instead of `medrag`, causing import failures for `vllm` and `medrag`. Fix: use `/data/wang/junh/envs/medrag/bin/python3.10` explicitly.

2. **Wrong working directory / PYTHONPATH** — Running from an arbitrary directory caused relative imports across the `MedRAG/` and `MIRAGE/src/` subtrees to fail silently. Fix: always run from `/data/wang/junh/githubs/mirage_medrag/MIRAGE` with `PYTHONPATH` explicitly set to all three roots.

3. **Silent hang on startup** — VLLM model loading (30–60 s), MedCPT retriever loading (15–30 s), and corpus DB loading (10–20 s) made the process appear frozen. Fix: `watch -n 1 nvidia-smi` on the target GPU; `ps aux | grep run_benchmark_vllm` to confirm it is running.

4. **GPU OOM at initialization** — VLLM's `gpu_memory_utilization=0.9` default clashes with other processes sharing the A100. The `test_output.log` captures the exact error: `Free memory on device (21.74/44.34 GiB) on startup is less than desired GPU memory utilization (0.9, 39.91 GiB)`. Fix: reduce `gpu_memory_utilization` to `0.5` or lower.

5. **PMC-LLaMA answer parsing failures** — Described in section 5 above. Fix: dump raw responses to `/tmp/pmc_llama_debug.txt` (`DEBUG_GUIDE.md:155–158`) and match the actual format in the regex.

The guide also documents two VS Code debug launch configurations added to the repo: a "Quick Test" (2 questions) and a "Full Debug" (10 questions), both using `--max_questions` which is a custom parameter not present in the upstream MIRAGE runner.

---

## 7. Top-Level Directory Structure

```
mirage_medrag/
├── MedRAG/                         Core retrieval system (forked from Teddy-XiongGZ/MedRAG)
│   ├── src/
│   │   ├── medrag.py               MedRAG class: corpus + retriever + LLM pipeline
│   │   ├── utils.py                Retrieval utilities (MedCPT, BM25, RRF, HNSW)
│   │   ├── template.py             System prompt templates (general_medrag_system etc.)
│   │   └── config.py               Model configs (max_length, context_length per model)
│   ├── templates/
│   │   └── pmc_llama.jinja         Chat template for PMC-LLaMA tokenizer
│   └── run_medrag_vllm.py          ** Custom VLLM integration layer + answer parser **
│
├── MIRAGE/                         Benchmark harness (forked from Teddy-XiongGZ/MIRAGE)
│   ├── src/
│   │   ├── evaluate.py             Accuracy evaluation (modified for dict format)
│   │   └── utils.py                locate_answer() + locate_answer4pub_llama()
│   ├── data/                       MMLU / MedQA / MedMCQA / PubMedQA / BioASQ datasets
│   ├── run_benchmark_vllm.py       ** Custom benchmark runner (--resume, --max_questions) **
│   └── run_benchmark.sh            Shell convenience wrapper
│
├── prediction/                     Per-question JSON outputs (5 datasets × model × config)
│   ├── mmlu/, medqa/, medmcqa/, pubmedqa/, bioasq/
│
├── prediction_by_source/           Same format but predictions broken down by corpus source
│   ├── mmlu/, medqa/, medmcqa/, pubmedqa/, bioasq/
│
├── prediction_by_source_new/       Newer variant of by-source predictions (mmlu only so far)
│
├── test_pmc_llama_consistency.py   5-test harness verifying PMC-LLaMA integration matches upstream
├── test_prewarmed_medcorp2.py      Integration test for pre-warmed MedCorp2 source retrievers
├── test_output.log                 Captured run showing VLLM OOM failure during setup test
├── ANSWER_PARSING_ANALYSIS.md      Analysis of two-stage answer parsing pipeline + recommended fix
├── DEBUG_GUIDE.md                  Operational debugging guide for 5 failure categories
├── PMC_LLAMA_CONSISTENCY_NOTES.md  Decision log: why custom PMC-LLaMA parser was removed
├── environment.yml                 Frozen conda env (Python 3.10, torch 2.8, vllm 0.11.0)
└── README.md                       Quick-start, repo structure, corpus download instructions
```

### `prediction/` Naming Convention

Files follow the pattern:
```
prediction/<dataset>/rag_<k>/<hf_org>/<model_name>/<corpus>/<retriever>/<question_id>.json
```
For example: `prediction/mmlu/rag_32/chaoyi-wu/PMC_LLAMA_7B/medcorp/MedCPT/test_0001.json`

Each `.json` contains a list with one dict: `[{"step_by_step_thinking": "...", "answer_choice": "B"}]`.

`prediction_by_source/` and `prediction_by_source_new/` follow the same dataset-level structure but store outputs produced by the per-source retrieval path (`medrag_answer_by_source_prewarmed()`), allowing ablation analysis per corpus sub-source (Textbooks, PubMed, StatPearls, Wikipedia, and the additional MedCorp2 source).

---

*Summary generated 2026-05-27. Source files: `/data/wang/junh/githubs/mirage_medrag/`.*
