# Module Summary: MedRAG Core Library

**Repository:** `/data/wang/junh/githubs/mirage_medrag/MedRAG/`
**Role:** Retrieval + RAG implementation that the multi-agent debate project calls as a library.
**Upstream:** `Teddy-XiongGZ/MedRAG` (ACL 2024); this fork adds vLLM, MedCorp2, source-disaggregated retrieval, and PMC-LLaMA parsing fixes.

---

## 1. MedRAG Class (`src/medrag.py`)

### 1.1 Constructor Signature

```python
# src/medrag.py:45
def __init__(
    self,
    llm_name="OpenAI/gpt-3.5-turbo-16k",
    rag=True,
    follow_up=False,
    retriever_name="MedCPT",
    corpus_name="Textbooks",
    db_dir="./corpus",
    cache_dir=None,
    corpus_cache=False,
    HNSW=False,
    retriever_device=None
):
```

| Parameter | Purpose |
|-----------|---------|
| `llm_name` | `"OpenAI/<model>"`, `"Google/gemini-*"`, or any HuggingFace model path. Determines tokenizer, context window, and generation backend. |
| `rag` | Enable retrieval. When `False`, uses chain-of-thought only. |
| `follow_up` | Activates i-MedRAG iterative follow-up query mode. |
| `retriever_name` | Key into `retriever_names` dict in `utils.py`. See Section 2. |
| `corpus_name` | Key into `corpus_names` dict in `utils.py`. See Section 2. |
| `db_dir` | Root directory for corpus chunk files and FAISS indexes. |
| `cache_dir` | HuggingFace model cache directory. |
| `corpus_cache` | If `True`, pre-loads a flat `id2text.json` lookup for fast document retrieval (avoids disk seeks per query). |
| `HNSW` | Build HNSW-accelerated FAISS index instead of flat index (set at first-build time; faster approximate search). |
| `retriever_device` | Override device for embedding models (e.g., `"cuda:0"`). Enables GPU separation from the LLM. |

**Construction side-effects by corpus:**

- **Standard corpus** (`Textbooks`, `MedCorp`, etc.): instantiates one `RetrievalSystem` with the named retriever and corpus.
- **`MedCorp2`**: skips the single `RetrievalSystem`; instead calls `_initialize_source_retrievers()` which builds two separate GPU-pinned retrieval systems (`medcorp` on GPU 1, `umls` on GPU 0) sharing one embedding model.
- **`follow_up=True`**: overrides `self.answer` to point at `i_medrag_answer`; replaces prompt templates with iterative variants.
- Otherwise: `self.answer` is set to `medrag_answer` (standard) or `medrag_answer_by_source` (MedCorp2).

**Context window constants set in `__init__`** (`src/medrag.py:68-139`):

| Model family | `max_length` | `context_length` |
|---|---|---|
| GPT-3.5 | 16384 | 15000 |
| GPT-4 | 32768 | 30000 |
| Gemini 1.5 | 1048576 | 1040384 |
| Mixtral | 32768 | 30000 |
| Llama-2 | 4096 | 3072 |
| Llama-3 / 3.1 / 3.2 | 8192 / 131072 | 7168 / 128000 |
| PMC-LLaMA | 2048 | 1024 |
| Qwen | 32768 | 7168 |
| Gemma | 8192 | 7168 |

---

### 1.2 Supported `llm_name` Values

The prefix before `/` controls which code branch is used:

- **`OpenAI/...`** — uses `openai_client` lambda; works with both openai v0 and v1 SDKs, Azure endpoint, or plain OpenAI. Temperature=0.
- **`Google/gemini-*`** — uses `google.generativeai`; temperature=0.7, max\_output\_tokens=2048.
- **Any other string** — loads a HuggingFace model via `transformers.pipeline("text-generation", ...)` with `bfloat16` and `device_map="auto"`. The vLLM monkey-patch in `run_medrag_vllm.py` intercepts this for supported model families (see Section 4).

Officially tested models (README): `OpenAI/gpt-4`, `OpenAI/gpt-3.5-turbo`, `Google/gemini-1.0-pro`, `meta-llama/Meta-Llama-3-70B-Instruct`, `meta-llama/Llama-2-70b-chat-hf`, `mistralai/Mixtral-8x7B-Instruct-v0.1`, `epfl-llm/meditron-70b`, `axiong/PMC_LLaMA_13B`. Also tested (README): `OpenAI/gpt-4o`, `meta-llama/Meta-Llama-3.1-*`, `meta-llama/Llama-3.2-*`.

---

### 1.3 Supported `corpus_name` Values

See `src/utils.py:10-18`; full list: `PubMed`, `Textbooks`, `StatPearls`, `Wikipedia`, `UMLS`, `MedText`, `MedCorp`, `MedCorp2`.

---

### 1.4 Primary Method: `medrag_answer`

```python
# src/medrag.py:310
def medrag_answer(self, question, options=None, k=32, rrf_k=100,
                  save_dir=None, snippets=None, snippets_ids=None, **kwargs):
    -> (answer_str_or_list, retrieved_snippets, scores)
```

**Call path:**
1. Formats `options` dict to `"A. ...\nB. ..."` string.
2. **Retrieval** (three modes, in priority order):
   - If `snippets` is provided: skip retrieval, use provided list.
   - If `snippets_ids` is provided: materialise docs from a `DocExtracter`.
   - Otherwise: call `self.retrieval_system.retrieve(question, k=k, rrf_k=rrf_k)`.
3. Concatenates snippets as `"Document [idx] (Title: ...) ..."`, then token-truncates to `self.context_length`.
4. Renders the Liquid template `medrag_prompt` with `context`, `question`, `options`.
5. Calls `self.generate(messages)`.
6. If `save_dir` given: saves `snippets.json` and `response.json`.
7. Returns `(answer, retrieved_snippets, scores)`.

When `rag=False`, skips retrieval and uses `cot_prompt` / `cot_system` templates instead.

---

### 1.5 MedCorp2-Specific Answer Method: `medrag_answer_by_source`

```python
# src/medrag.py:385
def medrag_answer_by_source(self, question, options=None, k=32, rrf_k=100,
                             save_dir=None, **kwargs):
```

Used automatically when `corpus_name="MedCorp2"`. Key differences:

- Uses pre-warmed `self.source_retrievers` dict (initialised at construction).
- Splits `k` as `k_medcorp = k//2 + k%2`, `k_umls = k//2`.
- Queries each source with the **original question** directly (no LLM-generated sub-query).
- Labels each retrieved snippet with `source_type` and `query_used`.
- Combines contexts as `"Source: MEDCORP\n...\n\nSource: UMLS\n..."`.
- Saves three files: `snippets_by_source.json`, `response_by_source.json`, `source_contexts.json`.

---

### 1.6 i-MedRAG Method: `i_medrag_answer`

```python
# src/medrag.py:514
def i_medrag_answer(self, question, options=None, k=32, rrf_k=100,
                    save_path=None, n_rounds=4, n_queries=3,
                    qa_cache_path=None, **kwargs):
```

Iterative retrieval loop:

- `n_rounds` rounds of "generate follow-up queries → retrieve → accumulate context".
- Each round the LLM generates sub-queries under `## Queries` header; these are parsed with a separate LLM call and each fed to `medrag_answer`.
- After `n_rounds` exhausted (or if answer appears), prompts LLM for final answer in JSON.
- Saves full conversation to `save_path` as a list of message dicts.
- `qa_cache_path` allows resuming from a checkpoint.

---

### 1.7 `generate` Method

```python
# src/medrag.py:255
def generate(self, messages, **kwargs):
```

Dispatches to the correct backend:
- **OpenAI**: `openai_client(model=..., messages=..., temperature=0.0)`.
- **Gemini**: concatenates system + user content, calls `model.generate_content`.
- **HuggingFace / vLLM**: applies `tokenizer.apply_chat_template(..., add_generation_prompt=True)`, then calls the pipeline. Meditron gets custom stopping criteria (`["###", "User:", "\n\n\n"]`). Llama-3 gets `<|eot_id|>` as extra EOS. Output sliced to strip prompt prefix (`response[0]["generated_text"][len(prompt):]`).

---

### 1.8 `_initialize_source_retrievers` (MedCorp2 pre-warm)

```python
# src/medrag.py:168
def _initialize_source_retrievers(self):
```

Builds two `RetrievalSystem` instances:

| Source key | Corpus | GPU |
|---|---|---|
| `"medcorp"` | `"MedCorp"` (pubmed+textbooks+statpearls+wikipedia) | cuda:1 |
| `"umls"` | `"UMLS"` | cuda:0 |

One shared embedding model is loaded once and injected into both retrieval systems to save GPU memory. BM25 skips this step. This replaces the per-query cost of spinning up retrievers.

---

## 2. RetrievalSystem and Per-Corpus Retrievers (`src/utils.py`)

### 2.1 Corpus and Retriever Registries

```python
# src/utils.py:10-18
corpus_names = {
    "PubMed":    ["pubmed"],
    "Textbooks": ["textbooks"],
    "StatPearls":["statpearls"],
    "Wikipedia": ["wikipedia"],
    "UMLS":      ["umls"],
    "MedText":   ["textbooks", "statpearls"],
    "MedCorp":   ["pubmed", "textbooks", "statpearls", "wikipedia"],
    "MedCorp2":  ["pubmed", "textbooks", "statpearls", "wikipedia", "umls"],
}

# src/utils.py:21-27
retriever_names = {
    "BM25":      ["bm25"],
    "Contriever":["facebook/contriever"],
    "SPECTER":   ["allenai/specter"],
    "MedCPT":    ["ncbi/MedCPT-Query-Encoder"],
    "RRF-2":     ["bm25", "ncbi/MedCPT-Query-Encoder"],
    "RRF-4":     ["bm25", "facebook/contriever", "allenai/specter", "ncbi/MedCPT-Query-Encoder"],
}
```

`MedCorp` is PubMed + Textbooks + StatPearls + Wikipedia (4 corpora, 54.2M snippets).
`MedCorp2` adds `umls`, making it the 5-corpus superset.

---

### 2.2 `RetrievalSystem` Class

```python
# src/utils.py:298-386
class RetrievalSystem:
    def __init__(self, retriever_name="MedCPT", corpus_name="Textbooks",
                 db_dir="./corpus", HNSW=False, cache=False, device=None,
                 shared_embedding_function=None):
```

**Internal structure:** `self.retrievers` is a 2-D list — `self.retrievers[i][j]` is the `Retriever` for the i-th retrieval model applied to the j-th sub-corpus. For `RRF-4` on `MedCorp` this is a 4×4 grid.

**`retrieve(question, k=32, rrf_k=100, id_only=False)`** (`src/utils.py:318`):

1. Expands `k` to `max(k*2, 100)` for RRF methods to over-fetch before fusion.
2. Calls `get_relevant_documents` on every `[i][j]` cell.
3. Calls `merge(texts, scores, k=k, rrf_k=rrf_k)`.
4. If `cache=True`: translates IDs back to full text via `DocExtracter`.

**`merge` (Reciprocal Rank Fusion)** (`src/utils.py:347`):

- For each retriever `i`: concatenates results across all sub-corpora `j`, sorts by score (descending for IP metrics, ascending for SPECTER L2).
- Fuses with RRF: `score += 1 / (rrf_k + rank + 1)` for each doc ID.
- If only one retriever (non-RRF): returns top-k from single sorted list.
- If multiple retrievers: returns top-k from fused RRF ranking.

---

### 2.3 `Retriever` Class

```python
# src/utils.py:133-295
class Retriever:
    def __init__(self, retriever_name, corpus_name, db_dir, HNSW=False,
                 device=None, shared_embedding_function=None, **kwarg):
```

**Index bootstrap logic:**
- On first run with a dense retriever: checks for `faiss.index` in `<db_dir>/<corpus>/index/<model-name>/`. If missing, checks if pre-computed embeddings can be downloaded from SharePoint (available for textbooks/pubmed/wikipedia × SPECTER/Contriever/MedCPT). Otherwise embeds from scratch.
- StatPearls chunks must be built locally (NCBI tar.gz + `src/data/statpearls.py`).
- BM25: builds a Pyserini Lucene index in the same path. Requires Java.

**`get_relevant_documents(question, k=32, id_only=False)`** (`src/utils.py:216`):
- BM25: calls `LuceneSearcher.search`, parses docids with special handling for UMLS-style IDs (`UMLS_R*_L*_C*` format).
- Dense: encodes query, calls `faiss_index.search`, maps result indices back via `metadatas.jsonl`.
- Returns `(List[Dict], List[float])` — each dict has `id`, `title`, `content`.

**Embedding functions:**
- `MedCPT` and `SPECTER` use `CustomizeSentenceTransformer` (CLS pooling override; default sentence-transformers uses MEAN).
- `Contriever` uses plain `SentenceTransformer` (MEAN pooling is correct for that model).
- `shared_embedding_function`: when provided from outside (MedCorp2 pre-warm), replaces the locally constructed model to avoid redundant GPU allocations.

**Text encoding per model** (`src/utils.py:92-99`):
- SPECTER: `sep_token.join([title, content])`
- Contriever: `". ".join([title, content])`
- MedCPT: `[title, content]` (list form, asymmetric encoding)
- Others: `concat(title, content)` (adds period if title lacks terminal punctuation)

---

### 2.4 `DocExtracter` Class

```python
# src/utils.py:389-563
class DocExtracter:
    def __init__(self, db_dir="./corpus", cache=False, corpus_name="MedCorp"):
```

Two modes:
- **`cache=True`**: builds `<corpus_name>_id2text.json` — a flat dict mapping every document ID to its full content dict. Fast random access for repeated queries. Skips Git-LFS pointer files.
- **`cache=False`**: builds `<corpus_name>_id2path.json` — maps each ID to `{"fpath": "...", "index": line_number}`. Reads on-demand from JSONL at query time.

**`extract(ids)`** (`src/utils.py:504`): returns list of document dicts for given IDs. Has special UMLS ID resolution logic for both `umls_run*_<n>` format (dense retriever) and `UMLS_R*_L*_C*` format (BM25).

---

### 2.5 MedCorp vs. MedCorp2

| | MedCorp | MedCorp2 |
|---|---|---|
| Sub-corpora | pubmed, textbooks, statpearls, wikipedia | + umls |
| # snippets (approx.) | 54.2M | 54.2M + UMLS concepts |
| `RetrievalSystem` | Single system over 4 corpora | **Not** a single `RetrievalSystem`; two pre-warmed systems (MedCorp + UMLS separately) |
| `self.answer` method | `medrag_answer` | `medrag_answer_by_source` |
| GPU assignment | Single device | MedCorp→cuda:1, UMLS→cuda:0 |
| k split | full k to one system | k split roughly 50/50 between sources |
| Saved files | `snippets.json`, `response.json` | `snippets_by_source.json`, `response_by_source.json`, `source_contexts.json` |

The key architectural decision: rather than fusing UMLS into a single MedCorp retrieval run (which would require one large multi-corpus FAISS index), MedCorp2 keeps two separate retrieval pipelines and concatenates their contexts in the prompt. This also makes it easy to attribute retrieved snippets to their source.

---

## 3. Template Engine (`src/template.py`)

Templates use the **Liquid** templating library (`from liquid import Template`). All templates use `{{variable}}` interpolation.

### 3.1 Template Objects Defined

| Name | Type | Used For |
|---|---|---|
| `general_cot_system` | plain string | System message for chain-of-thought (no RAG). Forces JSON output. |
| `general_cot` | `Template` | User prompt for CoT: question + options → JSON `{step_by_step_thinking, answer_choice}`. |
| `general_medrag_system` | plain string | System message for RAG. Forces JSON output. |
| `general_medrag` | `Template` | User prompt with context + question + options → same JSON format. |
| `meditron_cot` | `Template` | Few-shot CoT prompt in Meditron `### User / ### Assistant` format (with a one-shot example). |
| `meditron_medrag` | `Template` | Few-shot MedRAG prompt in Meditron format. |
| `simple_medrag_system` | plain string | Short system message for i-MedRAG mode. |
| `simple_medrag_prompt` | `Template` | Minimal prompt: context + question (no options, for sub-queries in iterative mode). |
| `i_medrag_system` | plain string | System message for iterative MedRAG assistant. |
| `follow_up_instruction_ask` | plain string (format placeholder) | Instructs LLM to generate N follow-up queries under `## Queries` section. |
| `follow_up_instruction_answer` | plain string | Instructs LLM to produce final answer under `## Answer` section. |
| `dual_query_generation` | `Template` | (Defined but not currently used) Generates two queries for MedCorp and UMLS. |
| `source_descriptions` | dict | (Defined but not currently used) Per-source retrieval guidance strings. |

**Which template set is activated** is determined at `MedRAG.__init__` time:

```python
# src/medrag.py:64-66
self.templates = {
    "cot_system": general_cot_system, "cot_prompt": general_cot,
    "medrag_system": general_medrag_system, "medrag_prompt": general_medrag
}
```

If `follow_up=True`, the template dict is extended with `simple_medrag_system`, `simple_medrag_prompt`, `i_medrag_system`, `follow_up_ask`, `follow_up_answer`. The `meditron_*` templates are defined but are **not** automatically activated — they would need explicit wiring.

The JSON output format enforced by `general_medrag_system` / `general_cot_system`:
```json
{"step_by_step_thinking": "...", "answer_choice": "A"}
```

---

## 4. Jinja Chat Templates (`templates/`)

These are loaded into `self.tokenizer.chat_template` in `MedRAG.__init__` for models that lack a suitable built-in template.

| File | Loaded for | Format produced |
|---|---|---|
| `meditron.jinja` | `"meditron"` in `llm_name` (not currently loaded in code; template exists for reference) | Prepends system content to first user turn; no special tokens; assistant turns end with `eos_token`. |
| `mistral-instruct.jinja` | `"mixtral"` in `llm_name` (`src/medrag.py:96`) | `<bos>[INST] system+user [/INST] response </s>` |
| `pmc_llama.jinja` | `"pmc"` in `llm_name` (`src/medrag.py:117`) | Simple: system content prepended to first user turn; no special wrapper tokens; `eos_token` after each assistant turn. |
| `pmc_llama_new.jinja` | Not loaded by default (available) | Same as `pmc_llama.jinja` but filters repeated question blocks in multi-turn conversations to avoid redundancy. |

Models using HuggingFace built-in templates (no jinja file): Llama-2, Llama-3/3.1/3.2, Qwen 2/3, Gemma.

---

## 5. `run_medrag_vllm.py` — Top-Level vLLM Inference Runner

This is the script that users invoke directly (or import from) to run inference with vLLM as the generation backend.

### 5.1 Purpose and Design

Not a CLI batch inference script; rather a **library module** containing:
- `VLLMWrapper`: a drop-in replacement for `transformers.pipeline` that routes generation to vLLM.
- `patch_medrag_for_vllm()`: monkey-patches `transformers.pipeline` globally.
- `vllm_medrag_answer()`: wrapper around `medrag_instance.answer()` with response parsing.
- `parse_response_standard()`: universal response parser.
- `init_medrag_with_device_separation()`: convenience initializer.
- `run_medrag_with_vllm()`: example demo function (hardcoded `Qwen/Qwen3-8B`).

### 5.2 Device Constants

```python
# run_medrag_vllm.py:11-12
RETRIEVER_DEVICE = "cuda:0"   # GPU 6 in practice (first visible)
LLM_DEVICE = "cuda:1"         # GPU 7 in practice (second visible)
```

### 5.3 `VLLMWrapper` Class

```python
# run_medrag_vllm.py:29
class VLLMWrapper:
    def __init__(self, model_name, **kwargs): ...
    def __call__(self, prompt, **kwargs): ...
    def generate_with_system(self, system_message, user_prompt, ...): ...
```

**Constructor:** initialises `vllm.LLM` with model-specific `max_model_len`:
- PMC-LLaMA: 2048
- Llama-3: 4096
- Qwen / fine-tuned variants: 32768
- Default: 32768

`enforce_eager=True` disables CUDA graphs (fixes token-limit issues). GPU utilisation defaults to 0.6.

**`__call__` (callable interface):** accepts `prompt` string plus optional kwargs:
- `do_sample`, `temperature` (default 0.7)
- `repetition_penalty` (clamped to ≥1.05; default 1.15)
- `max_tokens` or `max_length` (default 4096)
- `stop_sequences` (defaults to boxed-format stops + `<|im_end|>`, `</s>`, `###`, `\n\n\n`)
- `return_format="string"` returns plain text; otherwise returns `[{"generated_text": prompt+text}]` for MedRAG compatibility.

`generate_with_system`: formats Qwen-style `<|im_start|>system...` prompts explicitly; used for verification tasks.

### 5.4 `patch_medrag_for_vllm()`

```python
# run_medrag_vllm.py:409-424
def patch_medrag_for_vllm():
    original_pipeline = transformers.pipeline
    def vllm_pipeline(task, model=None, **kwargs):
        supported_models = ["llama", "qwen", "meta-llama", "mistral", "mixtral", "pmc"]
        if task == "text-generation" and model and any(name in model.lower() for name in supported_models):
            return VLLMWrapper(model, **kwargs)
        else:
            return original_pipeline(task, model=model, **kwargs)
    transformers.pipeline = vllm_pipeline
```

Must be called **before** `MedRAG(...)` so that the HuggingFace model branch in `__init__` picks up vLLM.

### 5.5 `vllm_medrag_answer()`

```python
# run_medrag_vllm.py:339
def vllm_medrag_answer(medrag_instance, question, options=None, k=32,
                        question_id=None, log_dir=None, **kwargs):
    -> (parsed_answer_dict, processed_snippets, scores)
```

- Calls `medrag_instance.answer(question, options, k)`.
- If answer is a string, passes it to `parse_response_standard()`.
- Normalises snippets: adds `contents` field if missing, ensures `title` field.
- Periodic memory cleanup every 50 iterations (gc + cuda cache clear) keyed on `question_id`.

### 5.6 `parse_response_standard()`

```python
# run_medrag_vllm.py:211
def parse_response_standard(raw_response, model_name=None, question_id=None, log_dir=None):
    -> {"step_by_step_thinking": str, "answer_choice": "A"|"B"|"C"|"D"}
```

Parsing waterfall:
1. **PMC-LLaMA specific** (if `"pmc"` in model_name):
   - Look for JSON array format `[{"answer_choice": "X", ...}]`.
   - Look for `### Answer: OPTION X IS CORRECT` pattern.
2. **Generic JSON**: regex for `{...}` block, parse with `json.loads`, validate `step_by_step_thinking` + `answer_choice` fields.
3. **Regex fallback**: try 7 different patterns for `answer_choice` and 5 for `step_by_step_thinking`.
4. **Single-letter fallback**: find lone `A/B/C/D`.
5. **Ultimate fallback**: return `{"answer_choice": "A", "step_by_step_thinking": "Error: ..."}`.

PMC-LLaMA raw responses are always logged to disk (`<question_id>_raw_response.txt`) when `log_dir` is provided.

### 5.7 `init_medrag_with_device_separation()`

```python
# run_medrag_vllm.py:427
def init_medrag_with_device_separation(
    llm_name, rag=True, retriever_name="MedCPT", corpus_name="MedCorp",
    db_dir="./src/data/corpus", corpus_cache=True, HNSW=True,
    retriever_device=None, **kwargs
) -> MedRAG:
```

Convenience wrapper ensuring `retriever_device=RETRIEVER_DEVICE` is always set. Works with any corpus name.

### 5.8 Datasets / Output Format

`run_medrag_vllm.py` does not contain a batch inference loop or dataset iteration. It is a **library** consumed by the debate project. The `run_medrag_with_vllm()` function at the bottom is a demo entry-point (hardcoded question, not a dataset loop).

Output from `vllm_medrag_answer` is a tuple:
```python
(
    {"step_by_step_thinking": "...", "answer_choice": "A"},   # parsed dict
    [{"id": ..., "title": ..., "content": ..., "contents": ..., ...}, ...],  # snippets
    [0.234, 0.189, ...]   # retrieval scores
)
```

---

## 6. Differences from Upstream MedRAG

The upstream is `Teddy-XiongGZ/MedRAG` (ACL 2024). Customisations in this fork:

### 6.1 vLLM Integration
- **New file:** `run_medrag_vllm.py` — entire file is custom.
- `VLLMWrapper` + `patch_medrag_for_vllm()` transparently substitute vLLM for `transformers.pipeline`, providing higher throughput and better memory management.
- `enforce_eager=True` with explicit `max_model_len` overrides avoids the 512-token generation limit that CUDA graph optimisation can impose.
- Model-specific `max_model_len` table in `VLLMWrapper.__init__`.

### 6.2 MedCorp2 Corpus
- **`utils.py:18`**: added `"MedCorp2": ["pubmed", "textbooks", "statpearls", "wikipedia", "umls"]`.
- **`medrag.py:58-63`**: MedCorp2 branch skips single-system instantiation.
- **`medrag.py:168-249`**: `_initialize_source_retrievers()` — new method; 2-GPU pre-warmed architecture, shared embedding model.
- **`medrag.py:385-511`**: `medrag_answer_by_source()` — new method for source-segregated retrieval and context assembly.

### 6.3 Source-Disaggregated Output
- MedCorp2 saves three JSON files per query instead of one, annotating each snippet with `source_type` and `query_used`.
- This allows the calling debate system to inspect which source contributed which evidence.

### 6.4 PMC-LLaMA Fixes
- **`medrag.py:114-124`**: explicit `model_max_length` override to 2048 (PMC-LLaMA's HF config sometimes reports incorrect values causing truncation).
- **`run_medrag_vllm.py:231-265`**: two PMC-LLaMA-specific parsing paths (array format and `### Answer: OPTION X IS CORRECT` format).
- **`run_medrag_vllm.py:191-209`**: raw response logging to disk for debugging.
- `pmc_llama.jinja` and `pmc_llama_new.jinja` custom chat templates.

### 6.5 GPU Device Separation
- `retriever_device` parameter added to `MedRAG.__init__` and propagated to `RetrievalSystem` and `Retriever` (not in upstream).
- `RETRIEVER_DEVICE` / `LLM_DEVICE` constants in `run_medrag_vllm.py` for 2-GPU workflows.

### 6.6 UMLS BM25 ID Handling
- `utils.py:225-237` (`get_relevant_documents`): custom parsing for UMLS BM25 docids (`UMLS_R*_L*_C*`).
- `utils.py:258-296` (`idx2txt`): UMLS-specific document lookup via glob over `umls_run*.jsonl`.
- `DocExtracter.extract` (lines 510-555): similar UMLS fallback resolution.

### 6.7 Git-LFS-Resistant Document Loading
- `DocExtracter.__init__` (lines ~425, 475): checks for and skips Git-LFS pointer files (`"version https://git-lfs"`). Robust `json.JSONDecodeError` handling throughout.

### 6.8 Qwen / Gemma Support
- `medrag.py:125-135`: added `qwen` and `gemma` branches in `__init__` with appropriate context windows. Not in upstream.

---

## 7. `download_medcorp.py`

**Purpose:** Downloads MedRAG corpus chunk files from HuggingFace without requiring Git-LFS.

**What it downloads:**

| Corpus | HuggingFace repo | Size | File count |
|---|---|---|---|
| Textbooks | `MedRAG/textbooks` | ~20 MB | 18 JSONL files |
| Wikipedia | `MedRAG/wikipedia` | ~650 MB | 646 JSONL files |
| PubMed | `MedRAG/pubmed` | ~2 GB | 1166 JSONL files |

StatPearls is not included (already available locally; updated frequently).

**Mechanism:**
1. Uses `huggingface_hub.hf_hub_download()` to fetch each `chunk/*.jsonl` file to a local cache (`/data/wang/junh/hf_cache/`).
2. Creates symlinks from `./src/data/corpus/<corpus>/chunk/<filename>` to the cached file (copies if symlink fails).
3. Interactive: asks user to choose 1 (textbooks only), 2 (textbooks + wikipedia), or 3 (all three).

**Target path:** `./src/data/corpus/` relative to the MedRAG directory. This maps to `db_dir` used in `MedRAG(...)` initialisation.

---

## 8. Documentation Files — Synopses

### `VLLM_SETUP.md`
Step-by-step guide for running MedRAG with vLLM on a local GPU server. Covers prerequisites (pip installs, Java for BM25, GPU memory requirements ≥8 GB), quick-start commands, a code snippet showing `patch_medrag_for_vllm()` + `MedRAG(...)`, all corpus and retriever options, memory optimisation tips (`max_model_len`, `gpu_memory_utilization`, `tensor_parallel_size`), and troubleshooting for OOM, corpus download failures, Java errors, and HuggingFace model access. Includes a sample terminal output showing a successful run.

### `TEMPLATE_EXPLANATION.md`
Documents which chat template format each model family uses and why. Explains the two-category split: models with custom `.jinja` files (PMC-LLaMA, Mixtral/Mistral, Meditron — chosen because their tokenizers lack canonical templates or have inconsistent behavior) vs. models that use HuggingFace built-in templates (Llama-2, Llama-3, Qwen, Gemma). Provides the format string produced by each template, shows the code path where templates are loaded (`medrag.py` constructor), and includes example code for creating new jinja templates and debugging template rendering. Also provides a complete comparison table (6 model families × custom template / template file / format style).

### `CORPUS_DOWNLOAD_SOLUTION.md`
Documents the Git-LFS workaround that makes corpus download root-access-free. The original HuggingFace repos store `.jsonl` chunk files in Git-LFS, which normally requires `git lfs pull`. This solution uses `huggingface_hub.hf_hub_download()` instead, which fetches actual file content directly. The document records that this approach was verified on textbooks (3 files tested, valid JSON confirmed), lists all four corpus sizes, explains the symlink-to-cache storage strategy, and provides the three-option interactive download procedure. Also documents the file layout (cache at `/data/wang/junh/hf_cache/`, corpus at `./src/data/corpus/*/chunk/`) and dependencies (sentence-transformers ≥5.1.1, numpy <2.0).

---

## 9. Test Files (Brief)

- **`test_vllm_integration.py`**: Tests `parse_llama_response` with four hardcoded response strings, then checks corpus directory for JSONL files with real content (vs. LFS pointers). Integration smoke test; not a pytest suite.
- **`test_vllm_setup.py`**: (Not read in detail) Presumably verifies that vLLM can be imported and `VLLMWrapper` can be constructed.
- **`test_pmc_parsing.py`**: Standalone copy of `parse_response_standard` used for unit-testing the PMC-LLaMA parsing paths. Contains the same logic as `run_medrag_vllm.py:211`; confirms the two PMC-specific formats (array format and `### Answer: OPTION X IS CORRECT`) parse correctly before the generic JSON fallback.

---

## 10. Key File References

| Item | Location |
|---|---|
| `MedRAG.__init__` signature | `src/medrag.py:45` |
| Corpus registry `corpus_names` | `src/utils.py:10` |
| Retriever registry `retriever_names` | `src/utils.py:21` |
| `RetrievalSystem.__init__` | `src/utils.py:300` |
| `RetrievalSystem.retrieve` | `src/utils.py:318` |
| `RetrievalSystem.merge` (RRF) | `src/utils.py:347` |
| `Retriever.__init__` | `src/utils.py:135` |
| `Retriever.get_relevant_documents` | `src/utils.py:216` |
| `DocExtracter.__init__` | `src/utils.py:391` |
| `medrag_answer` | `src/medrag.py:310` |
| `medrag_answer_by_source` | `src/medrag.py:385` |
| `i_medrag_answer` | `src/medrag.py:514` |
| `_initialize_source_retrievers` | `src/medrag.py:168` |
| `VLLMWrapper` | `run_medrag_vllm.py:29` |
| `patch_medrag_for_vllm` | `run_medrag_vllm.py:409` |
| `parse_response_standard` | `run_medrag_vllm.py:211` |
| `vllm_medrag_answer` | `run_medrag_vllm.py:339` |
| `init_medrag_with_device_separation` | `run_medrag_vllm.py:427` |
| OpenAI config placeholder | `src/config.py:1` |
| Liquid template definitions | `src/template.py:1-136` |
| Download script entry point | `download_medcorp.py:73` |
