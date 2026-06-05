# Project Overview — Multi-Agent Medical Debate on MedRAG

This document is the cross-repo index for two related research projects:

| Repo | Path | Role |
|---|---|---|
| **mirage_medrag** | `/data/wang/junh/githubs/mirage_medrag/` | Upstream retrieval backbone — customized MedRAG + MIRAGE benchmark |
| **Debate** | `/data/wang/junh/githubs/Debate/` | Downstream multi-agent debate built on top of MedRAG, plus KARE-style mortality prediction |

The retrieval layer (MedCPT + MedCorp2 + UMLS + a custom MIMIC-mortality corpus) lives in `mirage_medrag`. The debate / mortality-prediction / RL / GPT-ablation tracks all live in `Debate`. Every retrieval call in `Debate` hard-codes the path `/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus`.

Detailed per-module summaries live alongside this file in `module_summaries/`. This overview is the map.

---

## At-a-glance map

```
mirage_medrag/                        ~620 GB corpus on disk
├── MedRAG/                           customized MedRAG core
│   ├── src/medrag.py                 main MedRAG class
│   ├── src/utils.py                  RetrievalSystem + per-corpus retrievers
│   ├── src/template.py + templates/  prompt templates (incl. PMC-LLaMA fixes)
│   ├── src/data/                     per-corpus builders (pubmed, statpearls,
│   │                                 textbooks, wikipedia, umls, mimic_mor)
│   ├── src/data/corpus/              indexed corpora + FAISS shards (~620 GB)
│   ├── run_medrag_vllm.py            vLLM-patched inference runner
│   └── download_medcorp.py
├── MIRAGE/                           7,663-question medical-QA benchmark
│   ├── run_benchmark_vllm.py
│   ├── benchmark.json                MedQA, MedMCQA, PubMedQA*, BioASQ, MMLU-Med
│   └── src/{evaluate,utils}.py
├── prediction/                       baseline MIRAGE results
├── prediction_by_source/             per-corpus result breakdowns (MedCorp2)
├── prediction_by_source_new/         most recent ablation (MMLU + Qwen2.5-7B)
└── test_pmc_llama_consistency.py     regression test for the parsing fix

Debate/                               multi-agent debate + KARE mortality
├── run_debate_medrag_rag.py          2-agent Analyst/Skeptic + Judge QA debate
├── run_debate_medrag_inter.py        intermediate variant (CoT + RAG)
├── debate_logs / debate_logs_boxed   per-question QA debate transcripts
└── KARE/                             mortality-prediction sub-project
    ├── mortality_debate_rag.py       4-agent canonical debate
    ├── mortality_debate_rag_{fast,binary,unbiased,cot}.py
    │                                 variants of the above
    ├── old_mortality_debate_rag.py   legacy
    ├── run_kare_debate_mortality{,_fast}.py
    ├── kare_data_adapter.py          loads KARE EHR + similar-patient context
    ├── kare_contrastive_preprocessing.py
    ├── improved_faiss_retrieval.py   offline KARE-side FAISS preprocessing
    ├── analyze_* / fix_*             post-hoc analysis utilities
    ├── data/                         MIMIC-III/IV patient contexts + embeddings
    ├── results/ results_unbiased/    experimental outputs
    ├── effgen/                       newest port (Feb 2026) — effGen framework
    │                                 + agentic-retrieve ReAct variant
    ├── gpt/                          GPT condition A–F factorial ablation
    ├── searchr1/                     Search-R1 RL retriever (GRPO)
    └── verl/                         VERL/GRPO integrator training
```

---

## Module summary index

Each entry links to its detailed write-up. Lines after the bullet are a one-sentence hook.

### mirage_medrag (upstream)

- [`mirage_medrag/01_toplevel.md`](mirage_medrag/01_toplevel.md) — Conda env, debug guide, PMC-LLaMA consistency-test rationale.
- [`mirage_medrag/02_medrag_core.md`](mirage_medrag/02_medrag_core.md) — `MedRAG.answer()` pipeline, `RetrievalSystem` fan-out + RRF fusion, prompt templates, vLLM monkey-patch.
- [`mirage_medrag/03_corpus_builders.md`](mirage_medrag/03_corpus_builders.md) — Per-corpus chunking + indexing scripts; the **custom MIMIC-mortality corpus** and its design.
- [`mirage_medrag/04_mirage_benchmark.md`](mirage_medrag/04_mirage_benchmark.md) — `benchmark.json` schema, `run_benchmark_vllm.py` CLI, evaluator, prediction-dir naming conventions.

### Debate (downstream)

- [`01_qa_debate_toplevel.md`](01_qa_debate_toplevel.md) — 2-agent Analyst/Skeptic + Judge on medical QA; `_rag` vs `_inter` differences.
- [`02_kare_debate_core.md`](02_kare_debate_core.md) — 4-agent KARE mortality debate; the `_rag` / `_fast` / `_binary` / `_unbiased` / `_cot` / `old_` variant deltas.
- [`03_kare_data_and_utils.md`](03_kare_data_and_utils.md) — `KAREDataAdapter`, contrastive preprocessing, analyzers, single-agent baselines.
- [`04_kare_effgen.md`](04_kare_effgen.md) — Newest track (Feb 2026): port to the third-party effGen framework + agentic-retrieve ReAct variant.
- [`05_kare_searchr1.md`](05_kare_searchr1.md) — Search-R1 RL retriever (GRPO), local MedRAG HTTP server, FSDP→HF conversion.
- [`06_kare_verl.md`](06_kare_verl.md) — VERL/GRPO RL training of the integrator (format → Brier-calibration phases).
- [`07_kare_gpt.md`](07_kare_gpt.md) — GPT condition A–F factorial ablation (GPT/Qwen at Analyst × Retrieval × Integrator); survival-bias track.

---

## Data flow: how the pieces fit

### 1. Retrieval layer (mirage_medrag)
```
question / patient context
        |
        v
RetrievalSystem.retrieve()        ← MedRAG/src/utils.py
   [retriever] × [corpus] grid
   - MedCPT (default), BM25, Contriever, SPECTER
   - per-corpus: PubMed / Textbooks / StatPearls / Wikipedia / UMLS / MIMIC-mor
   - MedCorp = first 4; MedCorp2 = first 4 + UMLS
        |
        | RRF fusion across corpora
        v
top-k snippets (with source labels if by_source mode)
```

### 2. QA debate (Debate/run_debate_medrag_rag.py)
```
question
   |
   +-> Analyst subqueries  --> MedRAG --> Analyst-biased snippets
   |                                          \
   |                                           v
   +-> Skeptic subqueries  --> MedRAG --> Skeptic-biased snippets
                                              /
                                       Round 1 answers
                                              |
                                       (Round 2 with tool calls in _rag.py)
                                              |
                                              v
                                            Judge --> JSON
                                              |
                                              v
                              debate_logs/{dataset}_MedCPT_{corpus}/
```

### 3. KARE mortality debate (Debate/KARE/mortality_debate_rag.py)
```
patient EHR (MIMIC-III/IV)
   |
   v
KAREDataAdapter            ← KARE/kare_data_adapter.py
   - target_context
   - positive_similars (similar patients who died)
   - negative_similars (similar patients who survived)
   - ground_truth (hidden from analyst agents)
   |
   v
contrastive preprocessing  ← KARE/kare_contrastive_preprocessing.py
   - shared vs unique clinical concepts
   - label-blind formatting for analysts
   |
   v
Round 1 (batched in one vLLM call):
   - Mortality-Risk Assessor   (retrieves via MedRAG dual <search_umls>+<search_medcorp>)
   - Protective-Factor Analyst (same)
   |
   v
Round 2:
   - Balanced Clinical Integrator
     - sees analyst outputs + labels of similar patients
     - explicitly survival-biased prompt
     - outputs MORTALITY PROBABILITY: X.XX
   |
   v
results/<config_dir>/kare_debate_mortality_results.json
```

### 4. RL tracks (parallel offshoots)
```
searchr1/   trains a single-agent retriever-predictor end-to-end with GRPO,
            using <search>...</search> action tokens against the local MedRAG server.
            Output checkpoint plugged in as --integrator_model in the debate.

verl/       trains only the integrator (Agent 4) with GRPO.
            Phase 1: format-enforcement reward (binary regex on output line)
            Phase 2: Brier-score calibration reward
            Both base-model = Qwen/Qwen2.5-7B-Instruct.
```

---

## Status and results

| Track | When | Status | Outcome |
|---|---|---|---|
| QA debate (`run_debate_medrag_rag.py`) | Nov 2025 | Stable; logs in `debate_logs/` | Per-dataset acc tracked in `*_results.json` |
| KARE 4-agent debate (canonical) | Nov–Dec 2025 | Many variants exist; bugs noted below | Survival-bias problem dominates |
| searchr1 RL retriever | Dec 2025 | Checkpoint `step100_MedCPT` exists | Acc 0.895 but mortality recall only 0.09 |
| verl/GRPO integrator | Dec 2025, 1-week sprint, 29 wandb runs | Phase 1a checkpoint exists (`format_enforcer_7b_step57`) | Phase 2 calibration unfinished |
| gpt condition A–F | Jan 2026 | Complete factorial done | **Best result of project**: Condition D (GPT analysts + GPT retrieval + Qwen integrator) → 53% acc, 57.4% recall |
| effgen port + agentic-retrieve | Feb 2026 (newest) | Single-agent runs complete; debate version has no result file yet | Available runs degenerate to ~0% acc / ~100% fallback |

---

## Recurring issues across the whole project

### 1. Survival bias is the central, unresolved problem
Every track wrestles with it: the canonical KARE prompt is explicitly survival-biased ("mortality is rare"); `_unbiased.py` softens that language; `gpt_utils_bias.py` makes the bias *more* explicit and the `results_bias/` directory holds the actual experiments; the verl Brier-calibration phase was meant to attack it. **The one positive result (gpt Condition D, 57.4% recall) came not from prompt engineering but from swapping the integrator model** (gpt-4o → Qwen2.5-7B). Suggests integrator-model calibration matters more than prompt phrasing here.

### 2. Three concrete bugs in `KARE/mortality_debate_rag.py`
- `fallback = 1 - ground_truth` — when the integrator's probability output can't be parsed, the fallback is set to the *wrong* label by construction. Live code, not dead. Spotted in module 2.
- `round1_k` / `round3_k` CLI args are cosmetic — they only name the output directory; the actual retrieval `k` is hard-coded to 8.
- `format_integrator_history_with_labels` is imported everywhere but never called.

### 3. Logged-but-uncalled `_rag.py` quirks (QA debate)
- Timing-key mismatch in `_rag.py` (module 1).
- `SKEPTIC_SYS` dead prompt.
- `_inter.py` log-dir CLI flag bug + tuple-vs-string PARSE_FAILED comparison + missing per-question write that breaks `--resume`.

### 4. Hard-coded paths everywhere
Every script in `Debate/` hard-codes `/data/wang/junh/githubs/mirage_medrag/MedRAG/...` and `/data/wang/junh/.cache/huggingface`. Repo is not portable; moving either repo will break everything silently (the MedRAG retriever will return empty results, not crash loudly).

### 5. Two `mimic_mor.jsonl` files disagree
`mirage_medrag/MedRAG/src/data/mimic_mor.jsonl` (31 MB, FAISS-experiment format) vs `mirage_medrag/MedRAG/src/data/corpus/mimic_mor/chunk/mimic_mor.jsonl` (~15 MB, canonical production format). Always use the latter.

### 6. PMC-LLaMA answer-parsing trap (already fixed)
A custom array-format regex parser silently defaulted to "A" when it failed, tanking accuracy to ~25%. Documented in `ANSWER_PARSING_ANALYSIS.md` + `PMC_LLAMA_CONSISTENCY_NOTES.md`. Fix: drop the custom parser, use unified `parse_response_standard()`. `test_pmc_llama_consistency.py` is the regression test.

### 7. Variant proliferation
The codebase has many near-duplicate scripts (`_rag`, `_fast`, `_binary`, `_unbiased`, `_cot`, `old_`, `agentic_retrieve/...`). Typical research-leftover state — diffs between them are small but load-bearing. The variant matrix is documented inside each per-module summary.

---

## If you come back to this and want to revive it

The highest-signal direction the data suggests:

1. **Integrator-model swap is the lever**, not prompts. The gpt Condition D result (Qwen integrator beats gpt-4o integrator on mortality recall by ~14×) is the strongest signal in the project. Next step: try the verl-trained Brier-calibrated integrator (Phase 2, unfinished) in the same Condition-D-style configuration.

2. **Fix the `fallback = 1 - ground_truth` bug** before re-running anything — it silently destroys the worst-case behavior of the canonical debate.

3. **The effgen agentic-retrieve debate has no result yet** — that's the most recent code change, and the framework comparison was the original motivation for that track. Running it once would close the loop.

4. **Search-R1's calibrated-probability reward** (`reward_functions/kare_mortality_probability.py`) was a reasonable design but only reached step 100. Continuing training is cheap.

5. **Tracks to abandon if pruning**: `old_mortality_debate_rag.py`, the duplicate `mortality_debate_rag_fast/binary` variants once the canonical version's bugs are fixed, and the `prediction/` (non-by-source) MIRAGE results which are superseded by `prediction_by_source_new/`.

---

## Conventions / glossary

- **MedCorp** = PubMed + Textbooks + StatPearls + Wikipedia (4 sources)
- **MedCorp2** = MedCorp + UMLS (5 sources)
- **MedCPT** = the default dense retriever (asymmetric query/doc encoders)
- **RRF** = Reciprocal Rank Fusion (`K=60` everywhere)
- **KARE patient ID** = `"{patient_id}_{visit_index}"` (load-bearing — see `kare_data_adapter.py:214`)
- **"by_source"** = MIRAGE prediction mode that returns snippets keyed by which sub-corpus produced them, instead of fusing first
- **`step100_MedCPT`** = the Search-R1 RL checkpoint frequently used as `--integrator_model` in result-dir names

---

## Files in this directory

```
module_summaries/
├── PROJECT_OVERVIEW.md             this file
├── 01_qa_debate_toplevel.md        Debate: top-level QA debate
├── 02_kare_debate_core.md          Debate: KARE 4-agent debate
├── 03_kare_data_and_utils.md       Debate: KARE adapters + analyzers
├── 04_kare_effgen.md               Debate: effGen port (newest)
├── 05_kare_searchr1.md             Debate: Search-R1 RL retriever
├── 06_kare_verl.md                 Debate: VERL/GRPO integrator
├── 07_kare_gpt.md                  Debate: GPT condition A–F ablation
└── mirage_medrag/
    ├── 01_toplevel.md              MedRAG/MIRAGE top-level
    ├── 02_medrag_core.md           MedRAG core library
    ├── 03_corpus_builders.md       per-corpus builders + MIMIC-mor
    └── 04_mirage_benchmark.md      MIRAGE benchmark + result dirs
```

Total: **11 module summaries, ~4,400 lines / ~250 KB.**
