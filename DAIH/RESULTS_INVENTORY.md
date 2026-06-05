# Results Inventory & Data Availability

Status as of 2026-05-27. Source for all result rows: `/data/wang/junh/githubs/Debate/Related Work Summary.xlsx`.

---

## 1. Confirmations (answers to open questions from PLAN.md)

### Q2 — What are the 3 factorial positions?

Confirmed by reading `KARE/gpt/src/run_condition_D.py:7-13` and `KARE/mortality_debate_rag.py:158-225`:

| Position | What it does | Cost per patient |
|---|---|---|
| **P1: Contrastive analysts** | 2 calls, one per similar patient — compare Target vs Similar #i in terms of shared/unique clinical features. **Label-blind.** | 2 LLM calls |
| **P2: Retriever (query gen + docs)** | 1 LLM call to expand the integrator's needs into `<search_umls>` + `<search_medcorp>` queries. The docs themselves come from MedRAG (no LLM). | 1 LLM call |
| **P3: Integrator** | 1 LLM call that synthesizes both analyst outputs + similar-patient labels + retrieved evidence → MORTALITY PROBABILITY: X.XX | 1 LLM call |

→ 4 LLM calls per patient at full pipeline, all 3 swappable. **The factorial is real and 6-cell (excluding all-Qwen = "Cond G" baseline).**

**Critical efficiency note from `run_condition_D.py:7-13`:** Conditions B/C/D/E/F *reuse cached outputs* from Cond A for the unchanged positions. So we don't re-run all 4 calls per condition — only the swapped ones. This is what made the n=100 GPT-4o factorial affordable, and it carries over: once we run **Cond A′ (all-oss) once on n=996, Conds B′/C′/D′/E′/F′ are nearly free** (only the swapped position needs new calls).

### Q1 — The n=100 stratified sample (from `KARE/gpt/src/sample_select.py:1-15`)

The 100-sample subset is **not random**. It's a diagnostic-stratified slice of the n=996 MIMIC-III mortality test set:

- **54 positives** (ALL positive cases in the test set — every single 5.4%-prevalence mortality case)
- **23 hard negatives** — 1 from N3 (all-baseline-models-wrong) + 22 from N2 (≥2/3 baselines wrong)
- **23 easy negatives** — N0 (all baselines correct)

**Implications for the paper:**
- n=100 has **54% positive base rate** vs n=996's **5.4%**. Accuracy is NOT directly comparable across the two scales.
- n=100 is ideal for **role-attribution analysis** (diagnostic separation of which model fails where).
- n=996 is required for **deployment-realistic absolute numbers** (especially recall/precision at clinical base rate).
- **Plan: run n=100 with oss for role attribution → pick best combination → scale that one to n=996 for the headline.**

---

## 2. Result catalog (Excel → file path → status)

Legend: ✅ kept · ⚠️ partial · ❌ missing · 🆕 to run with gpt-oss-120b

### 2.1 GPT-4o factorial @ n=100 — Excel sheet `GPT-4o` (biased "be conservative" prompt)

| Cond | Pipeline | Acc | Recall | F1 | Source dir | Status |
|---|---|---|---|---|---|---|
| Qwen baseline | single-agent | 38.0 | 22.2 | 27.9 | `KARE/results/single_agent_*` | ✅ |
| A | GPT+GPT+GPT | 45.0 | 3.7 | 6.8 | `KARE/gpt/results_bias/condition_A_gpt_4o/` | ✅ |
| B | GPT+Qwen+GPT | 46.0 | 0.0 | 0.0 | `KARE/gpt/results_bias/condition_B_gpt_4o/` | ✅ |
| C | Qwen+GPT+GPT | 47.0 | 1.9 | 3.6 | `KARE/gpt/results_bias/condition_C_gpt_4o/` | ✅ |
| **D** | **GPT+GPT+Qwen** | **53.0** | **57.4** | **56.9** | `KARE/gpt/results_bias/condition_D_qwen/` | ✅ |
| E | GPT+Qwen+Qwen | 47.0 | 31.5 | 29.1 | `KARE/gpt/results_bias/condition_E_gpt_qwen_qwen/` | ✅ |
| F | Qwen+GPT+Qwen | 43.0 | 35.2 | 40.0 | `KARE/gpt/results_bias/condition_F_qwen_gpt_qwen/` | ✅ |
| G | Qwen+Qwen+Qwen | 37.0 | 16.7 | 22.2 | `DAIH/results/condition_G_all_qwen/` | ✅ |

Plus per-subgroup breakdowns (pos_all, neg_hard, neg_easy) for Conds A–F — all present in the Excel and likely derivable from the saved per-patient JSONs.

### 2.2 GPT-4o factorial @ n=100 — Excel sheet `GPT-4o-Unbias` (unbiased prompt)

| Cond | Acc | Recall | F1 | Status |
|---|---|---|---|---|
| Qwen baseline | 38.0 | 22.2 | 27.9 | ✅ |
| A | 40.0 | 37.0 | 40.0 | ✅ |
| B/C | — | — | — | ❌ never run (gap in Excel) |
| **D** | **48.5** | **83.0** | **63.3** | ✅ |
| E/F | — | — | — | ❌ never run |

→ Unbiased D shows even stronger recall (83%); **filling B/C/E/F at n=100 unbiased is a fast follow-up.**

### 2.3 KARE multi-axis ablation — Excel sheet `KARE Ablation` (~30 cells)

Two prompt regimes ("be conservative" vs unbiased) × {single,multi}-agent × {CoT, RAG} × {Qwen-7B, R1-Qwen-7B}.

Headline rows (full n=996, Qwen-only, biased prompt):
- multi-agent few-shot CoT Qwen-7B: 87.8 / 5.3 / 7.4 / 49.8 / 92.4 — acc dominant, recall collapse
- multi-agent few-shot RAG Qwen-7B: 76.51 / 5.9 / 22.22 / 47.9 / 79.62
- multi-agent few-shot RAG R1-Qwen-7B: 89.46 / 8.2 / 9.3 / 51.55 / 94.1
- Track 1+2+3 (analyst noRetrieve + dual-query + reduced bias) multi-agent RAG Qwen-7B: 68.1 / 9.0 / 53.7 / 47.9 / 68.9
- **Case-based analyst + Search-R1 trained (3 agents): 45.4 / 6.0 / 61.1 / 10.8 / 44.5**

Source: `KARE/results/multi_agent_binary/`, `KARE/results/rag_mor_*`, `KARE/searchr1/checkpoints/searchr1-binary-single-agent-step100/` All ✅ kept (results dir confirmed earlier — 550M total).

### 2.4 2-step KARE — Excel sheet `2step - KARE`

8 rows on alternate retrieval/integrator strategies. Most relevant for the paper:
- Single-step CoT vs RAG vs fallback-to-0 vs retry-twice
- Post-trained integrators: Format / Discrete reward / Brier dense reward — show how naïve RL post-training collapses (Brier → 94.7% acc, 0% recall)
- MedGemma-27B integrator: 54.3 / 5.9 / 90.7 / 16.7 / 11.0 — high recall but low precision

Source: `KARE/verl/checkpoints/` (model checkpoints) + `KARE/results/` (output JSONs). ✅ presumed kept; **need to verify which checkpoints survived.**

### 2.5 MIRAGE benchmark — Excel sheet `Mirage`

5 datasets × 3 retrievers (MedCPT/BM25/RRF-4) × 2 corpora (MedCorp / MedCorp+UMLS) × 3 models (Qwen3-8B / Qwen2.5-7B / Qwen3-4B-2507).

**Best result so far: RRF-4 + MedCorp + Qwen3-8B = 78.62 avg.**

Debate rows (partial — gaps to fill):

| Mode | Model | MMLU | MedQA | MedMCQA | PubMedQA | BioASQ |
|---|---|---|---|---|---|---|
| Debate+CoT | Qwen3-8B | 80.35 | 46.97 | ❌ | 34.80 | 72.17 |
| Debate+CoT | Qwen2.5-7B | 73.09 | 58.76 | ❌ | 44.0 | 74.11 |
| Debate+CoT | Qwen3-4B-2507 | 79.71 | 66.22 | ❌ | 51.4 | 75.89 |
| Debate+RAG | Qwen2.5-7B | 74.29 | 58.13 | ❌ | 43.8 | 68.61 |
| Debate+RAG | Qwen3-4B-2507 | 78.79 | ❌ | ❌ | ❌ | 63.27 |

→ **4 cells empty for Qwen3-4B-2507 debate+RAG (MedQA, MedMCQA, PubMedQA), plus the MedMCQA column.** BioASQ filled 2026-05-31 from `debate_logs_mirage_gapfill_rag/rag_bioasq_MedCPT_Qwen3_4B_Instruct_2507/.../bioasq_results.json` (618 Q, 391 correct = 63.27%). This is the **MIRAGE gap-fill task (~2 days local Qwen compute, no API)**.

Source: `/data/wang/junh/githubs/mirage_medrag/prediction*/` directory layout. ⚠️ partial — single-agent Mirage runs likely intact; debate runs need to be re-launched via `Debate/run_debate_medrag_rag.py`.

### 2.6 MIMIC-IV reference table — Excel sheet `MIMIC-IV`

**Competitor numbers only — your own system has not been run on MIMIC-IV mortality yet.** Reference numbers:

| Method | AUPRC | AUROC |
|---|---|---|
| ColaCare (SOTA) | 54.04 | 88.8 |
| AdaCare | 52.67 | 87.56 |
| ConCare | 49.71 | 87.21 |
| MAD | 30.95 | 78.43 |
| MedAgents | 27.73 | 74.38 |
| ReConcile | 27.91 | 74.51 |

🆕 **To run on MIMIC-IV with our pipeline: Cond G + Cond D′ + Qwen baseline + (optional) Cond A′.** Test set n=987, 19.3% positive (much higher base rate than MIMIC-III's 5.4%, so survival bias should be less catastrophic).

### 2.7 MedAgent + EffGen sheets

`MedAgent`: agent-capability scoring (Qwen2.5-7B: 10.67%; Qwen2.5-14B: 33.67%; Qwen3-8B: 0%). **Likely cut from paper** — useful only to justify Qwen2.5-7B as the open-weight floor.

`EffGen`: scaffold only, near-empty. **Cut entirely.**

`Temporal LLM`, `DataBase`: related-work surveys; populate §2 Related Work directly.

---

## 3. Input data availability (3 locations checked)

### 3.1 `/data/wang/junh/githubs/Debate/KARE/data/` (in-repo, 2.5 GB)

| Folder | Size | Contents | Status |
|---|---|---|---|
| `ehr_data/` | 152 M | MIMIC-III/IV mortality+readmission **test** JSONs + `pateint_*.json` per-patient raw | ✅ (test only) |
| `base_context_qwen/` | 1.5 G | `patient_contexts_*.json` + `patient_embeddings_*.pkl` (Qwen-encoded, MIMIC-III/IV mort+readm) | ✅ |
| `similar_patient_qwen/` | 686 M | `patient_to_top_{1,2}_patient_contexts_*.json` | ✅ |
| `patient_context/similar_patient_debate/` | 228 M | Debate-specific similar-patient contexts | ✅ |

### 3.2 `/data/wang/junh/datasets/KARE/` (external, 519 MB) — **SUPERSET of Debate/KARE/data/ehr_data**

| Folder | Size | Contents | Status |
|---|---|---|---|
| `ehr_data/` | 320 M | **train + val + test** for all 4 splits + `.pkl` files | ✅ richer than Debate copy |
| `patient_context/augmented_context_qwen/` | 124 M | `patient_contexts_*.json` only (no embeddings) | ✅ |
| `patient_context/similar_patient_debate/` | 76 M | **`patient_to_top_1_patient_contexts_mimic3_mortality_improved.json`** — the "_improved" file that `kare_data_adapter.py` prefers | ✅ canonical |

→ **Use `/data/wang/junh/datasets/KARE/` as the primary data source** for any new runs. It's the more complete location.

### 3.3 `/data/wang/junh/githubs/KARE/` (parent project, MB scale)

Original parent KARE repo (graph/, kg_construct/, kg_index/, multi_agent/, baselines/, prediction/, finetune/, ehr_prepare/). **No raw data files** — pure code. Used as a reference codebase only, not at runtime.

### 3.4 MedRAG corpora (~620 GB at `/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus/`)

All present per earlier inventory: MedCorp2, MedCorp, PubMed, Wikipedia, StatPearls, Textbooks, UMLS, mimic_mor. ✅ kept.

---

## 4. Code script availability

| Script | Path | Status | Purpose |
|---|---|---|---|
| Main debate (canonical) | `Debate/KARE/mortality_debate_rag.py` | ✅ | 4-call debate; hard-codes vLLM + MedRAG paths |
| Single-agent baselines | `Debate/KARE/mortality_single_agent_{rag,cot}.py` | ✅ | Single-LLM baselines |
| Data adapter | `Debate/KARE/kare_data_adapter.py` | ✅ | Loads target + similar patients; prefers `_improved` similar files |
| GPT factorial drivers | `Debate/KARE/gpt/src/run_condition_{A..F}.py` | ✅ | Each condition reuses cached outputs from prior conditions |
| GPT utilities | `Debate/KARE/gpt/src/gpt_utils.py`, `gpt_utils_bias.py` | ✅ | OpenAI client; AGENT_PROMPTS quoted from `mortality_debate_rag.py` |
| Sample selector | `Debate/KARE/gpt/src/sample_select.py` | ✅ | 100-sample diagnostic stratification |
| MIRAGE QA debate | `Debate/run_debate_medrag_rag.py` | ✅ | 2-agent + judge for medical QA |
| MIRAGE benchmark runner | `mirage_medrag/MIRAGE/run_benchmark_vllm.py` | ✅ | Single-agent MIRAGE eval |
| MedRAG library | `mirage_medrag/MedRAG/src/medrag.py` + `utils.py` | ✅ | |
| **ARC API ref impl** | `MAST/eval/full_run_eval_graph_inject_api_arc.py` | ✅ | RateLimiter + retry pattern to copy |

---

## 5. Gap analysis — what to run with gpt-oss-120b

### P0 — required for the headline claim

| # | Run | n | Position config | API budget | Local compute |
|---|---|---|---|---|---|
| P0a | Cond A′ @ n=100 (oss baseline) | 100 | oss+oss+oss | 400 calls (~25 min @ 1000/hr) | small Qwen warm |
| P0b | Cond D′ @ n=100 (oss-analyst+oss-retr+Qwen-int) | 100 | reuses A′ analysts+retrieval | ~0 (cache) | A100 for Qwen int |
| P0c | Conds B′/C′/E′/F′ @ n=100 | 100 each | reuse A′ + swap 1 position | ~100–200 each | A100 |
| P0d | Cond G @ n=100 (all-Qwen baseline parity) | 100 | Qwen+Qwen+Qwen | 0 | A100 ~1 hr |
| P0e | Best condition @ n=996 (full deployment scale) | 996 | TBD by P0a–d | up to 4000 calls (~4 hrs) | A100 ~6 hrs |

**Subtotal: ~5500 ARC calls = ~6 hrs API wall clock + ~12 hrs local A100.** Trivial cost; can finish in 2 days.

### P1 — strengthens the paper

| # | Run | Compute |
|---|---|---|
| P1a | Fill MIRAGE debate gaps (MedMCQA for 3 models + Qwen3-4B-2507 RAG row) | ~2 days local A100, no API |
| P1b | Calibration plots (reliability curves for Conds A/D/G integrators) | trivial post-processing |
| P1c | Paired bootstrap CIs for Cond D′ vs G | trivial |
| P1d | Unbiased-prompt B/C/E/F @ n=100 (fill the GPT-4o-Unbias sheet gaps) | small |

### P2 — high-payoff stretch

| # | Run | Compute |
|---|---|---|
| P2a | MIMIC-IV: Cond G + Cond D′ + Qwen baseline + Cond A′ @ n=987 | ~20 hrs API + ~10 hrs A100 |
| P2b | Cost/latency comparison plot | trivial |

### P3 — cut from this submission

- Search-R1 retrieval-policy training (mention as 1 row, don't re-run)
- VERL/GRPO integrator post-training (mention as cautionary tale in §5)
- EffGen framework comparison
- MedAgent capability scoring
- Temporal-LLM survey

---

## 6. Storage notes

**Risks if cleanup happens:**
- `Debate/KARE/data/base_context_qwen/` (1.5 G of embeddings) — only needed if we re-run KARE similarity computation. Safe to delete if we trust the existing `similar_patient_qwen/` outputs. **Recommendation: keep.**
- `Debate/KARE/data/similar_patient_qwen/` (686 M) — input to debate at runtime. **Keep.**
- `Debate/KARE/results/` (550 M) — historical results; many obsolete. **Can prune** the older variants (`single_agent_biased/`, `single_agent_fallback*`) once we confirm we've extracted what we need into RESULTS_INVENTORY.
- `Debate/KARE/searchr1/checkpoints/searchr1-binary-single-agent-step100/` — **KEEP**; cited in the ablation row.
- `Debate/KARE/verl/checkpoints/` — keep at least `format/global_step_57/` (cited in 2step-KARE results).
- `Debate/KARE/effgen/results/` — likely cuttable, results were degenerate per earlier audit.
- `mirage_medrag/MedRAG/src/data/corpus/` (620 G) — **DO NOT DELETE.** Required for any RAG run.

**What I should also save (for reproducibility):**
- Lock the 100-sample manifest at `Debate/KARE/gpt/manifests/gpt_experiment_samples.json` — this is the artifact that defines the diagnostic-stratified subset.
- Lock `Debate/KARE/gpt/manifests/selected_samples_full.parquet`.

---

## 7. Next concrete actions (proposed)

1. **Write `experiments/run_factorial_oss.py`** — wraps existing `mortality_debate_rag.py` with `--analyst_model / --retrieval_model / --integrator_model` triples, each independently `arc:gpt-oss-120b` or local Qwen. Reuses ARC RateLimiter from MAST.
2. **Smoke test on first 5 patients of n=100 manifest** to verify integration.
3. **Run P0a–d (the full n=100 oss factorial) in background** — finishes overnight.
4. **Inspect role-attribution numbers**, pick best mixed config.
5. **Scale that one config + Cond G + Qwen baseline to n=996** for the headline table.

After step 5 we'll have the data to make the *short vs long paper* decision.
