# Results Log — DAIH Workshop

> Live results ledger. `-` = experiment **running or missing** (not yet a number).
> Source-of-truth = the per-patient JSON logs under `DAIH/results/`; historical/past results
> = `/data/wang/junh/githubs/Debate/Related Work Summary.xlsx` and `DAIH/RESULTS_INVENTORY.md`.
> All numbers are aggregate metrics only (no patient-level content — PHI rule).
> Last recompute: 2026-06-05.

---

## Pillar 1 — Debate ablation (does the structure beat single-agent RAG?)

Cond **H′** = single-agent oss RAG (reuses A′ retrieved docs, drops the 2 contrastive analysts).
Treatment = Cond A′ (full debate, same model + retrieval). Isolates the debate structure.

| Cell | H′ Acc | H′ Rec | H′ F1 | H′ AUC | A′ Acc | A′ Rec | A′ F1 | A′ AUC |
|---|---|---|---|---|---|---|---|---|
| mortality · mimic3 | - | - | - | - | 74.4 | 22.2 | 8.6 | 0.533 |
| mortality · mimic4 | - | - | - | - | 78.7 | 14.7 | 21.1 | 0.730 |
| readmission · mimic3 | - | - | - | - | - | - | - | - |
| readmission · mimic4 | - | - | - | - | - | - | - | - |

*H′ control not yet run (script `run_condition_H_oss.py` pending).*

**Legacy single-vs-multi (Qwen2.5-7B, MIMIC-III mortality n=996, from Excel `KARE Ablation`):**

| Arm | Mode | Acc | Sens(Rec) | Macro-F1 | format-fallback |
|---|---|---|---|---|---|
| single-agent | CoT zero-shot | 74.2 | 22.2 | 46.8 | 14 ✅ clean |
| **multi-agent** | CoT few-shot | 87.8 | 7.4 | 49.8 | 47 ✅ clean |
| single-agent | RAG | **2–7** | 9–13 | 2–6 | **868–877 ⚠️ BROKEN** |
| **multi-agent** | RAG few-shot | 76.5 | 22.2 | 47.9 | 5 ✅ clean |

> **Critical:** the legacy single-agent *RAG* runs are ~87% format-fallback (868–877/996) — a
> broken control, **not** usable to claim "debate beats single-agent RAG." This is precisely why
> the clean **H′** control (oss single-agent, reusing A′'s retrieved docs, structured output) is
> required. The single-agent *CoT* arm is clean and can anchor the "RAG vs debate" decomposition.

---

## Pillar 2 — Factorial attribution (n=100, which position drives the gain?) ✅

n=100 diagnostic-stratified MIMIC-III mortality (54% pos by construction). All open-weight.
Positions: **A**nalysts / **R**etrieval / **I**ntegrator; oss = gpt-oss-120b, Qwen = Qwen2.5-7B.

| Cond | Analysts | Retrieval | Integrator | Acc | Recall | F1 |
|---|---|---|---|---|---|---|
| **A′** | oss | oss | oss | 41.0 | 25.9 | 32.2 |
| B′ | oss | Qwen | oss | 41.0 | 18.5 | 25.3 |
| C′ | Qwen | oss | oss | 37.0 | 24.1 | 29.2 |
| **D′** | oss | oss | **Qwen** | **52.0** | **75.9** | **63.1** |
| E′ | oss | Qwen | Qwen | 43.0 | 24.1 | 31.3 |
| F′ | Qwen | oss | Qwen | 41.0 | 27.8 | 33.7 |
| G | Qwen | Qwen | Qwen | 37.0 | 16.7 | 22.2 |

**Finding:** swapping **only the integrator** to local Qwen (A′→D′) lifts recall **25.9 → 75.9
(+50pp)** and F1 **32.2 → 63.1**. Swapping analysts (C′) or retrieval (B′) does nothing. D′ is
the **all-open-weight winner**. *(per-cond AUC: `-`, compute for camera-ready.)*

**Model-family invariance (from Excel `GPT-4o` sheet, same n=100 design, GPT-4o ↔ Qwen):**

| Cond | Pipeline | Acc | Recall | F1 |
|---|---|---|---|---|
| A | GPT+GPT+GPT | 45.0 | 3.7 | 6.8 |
| **D** | GPT+GPT+**Qwen** | 53.0 | **57.4** | 56.9 |

> Same lever, different family: A→D lifts recall **3.7 → 57.4** with GPT-4o analysts, mirroring
> A′→D′'s 25.9 → 75.9 with oss analysts. The authors' own note: *"Integrator choice is the primary
> driver… GPT is the conservative driver, Qwen is aggressive."* The effect is the integrator
> position, not the model family — the core factorial claim.

---

## Pillar 3 — Clinical generalization matrix (full test set) ⚠️

Cond A′ (all-oss debate) and D′ (oss analysts/retrieval + Qwen integrator). **AUC is the primary
metric** (cells imbalanced; majority-baseline acc shown for context).

| Cell | n | pos% | Cond | Acc | Recall | F1 | **AUC** | maj-acc |
|---|---|---|---|---|---|---|---|---|
| mortality · mimic3 | 996 | 5.4 | A′ | 74.4 | 22.2 | 8.6 | 0.533 | 94.6 |
| mortality · mimic3 | 996 | 5.4 | D′ | 36.4 | 59.3 | 9.2 | 0.501 | 94.6 |
| mortality · mimic4 | 987 | 19.3 | A′ | 78.7 | 14.7 | 21.1 | **0.730** | 80.7 |
| mortality · mimic4 | 987 | 19.3 | D′ | 66.4 | 61.1 | 41.1 | 0.706 | 80.7 |
| readmission · mimic3 | 996 | - | A′ | - | - | - | - | - |
| readmission · mimic3 | 996 | - | D′ | - | - | - | - | - |
| readmission · mimic4 | 1013 | - | A′ | - | - | - | - | - |
| readmission · mimic4 | 1013 | - | D′ | - | - | - | - | - |

**Findings (mortality):**
- **mimic4 mortality is the only genuinely discriminative cell** (AUC 0.73). mimic3 mortality is
  near-chance (0.50–0.53) — intrinsic to its 5.4% base rate + signal.
- **A′ and D′ have near-identical AUC per cell** → the integrator swap moves the **operating point
  (threshold)**, not discrimination. D′ trades accuracy for recall (e.g. mimic4: acc 78.7→66.4,
  rec 14.7→61.1) at the *same* AUC. The factorial's "+50pp recall" is this threshold shift; it is
  only clinically meaningful where AUC is high (mimic4 mortality).
- Readmission cells: re-running after the integrator-prompt bug fix (`-`).

---

## Pillar 4 — QA breadth (MIRAGE gap-fill, on ARC) `-`

Debate runs across 5 medical-QA datasets × 3 models. Scheduled on ARC (corpus staging fixed
2026-06-05). Only BioASQ landed so far.

| Mode | Model | MMLU | MedQA | MedMCQA | PubMedQA | BioASQ |
|---|---|---|---|---|---|---|
| Debate+CoT | Qwen3-8B | 80.35 | 46.97 | - | 34.80 | 72.17 |
| Debate+CoT | Qwen2.5-7B | 73.09 | 58.76 | - | 44.0 | 74.11 |
| Debate+CoT | Qwen3-4B-2507 | 79.71 | 66.22 | - | 51.4 | 75.89 |
| Debate+RAG | Qwen2.5-7B | 74.29 | 58.13 | - | 43.8 | 68.61 |
| Debate+RAG | Qwen3-4B-2507 | 78.79 | - | - | - | **63.27** |

*(Single-agent MIRAGE best so far for context: RRF-4 + MedCorp + Qwen3-8B = 78.62 avg.)*

---

## Reference: clinical EHR SOTA (competitor numbers, MIMIC-IV mortality) — for Related Work / Discussion

| Method | AUPRC | AUROC | Note |
|---|---|---|---|
| ColaCare (SOTA) | 54.04 | 88.8 | structured time-series EHR — NOT directly comparable to our text+RAG setup |
| AdaCare | 52.67 | 87.56 | |
| ConCare | 49.71 | 87.21 | |
| MAD | 30.95 | 78.43 | LLM multi-agent debate |
| MedAgents | 27.73 | 74.38 | |
| ReConcile | 27.91 | 74.51 | |

> Framing note: our 0.73 AUC is **not** comparable to ColaCare's 0.89 (they use structured
> time-series features; we use free-text + RAG debate). Use these only to position the
> *LLM-agent* family (MAD/MedAgents/ReConcile), not as a target to beat.
