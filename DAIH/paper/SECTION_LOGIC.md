# Section Logic & Current Results — DAIH 2026 Paper

> Companion to `OUTLINE.md` (page budget) and `RESULTS.md` (live number ledger). This doc fixes
> the **argument logic of each section** and records the **finished results with analysis**. It is
> not LaTeX — drafting the real sections comes once the pending cells land.

## Framing (confirmed with author)

Performance-oriented paper whose **contribution is the multi-agent system design** — careful *role
specialization* that lets multiple *small open-weight* models collectively match a *closed-weight*
model on clinical risk prediction. Privacy/HIPAA is the **motivation** for why open-weight matters,
not the subject; on-premise logistics are not foregrounded.

**Working title:** "Who Does the Reasoning? Role-Specialized Multi-Agent Debate Lets Small
Open-Weight Models Match Closed-Weight LLMs on Clinical Risk Prediction." (finalize at the end.)

**Spine = the factorial. Motivation = privacy-forced open-weight. Contribution = role design.**

## Venue constraints (recap)

8-page Research Paper track, COLM 2026 template; **double-blind / mandatory anonymization**
(desk-reject otherwise); non-archival; deadline **2026-06-23 AoE**. No real deployment required —
frame as a deployment-readiness study. Review rewards empiricism + honest evaluation (our
discrimination-vs-threshold honesty is an asset). PHI rule: aggregate metrics only.

---

## Section-by-section logic

Each section states the *job it does* and the *beats* that do it.

### §1 Introduction — "why this, why now, what we claim"
1. **Motivation:** LLM clinical risk prediction is attractive, but closed-weight APIs are infeasible
   on patient data (PHI) → the deployable question is whether *small open-weight* models can do it.
2. **Trend & gap:** "Debate + RAG" pipelines are widely proposed, yet two *design* questions are
   open: (i) does debate earn its cost over single-agent RAG? (ii) which agent *role* drives the
   gains — can a small open model fill it?
3. **Answer, previewed:** a controlled factorial isolates the **integrator** as the decisive role;
   a small open model *there* (not elsewhere) recovers closed-weight-level performance. Lever =
   role design, not model scale.
4. **Contributions C1–C3:** system design first (C1), role-attribution + family-invariance (C2),
   released factorial benchmark/artifacts (C3).
5. **Honest close:** the scale gain is largely a better *operating point*, not better
   discrimination — pre-empts §5.3 reading as a walk-back.

### §2 Related Work — "where we sit, what we don't claim to beat"
- KARE (base pipeline); debate lineage (MAD, MedAgents, ReConcile); medical RAG (MedRAG/MIRAGE);
  structured-EHR SOTA (ColaCare/AdaCare/ConCare).
- **Positioning move:** structured-EHR SOTA (AUROC ~0.88) uses time-series features and is *not* a
  target — we use free-text + RAG debate. Position only against the LLM-agent family (AUROC
  ~0.74–0.78). State explicitly to preempt the reviewer.

### §3 Method — "the design that is the contribution"
1. The 4-call pipeline: two **contrastive label-blind analysts** → **role-specific retrieval**
   (per-agent query-gen + MedRAG/MedCPT + RRF) → **integrator** (sees both analyst outputs +
   similar-patient outcomes, emits a probability).
2. **Why each choice:** label-blindness blocks shortcut/label-leak; role-specific (not shared)
   evidence biases each analyst toward its mandate (risk vs protective); batched Round-1 for
   efficiency. These are the design contributions and are defended here.
3. **The factorial:** 3 swappable positions × {capable open, small open}, **cached reuse** so each
   condition re-runs only the swapped position. Catalog A′–G + H′ ablation.
4. **Metric justification:** **AUC primary** on imbalanced cells; accuracy vs majority baseline for
   context. Stated *before* results so the threshold-vs-discrimination reading is pre-loaded.

### §4 Experimental Setup — "what we ran it on"
- Datasets: MIMIC-III/IV mortality + readmission; MIRAGE 5-dataset QA. Per-cell base rates
  (mortality 5.4% mimic3 / 19.3% mimic4) — these *explain* later AUC gaps.
- Models: capable open (gpt-oss-120b, free OpenAI-compatible endpoint) + small open (Qwen2.5-7B,
  local GPU); GPT-4o only as the closed-weight reference for the invariance check.
- Retrieval: MedRAG MedCorp2 + UMLS, MedCPT, RRF. Prompt regimes: biased/conservative (mortality)
  vs neutral (readmission) — named because it shapes the operating point.

### §5 Results — four pillars
- **§5.1 Debate vs single-agent RAG (Pillar 1).** Show debate earns its cost. H′ = single-agent RAG
  reusing A′ docs with structured output; A′ = full debate, same model/retrieval. The legacy
  single-agent RAG control was ~87% format-fallback (broken) → H′ is the clean test. **(running)**
- **§5.2 Factorial attribution (Pillar 2, spine).** Localize the gain to one role: swapping only
  the integrator (A′→D′) moves recall/F1 hugely; analysts/retrieval swaps do nothing; the GPT-4o
  sub-table shows the same lever under a different family ⇒ role, not family. **(done)**
- **§5.3 Generalization matrix (Pillar 3).** Where it holds at scale, stated honestly: A′/D′ have
  near-identical AUC → D′'s "+recall" is a threshold shift, useful only where AUC is high (mimic4
  mortality). **(mortality done; readmission running)**
- **§5.4 QA breadth / MIRAGE (Pillar 4).** Design generalizes beyond mortality across 3 model sizes.
  **(partial)**

### §6 Discussion
- (a) AUC parity = relocated operating point, not capability — report it as such.
- (b) **Why role design beats scale** (thesis restated through results); brief privacy tie-back.
- (c) Calibration as likely mechanism (forward-pointer; free re-analysis if a reviewer asks why).
- (d) When debate helps vs when single-agent RAG suffices.

### §7 Limitations
- One base model per family; MIMIC-only; modest absolute AUC; no prospective eval; cost not a
  controlled axis.

### §8 Conclusion
- Integrator is the lever; role design (not scale) lets small open models match closed-weight ones;
  debate beats single-agent RAG.

---

## Current results with analysis (finished cells)

> `-` = still running. Numbers from `RESULTS.md` (recompute 2026-06-05); aggregate metrics only.

### Pillar 2 — Factorial attribution (n=100, MIMIC-III mortality, 54% pos by design) ✅

Positions **A**nalysts / **R**etrieval / **I**ntegrator. oss = gpt-oss-120b; Qwen = Qwen2.5-7B.

| Cond | Analysts | Retrieval | Integrator | Acc | Recall | F1 |
|---|---|---|---|---|---|---|
| **A′** | oss | oss | oss | 41.0 | 25.9 | 32.2 |
| B′ | oss | Qwen | oss | 41.0 | 18.5 | 25.3 |
| C′ | Qwen | oss | oss | 37.0 | 24.1 | 29.2 |
| **D′** | oss | oss | **Qwen** | **52.0** | **75.9** | **63.1** |
| E′ | oss | Qwen | Qwen | 43.0 | 24.1 | 31.3 |
| F′ | Qwen | oss | Qwen | 41.0 | 27.8 | 33.7 |
| G | Qwen | Qwen | Qwen | 37.0 | 16.7 | 22.2 |

**Analysis.**
- **Integrator is the single lever.** A′→D′ (swap only the integrator to small-open Qwen): recall
  25.9→75.9 (+50pp), F1 32.2→63.1 (~2×). Retrieval-only (B′) or analyst-only (C′) swaps are
  flat-to-worse (F1 25.3 / 29.2).
- **Small models are positional, not globally good/bad.** All-Qwen (G) is the *worst* cell
  (F1 22.2); the *same* small model in the integrator slot (D′) is the *best*.
- **Lift is conditional on strong upstream analysts.** D′ ≫ F′ (small analysts + small integrator,
  F1 33.7) ≫ G — the small integrator only pays off when fed capable analyst evidence.

### Model-family invariance (GPT-4o ↔ Qwen, same n=100 design) ✅

| Cond | Pipeline | Acc | Recall | F1 |
|---|---|---|---|---|
| A | GPT-4o + GPT-4o + GPT-4o | 45.0 | 3.7 | 6.8 |
| **D** | GPT-4o + GPT-4o + **Qwen** | 53.0 | **57.4** | 56.9 |

**Analysis.** The swap fires identically across families: A→D recall 3.7→57.4 with GPT-4o analysts
mirrors A′→D′'s 25.9→75.9 with oss analysts. The closed-weight integrator is the conservative
driver (rarely predicts positive); the small Qwen integrator is aggressive-but-useful. **This is
the core claim's proof** — the effect is the *integrator role design*, not the model family, so a
small open model in that role recovers closed-weight-mixed performance.

### Pillar 3 — Clinical generalization matrix (full test set) ⚠️ mortality done / readmission running

AUC primary; maj-acc = majority-class baseline for context.

| Cell | n | pos% | Cond | Acc | Recall | F1 | **AUC** | maj-acc |
|---|---|---|---|---|---|---|---|---|
| mortality · mimic3 | 996 | 5.4 | A′ | 74.4 | 22.2 | 8.6 | 0.533 | 94.6 |
| mortality · mimic3 | 996 | 5.4 | D′ | 36.4 | 59.3 | 9.2 | 0.501 | 94.6 |
| mortality · mimic4 | 987 | 19.3 | A′ | 78.7 | 14.7 | 21.1 | **0.730** | 80.7 |
| mortality · mimic4 | 987 | 19.3 | D′ | 66.4 | 61.1 | 41.1 | 0.706 | 80.7 |
| readmission · {mimic3,mimic4} | — | — | A′/D′ | - | - | - | - | - |

**Analysis (the honesty pillar).**
- **AUC is near-identical between A′ and D′ within every cell** (mimic3 0.533 vs 0.501; mimic4 0.730
  vs 0.706) → the "+50pp recall" is a **threshold/operating-point shift, not better discrimination**.
  D′ trades accuracy for recall at the same AUC (mimic4: acc 78.7→66.4, recall 14.7→61.1).
- **Only mimic4 mortality is genuinely discriminative** (AUC 0.73); mimic3 mortality is near-chance
  (~0.50–0.53) intrinsic to its 5.4% base rate + weak text signal — so its +50pp recall is a
  diagnostic artifact. Where it counts (mimic4), the operating-point shift is real and useful
  (F1 21.1→41.1).
- Honest phrasing for §5.3/§6: the role-design win "recovers a clinically useful operating point at
  equal discrimination," not "predicts better."

### Pillar 1 — Debate vs single-agent RAG (H′) `-` running
Only the broken legacy control exists (single-agent RAG ~87% format-fallback). Clean H′ (oss
single-agent, A′ docs, structured output) pending. Tab 3 stays `-`.

### Pillar 4 — MIRAGE QA breadth `-` partial
Debate+CoT rows exist for 3 Qwen sizes across MMLU/MedQA/PubMedQA/BioASQ; MedMCQA column + several
Debate+RAG cells pending. Single-agent best for context: RRF-4 + MedCorp + Qwen3-8B = 78.62 avg.

---

## Anonymization guardrails (double-blind)
No author names/affiliations/acknowledgments/grants; never name the cluster/institution (say "a free
OpenAI-compatible endpoint" / "a local GPU"); artifact link anonymized or deferred to camera-ready;
PHI rule throughout.

## Deferred (not this task)
Real LaTeX section drafting, Fig 1 (pipeline), GitHub→Overleaf import, number refresh via
`compute_metrics.py`. Pending pillars (H′, readmission, MIRAGE gap-fill) drop in as their cells land.
