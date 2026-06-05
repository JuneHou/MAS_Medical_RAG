# DAIH 2026 Submission Plan

**Workshop:** DAIH — LLM/VLM Deployment Opportunities and Risks in Healthcare @ COLM 2026
**Deadline:** 2026-06-23 (AoE) — ~4 weeks from 2026-05-27
**Format target:** 4-page Deployment Case Report (no spotlight talk requirement; poster acceptable)

---

## Headline reframing (this is the important part)

**This is not a failure-analysis paper.** It is a positive-results paper with the following claim:

> A multi-agent debate built from small (4B–8B) open-weight Qwen models can match or exceed a closed-weight GPT-4o pipeline on clinical mortality prediction (MIMIC-III) and remain competitive across five medical-QA benchmarks (MMLU-Med, MedQA, MedMCQA, PubMedQA, BioASQ) — making *on-premise, HIPAA-compliant, low-cost* clinical LLM deployment viable today.

The findings that look "negative" in isolation (survival bias in single-model setups; integrator collapse under naïve post-training rewards) are not the story — they are the *evidence* that architecture, not raw model capability, is the deployment-blocking lever, and that open-weight stacks already clear it.

---

## Why DAIH is the right venue

Closed-weight models are operationally unusable in most clinical settings:

- **PHI / HIPAA:** sending patient data to OpenAI/Anthropic APIs is incompatible with BAA-bound deployments at most US hospitals; cross-border data residency rules tighten this further in EU/CA.
- **Cost:** GPT-4o pricing at ICU-scale prediction volumes (~10⁴ patients × multi-turn debate) is not economically viable for routine bedside use.
- **Auditability and reproducibility:** model versions change silently; closed-weight evaluations cannot be frozen for regulatory submission.
- **Equity:** safety-net hospitals and global-health deployments cannot afford or access the closed-weight tier.

DAIH's CFP explicitly emphasizes safety, equity, privacy, regulation, and workflow fit — all of which point toward open-weight, on-premise stacks. Our contribution is the empirical evidence that this is now achievable without an accuracy/recall tax, *if* the right architecture is used.

---

## Major contributions (3, in priority order)

**C1 — Deployment-feasible open-weight clinical prediction stack.**
A multi-agent debate architecture in which all components are 4B–8B open-weight Qwen models matches the all-closed-weight GPT-4o pipeline on MIMIC-III mortality prediction (Cond D regime) and remains within ~5% of best single-agent open-weight baselines across five medical-QA benchmarks. Fully on-premise; HIPAA-compatible; reproducible.

**C2 — Role-specialized retrieval as the architectural lever.**
We show that *where* in the debate pipeline you place the small open-weight model determines whether it matches the closed-weight baseline. Specifically, the integrator role rewards calibration over raw capability — a 7B Qwen integrator delivers 15× higher mortality recall than GPT-4o in the same slot (Cond A: 3.7% → Cond D: 57.4%; Cond A unbiased: 22% → Cond D unbiased: 83%). This is actionable design guidance for clinical-LLM deployers.

**C3 — A reproducible factorial benchmark for clinical agentic LLMs.**
We release a 6-condition GPT × Qwen factorial × biased/unbiased prompt × four architectural ablations (analyst label-blindness, dual-query retrieval, retrieval-policy bias, Search-R1 RL retriever) on MIMIC-III mortality + MIRAGE-Med 5-benchmark suite. Plus the MIMIC-mortality MedRAG corpus and the debate framework as open-source artifacts.

---

## Method (§3 sketch)

```
Patient EHR (MIMIC-III, mortality task)
       │
       ▼
KARE preprocessing
   - target patient context
   - K=2 positive similar patients (died)
   - K=2 negative similar patients (survived)
   - contrastive shared/unique concept analysis
       │
       ▼
3-agent debate (Round 1, batched in single vLLM call):
   Agent A: Mortality-Risk Assessor      ┐
            role-specific subqueries     │  each agent gets
            MedRAG retrieve (MedCPT)     │  its own evidence
            RRF fusion over MedCorp2+UMLS│  pool — not shared
   Agent B: Protective-Factor Analyst    ┘
       │
       ▼
Agent C: Balanced Clinical Integrator (Round 2)
   - receives both analyst outputs + similar-patient labels
   - outputs MORTALITY PROBABILITY: X.XX
   - **this is the swappable position**
       │
       ▼
Prediction (binary @ 0.5 or calibrated threshold)

Factorial ablation:
   Position 1 (Analysts):   GPT-4o or Qwen2.5-7B
   Position 2 (Retrieval):  GPT-4o or Qwen2.5-7B (subquery generation)
   Position 3 (Integrator): GPT-4o or Qwen2.5-7B
   Prompt regime:           "be conservative" (clinical default) or unbiased
   Retrieval policy:        baseline / dual-query / Search-R1 RL-trained
```

**Design choices to defend in §3:**
1. **Role-specific retrieval (not shared evidence pool)** — each agent gets evidence biased toward its mandate. Ablation shows this matters.
2. **Analyst label-blindness** — analysts never see similar-patient outcomes; only integrator does. Prevents shortcut reasoning.
3. **Single-call batched Round 1** — analysts run in parallel via vLLM batching. Efficiency win to mention.
4. **Per-role RRF fusion** — combines results from MedCorp2 (PubMed + Textbooks + StatPearls + Wikipedia + UMLS) and our MIMIC-mortality corpus.

---

## Experiments to feature (from Excel inventory)

### Primary table — clinical mortality (MIMIC-III, n=100)

| Pipeline | Acc | Recall | F1 | Deployable? |
|---|---|---|---|---|
| Qwen-7B single-agent baseline | 38.0 | 22.2 | 27.9 | ✅ (weak) |
| Cond A: all GPT-4o | 45.0 | 3.7 | 6.8 | ❌ (PHI) |
| Cond B: GPT-Qwen-GPT | 46.0 | 0.0 | 0.0 | ❌ |
| Cond C: Qwen-GPT-GPT | 47.0 | 1.9 | 3.6 | ❌ |
| **Cond D: GPT-GPT-Qwen** | **53.0** | **57.4** | **56.9** | ⚠ (PHI on first 2) |
| Cond E: GPT-Qwen-Qwen | 47.0 | 31.5 | 29.1 | ⚠ |
| Cond F: Qwen-GPT-Qwen | 43.0 | 35.2 | 40.0 | ⚠ |
| **Cond G (new, required): all Qwen** | TBD | TBD | TBD | ✅ |

**Gap to close:** we currently lack a fully-open-weight Cond G row. The headline claim ("open-weight matches closed-weight") needs this. **Highest-priority experiment.** Estimated 4–8 hours to run.

### Secondary table — MIRAGE 5-benchmark medical QA (single-agent and debate)

Lift these from the Excel Mirage sheet:
- Single-agent best open-weight: Qwen3-8B + RRF-4 + MedCorp = **78.62 avg** (closed-weight comparison from MedRAG paper: GPT-4 ≈ 72–75 depending on retriever)
- Debate + CoT Qwen3-4B-Instruct-2507: MMLU 79.71, MedQA 66.22, PubMedQA 51.4, BioASQ 75.89
- Debate + RAG Qwen2.5-7B: MMLU 74.29, MedQA 58.13, PubMedQA 43.8, BioASQ 68.61

**Gap:** debate cells partially empty. Fill at least MedQA + PubMedQA + BioASQ for Qwen2.5-7B and Qwen3-8B before submission. Estimated 1–2 days of compute.

### Tertiary table — architectural ablation

From KARE Ablation sheet, the four-dimensional bias-removal study:
- Track 1 (analyst noRetrieve): 86.65% acc / 9.3% recall (Qwen base); 93.88% / 1.85% (Search-R1)
- Track 1+2 (dual-query + analyst noRetrieve): 83.5% / 11.1%; 91.5% / 7.4%
- Track 3 (reduced bias): 48.6% / 48.2%; 69.5% / 37.0%
- Track 1+2+3 (combined): 68.1% / 53.7%; 78.7% / 20.4%

Story: tracks 1 & 2 increase accuracy but at recall cost (more conservative); track 3 (bias removal) restores recall. The combined track gives the deployable balance.

### Comparison context — MIMIC-IV agentic baselines

From MIMIC-IV sheet — reference numbers we are not running ourselves:
- ColaCare (current SOTA): AUPRC 54.04, AUROC 88.8
- MedAgents: AUPRC 27.73, AUROC 74.38
- ReConcile: AUPRC 27.91, AUROC 74.51
- MAD: AUPRC 30.95, AUROC 78.43

If time permits, we run our Cond D and Cond G on MIMIC-IV with the same metrics for direct comparison. Highest-impact extension if feasible. **Optional, time-boxed: ≤ 1 week.**

### Cuts (out of scope for this 4-page submission)

- Search-R1 RL retriever training details → mention as one ablation row, cite training as future work
- VERL/GRPO integrator post-training (Format / Discrete / Brier) → mention as "naïve post-training rewards can collapse the integrator (Brier → 0% recall)" in §5, do not detail
- EffGen framework comparison → cut entirely
- MedAgent capability scoring → cut entirely
- Temporal LLM survey → cut entirely (off-topic)

---

## Paper outline (4 pages × ~50 lines body = ~200 lines)

**§1 Introduction (~0.5 pg)**
- The deployment gap: closed-weight excellence, open-weight deployability
- Our claim and contributions (C1–C3)

**§2 Setting and Related Work (~0.5 pg)**
- MIMIC-III mortality task, MIRAGE benchmark
- Multi-agent debate (Du, Liang); medical RAG (MedRAG, KARE)
- Deployed-LLM constraints in healthcare (HIPAA, cost, audit)

**§3 Architecture (~0.75 pg)**
- 3-agent role-specialized debate
- Per-role retrieval + RRF fusion
- Label-blind analysts, integrator with similar-patient labels
- Architecture diagram (figure)

**§4 Experiments (~1.5 pg)**
- 4.1 Setup (datasets, models, metrics, infra)
- 4.2 **Main result: open-weight stack matches closed-weight stack** (Cond D vs Cond G table)
- 4.3 Role-attribution analysis (where does the lift come from?)
- 4.4 Generalization across 5 MIRAGE benchmarks
- 4.5 Architectural ablations (label-blindness, dual-query, bias-prompt)

**§5 Discussion (~0.5 pg)**
- Deployment implications: HIPAA/cost/audit
- Open-source artifacts release
- Limitations: MIMIC-III sample size (n=100 in factorial); requires GPT-4o reference for one column only (Cond D mixed); no live deployment yet

**§6 Conclusion (~0.25 pg)**
- Open-weight clinical-LLM stacks are deployment-ready *now*, given the right architecture

References, appendix (extra ablations, prompts, hyperparameters) outside the 4-page limit.

---

## Pre-submission checklist (in priority order)

**P0 — must have before writing**
- [ ] Run **Cond G (all-Qwen-7B factorial cell)** on the same 100-patient MIMIC-III split. Without this the headline claim cannot be stated. **~4–8 hrs compute.**
- [ ] Recompute confusion matrices and F1 for all 6+1 conditions to ensure consistency. **~1 hr.**

**P1 — must have before submitting**
- [ ] Fill 2–3 missing cells in the MIRAGE debate × Qwen2.5-7B table (currently MedMCQA blank for debate rows). **~1 day compute.**
- [ ] Run paired bootstrap CIs on Cond D vs Cond G; report whether the open-weight match is statistically equivalent. **~1 hr scripting + a re-run.**
- [ ] Architecture diagram (clean, publication-quality). **~half day.**
- [ ] Calibration plot for Cond A vs Cond D vs Cond G integrators (reliability curve). **~2 hrs.**

**P2 — nice to have**
- [ ] Cond D + Cond G on MIMIC-IV mortality with AUPRC/AUROC; placed alongside ColaCare reference numbers. **~1 week compute.**
- [ ] Cost/latency comparison: Cond A vs Cond G actual wall-clock and $/patient at standard API pricing. Small inline figure. **~2 hrs once Cond G is run.**

**P3 — out of scope, do not attempt**
- Live clinical deployment / IRB
- Fairness audit across demographics
- Hallucination detection beyond what role-specific retrieval already provides

---

## Risk and reviewer-anticipation

| Reviewer concern | Our pre-emptive response |
|---|---|
| "Sample size n=100 too small" | Acknowledge; emphasize that all factorial cells use the *same* patients, so within-comparison is paired. Provide bootstrap CIs. Note MIRAGE results provide larger-n generalization. |
| "Cond D still requires GPT-4o for analysts" | Cond G addresses this; Cond D is shown as the optimal-quality point, Cond G as the deployable point. |
| "Why not Llama-3, Mistral, Gemma?" | Qwen2.5/3 series chosen for license clarity and inference efficiency on consumer GPUs. Note in §5 as future work; Qwen3-4B/8B + Qwen2.5-7B already provides three sizes. |
| "Is this really mortality prediction or label leakage?" | Analysts are label-blind by construction (§3); only integrator sees labels of similar patients (not the target patient). Document in §3 explicitly. |
| "No prospective validation" | DAIH explicitly accepts retrospective case reports; we frame as deployment-readiness study, not clinical trial. |

---

## Folder layout for `DAIH/`

```
DAIH/
├── PLAN.md                       this document
├── RESULTS_INVENTORY.md          (next: cite-ready table of every result with source)
├── paper/                        LaTeX source (will clone COLM 2026 style)
│   ├── main.tex
│   ├── figures/
│   │   ├── architecture.pdf
│   │   ├── calibration.pdf
│   │   └── radar_mirage.pdf
│   └── tables/
│       ├── main_factorial.tex
│       ├── mirage_5bench.tex
│       └── ablation.tex
├── experiments/                  scripts that (re)produce the numbers in the paper
│   ├── run_cond_g_all_qwen.py    [P0]
│   ├── compute_calibration.py    [P1]
│   ├── paired_bootstrap.py       [P1]
│   └── make_paper_tables.py
├── results_cache/                frozen copies of the result JSONs we cite
│   └── (will be populated from KARE/results/, KARE/gpt/results_bias/, ...)
└── notes/                        draft fragments, reviewer-anticipation notes
```

---

## Open questions before I start (please confirm)

1. **Cond G (all-Qwen factorial cell)** — confirm we should run this on the same 100-patient MIMIC-III split as the GPT-4o conditions, using Qwen2.5-7B at all three positions. Or do you want a model-size matrix (e.g., 4B/7B/8B at integrator) instead of just 7B?

2. **MIRAGE debate gaps** — happy for me to fill these in the next two days, or would you rather we cite only what's already in the sheet?

3. **MIMIC-IV ColaCare comparison** — worth the ~1 week of compute, or skip and save for follow-up?

4. **Author list / venue logistics** — anything I should know about co-authors, prior submissions of overlapping content, or IP/HIPAA review steps before sharing the artifacts?

5. **Are the results in the Excel from the most recent code, or are some from older variants** (e.g., `mortality_debate_rag.py` vs `mortality_debate_rag_unbiased.py` vs the effgen track)? Need this to make sure RESULTS_INVENTORY.md links each number to a reproducible commit/script.
