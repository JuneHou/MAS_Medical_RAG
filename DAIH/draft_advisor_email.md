**Subject:** DAIH workshop submission — project status and proposed framing

Hi [Advisor name],

I'd like to put our multi-agent medical debate work in for the DAIH workshop at COLM 2026 (deployment-focused; deadline June 23, 4-page case-report format). Wanted to walk you through where we are, why I think it's a good venue fit, and what's left before submission.

---

## 1. What we've completed so far

On **MIMIC-III mortality prediction**:

- A **6-condition factorial ablation** (GPT-4o × Qwen2.5-7B at Analyst / Retrieval / Integrator positions) on a 100-patient stratified sample (54 positives + 46 negatives), under both biased ("be conservative") and unbiased prompts. The standout result is **Condition D (GPT analysts + GPT retrieval + Qwen integrator): 53% accuracy, 57% mortality recall** — vs Condition A (all GPT-4o): 45% accuracy, 3.7% recall. Swapping the small open-weight integrator delivered ~15× recall improvement.
- A **multi-axis architectural ablation** on the full n=996 test set: prompt bias × {single, multi}-agent × {CoT, RAG} × {Qwen-7B, Search-R1-trained Qwen-7B}, covering ~30 cells.
- **Search-R1 RL-trained retriever** (Qwen-7B, GRPO, MedRAG server) plugged into the debate as integrator.
- **VERL/GRPO post-training** of the integrator on Format / Discrete / Brier rewards (showing that naïve RL rewards can collapse calibration).

On **MIRAGE medical-QA benchmarks** (MMLU-Med, MedQA, MedMCQA, PubMedQA, BioASQ-Y/N):

- Single-agent runs across 3 retrievers (MedCPT, BM25, RRF-4), 2 corpora (MedCorp, MedCorp+UMLS), and 3 Qwen models (Qwen3-8B, Qwen2.5-7B, Qwen3-4B-Instruct-2507). **Best: RRF-4 + MedCorp + Qwen3-8B = 78.6 avg.**
- Partial debate runs filled in for CoT and RAG modes.

Plus retrieval infrastructure (custom MedRAG corpus with MIMIC-mortality records), the KARE data adapter pipeline, and reference numbers compiled from ColaCare / MAD / MedAgents / ReConcile on MIMIC-IV.

---

## 2. Why DAIH is the right venue

The CFP explicitly asks for safety, equity, privacy, deployment readiness — "beyond benchmark accuracy." Our work hits the target on:

- **Domain match:** clinical mortality prediction on MIMIC-III/IV is exactly DAIH's scope, not a general-NLP retrofit.
- **Deployment relevance (privacy/cost):** closed-weight APIs cannot legally process PHI under most hospital BAAs; our results show that an **all-open-weight Qwen 4–8B debate stack matches GPT-4o-based pipelines** on the safety-critical metric (mortality recall), enabling on-premise inference at zero per-query cost.
- **Safety/reliability story:** our factorial isolates *where in the agentic pipeline calibration matters most* (the integrator), giving deployers actionable architectural guidance rather than just a leaderboard number.
- **Failure-mode analysis welcomed:** the survival-bias collapse under naïve RL rewards (Brier → 94.7% accuracy / 0% recall) is the kind of operational lesson DAIH explicitly wants.
- **4-page Deployment Case Report format** matches our work's scope perfectly.

---

## 3. Proposed framing

Headline claim: *"Role-specialized multi-agent debate makes on-premise open-weight clinical mortality prediction deployment-ready, matching closed-weight GPT-4o on MIMIC-III at zero inference cost — driven by the finding that calibration-critical integration is the lever, and small open-weight models are sufficient there."*

Three contributions:

- **C1 (system):** a 3-agent debate architecture with role-specific retrieval and per-role RRF fusion, designed for clinical-scale longitudinal EHR context.
- **C2 (empirical):** the integrator-choice insight, demonstrated via the 6-condition factorial and verified to *transfer* from GPT-4o to gpt-oss-120b (open-weight). The small Qwen integrator delivers the largest single performance lever in the pipeline.
- **C3 (artifact):** open-source release of the factorial benchmark, the MIMIC-mortality MedRAG corpus, and the debate framework — including the bias-prompt vs unbiased comparison as a calibration stress test.

---

## 4. Remaining experiments before submission

I have free access to VT ARC's gpt-oss-120b endpoint, so the GPT-4o substitution work below adds no marginal cost.

**P0 — required for the headline (≈3 days):**

- Run Conditions A′–F′ on the same 100-patient manifest with gpt-oss-120b to verify transferability of the role-attribution finding from closed-weight to open-weight (already partially done; smoke test on 5 patients confirmed the pattern qualitatively).
- Scale the winning all-open-weight configuration + Cond G (all-Qwen) + Qwen baseline to **MIMIC-III n=996** for deployment-realistic recall/precision at the 5.4% positive base rate.

**P1 — strengthens the paper (≈2–3 days):**

- Fill the MIRAGE debate cells (MedMCQA column + Qwen3-4B-2507 RAG row).
- Calibration plots (reliability curves) for Cond A vs Cond D vs all-Qwen integrators.
- Paired bootstrap CIs to confirm the open-weight match is statistically supported.

**P2 — high-payoff stretch (≈1 week):**

- Run our pipeline on **MIMIC-IV mortality (n=987, 19.3% positive)** and report AUPRC / AUROC alongside ColaCare (SOTA: 54.0 / 88.8). This gives a direct head-to-head against the published baseline on the same dataset.

Total compute: ~1 week of local A100 time + ~40 hrs of ARC API calls (free). Comfortably within the June 23 deadline if I start the P0 runs this week.

---

I'd appreciate your thoughts on the framing — particularly whether **C1+C2+C3 is the right scope for the 4-page format**, or whether you'd want to lean harder into one of the three. I'd also welcome input on whether the MIMIC-IV comparison is worth the extra compute or if MIMIC-III alone tells the story.

Happy to set up a quick meeting to discuss before kicking off the large runs.

Best,
Jun
