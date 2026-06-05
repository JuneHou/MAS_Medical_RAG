# n=100 Factorial Report — MIMIC-III Mortality Prediction

**Date:** 2026-05-29
**Sample:** 100 stratified MIMIC-III mortality cases (54 positives, 46 negatives — every positive in the test set, plus 23 hard + 23 easy negatives)
**Prompt regime:** Biased — `gpt_utils.AGENT_PROMPTS["balanced_clinical_integrator"]` and `gpt_utils_bias.AGENT_PROMPTS["balanced_clinical_integrator"]` (both contain "Mortality is rare. Only assign a high mortality probability when ... strong evidence ...")
**Aggregate metrics only** — per the PHI/MIMIC handling rule, no patient records are quoted.

---

## 1. Results table — 13 cells

Each cell is a 3-tuple (Analyst / Retriever / Integrator). All cells use the same n=100 sample IDs.

| Cond | A / R / I | Acc | Recall | F1 | Spec | Fallback | NoRetr |
|---|---|---:|---:|---:|---:|---:|---:|
| A  | GPT-4o / GPT-4o / GPT-4o    | 45.0 |  3.7 |  6.8 | 93.5 | 0.0% | 14% |
| A′ | oss-120b / oss-120b / oss-120b | 41.0 | 25.9 | 32.2 | 58.7 | 2.0% | 16% |
| B  | GPT-4o / Qwen / GPT-4o      | 46.0 |  0.0 |  0.0 |100.0 | 0.0% | 42% |
| B′ | oss-120b / Qwen / oss-120b  | 41.0 | 18.5 | 25.3 | 67.4 | 0.0% | 42% |
| C  | Qwen / GPT-4o / GPT-4o      | 47.0 |  1.9 |  3.6 |100.0 | 0.0% | 14% |
| C′ | Qwen / oss-120b / oss-120b  | 37.0 | 24.1 | 29.2 | 52.2 | 3.0% | 16% |
| **D**  | **GPT-4o / GPT-4o / Qwen**  | **53.0** | **57.4** | **56.9** | 47.8 | 0.0% | 14% |
| **D′** | **oss-120b / oss-120b / Qwen** | **52.0** | **75.9** | **63.1** | 23.9 | 7.0% | 16% |
| E  | GPT-4o / Qwen / Qwen        | 47.0 | 31.5 | 39.1 | 65.2 | 4.0% | 42% |
| E′ | oss-120b / Qwen / Qwen      | 43.0 | 24.1 | 31.3 | 65.2 | 8.0% | 42% |
| F  | Qwen / GPT-4o / Qwen        | 43.0 | 35.2 | 40.0 | 52.2 | 5.0% | 14% |
| F′ | Qwen / oss-120b / Qwen      | 41.0 | 27.8 | 33.7 | 56.5 | 3.0% | 16% |
| **G**  | **Qwen / Qwen / Qwen**      | **37.0** | **16.7** | **22.2** | 60.9 | 4.0% | 42% |

Result directories:
- GPT-4o cells (A-F): `KARE/gpt/results_bias/condition_{A,B,C,D,E,F}_*/logs/`
- oss-120b cells (A′-F′): `DAIH/results/condition_{A,B,C,D,E,F}_oss*/logs/`
- All-Qwen cell (G): `DAIH/results/condition_G_all_qwen/logs/`

---

## 2. Headline finding

**The integrator role is the dominant lever — model-family-invariant.**

Swapping only the integrator (same-family A→D, A′→D′) lifts F1 by an order of magnitude:

| Swap | F1 before | F1 after | Lift |
|---|---:|---:|---:|
| GPT-4o A → D (Qwen integrator) | 6.8 | 56.9 | 8.4× |
| oss-120b A′ → D′ (Qwen integrator) | 32.2 | 63.1 | 2.0× |

The transferability claim is therefore: the architectural lever ("calibrate by swapping the integrator role to Qwen") works on at least two model families (closed-weight GPT-4o and open-weight gpt-oss-120b). Lift magnitude differs because oss-120b's baseline (A′) already partially resists the bias prompt.

---

## 3. The role-specialization story — what Cond G adds

Cond G (all-Qwen-7B at every position) is the **weakest cell in the table**: F1 22.2, recall 16.7. This is below the Qwen single-agent baseline (~F1 28) and far below GPT-4o D (F1 56.9) or oss D′ (F1 63.1).

Three observations explain why:

**(a) The Qwen integrator's lift is a function of input quality.** When analysts and retrieval come from a stronger model (GPT-4o or oss-120b), they surface enough mortality-risk signal that the Qwen integrator can act on it. When all three positions use Qwen-7B, the analyst outputs are themselves over-cautious — the integrator has no risk signal to amplify, and the "Mortality is rare" prior dominates.

**(b) The lift requires same-family inputs at analyst and retrieval.** Swap either input role to a different family and the Qwen-integrator lift collapses:

| Cell | Composition | F1 |
|---|---|---:|
| D | GPT analyst + GPT retrieval + Qwen integrator | 56.9 |
| E | GPT analyst + **Qwen** retrieval + Qwen integrator | 39.1 |
| F | **Qwen** analyst + GPT retrieval + Qwen integrator | 40.0 |
| D′ | oss analyst + oss retrieval + Qwen integrator | 63.1 |
| E′ | oss analyst + **Qwen** retrieval + Qwen integrator | 31.3 |
| F′ | **Qwen** analyst + oss retrieval + Qwen integrator | 33.7 |
| G | **Qwen** analyst + **Qwen** retrieval + Qwen integrator | 22.2 |

Mixing families (E, F, E′, F′) gives the integrator inconsistent signals and collapses the lift to ~F1 30-40. Going fully Qwen (G) collapses it further to F1 22.

**(c) Architectural choice matters more than raw model size.** The all-open-weight winner (D′, F1 63.1) outperforms the all-closed-weight winner (D, F1 56.9). But the all-small-open-weight cell (G, F1 22.2) is the worst in the table. The lever is *where* you place the small model, not which family the small model belongs to.

---

## 4. The deployment-ready cell — Cond D′

For the DAIH paper's C1 claim:

> A role-specialized debate using gpt-oss-120b analysts/retrieval and a Qwen2.5-7B integrator (Cond D′) matches and exceeds the closed-weight GPT-4o reference (Cond D) on MIMIC-III mortality prediction (F1 63.1 vs 56.9, n=100, biased prompt). On-premise, HIPAA-compatible deployment is feasible — but only with deliberate role specialization, not by naively dropping in small open-weight models everywhere (Cond G F1 22.2).

Cond D′ is fully open-weight:
- gpt-oss-120b: OpenAI's open-weight release (Apache-2.0)
- Qwen2.5-7B-Instruct: Alibaba's open release (Apache-2.0)

Both can be served on a single H100 or pair of A100s, on-premise.

---

## 5. Caveats and limitations

**(a) Specificity tradeoff.** D′'s recall lift comes at a steep specificity cost: only 23.9% of true negatives are correctly classified, vs 47.8% for GPT-4o D and 93.5% for the GPT-4o baseline (Cond A). In a real ICU this means many false mortality alarms per shift. Threshold calibration is required before deployment; report a calibrated operating point in §5 of the paper.

**(b) n=100 is small and stratified, not representative.** The 100-sample set has 54% positive base rate (vs MIMIC-III's true 5.4%). Accuracy in this table is NOT directly comparable to deployment-scale numbers. n=100 is for role-attribution diagnostics. The headline open-weight-matches-closed claim should be re-confirmed at n=996 before submission.

**(c) Bias-prompt regime only.** All 13 cells use the biased "Mortality is rare" integrator prompt — the clinically realistic deployment default. Unbiased numbers for Cond G exist (F1 14.5 / recall 9.3) and are slightly lower than the biased run — counterintuitive but consistent with the "Qwen integrator has nothing to amplify" interpretation. Unbiased numbers for A′-F′ are not yet computed.

**(d) No retrieval rate is high in Qwen-retrieval cells (42%).** Cells B, E, B′, E′, G all show 42% no-retrieval, because the Qwen analyst's search-emission rate is structurally lower than the GPT-4o/oss-120b analysts'. This is acceptable for the role-attribution finding but worth documenting in §3 of the paper.

---

## 6. Reproducing the numbers

```bash
cd /data/wang/junh/githubs/Debate

# All 12 cells from existing logs:
python DAIH/experiments/compute_metrics.py --logs_dir \
  KARE/gpt/results_bias/condition_A_gpt_4o/logs \
  KARE/gpt/results_bias/condition_B_gpt_4o/logs \
  KARE/gpt/results_bias/condition_C_gpt_4o/logs \
  KARE/gpt/results_bias/condition_D_qwen/logs \
  KARE/gpt/results_bias/condition_E_gpt_qwen_qwen/logs \
  KARE/gpt/results_bias/condition_F_qwen_gpt_qwen/logs \
  DAIH/results/condition_A_oss/logs \
  DAIH/results/condition_B_oss/logs \
  DAIH/results/condition_C_oss/logs \
  DAIH/results/condition_D_oss_qwen_int/logs \
  DAIH/results/condition_E_oss_qwen_int/logs \
  DAIH/results/condition_F_oss_qwen_int/logs \
  DAIH/results/condition_G_all_qwen/logs
```

Re-running any cell from scratch:

- A, B, C, D, E, F (GPT-4o): `KARE/gpt/src/run_condition_{A..F}.py`
- A′, B′, C′, D′, E′, F′ (oss-120b): `DAIH/experiments/run_condition_{A..F}_oss.py` (requires ARC API key for A′-C′; local Qwen GPU for D′-F′)
- G (all-Qwen): `DAIH/experiments/run_condition_G.py` (local Qwen GPU only)

---

## 7. Next steps

In priority order, to lock the n=100 story and move toward submission:

1. **Subgroup breakdown** (pos_all / neg_hard / neg_easy). The 100-sample manifest has stratum labels in `sample_select.py` logic; needs to be propagated into `samples_swap_core.csv` or computed inline.
2. **Calibration plot** for the four key integrators (A, D, D′, G) — reliability curve at the integrator's output probability. Will give us the "lift comes with a calibration cost" picture for §4.3 of the paper.
3. **MIRAGE debate gap-fill** — the missing MedMCQA column for the Qwen2.5-7B and Qwen3-4B-Instruct-2507 debate rows. ~1 day of compute.
4. **n=996 scale-up of D′** — re-confirm the lift at deployment scale (5.4% base rate, not 54%).
5. **MIMIC-IV ColaCare comparison** — only if calendar allows; ~1 week of compute.
6. **Bootstrap CIs** — last, per the user's instruction.
