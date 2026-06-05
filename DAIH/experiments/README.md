# DAIH experiments

Runners for the gpt-oss-120b transferability test on the existing n=100 stratified
manifest, plus the eventual scale-up to MIMIC-III n=996.

## Files

| File | Purpose |
|---|---|
| `arc_client.py` | Thin wrapper around VT ARC's OpenAI-compatible endpoint. Drop-in replacement for `KARE/gpt/src/gpt_utils.py:GPTClient`. Implements ARC's 30/min / 1000/hr / 3000/3hr fairshare sliding-window rate limiter and exponential-backoff retries. |
| `run_condition_A_oss.py` | **Cond A′**: gpt-oss-120b at all three positions (analysts, retrieval-query, integrator). Mirrors `KARE/gpt/src/run_condition_A.py` byte-for-byte except for the LLM client swap. |
| `run_condition_D_oss.py` | **Cond D′**: oss-120b analysts + oss-120b retrieval-query (both reused from Cond A′ cache) + local Qwen2.5-7B integrator. Mirrors `KARE/gpt/src/run_condition_D.py`. |
| `smoke_test.sh` | 3-step end-to-end smoke test on the first 5 patients. Run this first. |

## Prerequisites

1. **ARC API key** loaded into env:
   ```bash
   set -a; source /data/wang/junh/.cache/keys/arc_llm_api.sh; set +a
   export ARC_LLM_API_KEY="$API_KEY"
   ```

2. **MedRAG corpus** present at `/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus/` (~620 GB, MedCorp2 + UMLS). Already verified — do not delete.

3. **n=100 stratified manifest** at `KARE/gpt/manifests/selected_samples_full.parquet` (54 positives + 46 negatives). Already present.

4. **Local GPU** for Qwen2.5-7B-Instruct integrator (Cond D′ only). Single A100 is enough.

## Workflow

### Step 1 — smoke test
```bash
cd /data/wang/junh/githubs/Debate
bash DAIH/experiments/smoke_test.sh
```
Runs ARCClient self-test → Cond A′ on 5 patients → Cond D′ on those 5. Should
finish in <10 minutes. Verify per-sample JSON outputs look sane.

### Step 2 — full n=100 Cond A′
```bash
python DAIH/experiments/run_condition_A_oss.py
```
Resumes automatically (skips samples whose logs already exist). At ARC's 1000/hr
ceiling this finishes in roughly an hour for 100 patients × ~5 calls each.

### Step 3 — full n=100 Cond D′
```bash
python DAIH/experiments/run_condition_D_oss.py
```
Loads each patient's cached Cond A′ analyst+retrieval outputs and runs only the
Qwen integrator. No API calls; ~20 minutes on one A100.

### Step 4 — compare with the existing GPT-4o results
Cond A vs Cond A′ on the same 100 patients quantifies how much "all-oss" loses
to "all-GPT-4o" (closed-weight ceiling). Cond D vs Cond D′ shows whether the
integrator-swap conclusion transfers from GPT-4o analysts to oss-120b analysts.

GPT-4o counterparts:
- `KARE/gpt/results_bias/condition_A_gpt_4o/logs/`
- `KARE/gpt/results_bias/condition_D_qwen/logs/`

## What this verifies

- **Cond A′ ≈ Cond A?** → open-weight ceiling is competitive with closed-weight ceiling.
- **Cond D′ ≈ Cond D (or better)?** → integrator-swap finding is model-family-invariant; safe to scale.
- **Cond A′ ≪ Cond D′?** → integrator-position still dominates even in all-open-weight regime.

If all three hold, scale the winning configuration to MIMIC-III n=996 for the
deployment headline (separate runner — to be written after smoke test passes).

## Known carryovers from the existing code

- `process_sample_condition_a()` and `process_sample_condition_d()` are imported
  unchanged from `KARE/gpt/src/`. They retain the existing prompts, retrieval
  formatting, and **fallback policy** (when integrator probability parsing fails,
  prediction is set to `1 - label`). This is intentional for apples-to-apples
  comparison with the existing GPT-4o results; the policy will be revisited
  before any deployment claim.
- All MedRAG paths are hard-coded inside `gpt_utils.py` (already documented in
  `module_summaries/`).
