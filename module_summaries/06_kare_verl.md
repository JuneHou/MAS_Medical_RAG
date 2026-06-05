# Module 06: KARE/verl — VERL/GRPO RL Training Pipeline

**Directory**: `/data/wang/junh/githubs/Debate/KARE/verl/`
**Period**: December 2025 (first wandb run: 2025-12-13; last: 2025-12-19)
**Parent project**: Medical multi-agent debate system for MIMIC-III mortality prediction (KARE).

---

## 1. What Are VERL and GRPO?

**VERL** (Volcano Engine Reinforcement Learning) is an open-source RL training framework from ByteDance/Volcengine (`github.com/volcengine/verl`). It provides a PPO/GRPO trainer (`verl.trainer.main_ppo`) with support for FSDP-sharded model training, vLLM-based rollout generation, and pluggable custom reward functions. All training scripts here invoke VERL as:

```
python -m verl.trainer.main_ppo  algorithm.adv_estimator=grpo  ...
```

**GRPO** (Group Relative Policy Optimization) is the RL algorithm used, introduced in the DeepSeekMath paper. Instead of a learned critic, GRPO generates a *group* of `n` responses per prompt and computes advantages relative to the group mean reward. This removes the need for a separate value network. Key parameter: `actor_rollout_ref.rollout.n=2` (group size 2 in the actual scripts, though README mentions n=4 in some configs).

**How GRPO is applied here**: For each patient prompt, the vLLM engine samples `n=2` candidate responses. The custom reward function scores each. The relative reward within the group forms the advantage signal. A KL penalty (`kl_loss_coef=0.001`, type `low_var_kl`) keeps the policy close to the reference (frozen base model). No critic warmup is used (`trainer.critic_warmup=0`).

---

## 2. Model, Task, and Reward — End-to-End Map

### Base Model
`Qwen/Qwen2.5-7B-Instruct` — used as the starting point for both training phases.

### Role in the Debate System
The KARE debate system has four LLM roles:
- Agents 1–3 (target analyst, mortality risk assessor, protective factor analyst): standard Qwen2.5-7B-Instruct
- **Agent 4 (Integrator)**: the model trained here. It reads all three agents' analyses plus retrieved MedRAG evidence, then outputs a structured probability.

The integrator was failing in production because it often produced verbose free-form text instead of the required `MORTALITY PROBABILITY: X.XX` line, causing downstream parsing failures. This RL pipeline fixes that.

### Training Phases and Checkpoints

| Phase | Objective | Starting Model | Reward Function | Checkpoint |
|-------|-----------|---------------|-----------------|------------|
| Phase 1a — Mortality Format | Output `MORTALITY PROBABILITY: X.XX` correctly | Qwen2.5-7B-Instruct | Binary (0/1) | `checkpoints/format/global_step_57/` |
| Phase 1b — Survival Format | Output `SURVIVAL PROBABILITY: X.XX` correctly | Phase 1a checkpoint (step 57) | Binary (0/1) | TBD (ready to train) |
| Phase 2 — Prediction Accuracy | Output calibrated probabilities aligned with ground truth outcomes | Qwen2.5-7B-Instruct (or Phase 1 ckpt) | Continuous Brier-based [-1, +1] | `checkpoints/prediction/` (step to be determined) |

### Converted Model
After training, FSDP shards are merged and saved as:
- `models/format_enforcer_7b_step57/` — HuggingFace-loadable, used as drop-in integrator
  (README:176; convert_checkpoint.sh:10 shows the naming pattern `prediction_brier_unlabel_7b_step{N}` for prediction phase)

### Downstream Integration
The converted model replaces the integrator in `mortality_debate_rag.py`:
```
python run_kare_debate_mortality.py \
    --model Qwen/Qwen2.5-7B-Instruct \         # Agents 1-3
    --integrator_model .../format_enforcer_7b_step57 \  # Agent 4
    --integrator_gpu 4
```
(README:363-371)

---

## 3. Data Generation Pipeline

### Overview
Training data is assembled from existing KARE debate logs and retrieval logs — no new LLM inference is needed for format training. The pipeline converts completed debate runs into RL training episodes (prompt + reward signal).

### Source Data
- **Patient EHR records**: `KARE/data/ehr_data/mimic3_mortality_samples_test.json` — 996 MIMIC-III patients with conditions, procedures, medications, and `label` (0=survival, 1=mortality). (DATA_GENERATION_PLAN.md:14-30)
- **Debate logs**: `KARE/results/arc_rag_mor_.../debate_responses_{pid}_{vid}.log` (primary) or `fallback_rag_mor_.../` (fallback). Contains three agents' responses. (DATA_GENERATION_PLAN.md:34-56)
- **Retrieval logs**: `retrieve_mortality_assessment_{pid}_{vid}.json` — MedRAG-retrieved documents and the query that was used. (DATA_GENERATION_PLAN.md:62-88)

### Scripts in `data_generation/format/`

- `parse_debate_logs.py` — extracts the three agent sections from debate `.log` files with hierarchical fallback (primary dir → fallback dir → skip).
- `parse_retrieval_logs.py` — loads retrieval JSON, extracts the `query` string and top 16 retrieved documents.
- `find_hard_samples.py` — scans logs for the string `"EXTRACTED MORTALITY PROBABILITY: None"` to identify the 193 patients where the base model failed format extraction. These become the "hard mode" dataset.
- `generate_training_data.py` — top-level pipeline script. Args: `--class_type {mortality|survival}` and optionally `--hard`. Assembles prompts from all components and writes Parquet files.

### Scripts in `data_generation/prediction/`

- `sample_balanced_data.py` — samples balanced subsets (100 positive mortality + 100 negative survival) from available patients, requiring each patient to have complete similar-patient context. Outputs JSON files of patient IDs.
- `generate_prediction_training_data.py` — runs a full 3-round debate for each patient (using `MortalityDebateSystem` in `mortality_debate_rag_grpo.py`), including actual LLM tool calls for integrator-specific retrieval queries. Generates two prompts per patient (mortality + survival). Imports `KAREDataAdapter` from `KARE/kare_data_adapter.py` (shared with searchr1; resolved via `sys.path` going up 4 levels: `generate_prediction_training_data.py:17-19`).

### Prompt Schema (per training example)
Each Parquet row contains:

```
prompt:        Full chat-formatted prompt string (see below)
ground_truth:  int (0 or 1)
data_source:   'kare_integrator_mortality_format'
ability:       'medical_format'
reward_model:  {'ground_truth': <label>}   # VERL-required structural field
extra_info:    {'patient_id': ..., 'assessment_type': 'mortality'|'survival', ...}
```

The `prompt` string is assembled as:
1. System prompt (mortality or survival role, tool use instructions)
2. `## Target Patient EHR Context ##` — formatted conditions/procedures/medications
3. `## Previous Debate Analysis ##` — three agent responses
4. `You called: retrieve("<query>")` — simulated tool call output
5. Retrieved Evidence block (top 16 MedRAG documents)
6. Final instruction line

Mean prompt length: ~13,134 chars (~3,283 tokens); 95th percentile ~18,091 chars; max 60,932 chars. Context window set to 16,384 tokens to cover 99%+ of samples. (README:68-73)

### Dataset Sizes

| Dataset | Location | Size |
|---------|----------|------|
| Standard mortality | `data_generation/format/mortality_grpo_data/` | ~796 train / ~200 test |
| Hard mortality | `data_generation/format/mortality_grpo_data_hard/` | 154 train / 39 test |
| Hard survival | `data_generation/format/survival_grpo_data_hard/` | 154 train / 39 test |
| Prediction train | `data_generation/prediction/train.parquet` | ~400 examples (200 patients x 2) |
| Prediction val | `data_generation/prediction/val.parquet` | ~354 examples (177 patients x 2; only 77 positive available) |

Hard samples (193 total) are identical across mortality and survival datasets — same patients, same EHR and debate history, only the system prompt differs. (README:50-52)

---

## 4. Reward Functions in `reward_score/`

Three files, all exposing a `compute_score(...)` function that VERL calls during training.

### `kare_mortality_format.py`
Binary format-compliance reward for Phase 1a. Checks only the **last non-empty line** of the model output against the regex `^\s*MORTALITY\s+PROBABILITY\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$` (case-insensitive). Returns `1.0` if the extracted float is in `[0.0, 1.0]`, else `0.0`. Ground truth is ignored entirely (placeholder `"__FORMAT_ONLY__"`). Includes debug logging to `reward_debug.log` for the first 10 KB of calls. (kare_mortality_format.py:37-114)

### `kare_survival_format.py`
Identical structure to the mortality format reward, but matches `SURVIVAL PROBABILITY: X.XX` on the last line. Returns `1.0` / `0.0` with no accuracy component. (kare_survival_format.py:37-122)

### `kare_prediction_reward.py`
Accuracy reward for Phase 2. Uses a **modified Brier score** with aggressive scaling. Formula (kare_prediction_reward.py:208-210):

```python
brier_error = (p - y) ** 2
raw_reward = 1.0 - 4.0 * brier_error
reward = max(-1.0, min(1.0, raw_reward))
```

Where `p` is the extracted probability and `y` is the target label (0 or 1, flipped for survival assessments). Key properties:
- Perfect prediction (`p = y`): reward = +1.0
- Random prediction (`p = 0.5`): reward = 0.0 (neutral)
- Bad prediction (error ≥ 0.5): reward = -1.0 (clamped)

This replaced an earlier discrete threshold-based reward (`+1/0/-1`) that suffered from sparse gradients — most predictions fell into "uncertain" zones receiving reward 0.0. The Brier approach provides non-zero gradient everywhere. (README:862-944)

The function reads `assessment_type` from `extra_info` (either `'mortality'` or `'survival'`) to know which probability to extract and which direction to score. Includes debug logging of the first 50 calls to `logs/reward_debug.log`. (kare_prediction_reward.py:96-229)

---

## 5. Checkpoint Conversion: FSDP Shards to HuggingFace

### Why Conversion Is Needed
VERL trains with FSDP (Fully Sharded Data Parallel), which stores model weights split across multiple GPU rank files: `model_world_size_4_rank_{0,1,2,3}.pt`, optimizer states `optim_world_size_4_rank_*.pt`, and `extra_state_world_size_4_rank_*.pt`. These cannot be loaded directly by HuggingFace `transformers` or vLLM.

### `convert_fsdp_to_hf.py`
A thin wrapper around VERL's built-in model merger (convert_fsdp_to_hf.py:44-58):
```python
cmd = [
    "python", "-m", "verl.model_merger", "merge",
    "--backend", "fsdp",
    "--local_dir", checkpoint_dir,   # e.g. checkpoints/global_step_57/actor
    "--target_dir", output_dir       # e.g. models/format_enforcer_7b_step57
]
```
Validates that `fsdp_config.json` exists in the checkpoint dir before proceeding.

### `convert_checkpoint.sh`
Convenience wrapper (convert_checkpoint.sh:7-19). Takes an optional step number (default 150), constructs paths:
- Source: `checkpoints/prediction/global_step_{N}/actor`
- Output: `models/prediction_brier_unlabel_7b_step{N}`

Then calls `convert_fsdp_to_hf.py` with those paths.

### Disk Budget
Original FSDP checkpoint at step 57: ~86 GB (28 GB model shards + 58 GB optimizer states). Converted HuggingFace model: ~15 GB. Safe to delete shards after verifying the converted model works. (README:326-345)

---

## 6. SLURM Resources and Key Hyperparameters

### Local GPU Runs (`run_kare_mortality_grpo.sh`, `run_kare_prediction_grpo.sh`)
Both scripts use `CUDA_VISIBLE_DEVICES=3,4,5,6` (4x A40 44 GB GPUs on the local workstation). They set `VLLM_USE_V1=1` and redirect Ray's temp dir to `/data/wang/junh/tmp/` to avoid the 107-byte Unix socket path limit.

### SLURM Job (`run_kare_prediction_grpo.sbatch`)
Targets a cluster partition `a100_normal_q` under account `slmreasoning`:

```
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=72:00:00
```

Uses project-space paths: `/projects/slmreasoning/junh/Debate/KARE/verl/`. HuggingFace cache redirected to `/projects/slmreasoning/junh/.cache/huggingface` to avoid home quota. WandB auto-falls back to offline mode if `wandb verify` fails. (run_kare_prediction_grpo.sbatch:7-87)

### Shared GRPO Hyperparameters (both format and prediction scripts)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `algorithm.adv_estimator` | `grpo` | GRPO (not PPO) |
| `data.train_batch_size` | 8 | Total batch per step |
| `data.max_prompt_length` | 16384 | Covers 99%+ of prompts |
| `data.max_response_length` | 4096 (format) / 8192 (prediction) | Longer for accuracy reasoning |
| `actor_rollout_ref.actor.optim.lr` | 1e-6 | Conservative LR |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | 4 | |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | 1 | |
| `actor_rollout_ref.actor.use_kl_loss` | True | |
| `actor_rollout_ref.actor.kl_loss_coef` | 0.001 | |
| `actor_rollout_ref.actor.kl_loss_type` | `low_var_kl` | |
| `actor_rollout_ref.actor.fsdp_config.param_offload` | True | CPU offload for params |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` | True | CPU offload for optimizer |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | 2 | Model split across 2 GPUs |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | 0.5 | 50% GPU memory for vLLM |
| `actor_rollout_ref.rollout.n` | 2 | Group size for GRPO |
| `trainer.total_epochs` | 3 | |
| `trainer.save_freq` | 50 | Save checkpoint every 50 steps |
| `trainer.nnodes` | 1 | Single node |
| `trainer.n_gpus_per_node` | 4 | |

The format training script (`run_kare_mortality_grpo.sh`) was last configured to train survival format from the Phase 1a checkpoint (`actor_rollout_ref.model.path` pointing to `models/format_enforcer_7b_step57`). The prediction script uses `Qwen/Qwen2.5-7B-Instruct` as the base. (run_kare_mortality_grpo.sh:49; run_kare_prediction_grpo.sh:50)

WandB project names: `verl-kare-mortality-format` (format training) and `verl-kare-prediction` (accuracy training).

---

## 7. Relationship to `KARE/searchr1/`

`searchr1/` is a parallel RL track that trains a **single-agent** system to do retrieval-augmented mortality prediction end-to-end — one model that both searches and predicts. By contrast, `verl/` trains only the **integrator** (Agent 4) within the multi-agent debate system.

### Shared Infrastructure

| Component | Shared? | Detail |
|-----------|---------|--------|
| `KAREDataAdapter` | Yes — same file | Both import from `KARE/kare_data_adapter.py` (verl resolves via `sys.path` up 4 levels; searchr1 copies the adapter into its own `data_generation/` dir) |
| RL Framework | Yes | Both use `verl.trainer.main_ppo` with `algorithm.adv_estimator=grpo` |
| MIMIC-III data | Yes | Both draw from `KARE/data/ehr_data/mimic3_mortality_samples_test.json` |
| MedRAG retrieval | Yes | Both use the same retrieval infrastructure (HTTP server / local retriever) |
| Reward format | Overlapping | Both define a `compute_score` function targeting `MORTALITY PROBABILITY: X.XX`; searchr1 uses a positive-only reward (`mortality_prob` if GT=1 else `1-mortality_prob`), while verl uses binary format or Brier-based accuracy |
| Balanced sampling | Yes — same approach | Both use 100 positive + 100 negative patient samples for training |

### Do Their Outputs Feed Each Other?
Not directly. The `verl/` pipeline produces a fine-tuned integrator for the debate system. The `searchr1/` pipeline produces a standalone single-agent model. They are separate experiments exploring different RL strategies for the same underlying clinical task. There is no explicit pipeline where one module's checkpoint feeds into the other.

---

## 8. WandB Logs

**Total run directories**: 33 entries in `wandb/` (29 of which are prefixed `run-`, the rest are debug logs and `latest-run` symlink).

**Date range**: 2025-12-13 (earliest: `run-20251213_094438-e53wvp3y`) through 2025-12-19 (latest: `run-20251219_202724-ka9pt9e8`). All runs fall within a one-week sprint in December 2025.

The cluster of runs on 2025-12-13 (at least 3 runs close together in time) likely corresponds to early format-enforcement experiments. Later runs through 2025-12-19 correspond to refinements of the reward function and the shift toward prediction accuracy training.

WandB projects referenced: `verl-kare-mortality-format` (format runs) and `verl-kare-prediction` (accuracy runs).

---

## Key File Reference

| File | Purpose |
|------|---------|
| `README.md` | Primary reference; covers both format and prediction training end-to-end |
| `DATA_GENERATION_PLAN.md` | Design document for data pipeline; useful for understanding prompt schema |
| `run_kare_mortality_grpo.sh` | Local GPU launch for format training (currently set to survival format from step-57 ckpt) |
| `run_kare_prediction_grpo.sh` | Local GPU launch for prediction accuracy training |
| `run_kare_prediction_grpo.sbatch` | SLURM version of prediction training (A100 cluster) |
| `convert_fsdp_to_hf.py` | Merges FSDP shards into HuggingFace safetensors via `verl.model_merger` |
| `convert_checkpoint.sh` | Convenience wrapper; defaults to step 150, outputs `models/prediction_brier_unlabel_7b_step{N}` |
| `analyze_reward_function.py` | Visualization/analysis script for Brier reward behavior; not used in training |
| `reward_score/kare_mortality_format.py` | Binary last-line format reward for mortality |
| `reward_score/kare_survival_format.py` | Binary last-line format reward for survival |
| `reward_score/kare_prediction_reward.py` | Brier-based continuous accuracy reward (`r = clamp(1 - 4(p-y)^2, -1, 1)`) |
| `data_generation/format/generate_training_data.py` | Assembles format-training Parquet from debate/retrieval logs |
| `data_generation/format/find_hard_samples.py` | Identifies 193 "hard" patients where base model failed format extraction |
| `data_generation/prediction/generate_prediction_training_data.py` | Runs live 3-round debate to generate accuracy-training Parquet |
| `data_generation/prediction/sample_balanced_data.py` | Balanced 100+100 patient sampling |
| `checkpoints/format/global_step_57/` | Completed Phase 1a checkpoint (mortality format; ~75-81% reward) |
