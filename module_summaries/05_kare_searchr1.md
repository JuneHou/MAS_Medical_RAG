# Module 05: KARE/searchr1 — Search-R1 RL Retriever Training Pipeline

**Period:** ~December 2025  
**Directory:** `/data/wang/junh/githubs/Debate/KARE/searchr1/`  
**Upstream framework:** Search-R1 (FUDAN-FUEX/Search-R1) + veRL (GRPO trainer)  
**Retrieval backend:** MedRAG = MedCPT encoder + MedCorp2 corpus at `/data/wang/junh/githubs/mirage_medrag/MedRAG/`

---

## 1. What Is Search-R1?

Search-R1 is a reinforcement-learning framework that teaches a language model to interleave retrieval with reasoning. The model is fine-tuned from `Qwen/Qwen2.5-7B-Instruct` using **GRPO** (Group Relative Policy Optimization, a critic-free variant of PPO) implemented inside a fork of veRL (`python -m verl.trainer.main_ppo`).

In this project the task is binary ICU mortality prediction: given a patient's EHR context and two similar reference patients (one mortality, one survival), the model must decide whether to issue search queries for medical literature before emitting a final prediction.

### Action space

The model generates free text inside a multi-turn loop. On each turn the output is parsed for one of two tags:

| Tag | Meaning | Effect |
|-----|---------|--------|
| `<search>query</search>` | Retrieve evidence | Retriever called; results injected as `<information>…</information>`; episode continues |
| `<answer>X</answer>` | Final prediction | Episode terminates; reward computed |

Invalid output (neither tag found) produces an error message and the episode continues, consuming another turn without a retrieval call. (README_SEARCHR1_KARE.md lines 162–208)

### Multi-turn episode structure

```
Turn 1: prompt (patient EHR + similar cases, ~3.5k tokens)
        → model generates search or answer
Turn 2 (if search): prompt + retrieved docs (~7.5k total)
        → model generates answer
[max_turns=2; sequences capped to prevent >18k token growth]
```

Up to 8 samples run in parallel per step; an `active_mask` tracks which samples are still generating. Samples that emit `<answer>` early are excluded from later turns. (README_SEARCHR1_KARE.md lines 44–101)

### GRPO training algorithm

No critic model is maintained. Advantage is computed as `reward − mean(batch_rewards)` across the 8-sample batch. A single PPO-style clipped policy gradient step follows, with a low-variance KL penalty to the frozen reference model:

```
kl_loss_coef  = 0.001
kl_loss_type  = low_var_kl
clip ratio    = [0.8, 1.2]
lr            = 1e-6  (with 10% warmup)
epochs        = 5
```

FSDP is used with full parameter + gradient + optimizer CPU offloading to fit Qwen2.5-7B on 4 × A40 GPUs. vLLM handles rollout generation at `gpu_memory_utilization=0.35`. (train_searchr1_single_agent.sh lines 135–197)

---

## 2. Local MedRAG Retrieval Server

### Why a server at all

Search-R1's rollout workers run as separate Ray processes. They cannot directly import Python objects from the training process. The solution is to wrap MedRAG as a local HTTP API that all workers query over `localhost:8000`.

For SLURM environments (where launching background services is awkward), an alternative in-process path is provided — see Section 6.

### `medrag_retrieval_server.py` — FastAPI wrapper

**File:** `KARE/searchr1/medrag_retrieval_server.py`

MedRAG is initialized once at startup on a call to `initialize_medrag()` (`medrag_retrieval_server.py:49`), using `Qwen/Qwen2.5-7B-Instruct` as the LLM name only so the tokenizer initializes; the LLM itself is never invoked during retrieval.

#### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Health check; returns service name, retriever, corpus, version |
| `POST` | `/retrieve` | Main retrieval endpoint (Search-R1 compatible) |
| `GET` | `/stats` | Returns retriever/corpus status |

#### POST `/retrieve` — request / response schema

```python
# Request (medrag_retrieval_server.py:36-41)
class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = 8
    return_scores: bool = False

# Response (medrag_retrieval_server.py:43-46)
class RetrievalResponse(BaseModel):
    result: List[List[Dict[str, str]]]   # outer=queries, inner=docs
    scores: Optional[List[List[float]]]
```

Each document in `result` is wrapped in a `"document"` key with sub-keys `title`, `text`, `contents` — exactly the format Search-R1's rollout code expects (`medrag_retrieval_server.py:158–179`).

#### MedCorp2 split retrieval

When the corpus is `MedCorp2` and the MedRAG object exposes `source_retrievers`, the server splits the `topk` budget between two sub-corpora (`medrag_retrieval_server.py:126–145`):

```python
k_medcorp = topk // 2 + topk % 2   # ceiling half
k_umls    = topk // 2               # floor half
```

Each sub-corpus is queried directly via its `source_retrieval_system.retrieve(query, k, rrf_k=60)` call, bypassing the LLM generation layer entirely.

#### CLI arguments

```
--host      (default 0.0.0.0)
--port      (default 8000)
--retriever (default MedCPT)
--corpus    (default MedCorp2)
--db_dir    (default /data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus)
```

---

## 3. Training Pipeline

### 3a. Data generation

**Script:** `data_generation/prepare_searchr1_balanced_data.py`

The script reads a balanced JSON index file (e.g., `train_balanced_100pos_100neg.json`) produced separately, loads KARE patient records via `KAREDataAdapter`, and writes Parquet files in the format veRL expects.

**Two prompt modes** (`--mode binary|probability`):

- `binary` — `create_single_agent_prompt_binary()` (lines 45–86): model outputs `<answer>0</answer>` or `<answer>1</answer>`.
- `probability` — `create_single_agent_prompt()` (lines 90–147): model outputs structured probabilities inside `<answer>MORTALITY PROBABILITY: X.XX\nSURVIVAL PROBABILITY: X.XX</answer>`.

Both prompts include the target patient EHR, similar mortality cases, similar survival cases, and instructions for `<think>` / `<search>` / `<answer>` formatting.

**Output Parquet schema (per row):**

```python
{
    "data_source": "kare_mortality_single_agent",
    "prompt": [{"role": "user", "content": <prompt_text>}],
    "ability": "medical-mortality-prediction",
    "reward_model": {
        "style": "rule",
        "ground_truth": {"target": ["0"] or ["1"]}
    },
    "extra_info": {"split", "patient_id", "visit_id", "original_index"}
}
```
(prepare_searchr1_balanced_data.py lines 168–188)

**Default dataset sizes:** 200 train (100 pos / 100 neg), 50 val (25/25). A larger 900-sample variant is referenced in the SLURM sbatch script (`DATA_DIR` ending in `kare_mortality_single_agent_900`).

**Output paths:**
- Binary: `data/kare_mortality_single_agent/train.parquet`, `val.parquet`
- Probability: `data/kare_mortality_prob/train.parquet`, `val.parquet`

### 3b. Reward functions

#### Experiment 1 — Binary exact match (default Search-R1 reward)

**Location:** Built into Search-R1's `search_r1/reward/qa_em.py` (upstream code, not in this repo).

The reward function extracts the last `<answer>…</answer>` tag and compares it to the ground-truth string. Returns `1.0` for an exact match, `0.0` otherwise. No intermediate penalties. (README_SEARCHR1_KARE.md lines 216–266)

Used by: `train_searchr1_single_agent.sh` and `train_searchr1_slurm.sbatch`.

#### Experiment 2 — Probability-based calibration reward

**File:** `reward_functions/kare_mortality_probability.py`

**Function:** `compute_score(solution_str, ground_truth)` (line 70)

Extracts `MORTALITY PROBABILITY: X.XX` and `SURVIVAL PROBABILITY: X.XX` from the answer text via regex. Applies four validity checks (both present, both in [0,1], sum within ±0.05 of 1.0); any failure returns `0.0`.

If valid:
```
GT=1 (mortality): reward = mortality_prob          (range [0,1])
GT=0 (survival):  reward = 1.0 - mortality_prob    (range [0,1])
```

This "positive-only" smooth reward (Option A) avoids negative gradients, which destabilize GRPO. An alternative symmetric ±1/0/−1 reward with threshold logic (`compute_score_symmetric`, line 171) is provided but not used by default.

Used by: `train_searchr1_probability.sh` via env var `REWARD_FUNCTION_PATH`.

**Summary table:**

| File | Reward type | Range | Notes |
|------|-------------|-------|-------|
| `search_r1/reward/qa_em.py` (upstream) | Binary exact match | {0, 1} | Default; no custom code |
| `reward_functions/kare_mortality_probability.py` | Calibration (positive-only) | [0, 1] | Smooth; requires both probabilities |

### 3c. RL training invocation

All scripts call `python -m verl.trainer.main_ppo` (Search-R1's veRL fork) with Hydra-style CLI overrides. Key parameters shared across all training scripts:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `algorithm.adv_estimator` | `grpo` | Critic-free advantage |
| `actor_rollout_ref.model.path` | `Qwen/Qwen2.5-7B-Instruct` | Base model |
| `data.train_batch_size` | 8 | Samples per step |
| `data.max_prompt_length` | 18000 | Max prompt tokens |
| `data.max_response_length` | 3072 | Max per-turn generation |
| `data.max_obs_length` | 3072 | Max retrieved context |
| `max_turns` | 2 (test: 3) | Max retrieval turns |
| `retriever.url` | `http://127.0.0.1:8000/retrieve` | MedRAG server |
| `retriever.topk` | 5 | Docs per query |
| `trainer.total_epochs` | 5 | Training epochs |
| `trainer.save_freq` | 50 | Steps between checkpoint saves |
| `trainer.test_freq` | 25 | Steps between validation |
| `actor_rollout_ref.actor.fsdp_config.param_offload` | true | CPU offload for 7B |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | 0.35 | vLLM headroom |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Reduce fragmentation |
| `VLLM_ATTENTION_BACKEND` | `XFORMERS` | Attention backend |

Training logs to WandB project `searchr1-kare-mortality`.

---

## 4. The Trained Checkpoint and How It Is Used Downstream

### Checkpoint on disk

**Path:** `checkpoints/searchr1-binary-single-agent-step100/`

Contents are a standard Hugging Face model directory: `config.json`, `generation_config.json`, seven `model-*.safetensors` shards, tokenizer files (`tokenizer.json`, `vocab.json`, `merges.txt`, etc.).

The folder name on disk is `searchr1-binary-single-agent-step100`, which corresponds to the model saved at training step 100 from experiment `searchr1-kare-mortality-single-agent` (Experiment 1, binary reward).

### Downstream result directories

The checkpoint appears in result directory names under `KARE/results/`, where the path is mangled into the directory name:

```
results/rag_mor_Qwen_Qwen2.5_7B_Instruct_int__data_wang_junh_githubs_Debate_KARE_searchr1_checkpoints_searchr1_binary_single_agent_step100_MedCPT_8_8/
results/single_rag_mor__data_wang_junh_githubs_Debate_KARE_searchr1_checkpoints_searchr1_binary_single_agent_step100_MedCPT_few_shot/
results/single_rag_mor__data_wang_junh_githubs_Debate_KARE_searchr1_checkpoints_searchr1_binary_single_agent_step100_MedCPT_zero_shot/
```

The first directory's name encodes:
- Outer agents: `Qwen/Qwen2.5-7B-Instruct` (base Qwen, agents 1–3 in debate)
- Integrator agent: the searchr1 checkpoint path (agent 4 / integrator)
- Retriever: MedCPT, `round1_k=8`, `round3_k=8`

### How the drop-in replacement works

`run_kare_debate_mortality.py` exposes `--integrator_model` (line 484). When passed a local path instead of a HuggingFace model ID, the integrator agent (the final synthesizing agent in the 5-agent debate system) loads the RL-trained checkpoint rather than the base Qwen model. Agents 1–3 (the retrieval analysts) still use the base Qwen model.

This means the RL-trained checkpoint is used only as the **integrator** in the existing debate architecture, not as a replacement for the retrieval-augmented debate agents. The hypothesis being tested: does RL training on mortality prediction improve the integrator's ability to synthesize evidence into a final answer?

The result JSON for the run with the RL integrator:
- `accuracy: 0.895`, `macro_f1: 0.516` — most predictions are survival (class 0), with very few mortality positives detected (`recall: 0.093`, `f1_mortality: 0.087`) — suggesting the RL-trained checkpoint still predicts mortality conservatively.
(`results/.../kare_debate_mortality_results.json` lines 7–31)

The `analyze_retrieve_rate.py` script also references this result directory to compute per-condition retrieval statistics (`analyze_retrieve_rate.py:203`).

---

## 5. Reward Functions — Summary

| File | One-line description |
|------|----------------------|
| `search_r1/reward/qa_em.py` (upstream, not in repo) | Binary {0,1} exact-match on the `<answer>` tag content |
| `reward_functions/kare_mortality_probability.py` | Smooth positive-only calibration reward: `mort_prob` if GT=1, else `1−mort_prob`; requires both probabilities summing to 1±0.05 |

The probability reward file also contains an unused `compute_score_symmetric` function implementing a ±1/0/−1 threshold-based scheme.

---

## 6. Patch Script — `patch_searchr1_local_retriever.py`

**What it patches:** `Search-R1/search_r1/llm_agent/generation.py`, specifically the `_batch_search` method (~line 450). That method is the single hook that the Search-R1 rollout loop calls to retrieve documents for each batch of queries.

**Why it was needed:** The original `_batch_search` only supports HTTP retrieval (`requests.post(url, json=payload)`). On SLURM clusters it is difficult to run a persistent HTTP service on a specific port while the training job occupies all GPUs. The patch adds a branch controlled by a `retriever.local` config flag.

**What the patch does** (`patch_searchr1_local_retriever.py:9–61`):

```python
use_local = getattr(self.config, 'local', False)

if use_local:
    # Lazy-initialize singleton LocalMedRAGRetriever on a dedicated GPU
    from local_medrag_retriever import get_retriever
    self._local_retriever = get_retriever(
        corpus_name=..., retriever_name=..., db_dir=..., gpu_id=self.config.gpu_id
    )
    return self._local_retriever.retrieve(queries, topk=..., return_scores=True)
else:
    # Original HTTP path (unchanged)
    return requests.post(url, json=payload).json()
```

**`local_medrag_retriever.py`** implements the in-process alternative (`LocalMedRAGRetriever`, lines 14–117): loads MedRAG with `llm_name=None`, moves the MedCPT encoder to the designated GPU, and exposes a `retrieve()` method with the same dict schema as the HTTP server. A module-level singleton (`get_retriever()`, line 124) ensures only one retriever is loaded per process.

**GPU allocation under SLURM** (from `train_searchr1_slurm.sbatch:83–85`):
```
GPU 0,1,2,3 → CUDA_VISIBLE_DEVICES
GPU 3        → RETRIEVER_GPU (retrieval only)
GPU 0-3      → NUM_TRAIN_GPUS=4 (training)
```
In practice the sbatch file starts the HTTP server on GPU 3, but the SLURM guide describes the fully in-process alternative where no HTTP server is used at all.

The patch is applied manually: copy `patch_searchr1_local_retriever.py` into the Search-R1 repo and merge the `_batch_search` body into `generation.py`. No automated patching script exists.

---

## 7. CLI Arguments and Environment Variables

### Environment variables (set in all training scripts)

| Variable | Value | Purpose |
|----------|-------|---------|
| `CUDA_VISIBLE_DEVICES` | e.g., `1,3,4,5` | GPU selection (positional arg `$1`) |
| `BASE_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Model loaded by veRL |
| `EXPERIMENT_NAME` | e.g., `searchr1-kare-mortality-single-agent` | WandB run name and checkpoint subdir |
| `DATA_DIR` | `.../data/kare_mortality_single_agent` or `kare_mortality_prob` | Parquet input |
| `CHECKPOINT_DIR` | `.../checkpoints/$EXPERIMENT_NAME` | Where veRL saves model shards |
| `ROLLOUT_DATA_DIR` | `.../rollout_data/$EXPERIMENT_NAME` | Where rollout JSONL logs go |
| `VLLM_ATTENTION_BACKEND` | `XFORMERS` | vLLM attention kernel |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Reduce memory fragmentation |
| `RAY_TMPDIR` | `/data/wang/junh/tmp/` | Ray object store location |
| `RAY_LOG_TO_STDERR` | `0` | Suppress verbose Ray worker output |
| `REWARD_FUNCTION_PATH` | `.../reward_functions/kare_mortality_probability.py` | (probability script only) |

### Differences across training scripts

| Script | `DATA_DIR` | Reward | `max_turns` | GPU default |
|--------|-----------|--------|------------|-------------|
| `test_searchr1_training.sh` | `kare_mortality_single_agent` | binary (qa_em) | 3 | `0` |
| `train_searchr1_single_agent.sh` | `kare_mortality_single_agent` | binary (qa_em) | 2 | `1,3,4,5` |
| `train_searchr1_probability.sh` | `kare_mortality_prob` | `kare_mortality_probability.py` | 2 | `1,3,4,5` |
| `train_searchr1_slurm.sbatch` | `kare_mortality_single_agent_900` | binary (qa_em) | 2 | `0,1,2,3` |

### SLURM resource configuration (`train_searchr1_slurm.sbatch`)

```
#SBATCH --job-name=searchr1-prob-900
#SBATCH --account=slmreasoning
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=72:00:00
```

The sbatch uses an explicit conda env at `/projects/slmreasoning/junh/envs/searchr1` referenced as `${ENV_PATH}/bin/python` (line 36–37) to avoid HPC Python environment conflicts. It clears `PYTHONPATH` and `PYTHONHOME` and sets `PYTHONNOUSERSITE=1` for isolation (lines 39–41).

The sbatch also starts the MedRAG HTTP server on GPU 3 as a background process (lines 132–151) and kills it in the cleanup section (lines 236–240), making the job self-contained despite using HTTP mode rather than the in-process local retriever.

---

## 8. Additional Files

### `test_medrag_server.py`

Four-test smoke-test script: health check (`GET /`), single-query retrieval (`POST /retrieve`), batch retrieval, and stats endpoint (`GET /stats`). Verifies the Search-R1 document format (`doc['document']['title']`, `doc['document']['contents']`). (`test_medrag_server.py:13–60`)

### `utils/save_rollout_logs.py`

`RolloutLogger` class for manually persisting rollout batches (prompts, responses, rewards, ground truths) to disk. Needed because Search-R1's veRL fork lacks built-in rollout logging; the trainer's `+trainer.rollout_data_dir` argument enables this path.

### `download_log.py`

Not read in detail; based on name, likely a utility for logging or tracking model/data downloads.

### `rollout_data/`

Directory where live training rollout logs are written (one JSONL per checkpoint step).

---

## 9. End-to-End Workflow Summary

```
1. PREPARE DATA
   python data_generation/prepare_searchr1_balanced_data.py \
     --balanced_json train_balanced_100pos_100neg.json \
     --split train --mode binary
   → data/kare_mortality_single_agent/train.parquet, val.parquet

2. START MEDRAG SERVER (interactive / screen session)
   python searchr1/medrag_retrieval_server.py --port 8000
   → FastAPI serving MedCPT + MedCorp2 at localhost:8000/retrieve

3. VERIFY SERVER
   python searchr1/test_medrag_server.py

4. RUN RL TRAINING
   bash searchr1/train_searchr1_single_agent.sh 1,3,4,5
   → verl.trainer.main_ppo with GRPO, 5 epochs, checkpoints every 50 steps
   → checkpoints/searchr1-kare-mortality-single-agent/global_step_100/

5. USE CHECKPOINT IN DEBATE EXPERIMENTS
   python run_kare_debate_mortality.py \
     --model Qwen/Qwen2.5-7B-Instruct \
     --integrator_model /path/to/searchr1/checkpoints/searchr1-binary-single-agent-step100 \
     --mode rag --retriever_name MedCPT
   → RL-trained model acts as debate integrator (agent 4)
   → results saved to results/rag_mor_..._int_..._MedCPT_8_8/

6. ON SLURM (alternative, self-contained)
   sbatch searchr1/train_searchr1_slurm.sbatch
   → Starts its own MedRAG server on GPU 3, trains on GPUs 0-3
   → 900-sample dataset, 72-hour time limit
```

---

## Key Design Decisions and Limitations

- **Sparse reward only:** No intermediate penalties for extra searches or invalid tags. GRPO advantage normalization implicitly penalizes inefficiency by comparing samples within a batch.
- **max_turns=2 hard cap:** Prevents context growth beyond ~10k tokens per episode (3.5k prompt + 2 × retrieval + 2 × response).
- **Balanced training data:** 50/50 mortality/survival split for RL training, despite the natural KARE imbalance (~5–10% mortality). The resulting checkpoint still predicts mortality conservatively in downstream evaluations (recall 0.09 on full test set).
- **HTTP vs. in-process retrieval:** HTTP server is the default for interactive runs; the patched in-process retriever (`local_medrag_retriever.py`) is the SLURM-friendly alternative, but requires manually applying `patch_searchr1_local_retriever.py` to upstream Search-R1 code.
- **No critic model:** GRPO replaces the PPO value network. This reduces GPU memory but means advantage estimates are noisier (batch-level rather than state-level baselines).
