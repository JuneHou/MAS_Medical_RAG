# Module Summary: Top-Level Medical-QA Debate Scripts

**Files covered:**
- `run_debate_medrag_rag.py` (~2001 LOC) — the primary RAG-only debate driver
- `run_debate_medrag_inter.py` (~1865 LOC) — an "intermediate" variant supporting both CoT and RAG modes
- `run_debate_medrag_rag.sbatch` / `run_debate_medrag.sbatch` — SLURM launch configs
- `DEBATE_README_RAG.md`, `DEBATE_README_COT.md`, `ARCHITECTURE_COMPARISON.md` — reference docs

---

## 1. Purpose and Differences Between the Two Scripts

### `run_debate_medrag_rag.py` — RAG-only, production variant

- Designed exclusively for RAG (retrieval-augmented) debate.
- Agents are always named **Agent1** / **Agent2** (equal roles, not Analyst/Skeptic).
- **Round 1** uses `agent_turn_with_predocs()`: documents are pre-retrieved once via `unified_retrieval()` and split between the two agents before any generation happens. Both agents see their own pre-fetched document slice without seeing each other's reasoning.
- **Round 2** switches to `agent_turn_with_tools()`: each agent receives the other's Round 1 summary plus calls the `retrieve()` tool itself.
- Verification (`role_based_verification()`) uses the raw Round 2 agent responses, not summaries. Majority vote across rounds is available as a fallback.
- Output directory pattern: `rag_{dataset}_{retriever}_{model_name}/` under `--log_dir`.
- Has a `--corpus_name` CLI flag (default `MedCorp2`) and no `--mode` flag (always RAG).
- Hard-coded GPU pair: `CUDA_VISIBLE_DEVICES = '3,5'` (`rag.py:38`).

### `run_debate_medrag_inter.py` — CoT + RAG, intermediate/research variant

- Supports a `--mode` flag (`cot` or `rag`), making it a single entry-point for both approaches.
- In **CoT mode** (`mode="cot"`): no retrieval at all; agents are named `agent1`/`agent2` and use `AGENT_COT_SYS` / `AGENT_COT_DEBATE_SYS` prompts. `agent_turn_with_tools()` detects `mode=="cot"` and skips the tool-calling loop entirely, falling through to direct generation.
- In **RAG mode** (`mode="rag"`): agents are named **analyst** / **skeptic** (not agent1/agent2), use `ANALYST_SYS` / `SKEPTIC_SYS` prompts, and call `retrieve()` themselves every round. There is **no pre-retrieval step**; every round is tool-driven.
- Round 1 in RAG mode: **Analyst sees no previous context** (independent); **Skeptic sees Analyst's full Round 1 reasoning** (not just a summary) — this is the key structural difference from `_rag.py`.
- Round 2: each agent sees only the **other** agent's Round 1 summary (not their own), unlike `_rag.py` which feeds both summaries to each agent.
- Verification uses `generate_with_system()` (a named VLLMWrapper method) rather than the combined-prompt approach used in `_rag.py`.
- Hard-coded GPU pair: `CUDA_VISIBLE_DEVICES = '6,7'` (`inter.py:38`).
- Log directory pattern: `{args.log_dir}/{dataset}/{corpus_name}/{mode}/`.

### Shared code

Both scripts share nearly identical implementations of:
- All utility functions (`save_json`, `save_jsonl`, `parse_answer_from_text`, `parse_into_dict`, `clean_repetitive_response`)
- Tool infrastructure (`create_retrieval_tool`, `parse_tool_call`, `execute_tool_call`, `format_retrieved_docs_for_context`)
- Summarization system (`summarize_agent_response` and wrappers for round 1/2)
- Summary load/store functions (`load_round_summaries`, `load_all_round_summaries`, etc.)
- `unified_retrieval()` (though `_rag.py` actually calls it; `_inter.py` defines it but only uses it in RAG mode via the tool path)
- `run_debate_benchmark()` main loop and `QADataset` / `locate_answer` imports from MIRAGE

---

## 2. Pipeline: Single Question Flow

### `_rag.py` pipeline (RAG-only)

```
debate_question(medrag, llm, qid, question, options, k, log_dir)
│
├── unified_retrieval()                  # pre-fetch 2*k docs, split to agent1 and agent2
│       └── medrag.medrag_answer_by_source(query+options, k=2*k)
│           → agent1_predocs[:k], agent2_predocs[k:2k]
│
├── ROUND 1
│   ├── agent_turn_with_predocs(llm, "agent1", ..., agent1_predocs)
│   │       → llm(AGENT_RAG_ROUND1_SYS + question + docs)
│   │       → parse_into_dict()  →  save {qid}__agent1__round_1_response.json
│   │       → summarize_round1_agent_response()  → {qid}__round1__agent1__summary.json
│   │
│   └── agent_turn_with_predocs(llm, "agent2", ..., agent2_predocs, debate_history=None)
│           → same flow → {qid}__agent2__round_1_response.json + summary
│
├── ROUND 2
│   ├── load_round1_summaries()          # load both summaries from disk
│   ├── agent_turn_with_tools(llm, "agent1", ..., tools, debate_history=[self_R1, other_R1])
│   │       → llm(AGENT_RAG_DEBATE_SYS + prompt)   # generates retrieve("query")
│   │       → parse_tool_call()          # extract query string
│   │       → execute_tool_call()        # calls medrag.medrag_answer_by_source/medrag_answer
│   │       → llm(prompt + retrieved docs)          # final reasoning
│   │       → parse_into_dict()  →  save {qid}__agent1__round_2_response.json + summary
│   │
│   └── agent_turn_with_tools(llm, "agent2", ..., tools, debate_history=[self_R1, other_R1])
│           → same tool-calling flow
│
└── role_based_verification(llm, agent1_R2, agent2_R2, ...)
        → if agent1_choice == agent2_choice: return immediately (consensus_type="direct_agreement")
        → else: llm(verification_prompt + debate_summary[:4000])
                → parse_answer_from_text()
                → fallback: calculate_majority_vote_from_summaries() across rounds
        → save {qid}__verification.json
        → save {qid}__complete_debate.json
        → return final_answer dict
```

### `_inter.py` pipeline differences (RAG mode, analyst/skeptic)

- No pre-retrieval step; every round uses `agent_turn_with_tools()`.
- Round 1: Agent1 (analyst) gets `debate_history=None`; Agent2 (skeptic) gets the **full** Round 1 reasoning of Agent1 (not a summary).
- Round 2: Agent1 gets only Agent2's Round 1 summary; Agent2 gets only Agent1's Round 1 summary.
- Verification uses `llm.generate_with_system(verification_system, verification_prompt, ...)` when available.
- Fallback in verification falls back to Agent2's answer (last word), not Agent1's.

### Key functions and classes

| Name | Location | Role |
|---|---|---|
| `debate_question()` | both scripts | Orchestrates all rounds + verification for one question |
| `run_debate_benchmark()` | both scripts | Outer loop over dataset; handles resume, accuracy tracking, I/O |
| `unified_retrieval()` | both scripts | Pre-fetches 2*k docs and splits between agents (called in `_rag.py` Round 1) |
| `agent_turn_with_predocs()` | `_rag.py` only | Round 1 agent turn using pre-fetched docs |
| `agent_turn_with_tools()` | both scripts | Agent turn with 2-step (tool-call → reasoning) or CoT generation |
| `create_retrieval_tool()` | both scripts | Builds the `retrieve` tool dict wrapping MedRAG |
| `parse_tool_call()` | both scripts | Regex-extracts `retrieve("query")` from model output |
| `execute_tool_call()` | both scripts | Dispatches tool call, returns doc list |
| `role_based_verification()` | both scripts | Judge: direct agreement check → LLM verifier → majority vote |
| `summarize_agent_response()` | both scripts | Compresses agent response to ~500 tokens for cross-round context |
| `parse_answer_from_text()` | both scripts | Sequential regex parser; returns last `\boxed{X}` or equivalent |
| `parse_into_dict()` | both scripts | Wraps `parse_answer_from_text`, falls back to MedRAG's `parse_response_standard` |
| `clean_repetitive_response()` | both scripts | Deduplicates lines and answer-pattern repeats from model output |
| `calculate_majority_vote_from_summaries()` | both scripts | Aggregates per-round answers; in `_rag.py` only uses Round 2 answers |
| `VLLMWrapper` | imported from MedRAG | Local vLLM inference wrapper |
| `MedRAG` | imported from MedRAG | RAG retrieval class |
| `QADataset` | imported from MIRAGE | Dataset loader for all five benchmarks |

---

## 3. CLI Arguments

### `run_debate_medrag_rag.py`

```
--dataset     mmlu|medqa|medmcqa|pubmedqa|bioasq   (default: mmlu)
--k           int   number of docs retrieved per agent (default: 32)
--log_dir     str   base log directory (default: ./debate_logs_rag)
--rounds      int   number of debate rounds (default: 2)
--split       str   test/dev; auto-switched to dev for medmcqa (default: test)
--corpus_name str   MedCorp2 or MedCorp (default: MedCorp2)
--retriever_name str MedCPT (default: MedCPT)
--no-resume   flag  reprocess all questions (default: resume=True, skip processed)
```

Log directory is structured as `{log_dir}/rag_{dataset}_{retriever_name}_{model_name}/` and then further as `{dataset}_{retriever_name}_{corpus_name}/` inside `run_debate_benchmark`.

### `run_debate_medrag_inter.py`

```
--dataset     mmlu|medqa|medmcqa|pubmedqa|bioasq   (default: mmlu)
--mode        cot|rag   (default: cot)
--k           int   (default: 32; only used in rag mode)
--log_dir     str   (default: ./debate_logs)
--rounds      int   (default: 2)
--split       str   (default: test)
--corpus_name str   (default: MedCorp)
```

Log directory (inside `run_debate_benchmark`) is `{log_dir}/{dataset}/{corpus_name}/{mode}/`. Note: the CLI builds `os.path.join(args.log_dir, args.dataset, args.corpus_name, args.mode)` but then passes only `args.log_dir` to `run_debate_benchmark`, which builds its own `os.path.abspath(log_dir)` — there is a discrepancy here (see Quirks section).

---

## 4. External Dependencies and Hard-Coded Paths

### Python module imports (added to `sys.path` at startup)

```python
# run_debate_medrag_rag.py:15-19 and run_debate_medrag_inter.py:15-19
medrag_root = "/data/wang/junh/githubs/mirage_medrag/MedRAG"
mirage_src  = "/data/wang/junh/githubs/mirage_medrag/MIRAGE/src"
```

- `from run_medrag_vllm import patch_medrag_for_vllm, VLLMWrapper, parse_response_standard`  
  — local module at `{medrag_root}/run_medrag_vllm.py`
- `from medrag import MedRAG`  
  — `{medrag_root}/medrag.py` or `{medrag_root}/src/medrag.py`
- `QADataset`, `locate_answer` dynamically loaded from `{mirage_src}/utils.py`

### Corpus directory

```python
MEDCORP_DIR = "/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus"
```

This is passed as `db_dir` to `MedRAG(...)`. Contains the FAISS HNSW index and corpus JSON files for MedCorp2 (and possibly MedCorp).

### MedRAG initialization (both scripts, RAG mode)

```python
medrag = MedRAG(
    llm_name=HF_MODEL_NAME,        # "Qwen/Qwen2.5-7B-Instruct"
    rag=True,
    retriever_name="MedCPT",
    corpus_name="MedCorp2",        # or "MedCorp" in _inter.py
    db_dir=MEDCORP_DIR,
    corpus_cache=True,
    HNSW=True
)
```

`patch_medrag_for_vllm()` is called before constructing MedRAG to monkey-patch it for vLLM compatibility.

### GPU assignment (hard-coded at module level)

| Script | Value | FAISS GPU | VLLM GPU |
|---|---|---|---|
| `_rag.py` | `CUDA_VISIBLE_DEVICES = '3,5'` | 0 (→ physical 3) | 1 (→ physical 5) |
| `_inter.py` | `CUDA_VISIBLE_DEVICES = '6,7'` | 0 (→ physical 6) | 1 (→ physical 7) |

---

## 5. SLURM sbatch Configuration

### `run_debate_medrag_rag.sbatch`

Despite its filename, this sbatch file does **not** run `run_debate_medrag_rag.py`. It runs a KARE (mortality prediction) debate variant:

```bash
#SBATCH --job-name=KARE-mor-rag-qwen2.5-qwen3-70b
#SBATCH --partition=a100_normal_q
#SBATCH --gres=gpu:2            # 2 A100 GPUs
#SBATCH --mem=512G
#SBATCH --time=72:00:00

python -u KARE/run_kare_debate_mortality_fast.py \
    --mode rag \
    --integrator_model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --precomputed_log_dir .../rag_mor_Qwen_Qwen2.5_7B_Instruct_MedCPT_8_8/debate_logs \
    --output .../rag_mor_Qwen2.5_3-70b_8_16/kare_debate_mortality_results.json
```

This is a "fast mode" that reads precomputed agent 1–3 logs and only runs a large integrator model (Qwen3-30B). It is part of the KARE subdirectory, not the main QA debate pipeline.

### `run_debate_medrag.sbatch`

Runs a CoT boxed variant (not `_inter.py` either):

```bash
#SBATCH --job-name=resume-qwen3-8B-CoT-equal-mmlu
#SBATCH --gres=gpu:1            # 1 A100 GPU
#SBATCH --mem=256G
#SBATCH --time=72:00:00

python -u run_debate_medrag_cot_box.py \
    --dataset mmlu \
    --rounds 2 \
    --log_dir /projects/slmreasoning/junh/Debate/qwen3-8B-CoT-equal-mmlu_logs
```

Both sbatch files use conda environment at `/projects/slmreasoning/junh/envs/medrag`, run from `/projects/slmreasoning/junh/Debate` (not `/data/wang/junh/githubs/Debate`), and mirror stdout to `logs/run_{jobname}_{jobid}.log`.

---

## 6. Output Artifacts

### Per-question files (written into `{log_dir}/`)

All filenames use `{split}_{question_id}` as the `qid` (e.g., `test_0042`).

| File | Written by | Contents |
|---|---|---|
| `{qid}__agent1__round_1_response.json` | `save_json` in round loop | Full parsed response dict: `step_by_step_thinking`, `answer_choice`, `raw_response`, `generation_time`, `mode`, etc. |
| `{qid}__agent2__round_1_response.json` | same | Same structure for agent2 |
| `{qid}__agent1__round_2_response.json` | same | Round 2 response |
| `{qid}__agent2__round_2_response.json` | same | Round 2 response |
| `{qid}__agent1__latest_response.json` | `agent_turn_with_*` functions | Overwritten each round; contains the agent's last response |
| `{qid}__agent2__latest_response.json` | same | Overwritten each round |
| `{qid}__round1__agent1__summary.json` | `summarize_agent_response` | `{role, round, summary, answer_choice, formatted_summary, reasoning_type}` |
| `{qid}__round1__agent2__summary.json` | same | |
| `{qid}__round2__agent1__summary.json` | same | |
| `{qid}__round2__agent2__summary.json` | same | |
| `{qid}__verification.json` | `role_based_verification` | `{verification_prompt, verification_response, final_answer, consensus_type, timing}` — only written when agents disagree |
| `{qid}__complete_debate.json` | `debate_question` | Full debate log: `{qid, question, options, mode, rounds, history[], final_answer, timing_summary, debug_info}` |
| `{qid}__unified_retrieval.json` | `unified_retrieval` (`_rag.py` only) | Pre-retrieved docs: `{total_retrieved, agent1_count, agent2_count, agent1_snippets[], agent2_snippets[]}` |
| `{qid}__retrieval.json` | `retrieve_tool` | Per-query retrieval log when agents call `retrieve()` with a `log_dir` |
| `summarization_logs.jsonl` | `summarize_agent_response` | One JSONL line per summarization call: `{qid, role, round, original_length, summary_length, answer_choice, timestamp}` |

### Per-run file

| File | Written by | Contents |
|---|---|---|
| `{dataset}_results.json` | `run_debate_benchmark` | Array of `{id, question, gold, predicted, correct, reasoning, timing}` for all questions |
| `{qid}.json` | `run_debate_benchmark` (in `_rag.py` only) | Single-element list `[{step_by_step_thinking, answer_choice}]` used for per-question resume check |

### Log directory layout (actual observed structure)

```
debate_logs/
└── {dataset}/                              # e.g., medqa/, pubmedqa/
    └── cot/                                # or MedCorp/rag/ in inter mode
        ├── summarization_logs.jsonl
        ├── {qid}__complete_debate.json
        ├── {qid}__round1__agent1__summary.json
        ├── {qid}__round1__agent2__summary.json
        ├── {qid}__agent1__round_1_response.json
        ├── {qid}__agent2__round_1_response.json
        ├── {qid}__round2__agent1__summary.json
        ├── {qid}__round2__agent2__summary.json
        ├── {qid}__agent1__round_2_response.json
        ├── {qid}__agent2__round_2_response.json
        ├── {qid}__agent1__latest_response.json
        ├── {qid}__agent2__latest_response.json
        ├── {qid}__verification.json        # only when agents disagree
        └── {dataset}_results.json          # aggregated results
```

`debate_logs_boxed/` contains results from an earlier run of a "boxed" CoT variant (pubmedqa and medqa only, CoT mode). The `_rag.py` output goes under `debate_logs_rag/rag_{dataset}_{retriever}_{model}/`.

---

## 7. Notable Quirks, Dead Code, and In-Progress Signs

### Role-naming inconsistency in `_rag.py`

The script uses `agent1`/`agent2` role names (matching the CoT design) but prints timing with keys like `analyst_gen_time` and `skeptic_gen_time` at `_rag.py:1901–1904`, which don't exist in the timing dict (the dict uses `agent1_times`, `agent2_times`). These timing display lines will always print `0.0`.

### `_inter.py` log directory bug

The CLI block (`_inter.py:1853`) builds `log_dir = os.path.join(args.log_dir, args.dataset, args.corpus_name, args.mode)` but then passes only `args.log_dir` to `run_debate_benchmark`. `run_debate_benchmark` then uses `os.path.abspath(log_dir)` with the original `args.log_dir`. The extra path segments computed in the CLI are discarded.

### `_inter.py` verification return-value bug

At `_inter.py:1308`, verification calls `parse_answer_from_text(verification)` which returns a **tuple** `(answer, position)`, but the code then checks `if final_answer == "PARSE_FAILED"` — it is comparing against a tuple. This means the PARSE_FAILED branch never triggers for `_inter.py` verification; the raw tuple is used as `final_answer` if the fallback branch is not hit.

### CoT code path included in `_inter.py` `agent_turn_with_tools()`

The function contains both `if mode == "cot"` (direct generation, no tools) and `# RAG mode with tool calling` branches. The CoT branch returns early before any tool logic. This makes `agent_turn_with_tools` a dual-purpose function, which is somewhat confusing.

### Dead `ANALYST_SYS` / `SKEPTIC_SYS` in `_inter.py` for RAG mode

These role prompts exist but the `rag_prompts` dict maps `"agent1"` → `ANALYST_SYS` and `"agent2"` → `SKEPTIC_SYS`. However, in `_inter.py` RAG mode, roles are named `"analyst"` and `"skeptic"`, not `"agent1"` / `"agent2"`. So `rag_prompts.get(role, ANALYST_SYS)` always falls back to `ANALYST_SYS` for both agents — the SKEPTIC_SYS prompt is never actually used in practice (`_inter.py:863–867`).

### `clean_repetitive_response` signature differs

In `_rag.py`, `clean_repetitive_response(text)` takes only `text`. In `_inter.py`, it takes `clean_repetitive_response(text, max_length=1000)` and truncates at 1000 chars. The `_rag.py` version does not truncate. This reduces `_inter.py`'s reasoning context significantly.

### `calculate_majority_vote_from_summaries` logic differs

In `_rag.py`: only Round 2 answers participate in the vote; Round 1 answers are logged in `vote_breakdown` for inspection but not counted. In `_inter.py`: all four answers (both rounds, both agents) participate equally in the plurality vote, with a tie-break preferring Round 2.

### `run_debate_medrag_rag.sbatch` does not run `run_debate_medrag_rag.py`

The sbatch file is named for the RAG script but actually invokes `KARE/run_kare_debate_mortality_fast.py`. This is likely a repurposed template.

### `resume=True` only in `_rag.py`

`_rag.py` has a `--no-resume` flag and explicit resume logic. `_inter.py`'s `run_debate_benchmark` also has resume logic (skip if `{qid}.json` exists) but no CLI flag for it, and the function signature does not expose `resume`.

### Missing `result_file` write in `_inter.py`

In `_rag.py`, the per-question result is saved as `save_json([result], result_file)` enabling resume. In `_inter.py`'s benchmark loop, the per-question `.json` file is checked for resume but **never written** — only `{dataset}_results.json` is saved at the end. This means if the run is interrupted, nothing is resumable in `_inter.py`.

### Comment about 8-minute runtime

Both scripts have `# 8 mins for a question in mmlu` at line 6, indicating empirical timing observations from actual runs.

### `HF_MODEL_NAME` default

Both scripts default to `"Qwen/Qwen2.5-7B-Instruct"`. The sbatch files reference `Qwen3-8B` and `Qwen3-30B` variants in job names, suggesting the model field was manually changed for different experiments without updating the sbatch command.

---

## Key Constants (both scripts)

```python
HF_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MEDCORP_DIR   = "/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus"
DEFAULT_K     = 32        # docs retrieved per agent
RRF_K         = 60        # Reciprocal Rank Fusion parameter
MAX_ROUNDS    = 2         # debate rounds
FAISS_GPU_ID  = 0         # within CUDA_VISIBLE_DEVICES
VLLM_GPU_ID   = 1         # within CUDA_VISIBLE_DEVICES
```
