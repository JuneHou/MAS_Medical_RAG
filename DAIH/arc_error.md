# ARC runbook — MIRAGE gap-fill debate jobs (run_debate_medrag_rag.py)

Goal: run the MedMCQA (and other) RAG/CoT debate cells on VT ARC.
This doc captures every issue found while bringing the pipeline up, in order, with the
exact fix for each. Items marked ✅ are already done; ❌ are the remaining blockers.

Env: `/home/junh/envs/medrag` (python 3.10) · repo `/home/junh/repos/Debate` ·
corpus `/scratch/junh/mirage_medrag/MedRAG/src/data/corpus`.

```bash
PY=/home/junh/envs/medrag/bin/python
```

---

## 1. ✅ Tokenizer crash — transformers version drift (FIXED)

Symptom: `Qwen2Tokenizer has no attribute all_special_tokens_extended` in vllm.
Cause: the env had drifted to **transformers 5.9.0** (a v5 major release); vllm 0.11.0
needs the pinned **4.57.1**. v5 removed the API vllm reads.

Fix (already applied — keep for reference / rebuilds):
```bash
$PY -m pip install "transformers==4.57.1" "tokenizers==0.22.1"
# verify (all three must succeed):
$PY -c "import transformers,tokenizers;print(transformers.__version__,tokenizers.__version__)"  # 4.57.1 / 0.22.1
$PY -c "import vllm;print('vllm',vllm.__version__)"                                              # 0.11.0
$PY -c "from transformers.tokenization_utils_base import SpecialTokensMixin as S;print(hasattr(S,'all_special_tokens_extended'))"  # True
```
`environment.yml` already pins 4.57.1, so a clean rebuild is fine; the env had just been
pip-mutated after creation. If it drifts again, run `$PY -m pip check` to find which
package pulled transformers>=5.

---

## 2. ✅ Wrong Python interpreter on compute node (FIXED — but the batch jobs must guard it)

Symptom on the compute node: `ModuleNotFoundError: No module named 'transformers'`, with
the traceback pointing at `/home/junh/.local/lib/python3.13/...`.
Cause: bare `python` resolved to base Miniconda **python 3.13** + your `~/.local` user-site,
not the env's 3.10. `source activate` did not prepend the env bin in that shell.

Fix when running by hand: always call the env python explicitly and block user-site:
```bash
export PYTHONNOUSERSITE=1
which python; $PY -c "import sys;print(sys.executable, sys.version.split()[0])"   # confirm 3.10
# run scripts with $PY, e.g.:  $PY -u run_debate_medrag_rag.py ...
```

Fix for the **sbatch** files (do this so batch jobs can't silently fall back to 3.13):
in each `mirage_*.sbatch`, replace the activation block with:
```bash
module load Miniconda3/24.7.1-0          # pin! see note below
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/junh/envs/medrag
export PATH="/home/junh/envs/medrag/bin:$PATH"   # belt-and-suspenders; see note below
export PYTHONNOUSERSITE=1
# fail fast if python isn't 3.10 from the env:
python -c "import sys; assert sys.executable.startswith('/home/junh/envs/medrag/'), sys.executable" \
  || { echo 'WRONG PYTHON — env not active' >&2; exit 1; }
```

**Module-default flip (2026-06):** bare `module load Miniconda3` now resolves to the new
default **`Miniconda3/25.11.1-1`** (python 3.13). With that conda, `conda activate
/home/junh/envs/medrag` returns 0 and sets `CONDA_PREFIX`, but **does not prepend the env
`bin/` to PATH** — so the guard fired with `WRONG PYTHON` (base python
`/apps/common/software/Miniconda3/25.11.1-1/bin/python`). The env was created with
`24.7.1-0`, where activation works correctly. Fix: **pin `module load Miniconda3/24.7.1-0`**
and add the explicit `export PATH=.../bin:$PATH` after activate as a guarantee. Both are
applied to all 8 sbatch files.

---

## 3. ❌ Missing chunk text on /scratch — the main remaining blocker

Symptom: `Cloning the pubmed corpus from Huggingface... fatal: destination path
'.../corpus/pubmed' already exists`, repeated for textbooks and wikipedia, then OOM.
Cause: each `{source}/chunk` dir is a **symlink into the HF cache** (`hf_cache`), which was
excluded from the original `/scratch` staging. On `/scratch` those symlinks dangle, so
MedRAG thinks the corpus is missing and tries to git-clone it. `textbooks/chunk` is even an
**absolute** local path (`/data/wang/...`) that can never resolve on ARC.

Real chunk text that must be staged (statpearls/umls are already real on /scratch):

| source | size | files |
|---|---|---|
| pubmed | 66 G | 1166 |
| wikipedia | 43 G | 646 |
| textbooks | 0.2 G | 18 |

Fix — run from your **LOCAL workstation** (the data lives there; ARC can't pull it). The
`-L` flag dereferences the symlinks and writes real `.jsonl` files, so `/scratch` becomes
self-contained:
```bash
# LOCAL workstation, in tmux — ~109 G total
SRC=/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus
DST=junh@tinkercliffs1.arc.vt.edu:/scratch/junh/mirage_medrag/MedRAG/src/data/corpus
for s in pubmed textbooks wikipedia; do
  rsync -rL --no-times --partial --progress "$SRC/$s/chunk/" "$DST/$s/chunk__staged/"
done
```
Then on **ARC**, swap the dangling symlink for the real dir:
```bash
cd /scratch/junh/mirage_medrag/MedRAG/src/data/corpus
for s in pubmed textbooks wikipedia; do
  rm -f "$s/chunk" && mv "$s/chunk__staged" "$s/chunk"
  echo "$s: $(ls "$s/chunk" | wc -l) files"   # expect 1166 / 18 / 646
done
```
Adds ~1830 files to /scratch — well under the 10k guideline.
Affects all RAG cells (4, 5, 6, 7, 8). CoT cells (1, 2, 3) don't retrieve, so they're fine.

A preflight check is now in `_stage_corpus.sh` that aborts in ~1s if any chunk dir is still
dangling (instead of OOMing 30 min later) — re-rsync `DAIH/experiments/sbatch/` to ARC to
pick it up.

---

## 4. ❌ OOM — FAISS indexes need ~170 G host RAM + 2 GPUs

Symptom: `oom_kill event ... Out Of Memory`, job Killed.
Cause: the env uses **faiss-cpu**, so the MedCPT indexes load into **host RAM**, not GPU:

| index | size |
|---|---|
| pubmed faiss.index | 75 G |
| wikipedia faiss.index | 94 G |
| statpearls / textbooks | 1.1 G / 0.4 G |
| **total in RAM** | **~170 G** |

The interactive session used `--gres=gpu:1` with default memory → killed loading the 94 G
wikipedia index. Nothing wrong with the pipeline — it was under-resourced.

Fix: use the sbatch resources (`--mem=256G --gres=gpu:2`), which are sized for this. Don't
test on a small interactive shell.

---

## 6. ✅ FlashInfer sampler JIT needs nvcc (FIXED)

Symptom (during vLLM V1 engine init, right after the model finishes loading):
```
[EngineCore] Using FlashInfer for top-p & top-k sampling.
...
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
  (flashinfer/jit/cpp_ext.py:get_cuda_path → write_ninja → build_and_load)
```
Cause: vllm 0.11.0's V1 `TopKTopPSampler` defaults to **FlashInfer** for top-k/top-p
sampling, which **JIT-compiles** a CUDA kernel on first use. The ARC compute node has no
`nvcc` / no `/usr/local/cuda` / `CUDA_HOME` unset, so the JIT build aborts and the engine
dies. (Attention uses the FlashAttention backend, so the sampler is the only FlashInfer JIT
path triggered.)

Fix: disable the FlashInfer sampler so vllm uses its PyTorch-native top-k/top-p path (no JIT,
no nvcc, statistically equivalent). Added to all 8 sbatch env blocks:
```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```
vllm gates this at `vllm/v1/sample/ops/topk_topp_sampler.py` — when the var parses to `False`
it falls back to `forward_native`. Affects every cell (CoT and RAG share the sampling path).

---

## 7. ✅ Hardcoded benchmark.json path in MIRAGE QADataset (FIXED)

Symptom (right after vLLM init succeeds, at `Loading <dataset> dataset...`):
```
File ".../mirage_medrag/MIRAGE/src/utils.py", line 9, in __init__
    benchmark = json.load(open("/data/wang/junh/githubs/mirage_medrag/MIRAGE/benchmark.json"))
FileNotFoundError: '/data/wang/junh/githubs/mirage_medrag/MIRAGE/benchmark.json'
```
Cause: `QADataset.__init__` hardcoded the local-workstation path to `benchmark.json`, ignoring
`MIRAGE_MEDRAG_ROOT`. (`run_debate_medrag_inter.py`/`_rag.py` correctly honor the env var, but
`utils.py` did not.) The file exists on ARC at `…/mirage_medrag/MIRAGE/benchmark.json`.

Fix: resolve it relative to `utils.py`'s own location (benchmark.json sits one dir up):
```python
benchmark = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmark.json")))
```
Verified: `QADataset("medmcqa")` loads 4,183 questions. Affects all cells (every cell builds a
`QADataset`).

Note: `run_debate_medrag_inter.py:52` / `_rag.py:52` also hardcode `MEDCORP_DIR =
/data/wang/...corpus`, but that is RAG-only and the RAG sbatch files override it via
`--db_dir "$CORPUS_DIR"`, so it is not a blocker. Watch for it if running RAG without `--db_dir`.

---

## 5. Run order (after #3 chunk staging is done)

```bash
ssh arc
cd /home/junh/repos/Debate
# (re-rsync DAIH/experiments/sbatch/ first if you applied #2/#3 edits)

# end-to-end smoke test = the smallest RAG cell, via the real batch system:
sbatch DAIH/experiments/sbatch/mirage_05_rag_qwen3_4b_2507_medqa.sbatch
tail -f logs/run_mirage-05*.log
```
Watch for: corpus stages to NVMe → preflight prints `ok: pubmed/chunk (1166 files)` etc. →
**no** "Cloning the … corpus" lines → indexes load → first MedQA question gets a verdict.

If cell 5 runs clean, submit the MedMCQA cells:
```bash
bash DAIH/experiments/sbatch/submit_all.sh 1
bash DAIH/experiments/sbatch/submit_all.sh 2
bash DAIH/experiments/sbatch/submit_all.sh 3
bash DAIH/experiments/sbatch/submit_all.sh 4
bash DAIH/experiments/sbatch/submit_all.sh 6
squeue -u $USER
```
(Cells 1/2/3 = CoT MedMCQA, no retrieval; 4/6 = RAG MedMCQA. Cells 7/8 = PubMedQA/BioASQ are
being done locally — BioASQ already complete at 63.27%.)

MedMCQA = 4,183 questions; ~28 h wall-clock. On TIMEOUT, just re-submit — the runner resumes.

---

## Checklist
- [x] transformers 4.57.1 / tokenizers 0.22.1 in the env
- [x] preflight check added to `_stage_corpus.sh`
- [x] sbatch activation hardening (#2) applied to all 8 files (module pin + PATH prepend)
- [x] VLLM_USE_FLASHINFER_SAMPLER=0 in all 8 sbatch (#6)
- [x] MIRAGE benchmark.json path made relative in utils.py (#7)
- [ ] re-rsync `DAIH/experiments/sbatch/` to ARC to pick up #2/#6 edits
- [ ] chunk text staged to /scratch + symlinks swapped (#3)
- [ ] cell 5 smoke test clean
- [ ] MedMCQA cells 1,2,3,4,6 submitted
