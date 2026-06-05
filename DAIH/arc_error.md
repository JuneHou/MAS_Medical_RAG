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
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/junh/envs/medrag
export PYTHONNOUSERSITE=1
# fail fast if python isn't 3.10 from the env:
python -c "import sys; assert sys.executable.startswith('/home/junh/envs/medrag/'), sys.executable" \
  || { echo 'WRONG PYTHON — env not active' >&2; exit 1; }
```

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
- [ ] sbatch activation hardening (#2) applied to all 8 files + re-rsync to ARC
- [ ] chunk text staged to /scratch + symlinks swapped (#3)
- [ ] cell 5 smoke test clean
- [ ] MedMCQA cells 1,2,3,4,6 submitted
