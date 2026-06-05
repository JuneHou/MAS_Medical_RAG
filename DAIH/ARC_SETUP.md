# ARC Setup — MIRAGE Gap-Fill Preparation

> **Internal lab document.** Do not commit to a public repo. Paths and infrastructure details below are for our setup only.

Goal: run 8 multi-agent debate experiments on VT ARC to fill missing cells in the MIRAGE benchmark table for the DAIH workshop. Each experiment is one slurm job. All 8 are submitted in parallel and finish in roughly 1–2 days of wall-clock time depending on the queue.

Estimated one-time prep cost (everything in this document): roughly 4–6 hours of human time spread across two long-running transfers, plus a 30-minute conda env build and a 30-minute smoke test.

---

## What will be where

| What | Where | Size | Why |
|---|---|---:|---|
| `mirage_medrag` repo code | `/home/junh/repos/mirage_medrag/` | < 1 GB | small, permanent |
| `Debate` repo code | `/home/junh/repos/Debate/` | < 1 GB | small, permanent |
| Conda env `medrag` | `/home/junh/envs/medrag/` | ~15 GB | permanent, points all jobs at it |
| HF model cache | `/scratch/junh/hf_cache/` | ~40 GB after auto-downloads | big, transient — fine in /scratch (90-day window) |
| MedRAG corpus (everything except statpearls) | `/scratch/junh/mirage_medrag/MedRAG/src/data/corpus/` | ~450 GB | big, transient |
| MedRAG corpus statpearls (tarred) | `/scratch/junh/mirage_medrag/MedRAG/src/data/corpus/statpearls.tar` | ~6.5 GB | one tarball keeps the file count low |
| Per-job results | `/scratch/junh/mirage_outputs/...` | small | rsync the summaries to /home before /scratch ages them out |

ARC's `/scratch` has no per-user size quota but auto-deletes anything older than 90 days. The workshop finishes well inside that window, so we don't need to worry about purges during the run — just remember to rsync the final results to `/home` before walking away.

---

## Prerequisites — verify these once

Before starting:

1. **SSH access to ARC.** You already have it. Test:
   ```bash
   ssh arc 'hostname && whoami'
   ```
   It should print an ARC login node hostname and `junh`.

2. **Your local corpus path.** The data we need lives at:
   ```
   /data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus/
   ```
   Total local size: 511 GB (verified). We will skip a few files we don't need; the actual transfer size is ~457 GB.

3. **A long-running shell on your workstation.** Use `tmux` or `screen`. The corpus transfer will take several hours.

---

## Step 1 — Tar the StatPearls subdirectory locally (one minute)

ARC's `/scratch` has a soft guideline of fewer than 10,000 files per user. Our corpus has 38,729 files; 36,082 of those are in `statpearls/`. Tarring that one subdirectory drops the count to about 2,648 files, well under the limit.

On your local workstation:

```bash
cd /data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus
tar -cf statpearls.tar statpearls/
ls -lh statpearls.tar    # should be roughly 6.5 GB
```

(We do NOT delete the original `statpearls/` directory locally — we still want it intact for any local runs.)

---

## Step 2 — Push code to /home on ARC (a few minutes)

You either need an SSH alias `arc` in `~/.ssh/config`, or you can paste your ARC login hostname in place of `arc` in every command below (e.g., `junh@tinkercliffs1.arc.vt.edu`).

First, create all the directories you'll need on ARC in one shot:

```bash
ssh arc 'mkdir -p /home/junh/repos /home/junh/envs /scratch/junh/hf_cache /scratch/junh/mirage_medrag/MedRAG/src/data/corpus'
```

Now push the two repos from your local workstation. **Exclude the big data subdirectories** — the corpus goes to /scratch separately in Step 3, and the HF dataset cache is not needed at all on ARC.

```bash
# mirage_medrag repo (code only — corpus + hf_cache excluded)
rsync -av --exclude '__pycache__' --exclude '.git' \
  --exclude 'MedRAG/src/data/corpus' \
  --exclude 'MedRAG/src/data/hf_cache' \
  /data/wang/junh/githubs/mirage_medrag/ \
  arc:/home/junh/repos/mirage_medrag/

# Debate repo (code only — KARE's heavy data/checkpoints excluded; not needed for MIRAGE work)
rsync -av --exclude '__pycache__' --exclude '.git' \
  --exclude 'KARE/searchr1' \
  --exclude 'KARE/data' \
  --exclude 'KARE/results' \
  --exclude 'KARE/results_unbiased' \
  --exclude 'debate_logs' \
  --exclude 'debate_logs_boxed' \
  /data/wang/junh/githubs/Debate/ \
  arc:/home/junh/repos/Debate/
```

Total expected size after both rsyncs: about 1.5 GB in `/home/junh/repos/`.

Verify on ARC:
```bash
ssh arc 'ls /home/junh/repos/'
# Expect: mirage_medrag  Debate
```

---

## Step 3 — Push the corpus to /scratch on ARC (multi-hour)

This is the long-running step. Use tmux on your local workstation so it survives disconnects. (The destination directory `/scratch/junh/mirage_medrag/MedRAG/src/data/corpus/` was already created in Step 2.)

```bash
# In tmux on local workstation:

# Push everything except statpearls/ (we ship the tar instead) and three unused id2text files.
# --no-times = assign fresh modification times on ARC so the 90-day purge clock starts now.
rsync -av --partial --progress --no-times \
  --exclude 'statpearls/' \
  --exclude 'PubMed_id2text.json' \
  --exclude 'Wikipedia_id2text.json' \
  --exclude 'StatPearls_id2text.json' \
  /data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus/ \
  arc:/scratch/junh/mirage_medrag/MedRAG/src/data/corpus/

# Push the statpearls tarball (1 file)
rsync -av --partial --progress --no-times \
  /data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus/statpearls.tar \
  arc:/scratch/junh/mirage_medrag/MedRAG/src/data/corpus/
```

Estimated transfer time: depends entirely on your local-to-ARC link speed. Test the link with the first GB or so and use `--progress` to estimate the rest. At 100 MB/s, 457 GB takes about 75 minutes; on a slower WAN link, several hours.

Verify on ARC after it finishes:
```bash
ssh arc 'ls -la /scratch/junh/mirage_medrag/MedRAG/src/data/corpus/'
ssh arc 'du -sh /scratch/junh/mirage_medrag/MedRAG/src/data/corpus/'
# Expect ~457 GB.
ssh arc 'find /scratch/junh/ -type f | wc -l'
# Expect roughly 2,648 — well under the 10k file-count guideline.
```

---

## Step 4 — Build the conda env on ARC (~30 minutes)

The env file is at `/home/junh/repos/mirage_medrag/environment.yml` (pushed in Step 2). It pins every version (Python 3.10, vLLM 0.11.0, torch 2.8.0, transformers 4.57.1, faiss-cpu, sentence-transformers, pyserini, spacy — all of it).

SSH to an ARC login node, then:

```bash
module load Miniconda3
mkdir -p /home/junh/envs

conda env create \
  -p /home/junh/envs/medrag \
  -f /home/junh/repos/mirage_medrag/environment.yml
```

This takes 15–30 minutes depending on package solver speed and download bandwidth. If it fails on some pip-only packages, the conda-managed packages should still install; we can resolve the rest with `pip install` afterwards. Send me the error and I'll write the patch.

Activate and sanity check:
```bash
conda activate /home/junh/envs/medrag
python -c "import torch, vllm, transformers, faiss, sentence_transformers, pyserini; print('OK', torch.__version__, vllm.__version__)"
```

Expected output:
```
OK 2.8.0 0.11.0
```

If any import fails, stop here and tell me which one — installing the missing pieces is much easier before we kick off the long runs.

---

## Step 5 — Smoke test on ARC (~10 minutes)

Before submitting all 8 jobs, run one tiny check to confirm the pipeline works end-to-end on a compute node.

Grab an interactive 1-GPU session:

```bash
srun --account=slmreasoning --partition=a100_normal_q --gres=gpu:2 \
     --time=00:30:00 --cpus-per-task=8 --mem=64G --pty bash
```

Inside the interactive session:

```bash
module load Miniconda3
conda activate /home/junh/envs/medrag

export HF_HOME=/scratch/junh/hf_cache
export MIRAGE_MEDRAG_ROOT=/home/junh/repos/mirage_medrag
mkdir -p $HF_HOME
cd /home/junh/repos/Debate

# Untar statpearls into the node-local NVMe for the smoke test
mkdir -p $TMPNVME/corpus
tar -xf /scratch/junh/mirage_medrag/MedRAG/src/data/corpus/statpearls.tar -C $TMPNVME/corpus
ln -sn /scratch/junh/mirage_medrag/MedRAG/src/data/corpus/pubmed       $TMPNVME/corpus/
ln -sn /scratch/junh/mirage_medrag/MedRAG/src/data/corpus/wikipedia    $TMPNVME/corpus/
ln -sn /scratch/junh/mirage_medrag/MedRAG/src/data/corpus/textbooks    $TMPNVME/corpus/
ln -sn /scratch/junh/mirage_medrag/MedRAG/src/data/corpus/umls         $TMPNVME/corpus/
ln -sn /scratch/junh/mirage_medrag/MedRAG/src/data/corpus/MedCorp_id2text.json   $TMPNVME/corpus/
ln -sn /scratch/junh/mirage_medrag/MedRAG/src/data/corpus/MedCorp2_id2text.json  $TMPNVME/corpus/

# Run one cell on the smallest dataset (BioASQ, 618 questions) with --limit if available,
# or just let it process the first few and Ctrl-C once you've confirmed it's working.
python -u run_debate_medrag_rag.py \
  --dataset bioasq \
  --model Qwen/Qwen2.5-7B-Instruct \
  --gpu_ids 0,1 \
  --log_dir ./debate_logs_smoke \
  --corpus_name MedCorp2 \
  --retriever_name MedCPT
```

What to watch for:
- The HF model snapshot downloads to `$HF_HOME` on first run (expect a few minutes for ~15 GB).
- vLLM loads on GPU 1.
- FAISS indexes load on GPU 0.
- The first question gets processed; you see "Agent1 response", "Agent2 response", then a verdict.

Once you see a clean output for the first question or two, **Ctrl-C** to stop. Exit the interactive session.

If anything failed: copy the error and send it to me. Don't proceed to Step 6.

---

## Step 6 — Submit all 8 jobs (a few seconds)

The 8 sbatch files live under `/home/junh/repos/Debate/DAIH/experiments/sbatch/`. **They are currently written for the old `/projects/slmreasoning` paths and need to be rewritten with the new `/home` + `/scratch` setup before you submit them.** Please ask me to rewrite them (or I will do it as a follow-up step). Once rewritten:

```bash
ssh arc
cd /home/junh/repos/Debate
bash DAIH/experiments/sbatch/submit_all.sh
```

Submit all 8 cells. Each cell is its own slurm job. They queue in parallel and run as A100s free up. Check status:

```bash
squeue -u $USER
```

You should see 8 lines, each one named like `mirage-04-rag-qwen25-7b-medmcqa`.

---

## Step 7 — Monitor and re-submit if a job times out (over 1–2 days)

The MedMCQA cells (cells 1, 2, 3, 4, 6 — the ones running 4,183 questions) are the slowest. We requested a 36-hour walltime; if a job hits that limit before finishing, the script's built-in resume support means a fresh submission of the same sbatch will pick up where it left off. So if you see `TIMEOUT` in the slurm output for cell N, just re-submit it:

```bash
bash DAIH/experiments/sbatch/submit_all.sh N
```

The non-MedMCQA cells (5, 7, 8) should finish on the first try.

Each cell writes its per-question logs and a final results JSON to `/home/junh/repos/Debate/debate_logs_mirage_gapfill/...` (CoT) or `debate_logs_mirage_gapfill_rag/...` (RAG). Check the latest activity with:

```bash
tail -f /home/junh/repos/Debate/logs/run_mirage-*.log
```

---

## Step 8 — Pull results back to /home and to your local workstation (last step)

When all 8 jobs are done:

```bash
# On ARC: confirm every cell has a final results JSON
ssh arc 'find /home/junh/repos/Debate/debate_logs_mirage_gapfill* -name "*_results.json"'
# Expect 8 files.

# Pull results back to your local workstation
rsync -av arc:/home/junh/repos/Debate/debate_logs_mirage_gapfill/      /data/wang/junh/githubs/Debate/debate_logs_mirage_gapfill/
rsync -av arc:/home/junh/repos/Debate/debate_logs_mirage_gapfill_rag/  /data/wang/junh/githubs/Debate/debate_logs_mirage_gapfill_rag/
```

The big corpus on `/scratch` will purge itself in 90 days. The conda env and code on `/home` stay; you can clean them up later if you want the space back:

```bash
ssh arc 'rm -rf /home/junh/envs/medrag /home/junh/repos/mirage_medrag /home/junh/repos/Debate'
```

---

## What to do if something goes wrong

- **rsync drops the connection mid-transfer.** Just re-run the same rsync command. `--partial` means it resumes; no need to start over.
- **conda env build fails on some pip package.** Send me the error; usually a one-line `pip install` fix or a version pin update.
- **smoke test loads the model fine but the first FAISS query crashes.** Check that the corpus symlinks all resolve (`ls -la $TMPNVME/corpus/`) and that the tar extracted correctly (`ls $TMPNVME/corpus/statpearls/ | head`).
- **A job dies with OOM.** Bump `--mem` in that sbatch from 256G to 384G and re-submit.
- **A job dies right after vLLM startup.** Usually a CUDA version mismatch; check `nvidia-smi` in the job log and confirm the env's torch CUDA build matches the node's driver.

When in doubt, copy the failing log file's tail (last 50 lines) and paste it back to me.

---

## Summary of what you actually need to do in order

1. **Local workstation, ~1 min:** tar statpearls (Step 1).
2. **Local workstation, ~5 min:** rsync code to /home on ARC (Step 2).
3. **Local workstation, ~1–6 hours (in tmux):** rsync corpus to /scratch on ARC (Step 3).
4. **ARC login node, ~30 min:** create conda env (Step 4).
5. **ARC interactive job, ~10 min:** smoke test (Step 5).
6. **Tell me everything looks good and I'll rewrite the 8 sbatch files for the new paths.**
7. **ARC login node, ~5 sec:** submit_all.sh (Step 6).
8. **Over 1–2 days:** monitor and re-submit any time-outs (Step 7).
9. **End:** rsync results back to local; clean up if desired (Step 8).
