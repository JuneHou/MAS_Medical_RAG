# MIRAGE gap-fill — deferred work (RAG cells 4–8)

**Status as of 2026-06-05.** CoT cells are validated and running on ARC; RAG cells are
deferred to prioritize other experiments. This file tracks what's left so the RAG cells can
be resumed later without re-diagnosing. Detailed per-issue history is in
[`DAIH/arc_error.md`](DAIH/arc_error.md).

## ✅ Done (do not redo)

The environment + sbatch pipeline is fixed and the **CoT path is validated end-to-end** on
ARC (smoke test ran clean; cell 1 produced 98+ debates with zero errors). Fixes applied:
- **#2** conda activation — `module load Miniconda3/24.7.1-0` + `export PATH=.../medrag/bin:$PATH`
  in all 8 sbatch (the bare `module load Miniconda3` default flipped to 25.11.1-1 / py3.13,
  whose `conda activate` doesn't update PATH).
- **#6** `export VLLM_USE_FLASHINFER_SAMPLER=0` in all 8 sbatch (compute node has no nvcc, so
  FlashInfer's JIT sampler kernel can't build; use PyTorch-native sampling).
- **#7** `MIRAGE/src/utils.py` benchmark.json path made relative to `__file__` (was hardcoded
  to a `/data/wang/...` local path).

**Submitted / running (CoT, MedMCQA):** cell 1 (Qwen3-8B), cell 2 (Qwen2.5-7B),
cell 3 (Qwen3-4B-2507). These resume on TIMEOUT — just resubmit; the runner skips existing
`*.json`. Note: throughput is ~4–5 min/question (≈300 GPU-hours for 4,183 q), so each cell
needs several 36h resubmits.

## ⏸️ Deferred — RAG cells 4, 5, 6 (`run_debate_medrag_rag.py`)

- 4: RAG Qwen2.5-7B MedMCQA · 5: RAG Qwen3-4B-2507 MedQA · 6: RAG Qwen3-4B-2507 MedMCQA

**Blockers before these can run** (both still open):
1. **Corpus chunk staging (arc_error.md #3) — half-done.** On `/scratch/junh/mirage_medrag/
   MedRAG/src/data/corpus`: `pubmed/chunk__staged` (1166 files) and `wikipedia/chunk__staged`
   (646 files) are present, but the symlink swap was never done, and **`textbooks/chunk` is
   not staged at all** (its symlink points to a local `/data/wang/...` path that can't resolve
   on ARC). `_stage_corpus.sh`'s preflight checks all four sources and will abort RAG jobs in
   ~1s until fixed.
   - Step A (LOCAL workstation — ARC can't reach the source): rsync textbooks chunk text:
     ```bash
     SRC=/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus
     DST=junh@tinkercliffs1.arc.vt.edu:/scratch/junh/mirage_medrag/MedRAG/src/data/corpus
     rsync -rL --no-times --partial --progress "$SRC/textbooks/chunk/" "$DST/textbooks/chunk__staged/"
     ```
   - Step B (ARC): swap symlinks for all three:
     ```bash
     cd /scratch/junh/mirage_medrag/MedRAG/src/data/corpus
     for s in pubmed textbooks wikipedia; do rm -f "$s/chunk" && mv "$s/chunk__staged" "$s/chunk"; done
     # expect: pubmed 1166 / textbooks 18 / wikipedia 646 files
     ```
2. **RAG path not yet validated.** The smoke test only exercised CoT. After staging, smoke-test
   the smallest RAG cell first (short walltime), then submit 4 & 6:
   ```bash
   sbatch DAIH/experiments/sbatch/mirage_05_rag_qwen3_4b_2507_medqa.sbatch   # RAG smoke
   bash   DAIH/experiments/sbatch/submit_all.sh 4 6
   ```
   Watch for: preflight prints `ok: pubmed/chunk (1166 files)` etc. → **no** "Cloning the …
   corpus" lines → FAISS indexes load (needs ~170 G host RAM + the sbatch's `--mem=256G`) →
   first question gets a verdict.

## ⏸️ Deferred — cells 7, 8 (run locally, not on ARC)

- 7: RAG Qwen3-4B-2507 PubMedQA · 8: RAG Qwen3-4B-2507 BioASQ
- Per arc_error.md these are being run on the local workstation, not ARC (BioASQ was at
  63.27%). Do **not** submit 7/8 to ARC. Finish/collect locally; fold results into the repo.

## Open follow-ups (not blocking)
- `run_debate_medrag_inter.py:52` / `_rag.py:52` hardcode `MEDCORP_DIR = /data/wang/...` — only
  used by RAG and overridden by the sbatch `--db_dir "$CORPUS_DIR"`, so not a blocker, but
  worth parameterizing if the RAG runs are ever launched without `--db_dir`.
- ARC `Debate`/`mirage_medrag` are rsynced copies, **not** git clones — keep edits (sbatch
  fixes, utils.py, arc_error.md, this file) in sync with the real clone when merging.
