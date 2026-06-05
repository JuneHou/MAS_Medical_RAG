#!/bin/bash
# Stage the MedRAG corpus from /scratch into the compute node's local NVMe.
# Sourced (not executed) by each RAG sbatch file before the python invocation.
#
# After this runs:
#   $CORPUS_DIR  is set to a directory containing all the corpus pieces
#                MedRAG expects (statpearls untarred, others symlinked from /scratch).
#
# Why we do this: /scratch has a soft guideline of <10k files per user.
# We ship statpearls (36k files) as a single tar on /scratch and untar it
# per-job into $TMPNVME so the slow filesystem only sees ~2.6k files total.

set -euo pipefail

SCRATCH_CORPUS=/scratch/junh/mirage_medrag/MedRAG/src/data/corpus
CORPUS_DIR=$TMPNVME/corpus
export CORPUS_DIR

mkdir -p "$CORPUS_DIR"

echo "[stage_corpus] untarring statpearls into $CORPUS_DIR"
if [ -d "$CORPUS_DIR/statpearls" ]; then
  echo "[stage_corpus] statpearls already extracted, skipping tar"
else
  time tar -xf "$SCRATCH_CORPUS/statpearls.tar" -C "$CORPUS_DIR"
fi

# Symlink everything else in the scratch corpus dir into $CORPUS_DIR.
# Skip the statpearls.tar itself (we just untarred it).
# Use -f so re-running on the same node (e.g. after a job restart) is idempotent.
echo "[stage_corpus] symlinking other corpus files from $SCRATCH_CORPUS"
for item in "$SCRATCH_CORPUS"/*; do
  name=$(basename "$item")
  if [ "$name" = "statpearls.tar" ] || [ "$name" = "statpearls" ]; then
    continue
  fi
  ln -snf "$item" "$CORPUS_DIR/$name"
done

echo "[stage_corpus] done. CORPUS_DIR=$CORPUS_DIR"
ls -la "$CORPUS_DIR" | head -20

# Preflight: every MedCorp source must have a resolvable, non-empty chunk/ dir.
# The chunk dirs on /scratch are symlinks into the HF cache; if that content was
# not staged they resolve to nothing, and MedRAG silently tries to git-clone the
# corpus from HuggingFace, fails, then OOMs ~30 min later loading the FAISS index.
# Fail loudly here instead.
echo "[stage_corpus] preflight: verifying chunk dirs resolve..."
missing=0
for src in pubmed textbooks wikipedia statpearls; do
  chunk="$CORPUS_DIR/$src/chunk"
  n=$(ls -A "$chunk" 2>/dev/null | wc -l)
  if [ "$n" -eq 0 ]; then
    echo "[stage_corpus] ERROR: $chunk is missing or empty (dangling symlink?)." >&2
    echo "[stage_corpus]        Stage the real chunk text to /scratch (see ARC_SETUP.md Step 3)." >&2
    missing=1
  else
    echo "[stage_corpus]   ok: $src/chunk ($n files)"
  fi
done
if [ "$missing" -ne 0 ]; then
  echo "[stage_corpus] Aborting before model load to avoid a 30-min wait + OOM." >&2
  exit 1
fi
