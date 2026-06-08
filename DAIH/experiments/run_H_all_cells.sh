#!/bin/bash
# Cond H' (single-agent oss RAG, debate-ablation control) across all 4 clinical
# cells. API-only (reuses A' retrieved docs); resumable. Run from repo root.
set -uo pipefail
REPO_ROOT=/data/wang/junh/githubs/Debate
cd "$REPO_ROOT"
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate /data/wang/junh/envs/medrag
set -a; source /data/wang/junh/.cache/keys/arc_llm_api.sh; set +a
export ARC_LLM_API_KEY="${API_KEY:-${ARC_LLM_API_KEY:-}}"

for cell in mimic3:mortality mimic4:mortality mimic3:readmission mimic4:readmission; do
  DS="${cell%%:*}"; TASK="${cell##*:}"
  echo "############ Cond H' : $DS / $TASK  ############"
  python DAIH/experiments/run_condition_H_oss.py --task "$TASK" --dataset "$DS"
done
echo "############ ALL H' CELLS DONE ############"
