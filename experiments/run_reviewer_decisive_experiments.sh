#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/root/autodl-tmp/reliable_paper_experiments}"
MAIN_OUTPUT="${MAIN_OUTPUT:-/root/autodl-tmp/reliable_paper_main_v07}"
FINAL_OUTPUT="${FINAL_OUTPUT:-/root/autodl-tmp/reliable_paper_v07_nullswap_floor_rerun}"
FOUR_EVAL_OUTPUT="${FOUR_EVAL_OUTPUT:-/root/autodl-tmp/four_evaluator_supplement_v08}"
OUTPUT="${OUTPUT:-/root/autodl-tmp/reviewer_decisive_experiments_v04}"

cd "${EXPERIMENT_ROOT}"
mkdir -p "${OUTPUT}"
export PYTHONPATH="/root/autodl-tmp/reliable_face_defense_module05_visualft/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

python -u run_reviewer_decisive_experiments.py \
  --main-output "${MAIN_OUTPUT}" \
  --final-output "${FINAL_OUTPUT}" \
  --four-evaluator-output "${FOUR_EVAL_OUTPUT}" \
  --output "${OUTPUT}" \
  --stage preflight \
  2>&1 | tee -a "${OUTPUT}/00_preflight.log"

python -u run_reviewer_decisive_experiments.py \
  --main-output "${MAIN_OUTPUT}" \
  --final-output "${FINAL_OUTPUT}" \
  --four-evaluator-output "${FOUR_EVAL_OUTPUT}" \
  --output "${OUTPUT}" \
  --stage all \
  2>&1 | tee -a "${OUTPUT}/reviewer_decisive_experiments.log"

python - "${OUTPUT}" <<'PY'
from pathlib import Path
import sys
import tarfile

output = Path(sys.argv[1]).resolve()
archive = output.parent / f"{output.name}_results.tar.gz"
with tarfile.open(archive, "w:gz") as handle:
    handle.add(output, arcname=output.name)
print(f"RESULT_ARCHIVE={archive}", flush=True)
PY

