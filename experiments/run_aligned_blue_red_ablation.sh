#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/root/autodl-tmp/reliable_paper_experiments}"
MAIN_OUTPUT="${MAIN_OUTPUT:-/root/autodl-tmp/reliable_paper_main_v07}"
FINAL_OUTPUT="${FINAL_OUTPUT:-/root/autodl-tmp/reliable_paper_v07_nullswap_floor_rerun}"
ABLATION_OUTPUT="${ABLATION_OUTPUT:-/root/autodl-tmp/aligned_blue_red_ablation_v01}"

cd "${EXPERIMENT_ROOT}"
mkdir -p "${ABLATION_OUTPUT}"
export PYTHONPATH="/root/autodl-tmp/reliable_face_defense_module05_visualft/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

python -u run_aligned_blue_red_ablation.py \
  --main-output "${MAIN_OUTPUT}" \
  --final-output "${FINAL_OUTPUT}" \
  --ablation-output "${ABLATION_OUTPUT}" \
  2>&1 | tee -a "${ABLATION_OUTPUT}/aligned_blue_red_ablation.log"
