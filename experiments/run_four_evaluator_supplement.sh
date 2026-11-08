#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/reliable_paper_experiments
mkdir -p /root/autodl-tmp/reliable_paper_main_v07/four_evaluator_supplement
export PYTHONPATH="/root/autodl-tmp/reliable_face_defense_module05_visualft/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

python -u run_four_evaluator_supplement.py \
  --main-dir /root/autodl-tmp/reliable_paper_main_v07 \
  --config configs/main_experiment.json \
  2>&1 | tee -a /root/autodl-tmp/reliable_paper_main_v07/four_evaluator_supplement/run.log
