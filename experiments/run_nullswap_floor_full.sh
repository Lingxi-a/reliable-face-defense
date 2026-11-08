#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="/root/autodl-tmp/reliable_face_defense_module05_visualft/src:${PYTHONPATH:-}"

python -u run_nullswap_floor_selfcheck.py \
  --config configs/nullswap_floor_rerun.json

python -u run_nullswap_floor_rerun.py \
  --config configs/nullswap_floor_rerun.json \
  --stage all

