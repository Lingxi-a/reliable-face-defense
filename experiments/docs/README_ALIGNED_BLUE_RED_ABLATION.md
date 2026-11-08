# Protocol-aligned Blue-only versus Blue+Red ablation

This patch isolates the propagation-constraint contribution while keeping all
other factors fixed.

- Blue-only: frozen NullSwap anchor plus the final residual search, with
  candidate reliability checked on Clean only.
- Blue+Red: reuses the completed final Ours protections and results, generated
  with Clean/JPEG/Blur/OSN constraints.
- Both use the same candidate grid, internal ArcFace/FaceNet selection models,
  held-out VGGFace/SFace evaluation, target-index-0 pairs, gallery, detector,
  transforms, fixed eligibility denominator, and conservative error policy.

The only newly evaluated workload is Blue-only:

`200 sources x 4 environments x 3 swap models = 2400 swaps`.

The run is resumable and does not overwrite any previous result.

## Required completed directories

- `/root/autodl-tmp/reliable_paper_main_v07`
- `/root/autodl-tmp/reliable_paper_v07_nullswap_floor_rerun`

## Install and run

```bash
tar -xzf /root/autodl-tmp/aligned_blue_red_ablation_patch_v01.tar.gz \
  -C /root/autodl-tmp/reliable_paper_experiments

cd /root/autodl-tmp/reliable_paper_experiments

python -m py_compile \
  run_aligned_blue_red_ablation.py \
  paper_exp/aligned_blue_red_ablation.py \
  paper_exp/nullswap_floor_refinement.py

nohup bash run_aligned_blue_red_ablation.sh \
  > aligned_blue_red_launcher.out 2>&1 &

echo $! | tee aligned_blue_red.pid
tail -f aligned_blue_red_launcher.out
```

## Outputs

Results are written to:

`/root/autodl-tmp/aligned_blue_red_ablation_v01`

Main files:

- `aligned_blue_red_table.csv`
- `robust_all_pool.csv`
- `paired_transition_counts.csv`
- `blue_only_per_pair.csv`
- `summary.json`
- `protocol.json`

After completion:

```bash
cd /root/autodl-tmp
tar -czf aligned_blue_red_ablation_v01_results.tar.gz \
  aligned_blue_red_ablation_v01
```
