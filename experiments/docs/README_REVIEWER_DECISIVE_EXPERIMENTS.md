# Reviewer decisive experiments hotfix v05

This hotfix corrects the G3-to-Dynamic-Red alias used only when aggregating
the gate-ablation table. It intentionally keeps the v04 output directory and
frozen experiment identifier so a completed v04 run resumes without repeating
protection generation or already completed swap rows.

This patch runs the reviewer-requested experiments under the already frozen
paper protocol. It does not alter the completed baseline outputs.

## Included experiments

1. Dynamic Red strategy ablation using the same full reliability gate:
   - Clean-only optimization;
   - uniform multi-condition optimization;
   - anchor-frozen softmax Red weights;
   - current-candidate dynamic softmax Red weights.
2. Reliability gate ablation from the exact same Dynamic Red candidate pool:
   - G0 no gate;
   - G1 worst-risk gain;
   - G2 worst-risk gain plus condition-wise non-regression;
   - G3 full gate including visual non-regression.
3. Test-only unseen propagation conditions:
   - JPEG Q45 and Q75;
   - Gaussian blur sigma 0.5 and 1.5;
   - double JPEG Q65 then Q85.
4. Source-clustered paired bootstrap with 2000 replicates.
5. Saved-PNG residual budget, runtime, and peak-memory audits.

The held-out VGGFace and SFace evaluators are never used to select candidates.
No RPP, EMA routing, dynamic epsilon, RL, CVaR, Resize, extra dataset, or extra
face swapper is introduced.

## Install

Upload `reviewer_decisive_experiments_hotfix_v05.tar.gz` to `/root`, then run:

```bash
tar -xzf /root/reviewer_decisive_experiments_hotfix_v05.tar.gz \
  -C /root/autodl-tmp/reliable_paper_experiments

cd /root/autodl-tmp/reliable_paper_experiments

python -m py_compile \
  run_reviewer_decisive_experiments.py \
  paper_exp/reviewer_decisive_refinement.py \
  paper_exp/reviewer_decisive_experiments.py

nohup bash run_reviewer_decisive_experiments.sh \
  > reviewer_decisive_launcher.out 2>&1 &

echo $! | tee reviewer_decisive.pid
tail -f reviewer_decisive_launcher.out
```

Press `Ctrl+C` to leave `tail`; this does not stop the background experiment.

## Resume

Run the same `nohup` command again. Completed protection images, transformed
images, and per-pair evaluation rows are detected and skipped. Do not delete or
edit the frozen output directory between resumes.

## Output

The default result directory is:

```text
/root/autodl-tmp/reviewer_decisive_experiments_v04
```

After completion the launcher creates:

```text
/root/autodl-tmp/reviewer_decisive_experiments_v04_results.tar.gz
```

Upload that single archive for analysis.

Important paper-facing CSV files:

- `dynamic_red_strategy_ablation.csv`
- `dynamic_red_strategy_robust_all.csv`
- `reliability_gate_ablation.csv`
- `reliability_gate_robust_all.csv`
- `unseen_propagation_table.csv`
- `unseen_propagation_robust_all.csv`
- `cluster_bootstrap_main.csv`
- `paired_transitions_main.csv`
- `selection_gate_runtime_audit.csv`

## Comparison policy

The patch directly runs the requested supplementary experiments. It does not
try to reproduce or compare against the previously saved Ours PNGs. The Dynamic
Red branch is regenerated and evaluated in the same run as the other ablation
branches so that the controlled comparison uses identical inputs, denominators,
budgets, and evaluation definitions.

