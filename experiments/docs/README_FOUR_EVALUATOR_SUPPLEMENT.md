# Four-evaluator main-experiment supplement

This supplement adds ArcFace and FaceNet Acc@1/Acc@5 to the completed main
experiment while preserving its VGGFace/SFace rows and visual-quality metrics.

Protocol safeguards:

- reads the completed run's `frozen_main_config.json` and protocol;
- keeps the original SimSwap/BlendSwap/FaceShifter eligibility denominators;
- never edits the original `results/per_pair.jsonl` or `main_table.csv`;
- stops if replay disagrees with an originally successful swap instead of
  fabricating an identity score;
- labels ArcFace/FaceNet as project-internal/surrogate evaluators and keeps
  VGGFace/SFace labeled as held-out evaluators.

All outputs are written under:

`/root/autodl-tmp/reliable_paper_main_v07/four_evaluator_supplement/`

The paper-facing outputs follow the NullSwap layout instead of forcing all
dimensions into one oversized table:

- `paper_table_1_visual_quality.csv`: methods x PSNR/SSIM/LPIPS;
- `paper_table_2_source_cloaking.csv`: four recognizers as rows, with each
  method grouped into Acc@5/Acc@1 columns;
- `paper_table_3_face_swap_acc1.csv`: three face-swap models as rows, using
  the two held-out evaluators;
- `paper_table_3_face_swap_acc5.csv`: the corresponding Acc@5 table.

`main_table_four_evaluators.csv` is retained only as a complete appendix/data
record. The layout follows NullSwap, but accuracy is never mislabeled as cosine
similarity.

The command is resumable because completed rows are appended to
`per_pair_four_evaluators.jsonl`.
