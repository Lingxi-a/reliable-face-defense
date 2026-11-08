# Reliability-Anchored Dynamic Blue-Red Defense against Face Swapping

Official research code for a sample-level reliable and transmission-robust active defense framework against face swapping.

## Method

The v0.7 framework contains three principal components:

1. Dynamic Blue Agent: selects FacePoison- or NullSwap-based protection according to sample-level feedback.
2. Red Agent: simulates JPEG, Gaussian blur, and OSN transmission conditions.
3. RPP: refines protected images under robustness and visual-quality constraints.

Safety events:

- AttackSuccess = D_success AND I_success
- DefenseSuccess = D_fail OR I_fail

D denotes face-detection status and I denotes identity-retrieval status.

## Evaluation Protocol

- Dataset: CelebA-HQ
- Face-swapping models: SimSwap, BlendSwap, FaceShifter
- Training identity models: ArcFace, FaceNet
- Held-out evaluators: VGGFace, SFace
- Metrics: Acc@1, Acc@5, PSNR, SSIM, LPIPS
- Lower is better for Acc@1, Acc@5, and LPIPS
- Higher is better for PSNR and SSIM

The qualitative protocol retains 70 groups of Source, Target, Protected, Clean Swap, and Protected Swap images.

## Repository Structure

- core: core defense implementation
- experiments/configs: experiment configurations
- experiments/paper_exp: formal experiment modules
- experiments/tests: implementation tests
- experiments/docs: supplementary protocol notes
- experiments/run_*.py: Python entry points
- experiments/run_*.sh: AutoDL launch scripts

## Main Experiment Modules

- protection.py: protection generation and Blue Agent integration
- swappers.py: face-swapping model interfaces
- evaluators.py: detection and identity evaluation
- runner.py: main experiment runner
- nullswap_floor_refinement.py: candidate refinement
- nullswap_floor_rerun.py: anchor-based rerun experiment
- four_evaluator_supplement.py: four-evaluator analysis
- reviewer_decisive_experiments.py: robustness, gate, and cluster analyses
- aligned_blue_red_ablation.py: aligned Blue-Red ablation

## Installation and Configuration

Install the core package with: pip install -e ./core

Configuration files are under experiments/configs.

The configuration files preserve paths from the original AutoDL environment. Dataset, checkpoint, external repository, and output paths must be updated for a different environment.

## Release Scope

This repository does not contain:

- CelebA-HQ images or raw experimental data
- Model weights or checkpoints
- Generated protected images
- Clean or protected face-swapping outputs
- Per-sample records or aggregated results
- Logs, caches, or historical development artifacts

Third-party datasets, implementations, and weights must be obtained from their official sources.

## Ethical Use

This code is intended for academic research on identity protection and defensive analysis of face-swapping systems. Users are responsible for compliance with applicable laws, licenses, privacy requirements, and institutional ethical guidelines.
