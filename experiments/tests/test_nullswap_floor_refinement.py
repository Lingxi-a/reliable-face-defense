from __future__ import annotations

import torch

from paper_exp.nullswap_floor_refinement import (
    RefinementConfig,
    VisualMetrics,
    candidate_dominates_anchor,
    project_nonincreasing_l2_direction,
    quantize_png01,
)


def config():
    return RefinementConfig(
        steps=2,
        step_size_px=0.25,
        residual_budget_px=2.0,
        snapshot_steps=(1, 2),
        residual_scales=(0.5, 1.0),
        toward_clean_scales=(0.0,),
        environments=("clean", "jpeg_q60"),
        smoothmax_temperature=20.0,
        min_worst_similarity_gain=0.0005,
        max_condition_regression=0.0,
        max_psnr_drop_db=0.0,
        max_ssim_drop=0.0,
        max_lpips_increase=0.0,
    )


def test_png_quantization_is_idempotent():
    x = torch.tensor([0.0, 0.1, 0.5001, 1.0]).view(1, 1, 2, 2)
    assert torch.equal(quantize_png01(x), quantize_png01(quantize_png01(x)))


def test_direction_projection_is_not_outward():
    clean = torch.zeros(1, 1, 2, 2)
    current = torch.ones(1, 1, 2, 2)
    direction = torch.ones_like(current)
    projected = project_nonincreasing_l2_direction(direction, current, clean)
    dot = ((current - clean) * projected).sum()
    assert float(dot) <= 1e-7


def test_candidate_requires_security_and_all_visual_checks():
    anchor_visual = VisualMetrics(30.0, 0.95, 0.02, 100.0)
    candidate_visual = VisualMetrics(30.1, 0.951, 0.019, 99.0)
    anchor_signature = {"a::clean": 0.20, "f::clean": 0.18}
    candidate_signature = {"a::clean": 0.19, "f::clean": 0.17}
    accepted, checks = candidate_dominates_anchor(
        anchor_visual=anchor_visual,
        candidate_visual=candidate_visual,
        anchor_signature=anchor_signature,
        candidate_signature=candidate_signature,
        cfg=config(),
    )
    assert accepted
    assert checks["worst_similarity_gain"] > 0

    rejected, checks = candidate_dominates_anchor(
        anchor_visual=anchor_visual,
        candidate_visual=VisualMetrics(29.9, 0.951, 0.019, 99.0),
        anchor_signature=anchor_signature,
        candidate_signature=candidate_signature,
        cfg=config(),
    )
    assert not rejected
    assert not checks["psnr_ok"]


def test_conditionwise_regression_is_rejected():
    anchor_visual = VisualMetrics(30.0, 0.95, 0.02, 100.0)
    signature = {"a::clean": 0.20, "f::clean": 0.18}
    candidate = {"a::clean": 0.19, "f::clean": 0.181}
    accepted, checks = candidate_dominates_anchor(
        anchor_visual=anchor_visual,
        candidate_visual=anchor_visual,
        anchor_signature=signature,
        candidate_signature=candidate,
        cfg=config(),
    )
    assert not accepted
    assert not checks["conditionwise_no_regression"]

