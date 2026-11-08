from __future__ import annotations

import torch


def project_delta_box(
    *,
    clean01: torch.Tensor,
    delta: torch.Tensor,
    epsilon_norm: float,
) -> torch.Tensor:
    """
    Pi_B projection:
      ||delta||_inf <= epsilon
      0 <= clean + delta <= 1

    Both requirements are enforced exactly.
    """
    eps = float(epsilon_norm)
    if eps < 0:
        raise ValueError("epsilon_norm must be >= 0")

    out = torch.clamp(delta, -eps, eps)
    out = torch.maximum(out, -clean01)
    out = torch.minimum(out, 1.0 - clean01)
    return out


def linf_norm(delta: torch.Tensor) -> float:
    return float(delta.detach().abs().max().cpu())
