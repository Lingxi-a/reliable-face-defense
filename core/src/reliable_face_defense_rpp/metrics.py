from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def mae(a, b):
    return float((a-b).abs().mean().detach().cpu())


def mse(a, b):
    return float((a-b).pow(2).mean().detach().cpu())


def psnr(a, b, data_range=1.0):
    m = mse(a, b)
    if m == 0:
        return float("inf")
    return 10.0 * math.log10((data_range ** 2) / m)


def ssim(a, b, data_range=1.0, window_size=7):
    """
    Compact differentiable-style SSIM metric for reporting only.
    Uses local average pooling, not a trainable loss.
    """
    pad = window_size // 2
    mu_a = F.avg_pool2d(a, window_size, stride=1, padding=pad)
    mu_b = F.avg_pool2d(b, window_size, stride=1, padding=pad)

    sigma_a = F.avg_pool2d(a*a, window_size, 1, pad) - mu_a*mu_a
    sigma_b = F.avg_pool2d(b*b, window_size, 1, pad) - mu_b*mu_b
    sigma_ab = F.avg_pool2d(a*b, window_size, 1, pad) - mu_a*mu_b

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    score = (
        (2*mu_a*mu_b + c1) * (2*sigma_ab + c2)
        / ((mu_a.pow(2)+mu_b.pow(2)+c1) * (sigma_a+sigma_b+c2))
    )
    return float(score.mean().detach().cpu())
