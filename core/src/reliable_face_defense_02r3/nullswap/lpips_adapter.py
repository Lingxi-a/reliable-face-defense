from __future__ import annotations
import torch
import torch.nn as nn
from .range_policy import to_lpips_range


class LPIPSAlexNetAdapter(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        try:
            import lpips
        except Exception as exc:
            raise RuntimeError(
                "lpips package is unavailable. Module 02C/02R2 setup must pass first."
            ) from exc

        self.model = lpips.LPIPS(net="alex")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.to(device)

    def forward(self, clean01, protected01):
        # lpips package expects [-1,1] for normalized=True-style usage here.
        d = self.model(
            to_lpips_range(clean01),
            to_lpips_range(protected01),
        )
        return d.mean()
