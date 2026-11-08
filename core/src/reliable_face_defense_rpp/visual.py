from __future__ import annotations

import torch
import torch.nn as nn


class LPIPSAlexNet01(nn.Module):
    """
    LPIPS AlexNet adapter for RGB [0,1] tensors.

    LPIPS expects [-1,1], so this adapter performs only that documented range
    conversion. Pretrained AlexNet/LPIPS weights are handled by the `lpips`
    package/cache already prepared in the project environment.
    """
    def __init__(self, *, device="cpu"):
        super().__init__()
        import lpips
        self.model = lpips.LPIPS(net="alex").to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, clean01: torch.Tensor, protected01: torch.Tensor) -> torch.Tensor:
        a = clean01 * 2.0 - 1.0
        b = protected01 * 2.0 - 1.0
        return self.model(a, b).mean()


class PerceptualObjective(nn.Module):
    r"""
    Eq.(21):

        D_per(x, x+delta)
          = lambda_pix ||delta||_2^2
            + lambda_lpips LPIPS(x, x+delta)

    The squared L2 norm is computed per image then averaged over the batch.
    This preserves the mathematical ||delta||_2^2 form rather than silently
    converting it to MSE.
    """
    def __init__(
        self,
        *,
        lambda_pix: float,
        lambda_lpips: float,
        lpips_model: nn.Module | None,
    ):
        super().__init__()
        self.lambda_pix = float(lambda_pix)
        self.lambda_lpips = float(lambda_lpips)
        self.lpips_model = lpips_model

        if self.lambda_lpips > 0 and self.lpips_model is None:
            raise ValueError("lambda_lpips > 0 requires an LPIPS model.")

    def components(
        self,
        clean01: torch.Tensor,
        delta: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        protected = torch.clamp(clean01 + delta, 0.0, 1.0)
        pixel_l2_sq = delta.flatten(1).pow(2).sum(dim=1).mean()

        if self.lambda_lpips > 0:
            lp = self.lpips_model(clean01, protected)
        else:
            lp = torch.zeros(
                (),
                device=clean01.device,
                dtype=clean01.dtype,
            )

        total = (
            self.lambda_pix * pixel_l2_sq
            + self.lambda_lpips * lp
        )
        return {
            "total": total,
            "pixel_l2_sq": pixel_l2_sq,
            "lpips": lp,
        }

    def forward(
        self,
        clean01: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        return self.components(clean01, delta)["total"]
