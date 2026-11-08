from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import random
import numpy as np
import torch

from .facepoison_vendor.facepoison_retina_adapter import (
    FacePoisonRetinaAdapter,
)
from .repro import stable_int_seed


RETINA_MEAN_BGR = torch.tensor([104.0,117.0,123.0]).view(1,3,1,1)


def rgb01_to_retina(x01: torch.Tensor) -> torch.Tensor:
    """
    RGB [0,1] -> BGR pixel-domain -> subtract RetinaFace mean.
    Fully tensor based; no uint8 rounding.
    """
    if x01.ndim != 4 or x01.shape[1] != 3:
        raise ValueError(f"Expected BCHW RGB [0,1], got {tuple(x01.shape)}")
    if not torch.is_floating_point(x01):
        raise TypeError("FacePoison bridge expects a floating RGB tensor.")
    lo=float(x01.detach().min().cpu())
    hi=float(x01.detach().max().cpu())
    if lo < -1e-6 or hi > 1.0+1e-6:
        raise ValueError(f"Expected RGB [0,1], got [{lo},{hi}]")
    mean = RETINA_MEAN_BGR.to(device=x01.device, dtype=x01.dtype)
    bgr255 = x01[:, [2,1,0], :, :] * 255.0
    return bgr255 - mean


def retina_to_rgb01(x_ret: torch.Tensor) -> torch.Tensor:
    """
    Mean-subtracted BGR pixel-domain -> RGB [0,1].
    No quantization is introduced.
    """
    if x_ret.ndim != 4 or x_ret.shape[1] != 3:
        raise ValueError("Expected BCHW RetinaFace tensor.")
    mean = RETINA_MEAN_BGR.to(device=x_ret.device, dtype=x_ret.dtype)
    bgr255 = x_ret + mean
    rgb255 = bgr255[:, [2,1,0], :, :]
    return (rgb255 / 255.0).clamp(0.0,1.0)


@contextmanager
def _preserved_seed(seed: int, device: torch.device):
    """
    Make the stochastic official FacePoison attack reproducible for one
    (sample, epsilon) call without permanently mutating global RNG streams.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    devices = []
    if device.type == "cuda":
        devices = [device.index or 0]
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(int(seed))
            np.random.seed(int(seed) % (2**32 - 1))
            torch.manual_seed(int(seed))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


class FacePoisonRGBProvider:
    """
    Correct integrated FacePoison primitive.

    External system domain:
        RGB [0,1]

    Official FacePoison / RetinaFace domain:
        BGR [0,255] - mean

    The official FacePoison objective is NOT reimplemented.

    Reproducibility:
      the stochastic official attack is deterministically seeded from
      (base_seed, sample_id, epsilon) for validation/resume consistency.
    """
    def __init__(self, *, deps_root, device="cuda", base_seed=2026):
        deps_root = Path(deps_root)
        self.device = torch.device(device)
        self.base_seed = int(base_seed)
        self.adapter = FacePoisonRetinaAdapter(
            facepoison_root=deps_root/"source"/"FacePoison-main",
            retina_source_root=deps_root/"source"/"Pytorch_Retinaface-master",
            retina_weight_path=deps_root/"weights"/"detection"/"mobilenet0.25_Final.pth",
            device=device,
        )

    def protect(self, *, image01, epsilon_px, sample_id="unknown", **_):
        if epsilon_px is None:
            raise ValueError("FacePoison route requires epsilon_px.")
        eps = float(epsilon_px)
        if eps < 0 or eps > 8:
            raise ValueError("Audited FacePoison bound requires 0<=eps<=8.")

        x = image01.to(self.device)
        x_ret = rgb01_to_retina(x)
        seed = stable_int_seed(self.base_seed, str(sample_id), f"{eps:.8f}")

        with _preserved_seed(seed, self.device):
            adv_ret = self.adapter.generate(
                x_ret,
                overrides={"eps": eps},
            )

        adv = retina_to_rgb01(adv_ret)

        # Domain-conversion invariant.
        linf_px = float((adv-x).abs().max().detach().cpu()) * 255.0
        if linf_px > eps + 1e-3:
            raise RuntimeError(
                f"FacePoison RGB-domain L_inf={linf_px:.6f}px exceeds eps={eps}."
            )
        return adv
