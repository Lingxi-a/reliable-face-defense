from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import sys

import numpy as np
from PIL import Image
import torch


RETINAFACE_MEAN_BGR = np.array([104.0, 117.0, 123.0], dtype=np.float32)


def strip_module_prefix(state_dict: dict) -> dict:
    out = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        out[new_key] = value
    return out


def load_checkpoint_state_dict(path: str | Path) -> dict:
    obj = torch.load(str(path), map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(obj)!r}")
    return strip_module_prefix(obj)


def image_to_retina_tensor(
    image_path: str | Path,
    size: int = 640,
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB").resize((size, size), Image.BILINEAR)
    rgb = np.asarray(image, dtype=np.float32)
    bgr = rgb[..., ::-1].copy()
    bgr -= RETINAFACE_MEAN_BGR
    chw = np.transpose(bgr, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0)


def retina_tensor_to_rgb_image(x: torch.Tensor) -> np.ndarray:
    """
    x: [1,3,H,W] or [3,H,W], RetinaFace-preprocessed tensor (BGR, mean-subtracted)
    returns uint8 RGB image
    """
    if x.ndim == 4:
        x = x[0]
    arr = x.detach().cpu().numpy().astype(np.float32)
    chw_bgr = arr + RETINAFACE_MEAN_BGR[:, None, None]
    hwc_bgr = np.transpose(chw_bgr, (1, 2, 0))
    hwc_rgb = hwc_bgr[..., ::-1]
    hwc_rgb = np.clip(hwc_rgb, 0, 255).astype(np.uint8)
    return hwc_rgb


def save_retina_tensor_image(x: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = retina_tensor_to_rgb_image(x)
    Image.fromarray(rgb).save(path)


def save_perturbation_visual(delta: torch.Tensor, path: str | Path, scale: float = 8.0) -> None:
    """
    delta is in RetinaFace-preprocessed pixel space. We visualize the BGR perturbation
    converted to RGB and centered at 127.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if delta.ndim == 4:
        delta = delta[0]
    arr = delta.detach().cpu().numpy().astype(np.float32)
    hwc_bgr = np.transpose(arr, (1, 2, 0))
    hwc_rgb = hwc_bgr[..., ::-1]
    vis = np.clip(127.5 + (hwc_rgb / max(scale, 1e-6)) * 127.5, 0, 255).astype(np.uint8)
    Image.fromarray(vis).save(path)


@dataclass
class RetinaFaceLoadReport:
    checkpoint_keys: int
    model_keys: int
    used_keys: int
    missing_keys: int
    unexpected_keys: int


class RetinaFaceMobile025Adapter:
    def __init__(
        self,
        source_root: str | Path,
        weight_path: str | Path,
        device: str = "cpu",
    ) -> None:
        self.source_root = Path(source_root).resolve()
        self.weight_path = Path(weight_path).resolve()
        self.device = torch.device(device)

        if not self.source_root.exists():
            raise FileNotFoundError(f"RetinaFace source not found: {self.source_root}")
        if not self.weight_path.exists():
            raise FileNotFoundError(f"RetinaFace weight not found: {self.weight_path}")

        source_str = str(self.source_root)
        if source_str not in sys.path:
            sys.path.insert(0, source_str)

        from data.config import cfg_mnet
        from models.retinaface import RetinaFace

        cfg = dict(cfg_mnet)
        cfg["pretrain"] = False
        self.cfg = cfg
        self.model = RetinaFace(cfg=cfg, phase="test")

        state = load_checkpoint_state_dict(self.weight_path)
        model_keys = set(self.model.state_dict().keys())
        checkpoint_keys = set(state.keys())
        used = model_keys & checkpoint_keys

        incompatible = self.model.load_state_dict(state, strict=False)
        self.load_report = RetinaFaceLoadReport(
            checkpoint_keys=len(checkpoint_keys),
            model_keys=len(model_keys),
            used_keys=len(used),
            missing_keys=len(incompatible.missing_keys),
            unexpected_keys=len(incompatible.unexpected_keys),
        )

        if not used:
            raise RuntimeError(
                "The RetinaFace checkpoint shares zero keys with the model."
            )

        self.model.to(self.device)
        self.model.eval()

    def backbone_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = x.to(self.device)
        out = self.model.body(x)
        return {str(k): v for k, v in out.items()}

    def forward(self, x: torch.Tensor):
        return self.model(x.to(self.device))
