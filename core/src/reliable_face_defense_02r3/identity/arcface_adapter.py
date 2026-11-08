from __future__ import annotations
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import compatible_state_dict, freeze_module


class ArcFaceR50Adapter(nn.Module):
    """
    Existing project adapter convention:
      RGB [0,1] -> 112x112 bilinear -> (x - 0.5) / 0.5
    """
    def __init__(self, deps_root, device="cpu"):
        super().__init__()
        deps_root = Path(deps_root)
        arc_root = (
            deps_root
            / "source"
            / "insightface-master"
            / "recognition"
            / "arcface_torch"
        )
        if str(arc_root) not in sys.path:
            sys.path.insert(0, str(arc_root))

        from backbones import iresnet50

        self.model = iresnet50(fp16=False)
        weight_path = (
            deps_root
            / "weights"
            / "identity"
            / "arcface_r50"
            / "backbone.pth"
        )
        raw = torch.load(weight_path, map_location="cpu")
        used, missing, extra = compatible_state_dict(self.model, raw)
        result = self.model.load_state_dict(used, strict=False)

        self.load_report = {
            "weight": str(weight_path),
            "model_keys": len(self.model.state_dict()),
            "used_keys": len(used),
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
            "filtered_missing": missing,
            "filtered_extra": extra,
        }
        if result.missing_keys or result.unexpected_keys or missing:
            raise RuntimeError(
                "ArcFace checkpoint/model mismatch: "
                f"missing={len(result.missing_keys)} "
                f"unexpected={len(result.unexpected_keys)} "
                f"filtered_missing={len(missing)}"
            )

        freeze_module(self.model)
        self.model.to(device)

    def preprocess(self, x01):
        x = F.interpolate(
            x01, size=(112, 112),
            mode="bilinear", align_corners=False
        )
        return (x - 0.5) / 0.5

    def forward(self, x01):
        emb = self.model(self.preprocess(x01))
        return F.normalize(emb, dim=1)
