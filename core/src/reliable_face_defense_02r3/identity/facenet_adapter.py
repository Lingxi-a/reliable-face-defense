from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import compatible_state_dict, freeze_module


class FaceNetVGGFace2Adapter(nn.Module):
    """
    Existing project adapter convention:
      RGB [0,1] -> 160x160 bilinear -> x*255 -> (x-127.5)/128
    """
    def __init__(self, deps_root, device="cpu"):
        super().__init__()
        try:
            from facenet_pytorch import InceptionResnetV1
        except Exception as exc:
            raise RuntimeError(
                "facenet_pytorch import failed. Use the locally installed "
                "facenet-pytorch source package from FaceDefenseDependencies."
            ) from exc

        self.model = InceptionResnetV1(
            pretrained=None,
            classify=False,
        )
        deps_root = Path(deps_root)
        weight_path = (
            deps_root
            / "weights"
            / "identity"
            / "facenet"
            / "20180402-114759-vggface2.pt"
        )
        raw = torch.load(weight_path, map_location="cpu")
        used, missing, extra = compatible_state_dict(self.model, raw)
        result = self.model.load_state_dict(used, strict=False)

        # The previously audited checkpoint has two extra checkpoint keys and
        # no required model keys missing. We tolerate extras, not missing model keys.
        self.load_report = {
            "weight": str(weight_path),
            "model_keys": len(self.model.state_dict()),
            "used_keys": len(used),
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
            "filtered_missing": missing,
            "filtered_extra": extra,
        }
        if result.missing_keys or missing:
            raise RuntimeError(
                "FaceNet checkpoint/model mismatch: "
                f"missing={len(result.missing_keys)} "
                f"filtered_missing={len(missing)}"
            )

        freeze_module(self.model)
        self.model.to(device)

    def preprocess(self, x01):
        x = F.interpolate(
            x01, size=(160, 160),
            mode="bilinear", align_corners=False
        )
        x = x * 255.0
        return (x - 127.5) / 128.0

    def forward(self, x01):
        emb = self.model(self.preprocess(x01))
        return F.normalize(emb, dim=1)
