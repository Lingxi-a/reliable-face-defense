from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from PIL import Image

from .metrics import mae, mse, psnr, ssim


def save_image01(x, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (
        x[0].detach().cpu().clamp(0,1)
        .permute(1,2,0).numpy()
    )
    Image.fromarray(np.rint(arr*255).astype(np.uint8)).save(path)


def build_quality_report(clean01, initial01, refined01):
    return {
        "initial": {
            "mae": mae(clean01, initial01),
            "mse": mse(clean01, initial01),
            "psnr": psnr(clean01, initial01),
            "ssim": ssim(clean01, initial01),
        },
        "refined": {
            "mae": mae(clean01, refined01),
            "mse": mse(clean01, refined01),
            "psnr": psnr(clean01, refined01),
            "ssim": ssim(clean01, refined01),
        },
    }


def save_rpp_report(path, *, result, quality=None, extra=None):
    payload = result.summary_dict()
    if quality is not None:
        payload["quality"] = quality
    if extra:
        payload["extra"] = extra
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
