from __future__ import annotations

from io import BytesIO
import numpy as np
from PIL import Image, ImageFilter
import torch

from .contracts import RedTransformSpec


def _tensor_to_pil(x):
    arr = (
        x.detach().cpu().clamp(0, 1)
        .permute(1, 2, 0).numpy()
    )
    arr = np.rint(arr * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def _pil_to_tensor(img, device, dtype):
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2,0,1).to(device=device, dtype=dtype)


def _jpeg(img, q):
    buf = BytesIO()
    img.save(
        buf,
        format="JPEG",
        quality=int(q),
        optimize=False,
        progressive=False,
        subsampling="4:2:0",
    )
    buf.seek(0)
    out = Image.open(buf).convert("RGB")
    out.load()
    return out


def apply_red_transform(x01: torch.Tensor, spec: RedTransformSpec) -> torch.Tensor:
    if x01.ndim != 4 or x01.shape[1] != 3:
        raise ValueError("Expected BCHW RGB tensor")
    device, dtype = x01.device, x01.dtype
    outs = []
    for b in range(x01.shape[0]):
        img = _tensor_to_pil(x01[b])

        if spec.family == "jpeg":
            out = _jpeg(img, spec.params[0])

        elif spec.family == "blur":
            sigma = float(spec.params[0])
            # Exact library operator was not fixed by the method text.
            # Engineering implementation: Pillow GaussianBlur(radius=sigma).
            out = img.filter(ImageFilter.GaussianBlur(radius=sigma))

        elif spec.family == "osn":
            out = img
            for q in spec.params:
                out = _jpeg(out, q)

        else:
            raise ValueError(f"Unknown transform family: {spec.family}")

        outs.append(_pil_to_tensor(out, device, dtype))
    return torch.stack(outs, dim=0)
