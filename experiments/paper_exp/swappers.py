from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gc

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .common import isolated_import, load_rgb01


SWAP_OK = "SWAP_OK"
SOURCE_DETECTION_FAILED = "SOURCE_DETECTION_FAILED"
TARGET_DETECTION_FAILED = "TARGET_DETECTION_FAILED"
ENGINEERING_ERROR = "ENGINEERING_ERROR"


@dataclass
class SwapOutput:
    status: str
    image01: torch.Tensor | None = None
    error: str | None = None


def _unwrap_state(value):
    if not isinstance(value, dict):
        raise TypeError("Checkpoint is not a state dictionary.")
    for key in ("state_dict", "model", "generator", "G"):
        if key in value and isinstance(value[key], dict):
            return value[key]
    return value


def _load_checkpoint(path, *, mmap=False):
    kwargs = {"map_location": "cpu", "weights_only": False}
    if mmap:
        kwargs["mmap"] = True
    try:
        return _unwrap_state(torch.load(str(path), **kwargs))
    except RuntimeError as exc:
        if mmap and "mmap" in str(exc).lower():
            kwargs.pop("mmap", None)
            return _unwrap_state(torch.load(str(path), **kwargs))
        raise


def _load_meta_model(factory, state, device):
    try:
        with torch.device("meta"):
            model = factory()
        result = model.load_state_dict(state, strict=True, assign=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"strict load mismatch: missing={result.missing_keys}, "
                f"unexpected={result.unexpected_keys}"
            )
        return model.to(device).eval()
    except (TypeError, NotImplementedError):
        model = factory()
        model.load_state_dict(state, strict=True)
        return model.to(device).eval()


class SimSwapRunner:
    def __init__(self, cfg, device):
        from reliable_face_defense_fullsystem.config import FaceSwapConfig
        from reliable_face_defense_fullsystem.faceswap import SimSwapAdapter

        x = cfg["simswap"]
        fs_cfg = FaceSwapConfig(
            enabled=True,
            model="simswap",
            repo_root=x["repo_root"],
            checkpoints_dir=x["checkpoints_dir"],
            arcface_path=x["arcface_path"],
            detector_model_path=x["detector_model_path"],
            detector_fallbacks=tuple(x["detector_fallbacks"]),
            crop_size=int(x["crop_size"]),
            det_thresh=float(x["det_thresh"]),
            targets_per_source=int(cfg["targets_per_source"]),
            strict_engineering_errors=True,
        )
        self.adapter = SimSwapAdapter(cfg=fs_cfg, device=device)

    def swap(self, source_path, target_path):
        out = self.adapter.swap_paths(
            source_path=str(source_path), target_path=str(target_path)
        )
        return SwapOutput(out.status, out.image01, out.error)


class BlendSwapRunner:
    def __init__(self, cfg, device):
        x = cfg["blendswap"]
        self.device = torch.device(device)
        swapping_root = Path(x["repo_root"]) / "swapping"
        with isolated_import(swapping_root, ("model", "blendswap")):
            from blendswap import BlendSwap
            state = _load_checkpoint(x["checkpoint"], mmap=True)
            self.model = _load_meta_model(BlendSwap, state, self.device)
        del state
        gc.collect()

    def swap(self, source_path, target_path):
        try:
            source = load_rgb01(source_path, 112).to(self.device)
            target = load_rgb01(target_path, 256).to(self.device)
            with torch.no_grad():
                output = self.model(target, source).clamp(0.0, 1.0)
            return SwapOutput(SWAP_OK, output.detach())
        except Exception as exc:
            return SwapOutput(ENGINEERING_ERROR, error=f"{type(exc).__name__}: {exc}")


class FaceShifterRunner:
    def __init__(self, cfg, device):
        x = cfg["faceshifter"]
        self.device = torch.device(device)
        repo = Path(x["repo_root"])
        with isolated_import(repo, ("network", "face_modules")):
            from network.AEI_Net import AEI_Net
            from face_modules.model import Backbone

            g_state = _load_checkpoint(x["generator_checkpoint"], mmap=True)
            up1 = g_state.get("generator.up1.weight")
            if up1 is None:
                raise KeyError("FaceShifter checkpoint lacks generator.up1.weight")
            c_id = int(up1.shape[0])
            self.generator = _load_meta_model(
                lambda: AEI_Net(c_id=c_id), g_state, self.device
            )
            del g_state

            a_state = _load_checkpoint(x["arcface_checkpoint"], mmap=True)
            self.arcface = _load_meta_model(
                lambda: Backbone(50, 0.6, "ir_se"), a_state, self.device
            )
            del a_state
        gc.collect()

    def swap(self, source_path, target_path):
        try:
            source = load_rgb01(source_path, 256).to(self.device) * 2.0 - 1.0
            target = load_rgb01(target_path, 256).to(self.device) * 2.0 - 1.0
            with torch.no_grad():
                crop = source[:, :, 19:237, 19:237]
                crop = F.interpolate(
                    crop, (112, 112), mode="bilinear", align_corners=True
                )
                identity, _ = self.arcface(crop)
                output, _ = self.generator(target, identity)
                output = (output * 0.5 + 0.5).clamp(0.0, 1.0)
            return SwapOutput(SWAP_OK, output.detach())
        except Exception as exc:
            return SwapOutput(ENGINEERING_ERROR, error=f"{type(exc).__name__}: {exc}")


def make_swapper(name, cfg, device):
    if name == "simswap":
        return SimSwapRunner(cfg, device)
    if name == "blendswap":
        return BlendSwapRunner(cfg, device)
    if name == "faceshifter":
        return FaceShifterRunner(cfg, device)
    raise KeyError(name)


def release_swapper(value):
    for name in list(vars(value)):
        try:
            delattr(value, name)
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
