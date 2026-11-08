from __future__ import annotations

from typing import Protocol, runtime_checkable
import torch

from .contracts import Route


_RETINA_MEAN = torch.tensor([104.0,117.0,123.0]).view(1,3,1,1)


def _rgb01_to_retina(x):
    mean = _RETINA_MEAN.to(device=x.device,dtype=x.dtype)
    return x[:,[2,1,0]]*255.0 - mean


def _retina_to_rgb01(x):
    mean = _RETINA_MEAN.to(device=x.device,dtype=x.dtype)
    bgr = x + mean
    return (bgr[:,[2,1,0]]/255.0).clamp(0,1)


@runtime_checkable
class ProtectionProvider(Protocol):
    route: Route

    def protect(
        self,
        *,
        image01: torch.Tensor,
        state,
        epsilon_px: float | None,
    ):
        ...


class OfficialFacePoisonProvider:
    """
    Corrected RGB[0,1] -> official RetinaFace pixel domain -> RGB[0,1] bridge.
    """
    route = Route.FACEPOISON

    def __init__(self, official_adapter):
        self.adapter = official_adapter

    def protect(self, *, image01, state=None, epsilon_px=None):
        if epsilon_px is None:
            raise ValueError("FacePoison requires epsilon_px")
        xret = _rgb01_to_retina(image01)
        if not hasattr(self.adapter,"generate"):
            raise TypeError("Expected Module01B FacePoison adapter.generate")
        adv = self.adapter.generate(
            xret,
            overrides={"eps":float(epsilon_px)},
        )
        return _retina_to_rgb01(adv)


class NullSwapProvider:
    """
    Corrected NullSwap runtime bridge:
      protected_raw -> clamp_01
    """
    route = Route.NULLSWAP

    def __init__(self, generator, *, output_key="protected_raw"):
        self.generator=generator
        self.output_key=output_key

    def protect(self, *, image01, state=None, epsilon_px=None):
        if epsilon_px is not None:
            raise ValueError("NullSwap has no inference-time epsilon knob.")
        out=self.generator(image01)
        if isinstance(out,dict):
            out=out[self.output_key]
        return out.clamp(0,1)


class CallableProtectionProvider:
    def __init__(self, route: Route, fn):
        self.route=route
        self.fn=fn

    def protect(self, *, image01, state=None, epsilon_px=None):
        return self.fn(image01=image01,state=state,epsilon_px=epsilon_px)
