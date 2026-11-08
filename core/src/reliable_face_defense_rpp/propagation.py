from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import math

import torch
import torch.nn.functional as F


class IdentityDifferentiableTransform:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TorchGaussianBlurTransform:
    """
    Differentiable Gaussian blur.

    sigma is a method/experiment value.
    Kernel construction is an explicit engineering implementation detail.
    """
    def __init__(self, sigma: float, truncate: float = 3.0):
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        self.sigma = float(sigma)
        self.truncate = float(truncate)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        radius = max(1, int(math.ceil(self.truncate * self.sigma)))
        coords = torch.arange(
            -radius,
            radius + 1,
            device=x.device,
            dtype=x.dtype,
        )
        kernel1 = torch.exp(-(coords ** 2) / (2 * self.sigma ** 2))
        kernel1 = kernel1 / kernel1.sum()
        kernel2 = kernel1[:, None] * kernel1[None, :]
        kernel2 = kernel2.expand(x.shape[1], 1, -1, -1)

        pad = radius
        xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(xp, kernel2, groups=x.shape[1])


class StraightThroughHardTransform:
    """
    Engineering adapter for propagation operators whose forward path is
    nondifferentiable (e.g. hard JPEG / current OSN proxy).

    Forward:
        exactly hard_fn(x)

    Backward:
        straight-through identity gradient:
        y = x + stopgrad(hard_fn(x)-x)

    This is NOT claimed as a paper-specified JPEG gradient. It is an explicit
    engineering bridge so RPP can execute Eq.(25) through the currently used
    hard propagation forward operator until a calibrated differentiable proxy
    is plugged in.
    """
    def __init__(self, hard_fn: Callable[[torch.Tensor], torch.Tensor]):
        self.hard_fn = hard_fn

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            hard = self.hard_fn(x)
        return x + (hard - x).detach()


class DifferentiableTransformRegistry:
    def __init__(self, mapping: dict[str, Callable[[torch.Tensor], torch.Tensor]]):
        self.mapping = dict(mapping)
        self.mapping.setdefault("clean", IdentityDifferentiableTransform())

    def apply(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if name not in self.mapping:
            raise KeyError(
                f"No differentiable RPP propagation operator registered for '{name}'."
            )
        return self.mapping[name](x)

    @classmethod
    def from_03bc_hard_red_pool_with_ste(cls):
        """
        Concrete bridge to Module 03BC v0.3.11.

        - clean: identity
        - Blur: differentiable Torch Gaussian blur using the same sigma
        - JPEG/OSN: exact current hard forward wrapped with STE gradient

        No Resize is introduced.
        """
        from reliable_face_defense_agents.red.presets import (
            validation_red_pool,
            full_method_red_pool,
        )
        from reliable_face_defense_agents.red.transforms import apply_red_transform

        specs = {}
        for s in list(validation_red_pool()) + list(full_method_red_pool()):
            specs[s.name] = s

        mapping = {"clean": IdentityDifferentiableTransform()}
        for name, spec in specs.items():
            if spec.family == "blur":
                mapping[name] = TorchGaussianBlurTransform(float(spec.params[0]))
            elif spec.family in {"jpeg", "osn"}:
                def make_hard(sp):
                    return lambda x: apply_red_transform(x, sp)
                mapping[name] = StraightThroughHardTransform(make_hard(spec))
            else:
                raise ValueError(f"Unexpected 03BC Red family: {spec.family}")
        return cls(mapping)
