from __future__ import annotations

from pathlib import Path
import hashlib
import torch

from reliable_face_defense_02r3.nullswap.generator import (
    NullSwapGenerator,
    NullSwapEngineeringConfig,
)
from reliable_face_defense_02r3.nullswap.range_policy import (
    apply_protected_range_policy,
)


def _stable_seed(sample_id: str, base_seed: int) -> int:
    h = hashlib.sha256(f"{base_seed}:{sample_id}".encode()).digest()
    return (int.from_bytes(h[:8], "little") % (2**31-1))


class NullSwapRuntime:
    """
    Full-width paper-faithful NullSwap inference wrapper.

    Important correction vs the earlier Blue adapter:
      generator(...)[protected_raw]
        -> apply the SAME protected range policy used in training (`clamp_01`)
        -> RGB [0,1]
    """
    def __init__(
        self,
        *,
        checkpoint,
        device="cuda",
        base_channels=64,
        base_seed=2026,
    ):
        if int(base_channels) != 64:
            raise ValueError(
                "Integrated formal runtime requires NullSwap base_channels=64."
            )
        self.device = torch.device(device)
        self.base_seed = int(base_seed)
        self.generator = NullSwapGenerator(
            engineering=NullSwapEngineeringConfig(base_channels=64)
        ).to(self.device)

        state = torch.load(
            str(checkpoint),
            map_location=self.device,
            weights_only=False,
        )
        if isinstance(state, dict) and "generator" in state:
            state = state["generator"]
        if not isinstance(state, dict):
            raise TypeError("Unsupported NullSwap checkpoint format.")
        self.generator.load_state_dict(state, strict=True)
        self.generator.eval()

    def protect_with_diagnostics(self, *, image01, sample_id, **_):
        x = image01.to(self.device)
        seed = _stable_seed(sample_id, self.base_seed)

        devices = []
        if self.device.type == "cuda":
            devices = [self.device.index or 0]

        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            with torch.no_grad():
                raw = self.generator(x)["protected_raw"]
                protected = apply_protected_range_policy(raw, "clamp_01")
        diagnostics = {
            "protected_raw_min": float(raw.min().detach().cpu()),
            "protected_raw_max": float(raw.max().detach().cpu()),
            "raw_le_0_fraction": float((raw <= 0.0).float().mean().detach().cpu()),
            "raw_ge_1_fraction": float((raw >= 1.0).float().mean().detach().cpu()),
            "protected_near_0_fraction": float((protected < 0.01).float().mean().detach().cpu()),
            "protected_near_1_fraction": float((protected > 0.99).float().mean().detach().cpu()),
        }
        return protected, diagnostics

    def protect(self, *, image01, sample_id, **kwargs):
        return self.protect_with_diagnostics(
            image01=image01, sample_id=sample_id, **kwargs
        )[0]
