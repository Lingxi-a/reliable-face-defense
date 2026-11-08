from __future__ import annotations
import torch
from reliable_face_defense_02r3.nullswap.lpips_adapter import LPIPSAlexNetAdapter


class LPIPSDistortion:
    def __init__(self, device):
        self.model = LPIPSAlexNetAdapter(device=device)

    def __call__(self, clean01, protected01) -> float:
        with torch.no_grad():
            return float(
                self.model(
                    clean01.to(protected01.device),
                    protected01,
                ).detach().cpu()
            )
