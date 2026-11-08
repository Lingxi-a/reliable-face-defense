from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass(frozen=True)
class DynamicBudgetEngineeringConfig:
    """
    The project method fixes the budget parameterization, not the low-level
    neural architecture. These channel sizes are therefore engineering choices.
    """
    channels: tuple[int, int, int] = (16, 32, 64)
    hidden_dim: int = 64


class DynamicBudgetNet(nn.Module):
    r"""
    Sample-wise dynamic budget controller.

    Project method:
        epsilon_theta(x)
          = epsilon_min
            + (epsilon_max-epsilon_min) * sigmoid(a_theta(x))

    The method history fixes:
        epsilon_min = 4/255
        epsilon_max = 8/255

    Internally this class returns both normalized epsilon and pixel-domain
    epsilon, because the downloaded FacePoison implementation uses pixel-domain
    `eps` values such as 8.
    """

    def __init__(
        self,
        *,
        epsilon_min_norm: float = 4.0 / 255.0,
        epsilon_max_norm: float = 8.0 / 255.0,
        engineering: DynamicBudgetEngineeringConfig | None = None,
    ):
        super().__init__()
        if not (0 < epsilon_min_norm <= epsilon_max_norm):
            raise ValueError("Invalid dynamic-budget interval.")
        self.epsilon_min_norm = float(epsilon_min_norm)
        self.epsilon_max_norm = float(epsilon_max_norm)
        eng = engineering or DynamicBudgetEngineeringConfig()

        c1, c2, c3 = eng.channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, c1, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, eng.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(eng.hidden_dim, 1),
        )

    def forward(self, x01: torch.Tensor) -> dict[str, torch.Tensor]:
        if x01.ndim != 4 or x01.shape[1] != 3:
            raise ValueError(f"Expected BCHW RGB input, got {tuple(x01.shape)}")
        a = self.head(self.encoder(x01)).squeeze(1)
        sigma = torch.sigmoid(a)
        eps_norm = (
            self.epsilon_min_norm
            + (self.epsilon_max_norm - self.epsilon_min_norm) * sigma
        )
        eps_px = eps_norm * 255.0
        return {
            "a": a,
            "epsilon_norm": eps_norm,
            "epsilon_px": eps_px,
        }


def gated_minimum_sufficient_budget_loss(
    *,
    reliability_loss: torch.Tensor,
    epsilon_norm: torch.Tensor,
    epsilon_min_norm: float,
    epsilon_max_norm: float,
    lambda_budget: float,
    gamma_gate: float,
) -> dict[str, torch.Tensor]:
    r"""
    Implements the previously fixed minimum-sufficient-budget formulation:

        L_budget = (eps-eps_min)/(eps_max-eps_min)
        w = exp(-gamma * stopgrad(L_rel))
        L_MS = L_rel + lambda_b * w * L_budget

    lambda_budget and gamma_gate are deliberately required arguments because
    their numeric values are not fixed by the current method handoff.
    """
    if epsilon_max_norm <= epsilon_min_norm:
        raise ValueError("epsilon_max_norm must exceed epsilon_min_norm")
    l_budget = (
        (epsilon_norm - epsilon_min_norm)
        / (epsilon_max_norm - epsilon_min_norm)
    ).mean()
    rel = reliability_loss.mean()
    weight = torch.exp(-float(gamma_gate) * rel.detach())
    total = rel + float(lambda_budget) * weight * l_budget
    return {
        "total": total,
        "reliability": rel,
        "budget": l_budget,
        "gate": weight,
    }
