from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

import torch


Branch = Literal["detection", "identity"]


@dataclass(frozen=True)
class ConstraintTerm:
    """
    One differentiable inequality constraint:

        g_m(delta) <= 0

    `value` must be a scalar tensor connected to delta for RPP repair.
    """
    name: str
    branch: Branch
    environment: str
    value: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def violated(self) -> bool:
        return float(self.value.detach().cpu()) > 0.0

    def detached_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "branch": self.branch,
            "environment": self.environment,
            "value": float(self.value.detach().cpu()),
            "violated": self.violated,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ReliabilitySnapshot:
    """
    Task-level OR reliability and route-certificate reliability.

    task_reliable:
        detection_protected OR identity_protected

    certificate_reliable:
        all constraints of the Blue-selected certificate branch are <= 0.
        Preserving this is sufficient to preserve task reliability.
    """
    detection_protected: bool
    identity_protected: bool
    task_reliable: bool
    certificate_branch: Branch
    certificate_reliable: bool
    environments: tuple[str, ...]
    max_detection_violation: float
    max_identity_violation: float
    terms: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class RPPConfig:
    """
    Method-level RPP hyperparameters are explicit.

    No numeric values are silently claimed as paper values.

    Required:
      steps            J perceptual refinement steps
      eta_visual       eta_v in Eq.(22)
      rho_repair       rho in Eq.(25), 0 < rho <= 1
      xi               denominator stabilizer in Eq.(25)
      k_repair         repeated sequential repair passes
      lambda_pix       pixel L2 coefficient
      lambda_lpips     LPIPS coefficient

    `accept_only_if_visual_improves` implements the project's hard safety
    gate: a reliable candidate that does not improve perceptual objective can
    simply keep the previous reliable iterate.
    """
    steps: int
    eta_visual: float
    rho_repair: float
    xi: float
    k_repair: int
    lambda_pix: float
    lambda_lpips: float
    accept_only_if_visual_improves: bool = True
    visual_improvement_tolerance: float = 0.0

    def __post_init__(self):
        if self.steps < 0:
            raise ValueError("steps must be >= 0")
        if self.eta_visual <= 0:
            raise ValueError("eta_visual must be > 0")
        if not (0 < self.rho_repair <= 1):
            raise ValueError("rho_repair must satisfy 0 < rho <= 1")
        if self.xi <= 0:
            raise ValueError("xi must be > 0")
        if self.k_repair < 1:
            raise ValueError("k_repair must be >= 1")
        if self.lambda_pix < 0 or self.lambda_lpips < 0:
            raise ValueError("visual loss weights must be nonnegative")
        if self.lambda_pix == 0 and self.lambda_lpips == 0:
            raise ValueError("At least one visual loss term must be active.")


@dataclass(frozen=True)
class RPPStepRecord:
    step: int
    visual_before: float
    visual_after_perceptual_descent: float
    visual_after_repair: float
    accepted: bool
    rolled_back: bool
    repair_triggered: bool
    repair_passes_used: int
    sequential_projection_count: int
    snapshot_after_descent: ReliabilitySnapshot
    snapshot_after_repair: ReliabilitySnapshot
    reason: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass(frozen=True)
class RPPResult:
    status: str
    refined: torch.Tensor
    initial: torch.Tensor
    delta_initial: torch.Tensor
    delta_final: torch.Tensor
    epsilon_norm: float
    certificate_branch: Branch
    active_environments: tuple[str, ...]
    initial_snapshot: ReliabilitySnapshot
    final_snapshot: ReliabilitySnapshot
    steps: tuple[RPPStepRecord, ...]
    accepted_steps: int
    rollback_steps: int
    total_sequential_projections: int

    def summary_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "epsilon_norm": self.epsilon_norm,
            "certificate_branch": self.certificate_branch,
            "active_environments": list(self.active_environments),
            "accepted_steps": self.accepted_steps,
            "rollback_steps": self.rollback_steps,
            "total_sequential_projections": self.total_sequential_projections,
            "initial_snapshot": asdict(self.initial_snapshot),
            "final_snapshot": asdict(self.final_snapshot),
            "steps": [x.to_dict() for x in self.steps],
        }
