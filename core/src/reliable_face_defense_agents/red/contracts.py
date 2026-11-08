from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class RedTransformSpec:
    name: str
    family: str
    params: tuple[Any, ...]

    def to_dict(self):
        return {
            "name": self.name,
            "family": self.family,
            "params": list(self.params),
        }


@dataclass(frozen=True)
class ViolationObservation:
    """
    Continuous threshold-normalized violation values are supplied by the real
    detector/identity evaluation layer.

    Red does not define thresholds itself.

    q(x,T) = V_D + V_I is diagnostic only.
    Red selection uses attack_success = D_success AND I_success.
    """
    v_detection: float
    v_identity: float
    detection_success: bool
    identity_success: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def q(self) -> float:
        return float(self.v_detection) + float(self.v_identity)

    @property
    def attack_success(self) -> bool:
        return bool(self.detection_success and self.identity_success)


@dataclass(frozen=True)
class TransformRisk:
    """
    q_mean is diagnostic only.
    breach_rate is the current control-set attack-success rate.
    ema_risk is the EMA of breach_rate and drives Top-K selection.
    """
    transform: RedTransformSpec
    q_mean: float
    ema_risk: float
    breach_rate: float
    num_samples: int

    def to_dict(self):
        return {
            "transform": self.transform.to_dict(),
            "q_mean": self.q_mean,
            "ema_risk": self.ema_risk,
            "breach_rate": self.breach_rate,
            "num_samples": self.num_samples,
        }


@dataclass(frozen=True)
class RedDecision:
    epoch: int
    selected_topk: tuple[str, ...]
    active_constraints: tuple[str, ...]
    risks: tuple[TransformRisk, ...]
    reason: str

    def to_dict(self):
        return {
            "epoch": self.epoch,
            "selected_topk": list(self.selected_topk),
            "active_constraints": list(self.active_constraints),
            "reason": self.reason,
            "risks": [x.to_dict() for x in self.risks],
        }
