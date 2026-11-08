from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class Route(str, Enum):
    FACEPOISON = "facepoison"
    NULLSWAP = "nullswap"


@dataclass(frozen=True)
class BlueState:
    sample_id: str
    active_red_constraints: tuple[str, ...] = ()
    round_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvironmentOutcome:
    environment: str
    detection_success: bool
    identity_success: bool
    attack_success: bool
    defense_success: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateEvaluation:
    sample_id: str
    route: Route
    outcomes: tuple[EnvironmentOutcome, ...]
    distortion: float
    distortion_metric: str
    epsilon_px: float | None = None
    artifact: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def reliable(self) -> bool:
        return bool(self.outcomes) and all(x.defense_success for x in self.outcomes)

    @property
    def failed_environments(self) -> tuple[str, ...]:
        return tuple(
            x.environment for x in self.outcomes if not x.defense_success
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "route": self.route.value,
            "epsilon_px": self.epsilon_px,
            "reliable": self.reliable,
            "failed_environments": list(self.failed_environments),
            "distortion": self.distortion,
            "distortion_metric": self.distortion_metric,
            "outcomes": [asdict(x) for x in self.outcomes],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class BlueDecision:
    sample_id: str
    round_index: int
    chosen: CandidateEvaluation | None
    candidates: tuple[CandidateEvaluation, ...]
    predicted_epsilon_px: float | None
    searched_epsilon_px: tuple[float, ...]
    active_red_constraints: tuple[str, ...]
    status: str
    reason: str

    @property
    def feasible(self) -> bool:
        return self.status == "selected" and self.chosen is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "round_index": self.round_index,
            "status": self.status,
            "feasible": self.feasible,
            "predicted_epsilon_px": self.predicted_epsilon_px,
            "searched_epsilon_px": list(self.searched_epsilon_px),
            "active_red_constraints": list(self.active_red_constraints),
            "reason": self.reason,
            "chosen": None if self.chosen is None else self.chosen.to_dict(),
            "candidates": [x.to_dict() for x in self.candidates],
        }
