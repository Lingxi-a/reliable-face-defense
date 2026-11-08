from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Iterable

import torch

from .contracts import (
    RedTransformSpec,
    TransformRisk,
    RedDecision,
)
from .transforms import apply_red_transform


@dataclass(frozen=True)
class RedAgentConfig:
    """
    Full Dynamic-Red configuration.

    Method:
      A(x,T) = 1[D_success AND I_success]
      R_T    = mean attack success over the control set
      EMA_T  = alpha * EMA_prev + (1-alpha) * R_T
      A_e    = A_{e-1} UNION TopK(EMA_T)

    q=V_D+V_I is retained only as diagnostic information.
    Numeric alpha is experiment configuration and must be supplied.
    """
    ema_alpha: float
    top_k: int = 1

    def __post_init__(self):
        if not (0.0 <= float(self.ema_alpha) < 1.0):
            raise ValueError("ema_alpha must satisfy 0 <= alpha < 1")
        if int(self.top_k) < 1:
            raise ValueError("top_k must be >=1")


class RedAgent:
    """
    Corrected full Red implementation.

    This class:
      - evaluates every transform over a CONTROL SET;
      - computes the real attack-success rate D_success AND I_success;
      - ranks transforms by EMA of that attack-success rate;
      - retains q=V_D+V_I only for diagnostics;
      - accumulates selected Top-K conditions monotonically into A_e;
      - feeds the full active constraint set back to Blue.
    """

    def __init__(
        self,
        *,
        evaluator,
        transforms: Iterable[RedTransformSpec],
        config: RedAgentConfig,
    ):
        self.evaluator = evaluator
        self.transforms = tuple(transforms)
        self.config = config

        if not self.transforms:
            raise ValueError("Red transform pool is empty.")
        names = [x.name for x in self.transforms]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate Red transform names.")
        if any(x.family == "resize" for x in self.transforms):
            raise ValueError("Resize is not part of the current Red method.")

        self._order = {x.name: idx for idx, x in enumerate(self.transforms)}
        self._ema: dict[str, float] = {}
        self._seen: set[str] = set()
        self._active_ordered: list[str] = []
        self._epoch = 0

    @property
    def active_constraints(self) -> tuple[str, ...]:
        return tuple(self._active_ordered)

    def _update_ema(self, name, attack_rate):
        if name not in self._seen:
            ema = float(attack_rate)
            self._seen.add(name)
        else:
            a = float(self.config.ema_alpha)
            ema = (
                a * self._ema[name]
                + (1.0 - a) * float(attack_rate)
            )
        self._ema[name] = ema
        return ema

    def search(
        self,
        *,
        control_samples: list[tuple[str, torch.Tensor]],
    ) -> RedDecision:
        if not control_samples:
            raise ValueError("Red requires a non-empty control set.")

        risks = []
        for transform in self.transforms:
            q_values = []
            breach = 0

            for sample_id, protected in control_samples:
                transformed = apply_red_transform(protected, transform)
                obs = self.evaluator.evaluate(
                    sample_id=sample_id,
                    transform=transform,
                    transformed_protected=transformed,
                )
                q_values.append(obs.q)
                breach += int(obs.attack_success)

            # q_mean is diagnostic only. Red selection is driven by
            # the actual attack-success rate:
            # D_success AND I_success.
            q_mean = sum(q_values) / len(q_values)
            attack_rate = breach / len(q_values)
            ema = self._update_ema(
                transform.name,
                attack_rate,
            )

            risks.append(
                TransformRisk(
                    transform=transform,
                    q_mean=float(q_mean),
                    ema_risk=float(ema),
                    breach_rate=float(attack_rate),
                    num_samples=len(q_values),
                )
            )

        ranked = sorted(
            risks,
            key=lambda x: (
                -x.ema_risk,
                self._order[x.transform.name],
            ),
        )
        # Only a strictly positive EMA attack-success rate represents
        # an observed propagation breach. If all risks are zero, Red adds
        # no arbitrary constraint and the outer loop can stop as stable.
        positive_risks = [
            x for x in ranked
            if x.ema_risk > 0.0
        ]

        selected = tuple(
            x.transform.name
            for x in positive_risks[: self.config.top_k]
        )

        # Method Eq.(30): monotonic accumulation A_e = A_{e-1} U TopK.
        for name in selected:
            if name not in self._active_ordered:
                self._active_ordered.append(name)

        decision = RedDecision(
            epoch=self._epoch,
            selected_topk=selected,
            active_constraints=self.active_constraints,
            risks=tuple(risks),
            reason=(
                "Control-set worst-propagation search using the mean "
                "attack-success indicator D_success AND I_success, "
                "EMA attack-success risk, and monotonically accumulated "
                "Top-K active constraints. q=V_D+V_I is diagnostic only."
            ),
        )
        self._epoch += 1
        return decision

    def state_dict(self):
        return {
            "risk_definition": "attack_success_rate_v1",
            "ema_alpha": float(self.config.ema_alpha),
            "top_k": int(self.config.top_k),
            "transform_names": [x.name for x in self.transforms],
            "ema": deepcopy(self._ema),
            "seen": sorted(self._seen),
            "active_ordered": list(self._active_ordered),
            "epoch": int(self._epoch),
        }

    def load_state_dict(self, state):
        if state.get("risk_definition") != "attack_success_rate_v1":
            raise ValueError(
                "Red state uses an incompatible legacy risk definition. "
                "Start the corrected Red run in a new output directory."
            )
        if [x.name for x in self.transforms] != list(state["transform_names"]):
            raise ValueError("Red transform-pool mismatch.")
        if abs(float(state["ema_alpha"]) - float(self.config.ema_alpha)) > 1e-12:
            raise ValueError("Red EMA-alpha mismatch.")
        if int(state["top_k"]) != int(self.config.top_k):
            raise ValueError("Red Top-K mismatch.")

        self._ema = {str(k): float(v) for k, v in state["ema"].items()}
        self._seen = set(map(str, state["seen"]))
        self._active_ordered = list(map(str, state["active_ordered"]))
        self._epoch = int(state["epoch"])
