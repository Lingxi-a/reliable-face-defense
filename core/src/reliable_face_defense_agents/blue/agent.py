from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .contracts import (
    BlueState,
    Route,
    CandidateEvaluation,
    BlueDecision,
)
from .budget_search import MinimalSufficientBudgetSearch


@dataclass(frozen=True)
class BlueAgentConfig:
    epsilon_min_px: float = 4.0
    epsilon_max_px: float = 8.0
    compare_distortion_metric: str = "lpips"

    def __post_init__(self):
        if self.epsilon_min_px <= 0 or self.epsilon_max_px < self.epsilon_min_px:
            raise ValueError("Invalid Blue epsilon interval.")


class BlueAgent:
    """
    Corrected complete Blue policy.

    Dynamic behavior:
      - budget network predicts sample-specific epsilon_theta(x);
      - FacePoison route is verified and, if necessary, escalated within the
        externally supplied search grid up to eps_max;
      - NullSwap remains a fixed trained-generator action;
      - clean + all Red active constraints are hard reliability requirements;
      - among reliable routes, choose lower distortion.

    Blue does not invent a third defense route or an identity/detection score
    fusion.
    """

    def __init__(
        self,
        *,
        facepoison_provider,
        nullswap_provider,
        evaluator,
        budget_network,
        budget_search: MinimalSufficientBudgetSearch,
        config: BlueAgentConfig | None = None,
    ):
        self.facepoison_provider = facepoison_provider
        self.nullswap_provider = nullswap_provider
        self.evaluator = evaluator
        self.budget_network = budget_network
        self.budget_search = budget_search
        self.config = config or BlueAgentConfig()

    def predict_epsilon_px(self, image01: torch.Tensor) -> float:
        self.budget_network.eval()
        with torch.no_grad():
            out = self.budget_network(image01)
        eps = float(out["epsilon_px"].mean().detach().cpu())
        return min(
            max(eps, self.config.epsilon_min_px),
            self.config.epsilon_max_px,
        )

    def _validate_distortion(self, candidates):
        for x in candidates:
            if x.distortion_metric != self.config.compare_distortion_metric:
                raise ValueError(
                    f"Expected common distortion metric "
                    f"{self.config.compare_distortion_metric!r}, got "
                    f"{x.distortion_metric!r}"
                )
            if not math.isfinite(x.distortion):
                raise ValueError("Non-finite distortion")

    def act(
        self,
        *,
        state: BlueState,
        image01: torch.Tensor,
    ) -> BlueDecision:
        predicted = self.predict_epsilon_px(image01)

        # 1) Dynamic FacePoison epsilon route.
        fp = self.budget_search.search(
            image01=image01,
            state=state,
            provider=self.facepoison_provider,
            evaluator=self.evaluator,
            predicted_epsilon_px=predicted,
        )

        candidates = list(fp.evaluations)
        reliable_routes = []
        if fp.first_reliable is not None:
            reliable_routes.append(fp.first_reliable)

        # 2) Fixed NullSwap route (no invented inference strength).
        ns_artifact = self.nullswap_provider.protect(
            image01=image01,
            state=state,
            epsilon_px=None,
        )
        ns_eval = self.evaluator.evaluate(
            state=state,
            route=Route.NULLSWAP,
            clean01=image01,
            protected_artifact=ns_artifact,
            epsilon_px=None,
        )
        candidates.append(ns_eval)
        if ns_eval.reliable:
            reliable_routes.append(ns_eval)

        if not reliable_routes:
            return BlueDecision(
                sample_id=state.sample_id,
                round_index=state.round_index,
                chosen=None,
                candidates=tuple(candidates),
                predicted_epsilon_px=predicted,
                searched_epsilon_px=fp.tried_epsilon_px,
                active_red_constraints=state.active_red_constraints,
                status="infeasible",
                reason=(
                    "Neither verified dynamic-budget FacePoison nor fixed "
                    "NullSwap satisfies clean + all active Red constraints "
                    "within the allowed FacePoison budget interval."
                ),
            )

        self._validate_distortion(reliable_routes)
        chosen = min(
            reliable_routes,
            key=lambda x: (
                x.distortion,
                x.route.value,
            ),
        )

        return BlueDecision(
            sample_id=state.sample_id,
            round_index=state.round_index,
            chosen=chosen,
            candidates=tuple(candidates),
            predicted_epsilon_px=predicted,
            searched_epsilon_px=fp.tried_epsilon_px,
            active_red_constraints=state.active_red_constraints,
            status="selected",
            reason=(
                "Reliability-first selection: verified clean + active-Red "
                "constraints, then chose the reliable route with minimum "
                f"{self.config.compare_distortion_metric} distortion."
            ),
        )
