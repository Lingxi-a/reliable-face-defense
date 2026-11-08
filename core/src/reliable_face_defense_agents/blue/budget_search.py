from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .contracts import BlueState, Route, CandidateEvaluation


@dataclass(frozen=True)
class BudgetSearchResult:
    predicted_epsilon_px: float | None
    tried_epsilon_px: tuple[float, ...]
    first_reliable: CandidateEvaluation | None
    evaluations: tuple[CandidateEvaluation, ...]


class MinimalSufficientBudgetSearch:
    """
    Verification/search layer for the FacePoison route.

    It does NOT invent a budget grid. The experiment supplies `epsilon_grid_px`.

    Given the budget network prediction, search order is:
      1. predicted epsilon (clamped to method interval);
      2. then supplied grid values >= prediction, ascending;
      3. epsilon_max if not already present.

    The first candidate satisfying clean + all active Red constraints is the
    sample's verified sufficient FacePoison budget for this round.

    This is the runtime reliability guard around the learned dynamic budget.
    """

    def __init__(
        self,
        *,
        epsilon_grid_px: Iterable[float],
        epsilon_min_px: float = 4.0,
        epsilon_max_px: float = 8.0,
    ):
        if epsilon_max_px < epsilon_min_px:
            raise ValueError("Bad epsilon range")
        grid = sorted({float(x) for x in epsilon_grid_px})
        if not grid:
            raise ValueError(
                "epsilon_grid_px must be supplied by the experiment; "
                "Blue will not invent search levels."
            )
        for x in grid:
            if x < epsilon_min_px - 1e-9 or x > epsilon_max_px + 1e-9:
                raise ValueError(
                    f"Budget-grid value {x} is outside [{epsilon_min_px},{epsilon_max_px}]"
                )
        self.grid = tuple(grid)
        self.epsilon_min_px = float(epsilon_min_px)
        self.epsilon_max_px = float(epsilon_max_px)

    def build_order(self, predicted_epsilon_px: float | None) -> tuple[float, ...]:
        if predicted_epsilon_px is None:
            values = list(self.grid)
        else:
            p = min(
                max(float(predicted_epsilon_px), self.epsilon_min_px),
                self.epsilon_max_px,
            )
            values = [p]
            values.extend(x for x in self.grid if x > p + 1e-9)

        if not any(abs(x - self.epsilon_max_px) < 1e-9 for x in values):
            values.append(self.epsilon_max_px)

        out = []
        for x in values:
            if not any(abs(x-y) < 1e-9 for y in out):
                out.append(x)
        return tuple(out)

    def search(
        self,
        *,
        image01: torch.Tensor,
        state: BlueState,
        provider,
        evaluator,
        predicted_epsilon_px: float | None,
    ) -> BudgetSearchResult:
        evaluations = []
        tried = []
        first = None

        for eps in self.build_order(predicted_epsilon_px):
            artifact = provider.protect(
                image01=image01,
                state=state,
                epsilon_px=eps,
            )
            ev = evaluator.evaluate(
                state=state,
                route=Route.FACEPOISON,
                clean01=image01,
                protected_artifact=artifact,
                epsilon_px=eps,
            )
            evaluations.append(ev)
            tried.append(eps)
            if ev.reliable:
                first = ev
                break

        return BudgetSearchResult(
            predicted_epsilon_px=predicted_epsilon_px,
            tried_epsilon_px=tuple(tried),
            first_reliable=first,
            evaluations=tuple(evaluations),
        )
