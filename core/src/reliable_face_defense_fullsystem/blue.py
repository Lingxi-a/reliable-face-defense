from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from reliable_face_defense_agents.blue.contracts import Route
from .reliability import ReliabilityEvaluation


@dataclass(frozen=True)
class EnvironmentEval:
    environment: str
    evaluation: ReliabilityEvaluation

    def to_dict(self):
        return {
            "environment": self.environment,
            **self.evaluation.to_dict(),
        }


@dataclass(frozen=True)
class BlueCandidate:
    route: Route
    protected: torch.Tensor
    epsilon_px: float | None

    # LPIPS used by Blue for route comparison.
    distortion: float

    # Pre-RPP visual diagnostics.
    mse: float
    psnr_db: float

    outcomes: tuple[EnvironmentEval, ...]

    @property
    def security_reliable(self) -> bool:
        """
        Security reliability only.

        A candidate is security-reliable iff defense succeeds in:
            clean + every currently active Red environment.
        """
        return bool(self.outcomes) and all(
            x.evaluation.defense_success
            for x in self.outcomes
        )

    @property
    def reliable(self) -> bool:
        """
        Backward-compatible alias.

        Existing code/tests may still access candidate.reliable.
        Semantically this now means SECURITY reliability only.
        """
        return self.security_reliable

    def to_dict(self):
        return {
            "route": self.route.value,
            "epsilon_px": self.epsilon_px,

            # Preserve old field for compatibility.
            "distortion": self.distortion,
            "reliable": self.security_reliable,

            # New explicit semantics.
            "security_reliable": self.security_reliable,

            "visual_pre_rpp": {
                "mse": self.mse,
                "psnr_db": self.psnr_db,
                "lpips": self.distortion,
            },

            "outcomes": [
                x.to_dict()
                for x in self.outcomes
            ],
        }


@dataclass(frozen=True)
class BlueDecision:
    sample_id: str
    status: str
    chosen: BlueCandidate | None
    candidates: tuple[BlueCandidate, ...]
    searched_facepoison_eps: tuple[float, ...]
    active_constraints: tuple[str, ...]

    @property
    def security_feasible(self) -> bool:
        return (
            self.status == "selected"
            and self.chosen is not None
        )

    @property
    def feasible(self) -> bool:
        """
        Backward-compatible alias.

        At the Blue stage, feasible means SECURITY-feasible.
        Final deployability is checked after RPP.
        """
        return self.security_feasible

    def to_dict(self):
        return {
            "sample_id": self.sample_id,
            "status": self.status,

            # Preserve legacy name.
            "feasible": self.security_feasible,

            # New explicit name.
            "security_feasible": self.security_feasible,

            "searched_facepoison_eps":
                list(self.searched_facepoison_eps),

            "active_constraints":
                list(self.active_constraints),

            "chosen":
                None
                if self.chosen is None
                else self.chosen.to_dict(),

            "candidates": [
                x.to_dict()
                for x in self.candidates
            ],
        }


class VerifiedBlueAgent:
    """
    Reliability-first sample-level Blue.

    Actions:
        FacePoison(epsilon)
        NullSwap

    Security reliability is evaluated on:
        clean + all accumulated Red active constraints.

    Blue only selects among SECURITY-reliable candidates.

    If both routes are security-reliable:
        choose the lower-LPIPS candidate.

    Visual deployability is deliberately NOT enforced here.
    RPP is allowed to refine the selected candidate first.
    """

    def __init__(
        self,
        *,
        facepoison,
        nullswap,
        reliability,
        hard_propagation,
        distortion,
        epsilon_grid_px,
    ):
        self.facepoison = facepoison
        self.nullswap = nullswap
        self.reliability = reliability
        self.hard_propagation = hard_propagation
        self.distortion = distortion
        self.eps_grid = tuple(
            float(x)
            for x in epsilon_grid_px
        )

    @staticmethod
    def _pixel_metrics(clean01, protected):
        with torch.no_grad():
            diff = protected - clean01

            mse = float(
                diff.square().mean().detach().cpu()
            )

            if mse <= 0.0:
                psnr_db = float("inf")
            else:
                psnr_db = float(
                    10.0 * math.log10(1.0 / mse)
                )

        return mse, psnr_db

    def _evaluate_candidate(
        self,
        *,
        clean01,
        protected,
        context,
        route,
        epsilon_px,
        active_constraints,
    ):
        outcomes = []

        for env in (
            "clean",
        ) + tuple(active_constraints):

            y = (
                protected
                if env == "clean"
                else self.hard_propagation.apply(
                    env,
                    protected,
                )
            )

            ev = self.reliability.evaluate(
                y,
                context,
            )

            outcomes.append(
                EnvironmentEval(
                    env,
                    ev,
                )
            )

        mse, psnr_db = self._pixel_metrics(
            clean01,
            protected,
        )

        lpips_value = float(
            self.distortion(
                clean01,
                protected,
            )
        )

        return BlueCandidate(
            route=route,
            protected=protected.detach(),
            epsilon_px=epsilon_px,
            distortion=lpips_value,
            mse=mse,
            psnr_db=psnr_db,
            outcomes=tuple(outcomes),
        )

    def act(
        self,
        *,
        sample_id,
        clean01,
        context,
        active_constraints=(),
    ):
        candidates = []
        searched = []
        fp_reliable = None

        # FacePoison:
        # verified ascending budget search.
        for eps in self.eps_grid:
            protected = self.facepoison.protect(
                image01=clean01,
                epsilon_px=eps,
                sample_id=sample_id,
            )

            c = self._evaluate_candidate(
                clean01=clean01,
                protected=protected,
                context=context,
                route=Route.FACEPOISON,
                epsilon_px=eps,
                active_constraints=active_constraints,
            )

            candidates.append(c)
            searched.append(eps)

            if c.security_reliable:
                fp_reliable = c
                break

        # NullSwap:
        # one fixed trained generator action.
        ns_protected = self.nullswap.protect(
            image01=clean01,
            sample_id=sample_id,
        )

        ns = self._evaluate_candidate(
            clean01=clean01,
            protected=ns_protected,
            context=context,
            route=Route.NULLSWAP,
            epsilon_px=None,
            active_constraints=active_constraints,
        )

        candidates.append(ns)

        security_reliable_candidates = []

        if fp_reliable is not None:
            security_reliable_candidates.append(
                fp_reliable
            )

        if ns.security_reliable:
            security_reliable_candidates.append(
                ns
            )

        if not security_reliable_candidates:
            return BlueDecision(
                sample_id=str(sample_id),
                status="infeasible",
                chosen=None,
                candidates=tuple(candidates),
                searched_facepoison_eps=tuple(searched),
                active_constraints=tuple(
                    active_constraints
                ),
            )

        # Reliability first.
        # LPIPS only breaks the tie between reliable routes.
        chosen = min(
            security_reliable_candidates,
            key=lambda c: (
                c.distortion,
                c.route.value,
            ),
        )

        return BlueDecision(
            sample_id=str(sample_id),
            status="selected",
            chosen=chosen,
            candidates=tuple(candidates),
            searched_facepoison_eps=tuple(searched),
            active_constraints=tuple(
                active_constraints
            ),
        )
