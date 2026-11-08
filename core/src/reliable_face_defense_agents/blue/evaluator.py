from __future__ import annotations

from typing import Protocol, runtime_checkable, Any

import torch

from .contracts import (
    BlueState,
    Route,
    CandidateEvaluation,
)
from .reliability import make_environment_outcome


@runtime_checkable
class BlueEvaluator(Protocol):
    def evaluate(
        self,
        *,
        state: BlueState,
        route: Route,
        clean01: torch.Tensor,
        protected_artifact: Any,
        epsilon_px: float | None,
    ) -> CandidateEvaluation:
        ...


class CallableBlueEvaluator:
    """
    Complete evaluator composition.

    Required callables:
      transform_fn(name, protected_image) -> transformed image
      detection_fn(image, environment, state) -> bool or (bool, metadata)
      identity_fn(image, environment, state) -> bool or (bool, metadata)
      distortion_fn(clean, protected) -> float

    Thresholds and model-specific scoring stay inside detection_fn/identity_fn.
    Blue only consumes booleans under the frozen evaluation protocol.
    """

    def __init__(
        self,
        *,
        transform_fn,
        detection_fn,
        identity_fn,
        distortion_fn,
        distortion_metric: str,
        artifact_to_tensor=None,
    ):
        self.transform_fn = transform_fn
        self.detection_fn = detection_fn
        self.identity_fn = identity_fn
        self.distortion_fn = distortion_fn
        self.distortion_metric = str(distortion_metric)
        self.artifact_to_tensor = artifact_to_tensor or (lambda x: x)

    @staticmethod
    def _parse_result(x):
        if isinstance(x, tuple):
            return bool(x[0]), dict(x[1])
        return bool(x), {}

    def evaluate(
        self,
        *,
        state,
        route,
        clean01,
        protected_artifact,
        epsilon_px,
    ):
        protected = self.artifact_to_tensor(protected_artifact)

        environments = ("clean",) + tuple(state.active_red_constraints)
        outcomes = []

        for env in environments:
            y = protected if env == "clean" else self.transform_fn(env, protected)

            d_ok, d_meta = self._parse_result(
                self.detection_fn(y, env, state)
            )
            i_ok, i_meta = self._parse_result(
                self.identity_fn(y, env, state)
            )
            outcomes.append(
                make_environment_outcome(
                    env,
                    detection_success=d_ok,
                    identity_success=i_ok,
                    metadata={
                        "detection": d_meta,
                        "identity": i_meta,
                    },
                )
            )

        distortion = float(self.distortion_fn(clean01, protected))
        return CandidateEvaluation(
            sample_id=state.sample_id,
            route=route,
            outcomes=tuple(outcomes),
            distortion=distortion,
            distortion_metric=self.distortion_metric,
            epsilon_px=None if epsilon_px is None else float(epsilon_px),
            artifact=protected,
        )


class StaticBlueEvaluator:
    """
    Logic-test evaluator.
    """
    def __init__(self, specs: dict):
        self.specs = specs

    def evaluate(
        self,
        *,
        state,
        route,
        clean01,
        protected_artifact,
        epsilon_px,
    ):
        key = (route.value, None if epsilon_px is None else round(float(epsilon_px), 6))
        if key not in self.specs:
            raise KeyError(f"No static Blue evaluation for {key}")
        spec = self.specs[key]
        envs = ("clean",) + tuple(state.active_red_constraints)
        outcomes = []
        for env in envs:
            x = spec[env]
            outcomes.append(
                make_environment_outcome(
                    env,
                    detection_success=x["detection_success"],
                    identity_success=x["identity_success"],
                )
            )
        return CandidateEvaluation(
            sample_id=state.sample_id,
            route=route,
            outcomes=tuple(outcomes),
            distortion=float(spec["distortion"]),
            distortion_metric=str(spec["distortion_metric"]),
            epsilon_px=epsilon_px,
            artifact=protected_artifact,
        )
