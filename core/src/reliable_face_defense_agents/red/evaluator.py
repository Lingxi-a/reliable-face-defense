from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from .contracts import RedTransformSpec, ViolationObservation


@runtime_checkable
class RedViolationEvaluator(Protocol):
    def evaluate(
        self,
        *,
        sample_id: str,
        transform: RedTransformSpec,
        transformed_protected: torch.Tensor,
    ) -> ViolationObservation:
        ...


class CallableRedViolationEvaluator:
    """
    Concrete composition from a real detector-risk callable and identity-risk
    callable.

    Each callable must return:
      {
        "violation": nonnegative threshold-normalized V_D or V_I,
        "success": bool indicating attacker-side recovery,
        ... optional metadata
      }

    Red never sets the thresholds itself.
    """
    def __init__(self, detection_risk_fn, identity_risk_fn):
        self.detection_risk_fn = detection_risk_fn
        self.identity_risk_fn = identity_risk_fn

    def evaluate(
        self,
        *,
        sample_id,
        transform,
        transformed_protected,
    ):
        d = self.detection_risk_fn(
            transformed_protected,
            transform.name,
            sample_id,
        )
        i = self.identity_risk_fn(
            transformed_protected,
            transform.name,
            sample_id,
        )
        vd = max(float(d["violation"]), 0.0)
        vi = max(float(i["violation"]), 0.0)
        return ViolationObservation(
            v_detection=vd,
            v_identity=vi,
            detection_success=bool(d["success"]),
            identity_success=bool(i["success"]),
            metadata={
                "detection": {
                    k: v for k, v in d.items()
                    if k not in {"violation", "success"}
                },
                "identity": {
                    k: v for k, v in i.items()
                    if k not in {"violation", "success"}
                },
            },
        )


class StaticRedViolationEvaluator:
    """
    Logic-test evaluator:
      specs[(sample_id, transform_name)] -> observation fields
    """
    def __init__(self, specs):
        self.specs = specs

    def evaluate(
        self,
        *,
        sample_id,
        transform,
        transformed_protected,
    ):
        x = self.specs[(sample_id, transform.name)]
        return ViolationObservation(
            v_detection=float(x["v_detection"]),
            v_identity=float(x["v_identity"]),
            detection_success=bool(x["detection_success"]),
            identity_success=bool(x["identity_success"]),
            metadata=dict(x.get("metadata", {})),
        )
