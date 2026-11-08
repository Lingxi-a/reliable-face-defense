from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import numpy as np
import torch

from .detection_risk import DetectionContext, RetinaFaceLockedRisk
from .identity_risk import IdentityContext, IdentityRiskSystem


@dataclass(frozen=True)
class ReliabilityContext:
    sample_id: str
    detection: DetectionContext
    identity: IdentityContext


@dataclass(frozen=True)
class ReliabilityEvaluation:
    detection_risk: float
    identity_similarities: dict[str,float]
    v_detection: float
    v_identity: float
    detection_success: bool
    identity_success: bool
    attack_success: bool
    defense_success: bool

    @property
    def detection_protected(self):
        return not self.detection_success

    @property
    def identity_protected(self):
        return not self.identity_success

    def to_dict(self):
        return {
            "detection_risk": self.detection_risk,
            "identity_similarities": self.identity_similarities,
            "v_detection": self.v_detection,
            "v_identity": self.v_identity,
            "detection_success": self.detection_success,
            "identity_success": self.identity_success,
            "attack_success": self.attack_success,
            "defense_success": self.defense_success,
        }


class RealReliabilitySystem:
    """
    Unified real detection + identity evaluator.

    Detection:
      locked-anchor RetinaFace recovery risk.

    Identity:
      ArcFace + FaceNet cosine similarity to fixed clean references.

    Branch logic:
      detection_success = r_D > tau_D
      identity_success  = ANY identity model > its tau_I
      attack_success    = detection_success AND identity_success
      defense_success   = detection_fail OR identity_fail
    """
    def __init__(
        self,
        *,
        detector: RetinaFaceLockedRisk,
        identity: IdentityRiskSystem,
        detection_threshold: float,
        identity_thresholds: dict[str,float],
    ):
        self.detector = detector
        self.identity = identity
        self.detection_threshold = float(detection_threshold)
        required = {"arcface","facenet"}
        if set(identity_thresholds) != required:
            raise ValueError(
                f"identity_thresholds must be exactly {sorted(required)}"
            )
        self.identity_thresholds = {
            k: float(v) for k,v in identity_thresholds.items()
        }

    def prepare_context(self, sample_id, clean01):
        return ReliabilityContext(
            sample_id=str(sample_id),
            detection=self.detector.prepare_context(clean01),
            identity=self.identity.prepare_context(clean01),
        )

    def differentiable_values(self, image01, context: ReliabilityContext):
        d = self.detector.risk(image01, context.detection)
        sims = self.identity.similarities(image01, context.identity)
        d_g = d - self.detection_threshold
        i_g = {
            name: sim - self.identity_thresholds[name]
            for name, sim in sims.items()
        }
        return d_g, i_g, d, sims

    def evaluate(self, image01, context: ReliabilityContext):
        with torch.no_grad():
            d_g, i_g, d, sims = self.differentiable_values(image01, context)
            d_success = bool(float(d_g.detach().cpu()) > 0.0)
            i_success = any(
                float(v.detach().cpu()) > 0.0
                for v in i_g.values()
            )
            attack = d_success and i_success
            return ReliabilityEvaluation(
                detection_risk=float(d.detach().cpu()),
                identity_similarities={
                    k: float(v.detach().cpu()) for k,v in sims.items()
                },
                v_detection=max(float(d_g.detach().cpu()), 0.0),
                v_identity=max(
                    [max(float(v.detach().cpu()),0.0) for v in i_g.values()]
                ),
                detection_success=d_success,
                identity_success=i_success,
                attack_success=attack,
                defense_success=not attack,
            )

    def constraint_values(self, image01, context: ReliabilityContext):
        d_g, i_g, _, _ = self.differentiable_values(image01, context)
        return {
            "detection": {"retinaface_locked": d_g},
            "identity": i_g,
        }


def save_calibration(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_calibration(path):
    x = json.loads(Path(path).read_text(encoding="utf-8"))
    return x


def detection_threshold_from_clean_risks(
    risks,
    *,
    target_clean_tpr,
):
    risks = np.asarray(risks, dtype=np.float64)
    if risks.size < 2:
        raise ValueError("Need >=2 clean detection risks for calibration.")
    q = max(0.0, min(1.0, 1.0-float(target_clean_tpr)))
    return float(np.quantile(risks, q))
