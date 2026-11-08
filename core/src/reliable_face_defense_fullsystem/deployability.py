from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VisualGateConfig:
    enabled: bool = False
    min_psnr_db: float | None = None
    max_lpips: float | None = None


@dataclass(frozen=True)
class DeployabilityEvaluation:
    security_reliable: bool
    visual_gate_calibrated: bool

    psnr_db: float
    lpips: float

    psnr_ok: bool | None
    lpips_ok: bool | None
    visual_acceptable: bool | None
    deployable: bool | None

    def to_dict(self):
        return {
            "security_reliable": self.security_reliable,
            "visual_gate_calibrated": self.visual_gate_calibrated,
            "visual": {
                "psnr_db": self.psnr_db,
                "lpips": self.lpips,
                "psnr_ok": self.psnr_ok,
                "lpips_ok": self.lpips_ok,
                "acceptable": self.visual_acceptable,
            },
            "deployable": self.deployable,
        }


class FinalDeploymentGate:
    """
    Final post-RPP deployment decision.

    Security reliability:
        clean + current active Red constraints must remain defended.

    Visual acceptability:
        evaluated only after thresholds have been calibrated.

    Until calibration is enabled, visual/deployable fields are None rather
    than inventing arbitrary visual thresholds.
    """

    def __init__(self, config: VisualGateConfig):
        self.config = config

        if self.config.enabled:
            if self.config.min_psnr_db is None:
                raise ValueError(
                    "visual gate enabled but min_psnr_db is None"
                )
            if self.config.max_lpips is None:
                raise ValueError(
                    "visual gate enabled but max_lpips is None"
                )

    def evaluate(
        self,
        *,
        security_reliable: bool,
        psnr_db: float,
        lpips: float,
    ) -> DeployabilityEvaluation:

        if not self.config.enabled:
            return DeployabilityEvaluation(
                security_reliable=bool(security_reliable),
                visual_gate_calibrated=False,
                psnr_db=float(psnr_db),
                lpips=float(lpips),
                psnr_ok=None,
                lpips_ok=None,
                visual_acceptable=None,
                deployable=None,
            )

        psnr_ok = (
            float(psnr_db)
            >= float(self.config.min_psnr_db)
        )

        lpips_ok = (
            float(lpips)
            <= float(self.config.max_lpips)
        )

        visual_ok = bool(
            psnr_ok and lpips_ok
        )

        return DeployabilityEvaluation(
            security_reliable=bool(security_reliable),
            visual_gate_calibrated=True,
            psnr_db=float(psnr_db),
            lpips=float(lpips),
            psnr_ok=bool(psnr_ok),
            lpips_ok=bool(lpips_ok),
            visual_acceptable=visual_ok,
            deployable=bool(
                security_reliable and visual_ok
            ),
        )
