from __future__ import annotations

from pathlib import Path
import torch

from reliable_face_defense_agents.red.presets import (
    validation_red_pool,
    full_method_red_pool,
)

from .facepoison_rgb import FacePoisonRGBProvider
from .nullswap_runtime import NullSwapRuntime
from .detection_risk import RetinaFaceLockedRisk
from .identity_risk import IdentityRiskSystem
from .reliability import RealReliabilitySystem
from .propagation import HardPropagationRegistry
from .distortion import LPIPSDistortion
from .blue import VerifiedBlueAgent
from .rpp_integration import RPPFactory
from .orchestrator import FullSystemOrchestrator
from .deployability import (
    FinalDeploymentGate,
    VisualGateConfig,
)


def build_full_system(*, cfg, calibration):
    if cfg.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    detector=RetinaFaceLockedRisk(
        deps_root=cfg.deps_root,
        device=cfg.device,
        eta_anchor_iou=cfg.detection_calibration.eta_anchor_iou,
        lse_alpha=cfg.detection_calibration.lse_alpha,
    )
    identity=IdentityRiskSystem(
        deps_root=cfg.deps_root,
        device=cfg.device,
    )
    id_thresholds={
        name:float(info["threshold"])
        for name,info in calibration["identity"]["models"].items()
    }
    reliability=RealReliabilitySystem(
        detector=detector,
        identity=identity,
        detection_threshold=float(calibration["detection"]["threshold"]),
        identity_thresholds=id_thresholds,
    )

    fp=FacePoisonRGBProvider(
        deps_root=cfg.deps_root,
        device=cfg.device,
        base_seed=cfg.seed,
    )
    ns=NullSwapRuntime(
        checkpoint=cfg.nullswap.checkpoint,
        device=cfg.device,
        base_channels=64,
        base_seed=cfg.seed,
    )
    hard=HardPropagationRegistry(cfg.red.pool)
    distortion=LPIPSDistortion(cfg.device)

    blue=VerifiedBlueAgent(
        facepoison=fp,
        nullswap=ns,
        reliability=reliability,
        hard_propagation=hard,
        distortion=distortion,
        epsilon_grid_px=cfg.blue.epsilon_grid_px,
    )
    rpp=RPPFactory(cfg=cfg.rpp,device=cfg.device)

    deployment_gate=FinalDeploymentGate(
        VisualGateConfig(
            enabled=cfg.deployment.visual_gate_enabled,
            min_psnr_db=cfg.deployment.min_psnr_db,
            max_lpips=cfg.deployment.max_lpips,
        )
    )

    transforms=(
        validation_red_pool()
        if cfg.red.pool=="validation"
        else full_method_red_pool()
    )

    return FullSystemOrchestrator(
        cfg=cfg,
        reliability=reliability,
        blue=blue,
        rpp_factory=rpp,
        deployment_gate=deployment_gate,
        hard_propagation=hard,
        red_transforms=transforms,
        output_dir=cfg.output_dir,
    )
