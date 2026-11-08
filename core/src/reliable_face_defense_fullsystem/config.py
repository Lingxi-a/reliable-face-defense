from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


def _require(mapping: dict, key: str):
    if key not in mapping:
        raise KeyError(f"Missing required config key: {key}")
    return mapping[key]


@dataclass(frozen=True)
class SplitConfig:
    train: int
    calibration: int
    control: int
    test: int


@dataclass(frozen=True)
class NullSwapConfig:
    checkpoint: str
    train_if_missing: bool
    epochs: int
    batch_size: int
    base_channels: int
    disc_base_channels: int
    disc_blocks: int
    num_workers: int


@dataclass(frozen=True)
class DetectionCalibrationConfig:
    eta_anchor_iou: float
    lse_alpha: float
    target_clean_tpr: float


@dataclass(frozen=True)
class IdentityCalibrationConfig:
    far: float
    negative_pairs: int


@dataclass(frozen=True)
class BlueConfig:
    epsilon_grid_px: tuple[float, ...]
    distortion_metric: str


@dataclass(frozen=True)
class RPPConfigValues:
    steps: int
    eta_visual: float
    rho_repair: float
    xi: float
    k_repair: int
    lambda_pix: float
    lambda_lpips: float
    accept_only_if_visual_improves: bool


@dataclass(frozen=True)
class RedConfig:
    pool: str
    ema_alpha: float
    top_k: int
    max_rounds: int
    stop_when_no_new_constraint: bool


@dataclass(frozen=True)
class FaceSwapConfig:
    enabled: bool
    model: str
    repo_root: str
    checkpoints_dir: str
    arcface_path: str
    detector_model_path: str
    detector_fallbacks: tuple[str, ...]
    crop_size: int
    det_thresh: float
    targets_per_source: int
    strict_engineering_errors: bool


@dataclass(frozen=True)
class DeploymentConfig:
    visual_gate_enabled: bool
    min_psnr_db: float | None
    max_lpips: float | None


@dataclass(frozen=True)
class FullSystemConfig:
    deps_root: str
    device: str
    output_dir: str
    seed: int
    image_size: int
    split: SplitConfig
    nullswap: NullSwapConfig
    detection_calibration: DetectionCalibrationConfig
    identity_calibration: IdentityCalibrationConfig
    blue: BlueConfig
    rpp: RPPConfigValues
    deployment: DeploymentConfig
    red: RedConfig
    faceswap: FaceSwapConfig

    @property
    def dataset_dir(self) -> Path:
        return (
            Path(self.deps_root)
            / "dataset"
            / "CelebA-HQ"
            / "CelebA-HQ-img"
        )


def load_config(path: str | Path) -> FullSystemConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    split = _require(raw, "split")
    ns = _require(raw, "nullswap")
    rel = _require(raw, "reliability")
    det = _require(rel, "detection")
    ident = _require(rel, "identity")
    blue = _require(raw, "blue")
    rpp = _require(raw, "rpp")
    deployment = raw.get("deployment", {})
    red = _require(raw, "red")
    fs = _require(raw, "faceswap")

    cfg = FullSystemConfig(
        deps_root=str(_require(raw, "deps_root")),
        device=str(_require(raw, "device")),
        output_dir=str(_require(raw, "output_dir")),
        seed=int(_require(raw, "seed")),
        image_size=int(_require(raw, "image_size")),
        split=SplitConfig(
            train=int(_require(split, "train")),
            calibration=int(_require(split, "calibration")),
            control=int(_require(split, "control")),
            test=int(_require(split, "test")),
        ),
        nullswap=NullSwapConfig(
            checkpoint=str(_require(ns, "checkpoint")),
            train_if_missing=bool(_require(ns, "train_if_missing")),
            epochs=int(_require(ns, "epochs")),
            batch_size=int(_require(ns, "batch_size")),
            base_channels=int(_require(ns, "base_channels")),
            disc_base_channels=int(_require(ns, "disc_base_channels")),
            disc_blocks=int(_require(ns, "disc_blocks")),
            num_workers=int(_require(ns, "num_workers")),
        ),
        detection_calibration=DetectionCalibrationConfig(
            eta_anchor_iou=float(_require(det, "eta_anchor_iou")),
            lse_alpha=float(_require(det, "lse_alpha")),
            target_clean_tpr=float(_require(det, "target_clean_tpr")),
        ),
        identity_calibration=IdentityCalibrationConfig(
            far=float(_require(ident, "far")),
            negative_pairs=int(_require(ident, "negative_pairs")),
        ),
        blue=BlueConfig(
            epsilon_grid_px=tuple(float(x) for x in _require(blue, "epsilon_grid_px")),
            distortion_metric=str(_require(blue, "distortion_metric")),
        ),
        rpp=RPPConfigValues(
            steps=int(_require(rpp, "steps")),
            eta_visual=float(_require(rpp, "eta_visual")),
            rho_repair=float(_require(rpp, "rho_repair")),
            xi=float(_require(rpp, "xi")),
            k_repair=int(_require(rpp, "k_repair")),
            lambda_pix=float(_require(rpp, "lambda_pix")),
            lambda_lpips=float(_require(rpp, "lambda_lpips")),
            accept_only_if_visual_improves=bool(
                _require(rpp, "accept_only_if_visual_improves")
            ),
        ),
        deployment=DeploymentConfig(
            visual_gate_enabled=bool(
                deployment.get("visual_gate_enabled", False)
            ),
            min_psnr_db=(
                None
                if deployment.get("min_psnr_db") is None
                else float(deployment["min_psnr_db"])
            ),
            max_lpips=(
                None
                if deployment.get("max_lpips") is None
                else float(deployment["max_lpips"])
            ),
        ),
        red=RedConfig(
            pool=str(_require(red, "pool")),
            ema_alpha=float(_require(red, "ema_alpha")),
            top_k=int(_require(red, "top_k")),
            max_rounds=int(_require(red, "max_rounds")),
            stop_when_no_new_constraint=bool(
                _require(red, "stop_when_no_new_constraint")
            ),
        ),
        faceswap=FaceSwapConfig(
            enabled=bool(_require(fs, "enabled")),
            model=str(_require(fs, "model")).lower(),
            repo_root=str(_require(fs, "repo_root")),
            checkpoints_dir=str(_require(fs, "checkpoints_dir")),
            arcface_path=str(_require(fs, "arcface_path")),
            detector_model_path=str(_require(fs, "detector_model_path")),
            detector_fallbacks=tuple(
                str(x) for x in _require(fs, "detector_fallbacks")
            ),
            crop_size=int(_require(fs, "crop_size")),
            det_thresh=float(_require(fs, "det_thresh")),
            targets_per_source=int(_require(fs, "targets_per_source")),
            strict_engineering_errors=bool(
                _require(fs, "strict_engineering_errors")
            ),
        ),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: FullSystemConfig) -> None:
    if cfg.image_size != 256:
        raise ValueError(
            "This integrated NullSwap pipeline is fixed at 256x256. "
            "Use image_size=256."
        )
    if cfg.device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")

    counts = [
        cfg.split.train,
        cfg.split.calibration,
        cfg.split.control,
        cfg.split.test,
    ]
    if any(x < 1 for x in counts):
        raise ValueError("Every dataset split must contain at least one image.")

    grid = cfg.blue.epsilon_grid_px
    if not grid:
        raise ValueError("Blue epsilon_grid_px is empty.")
    if tuple(sorted(set(grid))) != grid:
        raise ValueError("epsilon_grid_px must be sorted ascending and unique.")
    if min(grid) < 4 or max(grid) > 8:
        raise ValueError(
            "Current method validation interval is 4<=eps<=8 pixel units, "
            "and the audited FacePoison outer bound is eps<=8."
        )

    if cfg.blue.distortion_metric != "lpips":
        raise ValueError(
            "The integrated runner currently uses one common LPIPS metric "
            "for route comparison; set distortion_metric='lpips'."
        )

    if not (0 < cfg.detection_calibration.eta_anchor_iou <= 1):
        raise ValueError("eta_anchor_iou must be in (0,1].")
    if cfg.detection_calibration.lse_alpha <= 0:
        raise ValueError("lse_alpha must be >0.")
    if not (0 < cfg.detection_calibration.target_clean_tpr <= 1):
        raise ValueError("target_clean_tpr must be in (0,1].")
    if not (0 < cfg.identity_calibration.far < 1):
        raise ValueError("identity FAR must be in (0,1).")
    if cfg.identity_calibration.negative_pairs < 1:
        raise ValueError("negative_pairs must be >=1.")

    if cfg.nullswap.base_channels != 64:
        raise ValueError(
            "Formal integrated code uses the full NullSwap generator width "
            "base_channels=64. Do not shrink architecture for GPU validation."
        )
    if cfg.nullswap.disc_base_channels != 32:
        raise ValueError(
            "Formal integrated code uses discriminator base_channels=32."
        )
    if cfg.nullswap.disc_blocks != 5:
        raise ValueError(
            "Current explicit engineering discriminator uses 5 ConvBlocks."
        )

    if cfg.deployment.visual_gate_enabled:
        if cfg.deployment.min_psnr_db is None:
            raise ValueError(
                "deployment.min_psnr_db is required when "
                "visual_gate_enabled=true."
            )
        if cfg.deployment.max_lpips is None:
            raise ValueError(
                "deployment.max_lpips is required when "
                "visual_gate_enabled=true."
            )
        if cfg.deployment.min_psnr_db <= 0:
            raise ValueError(
                "deployment.min_psnr_db must be >0."
            )
        if cfg.deployment.max_lpips < 0:
            raise ValueError(
                "deployment.max_lpips must be >=0."
            )

    if cfg.red.pool not in {"validation", "full"}:
        raise ValueError("red.pool must be 'validation' or 'full'.")
    if cfg.red.top_k < 1:
        raise ValueError("red.top_k must be >=1.")
    if not (0 <= cfg.red.ema_alpha < 1):
        raise ValueError("red.ema_alpha must satisfy 0<=alpha<1.")
    if cfg.red.max_rounds < 1:
        raise ValueError("red.max_rounds must be >=1.")

    if cfg.rpp.steps < 0:
        raise ValueError("rpp.steps must be >=0.")
    if cfg.rpp.eta_visual <= 0:
        raise ValueError("rpp.eta_visual must be >0.")
    if not (0 < cfg.rpp.rho_repair <= 1):
        raise ValueError("rpp.rho_repair must be in (0,1].")
    if cfg.rpp.xi <= 0:
        raise ValueError("rpp.xi must be >0.")
    if cfg.rpp.k_repair < 1:
        raise ValueError("rpp.k_repair must be >=1.")
    if cfg.rpp.lambda_pix < 0 or cfg.rpp.lambda_lpips < 0:
        raise ValueError("RPP visual weights must be nonnegative.")
    if cfg.rpp.lambda_pix == 0 and cfg.rpp.lambda_lpips == 0:
        raise ValueError("At least one RPP visual term must be active.")

    if cfg.faceswap.enabled:
        if cfg.faceswap.model != "simswap":
            raise ValueError(
                "v0.5.4 integrates the audited real SimSwap runner only. "
                "Do not silently substitute another face-swap model."
            )
        if cfg.faceswap.crop_size != 224:
            raise ValueError(
                "Current audited SimSwap `people` checkpoint uses crop_size=224."
            )
        if not (0 < cfg.faceswap.det_thresh < 1):
            raise ValueError("faceswap.det_thresh must be in (0,1).")
        if cfg.faceswap.targets_per_source < 1:
            raise ValueError("faceswap.targets_per_source must be >=1.")
        if cfg.split.test < 2:
            raise ValueError(
                "Face-swap evaluation requires at least two test images so a "
                "source is never paired with itself."
            )
        if cfg.faceswap.targets_per_source > cfg.split.test - 1:
            raise ValueError(
                "targets_per_source cannot exceed test_count-1 when targets "
                "are drawn deterministically from the frozen test split."
            )
