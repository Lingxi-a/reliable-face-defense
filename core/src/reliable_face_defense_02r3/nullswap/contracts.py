from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class NullSwapPaperTrainingConfig:
    # Paper-reported DLW values.
    dlw_alpha: float = 3.0
    dlw_beta_init: float = 0.5
    dlw_beta_max: float = 2.0
    dlw_gamma_schedule: float = 0.1
    dlw_k: int = 30
    dlw_epsilon_d: float = 1e-6
    dlw_epsilon_w: float = 1e-6

    # Paper Eq. (13).
    lambda_id: float = 0.08
    lambda_mse: float = 1.8
    lambda_lpips: float = 1.2
    lambda_d: float = 0.1

    # Paper implementation details.
    generator_lr: float = 5e-4
    discriminator_lr: float = 1e-4
    epochs: int = 60
    image_size: int = 256
    batch_size: int = 256
    training_identity_models: tuple[str, str] = ("ArcFace", "FaceNet")
    lpips_backbone: str = "alex"


@dataclass(frozen=True)
class NullSwapEngineeringTrainingConfig:
    # Paper-unreported.
    dlw_epsilon_p: float = 1e-6
    dlw_t_star: int | None = None
    dlw_warmup_policy: str = "available"

    # Paper text available to this implementation reports LRs, not optimizer family.
    optimizer_name: str = "Adam"
    adam_betas: tuple[float, float] = (0.9, 0.999)

    # Paper does not explicitly report final output activation/range handling.
    protected_range_policy: str = "clamp_01"

    # Printed Eq.(10) has a sign ambiguity for the fake term.
    discriminator_objective_interpretation: str = "standard_binary_bce"
