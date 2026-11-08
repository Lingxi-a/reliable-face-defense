from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def quantize_png01(x: torch.Tensor) -> torch.Tensor:
    """Match the actual 8-bit PNG that is consumed by the swap runners."""
    return torch.round(x.clamp(0.0, 1.0) * 255.0) / 255.0


def project_nonincreasing_l2_direction(
    direction: torch.Tensor,
    current: torch.Tensor,
    clean: torch.Tensor,
) -> torch.Tensor:
    """
    Project a proposed update onto the first-order half-space that does not
    increase squared pixel distance to the clean source.
    """
    radial = current.detach() - clean.detach()
    flat_r = radial.flatten(1)
    flat_d = direction.flatten(1)
    dot = (flat_r * flat_d).sum(dim=1, keepdim=True)
    denom = flat_r.square().sum(dim=1, keepdim=True).clamp_min(1e-12)
    correction = torch.relu(dot) / denom
    shape = (direction.shape[0],) + (1,) * (direction.ndim - 1)
    return direction - correction.view(shape) * radial


@dataclass(frozen=True)
class VisualMetrics:
    psnr: float
    ssim: float
    lpips: float
    linf_px: float

    def to_dict(self) -> dict:
        return {
            "psnr": float(self.psnr),
            "ssim": float(self.ssim),
            "lpips": float(self.lpips),
            "linf_px": float(self.linf_px),
        }


@dataclass(frozen=True)
class RefinementConfig:
    steps: int
    step_size_px: float
    residual_budget_px: float
    snapshot_steps: tuple[int, ...]
    residual_scales: tuple[float, ...]
    toward_clean_scales: tuple[float, ...]
    environments: tuple[str, ...]
    smoothmax_temperature: float
    min_worst_similarity_gain: float
    max_condition_regression: float
    max_psnr_drop_db: float
    max_ssim_drop: float
    max_lpips_increase: float

    @classmethod
    def from_dict(cls, raw: dict) -> "RefinementConfig":
        return cls(
            steps=int(raw["steps"]),
            step_size_px=float(raw["step_size_px"]),
            residual_budget_px=float(raw["residual_budget_px"]),
            snapshot_steps=tuple(int(x) for x in raw["snapshot_steps"]),
            residual_scales=tuple(float(x) for x in raw["residual_scales"]),
            toward_clean_scales=tuple(
                float(x) for x in raw["toward_clean_scales"]
            ),
            environments=tuple(str(x) for x in raw["environments"]),
            smoothmax_temperature=float(raw["smoothmax_temperature"]),
            min_worst_similarity_gain=float(
                raw["min_worst_similarity_gain"]
            ),
            max_condition_regression=float(raw["max_condition_regression"]),
            max_psnr_drop_db=float(raw["max_psnr_drop_db"]),
            max_ssim_drop=float(raw["max_ssim_drop"]),
            max_lpips_increase=float(raw["max_lpips_increase"]),
        )

    def validate(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if self.step_size_px <= 0 or self.residual_budget_px <= 0:
            raise ValueError("step size and residual budget must be positive")
        if not self.snapshot_steps or max(self.snapshot_steps) > self.steps:
            raise ValueError("snapshot_steps must be non-empty and <= steps")
        if not self.residual_scales or not self.toward_clean_scales:
            raise ValueError("candidate scale lists must be non-empty")
        if any(not 0.0 < x <= 1.0 for x in self.residual_scales):
            raise ValueError("residual_scales must be in (0,1]")
        if any(not 0.0 <= x < 1.0 for x in self.toward_clean_scales):
            raise ValueError("toward_clean_scales must be in [0,1)")
        if "clean" not in self.environments:
            raise ValueError("environments must include clean")
        if self.smoothmax_temperature <= 0:
            raise ValueError("smoothmax_temperature must be positive")


def candidate_dominates_anchor(
    *,
    anchor_visual: VisualMetrics,
    candidate_visual: VisualMetrics,
    anchor_signature: dict[str, float],
    candidate_signature: dict[str, float],
    cfg: RefinementConfig,
) -> tuple[bool, dict]:
    if set(anchor_signature) != set(candidate_signature):
        raise ValueError("signature keys differ")

    anchor_worst = max(anchor_signature.values())
    candidate_worst = max(candidate_signature.values())
    worst_gain = anchor_worst - candidate_worst
    condition_regression = max(
        candidate_signature[key] - anchor_signature[key]
        for key in anchor_signature
    )

    checks = {
        "worst_similarity_gain": float(worst_gain),
        "max_condition_regression": float(condition_regression),
        "worst_similarity_ok": (
            worst_gain >= cfg.min_worst_similarity_gain
        ),
        "conditionwise_no_regression": (
            condition_regression <= cfg.max_condition_regression
        ),
        "psnr_ok": (
            candidate_visual.psnr
            >= anchor_visual.psnr - cfg.max_psnr_drop_db
        ),
        "ssim_ok": (
            candidate_visual.ssim
            >= anchor_visual.ssim - cfg.max_ssim_drop
        ),
        "lpips_ok": (
            candidate_visual.lpips
            <= anchor_visual.lpips + cfg.max_lpips_increase
        ),
    }
    accepted = all(
        checks[key]
        for key in (
            "worst_similarity_ok",
            "conditionwise_no_regression",
            "psnr_ok",
            "ssim_ok",
            "lpips_ok",
        )
    )
    checks["accepted"] = bool(accepted)
    return bool(accepted), checks


class RobustNullSwapRefiner:
    """
    NullSwap-anchored robust residual search.

    The train-time identity models remain ArcFace and FaceNet.  The Red
    propagation family remains Clean/JPEG/Blur/OSN.  VGGFace and SFace are not
    loaded here and therefore cannot leak into candidate selection.
    """

    def __init__(self, *, deps_root: str, device: str, config: RefinementConfig):
        from reliable_face_defense_02r3.identity.arcface_adapter import (
            ArcFaceR50Adapter,
        )
        from reliable_face_defense_02r3.identity.facenet_adapter import (
            FaceNetVGGFace2Adapter,
        )
        from reliable_face_defense_fullsystem.distortion import LPIPSDistortion
        from reliable_face_defense_rpp.propagation import (
            DifferentiableTransformRegistry,
        )

        config.validate()
        self.cfg = config
        self.device = torch.device(device)
        self.arcface = ArcFaceR50Adapter(deps_root, device=device).eval()
        self.facenet = FaceNetVGGFace2Adapter(deps_root, device=device).eval()
        self.lpips = LPIPSDistortion(device)
        self.transforms = (
            DifferentiableTransformRegistry
            .from_03bc_hard_red_pool_with_ste()
        )

        for model in (self.arcface, self.facenet):
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    @torch.no_grad()
    def _references(self, clean: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "arcface": self.arcface(clean).detach(),
            "facenet": self.facenet(clean).detach(),
        }

    def _similarity_tensors(
        self,
        image: torch.Tensor,
        references: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        transformed = []
        for environment in self.cfg.environments:
            if environment == "clean":
                transformed.append(image)
            else:
                transformed.append(self.transforms.apply(environment, image))
        batch = torch.cat(transformed, dim=0)

        values = {}
        for name, model in (
            ("arcface", self.arcface),
            ("facenet", self.facenet),
        ):
            embeddings = model(batch)
            reference = references[name].expand(embeddings.shape[0], -1)
            sims = F.cosine_similarity(embeddings, reference, dim=1)
            for index, environment in enumerate(self.cfg.environments):
                values[f"{name}::{environment}"] = sims[index]
        return values

    def _smooth_worst(
        self,
        image: torch.Tensor,
        references: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        values = torch.stack(
            list(self._similarity_tensors(image, references).values())
        )
        temperature = self.cfg.smoothmax_temperature
        return torch.logsumexp(temperature * values, dim=0) / temperature

    @torch.no_grad()
    def signature(
        self,
        image: torch.Tensor,
        references: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        values = self._similarity_tensors(image, references)
        return {
            key: float(value.detach().cpu())
            for key, value in values.items()
        }

    def visual_metrics(
        self,
        clean: torch.Tensor,
        protected: torch.Tensor,
    ) -> VisualMetrics:
        from reliable_face_defense_rpp.metrics import psnr, ssim

        with torch.no_grad():
            lpips_value = float(self.lpips(clean, protected))
            linf_px = float(
                (protected - clean).abs().max().detach().cpu()
            ) * 255.0
        return VisualMetrics(
            psnr=float(psnr(clean, protected)),
            ssim=float(ssim(clean, protected)),
            lpips=lpips_value,
            linf_px=linf_px,
        )

    def _project_residual(
        self,
        value: torch.Tensor,
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        budget = self.cfg.residual_budget_px / 255.0
        return torch.maximum(
            torch.minimum(value, anchor + budget),
            anchor - budget,
        ).clamp(0.0, 1.0)

    @staticmethod
    def _candidate_key(record: dict) -> tuple:
        visual = record["visual"]
        return (
            float(record["worst_similarity"]),
            float(visual["lpips"]),
            -float(visual["ssim"]),
            -float(visual["psnr"]),
        )

    def refine(
        self,
        *,
        clean: torch.Tensor,
        nullswap_anchor: torch.Tensor,
    ) -> dict:
        clean = clean.to(self.device)
        anchor = quantize_png01(nullswap_anchor.to(self.device))
        references = self._references(clean)
        anchor_signature = self.signature(anchor, references)
        anchor_visual = self.visual_metrics(clean, anchor)

        current = anchor.detach().clone()
        best_record = None
        accepted_count = 0
        evaluated = 0

        for step in range(1, self.cfg.steps + 1):
            variable = current.detach().clone().requires_grad_(True)
            loss = self._smooth_worst(variable, references)
            gradient = torch.autograd.grad(
                loss,
                variable,
                create_graph=False,
                retain_graph=False,
            )[0]
            direction = -gradient.detach()
            direction = project_nonincreasing_l2_direction(
                direction,
                current,
                clean,
            )
            scale = direction.abs().flatten(1).amax(dim=1).clamp_min(1e-12)
            direction = direction / scale.view(-1, 1, 1, 1)
            current = self._project_residual(
                current + (self.cfg.step_size_px / 255.0) * direction,
                anchor,
            ).detach()

            if step not in self.cfg.snapshot_steps:
                continue

            raw_delta = current - anchor
            for residual_scale in self.cfg.residual_scales:
                residual_candidate = anchor + residual_scale * raw_delta
                for toward_clean in self.cfg.toward_clean_scales:
                    candidate = (
                        (1.0 - toward_clean) * residual_candidate
                        + toward_clean * clean
                    )
                    candidate = self._project_residual(candidate, anchor)
                    candidate = quantize_png01(candidate).detach()
                    evaluated += 1

                    signature = self.signature(candidate, references)
                    visual = self.visual_metrics(clean, candidate)
                    accepted, checks = candidate_dominates_anchor(
                        anchor_visual=anchor_visual,
                        candidate_visual=visual,
                        anchor_signature=anchor_signature,
                        candidate_signature=signature,
                        cfg=self.cfg,
                    )
                    if not accepted:
                        continue
                    accepted_count += 1
                    record = {
                        "tensor": candidate,
                        "step": int(step),
                        "residual_scale": float(residual_scale),
                        "toward_clean": float(toward_clean),
                        "signature": signature,
                        "worst_similarity": max(signature.values()),
                        "visual": visual.to_dict(),
                        "checks": checks,
                    }
                    if (
                        best_record is None
                        or self._candidate_key(record)
                        < self._candidate_key(best_record)
                    ):
                        best_record = record

        if best_record is not None:
            output = best_record["tensor"].detach()
            selected = True
            selection = {
                key: value
                for key, value in best_record.items()
                if key != "tensor"
            }
        else:
            output = anchor.detach()
            selected = False
            selection = {
                "step": None,
                "residual_scale": 0.0,
                "toward_clean": 0.0,
                "signature": dict(anchor_signature),
                "worst_similarity": max(anchor_signature.values()),
                "visual": anchor_visual.to_dict(),
                "checks": {
                    "accepted": False,
                    "reason": "no candidate strictly dominated the NullSwap anchor",
                },
            }

        return {
            "protected": output,
            "selected_refinement": bool(selected),
            "anchor_signature": anchor_signature,
            "anchor_worst_similarity": max(anchor_signature.values()),
            "anchor_visual": anchor_visual.to_dict(),
            "selection": selection,
            "candidates_evaluated": int(evaluated),
            "accepted_candidate_count": int(accepted_count),
        }

