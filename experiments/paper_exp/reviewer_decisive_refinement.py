from __future__ import annotations

from dataclasses import dataclass
import math
import time

import torch
import torch.nn.functional as F


def quantize_png01(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x.clamp(0.0, 1.0) * 255.0) / 255.0


def project_nonincreasing_l2_direction(
    direction: torch.Tensor,
    current: torch.Tensor,
    clean: torch.Tensor,
) -> torch.Tensor:
    radial = current.detach() - clean.detach()
    flat_r = radial.flatten(1)
    flat_d = direction.flatten(1)
    dot = (flat_r * flat_d).sum(dim=1, keepdim=True)
    denom = flat_r.square().sum(dim=1, keepdim=True).clamp_min(1e-12)
    correction = torch.relu(dot) / denom
    shape = (direction.shape[0],) + (1,) * (direction.ndim - 1)
    return direction - correction.view(shape) * radial


@dataclass(frozen=True)
class ExperimentConfig:
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
    def from_dict(cls, raw: dict) -> "ExperimentConfig":
        return cls(
            steps=int(raw["steps"]),
            step_size_px=float(raw["step_size_px"]),
            residual_budget_px=float(raw["residual_budget_px"]),
            snapshot_steps=tuple(int(x) for x in raw["snapshot_steps"]),
            residual_scales=tuple(float(x) for x in raw["residual_scales"]),
            toward_clean_scales=tuple(float(x) for x in raw["toward_clean_scales"]),
            environments=tuple(str(x) for x in raw["environments"]),
            smoothmax_temperature=float(raw["smoothmax_temperature"]),
            min_worst_similarity_gain=float(raw["min_worst_similarity_gain"]),
            max_condition_regression=float(raw["max_condition_regression"]),
            max_psnr_drop_db=float(raw["max_psnr_drop_db"]),
            max_ssim_drop=float(raw["max_ssim_drop"]),
            max_lpips_increase=float(raw["max_lpips_increase"]),
        )

    def validate(self) -> None:
        if self.steps < 1 or self.step_size_px <= 0 or self.residual_budget_px <= 0:
            raise ValueError("invalid optimization steps or residual parameters")
        if not self.snapshot_steps or max(self.snapshot_steps) > self.steps:
            raise ValueError("snapshot steps must be non-empty and <= total steps")
        if "clean" not in self.environments:
            raise ValueError("the full gate environment set must include clean")
        if self.smoothmax_temperature <= 0:
            raise ValueError("smoothmax temperature must be positive")


STRATEGIES = (
    "clean_full_gate",
    "uniform_full_gate",
    "frozen_softmax_full_gate",
    "dynamic_softmax_full_gate",
)

GATE_VARIANTS = (
    "g0_ungated",
    "g1_worst_gain",
    "g2_conditionwise",
    "g3_full",
)


class ReviewerDecisiveRefiner:
    """Generate aligned strategy and gate-ablation outputs from one frozen anchor."""

    def __init__(self, *, deps_root: str, device: str, config: ExperimentConfig):
        from reliable_face_defense_02r3.identity.arcface_adapter import ArcFaceR50Adapter
        from reliable_face_defense_02r3.identity.facenet_adapter import FaceNetVGGFace2Adapter
        from reliable_face_defense_fullsystem.distortion import LPIPSDistortion
        from reliable_face_defense_rpp.propagation import DifferentiableTransformRegistry

        config.validate()
        self.cfg = config
        self.device = torch.device(device)
        self.arcface = ArcFaceR50Adapter(deps_root, device=device).eval()
        self.facenet = FaceNetVGGFace2Adapter(deps_root, device=device).eval()
        self.lpips = LPIPSDistortion(device)
        self.transforms = DifferentiableTransformRegistry.from_03bc_hard_red_pool_with_ste()
        for model in (self.arcface, self.facenet):
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    @torch.no_grad()
    def references(self, clean: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "arcface": self.arcface(clean).detach(),
            "facenet": self.facenet(clean).detach(),
        }

    def similarity_tensors(
        self,
        image: torch.Tensor,
        references: dict[str, torch.Tensor],
        environments: tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        environments = environments or self.cfg.environments
        transformed = []
        for environment in environments:
            transformed.append(
                image if environment == "clean" else self.transforms.apply(environment, image)
            )
        batch = torch.cat(transformed, dim=0)
        values: dict[str, torch.Tensor] = {}
        for name, model in (("arcface", self.arcface), ("facenet", self.facenet)):
            embeddings = model(batch)
            reference = references[name].expand(embeddings.shape[0], -1)
            similarities = F.cosine_similarity(embeddings, reference, dim=1)
            for index, environment in enumerate(environments):
                values[f"{name}::{environment}"] = similarities[index]
        return values

    @torch.no_grad()
    def signature(
        self,
        image: torch.Tensor,
        references: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        return {
            key: float(value.detach().cpu())
            for key, value in self.similarity_tensors(image, references).items()
        }

    def visual_metrics(self, clean: torch.Tensor, protected: torch.Tensor) -> dict:
        from reliable_face_defense_rpp.metrics import psnr, ssim

        with torch.no_grad():
            lpips_value = float(self.lpips(clean, protected))
            clean_linf = float((protected - clean).abs().max().detach().cpu()) * 255.0
        return {
            "psnr": float(psnr(clean, protected)),
            "ssim": float(ssim(clean, protected)),
            "lpips": lpips_value,
            "clean_linf_px": clean_linf,
        }

    def project_residual(self, value: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        budget = self.cfg.residual_budget_px / 255.0
        return torch.maximum(
            torch.minimum(value, anchor + budget),
            anchor - budget,
        ).clamp(0.0, 1.0)

    def _loss_and_response(
        self,
        *,
        strategy: str,
        variable: torch.Tensor,
        references: dict[str, torch.Tensor],
        frozen_weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict]:
        if strategy == "clean_full_gate":
            values_dict = self.similarity_tensors(variable, references, ("clean",))
            keys = tuple(values_dict)
            values = torch.stack([values_dict[key] for key in keys])
            weights = torch.full_like(values, 1.0 / len(values))
            loss = values.mean()
        else:
            values_dict = self.similarity_tensors(variable, references)
            keys = tuple(values_dict)
            values = torch.stack([values_dict[key] for key in keys])
            if strategy == "uniform_full_gate":
                weights = torch.full_like(values, 1.0 / len(values))
                loss = values.mean()
            elif strategy == "frozen_softmax_full_gate":
                if frozen_weights is None:
                    raise RuntimeError("frozen Red weights are missing")
                weights = frozen_weights
                loss = (weights * values).sum()
            elif strategy == "dynamic_softmax_full_gate":
                temperature = self.cfg.smoothmax_temperature
                weights = torch.softmax(temperature * values.detach(), dim=0)
                loss = torch.logsumexp(temperature * values, dim=0) / temperature
            else:
                raise KeyError(strategy)
        detached = values.detach()
        weight_detached = weights.detach()
        entropy = float(
            (-(weight_detached * weight_detached.clamp_min(1e-12).log()).sum()).cpu()
        )
        return loss, {
            "keys": keys,
            "top_key": keys[int(torch.argmax(detached).item())],
            "top_risk": float(detached.max().cpu()),
            "weight_entropy": entropy,
            "effective_conditions": float(math.exp(entropy)),
            "weights": {
                key: float(weight_detached[index].cpu())
                for index, key in enumerate(keys)
            },
        }

    def _candidate_checks(
        self,
        *,
        anchor_visual: dict,
        candidate_visual: dict,
        anchor_signature: dict[str, float],
        candidate_signature: dict[str, float],
    ) -> dict:
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
            "worst_similarity_ok": worst_gain >= self.cfg.min_worst_similarity_gain,
            "conditionwise_no_regression": condition_regression <= self.cfg.max_condition_regression,
            "psnr_ok": candidate_visual["psnr"] >= anchor_visual["psnr"] - self.cfg.max_psnr_drop_db,
            "ssim_ok": candidate_visual["ssim"] >= anchor_visual["ssim"] - self.cfg.max_ssim_drop,
            "lpips_ok": candidate_visual["lpips"] <= anchor_visual["lpips"] + self.cfg.max_lpips_increase,
        }
        checks["visual_no_regression"] = bool(
            checks["psnr_ok"] and checks["ssim_ok"] and checks["lpips_ok"]
        )
        return checks

    @staticmethod
    def _candidate_key(record: dict) -> tuple:
        visual = record["visual"]
        return (
            float(record["worst_similarity"]),
            float(visual["lpips"]),
            -float(visual["ssim"]),
            -float(visual["psnr"]),
        )

    @staticmethod
    def _passes_gate(record: dict, gate: str) -> bool:
        checks = record["checks"]
        if gate == "g0_ungated":
            return True
        if gate == "g1_worst_gain":
            return bool(checks["worst_similarity_ok"])
        if gate == "g2_conditionwise":
            return bool(checks["worst_similarity_ok"] and checks["conditionwise_no_regression"])
        if gate == "g3_full":
            return bool(
                checks["worst_similarity_ok"]
                and checks["conditionwise_no_regression"]
                and checks["visual_no_regression"]
            )
        raise KeyError(gate)

    def _select(
        self,
        *,
        candidates: list[dict],
        gate: str,
        anchor: torch.Tensor,
        anchor_signature: dict[str, float],
        anchor_visual: dict,
    ) -> dict:
        accepted = [record for record in candidates if self._passes_gate(record, gate)]
        if accepted:
            best = min(accepted, key=self._candidate_key)
            return {
                "tensor": best["tensor"].detach(),
                "selected_refinement": True,
                "accepted_candidate_count": len(accepted),
                "selection": {key: value for key, value in best.items() if key != "tensor"},
            }
        return {
            "tensor": anchor.detach(),
            "selected_refinement": False,
            "accepted_candidate_count": 0,
            "selection": {
                "step": None,
                "residual_scale": 0.0,
                "toward_clean": 0.0,
                "signature": dict(anchor_signature),
                "worst_similarity": max(anchor_signature.values()),
                "visual": dict(anchor_visual),
                "checks": {"fallback": True, "reason": f"no candidate passed {gate}"},
            },
        }

    def refine_strategy(
        self,
        *,
        clean: torch.Tensor,
        nullswap_anchor: torch.Tensor,
        strategy: str,
        return_gate_variants: bool = False,
    ) -> dict:
        if strategy not in STRATEGIES:
            raise KeyError(strategy)
        started = time.perf_counter()
        clean = clean.to(self.device)
        anchor = quantize_png01(nullswap_anchor.to(self.device))
        references = self.references(clean)
        anchor_signature = self.signature(anchor, references)
        anchor_visual = self.visual_metrics(clean, anchor)

        frozen_weights = None
        if strategy == "frozen_softmax_full_gate":
            anchor_values = torch.tensor(
                list(anchor_signature.values()), device=self.device, dtype=anchor.dtype
            )
            frozen_weights = torch.softmax(
                self.cfg.smoothmax_temperature * anchor_values, dim=0
            ).detach()

        current = anchor.detach().clone()
        candidates: list[dict] = []
        response_trace: list[dict] = []
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        for step in range(1, self.cfg.steps + 1):
            variable = current.detach().clone().requires_grad_(True)
            loss, response = self._loss_and_response(
                strategy=strategy,
                variable=variable,
                references=references,
                frozen_weights=frozen_weights,
            )
            gradient = torch.autograd.grad(loss, variable, create_graph=False, retain_graph=False)[0]
            direction = project_nonincreasing_l2_direction(-gradient.detach(), current, clean)
            scale = direction.abs().flatten(1).amax(dim=1).clamp_min(1e-12)
            direction = direction / scale.view(-1, 1, 1, 1)
            current = self.project_residual(
                current + (self.cfg.step_size_px / 255.0) * direction,
                anchor,
            ).detach()
            response_trace.append({"step": step, **response})

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
                    candidate = quantize_png01(
                        self.project_residual(candidate, anchor)
                    ).detach()
                    signature = self.signature(candidate, references)
                    visual = self.visual_metrics(clean, candidate)
                    checks = self._candidate_checks(
                        anchor_visual=anchor_visual,
                        candidate_visual=visual,
                        anchor_signature=anchor_signature,
                        candidate_signature=signature,
                    )
                    candidates.append({
                        "tensor": candidate,
                        "step": int(step),
                        "residual_scale": float(residual_scale),
                        "toward_clean": float(toward_clean),
                        "signature": signature,
                        "worst_similarity": max(signature.values()),
                        "visual": visual,
                        "checks": checks,
                    })

        gates = GATE_VARIANTS if return_gate_variants else ("g3_full",)
        selections = {
            gate: self._select(
                candidates=candidates,
                gate=gate,
                anchor=anchor,
                anchor_signature=anchor_signature,
                anchor_visual=anchor_visual,
            )
            for gate in gates
        }
        top_keys = [row["top_key"] for row in response_trace]
        switches = sum(a != b for a, b in zip(top_keys, top_keys[1:]))
        peak_bytes = None
        if torch.cuda.is_available() and self.device.type == "cuda":
            peak_bytes = int(torch.cuda.max_memory_allocated(self.device))
        return {
            "strategy": strategy,
            "anchor_signature": anchor_signature,
            "anchor_visual": anchor_visual,
            "anchor_worst_similarity": max(anchor_signature.values()),
            "candidate_count": len(candidates),
            "selections": selections,
            "response_summary": {
                "top_condition_sequence": top_keys,
                "top_condition_switch_count": switches,
                "mean_weight_entropy": sum(x["weight_entropy"] for x in response_trace) / len(response_trace),
                "mean_effective_conditions": sum(x["effective_conditions"] for x in response_trace) / len(response_trace),
                "first_weights": response_trace[0]["weights"],
                "last_weights": response_trace[-1]["weights"],
            },
            "runtime_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": peak_bytes,
        }

