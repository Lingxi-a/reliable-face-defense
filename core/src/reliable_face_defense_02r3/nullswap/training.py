from __future__ import annotations
from dataclasses import dataclass, asdict
import torch

from .contracts import NullSwapPaperTrainingConfig, NullSwapEngineeringTrainingConfig
from .dlw import DynamicLossWeighting
from .losses import (
    cosine_identity_loss,
    mse_loss,
    discriminator_bce_loss,
    generator_adversarial_loss,
    total_loss,
)
from .range_policy import apply_protected_range_policy


class NullSwapTrainer:
    """
    Full fidelity-revision training wiring.

    Recognition models are frozen but protected-image forwards are NOT wrapped
    in no_grad, so gradients can flow from identity losses into the generator.
    """

    def __init__(
        self,
        *,
        generator,
        discriminator,
        arcface,
        facenet,
        lpips_model,
        device="cpu",
        paper=None,
        engineering=None,
    ):
        self.device = torch.device(device)
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.arcface = arcface.to(self.device)
        self.facenet = facenet.to(self.device)
        self.lpips_model = lpips_model.to(self.device)

        self.paper = paper or NullSwapPaperTrainingConfig()
        self.engineering = engineering or NullSwapEngineeringTrainingConfig()

        if self.engineering.optimizer_name != "Adam":
            raise ValueError("Only explicit engineering optimizer Adam is implemented.")

        self.optimizer_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.paper.generator_lr,
            betas=self.engineering.adam_betas,
        )
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.paper.discriminator_lr,
            betas=self.engineering.adam_betas,
        )

        self.dlw = DynamicLossWeighting(
            2,
            paper=self.paper,
            engineering=self.engineering,
        )

    def _protected_with_raw(self, clean):
        raw = self.generator(clean)["protected_raw"]
        protected = apply_protected_range_policy(
            raw,
            self.engineering.protected_range_policy,
        )
        return raw, protected

    def _protected(self, clean):
        return self._protected_with_raw(clean)[1]

    def one_step(self, clean01, epoch=0):
        clean01 = clean01.to(self.device)

        # Clean embeddings are constants for the generator objective.
        with torch.no_grad():
            clean_arc = self.arcface(clean01)
            clean_face = self.facenet(clean01)

        # ---- D update ----
        self.discriminator.train()
        self.generator.train()
        self.optimizer_d.zero_grad(set_to_none=True)

        raw_d, protected_d = self._protected_with_raw(clean01)
        real_logits = self.discriminator(clean01)
        fake_logits_d = self.discriminator(protected_d.detach())
        loss_d = discriminator_bce_loss(real_logits, fake_logits_d)
        loss_d.backward()
        self.optimizer_d.step()

        # ---- G update ----
        self.optimizer_g.zero_grad(set_to_none=True)
        raw_g, protected = self._protected_with_raw(clean01)

        protected_arc = self.arcface(protected)
        protected_face = self.facenet(protected)

        loss_arc = cosine_identity_loss(clean_arc, protected_arc)
        loss_face = cosine_identity_loss(clean_face, protected_face)

        dlw_result = self.dlw.compute(
            [loss_arc, loss_face],
            epoch=epoch,
            update_history=True,
        )
        loss_identity = dlw_result.weighted_loss

        loss_mse = mse_loss(clean01, protected)
        loss_lpips = self.lpips_model(clean01, protected)

        # Freeze D params for generator adversarial forward while preserving
        # gradient wrt protected input.
        d_requires = [p.requires_grad for p in self.discriminator.parameters()]
        for p in self.discriminator.parameters():
            p.requires_grad_(False)
        fake_logits_g = self.discriminator(protected)
        loss_adv = generator_adversarial_loss(fake_logits_g)

        loss_g = total_loss(
            loss_identity,
            loss_mse,
            loss_lpips,
            loss_adv,
            paper=self.paper,
        )
        loss_g.backward()
        self.optimizer_g.step()

        for p, req in zip(self.discriminator.parameters(), d_requires):
            p.requires_grad_(req)

        values = {
            "loss_d": float(loss_d.detach().cpu()),
            "loss_g_total": float(loss_g.detach().cpu()),
            "identity_arcface": float(loss_arc.detach().cpu()),
            "identity_facenet": float(loss_face.detach().cpu()),
            "identity_dlw": float(loss_identity.detach().cpu()),
            "dlw_weights": list(dlw_result.normalized_weights),
            "dlw_beta": float(dlw_result.beta_epoch),
            "mse": float(loss_mse.detach().cpu()),
            "lpips": float(loss_lpips.detach().cpu()),
            "generator_adv": float(loss_adv.detach().cpu()),
            "protected_min": float(protected.detach().min().cpu()),
            "protected_max": float(protected.detach().max().cpu()),
            # Paper does not report final output activation/range handling.
            # These diagnostics do not alter training; they expose how strongly
            # the engineering range policy is clipping the raw reconstruction.
            "protected_raw_min": float(raw_g.detach().min().cpu()),
            "protected_raw_max": float(raw_g.detach().max().cpu()),
            "raw_le_0_fraction": float((raw_g.detach() <= 0.0).float().mean().cpu()),
            "raw_ge_1_fraction": float((raw_g.detach() >= 1.0).float().mean().cpu()),
        }

        finite = all(
            torch.isfinite(torch.tensor(v)).item()
            for k, v in values.items()
            if isinstance(v, float)
        )
        return {
            "protected": protected.detach(),
            "metrics": values,
            "checks": {
                "losses_finite": bool(finite),
                "dlw_weight_sum_is_2": abs(sum(dlw_result.normalized_weights) - 2.0) < 1e-6,
                "protected_range_valid": (
                    values["protected_min"] >= 0.0
                    and values["protected_max"] <= 1.0
                ),
            },
        }
