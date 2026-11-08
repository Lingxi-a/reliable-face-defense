from __future__ import annotations
import torch
import torch.nn.functional as F
from .contracts import NullSwapPaperTrainingConfig


def cosine_identity_loss(clean_embedding, protected_embedding):
    return F.cosine_similarity(
        clean_embedding,
        protected_embedding,
        dim=1,
    ).mean()


def mse_loss(clean, protected):
    return F.mse_loss(protected, clean, reduction="mean")


def discriminator_bce_loss(real_logits, fake_logits):
    # Engineering interpretation of the paper's binary classifier objective.
    return (
        F.binary_cross_entropy_with_logits(
            real_logits, torch.ones_like(real_logits)
        )
        + F.binary_cross_entropy_with_logits(
            fake_logits, torch.zeros_like(fake_logits)
        )
    )


def generator_adversarial_loss(fake_logits):
    # Stable logits-space form of -E log D(fake).
    return F.binary_cross_entropy_with_logits(
        fake_logits,
        torch.ones_like(fake_logits),
    )


def total_loss(identity, mse, lpips_value, adversarial, paper=None):
    p = paper or NullSwapPaperTrainingConfig()
    return (
        p.lambda_id * identity
        + p.lambda_mse * mse
        + p.lambda_lpips * lpips_value
        + p.lambda_d * adversarial
    )
