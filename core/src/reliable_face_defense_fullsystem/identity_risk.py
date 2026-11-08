from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F

from reliable_face_defense_02r3.identity.arcface_adapter import ArcFaceR50Adapter
from reliable_face_defense_02r3.identity.facenet_adapter import FaceNetVGGFace2Adapter


@dataclass(frozen=True)
class IdentityContext:
    arcface_ref: torch.Tensor
    facenet_ref: torch.Tensor


class IdentityRiskSystem:
    def __init__(self, *, deps_root, device):
        self.device = torch.device(device)
        self.arcface = ArcFaceR50Adapter(deps_root, device=device)
        self.facenet = FaceNetVGGFace2Adapter(deps_root, device=device)

    def prepare_context(self, clean01):
        x = clean01.to(self.device)
        with torch.no_grad():
            a = self.arcface(x).detach()
            f = self.facenet(x).detach()
        return IdentityContext(a, f)

    def similarities(self, image01, context: IdentityContext):
        x = image01.to(self.device)
        a = self.arcface(x)
        f = self.facenet(x)
        return {
            "arcface": F.cosine_similarity(a, context.arcface_ref, dim=1).mean(),
            "facenet": F.cosine_similarity(f, context.facenet_ref, dim=1).mean(),
        }

    @torch.no_grad()
    def embed_batch(self, batch01):
        x = batch01.to(self.device)
        return {
            "arcface": self.arcface(x).detach().cpu(),
            "facenet": self.facenet(x).detach().cpu(),
        }


def calibrate_far_thresholds(
    embeddings: dict[str, torch.Tensor],
    *,
    far: float,
    negative_pairs: int,
    seed: int,
) -> dict:
    """
    Approximate impostor calibration without identity labels.

    Random non-self image pairs are used as impostor pairs. CelebA-HQ can
    contain repeated identities, so this is explicitly an engineering
    calibration approximation for smoke/validation.

    Publication-grade calibration should use identity labels / a dedicated
    verification set. The resulting JSON records this provenance.
    """
    names = tuple(embeddings.keys())
    if not names:
        raise ValueError("No embeddings")
    n = embeddings[names[0]].shape[0]
    if n < 2:
        raise ValueError("Need >=2 calibration images.")

    rng = random.Random(int(seed))
    pairs = []
    max_unique = n*(n-1)//2
    target = min(int(negative_pairs), max_unique)

    if max_unique <= 200000:
        all_pairs = [(i,j) for i in range(n) for j in range(i+1,n)]
        rng.shuffle(all_pairs)
        pairs = all_pairs[:target]
    else:
        seen = set()
        while len(pairs) < target:
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            if i > j:
                i,j = j,i
            if (i,j) in seen:
                continue
            seen.add((i,j))
            pairs.append((i,j))

    out = {}
    q = 1.0 - float(far)
    for name, emb in embeddings.items():
        emb = F.normalize(emb.float(), dim=1)
        scores = torch.tensor(
            [
                float((emb[i]*emb[j]).sum())
                for i,j in pairs
            ]
        ).numpy()
        threshold = float(np.quantile(scores, q))
        out[name] = {
            "threshold": threshold,
            "far": float(far),
            "num_pairs": len(pairs),
            "impostor_mean": float(scores.mean()),
            "impostor_std": float(scores.std()),
        }
    return {
        "models": out,
        "provenance": "unlabeled_random_nonself_pair_approximation",
        "warning": (
            "CelebA-HQ may contain repeated identities. Use identity-labeled "
            "calibration for publication-grade FAR thresholds."
        ),
    }
