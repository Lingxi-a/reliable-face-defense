from __future__ import annotations

from pathlib import Path
import json
import hashlib
import torch
from torch.utils.data import DataLoader

from .data import ImagePathDataset
from .detection_risk import RetinaFaceLockedRisk
from .identity_risk import IdentityRiskSystem, calibrate_far_thresholds
from .reliability import detection_threshold_from_clean_risks



def _paths_sha256(paths):
    h=hashlib.sha256()
    for p in paths:
        h.update(str(p).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


@torch.no_grad()
def calibrate(
    *,
    deps_root,
    paths,
    device,
    eta_anchor_iou,
    lse_alpha,
    target_clean_tpr,
    identity_far,
    negative_pairs,
    seed,
    batch_size=8,
):
    detector = RetinaFaceLockedRisk(
        deps_root=deps_root,
        device=device,
        eta_anchor_iou=eta_anchor_iou,
        lse_alpha=lse_alpha,
    )
    identity = IdentityRiskSystem(
        deps_root=deps_root,
        device=device,
    )

    detection_risks = []
    all_arc = []
    all_face = []

    ds = ImagePathDataset(paths,size=256)
    loader = DataLoader(ds,batch_size=int(batch_size),shuffle=False,num_workers=0)

    # Detector contexts are sample-specific, so calibrate individually.
    for idx, p in enumerate(paths):
        image = ds[idx]["image"].unsqueeze(0).to(device)
        ctx = detector.prepare_context(image)
        detection_risks.append(ctx.clean_risk)

    # Identity embeddings can be batched.
    for batch in loader:
        emb = identity.embed_batch(batch["image"])
        all_arc.append(emb["arcface"])
        all_face.append(emb["facenet"])

    embeddings = {
        "arcface":torch.cat(all_arc,dim=0),
        "facenet":torch.cat(all_face,dim=0),
    }
    id_cal = calibrate_far_thresholds(
        embeddings,
        far=identity_far,
        negative_pairs=negative_pairs,
        seed=seed,
    )
    d_thr = detection_threshold_from_clean_risks(
        detection_risks,
        target_clean_tpr=target_clean_tpr,
    )

    return {
        "_meta": {
            "calibration_paths_sha256": _paths_sha256(paths),
            "seed": int(seed),
            "eta_anchor_iou": float(eta_anchor_iou),
            "lse_alpha": float(lse_alpha),
            "target_clean_tpr": float(target_clean_tpr),
            "identity_far": float(identity_far),
            "negative_pairs": int(negative_pairs),
        },
        "detection": {
            "threshold":d_thr,
            "target_clean_tpr":float(target_clean_tpr),
            "eta_anchor_iou":float(eta_anchor_iou),
            "lse_alpha":float(lse_alpha),
            "num_images":len(paths),
            "clean_risk_min":min(detection_risks),
            "clean_risk_max":max(detection_risks),
            "clean_risk_mean":sum(detection_risks)/len(detection_risks),
            "provenance":"clean_calibration_quantile",
        },
        "identity":id_cal,
    }


def save(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2),encoding="utf-8")
