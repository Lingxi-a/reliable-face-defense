from __future__ import annotations

from pathlib import Path
import hashlib
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ImagePathDataset

from reliable_face_defense_02r3.nullswap.generator import (
    NullSwapGenerator,
    NullSwapEngineeringConfig,
)
from reliable_face_defense_02r3.nullswap.discriminator import (
    NullSwapDiscriminator,
    DiscriminatorEngineeringConfig,
)
from reliable_face_defense_02r3.nullswap.contracts import (
    NullSwapPaperTrainingConfig,
    NullSwapEngineeringTrainingConfig,
)
from reliable_face_defense_02r3.nullswap.lpips_adapter import LPIPSAlexNetAdapter
from reliable_face_defense_02r3.nullswap.training import NullSwapTrainer
from reliable_face_defense_02r3.nullswap.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)
from reliable_face_defense_02r3.identity.arcface_adapter import ArcFaceR50Adapter
from reliable_face_defense_02r3.identity.facenet_adapter import FaceNetVGGFace2Adapter


def _paths_sha256(paths) -> str:
    h=hashlib.sha256()
    for p in paths:
        h.update(str(p).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _seed_epoch(seed: int, epoch: int, device: str):
    v=int(seed)+int(epoch)
    random.seed(v)
    np.random.seed(v % (2**32-1))
    torch.manual_seed(v)
    if device=="cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(v)


def train_nullswap(
    *,
    deps_root,
    train_paths,
    checkpoint,
    device,
    epochs,
    batch_size,
    num_workers,
    seed,
    base_channels=64,
    disc_base_channels=32,
    disc_blocks=5,
    resume=True,
    log_path=None,
):
    """
    Complete NullSwap training stage using audited Module02R3.

    Formal architecture is NOT reduced for GPU validation.
    Only data amount / batch / epoch count may be reduced.

    Resume reproducibility:
      - epoch-specific RNG seed = seed + epoch
      - epoch-specific DataLoader generator
      - checkpoint records the exact train manifest hash / architecture / seed
    """
    if base_channels != 64:
        raise ValueError("Do not shrink formal NullSwap generator width.")
    if disc_base_channels != 32 or disc_blocks != 5:
        raise ValueError("Unexpected discriminator engineering architecture.")

    device = str(device)
    paper = NullSwapPaperTrainingConfig()
    eng = NullSwapEngineeringTrainingConfig()

    g = NullSwapGenerator(
        engineering=NullSwapEngineeringConfig(base_channels=64)
    )
    d = NullSwapDiscriminator(
        DiscriminatorEngineeringConfig(
            num_blocks=5,
            base_channels=32,
        )
    )
    arc = ArcFaceR50Adapter(deps_root,device=device)
    fac = FaceNetVGGFace2Adapter(deps_root,device=device)
    lp = LPIPSAlexNetAdapter(device=device)

    trainer = NullSwapTrainer(
        generator=g,
        discriminator=d,
        arcface=arc,
        facenet=fac,
        lpips_model=lp,
        device=device,
        paper=paper,
        engineering=eng,
    )

    train_paths=tuple(str(x) for x in train_paths)
    manifest_hash=_paths_sha256(train_paths)
    provenance={
        "train_manifest_sha256":manifest_hash,
        "seed":int(seed),
        "formal_generator_base_channels":64,
        "disc_base_channels":32,
        "disc_blocks":5,
        "image_size":256,
    }

    ckpt = Path(checkpoint)
    start_epoch = 0
    global_step = 0
    if resume and ckpt.exists():
        meta = load_checkpoint(
            ckpt,
            generator=trainer.generator,
            discriminator=trainer.discriminator,
            optimizer_g=trainer.optimizer_g,
            optimizer_d=trainer.optimizer_d,
            dlw=trainer.dlw,
            map_location=device,
        )
        extra=meta.get("extra",{})
        for k,v in provenance.items():
            if k not in extra or extra[k] != v:
                raise RuntimeError(
                    f"NullSwap resume checkpoint provenance mismatch for {k}: "
                    f"checkpoint={extra.get(k)!r}, current={v!r}"
                )
        start_epoch = meta["epoch"] + 1
        global_step = meta["global_step"]

    ds = ImagePathDataset(train_paths,size=256)

    log_file = None
    if log_path:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True,exist_ok=True)

    for epoch in range(start_epoch, int(epochs)):
        _seed_epoch(seed,epoch,device)

        loader_gen=torch.Generator()
        loader_gen.manual_seed(int(seed)+int(epoch))
        loader = DataLoader(
            ds,
            batch_size=int(batch_size),
            shuffle=True,
            generator=loader_gen,
            num_workers=int(num_workers),
            pin_memory=(device=="cuda"),
            drop_last=False,
        )

        for batch in loader:
            result = trainer.one_step(
                batch["image"],
                epoch=epoch,
            )
            global_step += 1
            rec = {
                "epoch":epoch,
                "global_step":global_step,
                "batch_size":int(batch["image"].shape[0]),
                **result["metrics"],
            }
            if log_file:
                with log_file.open("a",encoding="utf-8") as f:
                    f.write(json.dumps(rec)+"\n")

        save_checkpoint(
            ckpt,
            generator=trainer.generator,
            discriminator=trainer.discriminator,
            optimizer_g=trainer.optimizer_g,
            optimizer_d=trainer.optimizer_d,
            dlw=trainer.dlw,
            epoch=epoch,
            global_step=global_step,
            extra={
                **provenance,
                "validation_batch_size":int(batch_size),
                "requested_epochs":int(epochs),
            },
        )

    return {
        "checkpoint":str(ckpt),
        "last_epoch":max(start_epoch, int(epochs))-1,
        "global_step":global_step,
        **provenance,
    }
