from __future__ import annotations
from pathlib import Path
import torch


def save_checkpoint(
    path,
    *,
    generator,
    discriminator,
    optimizer_g,
    optimizer_d,
    dlw,
    epoch,
    global_step,
    extra=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "dlw": dlw.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "extra": extra or {},
    }, path)


def load_checkpoint(
    path,
    *,
    generator,
    discriminator,
    optimizer_g,
    optimizer_d,
    dlw,
    map_location="cpu",
):
    state = torch.load(path, map_location=map_location)
    generator.load_state_dict(state["generator"])
    discriminator.load_state_dict(state["discriminator"])
    optimizer_g.load_state_dict(state["optimizer_g"])
    optimizer_d.load_state_dict(state["optimizer_d"])
    dlw.load_state_dict(state["dlw"])
    return {
        "epoch": int(state["epoch"]),
        "global_step": int(state["global_step"]),
        "extra": state.get("extra", {}),
    }
