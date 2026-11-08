from __future__ import annotations
from pathlib import Path
import torch


def extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "backbone", "net"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def normalize_checkpoint_keys(state):
    out = {}
    for k, v in state.items():
        nk = k
        for prefix in ("module.", "model.", "backbone."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        out[nk] = v
    return out


def compatible_state_dict(model, raw_state):
    state = normalize_checkpoint_keys(extract_state_dict(raw_state))
    model_state = model.state_dict()
    used = {}
    extra = []
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            used[k] = v
        else:
            extra.append(k)
    missing = [k for k in model_state.keys() if k not in used]
    return used, missing, extra


def freeze_module(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
