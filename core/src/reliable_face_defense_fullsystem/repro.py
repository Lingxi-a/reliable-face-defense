from __future__ import annotations

import hashlib
import os
import random
import numpy as np
import torch


def stable_int_seed(*parts, modulo=2**31 - 1) -> int:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x00")
    return int.from_bytes(h.digest()[:8], "little") % int(modulo)


def set_global_seed(seed: int) -> dict:
    """
    Reproducibility setup for the integrated runner.

    This fixes Python / NumPy / PyTorch RNG seeds and disables cuDNN benchmark.
    We deliberately do not force `torch.use_deterministic_algorithms(True)`
    because several third-party/legacy model operators may not have a strict
    deterministic CUDA implementation. That limitation is reported rather
    than hidden.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    return {
        "seed": seed,
        "python_hash_seed_env": os.environ.get("PYTHONHASHSEED"),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "strict_deterministic_algorithms": False,
    }
