from __future__ import annotations

from contextlib import contextmanager
import importlib
import importlib.machinery
import inspect
from pathlib import Path
from typing import Any, Dict
import sys
import types

import numpy as np
import torch

from .retinaface_adapter import RetinaFaceMobile025Adapter


RETINA_MEAN_BGR = torch.tensor(
    [104.0, 117.0, 123.0],
    dtype=torch.float32,
).view(1, 3, 1, 1)


def _parse_csv_numbers(s: str, cast=float):
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


_FACEPOISON_TOPLEVEL_NAMES = {
    "utils",
    "detectors",
    "attacks",
    "attack_public_datasets",
}


def _module_is_under(module, root: Path) -> bool:
    try:
        f = getattr(module, "__file__", None)
        if f:
            Path(f).resolve().relative_to(root)
            return True
    except Exception:
        pass

    try:
        for p in getattr(module, "__path__", []) or []:
            try:
                Path(p).resolve().relative_to(root)
                return True
            except Exception:
                continue
    except Exception:
        pass
    return False


def _install_explicit_namespace(name: str, directory: Path) -> None:
    """
    Pin a top-level namespace package to one repository directory.

    FacePoison's `utils/` directory can be a PEP-420 namespace package
    (no __init__.py). A later regular package named `utils` can otherwise win
    import resolution even when FacePoison is first on sys.path. Installing the
    namespace explicitly in sys.modules removes that ambiguity.
    """
    directory = directory.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)

    module = types.ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(directory)]
    spec = importlib.machinery.ModuleSpec(
        name=name,
        loader=None,
        is_package=True,
    )
    spec.submodule_search_locations = [str(directory)]
    module.__spec__ = spec
    sys.modules[name] = module


@contextmanager
def facepoison_import_scope(facepoison_root: str | Path):
    """
    Import the official FacePoison repository in an isolated top-level
    namespace window.

    The real collision seen on AutoDL is:
      Pytorch_Retinaface -> top-level regular package `utils`
      FacePoison         -> top-level `utils/evaluation.py`

    FacePoison's `utils` may be a namespace package. Merely putting FacePoison
    first on sys.path is therefore insufficient: Python can still select the
    later regular RetinaFace `utils` package.

    We explicitly pin `utils` to FacePoison's own directory for the duration of
    the official import, then restore the previous interpreter state.
    """
    root = Path(facepoison_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"FacePoison root not found: {root}")
    if not (root / "utils" / "evaluation.py").is_file():
        raise FileNotFoundError(
            "Official FacePoison source is incomplete: "
            f"{root / 'utils' / 'evaluation.py'} is missing."
        )

    old_path = list(sys.path)
    displaced = {}

    for name, module in list(sys.modules.items()):
        top = name.split(".", 1)[0]
        if top not in _FACEPOISON_TOPLEVEL_NAMES:
            continue
        if _module_is_under(module, root):
            continue
        displaced[name] = module
        sys.modules.pop(name, None)

    root_s = str(root)
    sys.path[:] = [p for p in sys.path if p != root_s]
    sys.path.insert(0, root_s)
    importlib.invalidate_caches()

    # Critical v0.5.8 fix: pin FacePoison's namespace-style utils explicitly.
    _install_explicit_namespace("utils", root / "utils")

    try:
        yield
    finally:
        for name, module in list(sys.modules.items()):
            top = name.split(".", 1)[0]
            if top not in _FACEPOISON_TOPLEVEL_NAMES:
                continue
            if top == "utils" or _module_is_under(module, root):
                sys.modules.pop(name, None)

        for name, module in displaced.items():
            sys.modules[name] = module

        sys.path[:] = old_path
        importlib.invalidate_caches()


def _retina_to_official_rgb_pixels(x_ret: torch.Tensor) -> np.ndarray:
    """
    RetinaFace carrier tensor [1,3,H,W], BGR-mean-subtracted
    -> FacePoison official image domain: RGB HWC float pixels [0,255].

    The official FacePoison attack itself performs its detector transform.
    """
    if x_ret.ndim != 4 or x_ret.shape[0] != 1 or x_ret.shape[1] != 3:
        raise ValueError(
            "Official FacePoison attack is single-image; expected [1,3,H,W]."
        )
    mean = RETINA_MEAN_BGR.to(
        device=x_ret.device,
        dtype=x_ret.dtype,
    )
    bgr = x_ret + mean
    rgb = bgr[:, [2, 1, 0], :, :]
    return (
        rgb[0]
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=True)
    )


def _official_rgb_pixels_to_retina(
    rgb_hwc: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(rgb_hwc, np.ndarray):
        raise TypeError(
            f"Expected numpy array from official FacePoison, got {type(rgb_hwc)!r}"
        )
    if rgb_hwc.ndim != 3 or rgb_hwc.shape[2] != 3:
        raise ValueError(
            f"Expected official FacePoison RGB HWC output, got {rgb_hwc.shape}"
        )

    rgb = torch.from_numpy(
        np.asarray(rgb_hwc, dtype=np.float32)
    ).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    bgr = rgb[:, [2, 1, 0], :, :]
    mean = RETINA_MEAN_BGR.to(device=device, dtype=dtype)
    return bgr - mean


class FacePoisonRetinaAdapter:
    """
    Adapter around the downloaded official FacePoison implementation.

    Important:
      * the attack class is imported from the official repository;
      * the local official config is passed unchanged except for the requested
        dynamic epsilon override;
      * the official `.attack(image)` method is used when present;
      * no FacePoison objective/loss is reimplemented here.
    """

    def __init__(
        self,
        facepoison_root: str | Path,
        retina_source_root: str | Path,
        retina_weight_path: str | Path,
        device: str = "cuda",
    ) -> None:
        self.facepoison_root = Path(facepoison_root).resolve()
        self.retina_source_root = Path(retina_source_root).resolve()
        self.retina_weight_path = Path(retina_weight_path).resolve()
        self.device = torch.device(device)

        if not self.facepoison_root.exists():
            raise FileNotFoundError(
                f"FacePoison root not found: {self.facepoison_root}"
            )

        # Build the actual RetinaFace proxy network first.
        self.retina = RetinaFaceMobile025Adapter(
            self.retina_source_root,
            self.retina_weight_path,
            device=device,
        )
        self.model = self.retina.model

        with facepoison_import_scope(self.facepoison_root):
            from attack_public_datasets.adv_config_default import (
                FacePoisonPP_cfg,
            )
            from attacks.adv_attack import FacePoisonPP

            self.official_cfg = dict(FacePoisonPP_cfg["retinaface"])
            self.attack_layers = [
                x.strip()
                for x in self.official_cfg["layer"].split(",")
            ]
            self.feat_weights = _parse_csv_numbers(
                self.official_cfg["feat_weight"],
                float,
            )

            init_sig = inspect.signature(FacePoisonPP.__init__)
            init_names = list(init_sig.parameters.keys())

            # Current official repository API:
            # FacePoisonPP(net, params, device='cuda')
            if "params" in init_sig.parameters:
                self.attack_impl = FacePoisonPP(
                    self.model,
                    dict(self.official_cfg),
                    device=str(self.device),
                )
                self.api_mode = "official_attack_method"
            else:
                # Compatibility with the previously audited vendored adapter
                # shape, if an older local checkout is used.
                self.attack_impl = FacePoisonPP(
                    self.model,
                    self.attack_layers,
                    self.feat_weights,
                )
                self.api_mode = "legacy_callable_adapter"

            if hasattr(self.attack_impl, "attack"):
                self.attack_signature = str(
                    inspect.signature(self.attack_impl.attack)
                )
            else:
                self.attack_signature = str(
                    inspect.signature(self.attack_impl.__call__)
                )
            self.constructor_signature = str(init_sig)
            self.constructor_parameters = init_names

    def official_parameters(self) -> Dict[str, Any]:
        cfg = self.official_cfg
        return {
            "layer": cfg["layer"],
            "feat_weight": cfg["feat_weight"],
            "eps": float(cfg["eps"]),
            "alpha": float(cfg["alpha"]),
            "mask_p": float(cfg["mask_p"]),
            "ens": int(cfg["ens"]),
            "n_iter": int(cfg["n_iter"]),
            "lambda": float(cfg["lambda"]),
            "term_weight": cfg["term_weight"],
            "fdmean": float(cfg["fdmean"]),
            "hybrid_mode": int(cfg["hybrid_mode"]),
            "avgerage_grad": int(cfg["avgerage_grad"]),
            "flip_sign_p": float(cfg["flip_sign_p"]),
        }

    def _generate_official_attack(
        self,
        x_ret: torch.Tensor,
        overrides: Dict[str, Any] | None,
    ) -> torch.Tensor:
        rgb = _retina_to_official_rgb_pixels(x_ret)

        params = getattr(self.attack_impl, "params", None)
        if not isinstance(params, dict):
            raise TypeError(
                "Official FacePoisonPP exposes .attack but has no dict .params."
            )

        overrides = dict(overrides or {})
        unknown = sorted(set(overrides) - set(params))
        if unknown:
            raise KeyError(
                f"Unsupported FacePoison parameter override(s): {unknown}"
            )

        previous = {k: params[k] for k in overrides}
        try:
            params.update(overrides)
            adv_rgb = self.attack_impl.attack(rgb)
        finally:
            params.update(previous)

        return _official_rgb_pixels_to_retina(
            adv_rgb,
            device=x_ret.device,
            dtype=x_ret.dtype,
        )

    def _build_legacy_call_kwargs(
        self,
        overrides: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        params = self.official_parameters()
        if overrides:
            params.update(overrides)

        alias_table = {
            "eps": ["eps", "epsilon"],
            "alpha": ["alpha"],
            "mask_p": ["mask_p"],
            "ens": ["ens"],
            "n_iter": ["n_iter", "n_iters", "iters"],
            "lambda": ["lambda_", "lamb", "lmbd", "lambda"],
            "term_weight": ["term_weight"],
            "fdmean": ["fdmean"],
            "hybrid_mode": ["hybrid_mode"],
            "avgerage_grad": ["avgerage_grad", "average_grad"],
            "flip_sign_p": ["flip_sign_p"],
        }

        sig = inspect.signature(self.attack_impl.__call__)
        accepted = set(sig.parameters.keys())
        call_kwargs: Dict[str, Any] = {}

        for cfg_key, candidates in alias_table.items():
            for name in candidates:
                if name in accepted:
                    value = params[cfg_key]
                    if cfg_key == "term_weight" and isinstance(value, str):
                        value = tuple(_parse_csv_numbers(value, float))
                    call_kwargs[name] = value
                    break
        return call_kwargs

    def generate(
        self,
        x: torch.Tensor,
        overrides: Dict[str, Any] | None = None,
    ) -> torch.Tensor:
        x = x.to(self.retina.device)

        if self.api_mode == "official_attack_method":
            with torch.enable_grad():
                return self._generate_official_attack(x, overrides)

        call_kwargs = self._build_legacy_call_kwargs(overrides)
        sig = inspect.signature(self.attack_impl.__call__)
        params = list(sig.parameters.keys())
        with torch.enable_grad():
            if len(params) >= 2 and params[1] == "masks":
                adv = self.attack_impl(x, None, **call_kwargs)
            else:
                adv = self.attack_impl(x, **call_kwargs)

        if not isinstance(adv, torch.Tensor):
            raise TypeError(
                f"Expected torch.Tensor from FacePoisonPP, got {type(adv)!r}"
            )
        return adv

    def build_report(
        self,
        x: torch.Tensor,
        adv: torch.Tensor,
    ) -> Dict[str, Any]:
        delta = (adv - x).detach()
        linf = float(delta.abs().max().item())
        l2 = float(delta.pow(2).sum().sqrt().item())
        mean_abs = float(delta.abs().mean().item())
        return {
            "api_mode": self.api_mode,
            "constructor_signature": self.constructor_signature,
            "attack_signature": self.attack_signature,
            "attack_layers": self.attack_layers,
            "feat_weights": self.feat_weights,
            "official_parameters": self.official_parameters(),
            "retina_load_report": self.retina.load_report.__dict__,
            "perturbation": {
                "linf": linf,
                "l2": l2,
                "mean_abs": mean_abs,
                "eps_bound_expected": float(self.official_cfg["eps"]),
                "linf_within_bound": bool(
                    linf <= float(self.official_cfg["eps"]) + 1e-4
                ),
            },
        }
