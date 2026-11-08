from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import inspect
import importlib
import json
import os
import random
import shutil
import sys
import tempfile
import traceback

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .data import load_rgb01
from .reporting import save_image01, json_dump, write_csv


SWAP_OK = "SWAP_OK"
SOURCE_DETECTION_FAILED = "SOURCE_DETECTION_FAILED"
TARGET_DETECTION_FAILED = "TARGET_DETECTION_FAILED"
ENGINEERING_ERROR = "ENGINEERING_ERROR"
PROTECTION_INFEASIBLE = "PROTECTION_INFEASIBLE"


@dataclass(frozen=True)
class FaceSwapPair:
    source_id: str
    source_path: str
    target_id: str
    target_path: str
    target_index: int

    @property
    def pair_id(self) -> str:
        return f"{self.source_id}__T{self.target_index:02d}_{self.target_id}"


@dataclass
class SwapResult:
    status: str
    image01: torch.Tensor | None = None
    error: str | None = None
    metadata: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.status == SWAP_OK and self.image01 is not None

    def serializable(self):
        return {
            "status": self.status,
            "error": self.error,
            "metadata": self.metadata or {},
        }


def deterministic_test_pairs(
    test_paths,
    *,
    targets_per_source: int,
    seed: int,
) -> tuple[FaceSwapPair, ...]:
    """
    Deterministic deranged source-target pairing on the frozen test split.

    No source is ever paired with itself. For K targets/source, distinct cyclic
    offsets are used. The seed only chooses the starting offset; reruns with the
    same manifest+seed are identical.
    """
    paths = tuple(str(x) for x in test_paths)
    n = len(paths)
    if n < 2:
        raise ValueError("Need at least two test images for face-swap pairing.")
    if targets_per_source < 1 or targets_per_source > n - 1:
        raise ValueError("targets_per_source must be in [1, n-1].")

    start = 1 + (int(seed) % (n - 1))
    offsets = []
    cur = start
    while len(offsets) < targets_per_source:
        if cur not in offsets:
            offsets.append(cur)
        cur += 1
        if cur >= n:
            cur = 1

    out = []
    for i, src in enumerate(paths):
        sid = Path(src).stem
        for t_idx, offset in enumerate(offsets):
            tgt = paths[(i + offset) % n]
            if tgt == src:
                raise RuntimeError("Internal pairing bug produced self-pair.")
            out.append(
                FaceSwapPair(
                    source_id=sid,
                    source_path=src,
                    target_id=Path(tgt).stem,
                    target_path=tgt,
                    target_index=t_idx,
                )
            )
    return tuple(out)


def build_faceswap_candidate_pool(
    *,
    dataset_dir: str | Path,
    excluded_paths,
    seed: int,
) -> tuple[str, ...]:
    """
    Build a deterministic held-out candidate pool for face-swap evaluation.

    Training, calibration and Red-control images are excluded. The existing
    reliability-test images remain eligible because they are also held out
    from all method fitting/control. Remaining unused dataset images provide
    extra candidates when a tiny smoke test split contains no clean
    attack-eligible source.
    """
    root = Path(dataset_dir)
    items = []
    for pat in ("*.jpg", "*.jpeg", "*.png"):
        items.extend(root.glob(pat))
    excluded = {str(Path(x).resolve()) for x in excluded_paths}
    pool = sorted(
        str(x.resolve())
        for x in set(items)
        if str(x.resolve()) not in excluded
    )
    rng = random.Random(int(seed))
    rng.shuffle(pool)
    return tuple(pool)


def select_clean_attack_eligible_pairs(
    *,
    candidate_paths,
    swapper,
    device,
    source_count: int,
    targets_per_source: int,
    strict_engineering_errors: bool,
    output_path: str | Path | None = None,
):
    """
    Deterministically select clean SimSwap-eligible source-target pairs.

    Eligibility is defined only by the clean attack: a selected pair must
    produce a real clean SimSwap output (SWAP_OK). SOURCE/TARGET detection
    failures are recorded as attack-ineligible and are never counted as
    defense success. ENGINEERING_ERROR is never an eligible outcome.

    Returns:
      selected_pairs, eligibility_records, clean_swap_cache
    """
    paths = tuple(str(x) for x in candidate_paths)
    if source_count < 1:
        raise ValueError("source_count must be >= 1")
    if targets_per_source < 1:
        raise ValueError("targets_per_source must be >= 1")
    if len(paths) < 2:
        raise ValueError("Need at least two candidate images.")

    device = torch.device(device)
    selected_pairs = []
    records = []
    clean_cache = {}
    selected_source_ids = set()

    for src_index, source_path in enumerate(paths):
        source_id = Path(source_path).stem
        if source_id in selected_source_ids:
            continue

        clean = load_rgb01(source_path, 256).unsqueeze(0).to(device)
        successes = []
        attempts = []

        target_paths = paths[src_index + 1:] + paths[:src_index]
        for target_path in target_paths:
            if target_path == source_path:
                continue

            result = swapper.swap(
                source01=clean,
                target_path=target_path,
            )
            attempts.append({
                "target_id": Path(target_path).stem,
                "target_path": str(target_path),
                "status": result.status,
                "error": result.error,
            })

            if result.status == ENGINEERING_ERROR:
                if strict_engineering_errors:
                    raise RuntimeError(
                        "Clean SimSwap engineering failure during "
                        "attack-eligibility selection:\n"
                        + (result.error or "unknown")
                    )
                continue

            if result.status == SOURCE_DETECTION_FAILED:
                break

            if result.status != SWAP_OK:
                continue

            target_index = len(successes)
            pair = FaceSwapPair(
                source_id=source_id,
                source_path=str(source_path),
                target_id=Path(target_path).stem,
                target_path=str(target_path),
                target_index=target_index,
            )
            successes.append(pair)
            clean_cache[pair.pair_id] = result
            if len(successes) >= targets_per_source:
                break

        accepted = len(successes) == targets_per_source
        records.append({
            "candidate_source_rank": src_index,
            "source_id": source_id,
            "source_path": str(source_path),
            "accepted": accepted,
            "successful_targets": len(successes),
            "required_targets": targets_per_source,
            "attempts": attempts,
        })

        if not accepted:
            for pair in successes:
                clean_cache.pop(pair.pair_id, None)
            continue

        selected_source_ids.add(source_id)
        selected_pairs.extend(successes)
        if len(selected_source_ids) >= source_count:
            break

    if len(selected_source_ids) < source_count:
        raise RuntimeError(
            "Could not find enough clean SimSwap attack-eligible held-out "
            f"sources: required={source_count}, found={len(selected_source_ids)}."
        )

    payload = {
        "selection_rule": (
            "held-out deterministic pool; select source only when every "
            "requested target pair has clean SimSwap status SWAP_OK"
        ),
        "source_count_required": int(source_count),
        "targets_per_source": int(targets_per_source),
        "selected_source_count": len(selected_source_ids),
        "selected_pair_count": len(selected_pairs),
        "selected_pairs": [
            asdict(x) | {"pair_id": x.pair_id}
            for x in selected_pairs
        ],
        "candidate_records": records,
        "identity_label_limitation": (
            "different image file is enforced; CelebA-HQ identity labels are "
            "not used here, so different file does not guarantee different identity"
        ),
    }
    if output_path is not None:
        json_dump(Path(output_path), payload)

    return tuple(selected_pairs), records, clean_cache


_SIMSWAP_TOPLEVEL_NAMES = {
    "models",
    "options",
    "util",
    "insightface_func",
    "insightface",
}


def _resolve_insightface_python_root() -> Path:
    """Locate the already-downloaded InsightFace Python package."""
    candidates = [
        Path(
            "/root/autodl-tmp/FaceDefenseDependencies/source/"
            "insightface-master/python-package"
        ),
        Path(
            "/root/autodl-tmp/FaceDefenseDependencies/source/"
            "insightface-master"
        ),
    ]
    for root in candidates:
        if (root / "insightface" / "__init__.py").is_file():
            return root.resolve()
    raise FileNotFoundError(
        "InsightFace Python package source was not found. Expected "
        "/root/autodl-tmp/FaceDefenseDependencies/source/"
        "insightface-master/python-package/insightface/__init__.py"
    )


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


@contextmanager
def _repo_import_context(
    repo_root: Path,
    *,
    extra_import_roots: tuple[Path, ...] = (),
):
    """
    Collision-safe import scope for the official SimSwap repository.

    SimSwap intentionally imports generic top-level packages such as
    `models`, `options` and `util`. Earlier Module05 stages and third-party
    dependencies may already have modules with the same names in sys.modules.
    In that case, merely putting SimSwap first on sys.path is not enough:
    Python reuses the already-loaded foreign module and imports like
    `from models.models import create_model` fail.

    This scope temporarily displaces only conflicting top-level module trees,
    imports/runs SimSwap with its repository root first, then restores the
    previous interpreter state. No SimSwap network or inference logic changes.
    """
    repo_root = repo_root.resolve()
    if not repo_root.is_dir():
        raise FileNotFoundError(f"SimSwap repository not found: {repo_root}")
    if not (repo_root / "models" / "models.py").is_file():
        raise FileNotFoundError(
            f"SimSwap source incomplete: {repo_root / 'models' / 'models.py'}"
        )

    extra_import_roots = tuple(Path(p).resolve() for p in extra_import_roots)

    old_cwd = Path.cwd()
    old_path = list(sys.path)
    displaced = {}

    allowed_roots = (repo_root, *extra_import_roots)

    for name, module in list(sys.modules.items()):
        top = name.split(".", 1)[0]
        if top not in _SIMSWAP_TOPLEVEL_NAMES:
            continue
        if any(_module_is_under(module, root) for root in allowed_roots):
            continue
        displaced[name] = module
        sys.modules.pop(name, None)

    prepend = [str(repo_root), *[str(p) for p in extra_import_roots]]
    sys.path[:] = [p for p in sys.path if p not in prepend]
    sys.path[:0] = prepend

    os.chdir(repo_root)
    importlib.invalidate_caches()

    try:
        yield
    finally:
        # Remove SimSwap's generic top-level packages before restoring the
        # process state that existed before entering the scope.
        for name, module in list(sys.modules.items()):
            top = name.split(".", 1)[0]
            if top not in _SIMSWAP_TOPLEVEL_NAMES:
                continue
            if any(_module_is_under(module, root) for root in allowed_roots):
                sys.modules.pop(name, None)

        for name, module in displaced.items():
            sys.modules[name] = module

        os.chdir(old_cwd)
        sys.path[:] = old_path
        importlib.invalidate_caches()


def _tensor01_to_bgr_u8(x01: torch.Tensor) -> np.ndarray:
    if x01.ndim != 4 or x01.shape[0] != 1 or x01.shape[1] != 3:
        raise ValueError("Expected source tensor shape [1,3,H,W].")
    arr = (
        x01[0].detach().cpu().clamp(0, 1)
        .permute(1, 2, 0).numpy()
    )
    rgb = np.rint(arr * 255.0).astype(np.uint8)
    return rgb[:, :, ::-1].copy()


def _bgr_u8_to_tensor01(arr: np.ndarray, device) -> torch.Tensor:
    if arr is None or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected BGR uint8 HWC image.")
    rgb = arr[:, :, ::-1].copy()
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)[None]
    return x.to(device)


def _first_existing(paths) -> Path | None:
    for p in paths:
        p = Path(p)
        if p.is_file():
            return p
    return None


def ensure_detector_model(cfg) -> dict:
    """
    Ensure SimSwap's `antelope` directory contains the audited SCRFD detector.

    Historical AutoDL deployment stored the same `scrfd_10g_bnkps.onnx` in a
    different project directory. We stage it only when the canonical SimSwap
    path is missing. No detector is downloaded and no alternative detector is
    silently selected.
    """
    dst = Path(cfg.detector_model_path)
    if dst.is_file():
        return {
            "detector_path": str(dst),
            "staged": False,
            "source": str(dst),
        }

    src = _first_existing(cfg.detector_fallbacks)
    if src is None:
        raise FileNotFoundError(
            "SimSwap SCRFD detector missing. Expected either canonical path "
            f"{dst} or one configured fallback."
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "detector_path": str(dst),
        "staged": True,
        "source": str(src),
    }


class SimSwapAdapter:
    """
    Real SimSwap single-face inference adapter.

    The inference sequence follows the official SimSwap whole-image single-face
    script:
      1. detect/align source;
      2. ArcFace source embedding -> L2-normalized latent ID;
      3. detect/align target;
      4. run the official SimSwap generator for each target crop;
      5. reverse the crop(s) back to the target canvas.

    SimSwap is evaluation-only here. No gradient is propagated back into Blue,
    NullSwap, RPP or Red.
    """
    def __init__(self, *, cfg, device="cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError(
                "The integrated real SimSwap evaluation is GPU-only."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for SimSwap.")

        self.repo_root = Path(cfg.repo_root)
        self.asset_report = ensure_detector_model(cfg)
        self.insightface_python_root = _resolve_insightface_python_root()

        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

        # Official SimSwap's reverse2original.py still uses the removed
        # NumPy alias `np.float`. Keep compatibility local to this adapter;
        # this does not alter any numerical operation (np.float historically
        # aliased Python float).
        if not hasattr(np, "float"):
            np.float = float  # type: ignore[attr-defined]

        with _repo_import_context(
            self.repo_root,
            extra_import_roots=(self.insightface_python_root,),
        ):
            import cv2
            from torchvision import transforms
            from models.models import create_model
            from options.test_options import TestOptions
            from insightface_func.face_detect_crop_single import Face_detect_crop
            from util.reverse2original import reverse2wholeimage
            from util.norm import SpecificNorm

            self.cv2 = cv2
            self.transforms = transforms
            self.reverse2wholeimage = reverse2wholeimage
            self.sp_norm = SpecificNorm()

            # Match official source-identity preprocessing.
            self.transformer_arcface = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ])

            # Construct options without touching sys.argv.
            options = TestOptions()
            options.initialize()
            args = [
                "--name", "people",
                "--Arc_path", str(Path(cfg.arcface_path).resolve()),
                "--checkpoints_dir", str(Path(cfg.checkpoints_dir).resolve()),
                "--crop_size", str(int(cfg.crop_size)),
                "--gpu_ids", "0",
                "--no_simswaplogo",
            ]
            opt = options.parser.parse_args(args)
            opt.isTrain = False
            opt.no_simswaplogo = True
            opt.use_mask = False
            opt.gpu_ids = [0]
            torch.cuda.set_device(0)
            torch.nn.Module.dump_patches = True

            self.model = create_model(opt)
            self.model.eval()
            self.opt = opt

            detector_root = (
                Path(cfg.detector_model_path).resolve().parent.parent
            )
            self.detector = Face_detect_crop(
                name="antelope",
                root=str(detector_root),
            )
            self.detector.prepare(
                ctx_id=0,
                det_thresh=float(cfg.det_thresh),
                det_size=(640, 640),
                mode="None",
            )
            self._patch_detect_keyword_compatibility()

    def _patch_detect_keyword_compatibility(self):
        """
        Bridge historical SimSwap/InsightFace SCRFD API drift.

        SimSwap's bundled face_detect_crop_single.py calls:
            det_model.detect(..., threshold=self.det_thresh, ...)

        SCRFD variants use one of:
          * threshold=...
          * thresh=...
          * det_thresh=...
          * no threshold kwarg, reading self.det_thresh instead

        Keep SimSwap unchanged and adapt only the detector-instance boundary.
        """
        det = getattr(self.detector, "det_model", None)
        if det is None or not hasattr(det, "detect"):
            return

        original = det.detect
        sig = inspect.signature(original)
        params = sig.parameters
        self.scrfd_detect_signature = str(sig)

        if "threshold" in params:
            self.scrfd_threshold_mode = "threshold"
            return
        if "det_thresh" in params:
            mode = "det_thresh"
        elif "thresh" in params:
            mode = "thresh"
        else:
            mode = "attribute"

        def compatible_detect(*args, **kwargs):
            if "threshold" not in kwargs:
                return original(*args, **kwargs)

            threshold = kwargs.pop("threshold")

            if mode == "det_thresh":
                kwargs.setdefault("det_thresh", threshold)
                return original(*args, **kwargs)

            if mode == "thresh":
                kwargs.setdefault("thresh", threshold)
                return original(*args, **kwargs)

            had_attr = hasattr(det, "det_thresh")
            old_value = getattr(det, "det_thresh", None)
            try:
                setattr(det, "det_thresh", float(threshold))
                return original(*args, **kwargs)
            finally:
                if had_attr:
                    setattr(det, "det_thresh", old_value)
                else:
                    try:
                        delattr(det, "det_thresh")
                    except AttributeError:
                        pass

        det.detect = compatible_detect
        self.scrfd_threshold_mode = mode

    @staticmethod
    def _totensor(rgb_u8):
        tensor = torch.from_numpy(rgb_u8)
        img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255.0)

    def _get_faces(self, bgr, crop_size):
        result = self.detector.get(bgr, crop_size)
        if result is None:
            return [], []
        crops, mats = result
        if crops is None:
            crops = []
        if mats is None:
            mats = []
        return list(crops), list(mats)

    def _swap_bgr(self, *, src_bgr: np.ndarray, target_bgr: np.ndarray, target_path: str) -> SwapResult:
        """Run the official SimSwap whole-image single-face sequence.

        This method intentionally mirrors `test_wholeimage_swapsingle.py`:
        source detection/alignment -> ArcFace latent -> target detection/alignment
        -> generator -> reverse2wholeimage.  Module05 does not substitute a
        detector, alignment rule, identity extractor, or generator.  The only
        compatibility shim retained elsewhere in this adapter maps the old
        SimSwap `threshold=` keyword to the installed SCRFD API.
        """
        source_crops, _ = self._get_faces(
            src_bgr,
            int(self.cfg.crop_size),
        )
        if not source_crops:
            return SwapResult(
                status=SOURCE_DETECTION_FAILED,
                metadata={"target_path": str(target_path)},
            )

        target_crops, target_mats = self._get_faces(
            target_bgr,
            int(self.cfg.crop_size),
        )
        if not target_crops:
            return SwapResult(
                status=TARGET_DETECTION_FAILED,
                metadata={"target_path": str(target_path)},
            )

        # Official SimSwap source identity path.
        src_pil = Image.fromarray(
            self.cv2.cvtColor(
                source_crops[0],
                self.cv2.COLOR_BGR2RGB,
            )
        )
        img_a = self.transformer_arcface(src_pil)
        img_id = img_a.view(
            -1,
            img_a.shape[0],
            img_a.shape[1],
            img_a.shape[2],
        ).to(self.device)

        with torch.no_grad():
            img_id_downsample = F.interpolate(
                img_id,
                size=(112, 112),
            )
            latent_id = self.model.netArc(img_id_downsample)
            latent_id = F.normalize(latent_id, p=2, dim=1)

            swap_result_list = []
            target_tensor_list = []
            for crop in target_crops:
                rgb = self.cv2.cvtColor(
                    crop,
                    self.cv2.COLOR_BGR2RGB,
                )
                target_tensor = (
                    self._totensor(rgb)[None, ...]
                    .to(self.device)
                )
                swap_result = self.model(
                    None,
                    target_tensor,
                    latent_id,
                    None,
                    True,
                )[0]
                swap_result_list.append(swap_result)
                target_tensor_list.append(target_tensor)

            with tempfile.TemporaryDirectory(
                prefix="module05_simswap_"
            ) as td:
                out_path = Path(td) / "swap.png"
                self.reverse2wholeimage(
                    target_tensor_list,
                    swap_result_list,
                    target_mats,
                    int(self.cfg.crop_size),
                    target_bgr,
                    None,
                    str(out_path),
                    True,  # no_simswaplogo
                    pasring_model=None,
                    use_mask=False,
                    norm=self.sp_norm,
                )
                out_bgr = self.cv2.imread(str(out_path))
                if out_bgr is None:
                    raise RuntimeError(
                        "SimSwap reverse2wholeimage did not produce "
                        "a readable output image."
                    )
                out01 = _bgr_u8_to_tensor01(
                    out_bgr,
                    self.device,
                )

        return SwapResult(
            status=SWAP_OK,
            image01=out01.detach(),
            metadata={
                "target_path": str(target_path),
                "target_size_hw": [
                    int(target_bgr.shape[0]),
                    int(target_bgr.shape[1]),
                ],
                "source_faces": len(source_crops),
                "target_faces": len(target_crops),
                "simswap_path": "official_wholeimage_single",
            },
        )

    def swap_paths(self, *, source_path: str, target_path: str) -> SwapResult:
        """Run SimSwap from image files, matching its official test script.

        `source_path` should point to the exact 256x256 clean/protected source
        image used by NullSwap evaluation.  SimSwap remains responsible for
        its own detection, alignment, 224 crop, ArcFace preprocessing and
        whole-image reconstruction.
        """
        try:
            with _repo_import_context(
                self.repo_root,
                extra_import_roots=(self.insightface_python_root,),
            ):
                src_bgr = self.cv2.imread(str(source_path))
                if src_bgr is None:
                    return SwapResult(
                        status=ENGINEERING_ERROR,
                        error=f"cv2.imread failed for source: {source_path}",
                    )
                target_bgr = self.cv2.imread(str(target_path))
                if target_bgr is None:
                    return SwapResult(
                        status=ENGINEERING_ERROR,
                        error=f"cv2.imread failed for target: {target_path}",
                    )
                return self._swap_bgr(
                    src_bgr=src_bgr,
                    target_bgr=target_bgr,
                    target_path=target_path,
                )
        except Exception as exc:
            return SwapResult(
                status=ENGINEERING_ERROR,
                error=(
                    f"{type(exc).__name__}: {exc}\n"
                    f"{traceback.format_exc(limit=8)}"
                ),
            )

    def swap(self, *, source01: torch.Tensor, target_path: str) -> SwapResult:
        """Compatibility entry point for tensor callers.

        The tensor is serialized once to a PNG and then passed through the same
        path-based official SimSwap sequence as `swap_paths`. This keeps old
        Module05 callers working without maintaining a second inference path.
        """
        try:
            with tempfile.TemporaryDirectory(
                prefix="module05_simswap_source_"
            ) as td:
                source_path = Path(td) / "source.png"
                x = source01.detach().cpu().clamp(0, 1)
                if x.ndim != 4 or x.shape[0] != 1 or x.shape[1] != 3:
                    raise ValueError(
                        "Expected source tensor shape [1,3,H,W]."
                    )
                arr = (
                    x[0].permute(1, 2, 0).numpy() * 255.0
                ).round().astype(np.uint8)
                Image.fromarray(arr, mode="RGB").save(source_path)
                return self.swap_paths(
                    source_path=str(source_path),
                    target_path=target_path,
                )
        except Exception as exc:
            return SwapResult(
                status=ENGINEERING_ERROR,
                error=(
                    f"{type(exc).__name__}: {exc}\n"
                    f"{traceback.format_exc(limit=8)}"
                ),
            )


def _copy_target_as_png(src_path: str, dst_path: Path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    Image.open(src_path).convert("RGB").save(dst_path)


class FaceSwapEvaluator:
    """
    Final black-box/transfer face-swap evaluation.

    Output pair directory contains the requested visual categories:
      source.png
      target.png
      protected.png
      swap_clean.png       (when a real swap is produced)
      swap_protected.png   (when a real swap is produced)

    A detection-blocked case is never replaced by a fabricated swap image.
    Instead `swap.json` records the exact status.
    """
    def __init__(
        self,
        *,
        cfg,
        swapper,
        identity_system,
        device,
        output_dir,
        seed,
    ):
        self.cfg = cfg
        self.swapper = swapper
        self.identity = identity_system
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        # Reproducibility belongs to FullSystemConfig, not FaceSwapConfig.
        self.seed = int(seed)

    def _identity_to_source(self, image01, source_identity_context):
        with torch.no_grad():
            sims = self.identity.similarities(
                image01,
                source_identity_context,
            )
        return {
            k: float(v.detach().cpu())
            for k, v in sims.items()
        }

    def _swap_saved_source(self, source_path: Path, target_path: str):
        """Prefer the official path-based SimSwap route when available."""
        if hasattr(self.swapper, "swap_paths"):
            return self.swapper.swap_paths(
                source_path=str(source_path),
                target_path=str(target_path),
            )
        source01 = load_rgb01(source_path, 256).unsqueeze(0).to(self.device)
        return self.swapper.swap(
            source01=source01,
            target_path=str(target_path),
        )

    def evaluate(
        self,
        *,
        test_paths,
        defense_results,
        source_contexts,
        pairs=None,
        precomputed_clean_swaps=None,
    ):
        by_id = {str(x["sample_id"]): x for x in defense_results}
        if pairs is None:
            pairs = deterministic_test_pairs(
                test_paths,
                targets_per_source=self.cfg.targets_per_source,
                seed=self.seed,
            )
        else:
            pairs = tuple(pairs)
        precomputed_clean_swaps = dict(precomputed_clean_swaps or {})

        pair_manifest = [asdict(x) | {"pair_id": x.pair_id} for x in pairs]
        json_dump(
            self.output_dir / "faceswap_pairs.json",
            pair_manifest,
        )

        rows = []
        for pair in pairs:
            source_row = by_id[pair.source_id]
            pair_dir = (
                self.output_dir
                / self.cfg.model
                / pair.pair_id
            )
            pair_dir.mkdir(parents=True, exist_ok=True)

            clean = load_rgb01(
                pair.source_path,
                256,
            ).unsqueeze(0).to(self.device)
            save_image01(clean, pair_dir / "source.png")
            _copy_target_as_png(
                pair.target_path,
                pair_dir / "target.png",
            )

            # Always execute the clean-source baseline. This guarantees that
            # GPU smoke actually validates the real SimSwap path even when the
            # under-trained defense marks a protected sample infeasible.
            clean_swap = precomputed_clean_swaps.get(pair.pair_id)
            if clean_swap is None:
                clean_swap = self._swap_saved_source(
                    pair_dir / "source.png",
                    pair.target_path,
                )
            if (
                self.cfg.strict_engineering_errors
                and clean_swap.status == ENGINEERING_ERROR
            ):
                raise RuntimeError(
                    "Clean SimSwap engineering failure:\n"
                    + (clean_swap.error or "unknown")
                )
            if clean_swap.ok:
                clean_swap.image01 = clean_swap.image01.to(self.device)
                save_image01(
                    clean_swap.image01,
                    pair_dir / "swap_clean.png",
                )

            protected_path = (
                Path(source_row["_sample_dir"]) / "protected.png"
            )
            if source_row["status"] != "OK" or not protected_path.is_file():
                clean_sim = (
                    self._identity_to_source(
                        clean_swap.image01,
                        source_contexts[pair.source_id].identity,
                    )
                    if clean_swap.ok
                    else None
                )
                rec = {
                    **asdict(pair),
                    "pair_id": pair.pair_id,
                    "status": PROTECTION_INFEASIBLE,
                    "clean_swap_status": clean_swap.status,
                    "protected_swap_status": None,
                    "clean_swap_error": clean_swap.error,
                    "protected_swap_error": None,
                    "clean_swap_source_similarity": clean_sim,
                    "protected_swap_source_similarity": None,
                    "identity_similarity_drop": None,
                    "protected_detection_blocked": False,
                    "visual_five_class_complete": False,
                }
                json_dump(
                    pair_dir / "swap.json",
                    {
                        **rec,
                        "clean_swap_metadata": clean_swap.metadata,
                        "protected_swap_metadata": None,
                    },
                )
                rows.append(rec)
                continue

            protected = load_rgb01(
                protected_path,
                256,
            ).unsqueeze(0).to(self.device)
            save_image01(protected, pair_dir / "protected.png")

            protected_swap = self._swap_saved_source(
                pair_dir / "protected.png",
                pair.target_path,
            )

            if (
                self.cfg.strict_engineering_errors
                and protected_swap.status == ENGINEERING_ERROR
            ):
                raise RuntimeError(
                    "Protected SimSwap engineering failure is never counted "
                    "as defense success:\n"
                    + (protected_swap.error or "unknown")
                )

            if protected_swap.ok:
                save_image01(
                    protected_swap.image01,
                    pair_dir / "swap_protected.png",
                )

            source_identity_context = (
                source_contexts[pair.source_id].identity
            )

            clean_sim = (
                self._identity_to_source(
                    clean_swap.image01,
                    source_identity_context,
                )
                if clean_swap.ok
                else None
            )
            protected_sim = (
                self._identity_to_source(
                    protected_swap.image01,
                    source_identity_context,
                )
                if protected_swap.ok
                else None
            )

            drop = None
            if clean_sim is not None and protected_sim is not None:
                drop = {
                    k: clean_sim[k] - protected_sim[k]
                    for k in clean_sim
                }

            five_complete = all(
                (pair_dir / name).is_file()
                for name in (
                    "source.png",
                    "target.png",
                    "protected.png",
                    "swap_clean.png",
                    "swap_protected.png",
                )
            )

            rec = {
                **asdict(pair),
                "pair_id": pair.pair_id,
                "status": "EVALUATED",
                "clean_swap_status": clean_swap.status,
                "protected_swap_status": protected_swap.status,
                "clean_swap_error": clean_swap.error,
                "protected_swap_error": protected_swap.error,
                "clean_swap_source_similarity": clean_sim,
                "protected_swap_source_similarity": protected_sim,
                "identity_similarity_drop": drop,
                "protected_detection_blocked": (
                    protected_swap.status == SOURCE_DETECTION_FAILED
                ),
                "visual_five_class_complete": five_complete,
            }
            json_dump(
                pair_dir / "swap.json",
                {
                    **rec,
                    "clean_swap_metadata": clean_swap.metadata,
                    "protected_swap_metadata": protected_swap.metadata,
                },
            )
            rows.append(rec)

        json_dump(
            self.output_dir / "faceswap_results.json",
            rows,
        )

        flat = []
        for x in rows:
            clean_sim = x.get("clean_swap_source_similarity") or {}
            prot_sim = x.get("protected_swap_source_similarity") or {}
            drop = x.get("identity_similarity_drop") or {}
            flat.append({
                "pair_id": x.get("pair_id"),
                "source_id": x.get("source_id"),
                "target_id": x.get("target_id"),
                "status": x.get("status"),
                "clean_swap_status": x.get("clean_swap_status"),
                "protected_swap_status": x.get("protected_swap_status"),
                "protected_detection_blocked": x.get(
                    "protected_detection_blocked"
                ),
                "visual_five_class_complete": x.get(
                    "visual_five_class_complete"
                ),
                "clean_arcface_similarity": clean_sim.get("arcface"),
                "protected_arcface_similarity": prot_sim.get("arcface"),
                "arcface_similarity_drop": drop.get("arcface"),
                "clean_facenet_similarity": clean_sim.get("facenet"),
                "protected_facenet_similarity": prot_sim.get("facenet"),
                "facenet_similarity_drop": drop.get("facenet"),
            })
        write_csv(
            self.output_dir / "faceswap_results.csv",
            flat,
            fieldnames=[
                "pair_id",
                "source_id",
                "target_id",
                "status",
                "clean_swap_status",
                "protected_swap_status",
                "protected_detection_blocked",
                "visual_five_class_complete",
                "clean_arcface_similarity",
                "protected_arcface_similarity",
                "arcface_similarity_drop",
                "clean_facenet_similarity",
                "protected_facenet_similarity",
                "facenet_similarity_drop",
            ],
        )
        return rows


def summarize_faceswap(rows):
    evaluated = [x for x in rows if x.get("status") == "EVALUATED"]
    engineering_errors = sum(
        (x.get("clean_swap_status") == ENGINEERING_ERROR)
        or (x.get("protected_swap_status") == ENGINEERING_ERROR)
        for x in rows
    )
    clean_ok = sum(x.get("clean_swap_status") == SWAP_OK for x in rows)
    protected_ok = sum(
        x.get("protected_swap_status") == SWAP_OK for x in rows
    )
    protected_blocked = sum(
        bool(x.get("protected_detection_blocked"))
        for x in evaluated
    )
    five_complete = sum(
        bool(x.get("visual_five_class_complete"))
        for x in evaluated
    )

    arc_drops = [
        x["identity_similarity_drop"]["arcface"]
        for x in evaluated
        if x.get("identity_similarity_drop")
        and "arcface" in x["identity_similarity_drop"]
    ]
    facenet_drops = [
        x["identity_similarity_drop"]["facenet"]
        for x in evaluated
        if x.get("identity_similarity_drop")
        and "facenet" in x["identity_similarity_drop"]
    ]

    def mean_or_none(xs):
        return sum(xs) / len(xs) if xs else None

    attack_eligible = [
        x for x in rows if x.get("clean_swap_status") == SWAP_OK
    ]

    return {
        "faceswap_model": "simswap",
        "faceswap_pairs_total": len(rows),
        "faceswap_pairs_evaluated": len(evaluated),
        "attack_eligible_pairs": len(attack_eligible),
        "defense_denominator_pairs": len(attack_eligible),
        "clean_swap_ok": clean_ok,
        "protected_swap_ok": protected_ok,
        "protected_source_detection_blocked": protected_blocked,
        "engineering_error_pairs": engineering_errors,
        "five_class_visual_complete_pairs": five_complete,
        "mean_arcface_similarity_drop": mean_or_none(arc_drops),
        "mean_facenet_similarity_drop": mean_or_none(facenet_drops),
    }
