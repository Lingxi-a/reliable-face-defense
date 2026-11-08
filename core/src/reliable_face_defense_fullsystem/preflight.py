from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import torch


REQUIRED = [
    ("source/FacePoison-main", "dir"),
    ("source/Pytorch_Retinaface-master", "dir"),
    ("source/insightface-master", "dir"),
    ("weights/detection/mobilenet0.25_Final.pth", "file"),
    ("weights/identity/arcface_r50/backbone.pth", "file"),
    ("weights/identity/facenet/20180402-114759-vggface2.pt", "file"),
    ("dataset/CelebA-HQ/CelebA-HQ-img", "dir"),
]

EXPECTED_FACEPOISON = {
    "eps":8,
    "alpha":2,
    "mask_p":0.4,
    "ens":30,
    "n_iter":10,
    "lambda":1,
    "term_weight":"1,1",
    "fdmean":1,
    "hybrid_mode":2,
    "avgerage_grad":1,
    "flip_sign_p":0.01,
    "layer":"body.stage1,body.stage2,body.stage3",
    "feat_weight":"0.2,0.3,0.5",
}


def _audit_facepoison_config(deps):
    root=(deps/"source"/"FacePoison-main").resolve()
    if str(root) not in sys.path:
        sys.path.insert(0,str(root))
    from attack_public_datasets.adv_config_default import FacePoisonPP_cfg
    cfg=dict(FacePoisonPP_cfg["retinaface"])
    mismatch={}
    for k,v in EXPECTED_FACEPOISON.items():
        if k not in cfg or cfg[k] != v:
            mismatch[k]={"expected":v,"actual":cfg.get(k)}
    if mismatch:
        raise RuntimeError(
            "Downloaded FacePoison config drifted from the audited local "
            f"configuration: {mismatch}"
        )
    return True


def _faceswap_checks(cfg):
    fs = cfg.faceswap
    if not fs.enabled:
        return {}

    repo = Path(fs.repo_root)
    required_repo_files = [
        "models/models.py",
        "options/test_options.py",
        "insightface_func/face_detect_crop_single.py",
        "util/reverse2original.py",
        "util/norm.py",
    ]
    checks = {
        "faceswap_enabled": True,
        "simswap_repo_exists": repo.is_dir(),
        "simswap_generator_checkpoint_exists": (
            Path(fs.checkpoints_dir) / "people" / "latest_net_G.pth"
        ).is_file(),
        "simswap_arcface_checkpoint_exists": Path(fs.arcface_path).is_file(),
    }
    for rel in required_repo_files:
        checks[f"simswap_repo_file:{rel}"] = (repo / rel).is_file()

    canonical = Path(fs.detector_model_path)
    fallback = next(
        (Path(p) for p in fs.detector_fallbacks if Path(p).is_file()),
        None,
    )
    checks["simswap_detector_canonical_exists"] = canonical.is_file()
    checks["simswap_detector_fallback_available"] = fallback is not None
    checks["simswap_detector_ready_or_stageable"] = (
        canonical.is_file() or fallback is not None
    )
    checks["simswap_detector_selected_source"] = str(
        canonical if canonical.is_file() else fallback
    )
    return checks


def run_preflight(cfg, *, require_nullswap_checkpoint=False):
    deps=Path(cfg.deps_root)
    checks={}
    for rel,kind in REQUIRED:
        p=deps/rel
        ok=p.is_dir() if kind=="dir" else p.is_file()
        checks[rel]=ok

    checks["lpips_importable"]=importlib.util.find_spec("lpips") is not None
    checks["facenet_pytorch_importable"]=(
        importlib.util.find_spec("facenet_pytorch") is not None
    )
    checks["cv2_importable"]=importlib.util.find_spec("cv2") is not None
    checks["scipy_importable"]=importlib.util.find_spec("scipy") is not None
    checks["torchvision_importable"]=(
        importlib.util.find_spec("torchvision") is not None
    )
    checks["onnxruntime_importable"]=(
        importlib.util.find_spec("onnxruntime") is not None
    )
    insightface_pkg = (
        deps / "source" / "insightface-master" /
        "python-package" / "insightface" / "__init__.py"
    )
    checks["insightface_python_package_source_exists"] = (
        insightface_pkg.is_file()
    )
    checks["cuda_available"]=torch.cuda.is_available()

    alex=Path(torch.hub.get_dir())/"checkpoints"/"alexnet-owt-7be5be79.pth"
    checks["lpips_alexnet_cache_exists"]=alex.is_file()
    checks["lpips_alexnet_cache_path"]=str(alex)

    ckpt=Path(cfg.nullswap.checkpoint)
    checks["nullswap_checkpoint_exists"]=ckpt.exists()

    if checks["source/FacePoison-main"]:
        checks["facepoison_official_config_audited"]=_audit_facepoison_config(deps)
    else:
        checks["facepoison_official_config_audited"]=False

    dataset_dir=deps/"dataset"/"CelebA-HQ"/"CelebA-HQ-img"
    if dataset_dir.is_dir():
        n=sum(
            1 for p in dataset_dir.iterdir()
            if p.suffix.lower() in {".jpg",".jpeg",".png"}
        )
    else:
        n=0
    checks["dataset_image_count"]=n
    checks["dataset_count_sufficient"]=(
        n >= cfg.split.train+cfg.split.calibration+cfg.split.control+cfg.split.test
    )

    checks.update(_faceswap_checks(cfg))

    failed=[]
    bool_required=[
        *[rel for rel,_ in REQUIRED],
        "lpips_importable",
        "facenet_pytorch_importable",
        "cv2_importable",
        "scipy_importable",
        "torchvision_importable",
        "onnxruntime_importable",
        "insightface_python_package_source_exists",
        "lpips_alexnet_cache_exists",
        "facepoison_official_config_audited",
        "dataset_count_sufficient",
    ]
    if cfg.faceswap.enabled:
        bool_required.extend([
            "simswap_repo_exists",
            "simswap_generator_checkpoint_exists",
            "simswap_arcface_checkpoint_exists",
            "simswap_detector_ready_or_stageable",
            "simswap_repo_file:models/models.py",
            "simswap_repo_file:options/test_options.py",
            "simswap_repo_file:insightface_func/face_detect_crop_single.py",
            "simswap_repo_file:util/reverse2original.py",
            "simswap_repo_file:util/norm.py",
        ])

    for k in bool_required:
        if not bool(checks[k]):
            failed.append(k)
    if cfg.device=="cuda" and not checks["cuda_available"]:
        failed.append("cuda_available")
    if require_nullswap_checkpoint or not cfg.nullswap.train_if_missing:
        if not checks["nullswap_checkpoint_exists"]:
            failed.append("nullswap_checkpoint_exists")

    if failed:
        raise RuntimeError("Preflight failed: "+", ".join(failed))

    from .facepoison_vendor.facepoison_retina_adapter import (
        FacePoisonRetinaAdapter,
    )
    from reliable_face_defense_02r3.nullswap.generator import NullSwapGenerator
    checks["facepoison_adapter_importable"]=FacePoisonRetinaAdapter is not None
    checks["nullswap_generator_importable"]=NullSwapGenerator is not None

    if torch.cuda.is_available():
        checks["gpu_name"]=torch.cuda.get_device_name(0)
        free,total=torch.cuda.mem_get_info()
        checks["gpu_free_gib"]=round(free/(1024**3),3)
        checks["gpu_total_gib"]=round(total/(1024**3),3)

    return checks
