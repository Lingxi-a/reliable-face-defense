from __future__ import annotations

from pathlib import Path
import shutil

import torch

from .common import (
    append_jsonl,
    load_rgb01,
    read_jsonl,
    save_rgb01,
    stable_seed,
    write_csv,
    write_json,
)


def make_core_config(cfg, protocol, out_dir):
    simswap = cfg["simswap"]
    payload = {
        "deps_root": cfg["deps_root"],
        "device": cfg["device"],
        "output_dir": str(out_dir / "core"),
        "seed": int(cfg["seed"]),
        "image_size": 256,
        "split": {
            "train": 1,
            "calibration": len(protocol["calibration_paths"]),
            "control": len(protocol["control_paths"]),
            "test": len(protocol["source_paths"]),
        },
        "nullswap": {
            "checkpoint": cfg["nullswap_checkpoint"],
            "train_if_missing": False,
            "epochs": 1,
            "batch_size": 8,
            "base_channels": 64,
            "disc_base_channels": 32,
            "disc_blocks": 5,
            "num_workers": 0,
        },
        "reliability": {
            "detection": {
                "eta_anchor_iou": 0.5,
                "lse_alpha": 10.0,
                "target_clean_tpr": float(cfg["detection_clean_tpr"]),
            },
            "identity": {
                "far": float(cfg["identity_far"]),
                "negative_pairs": int(cfg["identity_negative_pairs"]),
            },
        },
        "blue": {
            "epsilon_grid_px": cfg["blue"]["epsilon_grid_px"],
            "distortion_metric": "lpips",
        },
        "rpp": dict(cfg["rpp"]),
        "red": dict(cfg["red"]),
        "deployment": {
            "visual_gate_enabled": False,
            "min_psnr_db": None,
            "max_lpips": None,
        },
        "faceswap": {
            "enabled": False,
            "model": "simswap",
            "repo_root": simswap["repo_root"],
            "checkpoints_dir": simswap["checkpoints_dir"],
            "arcface_path": simswap["arcface_path"],
            "detector_model_path": simswap["detector_model_path"],
            "detector_fallbacks": simswap["detector_fallbacks"],
            "crop_size": int(simswap["crop_size"]),
            "det_thresh": float(simswap["det_thresh"]),
            "targets_per_source": int(cfg["targets_per_source"]),
            "strict_engineering_errors": True,
        },
    }
    path = out_dir / "frozen_core_config.json"
    write_json(path, payload)
    return path


def _visual_metrics(system, clean, protected):
    from reliable_face_defense_rpp.metrics import psnr, ssim

    with torch.no_grad():
        lpips = float(system.blue.distortion(clean, protected))
    return {
        "psnr": float(psnr(clean, protected)),
        "ssim": float(ssim(clean, protected)),
        "lpips": lpips,
        "linf_px": float((clean - protected).abs().max().detach().cpu()) * 255.0,
    }


def _random_noise(clean, *, seed, epsilon_px):
    generator = torch.Generator(device=clean.device)
    generator.manual_seed(int(seed))
    noise = torch.empty_like(clean).uniform_(
        -float(epsilon_px) / 255.0,
        float(epsilon_px) / 255.0,
        generator=generator,
    )
    return (clean + noise).clamp(0.0, 1.0)


def generate_protections(cfg, protocol, out_dir):
    from reliable_face_defense_fullsystem.calibration import calibrate
    from reliable_face_defense_fullsystem.config import load_config
    from reliable_face_defense_fullsystem.build import build_full_system
    from reliable_face_defense_fullsystem.reliability import load_calibration

    core_config_path = make_core_config(cfg, protocol, out_dir)
    core_cfg = load_config(core_config_path)
    calibration_path = out_dir / "calibration_official_valid.json"

    if calibration_path.is_file():
        calibration = load_calibration(calibration_path)
    else:
        calibration = calibrate(
            deps_root=cfg["deps_root"],
            paths=protocol["calibration_paths"],
            device=cfg["device"],
            eta_anchor_iou=0.5,
            lse_alpha=10.0,
            target_clean_tpr=float(cfg["detection_clean_tpr"]),
            identity_far=float(cfg["identity_far"]),
            negative_pairs=int(cfg["identity_negative_pairs"]),
            seed=int(cfg["seed"]),
            batch_size=8,
        )
        calibration["_formal_note"] = (
            "One image per official-validation identity was selected, so every "
            "non-self calibration pair is an impostor pair."
        )
        write_json(calibration_path, calibration)

    system = build_full_system(cfg=core_cfg, calibration=calibration)
    active, history = system.run_control_rounds(protocol["control_paths"])
    write_json(out_dir / "active_constraints.json", {
        "active_constraints": list(active),
        "history": history,
    })

    manifest_jsonl = out_dir / "protections" / "manifest.jsonl"
    existing = read_jsonl(manifest_jsonl)
    done = {(x["method"], str(x["source_identity"])) for x in existing}

    for source_identity in protocol["source_identities"]:
        sid = str(source_identity)
        source_path = protocol["source_paths"][sid]
        clean = load_rgb01(source_path, 256).to(cfg["device"])

        method_tensors = {
            "no_protection": clean,
            "random_noise": _random_noise(
                clean,
                seed=stable_seed(cfg["seed"], sid, "random_noise"),
                epsilon_px=cfg["random_noise_epsilon_px"],
            ),
        }
        if ("facepoison", sid) not in done:
            method_tensors["facepoison"] = system.blue.facepoison.protect(
                image01=clean,
                epsilon_px=float(cfg["facepoison_epsilon_px"]),
                sample_id=sid,
            )
        if ("nullswap", sid) not in done:
            method_tensors["nullswap"] = system.blue.nullswap.protect(
                image01=clean,
                sample_id=sid,
            )

        for method, protected in method_tensors.items():
            key = (method, sid)
            if key in done:
                continue
            path = out_dir / "protections" / method / f"{sid}.png"
            save_rgb01(protected, path)
            row = {
                "source_identity": source_identity,
                "source_path": source_path,
                "method": method,
                "status": "OK",
                "protected_path": str(path),
                "route": method,
                **_visual_metrics(system, clean, protected),
            }
            append_jsonl(manifest_jsonl, row)
            existing.append(row)
            done.add(key)

        if ("ours", sid) not in done:
            context = system.reliability.prepare_context(sid, clean)
            rec = system._process_one(
                sample_id=sid,
                path=source_path,
                clean=clean,
                context=context,
                active_constraints=active,
                round_dir=Path(core_cfg.output_dir) / "formal_sources",
            )
            ours_path = out_dir / "protections" / "ours" / f"{sid}.png"
            if rec["status"] == "OK":
                save_rgb01(rec["protected"], ours_path)
                metrics = _visual_metrics(system, clean, rec["protected"])
                route = rec["blue"].chosen.route.value
                epsilon = rec["blue"].chosen.epsilon_px
            else:
                metrics = {"psnr": None, "ssim": None, "lpips": None, "linf_px": None}
                route = None
                epsilon = None
            row = {
                "source_identity": source_identity,
                "source_path": source_path,
                "method": "ours",
                "status": rec["status"],
                "protected_path": str(ours_path) if ours_path.is_file() else None,
                "route": route,
                "epsilon_px": epsilon,
                **metrics,
            }
            append_jsonl(manifest_jsonl, row)
            existing.append(row)
            done.add(("ours", sid))

        del clean, method_tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows = read_jsonl(manifest_jsonl)
    write_json(out_dir / "protections" / "manifest.json", rows)
    write_csv(out_dir / "protections" / "manifest.csv", rows)
    return rows
