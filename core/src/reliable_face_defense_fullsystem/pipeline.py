from __future__ import annotations

from pathlib import Path
import hashlib
import json
import statistics

from .config import load_config
from .data import build_manifest, DatasetManifest
from .preflight import run_preflight
from .nullswap_training import train_nullswap
from .calibration import calibrate, save as save_calibration
from .reliability import load_calibration
from .build import build_full_system
from .repro import set_global_seed
from .faceswap import (
    SimSwapAdapter,
    FaceSwapEvaluator,
    summarize_faceswap,
    build_faceswap_candidate_pool,
    select_clean_attack_eligible_pairs,
)


def _paths_sha256(paths):
    h=hashlib.sha256()
    for p in paths:
        h.update(str(p).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _validate_manifest(manifest, cfg):
    expected={
        "train":cfg.split.train,
        "calibration":cfg.split.calibration,
        "control":cfg.split.control,
        "test":cfg.split.test,
    }
    for name,n in expected.items():
        xs=getattr(manifest,name)
        if len(xs)!=n:
            raise RuntimeError(
                f"Existing manifest split '{name}' has {len(xs)} items, "
                f"but config requires {n}. Use a new output_dir or remove "
                "the stale manifest intentionally."
            )
        missing=[p for p in xs if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Manifest contains missing file(s), first={missing[0]}"
            )
    if int(manifest.seed)!=int(cfg.seed):
        raise RuntimeError(
            f"Manifest seed {manifest.seed} != config seed {cfg.seed}."
        )


def _validate_calibration(calibration, cfg, calibration_paths):
    meta=calibration.get("_meta")
    if not isinstance(meta,dict):
        raise RuntimeError(
            "Existing calibration lacks Module05 provenance metadata. "
            "Use a new output_dir or remove calibration.json to recalibrate."
        )
    expected={
        "calibration_paths_sha256":_paths_sha256(calibration_paths),
        "seed":int(cfg.seed),
        "eta_anchor_iou":float(cfg.detection_calibration.eta_anchor_iou),
        "lse_alpha":float(cfg.detection_calibration.lse_alpha),
        "target_clean_tpr":float(cfg.detection_calibration.target_clean_tpr),
        "identity_far":float(cfg.identity_calibration.far),
        "negative_pairs":int(cfg.identity_calibration.negative_pairs),
    }
    for k,v in expected.items():
        if meta.get(k)!=v:
            raise RuntimeError(
                f"Calibration provenance mismatch for {k}: "
                f"file={meta.get(k)!r}, config={v!r}"
            )


def run_pipeline(config_path):
    cfg=load_config(config_path)
    out=Path(cfg.output_dir)
    out.mkdir(parents=True,exist_ok=True)

    reproducibility=set_global_seed(cfg.seed)
    (out/"reproducibility.json").write_text(
        json.dumps(reproducibility,indent=2),encoding="utf-8"
    )

    pre=run_preflight(cfg,require_nullswap_checkpoint=False)
    (out/"preflight.json").write_text(
        json.dumps(pre,indent=2),encoding="utf-8"
    )

    manifest_path=out/"manifest.json"
    if manifest_path.exists():
        manifest=DatasetManifest.load(manifest_path)
        _validate_manifest(manifest,cfg)
    else:
        manifest=build_manifest(
            dataset_dir=cfg.dataset_dir,
            train=cfg.split.train,
            calibration=cfg.split.calibration,
            control=cfg.split.control,
            test=cfg.split.test,
            seed=cfg.seed,
        )
        manifest.save(manifest_path)

    ckpt=Path(cfg.nullswap.checkpoint)
    # Reuse an existing NullSwap checkpoint.  `train_if_missing` means exactly
    # that: train only when the checkpoint is absent.  Earlier revisions
    # retrained on every run whenever this flag was true, which made visual
    # sanity checks non-reproducible and wasted GPU time.
    if not ckpt.exists():
        if not cfg.nullswap.train_if_missing:
            raise FileNotFoundError(ckpt)
        train_nullswap(
            deps_root=cfg.deps_root,
            train_paths=manifest.train,
            checkpoint=ckpt,
            device=cfg.device,
            epochs=cfg.nullswap.epochs,
            batch_size=cfg.nullswap.batch_size,
            num_workers=cfg.nullswap.num_workers,
            seed=cfg.seed,
            base_channels=cfg.nullswap.base_channels,
            disc_base_channels=cfg.nullswap.disc_base_channels,
            disc_blocks=cfg.nullswap.disc_blocks,
            resume=True,
            log_path=out/"nullswap_train.jsonl",
        )

    cal_file=out/"calibration.json"
    if cal_file.exists():
        calibration=load_calibration(cal_file)
        _validate_calibration(calibration,cfg,manifest.calibration)
    else:
        calibration=calibrate(
            deps_root=cfg.deps_root,
            paths=manifest.calibration,
            device=cfg.device,
            eta_anchor_iou=cfg.detection_calibration.eta_anchor_iou,
            lse_alpha=cfg.detection_calibration.lse_alpha,
            target_clean_tpr=cfg.detection_calibration.target_clean_tpr,
            identity_far=cfg.identity_calibration.far,
            negative_pairs=cfg.identity_calibration.negative_pairs,
            seed=cfg.seed,
            batch_size=max(1,min(8,cfg.nullswap.batch_size)),
        )
        save_calibration(cal_file,calibration)

    system=build_full_system(cfg=cfg,calibration=calibration)
    active,history=system.run_control_rounds(manifest.control)
    test_results=system.run_test(manifest.test,active)

    faceswap_summary = None
    if cfg.faceswap.enabled:
        # Face-swap evaluation uses an attack-eligible held-out subset. Clean
        # SimSwap must succeed before the protected result can enter the
        # defense denominator. Training/calibration/control images are never
        # candidates. This prevents clean attack failure from being credited to
        # the defense, especially in tiny smoke splits.
        swapper = SimSwapAdapter(
            cfg=cfg.faceswap,
            device=cfg.device,
        )
        fs_out = out/"final_test"/"faceswap"
        candidate_pool = build_faceswap_candidate_pool(
            dataset_dir=cfg.dataset_dir,
            excluded_paths=(
                *manifest.train,
                *manifest.calibration,
                *manifest.control,
            ),
            seed=cfg.seed,
        )
        selected_pairs, eligibility_records, clean_cache = (
            select_clean_attack_eligible_pairs(
                candidate_paths=candidate_pool,
                swapper=swapper,
                device=cfg.device,
                source_count=cfg.split.test,
                targets_per_source=cfg.faceswap.targets_per_source,
                strict_engineering_errors=cfg.faceswap.strict_engineering_errors,
                output_path=fs_out/"faceswap_eligibility.json",
            )
        )
        selected_source_paths = tuple(dict.fromkeys(
            pair.source_path for pair in selected_pairs
        ))

        # Only attack-eligible source images need defense processing here. The
        # original reliability final-test results remain untouched.
        fs_defense_results = system.run_test(
            selected_source_paths,
            active,
            output_dir=out/"final_test"/"faceswap_source_defense",
        )
        _, source_contexts = system._prepare_split(selected_source_paths)

        fs_eval = FaceSwapEvaluator(
            cfg=cfg.faceswap,
            swapper=swapper,
            identity_system=system.reliability.identity,
            device=cfg.device,
            output_dir=fs_out,
            seed=cfg.seed,
        )
        faceswap_rows = fs_eval.evaluate(
            test_paths=selected_source_paths,
            defense_results=fs_defense_results,
            source_contexts=source_contexts,
            pairs=selected_pairs,
            precomputed_clean_swaps=clean_cache,
        )
        faceswap_summary = summarize_faceswap(faceswap_rows)
        faceswap_summary.update({
            "targets_per_source": int(cfg.faceswap.targets_per_source),
            "eligibility_candidate_sources_checked": len(eligibility_records),
            "eligibility_selected_sources": len(selected_source_paths),
            "eligibility_rule": (
                "clean SimSwap SWAP_OK required before defense evaluation"
            ),
        })

    feasible=[x for x in test_results if x["status"]=="OK"]
    route_counts={}
    eps_values=[]
    all_pool_ok=[]
    for x in feasible:
        route_counts[x["route"]]=route_counts.get(x["route"],0)+1
        if x["epsilon_px"] is not None:
            eps_values.append(float(x["epsilon_px"]))
        all_pool_ok.append(bool(x["robust_all_pool_reliable"]))

    eps_max=max(cfg.blue.epsilon_grid_px)
    summary={
        "status":"OK",
        "active_constraints":list(active),
        "control_rounds":history,
        "test_total":len(test_results),
        "test_feasible":len(feasible),
        "test_infeasible":len(test_results)-len(feasible),
        "route_counts":route_counts,
        "facepoison_epsilon_values":eps_values,
        "facepoison_epsilon_mean":(
            sum(eps_values)/len(eps_values) if eps_values else None
        ),
        "facepoison_epsilon_median":(
            statistics.median(eps_values) if eps_values else None
        ),
        "facepoison_epsilon_saturation_rate":(
            sum(abs(x-eps_max)<1e-9 for x in eps_values)/len(eps_values)
            if eps_values else None
        ),
        "robust_all_pool_reliable_rate":(
            sum(all_pool_ok)/len(all_pool_ok) if all_pool_ok else None
        ),
        "faceswap":faceswap_summary,
        "output_dir":str(out),
    }
    (out/"summary.json").write_text(
        json.dumps(summary,indent=2),encoding="utf-8"
    )
    return summary
