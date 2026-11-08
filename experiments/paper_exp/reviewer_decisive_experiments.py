from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import csv
import gc
import json
import random
import shutil
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .common import (
    append_jsonl,
    load_rgb01,
    read_json,
    read_jsonl,
    save_rgb01,
    sha256_file,
    write_csv,
    write_json,
)
from .evaluators import DetectionGate, HeldOutEvaluators, build_gallery, rank_hits
from .reviewer_decisive_refinement import (
    ExperimentConfig,
    GATE_VARIANTS,
    ReviewerDecisiveRefiner,
    STRATEGIES,
)
from .swappers import (
    ENGINEERING_ERROR,
    SOURCE_DETECTION_FAILED,
    SWAP_OK,
    make_swapper,
    release_swapper,
)


HELD_OUT = ("vggface", "sface")
KNOWN_ENVIRONMENTS = {
    "clean": None,
    "jpeg_q60": ("jpeg", (60,)),
    "blur_sigma_1.0": ("blur", (1.0,)),
    "osn_QW": ("osn", (75, 80)),
}
UNSEEN_ENVIRONMENTS = {
    "jpeg_q45": ("jpeg", (45,)),
    "jpeg_q75": ("jpeg", (75,)),
    "blur_sigma_0.5": ("blur", (0.5,)),
    "blur_sigma_1.5": ("blur", (1.5,)),
    "osn_65_85": ("osn", (65, 85)),
}
NEW_KNOWN_METHODS = (
    "clean_full_gate",
    "uniform_full_gate",
    "frozen_softmax_full_gate",
    "g0_ungated",
    "g1_worst_gain",
    "g2_conditionwise",
)
DYNAMIC_METHOD = "dynamic_softmax_full_gate"
ALL_KNOWN_METHODS = (*NEW_KNOWN_METHODS[:3], DYNAMIC_METHOD, *NEW_KNOWN_METHODS[3:])
GATE_TABLE_METHODS = ("g0_ungated", "g1_worst_gain", "g2_conditionwise", "g3_full")


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_manifest(path_root: Path) -> list[dict]:
    jsonl = path_root / "protections" / "manifest.jsonl"
    if jsonl.is_file():
        return read_jsonl(jsonl)
    return read_json(path_root / "protections" / "manifest.json")


def _pixel_linf_px(path_a: str | Path, path_b: str | Path) -> float:
    a = np.asarray(Image.open(path_a).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(path_b).convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        raise ValueError(f"image shape mismatch: {a.shape} != {b.shape}")
    return float(np.abs(a - b).max())


def _target_zero_pairs(protocol: dict, source_limit: int | None) -> list[dict]:
    allowed = None
    if source_limit is not None:
        allowed = set(map(int, protocol["source_identities"][:source_limit]))
    pairs = [
        row for row in protocol["pairs"]
        if int(row["target_index"]) == 0
        and (allowed is None or int(row["source_identity"]) in allowed)
    ]
    expected = len(protocol["source_identities"]) if source_limit is None else source_limit
    if len(pairs) != expected:
        raise RuntimeError(f"target-index-0 pair mismatch: {len(pairs)} != {expected}")
    return pairs


def _eligible_target0(main_dir: Path, cfg: dict, pairs: list[dict]) -> dict[str, set[str]]:
    pair_ids = {row["pair_id"] for row in pairs}
    rows = read_jsonl(main_dir / "results" / "per_pair.jsonl")
    eligible: dict[str, set[str]] = {}
    for model in cfg["swap_models"]:
        eligible[model] = {
            row["pair_id"]
            for row in rows
            if row["swap_model"] == model
            and row["method"] == "no_protection"
            and row["pair_id"] in pair_ids
            and row["status"] == SWAP_OK
            and _as_bool(row["detection_success"])
        }
    return eligible


def _frozen_payload(
    main_dir: Path,
    final_dir: Path,
    output_dir: Path,
    four_eval_dir: Path | None,
    source_limit: int | None,
) -> dict:
    payload = {
        "experiment": "reviewer_decisive_experiments_v04",
        "main_output": str(main_dir),
        "final_output": str(final_dir),
        "output": str(output_dir),
        "four_evaluator_output": None if four_eval_dir is None else str(four_eval_dir),
        "source_limit": source_limit,
        "known_strategies": list(STRATEGIES),
        "gate_variants": list(GATE_VARIANTS),
        "known_environments": {
            key: value for key, value in KNOWN_ENVIRONMENTS.items()
        },
        "unseen_test_only_environments": UNSEEN_ENVIRONMENTS,
        "bootstrap": {"cluster": "source_identity", "replicates": 2000, "seed": 2026},
        "held_out_evaluators_used_for_selection": False,
        "forbidden_modules": ["RPP", "EMA routing", "dynamic epsilon", "RL", "CVaR", "Resize"],
    }
    # Canonicalize tuples to JSON arrays before both writing and comparison so
    # a preflight run and the following full run produce identical payloads.
    return json.loads(json.dumps(payload))


def _validate_and_freeze(
    main_dir: Path,
    final_dir: Path,
    output_dir: Path,
    four_eval_dir: Path | None,
    source_limit: int | None,
) -> tuple[dict, dict, dict, list[dict], dict[str, set[str]]]:
    required = [
        main_dir / "frozen_main_config.json",
        main_dir / "protocol_manifest.json",
        main_dir / "results" / "per_pair.jsonl",
        main_dir / "protections" / "manifest.jsonl",
        final_dir / "frozen_nullswap_floor_config.json",
        final_dir / "protections" / "manifest.csv",
        final_dir / "main" / "per_pair_four_evaluators.csv",
        final_dir / "robustness" / "evaluation_heldout.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required completed outputs:\n" + "\n".join(missing))
    cfg = read_json(main_dir / "frozen_main_config.json")
    protocol = read_json(main_dir / "protocol_manifest.json")
    floor_cfg = read_json(final_dir / "frozen_nullswap_floor_config.json")["config"]
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full experiment")
    if tuple(cfg["swap_models"]) != ("simswap", "blendswap", "faceshifter"):
        raise RuntimeError(f"unexpected swap models: {cfg['swap_models']}")
    refinement = floor_cfg["refinement"]
    expected_environments = tuple(KNOWN_ENVIRONMENTS)
    if tuple(refinement["environments"]) != expected_environments:
        raise RuntimeError("the frozen method environment protocol does not match the paper settings")
    if int(refinement["steps"]) != 32 or float(refinement["residual_budget_px"]) != 2.0:
        raise RuntimeError("the frozen optimization protocol does not match the paper settings")
    pairs = _target_zero_pairs(protocol, source_limit)
    eligible = _eligible_target0(main_dir, cfg, pairs)
    payload = _frozen_payload(main_dir, final_dir, output_dir, four_eval_dir, source_limit)
    frozen_path = output_dir / "frozen_reviewer_decisive_config.json"
    if frozen_path.is_file() and read_json(frozen_path) != payload:
        raise RuntimeError("output directory contains a different frozen configuration")
    write_json(frozen_path, payload)
    write_json(output_dir / "fixed_target0_eligible.json", {
        model: sorted(values) for model, values in eligible.items()
    })
    return cfg, protocol, floor_cfg, pairs, eligible


def _method_manifest_path(output_dir: Path, method: str) -> Path:
    return output_dir / "protections" / method / "manifest.jsonl"


def _selection_row(
    *,
    sid: str,
    source_path: str,
    method: str,
    strategy: str,
    gate: str,
    selection_result: dict,
    strategy_result: dict,
    output_path: Path,
    anchor_path: str,
) -> dict:
    selected = selection_result["selection"]
    visual = selected["visual"]
    saved_anchor_linf = _pixel_linf_px(output_path, anchor_path)
    return {
        "source_identity": int(sid),
        "source_path": source_path,
        "method": method,
        "optimization_strategy": strategy,
        "gate_variant": gate,
        "status": "OK",
        "protected_path": str(output_path),
        "anchor_path": anchor_path,
        "selected_refinement": bool(selection_result["selected_refinement"]),
        "accepted_candidate_count": int(selection_result["accepted_candidate_count"]),
        "candidate_count": int(strategy_result["candidate_count"]),
        "selected_step": selected["step"],
        "selected_residual_scale": selected["residual_scale"],
        "selected_toward_clean": selected["toward_clean"],
        "anchor_worst_similarity": strategy_result["anchor_worst_similarity"],
        "selected_worst_similarity": selected["worst_similarity"],
        "worst_similarity_gain": (
            strategy_result["anchor_worst_similarity"] - selected["worst_similarity"]
        ),
        "psnr": visual["psnr"],
        "ssim": visual["ssim"],
        "lpips": visual["lpips"],
        "clean_linf_px": visual["clean_linf_px"],
        "saved_anchor_linf_px": saved_anchor_linf,
        "selection_checks": selected["checks"],
        "response_summary": strategy_result["response_summary"],
        "strategy_runtime_seconds": strategy_result["runtime_seconds"],
        "peak_cuda_memory_bytes": strategy_result["peak_cuda_memory_bytes"],
        "output_sha256": sha256_file(output_path),
        "held_out_evaluators_used_for_selection": False,
    }


def generate_protections(
    *,
    cfg: dict,
    protocol: dict,
    floor_cfg: dict,
    main_dir: Path,
    output_dir: Path,
    source_limit: int | None,
) -> dict[str, list[dict]]:
    source_ids = list(protocol["source_identities"])
    if source_limit is not None:
        source_ids = source_ids[:source_limit]
    base_manifest = {
        (str(row["source_identity"]), row["method"]): row
        for row in _load_manifest(main_dir)
    }
    methods = (*STRATEGIES, "g0_ungated", "g1_worst_gain", "g2_conditionwise", "g3_full")
    existing = {method: read_jsonl(_method_manifest_path(output_dir, method)) for method in methods}
    done = {
        method: {str(row["source_identity"]) for row in rows}
        for method, rows in existing.items()
    }
    refine_cfg = ExperimentConfig.from_dict(floor_cfg["refinement"])
    refiner = ReviewerDecisiveRefiner(
        deps_root=cfg["deps_root"], device=cfg["device"], config=refine_cfg
    )

    for index, source_identity in enumerate(source_ids, 1):
        sid = str(source_identity)
        anchor_row = base_manifest.get((sid, "nullswap"))
        if not anchor_row or anchor_row.get("status") != "OK" or not anchor_row.get("protected_path"):
            raise RuntimeError(f"missing frozen NullSwap anchor for source={sid}")
        clean = load_rgb01(protocol["source_paths"][sid], 256).to(cfg["device"])
        anchor = load_rgb01(anchor_row["protected_path"], 256).to(cfg["device"])

        for strategy in STRATEGIES:
            required_methods = (
                (strategy, "g0_ungated", "g1_worst_gain", "g2_conditionwise", "g3_full")
                if strategy == DYNAMIC_METHOD else (strategy,)
            )
            if all(sid in done[method] for method in required_methods):
                continue
            result = refiner.refine_strategy(
                clean=clean,
                nullswap_anchor=anchor,
                strategy=strategy,
                return_gate_variants=(strategy == DYNAMIC_METHOD),
            )
            gate_map = (
                {gate: gate for gate in GATE_VARIANTS}
                if strategy == DYNAMIC_METHOD else {strategy: "g3_full"}
            )
            for method, gate in gate_map.items():
                if sid in done[method]:
                    continue
                selected = result["selections"][gate]
                image_path = output_dir / "protections" / method / "images" / f"{sid}.png"
                save_rgb01(selected["tensor"], image_path)
                row = _selection_row(
                    sid=sid,
                    source_path=protocol["source_paths"][sid],
                    method=method,
                    strategy=strategy,
                    gate=gate,
                    selection_result=selected,
                    strategy_result=result,
                    output_path=image_path,
                    anchor_path=anchor_row["protected_path"],
                )
                append_jsonl(_method_manifest_path(output_dir, method), row)
                existing[method].append(row)
                done[method].add(sid)
            if strategy == DYNAMIC_METHOD and sid not in done[DYNAMIC_METHOD]:
                g3_row = existing["g3_full"][-1]
                dynamic_path = output_dir / "protections" / DYNAMIC_METHOD / "images" / f"{sid}.png"
                dynamic_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(g3_row["protected_path"], dynamic_path)
                dynamic_row = dict(g3_row)
                dynamic_row.update({
                    "method": DYNAMIC_METHOD,
                    "protected_path": str(dynamic_path),
                    "output_sha256": sha256_file(dynamic_path),
                })
                append_jsonl(_method_manifest_path(output_dir, DYNAMIC_METHOD), dynamic_row)
                existing[DYNAMIC_METHOD].append(dynamic_row)
                done[DYNAMIC_METHOD].add(sid)

        if index % 5 == 0 or index == len(source_ids):
            print(f"REVIEWER_PROTECTION_PROGRESS={index}/{len(source_ids)}", flush=True)
        del clean, anchor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    expected_ids = {str(x) for x in source_ids}
    for method in methods:
        rows = read_jsonl(_method_manifest_path(output_dir, method))
        by_sid = {str(row["source_identity"]): row for row in rows}
        if set(by_sid) != expected_ids:
            raise RuntimeError(f"incomplete protection manifest for {method}")
        ordered = [by_sid[str(x)] for x in source_ids]
        write_json(output_dir / "protections" / method / "manifest.json", ordered)
        write_csv(output_dir / "protections" / method / "manifest.csv", ordered)
        existing[method] = ordered

    del refiner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return existing


def _materialize_transformed(
    *,
    cfg: dict,
    output_dir: Path,
    method: str,
    protection_by_sid: dict[str, dict],
    environments: dict,
) -> dict[tuple[str, str], str]:
    project_src = str(Path(cfg["project_root"]) / "src")
    if project_src not in sys.path:
        sys.path.insert(0, project_src)
    from reliable_face_defense_agents.red.contracts import RedTransformSpec
    from reliable_face_defense_agents.red.transforms import apply_red_transform

    transformed: dict[tuple[str, str], str] = {}
    for sid, row in protection_by_sid.items():
        for env_name, spec_tuple in environments.items():
            if spec_tuple is None:
                transformed[(sid, env_name)] = row["protected_path"]
                continue
            family, params = spec_tuple
            path = output_dir / "transformed" / method / env_name / f"{sid}.png"
            if not path.is_file():
                image = load_rgb01(row["protected_path"], 256)
                transformed_image = apply_red_transform(
                    image, RedTransformSpec(env_name, family, tuple(params))
                )
                save_rgb01(transformed_image, path)
            transformed[(sid, env_name)] = str(path)
    return transformed


def _error_result(base: dict, status: str, error: str | None) -> dict:
    return {
        **base,
        "status": status,
        "error": error,
        "detection_success": None,
        "detection_score": None,
        "vggface_acc1": True,
        "vggface_acc5": True,
        "sface_acc1": True,
        "sface_acc5": True,
        "error_policy": "target/engineering error counts as attack success",
    }


def _flush_eval(batch, gate, evaluators, gallery, result_path, done):
    if not batch:
        return
    images = torch.cat([
        F.interpolate(item["image"].detach().cpu(), (256, 256), mode="bilinear", align_corners=False)
        for item in batch
    ])
    detected, scores = gate.success(images)
    query = evaluators.embed(images)
    hits = rank_hits(query, gallery, [item["source_identity"] for item in batch])
    for index, item in enumerate(batch):
        detection_success = bool(detected[index])
        row = {
            **item["base"],
            "status": SWAP_OK,
            "error": None,
            "detection_success": detection_success,
            "detection_score": float(scores[index]),
        }
        for evaluator in HELD_OUT:
            row[f"{evaluator}_acc1"] = bool(hits[evaluator]["acc1"][index]) if detection_success else False
            row[f"{evaluator}_acc5"] = bool(hits[evaluator]["acc5"][index]) if detection_success else False
        append_jsonl(result_path, row)
        done.add((row["method"], row["swap_model"], row["environment"], row["pair_id"]))
    batch.clear()


def evaluate_methods(
    *,
    cfg: dict,
    protocol: dict,
    main_dir: Path,
    output_dir: Path,
    pairs: list[dict],
    eligible: dict[str, set[str]],
    method_manifests: dict[str, list[dict]],
    methods: tuple[str, ...],
    environments: dict,
    result_name: str,
) -> list[dict]:
    result_path = output_dir / result_name
    previous = read_jsonl(result_path)
    done = {
        (row["method"], row["swap_model"], row["environment"], row["pair_id"])
        for row in previous
    }
    if len(done) != len(previous):
        raise RuntimeError(f"duplicate rows in {result_path}")

    transformed: dict[tuple[str, str, str], str] = {}
    for method in methods:
        by_sid = {str(row["source_identity"]): row for row in method_manifests[method]}
        materialized = _materialize_transformed(
            cfg=cfg,
            output_dir=output_dir,
            method=method,
            protection_by_sid=by_sid,
            environments=environments,
        )
        for (sid, env), path in materialized.items():
            transformed[(method, sid, env)] = path

    gate = DetectionGate(cfg["deps_root"], cfg["device"])
    gate.calibrate(
        protocol["calibration_paths"],
        target_tpr=float(cfg["detection_clean_tpr"]),
        cache_path=main_dir / "detection_gate.json",
    )
    evaluators = HeldOutEvaluators(cfg, cfg["device"])
    gallery = build_gallery(evaluators, protocol["gallery"], main_dir / "gallery_embeddings.pt")
    pair_by_id = {row["pair_id"]: row for row in pairs}
    total = sum(
        len(eligible[model]) * len(environments) * len(methods)
        for model in cfg["swap_models"]
    )
    print(f"REVIEWER_SWAP_ROWS_EXPECTED={total}", flush=True)

    for model in cfg["swap_models"]:
        swapper = make_swapper(model, cfg, cfg["device"])
        batch = []
        try:
            model_pairs = [pair_by_id[pair_id] for pair_id in sorted(eligible[model])]
            for method in methods:
                for environment in environments:
                    for pair in model_pairs:
                        key = (method, model, environment, pair["pair_id"])
                        if key in done:
                            continue
                        sid = str(pair["source_identity"])
                        source_path = transformed[(method, sid, environment)]
                        base = {
                            "method": method,
                            "swap_model": model,
                            "environment": environment,
                            "pair_id": pair["pair_id"],
                            "source_identity": int(pair["source_identity"]),
                            "target_identity": int(pair["target_identity"]),
                            "target_path": pair["target_path"],
                            "protected_path": source_path,
                        }
                        output = swapper.swap(source_path, pair["target_path"])
                        if output.status != SWAP_OK or output.image01 is None:
                            if output.status == SOURCE_DETECTION_FAILED:
                                row = {
                                    **base,
                                    "status": output.status,
                                    "error": output.error,
                                    "detection_success": False,
                                    "detection_score": None,
                                    "vggface_acc1": False,
                                    "vggface_acc5": False,
                                    "sface_acc1": False,
                                    "sface_acc5": False,
                                }
                            else:
                                row = _error_result(base, output.status, output.error)
                            append_jsonl(result_path, row)
                            done.add(key)
                            continue
                        batch.append({
                            "base": base,
                            "source_identity": pair["source_identity"],
                            "image": output.image01,
                        })
                        if len(batch) >= 12:
                            _flush_eval(batch, gate, evaluators, gallery, result_path, done)
                    _flush_eval(batch, gate, evaluators, gallery, result_path, done)
                    print(
                        f"REVIEWER_SWAP_PROGRESS model={model} method={method} "
                        f"env={environment} rows={len(done)}/{total}",
                        flush=True,
                    )
        finally:
            release_swapper(swapper)

    rows = read_jsonl(result_path)
    expected_keys = {
        (method, model, env, pair_id)
        for method in methods
        for model in cfg["swap_models"]
        for env in environments
        for pair_id in eligible[model]
    }
    actual_keys = {
        (row["method"], row["swap_model"], row["environment"], row["pair_id"])
        for row in rows
        if row["method"] in methods and row["environment"] in environments
    }
    if actual_keys != expected_keys:
        raise RuntimeError(
            f"incomplete evaluation {result_name}: missing={len(expected_keys-actual_keys)} "
            f"extra={len(actual_keys-expected_keys)}"
        )
    selected_rows = [
        row for row in rows
        if row["method"] in methods and row["environment"] in environments
    ]
    write_csv(output_dir / result_name.replace(".jsonl", ".csv"), selected_rows)
    write_json(output_dir / result_name.replace(".jsonl", ".json"), selected_rows)
    del gate, evaluators, gallery
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return selected_rows


def _aggregate(rows: list[dict], methods: tuple[str, ...], environments: tuple[str, ...]) -> tuple[list[dict], list[dict]]:
    source_methods = {
        DYNAMIC_METHOD if method == "g3_full" else method
        for method in methods
    }
    index = defaultdict(list)
    for row in rows:
        if row["method"] in source_methods and row["environment"] in environments:
            index[(row["method"], row["environment"])].append(row)
    table: list[dict] = []
    for method in methods:
        alias = DYNAMIC_METHOD if method == "g3_full" else method
        for environment in environments:
            scoped = index[(alias, environment)]
            if not scoped:
                raise RuntimeError(f"no rows for {alias}/{environment}")
            n = len(scoped)
            for evaluator in HELD_OUT:
                table.append({
                    "method": method,
                    "environment": environment,
                    "evaluator": evaluator,
                    "N_fixed": n,
                    "detection_success_n": sum(_as_bool(x["detection_success"]) for x in scoped),
                    "DetSR": sum(_as_bool(x["detection_success"]) for x in scoped) / n,
                    "attack_success_n_at1": sum(_as_bool(x[f"{evaluator}_acc1"]) for x in scoped),
                    "ASR_at1": sum(_as_bool(x[f"{evaluator}_acc1"]) for x in scoped) / n,
                    "attack_success_n_at5": sum(_as_bool(x[f"{evaluator}_acc5"]) for x in scoped),
                    "ASR_at5": sum(_as_bool(x[f"{evaluator}_acc5"]) for x in scoped) / n,
                })

    robust: list[dict] = []
    for method in methods:
        alias = DYNAMIC_METHOD if method == "g3_full" else method
        method_rows = [row for row in rows if row["method"] == alias]
        by_pair = defaultdict(dict)
        for row in method_rows:
            by_pair[(row["swap_model"], row["pair_id"])][row["environment"]] = row
        for evaluator in HELD_OUT:
            for k in (1, 5):
                reliable = 0
                for env_rows in by_pair.values():
                    if set(env_rows) != set(environments):
                        raise RuntimeError(f"missing environment in robust-all for {method}")
                    reliable += not any(_as_bool(env_rows[env][f"{evaluator}_acc{k}"]) for env in environments)
                n = len(by_pair)
                robust.append({
                    "method": method,
                    "evaluator": evaluator,
                    "metric": f"Robust-All@{k}",
                    "N_fixed": n,
                    "reliable_all_n": reliable,
                    "robust_all_rate": reliable / n,
                })
    return table, robust


def _protection_summary(method_manifests: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for method, manifest in method_manifests.items():
        if method not in {*STRATEGIES, *GATE_VARIANTS}:
            continue
        n = len(manifest)
        selected = sum(_as_bool(row["selected_refinement"]) for row in manifest)
        checks = [row.get("selection_checks") or {} for row in manifest]
        rows.append({
            "method": method,
            "source_count": n,
            "selected_refinement_count": selected,
            "fallback_count": n - selected,
            "selection_rate": selected / n,
            "mean_worst_similarity_gain": float(np.mean([float(row["worst_similarity_gain"]) for row in manifest])),
            "condition_regression_selected_n": sum(float(row.get("selection_checks", {}).get("max_condition_regression", 0.0)) > 0 for row in manifest),
            "visual_regression_selected_n": sum(
                bool(c) and not bool(c.get("visual_no_regression", True)) for c in checks
            ),
            "mean_runtime_seconds": float(np.mean([float(row["strategy_runtime_seconds"]) for row in manifest])),
            "peak_cuda_memory_gib": max(
                (float(row["peak_cuda_memory_bytes"] or 0) for row in manifest), default=0
            ) / (1024 ** 3),
            "max_saved_anchor_linf_px": max(float(row["saved_anchor_linf_px"]) for row in manifest),
            "residual_budget_violation_n": sum(float(row["saved_anchor_linf_px"]) > 2.0 for row in manifest),
        })
    return rows


def _find_four_eval_rows(four_eval_dir: Path | None, main_dir: Path) -> list[dict] | None:
    candidates = []
    if four_eval_dir is not None:
        candidates.extend([
            four_eval_dir / "per_pair_four_evaluators.jsonl",
            four_eval_dir / "per_pair_four_evaluators.json",
        ])
    candidates.extend([
        main_dir / "four_evaluator_supplement" / "per_pair_four_evaluators.jsonl",
        main_dir / "results" / "per_pair_four_evaluators.jsonl",
    ])
    for path in candidates:
        if path.is_file():
            return read_jsonl(path) if path.suffix == ".jsonl" else read_json(path)
    return None


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def cluster_bootstrap(
    *,
    main_dir: Path,
    final_dir: Path,
    four_eval_dir: Path | None,
    output_dir: Path,
) -> tuple[list[dict], list[dict], list[str]]:
    warnings: list[str] = []
    baseline_all = _find_four_eval_rows(four_eval_dir, main_dir)
    if baseline_all is None:
        baseline_all = read_jsonl(main_dir / "results" / "per_pair.jsonl")
        warnings.append("four-evaluator baseline rows not found; bootstrap limited to held-out evaluators")
    final_rows = _read_csv(final_dir / "main" / "per_pair_four_evaluators.csv")
    eligible = read_json(final_dir / "main" / "eligible_pairs.json")
    eligible_keys = {
        (model, pair_id)
        for model, pair_ids in eligible.items()
        for pair_id in pair_ids
    }
    baseline = {
        (row["swap_model"], row["pair_id"]): row
        for row in baseline_all
        if row.get("method") == "nullswap"
        and (row["swap_model"], row["pair_id"]) in eligible_keys
    }
    final = {
        (row["swap_model"], row["pair_id"]): row
        for row in final_rows
        if row.get("method") == "ours"
        and (row["swap_model"], row["pair_id"]) in eligible_keys
    }
    if set(baseline) != set(final) or set(final) != eligible_keys:
        raise RuntimeError("bootstrap rows do not match the frozen main denominator")
    evaluators = [
        name for name in ("arcface", "facenet", "vggface", "sface")
        if all(f"{name}_acc5" in row for row in baseline.values())
    ]
    clusters = defaultdict(list)
    for key in sorted(eligible_keys):
        source_identity = int(final[key]["source_identity"])
        clusters[source_identity].append((baseline[key], final[key]))
    source_ids = sorted(clusters)
    if len(source_ids) != 200:
        warnings.append(f"expected 200 source clusters, found {len(source_ids)}")
    rng = random.Random(2026)
    bootstrap_rows: list[dict] = []
    transition_rows: list[dict] = []
    for evaluator in evaluators:
        for k in (1, 5):
            field = f"{evaluator}_acc{k}"
            all_pairs = [pair for source_id in source_ids for pair in clusters[source_id]]
            b_values = [_as_bool(pair[0][field]) for pair in all_pairs]
            o_values = [_as_bool(pair[1][field]) for pair in all_pairs]
            observed = sum(o_values) / len(o_values) - sum(b_values) / len(b_values)
            draws = []
            for _ in range(2000):
                sampled = [rng.choice(source_ids) for _ in source_ids]
                numerator = 0
                denominator = 0
                for source_id in sampled:
                    for base_row, ours_row in clusters[source_id]:
                        numerator += int(_as_bool(ours_row[field])) - int(_as_bool(base_row[field]))
                        denominator += 1
                draws.append(numerator / denominator)
            lower = _percentile(draws, 2.5)
            upper = _percentile(draws, 97.5)
            bootstrap_rows.append({
                "evaluator": evaluator,
                "metric": f"ASR@{k}",
                "N_fixed": len(all_pairs),
                "source_clusters": len(source_ids),
                "nullswap_asr": sum(b_values) / len(b_values),
                "ours_asr": sum(o_values) / len(o_values),
                "delta_ours_minus_nullswap": observed,
                "bootstrap_replicates": 2000,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "ci_excludes_zero": lower > 0 or upper < 0,
            })
            transition_rows.append({
                "evaluator": evaluator,
                "metric": f"ASR@{k}",
                "nullswap_attack_ours_defend_n": sum(b and not o for b, o in zip(b_values, o_values)),
                "nullswap_defend_ours_attack_n": sum((not b) and o for b, o in zip(b_values, o_values)),
                "both_attack_n": sum(b and o for b, o in zip(b_values, o_values)),
                "both_defend_n": sum((not b) and (not o) for b, o in zip(b_values, o_values)),
            })
    write_csv(output_dir / "cluster_bootstrap_main.csv", bootstrap_rows)
    write_json(output_dir / "cluster_bootstrap_main.json", bootstrap_rows)
    write_csv(output_dir / "paired_transitions_main.csv", transition_rows)
    return bootstrap_rows, transition_rows, warnings


def _baseline_manifests(main_dir: Path, final_dir: Path) -> dict[str, list[dict]]:
    main_rows = _load_manifest(main_dir)
    nullswap = [row for row in main_rows if row["method"] == "nullswap" and row["status"] == "OK"]
    ours = _read_csv(final_dir / "protections" / "manifest.csv")
    return {"nullswap": nullswap, "ours": ours}


def run(
    *,
    main_output: str,
    final_output: str,
    output: str,
    four_evaluator_output: str | None = None,
    source_limit: int | None = None,
    stage: str = "all",
) -> dict:
    main_dir = Path(main_output)
    final_dir = Path(final_output)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    four_eval_dir = Path(four_evaluator_output) if four_evaluator_output else None
    cfg, protocol, floor_cfg, pairs, eligible = _validate_and_freeze(
        main_dir, final_dir, output_dir, four_eval_dir, source_limit
    )
    if stage == "preflight":
        return {
            "status": "PREFLIGHT_PASS",
            "sources": len(pairs),
            "eligible": {model: len(values) for model, values in eligible.items()},
        }

    method_manifests = generate_protections(
        cfg=cfg,
        protocol=protocol,
        floor_cfg=floor_cfg,
        main_dir=main_dir,
        output_dir=output_dir,
        source_limit=source_limit,
    )
    if stage == "protections":
        return {"status": "PROTECTIONS_COMPLETE", "sources": len(pairs)}

    known_rows = evaluate_methods(
        cfg=cfg,
        protocol=protocol,
        main_dir=main_dir,
        output_dir=output_dir,
        pairs=pairs,
        eligible=eligible,
        method_manifests=method_manifests,
        methods=ALL_KNOWN_METHODS,
        environments=KNOWN_ENVIRONMENTS,
        result_name="known_conditions_per_pair.jsonl",
    )
    strategy_table, strategy_robust = _aggregate(
        known_rows, STRATEGIES, tuple(KNOWN_ENVIRONMENTS)
    )
    gate_table, gate_robust = _aggregate(
        known_rows, GATE_TABLE_METHODS, tuple(KNOWN_ENVIRONMENTS)
    )
    write_csv(output_dir / "dynamic_red_strategy_ablation.csv", strategy_table)
    write_csv(output_dir / "dynamic_red_strategy_robust_all.csv", strategy_robust)
    write_csv(output_dir / "reliability_gate_ablation.csv", gate_table)
    write_csv(output_dir / "reliability_gate_robust_all.csv", gate_robust)

    baseline_manifests = _baseline_manifests(main_dir, final_dir)
    source_ids = {str(row["source_identity"]) for row in pairs}
    baseline_manifests = {
        method: [row for row in rows if str(row["source_identity"]) in source_ids]
        for method, rows in baseline_manifests.items()
    }
    unseen_rows = evaluate_methods(
        cfg=cfg,
        protocol=protocol,
        main_dir=main_dir,
        output_dir=output_dir,
        pairs=pairs,
        eligible=eligible,
        method_manifests=baseline_manifests,
        methods=("nullswap", "ours"),
        environments=UNSEEN_ENVIRONMENTS,
        result_name="unseen_conditions_per_pair.jsonl",
    )
    unseen_table, unseen_robust = _aggregate(
        unseen_rows, ("nullswap", "ours"), tuple(UNSEEN_ENVIRONMENTS)
    )
    write_csv(output_dir / "unseen_propagation_table.csv", unseen_table)
    write_csv(output_dir / "unseen_propagation_robust_all.csv", unseen_robust)

    protection_summary = _protection_summary(method_manifests)
    write_csv(output_dir / "selection_gate_runtime_audit.csv", protection_summary)
    bootstrap_rows, transitions, warnings = cluster_bootstrap(
        main_dir=main_dir,
        final_dir=final_dir,
        four_eval_dir=four_eval_dir,
        output_dir=output_dir,
    )
    summary = {
        "status": "COMPLETE",
        "experiment": "reviewer_decisive_experiments_v04",
        "source_count": len(pairs),
        "fixed_target0_denominators": {model: len(values) for model, values in eligible.items()},
        "fixed_target0_micro": sum(len(values) for values in eligible.values()),
        "protection_summary": protection_summary,
        "known_evaluation_rows": len(known_rows),
        "unseen_evaluation_rows": len(unseen_rows),
        "bootstrap_rows": len(bootstrap_rows),
        "transition_rows": len(transitions),
        "warnings": warnings,
        "outputs": [
            "dynamic_red_strategy_ablation.csv",
            "dynamic_red_strategy_robust_all.csv",
            "reliability_gate_ablation.csv",
            "reliability_gate_robust_all.csv",
            "unseen_propagation_table.csv",
            "unseen_propagation_robust_all.csv",
            "cluster_bootstrap_main.csv",
            "paired_transitions_main.csv",
            "selection_gate_runtime_audit.csv",
        ],
    }
    write_json(output_dir / "summary.json", summary)
    return summary

