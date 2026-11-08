from __future__ import annotations

"""Protocol-aligned Blue-only versus Blue+Red ablation.

Both variants start from the same frozen NullSwap image and use the same
residual search.  Blue-only validates candidates on Clean.  Blue+Red is the
already completed final Ours run, which validates the same candidate family on
Clean/JPEG/Blur/OSN.  VGGFace and SFace are held out from selection.
"""

from collections import Counter, defaultdict
from pathlib import Path
import csv
import gc
import sys

import torch
import torch.nn.functional as F

from .common import (
    append_jsonl,
    load_rgb01,
    read_json,
    read_jsonl,
    save_rgb01,
    write_csv,
    write_json,
)
from .evaluators import DetectionGate, HeldOutEvaluators, build_gallery, rank_hits
from .nullswap_floor_refinement import RefinementConfig, RobustNullSwapRefiner
from .swappers import (
    ENGINEERING_ERROR,
    SOURCE_DETECTION_FAILED,
    SWAP_OK,
    make_swapper,
    release_swapper,
)


METHODS = ("blue_only", "blue_plus_red")
HELD_OUT = ("vggface", "sface")
ENVIRONMENTS = {
    "jpeg_q60": ("jpeg", (60,)),
    "blur_sigma_1.0": ("blur", (1.0,)),
    "osn_QW": ("osn", (75, 80)),
}
ENV_ORDER = ("clean", *ENVIRONMENTS.keys())


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _read_csv(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_files(paths):
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing required file(s):\n" + "\n".join(missing))


def _target_zero_pairs(protocol):
    pairs = [row for row in protocol["pairs"] if int(row["target_index"]) == 0]
    expected = len(protocol["source_identities"])
    if len(pairs) != expected:
        raise RuntimeError(f"Expected {expected} target-index-0 pairs, got {len(pairs)}")
    return pairs


def _load_main_manifest(main_dir):
    path = main_dir / "protections" / "manifest.jsonl"
    if path.is_file():
        return read_jsonl(path)
    return read_json(main_dir / "protections" / "manifest.json")


def _generate_blue_only(base_cfg, floor_cfg, protocol, main_dir, output_dir):
    manifest_path = output_dir / "protections" / "blue_only_manifest.jsonl"
    previous = read_jsonl(manifest_path)
    done = {str(row["source_identity"]) for row in previous}
    if len(done) != len(previous):
        raise RuntimeError("Blue-only manifest contains duplicate sources.")

    baseline = {
        (str(row["source_identity"]), row["method"]): row
        for row in _load_main_manifest(main_dir)
    }
    raw = dict(floor_cfg["refinement"])
    original_environments = tuple(raw["environments"])
    if original_environments != tuple(ENV_ORDER):
        raise RuntimeError(
            f"Unexpected final Ours environments: {original_environments}"
        )
    raw["environments"] = ["clean"]
    clean_only_cfg = RefinementConfig.from_dict(raw)
    refiner = RobustNullSwapRefiner(
        deps_root=base_cfg["deps_root"],
        device=base_cfg["device"],
        config=clean_only_cfg,
    )

    total = len(protocol["source_identities"])
    for index, source_identity in enumerate(protocol["source_identities"], 1):
        sid = str(source_identity)
        if sid in done:
            continue
        anchor_row = baseline.get((sid, "nullswap"))
        if (
            not anchor_row
            or anchor_row.get("status") != "OK"
            or not anchor_row.get("protected_path")
        ):
            raise RuntimeError(f"Missing frozen NullSwap anchor for source={sid}")
        clean = load_rgb01(protocol["source_paths"][sid], 256).to(base_cfg["device"])
        anchor = load_rgb01(anchor_row["protected_path"], 256).to(base_cfg["device"])
        result = refiner.refine(clean=clean, nullswap_anchor=anchor)
        output_path = output_dir / "protections" / "blue_only" / f"{sid}.png"
        save_rgb01(result["protected"], output_path)
        selected = result["selection"]
        visual = selected["visual"]
        row = {
            "source_identity": int(source_identity),
            "source_path": protocol["source_paths"][sid],
            "method": "blue_only",
            "status": "OK",
            "protected_path": str(output_path),
            "route": (
                "nullswap_floor_clean_refined"
                if result["selected_refinement"]
                else "nullswap_exact_fallback"
            ),
            "selected_refinement": bool(result["selected_refinement"]),
            "psnr": visual["psnr"],
            "ssim": visual["ssim"],
            "lpips": visual["lpips"],
            "anchor_psnr": result["anchor_visual"]["psnr"],
            "anchor_ssim": result["anchor_visual"]["ssim"],
            "anchor_lpips": result["anchor_visual"]["lpips"],
            "anchor_worst_similarity": result["anchor_worst_similarity"],
            "selected_worst_similarity": selected["worst_similarity"],
            "selected_step": selected["step"],
            "selected_residual_scale": selected["residual_scale"],
            "selected_toward_clean": selected["toward_clean"],
            "candidates_evaluated": result["candidates_evaluated"],
            "accepted_candidate_count": result["accepted_candidate_count"],
            "selection_checks": selected["checks"],
            "selection_environments": ["clean"],
            "held_out_evaluators_used_for_selection": False,
        }
        append_jsonl(manifest_path, row)
        previous.append(row)
        done.add(sid)
        if index % 10 == 0 or index == total:
            selected_count = sum(_as_bool(x["selected_refinement"]) for x in previous)
            print(
                f"ALIGNED_BLUE_PROTECTION sources={index}/{total} "
                f"selected={selected_count}",
                flush=True,
            )
        del clean, anchor, result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    by_sid = {str(row["source_identity"]): row for row in read_jsonl(manifest_path)}
    if set(by_sid) != {str(x) for x in protocol["source_identities"]}:
        raise RuntimeError("Blue-only manifest key set is incomplete.")
    rows = [by_sid[str(x)] for x in protocol["source_identities"]]
    write_json(output_dir / "protections" / "blue_only_manifest.json", rows)
    write_csv(output_dir / "protections" / "blue_only_manifest.csv", rows)
    del refiner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def _materialize_transforms(base_cfg, output_dir, protections):
    project_src = str(Path(base_cfg["project_root"]) / "src")
    if project_src not in sys.path:
        sys.path.insert(0, project_src)
    from reliable_face_defense_agents.red.contracts import RedTransformSpec
    from reliable_face_defense_agents.red.transforms import apply_red_transform

    transformed = {}
    for row in protections:
        sid = str(row["source_identity"])
        for env_name, (family, params) in ENVIRONMENTS.items():
            path = output_dir / "transformed" / env_name / f"{sid}.png"
            if not path.is_file():
                image = load_rgb01(row["protected_path"], 256)
                transformed_image = apply_red_transform(
                    image, RedTransformSpec(env_name, family, tuple(params))
                )
                save_rgb01(transformed_image, path)
            transformed[(sid, env_name)] = str(path)
    return transformed


def _error_row(base, status, error):
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


def _flush(batch, gate, evaluators, gallery, path, done):
    if not batch:
        return
    images = torch.cat([
        F.interpolate(
            item["image"].detach().cpu(),
            (256, 256),
            mode="bilinear",
            align_corners=False,
        )
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
            row[f"{evaluator}_acc1"] = (
                bool(hits[evaluator]["acc1"][index]) if detection_success else False
            )
            row[f"{evaluator}_acc5"] = (
                bool(hits[evaluator]["acc5"][index]) if detection_success else False
            )
        append_jsonl(path, row)
        done.add((row["swap_model"], row["environment"], row["pair_id"]))
    batch.clear()


def _run_blue_only(base_cfg, protocol, main_dir, output_dir, pairs, protections, transformed):
    path = output_dir / "blue_only_per_pair.jsonl"
    previous = read_jsonl(path)
    done = {
        (row["swap_model"], row["environment"], row["pair_id"])
        for row in previous
    }
    if len(done) != len(previous):
        raise RuntimeError("Blue-only resume file contains duplicate keys.")
    by_sid = {str(row["source_identity"]): row for row in protections}

    gate = DetectionGate(base_cfg["deps_root"], base_cfg["device"])
    gate.calibrate(
        protocol["calibration_paths"],
        target_tpr=float(base_cfg["detection_clean_tpr"]),
        cache_path=main_dir / "detection_gate.json",
    )
    evaluators = HeldOutEvaluators(base_cfg, base_cfg["device"])
    gallery = build_gallery(
        evaluators, protocol["gallery"], main_dir / "gallery_embeddings.pt"
    )
    total = len(base_cfg["swap_models"]) * len(ENV_ORDER) * len(pairs)
    print(f"ALIGNED_BLUE_SWAP_TOTAL={total}", flush=True)
    for model_name in base_cfg["swap_models"]:
        swapper = make_swapper(model_name, base_cfg, base_cfg["device"])
        batch = []
        try:
            for environment in ENV_ORDER:
                for pair in pairs:
                    key = (model_name, environment, pair["pair_id"])
                    if key in done:
                        continue
                    sid = str(pair["source_identity"])
                    source_path = (
                        by_sid[sid]["protected_path"]
                        if environment == "clean"
                        else transformed[(sid, environment)]
                    )
                    base = {
                        "swap_model": model_name,
                        "environment": environment,
                        "method": "blue_only",
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
                            row = _error_row(base, output.status, output.error)
                        append_jsonl(path, row)
                        done.add(key)
                        continue
                    batch.append({
                        "base": base,
                        "source_identity": pair["source_identity"],
                        "image": output.image01,
                    })
                    if len(batch) >= 12:
                        _flush(batch, gate, evaluators, gallery, path, done)
                _flush(batch, gate, evaluators, gallery, path, done)
                print(
                    f"ALIGNED_BLUE_SWAP_PROGRESS model={model_name} "
                    f"env={environment} completed={len(done)}/{total}",
                    flush=True,
                )
        finally:
            release_swapper(swapper)

    rows = read_jsonl(path)
    expected_keys = {
        (model, environment, pair["pair_id"])
        for model in base_cfg["swap_models"]
        for environment in ENV_ORDER
        for pair in pairs
    }
    actual_keys = {
        (row["swap_model"], row["environment"], row["pair_id"])
        for row in rows
    }
    if len(rows) != total or actual_keys != expected_keys:
        raise RuntimeError(
            f"Incomplete Blue-only results rows={len(rows)}/{total} "
            f"missing={len(expected_keys - actual_keys)} extra={len(actual_keys - expected_keys)}"
        )
    write_json(output_dir / "blue_only_per_pair.json", rows)
    write_csv(output_dir / "blue_only_per_pair.csv", rows)
    return rows


def _index_rows(rows, method, environment_field=True):
    index = {}
    for row in rows:
        environment = row["environment"] if environment_field else "clean"
        key = (row["swap_model"], method, environment, row["pair_id"])
        if key in index:
            raise RuntimeError(f"Duplicate result key: {key}")
        index[key] = row
    return index


def _aggregate(rows, evaluator):
    n = len(rows)
    detected = sum(_as_bool(row["detection_success"]) for row in rows)
    output = {
        "N_fixed_target0": n,
        "detection_success_n": detected,
        "detection_success_rate": detected / n,
        "engineering_error_n": sum(row["status"] == ENGINEERING_ERROR for row in rows),
    }
    for k in (1, 5):
        attacks = sum(_as_bool(row[f"{evaluator}_acc{k}"]) for row in rows)
        conditional = sum(
            _as_bool(row["detection_success"])
            and _as_bool(row[f"{evaluator}_acc{k}"])
            for row in rows
        )
        output[f"attack_success_n_at{k}"] = attacks
        output[f"asr_at{k}"] = attacks / n
        output[f"defsr_at{k}"] = 1.0 - attacks / n
        output[f"conditional_identity_success_n_at{k}"] = conditional
        output[f"conditional_identity_rate_at{k}"] = (
            conditional / detected if detected else None
        )
    return output


def _summarize(base_cfg, protocol, main_dir, final_dir, output_dir, pairs, blue_rows, blue_manifest):
    main_raw = read_jsonl(main_dir / "results" / "per_pair.jsonl")
    eligible = {}
    pair_ids = {row["pair_id"] for row in pairs}
    for model in base_cfg["swap_models"]:
        eligible[model] = sorted({
            row["pair_id"]
            for row in main_raw
            if row["pair_id"] in pair_ids
            and row["swap_model"] == model
            and row["method"] == "no_protection"
            and row["status"] == SWAP_OK
            and row["detection_success"] is True
        })

    blue_index = _index_rows(blue_rows, "blue_only", True)
    red_clean = _read_csv(final_dir / "main" / "per_pair_four_evaluators.csv")
    red_clean = [row for row in red_clean if row["method"] == "ours" and row["pair_id"] in pair_ids]
    red_robust = _read_csv(final_dir / "robustness" / "evaluation_heldout.csv")
    red_robust = [row for row in red_robust if row["method"] == "ours"]
    red_index = _index_rows(red_clean, "blue_plus_red", False)
    red_index.update(_index_rows(red_robust, "blue_plus_red", True))
    combined = {**blue_index, **red_index}

    table = []
    paired = []
    for scope in (*base_cfg["swap_models"], "micro_3_models"):
        scoped_pairs = []
        models = base_cfg["swap_models"] if scope == "micro_3_models" else (scope,)
        for model in models:
            scoped_pairs.extend((model, pair_id) for pair_id in eligible[model])
        for evaluator in HELD_OUT:
            for environment in ENV_ORDER:
                method_rows = {}
                for method in METHODS:
                    rows = [
                        combined[(model, method, environment, pair_id)]
                        for model, pair_id in scoped_pairs
                    ]
                    method_rows[method] = rows
                    table.append({
                        "swap_model": scope,
                        "evaluator": evaluator,
                        "environment": environment,
                        "method": method,
                        **_aggregate(rows, evaluator),
                    })
                for k in (1, 5):
                    improve = regress = both_attack = both_defend = 0
                    for blue, red in zip(method_rows["blue_only"], method_rows["blue_plus_red"]):
                        b = _as_bool(blue[f"{evaluator}_acc{k}"])
                        r = _as_bool(red[f"{evaluator}_acc{k}"])
                        improve += b and not r
                        regress += not b and r
                        both_attack += b and r
                        both_defend += (not b) and (not r)
                    paired.append({
                        "swap_model": scope,
                        "evaluator": evaluator,
                        "environment": environment,
                        "metric": f"at{k}",
                        "N_fixed_target0": len(scoped_pairs),
                        "blue_attack_red_defend_n": improve,
                        "blue_defend_red_attack_n": regress,
                        "both_attack_n": both_attack,
                        "both_defend_n": both_defend,
                        "net_improved_pairs": improve - regress,
                    })

    robust_all = []
    for scope in (*base_cfg["swap_models"], "micro_3_models"):
        models = base_cfg["swap_models"] if scope == "micro_3_models" else (scope,)
        scoped_pairs = [
            (model, pair_id)
            for model in models
            for pair_id in eligible[model]
        ]
        for evaluator in HELD_OUT:
            for k in (1, 5):
                for method in METHODS:
                    reliable = 0
                    for model, pair_id in scoped_pairs:
                        attacks = [
                            _as_bool(combined[(model, method, env, pair_id)][f"{evaluator}_acc{k}"])
                            for env in ENV_ORDER
                        ]
                        reliable += not any(attacks)
                    robust_all.append({
                        "swap_model": scope,
                        "evaluator": evaluator,
                        "metric": f"at{k}",
                        "method": method,
                        "N_fixed_target0": len(scoped_pairs),
                        "reliable_all_n": reliable,
                        "robust_all_pool_rate": reliable / len(scoped_pairs),
                        "attack_success_any_environment_n": len(scoped_pairs) - reliable,
                    })

    final_manifest = _read_csv(final_dir / "protections" / "manifest.csv")
    expected_new_rows = (
        len(base_cfg["swap_models"]) * len(ENV_ORDER) * len(pairs)
    )
    write_csv(output_dir / "aligned_blue_red_table.csv", table)
    write_json(output_dir / "aligned_blue_red_table.json", table)
    write_csv(output_dir / "paired_transition_counts.csv", paired)
    write_csv(output_dir / "robust_all_pool.csv", robust_all)
    summary = {
        "status": "COMPLETE",
        "experiment": "protocol-aligned Blue-only versus Blue+Red ablation",
        "definition": {
            "blue_only": "same NullSwap anchor and residual search; candidate reliability checked on Clean only",
            "blue_plus_red": "completed final Ours; same anchor/search with Clean/JPEG/Blur/OSN constraints",
        },
        "held_out_evaluators_used_for_selection": False,
        "source_count": len(protocol["source_identities"]),
        "new_blue_only_swap_rows": len(blue_rows),
        "expected_new_blue_only_swap_rows": expected_new_rows,
        "fixed_target0_pairs": {model: len(ids) for model, ids in eligible.items()},
        "blue_only_selection": dict(Counter(
            "refined" if _as_bool(row["selected_refinement"]) else "fallback"
            for row in blue_manifest
        )),
        "blue_plus_red_selection": dict(Counter(
            "refined" if _as_bool(row["selected_refinement"]) else "fallback"
            for row in final_manifest
        )),
        "blue_only_status_counts": dict(Counter(row["status"] for row in blue_rows)),
        "files": {
            "main_table": str(output_dir / "aligned_blue_red_table.csv"),
            "paired_counts": str(output_dir / "paired_transition_counts.csv"),
            "robust_all": str(output_dir / "robust_all_pool.csv"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "protocol.json", {
        "pair_selection": "frozen target_index == 0; one pair per source",
        "fixed_denominator": "frozen clean no-protection SWAP_OK and detection-success pairs",
        "environments": list(ENV_ORDER),
        "evaluators": list(HELD_OUT),
        "aggregation": "micro aggregation by event counts; no equal-weight macro average",
        "error_policy": {
            "source_detection_failure": "attack failure",
            "target_or_engineering_failure": "conservative attack success",
        },
    })
    return summary


def run(main_output, final_output, ablation_output):
    main_dir = Path(main_output)
    final_dir = Path(final_output)
    output_dir = Path(ablation_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    required = [
        main_dir / "frozen_main_config.json",
        main_dir / "protocol_manifest.json",
        main_dir / "results" / "per_pair.jsonl",
        final_dir / "frozen_nullswap_floor_config.json",
        final_dir / "protections" / "manifest.csv",
        final_dir / "main" / "per_pair_four_evaluators.csv",
        final_dir / "robustness" / "evaluation_heldout.csv",
    ]
    _validate_files(required)
    base_cfg = read_json(main_dir / "frozen_main_config.json")
    protocol = read_json(main_dir / "protocol_manifest.json")
    frozen_floor = read_json(final_dir / "frozen_nullswap_floor_config.json")
    floor_cfg = frozen_floor["config"]
    if base_cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the frozen experiment config.")
    if len(protocol["source_identities"]) != 200:
        raise RuntimeError("Expected the frozen 200-source protocol.")

    run_config = {
        "main_output": str(main_dir),
        "final_output": str(final_dir),
        "ablation_output": str(output_dir),
        "blue_only_environments": ["clean"],
        "blue_plus_red_environments": list(ENV_ORDER),
        "refinement_except_environments": {
            key: value for key, value in floor_cfg["refinement"].items()
            if key != "environments"
        },
    }
    frozen_path = output_dir / "frozen_aligned_ablation_config.json"
    if frozen_path.is_file() and read_json(frozen_path) != run_config:
        raise RuntimeError("Output directory contains a different frozen config.")
    write_json(frozen_path, run_config)

    pairs = _target_zero_pairs(protocol)
    blue_manifest = _generate_blue_only(
        base_cfg, floor_cfg, protocol, main_dir, output_dir
    )
    transformed = _materialize_transforms(base_cfg, output_dir, blue_manifest)
    blue_rows = _run_blue_only(
        base_cfg,
        protocol,
        main_dir,
        output_dir,
        pairs,
        blue_manifest,
        transformed,
    )
    summary = _summarize(
        base_cfg,
        protocol,
        main_dir,
        final_dir,
        output_dir,
        pairs,
        blue_rows,
        blue_manifest,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary
