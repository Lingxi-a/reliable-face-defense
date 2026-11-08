from __future__ import annotations

from collections import defaultdict
from pathlib import Path
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
from .four_evaluator_supplement import ArcFaceFaceNetEvaluators
from .nullswap_floor_refinement import (
    RefinementConfig,
    RobustNullSwapRefiner,
)
from .swappers import (
    SOURCE_DETECTION_FAILED,
    SWAP_OK,
    make_swapper,
    release_swapper,
)


EVALUATORS = ("arcface", "facenet", "vggface", "sface")
HELD_OUT = ("vggface", "sface")
INTERNAL = ("arcface", "facenet")
ROBUST_ENVIRONMENTS = {
    "jpeg_q60": ("jpeg", (60,)),
    "blur_sigma_1.0": ("blur", (1.0,)),
    "osn_QW": ("osn", (75, 80)),
}


def _mean(values):
    values = [float(x) for x in values if x is not None]
    return sum(values) / len(values) if values else None


def _manifest(main_dir: Path) -> list[dict]:
    jsonl = main_dir / "protections" / "manifest.jsonl"
    if jsonl.is_file():
        return read_jsonl(jsonl)
    return read_json(main_dir / "protections" / "manifest.json")


def _filtered_protocol(protocol: dict, limit: int | None) -> dict:
    if limit is None:
        return protocol
    source_ids = list(protocol["source_identities"])[: int(limit)]
    selected = set(int(x) for x in source_ids)
    return {
        **protocol,
        "source_identities": source_ids,
        "source_paths": {
            str(key): value
            for key, value in protocol["source_paths"].items()
            if int(key) in selected
        },
        "pairs": [
            row for row in protocol["pairs"]
            if int(row["source_identity"]) in selected
        ],
        "pair_count": sum(
            int(row["source_identity"]) in selected
            for row in protocol["pairs"]
        ),
        "run_scope": f"first_{len(source_ids)}_frozen_source_identities",
    }


def _prepare(cfg: dict, limit: int | None):
    main_dir = Path(cfg["baseline_main_dir"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = read_json(main_dir / "frozen_main_config.json")
    protocol = _filtered_protocol(
        read_json(main_dir / "protocol_manifest.json"),
        limit,
    )
    project_src = str(Path(base_cfg["project_root"]) / "src")
    if project_src not in sys.path:
        sys.path.insert(0, project_src)
    if base_cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal rerun")
    frozen_path = output_dir / "frozen_nullswap_floor_config.json"
    frozen_payload = {"config": cfg, "source_limit": limit}
    if frozen_path.is_file() and read_json(frozen_path) != frozen_payload:
        raise RuntimeError(
            "Output directory already contains a different frozen rerun "
            "configuration. Use a new output directory."
        )
    protocol_path = output_dir / "protocol_manifest.json"
    if protocol_path.is_file() and read_json(protocol_path) != protocol:
        raise RuntimeError(
            "Output directory already contains a different protocol subset. "
            "Use a new output directory."
        )
    write_json(frozen_path, frozen_payload)
    write_json(protocol_path, protocol)
    return main_dir, output_dir, base_cfg, protocol


def generate_refined_protections(
    cfg: dict,
    *,
    main_dir: Path,
    output_dir: Path,
    base_cfg: dict,
    protocol: dict,
) -> list[dict]:
    manifest_path = output_dir / "protections" / "manifest.jsonl"
    rows = read_jsonl(manifest_path)
    done = {str(row["source_identity"]) for row in rows}
    baseline_by_key = {
        (str(row["source_identity"]), row["method"]): row
        for row in _manifest(main_dir)
    }

    refine_cfg = RefinementConfig.from_dict(cfg["refinement"])
    refiner = RobustNullSwapRefiner(
        deps_root=base_cfg["deps_root"],
        device=base_cfg["device"],
        config=refine_cfg,
    )

    total = len(protocol["source_identities"])
    for index, source_identity in enumerate(protocol["source_identities"], 1):
        sid = str(source_identity)
        if sid in done:
            continue
        anchor_row = baseline_by_key.get((sid, "nullswap"))
        if (
            not anchor_row
            or anchor_row.get("status") != "OK"
            or not anchor_row.get("protected_path")
        ):
            raise RuntimeError(f"Missing NullSwap anchor for source {sid}")

        clean = load_rgb01(protocol["source_paths"][sid], 256).to(
            base_cfg["device"]
        )
        anchor = load_rgb01(anchor_row["protected_path"], 256).to(
            base_cfg["device"]
        )
        result = refiner.refine(clean=clean, nullswap_anchor=anchor)
        output_path = output_dir / "protections" / "ours" / f"{sid}.png"
        save_rgb01(result["protected"], output_path)

        selection = result["selection"]
        visual = selection["visual"]
        row = {
            "source_identity": int(source_identity),
            "source_path": protocol["source_paths"][sid],
            "method": "ours",
            "status": "OK",
            "protected_path": str(output_path),
            "route": (
                "nullswap_floor_refined"
                if result["selected_refinement"]
                else "nullswap_exact_fallback"
            ),
            "selected_refinement": result["selected_refinement"],
            "psnr": visual["psnr"],
            "ssim": visual["ssim"],
            "lpips": visual["lpips"],
            "linf_px": visual["linf_px"],
            "anchor_psnr": result["anchor_visual"]["psnr"],
            "anchor_ssim": result["anchor_visual"]["ssim"],
            "anchor_lpips": result["anchor_visual"]["lpips"],
            "anchor_worst_similarity": result["anchor_worst_similarity"],
            "selected_worst_similarity": selection["worst_similarity"],
            "worst_similarity_gain": (
                result["anchor_worst_similarity"]
                - selection["worst_similarity"]
            ),
            "selected_step": selection["step"],
            "selected_residual_scale": selection["residual_scale"],
            "selected_toward_clean": selection["toward_clean"],
            "candidates_evaluated": result["candidates_evaluated"],
            "accepted_candidate_count": result["accepted_candidate_count"],
            "anchor_signature": result["anchor_signature"],
            "selected_signature": selection["signature"],
            "selection_checks": selection["checks"],
        }
        append_jsonl(manifest_path, row)
        rows.append(row)
        done.add(sid)

        if index % 10 == 0 or index == total:
            selected_count = sum(
                bool(x.get("selected_refinement")) for x in rows
            )
            print(
                "NULLSWAP_FLOOR_PROGRESS "
                f"sources={index}/{total} selected={selected_count}",
                flush=True,
            )
        del clean, anchor, result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ordered = {
        str(row["source_identity"]): row for row in read_jsonl(manifest_path)
    }
    rows = [ordered[str(x)] for x in protocol["source_identities"]]
    write_json(output_dir / "protections" / "manifest.json", rows)
    write_csv(output_dir / "protections" / "manifest.csv", rows)
    summary = {
        "status": "COMPLETE",
        "source_count": len(rows),
        "selected_refinement_count": sum(
            bool(row["selected_refinement"]) for row in rows
        ),
        "exact_nullswap_fallback_count": sum(
            not bool(row["selected_refinement"]) for row in rows
        ),
        "selection_rate": _mean([
            bool(row["selected_refinement"]) for row in rows
        ]),
        "nullswap_anchor": "frozen baseline NullSwap PNG per source",
        "held_out_evaluators_used_for_selection": False,
        "visual_average": {
            "psnr": _mean([row["psnr"] for row in rows]),
            "ssim": _mean([row["ssim"] for row in rows]),
            "lpips": _mean([row["lpips"] for row in rows]),
        },
        "anchor_visual_average": {
            "psnr": _mean([row["anchor_psnr"] for row in rows]),
            "ssim": _mean([row["anchor_ssim"] for row in rows]),
            "lpips": _mean([row["anchor_lpips"] for row in rows]),
        },
        "mean_worst_similarity_gain": _mean([
            row["worst_similarity_gain"] for row in rows
        ]),
    }
    write_json(output_dir / "protections" / "summary.json", summary)
    return rows


def _swap_key(row: dict) -> tuple:
    return row["swap_model"], row.get("environment"), row["pair_id"]


def _run_swap_cache(
    cfg: dict,
    *,
    base_cfg: dict,
    protocol: dict,
    protections: list[dict],
    output_dir: Path,
) -> list[dict]:
    cache_dir = output_dir / "main" / "swap_cache"
    jsonl = output_dir / "main" / "swap_cache_manifest.jsonl"
    rows = read_jsonl(jsonl)
    done = {_swap_key(row) for row in rows}
    protection_by_sid = {
        str(row["source_identity"]): row for row in protections
    }
    total = len(base_cfg["swap_models"]) * len(protocol["pairs"])

    for model_name in base_cfg["swap_models"]:
        model_keys = {
            (model_name, None, pair["pair_id"])
            for pair in protocol["pairs"]
        }
        if model_keys <= done:
            continue
        swapper = make_swapper(model_name, base_cfg, base_cfg["device"])
        for pair in protocol["pairs"]:
            key = (model_name, None, pair["pair_id"])
            if key in done:
                continue
            protected = protection_by_sid[str(pair["source_identity"])]
            output = swapper.swap(
                protected["protected_path"],
                pair["target_path"],
            )
            image_path = None
            if output.status == SWAP_OK and output.image01 is not None:
                image_path = cache_dir / model_name / f"{pair['pair_id']}.png"
                save_rgb01(
                    F.interpolate(
                        output.image01,
                        (256, 256),
                        mode="bilinear",
                        align_corners=False,
                    ),
                    image_path,
                )
            row = {
                "swap_model": model_name,
                "method": "ours",
                "pair_id": pair["pair_id"],
                "source_identity": int(pair["source_identity"]),
                "source_path": pair["source_path"],
                "target_identity": int(pair["target_identity"]),
                "target_path": pair["target_path"],
                "status": output.status,
                "error": output.error,
                "swap_image_path": None if image_path is None else str(image_path),
            }
            append_jsonl(jsonl, row)
            rows.append(row)
            done.add(key)
            if len(done) % 100 == 0 or len(done) == total:
                print(
                    f"NULLSWAP_FLOOR_SWAP_PROGRESS={len(done)}/{total}",
                    flush=True,
                )
        release_swapper(swapper)

    rows = read_jsonl(jsonl)
    if len({_swap_key(row) for row in rows}) != total:
        raise RuntimeError(f"Incomplete main swap cache: {len(rows)} != {total}")
    write_json(output_dir / "main" / "swap_cache_manifest.json", rows)
    write_csv(output_dir / "main" / "swap_cache_manifest.csv", rows)
    return rows


def _build_internal_gallery(evaluators, gallery, cache_path, batch_size=12):
    cache_path = Path(cache_path)
    if cache_path.is_file():
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    identities = sorted(int(x) for x in gallery)
    embeddings = {name: [] for name in INTERNAL}
    for start in range(0, len(identities), batch_size):
        ids = identities[start:start + batch_size]
        images = torch.cat([
            load_rgb01(gallery[str(identity)], 256) for identity in ids
        ])
        values = evaluators.embed(images)
        for name in embeddings:
            embeddings[name].append(values[name])
    payload = {
        "identities": identities,
        "embeddings": {
            name: F.normalize(torch.cat(parts).float(), dim=1)
            for name, parts in embeddings.items()
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def _error_metrics(names: tuple[str, ...], value: bool) -> dict:
    return {
        f"{name}_acc{k}": bool(value)
        for name in names for k in (1, 5)
    }


def _evaluate_swap_cache_group(
    *,
    group: str,
    base_cfg: dict,
    protocol: dict,
    cache_rows: list[dict],
    baseline_main_dir: Path,
    output_dir: Path,
    output_prefix: str = "main",
) -> list[dict]:
    if group not in {"heldout", "internal"}:
        raise ValueError(group)
    names = HELD_OUT if group == "heldout" else INTERNAL
    jsonl = output_dir / output_prefix / f"evaluation_{group}.jsonl"
    previous = read_jsonl(jsonl)
    done = {_swap_key(row) for row in previous}

    gate = DetectionGate(base_cfg["deps_root"], base_cfg["device"])
    gate.calibrate(
        protocol["calibration_paths"],
        target_tpr=float(base_cfg["detection_clean_tpr"]),
        cache_path=baseline_main_dir / "detection_gate.json",
    )
    if group == "heldout":
        evaluators = HeldOutEvaluators(base_cfg, base_cfg["device"])
        gallery = build_gallery(
            evaluators,
            protocol["gallery"],
            baseline_main_dir / "gallery_embeddings.pt",
        )
    else:
        evaluators = ArcFaceFaceNetEvaluators(base_cfg, base_cfg["device"])
        gallery = _build_internal_gallery(
            evaluators,
            protocol["gallery"],
            baseline_main_dir / "four_evaluator_supplement"
            / "gallery_embeddings_arcface_facenet.pt",
        )

    batch = []

    def flush():
        if not batch:
            return
        images = torch.cat([
            load_rgb01(item["swap_image_path"], 256) for item in batch
        ])
        detected, scores = gate.success(images)
        query = evaluators.embed(images)
        hits = rank_hits(
            query,
            gallery,
            [item["source_identity"] for item in batch],
        )
        for index, item in enumerate(batch):
            detection_success = bool(detected[index])
            row = {
                **item,
                "detection_success": detection_success,
                "detection_score": float(scores[index]),
            }
            for name in names:
                row[f"{name}_acc1"] = (
                    bool(hits[name]["acc1"][index])
                    if detection_success else False
                )
                row[f"{name}_acc5"] = (
                    bool(hits[name]["acc5"][index])
                    if detection_success else False
                )
            append_jsonl(jsonl, row)
            done.add(_swap_key(row))
        batch.clear()

    for item in cache_rows:
        if _swap_key(item) in done:
            continue
        if item["status"] == SWAP_OK and item.get("swap_image_path"):
            batch.append(item)
            if len(batch) >= 12:
                flush()
            continue
        flush()
        if item["status"] == SOURCE_DETECTION_FAILED:
            detected = False
            metrics = _error_metrics(names, False)
        else:
            detected = None
            metrics = _error_metrics(names, True)
        row = {
            **item,
            "detection_success": detected,
            "detection_score": None,
            **metrics,
        }
        append_jsonl(jsonl, row)
        done.add(_swap_key(row))
    flush()

    del evaluators, gate, gallery
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    rows = read_jsonl(jsonl)
    write_json(output_dir / output_prefix / f"evaluation_{group}.json", rows)
    write_csv(output_dir / output_prefix / f"evaluation_{group}.csv", rows)
    return rows


def _merge_main_evaluations(
    heldout_rows: list[dict],
    internal_rows: list[dict],
    output_dir: Path,
) -> list[dict]:
    heldout = {_swap_key(row): row for row in heldout_rows}
    internal = {_swap_key(row): row for row in internal_rows}
    if set(heldout) != set(internal):
        raise RuntimeError("Held-out and internal evaluation key sets differ")
    merged = []
    for key in sorted(heldout):
        a, b = heldout[key], internal[key]
        row = dict(a)
        for name in INTERNAL:
            for k in (1, 5):
                row[f"{name}_acc{k}"] = b[f"{name}_acc{k}"]
        merged.append(row)
    write_json(output_dir / "main" / "per_pair_four_evaluators.json", merged)
    write_csv(output_dir / "main" / "per_pair_four_evaluators.csv", merged)
    return merged


def _eligible_pairs(
    baseline_rows: list[dict],
    protocol: dict,
) -> dict[str, set[str]]:
    allowed = {row["pair_id"] for row in protocol["pairs"]}
    return {
        model: {
            row["pair_id"] for row in baseline_rows
            if row["pair_id"] in allowed
            and row["swap_model"] == model
            and row["method"] == "no_protection"
            and row["status"] == SWAP_OK
            and row["detection_success"] is True
        }
        for model in {row["swap_model"] for row in baseline_rows}
    }


def _summarize_main(
    *,
    base_cfg: dict,
    protocol: dict,
    baseline_main_dir: Path,
    ours_rows: list[dict],
    protections: list[dict],
    output_dir: Path,
) -> list[dict]:
    baseline_original = read_jsonl(
        baseline_main_dir / "results" / "per_pair.jsonl"
    )
    baseline_four = read_jsonl(
        baseline_main_dir / "four_evaluator_supplement"
        / "per_pair_four_evaluators.jsonl"
    )
    if not baseline_four:
        raise FileNotFoundError(
            "Run the frozen four-evaluator supplement before this rerun"
        )
    allowed = {row["pair_id"] for row in protocol["pairs"]}
    baseline_nullswap = [
        row for row in baseline_four
        if row["method"] == "nullswap" and row["pair_id"] in allowed
    ]
    eligible = _eligible_pairs(baseline_original, protocol)
    grouped = defaultdict(list)
    for row in baseline_nullswap + ours_rows:
        grouped[(row["swap_model"], row["method"])].append(row)

    table = []
    for model in base_cfg["swap_models"]:
        for evaluator in EVALUATORS:
            for method in ("nullswap", "ours"):
                selected = [
                    row for row in grouped[(model, method)]
                    if row["pair_id"] in eligible[model]
                ]
                if len(selected) != len(eligible[model]):
                    raise RuntimeError(
                        f"Missing main rows: {model}/{method} "
                        f"{len(selected)} != {len(eligible[model])}"
                    )
                table.append({
                    "swap_model": model,
                    "evaluator": evaluator,
                    "method": method,
                    "denominator": len(selected),
                    "acc1": _mean([
                        row[f"{evaluator}_acc1"] for row in selected
                    ]),
                    "acc5": _mean([
                        row[f"{evaluator}_acc5"] for row in selected
                    ]),
                })
    for evaluator in EVALUATORS:
        for method in ("nullswap", "ours"):
            parts = [
                row for row in table
                if row["evaluator"] == evaluator and row["method"] == method
            ]
            table.append({
                "swap_model": "average_macro_3_models",
                "evaluator": evaluator,
                "method": method,
                "denominator": sum(row["denominator"] for row in parts),
                "acc1": _mean([row["acc1"] for row in parts]),
                "acc5": _mean([row["acc5"] for row in parts]),
            })

    visual_table = [
        {
            "method": "nullswap",
            "count": len(protections),
            "psnr": _mean([row["anchor_psnr"] for row in protections]),
            "ssim": _mean([row["anchor_ssim"] for row in protections]),
            "lpips": _mean([row["anchor_lpips"] for row in protections]),
        },
        {
            "method": "ours",
            "count": len(protections),
            "psnr": _mean([row["psnr"] for row in protections]),
            "ssim": _mean([row["ssim"] for row in protections]),
            "lpips": _mean([row["lpips"] for row in protections]),
        },
    ]
    write_json(output_dir / "main" / "main_table_vs_nullswap.json", table)
    write_csv(output_dir / "main" / "main_table_vs_nullswap.csv", table)
    write_json(output_dir / "main" / "visual_table_vs_nullswap.json", visual_table)
    write_csv(output_dir / "main" / "visual_table_vs_nullswap.csv", visual_table)
    write_json(output_dir / "main" / "eligible_pairs.json", {
        key: sorted(value) for key, value in eligible.items()
    })
    return table


def run_main_evaluation(
    cfg: dict,
    *,
    main_dir: Path,
    output_dir: Path,
    base_cfg: dict,
    protocol: dict,
    protections: list[dict],
) -> tuple[list[dict], list[dict]]:
    cache_rows = _run_swap_cache(
        cfg,
        base_cfg=base_cfg,
        protocol=protocol,
        protections=protections,
        output_dir=output_dir,
    )
    heldout = _evaluate_swap_cache_group(
        group="heldout",
        base_cfg=base_cfg,
        protocol=protocol,
        cache_rows=cache_rows,
        baseline_main_dir=main_dir,
        output_dir=output_dir,
    )
    internal = _evaluate_swap_cache_group(
        group="internal",
        base_cfg=base_cfg,
        protocol=protocol,
        cache_rows=cache_rows,
        baseline_main_dir=main_dir,
        output_dir=output_dir,
    )
    merged = _merge_main_evaluations(heldout, internal, output_dir)
    table = _summarize_main(
        base_cfg=base_cfg,
        protocol=protocol,
        baseline_main_dir=main_dir,
        ours_rows=merged,
        protections=protections,
        output_dir=output_dir,
    )
    return merged, table


def _target_zero_pairs(protocol: dict) -> list[dict]:
    return [
        row for row in protocol["pairs"] if int(row["target_index"]) == 0
    ]


def _materialize_robust_sources(
    *,
    base_cfg: dict,
    pairs: list[dict],
    protections: list[dict],
    output_dir: Path,
) -> dict[tuple[str, str], str]:
    from reliable_face_defense_agents.red.contracts import RedTransformSpec
    from reliable_face_defense_agents.red.transforms import apply_red_transform

    by_sid = {str(row["source_identity"]): row for row in protections}
    transformed = {}
    for pair in pairs:
        sid = str(pair["source_identity"])
        source = load_rgb01(by_sid[sid]["protected_path"], 256)
        for environment, (family, params) in ROBUST_ENVIRONMENTS.items():
            path = (
                output_dir / "robustness" / "transformed"
                / environment / f"{sid}.png"
            )
            if not path.is_file():
                spec = RedTransformSpec(environment, family, tuple(params))
                save_rgb01(apply_red_transform(source, spec), path)
            transformed[(sid, environment)] = str(path)
    write_json(output_dir / "robustness" / "transform_protocol.json", {
        "pair_selection": "frozen target_index == 0",
        "environments": {
            "clean": {"source": "main rerun target_index == 0"},
            **{
                name: {"family": family, "params": list(params)}
                for name, (family, params) in ROBUST_ENVIRONMENTS.items()
            },
        },
        "osn_note": "Frozen project proxy: JPEG75 followed by JPEG80",
    })
    return transformed


def _run_robust_swap_cache(
    *,
    base_cfg: dict,
    pairs: list[dict],
    transformed: dict[tuple[str, str], str],
    output_dir: Path,
) -> list[dict]:
    jsonl = output_dir / "robustness" / "swap_cache_manifest.jsonl"
    rows = read_jsonl(jsonl)
    done = {
        (row["swap_model"], row["environment"], row["pair_id"])
        for row in rows
    }
    total = (
        len(base_cfg["swap_models"])
        * len(ROBUST_ENVIRONMENTS)
        * len(pairs)
    )
    for model in base_cfg["swap_models"]:
        swapper = make_swapper(model, base_cfg, base_cfg["device"])
        for environment in ROBUST_ENVIRONMENTS:
            for pair in pairs:
                key = (model, environment, pair["pair_id"])
                if key in done:
                    continue
                sid = str(pair["source_identity"])
                output = swapper.swap(
                    transformed[(sid, environment)],
                    pair["target_path"],
                )
                image_path = None
                if output.status == SWAP_OK and output.image01 is not None:
                    image_path = (
                        output_dir / "robustness" / "swap_cache"
                        / model / environment / f"{pair['pair_id']}.png"
                    )
                    save_rgb01(
                        F.interpolate(
                            output.image01,
                            (256, 256),
                            mode="bilinear",
                            align_corners=False,
                        ),
                        image_path,
                    )
                row = {
                    "swap_model": model,
                    "environment": environment,
                    "method": "ours",
                    "pair_id": pair["pair_id"],
                    "source_identity": int(pair["source_identity"]),
                    "target_identity": int(pair["target_identity"]),
                    "target_path": pair["target_path"],
                    "status": output.status,
                    "error": output.error,
                    "swap_image_path": (
                        None if image_path is None else str(image_path)
                    ),
                }
                append_jsonl(jsonl, row)
                rows.append(row)
                done.add(key)
                if len(done) % 100 == 0 or len(done) == total:
                    print(
                        f"NULLSWAP_FLOOR_ROBUST_SWAP={len(done)}/{total}",
                        flush=True,
                    )
        release_swapper(swapper)
    rows = read_jsonl(jsonl)
    if len(done) != total:
        raise RuntimeError(f"Incomplete robust swap cache: {len(done)} != {total}")
    write_json(output_dir / "robustness" / "swap_cache_manifest.json", rows)
    write_csv(output_dir / "robustness" / "swap_cache_manifest.csv", rows)
    return rows


def _summarize_robustness(
    *,
    base_cfg: dict,
    protocol: dict,
    baseline_main_dir: Path,
    new_main_rows: list[dict],
    transformed_ours_rows: list[dict],
    output_dir: Path,
) -> list[dict]:
    pairs = _target_zero_pairs(protocol)
    pair_ids = {row["pair_id"] for row in pairs}
    baseline_main = [
        row for row in read_jsonl(
            baseline_main_dir / "results" / "per_pair.jsonl"
        )
        if row["pair_id"] in pair_ids
    ]
    baseline_transformed = read_jsonl(
        baseline_main_dir / "robustness_v07"
        / "per_pair_transformed.jsonl"
    )
    eligible = _eligible_pairs(baseline_main, {
        **protocol,
        "pairs": pairs,
    })

    combined = []
    for row in baseline_main:
        if row["method"] == "nullswap":
            combined.append({**row, "environment": "clean"})
    for row in baseline_transformed:
        if (
            row["method"] == "nullswap"
            and row["pair_id"] in pair_ids
        ):
            combined.append(row)
    for row in new_main_rows:
        if row["pair_id"] in pair_ids:
            combined.append({**row, "environment": "clean"})
    combined.extend(transformed_ours_rows)

    grouped = defaultdict(list)
    for row in combined:
        grouped[(
            row["swap_model"], row["environment"], row["method"]
        )].append(row)

    environments = ("clean", *ROBUST_ENVIRONMENTS.keys())
    table = []
    for model in base_cfg["swap_models"]:
        for evaluator in HELD_OUT:
            for method in ("nullswap", "ours"):
                for environment in environments:
                    selected = [
                        row for row in grouped[(model, environment, method)]
                        if row["pair_id"] in eligible[model]
                    ]
                    if len(selected) != len(eligible[model]):
                        raise RuntimeError(
                            f"Missing robustness rows: {model}/{evaluator}/"
                            f"{method}/{environment} {len(selected)} != "
                            f"{len(eligible[model])}"
                        )
                    table.append({
                        "swap_model": model,
                        "evaluator": evaluator,
                        "method": method,
                        "environment": environment,
                        "denominator": len(selected),
                        "acc1": _mean([
                            row[f"{evaluator}_acc1"] for row in selected
                        ]),
                        "acc5": _mean([
                            row[f"{evaluator}_acc5"] for row in selected
                        ]),
                    })
    for evaluator in HELD_OUT:
        for method in ("nullswap", "ours"):
            for environment in environments:
                parts = [
                    row for row in table
                    if row["evaluator"] == evaluator
                    and row["method"] == method
                    and row["environment"] == environment
                ]
                table.append({
                    "swap_model": "average_macro_3_models",
                    "evaluator": evaluator,
                    "method": method,
                    "environment": environment,
                    "denominator": sum(row["denominator"] for row in parts),
                    "acc1": _mean([row["acc1"] for row in parts]),
                    "acc5": _mean([row["acc5"] for row in parts]),
                })
    write_json(output_dir / "robustness" / "robustness_table_vs_nullswap.json", table)
    write_csv(output_dir / "robustness" / "robustness_table_vs_nullswap.csv", table)
    return table


def run_robustness_evaluation(
    cfg: dict,
    *,
    main_dir: Path,
    output_dir: Path,
    base_cfg: dict,
    protocol: dict,
    protections: list[dict],
    new_main_rows: list[dict],
) -> list[dict]:
    pairs = _target_zero_pairs(protocol)
    transformed = _materialize_robust_sources(
        base_cfg=base_cfg,
        pairs=pairs,
        protections=protections,
        output_dir=output_dir,
    )
    cache_rows = _run_robust_swap_cache(
        base_cfg=base_cfg,
        pairs=pairs,
        transformed=transformed,
        output_dir=output_dir,
    )
    heldout = _evaluate_swap_cache_group(
        group="heldout",
        base_cfg=base_cfg,
        protocol=protocol,
        cache_rows=cache_rows,
        baseline_main_dir=main_dir,
        output_dir=output_dir,
        output_prefix="robustness",
    )
    return _summarize_robustness(
        base_cfg=base_cfg,
        protocol=protocol,
        baseline_main_dir=main_dir,
        new_main_rows=new_main_rows,
        transformed_ours_rows=heldout,
        output_dir=output_dir,
    )


def _comparison(table: list[dict], *, dimensions: tuple[str, ...]) -> dict:
    by_key = {
        tuple(row[name] for name in dimensions) + (row["method"],): row
        for row in table
    }
    outcomes = []
    base_keys = sorted({
        tuple(row[name] for name in dimensions) for row in table
    })
    for key in base_keys:
        baseline = by_key[key + ("nullswap",)]
        ours = by_key[key + ("ours",)]
        for metric in ("acc1", "acc5"):
            delta = float(ours[metric]) - float(baseline[metric])
            outcomes.append({
                **{name: value for name, value in zip(dimensions, key)},
                "metric": metric,
                "nullswap": baseline[metric],
                "ours": ours[metric],
                "ours_minus_nullswap": delta,
                "outcome": "better" if delta < 0 else "equal" if delta == 0 else "worse",
            })
    better = sum(row["outcome"] == "better" for row in outcomes)
    equal = sum(row["outcome"] == "equal" for row in outcomes)
    worse = sum(row["outcome"] == "worse" for row in outcomes)
    return {
        "cells": outcomes,
        "better": better,
        "equal": equal,
        "worse": worse,
        "pareto_better": worse == 0 and better > 0,
        "strictly_better_everywhere": all(
            row["outcome"] == "better" for row in outcomes
        ),
    }


def run(config: dict, *, stage: str = "all", limit: int | None = None) -> dict:
    main_dir, output_dir, base_cfg, protocol = _prepare(config, limit)
    protections = generate_refined_protections(
        config,
        main_dir=main_dir,
        output_dir=output_dir,
        base_cfg=base_cfg,
        protocol=protocol,
    )
    if stage == "protections":
        return read_json(output_dir / "protections" / "summary.json")

    main_rows, main_table = run_main_evaluation(
        config,
        main_dir=main_dir,
        output_dir=output_dir,
        base_cfg=base_cfg,
        protocol=protocol,
        protections=protections,
    )
    if stage == "main":
        robustness_table = None
    else:
        robustness_table = run_robustness_evaluation(
            config,
            main_dir=main_dir,
            output_dir=output_dir,
            base_cfg=base_cfg,
            protocol=protocol,
            protections=protections,
            new_main_rows=main_rows,
        )

    main_comparison = _comparison(
        main_table,
        dimensions=("swap_model", "evaluator"),
    )
    robust_comparison = None
    if robustness_table is not None:
        robust_comparison = _comparison(
            robustness_table,
            dimensions=("swap_model", "evaluator", "environment"),
        )
    selection = read_json(output_dir / "protections" / "summary.json")
    positive_probe_signal = (
        selection["selected_refinement_count"] > 0
        and main_comparison["pareto_better"]
        and robust_comparison is not None
        and robust_comparison["pareto_better"]
    )
    summary = {
        "status": "COMPLETE",
        "scope": protocol.get("run_scope", "full_frozen_protocol"),
        "source_count": len(protocol["source_identities"]),
        "pair_count": len(protocol["pairs"]),
        "selection": selection,
        "main_comparison": main_comparison,
        "robustness_comparison": robust_comparison,
        "positive_probe_signal": positive_probe_signal,
        "formal_success_rule": (
            "Ours must be lower than NullSwap; equal is reported as equal, "
            "not as an improvement."
        ),
        "original_v07_outputs_modified": False,
    }
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "main" / "cellwise_comparison.csv", main_comparison["cells"])
    if robust_comparison is not None:
        write_csv(
            output_dir / "robustness" / "cellwise_comparison.csv",
            robust_comparison["cells"],
        )
    return summary

