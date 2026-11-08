from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import gc

import torch
import torch.nn.functional as F

from .common import (
    append_jsonl,
    load_rgb01,
    read_json,
    read_jsonl,
    write_csv,
    write_json,
)
from .evaluators import HeldOutEvaluators, rank_hits
from .swappers import (
    ENGINEERING_ERROR,
    SOURCE_DETECTION_FAILED,
    SWAP_OK,
    make_swapper,
    release_swapper,
)


NEW_EVALUATORS = ("arcface", "facenet")
ALL_EVALUATORS = ("arcface", "facenet", "vggface", "sface")


class ArcFaceFaceNetEvaluators:
    """Project-native ArcFace and FaceNet adapters used as added evaluators."""

    def __init__(self, cfg, device):
        from reliable_face_defense_02r3.identity.arcface_adapter import (
            ArcFaceR50Adapter,
        )
        from reliable_face_defense_02r3.identity.facenet_adapter import (
            FaceNetVGGFace2Adapter,
        )

        self.device = torch.device(device)
        self.arcface = ArcFaceR50Adapter(cfg["deps_root"], device=device).eval()
        self.facenet = FaceNetVGGFace2Adapter(cfg["deps_root"], device=device).eval()

    @torch.no_grad()
    def embed(self, images01):
        images01 = images01.to(self.device)
        return {
            "arcface": self.arcface(images01).detach().cpu(),
            "facenet": self.facenet(images01).detach().cpu(),
        }


def _row_key(row):
    return (row["swap_model"], row["method"], row["pair_id"])


def _validate_unique(rows, label):
    seen = set()
    duplicates = []
    for row in rows:
        key = _row_key(row)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        raise RuntimeError(f"{label} contains duplicate keys; first={duplicates[0]}")
    return seen


def _validate_original(cfg, protocol, original_rows):
    expected = {
        (model, method, pair["pair_id"])
        for model in cfg["swap_models"]
        for method in cfg["methods"]
        for pair in protocol["pairs"]
    }
    actual = _validate_unique(original_rows, "original per_pair.jsonl")
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise RuntimeError(
            "Original main result cardinality mismatch: "
            f"expected={len(expected)} actual={len(actual)} "
            f"missing={len(missing)} extra={len(extra)}"
        )
    required = {
        "vggface_acc1", "vggface_acc5", "sface_acc1", "sface_acc5",
        "status", "detection_success",
    }
    for row in original_rows:
        absent = required - set(row)
        if absent:
            raise RuntimeError(f"Original row {_row_key(row)} lacks {sorted(absent)}")
    return expected


def _build_gallery(evaluators, gallery, cache_path, batch_size=12):
    cache_path = Path(cache_path)
    identities = sorted(int(x) for x in gallery)
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("identities") != identities:
            raise RuntimeError("ArcFace/FaceNet gallery cache identity mismatch.")
        if set(payload.get("embeddings", {})) != set(NEW_EVALUATORS):
            raise RuntimeError("ArcFace/FaceNet gallery cache evaluator mismatch.")
        return payload

    embeddings = {name: [] for name in NEW_EVALUATORS}
    for start in range(0, len(identities), batch_size):
        ids = identities[start:start + batch_size]
        images = torch.cat([load_rgb01(gallery[str(x)], 256) for x in ids])
        batch_embeddings = evaluators.embed(images)
        for name in NEW_EVALUATORS:
            embeddings[name].append(batch_embeddings[name])
    payload = {
        "identities": identities,
        "embeddings": {
            name: F.normalize(torch.cat(parts).float(), dim=1)
            for name, parts in embeddings.items()
        },
        "evaluator_roles": {
            "arcface": "project-internal/surrogate evaluator",
            "facenet": "project-internal/surrogate evaluator",
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def _protection_index(rows):
    return {
        (str(row["source_identity"]), row["method"]): row
        for row in rows
    }


def _copy_non_replay_row(original):
    """Extend rows that cannot/need not be identity-evaluated after swapping."""
    row = dict(original)
    if original["status"] == SOURCE_DETECTION_FAILED:
        value = False
        reason = "original source detection failure"
    elif original["status"] == SWAP_OK and original["detection_success"] is False:
        value = False
        reason = "original held-out detection gate failure"
    elif original["status"] != SWAP_OK:
        value = True
        reason = "original conservative engineering-error policy"
    else:
        raise ValueError("A detected SWAP_OK row requires replay evaluation.")

    for evaluator in NEW_EVALUATORS:
        row[f"{evaluator}_acc1"] = value
        row[f"{evaluator}_acc5"] = value
    row["supplement_replay_status"] = "NOT_REQUIRED"
    row["supplement_replay_note"] = reason
    return row


def _flush_batch(batch, evaluators, gallery, results_path, done):
    if not batch:
        return 0
    images = torch.cat([
        F.interpolate(
            item["image"].detach().cpu(),
            (256, 256),
            mode="bilinear",
            align_corners=False,
        )
        for item in batch
    ])
    query = evaluators.embed(images)
    hits = rank_hits(query, gallery, [x["source_identity"] for x in batch])
    for index, item in enumerate(batch):
        row = dict(item["original"])
        for evaluator in NEW_EVALUATORS:
            row[f"{evaluator}_acc1"] = bool(hits[evaluator]["acc1"][index])
            row[f"{evaluator}_acc5"] = bool(hits[evaluator]["acc5"][index])
        row["supplement_replay_status"] = SWAP_OK
        row["supplement_replay_note"] = None
        append_jsonl(results_path, row)
        done.add(_row_key(row))
    count = len(batch)
    batch.clear()
    return count


def _flush_source_batch(
    batch, added_evaluators, held_out_evaluators, gallery, results_path, done
):
    if not batch:
        return 0
    images = torch.cat([item["image"] for item in batch])
    query = {
        **added_evaluators.embed(images),
        **held_out_evaluators.embed(images),
    }
    hits = rank_hits(query, gallery, [x["source_identity"] for x in batch])
    for index, item in enumerate(batch):
        row = dict(item["base"])
        for evaluator in ALL_EVALUATORS:
            row[f"{evaluator}_acc1"] = bool(hits[evaluator]["acc1"][index])
            row[f"{evaluator}_acc5"] = bool(hits[evaluator]["acc5"][index])
        append_jsonl(results_path, row)
        done.add((row["method"], str(row["source_identity"])))
    count = len(batch)
    batch.clear()
    return count


def evaluate_source_cloaking(
    cfg, protocol, protections, main_dir, output_dir, added_evaluators, added_gallery
):
    """NullSwap Table-2-style evaluation directly on protected source faces."""
    held_gallery_path = Path(main_dir) / "gallery_embeddings.pt"
    if not held_gallery_path.is_file():
        raise FileNotFoundError(
            f"Missing frozen VGGFace/SFace gallery: {held_gallery_path}"
        )
    held_gallery = torch.load(
        held_gallery_path, map_location="cpu", weights_only=False
    )
    if held_gallery.get("identities") != added_gallery.get("identities"):
        raise RuntimeError("Held-out and added-evaluator gallery identities differ.")
    if set(held_gallery.get("embeddings", {})) != {"vggface", "sface"}:
        raise RuntimeError("Frozen held-out gallery evaluator mismatch.")
    gallery = {
        "identities": added_gallery["identities"],
        "embeddings": {
            **added_gallery["embeddings"],
            **held_gallery["embeddings"],
        },
    }

    held_out = HeldOutEvaluators(cfg, cfg["device"])
    results_path = Path(output_dir) / "source_cloaking_per_image.jsonl"
    previous = read_jsonl(results_path)
    done = set()
    for row in previous:
        key = (row["method"], str(row["source_identity"]))
        if key in done:
            raise RuntimeError(f"Duplicate source-cloaking row: {key}")
        if not all(
            f"{name}_acc{k}" in row
            for name in ALL_EVALUATORS for k in (1, 5)
        ):
            raise RuntimeError(f"Incomplete source-cloaking row: {key}")
        done.add(key)

    protection_by_key = _protection_index(protections)
    expected = {
        (method, str(source_identity))
        for method in cfg["methods"]
        for source_identity in protocol["source_identities"]
    }
    if not done <= expected:
        raise RuntimeError("Source-cloaking resume file contains unexpected keys.")

    batch = []
    completed = len(done)
    for method in cfg["methods"]:
        for source_identity in protocol["source_identities"]:
            sid = str(source_identity)
            key = (method, sid)
            if key in done:
                continue
            protected = protection_by_key.get((sid, method))
            base = {
                "method": method,
                "source_identity": int(source_identity),
                "status": None if protected is None else protected.get("status"),
                "protected_path": (
                    None if protected is None else protected.get("protected_path")
                ),
            }
            if (
                not protected
                or protected.get("status") != "OK"
                or not protected.get("protected_path")
            ):
                row = dict(base)
                for evaluator in ALL_EVALUATORS:
                    row[f"{evaluator}_acc1"] = True
                    row[f"{evaluator}_acc5"] = True
                row["error_policy"] = (
                    "protection error counts as identity-match success; "
                    "never as defense success"
                )
                append_jsonl(results_path, row)
                done.add(key)
                completed += 1
                continue
            batch.append({
                "base": base,
                "source_identity": int(source_identity),
                "image": load_rgb01(protected["protected_path"], 256),
            })
            if len(batch) >= 8:
                completed += _flush_source_batch(
                    batch, added_evaluators, held_out, gallery, results_path, done
                )
        completed += _flush_source_batch(
            batch, added_evaluators, held_out, gallery, results_path, done
        )
        print(
            "SOURCE_CLOAKING_PROGRESS "
            f"method={method} completed={completed}/{len(expected)}",
            flush=True,
        )

    del held_out
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rows = read_jsonl(results_path)
    actual = {(x["method"], str(x["source_identity"])) for x in rows}
    if actual != expected or len(rows) != len(expected):
        raise RuntimeError(
            f"Source-cloaking evaluation incomplete: expected={len(expected)} "
            f"actual={len(rows)}"
        )
    write_json(Path(output_dir) / "source_cloaking_per_image.json", rows)
    write_csv(Path(output_dir) / "source_cloaking_per_image.csv", rows)
    return rows


def _mean(values):
    values = [float(x) for x in values if x is not None]
    return sum(values) / len(values) if values else None


def summarize_four_evaluators(cfg, protections, rows, output_dir):
    """Use the original no-protection eligibility denominator for all evaluators."""
    by_model_method = defaultdict(list)
    for row in rows:
        by_model_method[(row["swap_model"], row["method"])].append(row)

    eligible = {}
    for model_name in cfg["swap_models"]:
        eligible[model_name] = {
            row["pair_id"]
            for row in by_model_method[(model_name, "no_protection")]
            if row["status"] == SWAP_OK and row["detection_success"] is True
        }

    visual_by_method = defaultdict(list)
    for row in protections:
        if row.get("status") == "OK":
            visual_by_method[row["method"]].append(row)

    table = []
    for model_name in cfg["swap_models"]:
        denominator_ids = eligible[model_name]
        for evaluator in ALL_EVALUATORS:
            for method in cfg["methods"]:
                selected = [
                    row for row in by_model_method[(model_name, method)]
                    if row["pair_id"] in denominator_ids
                ]
                visual = visual_by_method[method]
                table.append({
                    "swap_model": model_name,
                    "evaluator": evaluator,
                    "evaluator_role": (
                        "held-out" if evaluator in {"vggface", "sface"}
                        else "project-internal/surrogate"
                    ),
                    "method": method,
                    "denominator": len(selected),
                    "acc1": _mean([x[f"{evaluator}_acc1"] for x in selected]),
                    "acc5": _mean([x[f"{evaluator}_acc5"] for x in selected]),
                    "psnr": _mean([x.get("psnr") for x in visual]),
                    "ssim": _mean([x.get("ssim") for x in visual]),
                    "lpips": _mean([x.get("lpips") for x in visual]),
                    "engineering_errors": sum(
                        x["status"] == ENGINEERING_ERROR for x in selected
                    ),
                })

    for evaluator in ALL_EVALUATORS:
        for method in cfg["methods"]:
            parts = [
                x for x in table
                if x["evaluator"] == evaluator and x["method"] == method
            ]
            table.append({
                "swap_model": "平均",
                "evaluator": evaluator,
                "evaluator_role": (
                    "held-out" if evaluator in {"vggface", "sface"}
                    else "project-internal/surrogate"
                ),
                "method": method,
                "denominator": sum(x["denominator"] for x in parts),
                "acc1": _mean([x["acc1"] for x in parts]),
                "acc5": _mean([x["acc5"] for x in parts]),
                "psnr": parts[0]["psnr"] if parts else None,
                "ssim": parts[0]["ssim"] if parts else None,
                "lpips": parts[0]["lpips"] if parts else None,
                "engineering_errors": sum(x["engineering_errors"] for x in parts),
            })

    output_dir = Path(output_dir)
    write_csv(output_dir / "main_table_four_evaluators.csv", table)
    write_json(output_dir / "main_table_four_evaluators.json", table)
    return table, eligible


def write_nullswap_style_tables(
    cfg, protections, source_rows, swap_rows, eligible, output_dir
):
    """Paper-facing pivots following NullSwap Tables 1, 2, and 3."""
    output_dir = Path(output_dir)

    visual_by_method = defaultdict(list)
    for row in protections:
        if row.get("status") == "OK":
            visual_by_method[row["method"]].append(row)
    visual_table = []
    for method in cfg["methods"]:
        selected = visual_by_method[method]
        visual_table.append({
            "method": method,
            "psnr_up": _mean([x.get("psnr") for x in selected]),
            "ssim_up": _mean([x.get("ssim") for x in selected]),
            "lpips_down": _mean([x.get("lpips") for x in selected]),
        })
    write_csv(output_dir / "paper_table_1_visual_quality.csv", visual_table)
    write_json(output_dir / "paper_table_1_visual_quality.json", visual_table)

    source_by_evaluator_method = {}
    for evaluator in ALL_EVALUATORS:
        for method in cfg["methods"]:
            selected = [x for x in source_rows if x["method"] == method]
            source_by_evaluator_method[(evaluator, method)] = {
                "acc5": _mean([x[f"{evaluator}_acc5"] for x in selected]),
                "acc1": _mean([x[f"{evaluator}_acc1"] for x in selected]),
            }
    source_table = []
    for evaluator in ALL_EVALUATORS:
        row = {"evaluator": evaluator}
        for method in cfg["methods"]:
            values = source_by_evaluator_method[(evaluator, method)]
            row[f"{method}__acc5_down"] = values["acc5"]
            row[f"{method}__acc1_down"] = values["acc1"]
        source_table.append(row)
    average = {"evaluator": "Average"}
    for method in cfg["methods"]:
        average[f"{method}__acc5_down"] = _mean([
            source_by_evaluator_method[(evaluator, method)]["acc5"]
            for evaluator in ALL_EVALUATORS
        ])
        average[f"{method}__acc1_down"] = _mean([
            source_by_evaluator_method[(evaluator, method)]["acc1"]
            for evaluator in ALL_EVALUATORS
        ])
    source_table.append(average)
    write_csv(output_dir / "paper_table_2_source_cloaking.csv", source_table)
    write_json(output_dir / "paper_table_2_source_cloaking.json", source_table)

    by_model_method = defaultdict(list)
    for row in swap_rows:
        by_model_method[(row["swap_model"], row["method"])].append(row)

    face_swap_tables = {}
    for k in (1, 5):
        table = []
        for model_name in cfg["swap_models"]:
            row = {
                "swap_model": model_name,
                "eligible_pairs": len(eligible[model_name]),
            }
            for method in cfg["methods"]:
                selected = [
                    x for x in by_model_method[(model_name, method)]
                    if x["pair_id"] in eligible[model_name]
                ]
                for evaluator in ("vggface", "sface"):
                    row[f"{method}__{evaluator}__acc{k}_down"] = _mean([
                        x[f"{evaluator}_acc{k}"] for x in selected
                    ])
            table.append(row)
        average_row = {
            "swap_model": "Average",
            "eligible_pairs": sum(len(x) for x in eligible.values()),
        }
        for method in cfg["methods"]:
            for evaluator in ("vggface", "sface"):
                field = f"{method}__{evaluator}__acc{k}_down"
                average_row[field] = _mean([x[field] for x in table])
        table.append(average_row)
        face_swap_tables[k] = table
        write_csv(output_dir / f"paper_table_3_face_swap_acc{k}.csv", table)
        write_json(output_dir / f"paper_table_3_face_swap_acc{k}.json", table)

    layout_note = {
        "paper_table_1": (
            "NullSwap Table 1 layout: methods as rows; PSNR, SSIM, LPIPS columns."
        ),
        "paper_table_2": (
            "NullSwap Table 2 layout: four recognizers as rows; methods grouped "
            "by Acc@5 and Acc@1. Evaluated directly on protected source faces."
        ),
        "paper_table_3": (
            "NullSwap Table 3 layout: face-swap models as rows; methods grouped "
            "by two held-out evaluators. Acc@1 and Acc@5 are separated to avoid "
            "one unreadably wide table."
        ),
        "metric_note": (
            "The table layout follows NullSwap, but our registered metric remains "
            "identity-matching accuracy; it is not relabeled as cosine similarity."
        ),
        "complete_four_evaluator_detail": "main_table_four_evaluators.csv",
    }
    write_json(output_dir / "paper_table_layout.json", layout_note)
    return {
        "visual_quality": visual_table,
        "source_cloaking": source_table,
        "face_swap_acc1": face_swap_tables[1],
        "face_swap_acc5": face_swap_tables[5],
    }


def _release_evaluators(evaluators):
    for name in ("arcface", "facenet"):
        if hasattr(evaluators, name):
            delattr(evaluators, name)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(main_dir, config_path=None):
    main_dir = Path(main_dir)
    supplement_dir = main_dir / "four_evaluator_supplement"
    supplement_dir.mkdir(parents=True, exist_ok=True)

    frozen_config = main_dir / "frozen_main_config.json"
    if frozen_config.is_file():
        cfg = read_json(frozen_config)
        config_source = frozen_config
    elif config_path is not None:
        cfg = read_json(config_path)
        config_source = Path(config_path)
    else:
        raise FileNotFoundError(f"Missing frozen config: {frozen_config}")

    project_src = str(Path(cfg["project_root"]) / "src")
    import sys
    if project_src not in sys.path:
        sys.path.insert(0, project_src)

    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal supplement.")

    required_paths = {
        "protocol": main_dir / "protocol_manifest.json",
        "protections": main_dir / "protections" / "manifest.jsonl",
        "original results": main_dir / "results" / "per_pair.jsonl",
        "ArcFace weight": (
            Path(cfg["deps_root"])
            / "weights/identity/arcface_r50/backbone.pth"
        ),
        "FaceNet weight": (
            Path(cfg["deps_root"])
            / "weights/identity/facenet/20180402-114759-vggface2.pt"
        ),
    }
    missing = [f"{name}: {path}" for name, path in required_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing supplement input(s):\n" + "\n".join(missing))

    protocol = read_json(required_paths["protocol"])
    protections = read_jsonl(required_paths["protections"])
    original_rows = read_jsonl(required_paths["original results"])
    expected_keys = _validate_original(cfg, protocol, original_rows)
    original_by_key = {_row_key(row): row for row in original_rows}
    protection_by_key = _protection_index(protections)

    results_path = supplement_dir / "per_pair_four_evaluators.jsonl"
    previous = read_jsonl(results_path)
    previous_keys = _validate_unique(previous, "supplement JSONL")
    if not previous_keys <= expected_keys:
        raise RuntimeError("Supplement JSONL contains keys outside the frozen main run.")
    done = {
        _row_key(row) for row in previous
        if all(f"{name}_acc{k}" in row for name in NEW_EVALUATORS for k in (1, 5))
    }
    if len(done) != len(previous):
        raise RuntimeError(
            "Supplement JSONL has an incomplete row. Move it aside before resuming."
        )

    write_json(supplement_dir / "supplement_config.json", {
        "config_source": str(config_source),
        "main_dir": str(main_dir),
        "output_dir": str(supplement_dir),
        "new_evaluators": list(NEW_EVALUATORS),
        "preserved_evaluators": ["vggface", "sface"],
        "swap_models": cfg["swap_models"],
        "methods": cfg["methods"],
        "eligibility": "frozen original no_protection SWAP_OK and detection_success=True",
        "original_results_are_never_modified": True,
    })

    evaluators = ArcFaceFaceNetEvaluators(cfg, cfg["device"])
    gallery = _build_gallery(
        evaluators,
        protocol["gallery"],
        supplement_dir / "gallery_embeddings_arcface_facenet.pt",
    )
    source_rows = evaluate_source_cloaking(
        cfg,
        protocol,
        protections,
        main_dir,
        supplement_dir,
        evaluators,
        gallery,
    )

    completed = len(done)
    expected_count = len(expected_keys)
    for model_name in cfg["swap_models"]:
        model_keys = {
            (model_name, method, pair["pair_id"])
            for method in cfg["methods"]
            for pair in protocol["pairs"]
        }
        if model_keys <= done:
            continue

        swapper = make_swapper(model_name, cfg, cfg["device"])
        batch = []
        for method in cfg["methods"]:
            for pair in protocol["pairs"]:
                key = (model_name, method, pair["pair_id"])
                if key in done:
                    continue
                original = original_by_key[key]
                needs_replay = (
                    original["status"] == SWAP_OK
                    and original["detection_success"] is True
                )
                if not needs_replay:
                    row = _copy_non_replay_row(original)
                    append_jsonl(results_path, row)
                    done.add(key)
                    completed += 1
                    continue

                protected = protection_by_key.get(
                    (str(pair["source_identity"]), method)
                )
                if (
                    not protected
                    or protected.get("status") != "OK"
                    or not protected.get("protected_path")
                ):
                    raise RuntimeError(
                        f"Original successful row {key} has no reusable protection image."
                    )

                output = swapper.swap(
                    protected["protected_path"], pair["target_path"]
                )
                if output.status != SWAP_OK or output.image01 is None:
                    raise RuntimeError(
                        "Replay disagrees with the frozen successful main row: "
                        f"key={key} replay_status={output.status} error={output.error}. "
                        "The supplement stopped instead of inventing a metric."
                    )
                batch.append({
                    "original": original,
                    "source_identity": pair["source_identity"],
                    "image": output.image01,
                })
                if len(batch) >= 12:
                    completed += _flush_batch(
                        batch, evaluators, gallery, results_path, done
                    )
                if completed and completed % 200 == 0:
                    print(
                        "FOUR_EVAL_PROGRESS "
                        f"model={model_name} method={method} "
                        f"completed={completed}/{expected_count}",
                        flush=True,
                    )
            completed += _flush_batch(
                batch, evaluators, gallery, results_path, done
            )
            print(
                "FOUR_EVAL_PROGRESS "
                f"model={model_name} method={method} "
                f"completed={completed}/{expected_count}",
                flush=True,
            )
        release_swapper(swapper)

    _release_evaluators(evaluators)

    final_rows = read_jsonl(results_path)
    final_keys = _validate_unique(final_rows, "completed supplement JSONL")
    if final_keys != expected_keys:
        raise RuntimeError(
            f"Supplement incomplete: expected={len(expected_keys)} actual={len(final_keys)}"
        )
    by_key = {_row_key(row): row for row in final_rows}
    ordered_rows = [by_key[_row_key(row)] for row in original_rows]
    write_json(supplement_dir / "per_pair_four_evaluators.json", ordered_rows)
    write_csv(supplement_dir / "per_pair_four_evaluators.csv", ordered_rows)

    table, eligible = summarize_four_evaluators(
        cfg, protections, ordered_rows, supplement_dir
    )
    paper_tables = write_nullswap_style_tables(
        cfg,
        protections,
        source_rows,
        ordered_rows,
        eligible,
        supplement_dir,
    )
    summary = {
        "status": "COMPLETE",
        "original_rows_preserved": len(ordered_rows),
        "pair_count_per_model": len(protocol["pairs"]),
        "methods": cfg["methods"],
        "swap_models": cfg["swap_models"],
        "evaluators": list(ALL_EVALUATORS),
        "evaluator_roles": {
            "arcface": "project-internal/surrogate",
            "facenet": "project-internal/surrogate",
            "vggface": "held-out",
            "sface": "held-out",
        },
        "eligible_pairs": {name: len(ids) for name, ids in eligible.items()},
        "table_rows": len(table),
        "paper_tables": {
            "visual_quality": str(
                supplement_dir / "paper_table_1_visual_quality.csv"
            ),
            "source_cloaking": str(
                supplement_dir / "paper_table_2_source_cloaking.csv"
            ),
            "face_swap_acc1": str(
                supplement_dir / "paper_table_3_face_swap_acc1.csv"
            ),
            "face_swap_acc5": str(
                supplement_dir / "paper_table_3_face_swap_acc5.csv"
            ),
        },
        "appendix_full_table": str(
            supplement_dir / "main_table_four_evaluators.csv"
        ),
        "paper_table_row_counts": {
            name: len(value) for name, value in paper_tables.items()
        },
        "output_dir": str(supplement_dir),
        "original_main_results_modified": False,
    }
    write_json(supplement_dir / "summary.json", summary)
    return summary
