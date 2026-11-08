from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import gc
import json
import shutil

from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from .common import (
    append_jsonl,
    build_protocol,
    load_rgb01,
    read_json,
    read_jsonl,
    save_rgb01,
    write_csv,
    write_json,
)
from .evaluators import DetectionGate, HeldOutEvaluators, build_gallery, rank_hits
from .protection import generate_protections
from .swappers import (
    ENGINEERING_ERROR,
    SOURCE_DETECTION_FAILED,
    SWAP_OK,
    make_swapper,
    release_swapper,
)


def _required_paths(cfg):
    return [
        cfg["project_root"], cfg["deps_root"], cfg["official_split_manifest"],
        cfg["hq_mapping"], cfg["celeba_identity"], cfg["nullswap_checkpoint"],
        cfg["simswap"]["repo_root"], cfg["simswap"]["checkpoints_dir"],
        cfg["simswap"]["arcface_path"], cfg["blendswap"]["repo_root"],
        cfg["blendswap"]["checkpoint"], cfg["faceshifter"]["repo_root"],
        cfg["faceshifter"]["generator_checkpoint"],
        cfg["faceshifter"]["arcface_checkpoint"],
        cfg["evaluators"]["vggface"], cfg["evaluators"]["sface"],
    ]


def preflight(cfg, out_dir):
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal main experiment.")
    missing = [str(x) for x in _required_paths(cfg) if not Path(x).exists()]
    if missing:
        raise FileNotFoundError("Missing required path(s):\n" + "\n".join(missing))
    expected = {
        "nullswap_checkpoint": "4334878d7ca42e310a92071dfcba9730ce852945ff7193531dbcf37b85090f56",
        "blendswap_checkpoint": "3b3715afb2b5ee55680f0f7be46f06c85ed663c5874967ab279e3e66ce0b106a",
        "faceshifter_generator": "8b9feda5071eaadcaa88d6f58e947a7a3c8100af81178e764217be662c475854",
        "faceshifter_arcface": "a035c768259b98ab1ce0e646312f48b9e1e218197a0f80ac6765e88f8b6ddf28",
        "vggface": "759266b9614d0fd5d65b97bf716818b746cc77ab5944c7bffc937c6ba9455d8c",
        "sface": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    }
    payload = {
        "status": "PASS",
        "gpu": torch.cuda.get_device_name(0),
        "hash_verification": "reused previously verified frozen hashes; no multi-GB rehash at launch",
        "expected_hashes": expected,
    }
    write_json(out_dir / "preflight.json", payload)
    return payload


def smoke_components(cfg, protocol, out_dir):
    """One real GPU inference per external component, then the same run continues."""
    smoke_path = out_dir / "gpu_component_smoke.json"
    if smoke_path.is_file():
        payload = read_json(smoke_path)
        if payload.get("status") == "PASS":
            return payload

    first_source = protocol["pairs"][0]["source_path"]
    image = load_rgb01(first_source, 256)
    evaluators = HeldOutEvaluators(cfg, cfg["device"])
    evaluator_shapes = {
        name: list(value.shape)
        for name, value in evaluators.embed(image).items()
    }
    del evaluators
    gc.collect()
    torch.cuda.empty_cache()

    swap_status = {}
    for model_name in cfg["swap_models"]:
        swapper = make_swapper(model_name, cfg, cfg["device"])
        successful = None
        last = None
        for pair in protocol["pairs"][:20]:
            last = swapper.swap(pair["source_path"], pair["target_path"])
            if last.status == ENGINEERING_ERROR:
                raise RuntimeError(f"{model_name} GPU smoke failed: {last.error}")
            if last.status == SWAP_OK and last.image01 is not None:
                successful = pair["pair_id"]
                break
        if successful is None:
            raise RuntimeError(
                f"{model_name} produced no SWAP_OK result in 20 smoke pairs; "
                f"last_status={None if last is None else last.status}"
            )
        swap_status[model_name] = {"status": "PASS", "pair_id": successful}
        release_swapper(swapper)

    payload = {
        "status": "PASS",
        "evaluators": evaluator_shapes,
        "swap_models": swap_status,
        "note": "This is the only launch smoke; the same command proceeds to the full run.",
    }
    write_json(smoke_path, payload)
    return payload


def _protection_index(rows):
    return {
        (str(row["source_identity"]), row["method"]): row
        for row in rows
    }


def _conservative_error_row(base, status, error):
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
        "error_policy": "engineering/protection error counts as attack success; never as defense success",
    }


def _save_visual_candidate(out_dir, model_name, pair, method, protected_path, swap_image):
    root = out_dir / "visual_candidates" / model_name / pair["pair_id"]
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "source.png").is_file():
        Image.open(pair["source_path"]).convert("RGB").resize((256, 256)).save(root / "source.png")
        Image.open(pair["target_path"]).convert("RGB").resize((256, 256)).save(root / "target.png")
    if method == "no_protection":
        save_rgb01(swap_image, root / "swap_clean.png")
    elif method == "ours":
        if protected_path:
            Image.open(protected_path).convert("RGB").save(root / "protected.png")
        save_rgb01(swap_image, root / "swap_protected.png")


def _flush_batch(batch, gate, evaluators, gallery, results_path, done):
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
    detection_success, detection_scores = gate.success(images)
    query = evaluators.embed(images)
    hits = rank_hits(query, gallery, [x["source_identity"] for x in batch])
    for index, item in enumerate(batch):
        detected = bool(detection_success[index])
        row = {
            **item["base"],
            "status": SWAP_OK,
            "error": None,
            "detection_success": detected,
            "detection_score": float(detection_scores[index]),
        }
        for evaluator_name in ("vggface", "sface"):
            row[f"{evaluator_name}_acc1"] = bool(hits[evaluator_name]["acc1"][index]) if detected else False
            row[f"{evaluator_name}_acc5"] = bool(hits[evaluator_name]["acc5"][index]) if detected else False
        append_jsonl(results_path, row)
        done.add((row["swap_model"], row["method"], row["pair_id"]))
    batch.clear()


def run_swaps_and_evaluation(cfg, protocol, protections, out_dir):
    results_path = out_dir / "results" / "per_pair.jsonl"
    previous = read_jsonl(results_path)
    done = {(x["swap_model"], x["method"], x["pair_id"]) for x in previous}
    protection_by_key = _protection_index(protections)

    gate = DetectionGate(cfg["deps_root"], cfg["device"])
    gate.calibrate(
        protocol["calibration_paths"],
        target_tpr=float(cfg["detection_clean_tpr"]),
        cache_path=out_dir / "detection_gate.json",
    )
    evaluators = HeldOutEvaluators(cfg, cfg["device"])
    gallery = build_gallery(
        evaluators, protocol["gallery"], out_dir / "gallery_embeddings.pt"
    )

    visual_candidate_limit = max(120, int(cfg["visual_groups"]) * 2)
    for model_name in cfg["swap_models"]:
        if all(
            (model_name, method, pair["pair_id"]) in done
            for method in cfg["methods"] for pair in protocol["pairs"]
        ):
            continue

        swapper = make_swapper(model_name, cfg, cfg["device"])
        model_ok = False
        batch = []
        for method in cfg["methods"]:
            for pair_index, pair in enumerate(protocol["pairs"]):
                key = (model_name, method, pair["pair_id"])
                if key in done:
                    continue
                protected_row = protection_by_key.get((str(pair["source_identity"]), method))
                base = {
                    "swap_model": model_name,
                    "method": method,
                    "pair_id": pair["pair_id"],
                    "source_identity": int(pair["source_identity"]),
                    "source_path": pair["source_path"],
                    "target_identity": int(pair["target_identity"]),
                    "target_path": pair["target_path"],
                }
                if not protected_row or protected_row.get("status") != "OK" or not protected_row.get("protected_path"):
                    row = _conservative_error_row(base, "PROTECTION_INFEASIBLE", None)
                    append_jsonl(results_path, row)
                    done.add(key)
                    continue

                output = swapper.swap(protected_row["protected_path"], pair["target_path"])
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
                        row = _conservative_error_row(base, output.status, output.error)
                    append_jsonl(results_path, row)
                    done.add(key)
                    continue

                model_ok = True
                standardized = F.interpolate(
                    output.image01,
                    (256, 256),
                    mode="bilinear",
                    align_corners=False,
                )
                if bool(cfg["save_all_swap_images"]):
                    save_rgb01(
                        standardized,
                        out_dir / "swap_images" / model_name / method / f"{pair['pair_id']}.png",
                    )
                if pair_index < visual_candidate_limit and method in {"no_protection", "ours"}:
                    _save_visual_candidate(
                        out_dir, model_name, pair, method,
                        protected_row.get("protected_path"), standardized,
                    )
                batch.append({
                    "base": base,
                    "source_identity": pair["source_identity"],
                    "image": standardized,
                })
                if len(batch) >= 12:
                    _flush_batch(batch, gate, evaluators, gallery, results_path, done)
            _flush_batch(batch, gate, evaluators, gallery, results_path, done)

        if not model_ok:
            raise RuntimeError(f"{model_name} produced no successful output in the main experiment.")
        release_swapper(swapper)

    rows = read_jsonl(results_path)
    write_json(out_dir / "results" / "per_pair.json", rows)
    write_csv(out_dir / "results" / "per_pair.csv", rows)
    return rows


def _mean(values):
    values = [float(x) for x in values if x is not None]
    return sum(values) / len(values) if values else None


def summarize(cfg, protocol, protections, rows, out_dir):
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
        for evaluator in ("vggface", "sface"):
            for method in cfg["methods"]:
                selected = [
                    row for row in by_model_method[(model_name, method)]
                    if row["pair_id"] in denominator_ids
                ]
                visual = visual_by_method[method]
                table.append({
                    "swap_model": model_name,
                    "evaluator": evaluator,
                    "method": method,
                    "denominator": len(selected),
                    "acc1": _mean([x[f"{evaluator}_acc1"] for x in selected]),
                    "acc5": _mean([x[f"{evaluator}_acc5"] for x in selected]),
                    "psnr": _mean([x.get("psnr") for x in visual]),
                    "ssim": _mean([x.get("ssim") for x in visual]),
                    "lpips": _mean([x.get("lpips") for x in visual]),
                    "engineering_errors": sum(x["status"] == ENGINEERING_ERROR for x in selected),
                })

    for evaluator in ("vggface", "sface"):
        for method in cfg["methods"]:
            parts = [x for x in table if x["evaluator"] == evaluator and x["method"] == method]
            table.append({
                "swap_model": "平均",
                "evaluator": evaluator,
                "method": method,
                "denominator": sum(x["denominator"] for x in parts),
                "acc1": _mean([x["acc1"] for x in parts]),
                "acc5": _mean([x["acc5"] for x in parts]),
                "psnr": parts[0]["psnr"] if parts else None,
                "ssim": parts[0]["ssim"] if parts else None,
                "lpips": parts[0]["lpips"] if parts else None,
                "engineering_errors": sum(x["engineering_errors"] for x in parts),
            })
    write_csv(out_dir / "results" / "main_table.csv", table)
    write_json(out_dir / "results" / "main_table.json", table)
    return table, eligible


def _make_grid(files, output_path):
    labels = ["Source", "Target", "Protected", "Clean Swap", "Protected Swap"]
    images = [Image.open(path).convert("RGB").resize((256, 256)) for path in files]
    canvas = Image.new("RGB", (256 * 5, 292), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(zip(labels, images)):
        canvas.paste(image, (index * 256, 36))
        draw.text((index * 256 + 8, 10), label, fill="black")
    canvas.save(output_path)


def export_visual70(cfg, protocol, rows, eligible, out_dir):
    quotas = {"simswap": 24, "blendswap": 23, "faceshifter": 23}
    requested = int(cfg["visual_groups"])
    if requested != sum(quotas.values()):
        base = requested // len(cfg["swap_models"])
        quotas = {name: base for name in cfg["swap_models"]}
        for name in cfg["swap_models"][: requested - base * len(cfg["swap_models"])]:
            quotas[name] += 1

    row_index = {(x["swap_model"], x["method"], x["pair_id"]): x for x in rows}
    manifest = []
    visual_root = out_dir / "visual70"
    for model_name in cfg["swap_models"]:
        accepted = 0
        for pair in protocol["pairs"]:
            if accepted >= quotas[model_name]:
                break
            pair_id = pair["pair_id"]
            candidate = out_dir / "visual_candidates" / model_name / pair_id
            names = ["source.png", "target.png", "protected.png", "swap_clean.png", "swap_protected.png"]
            if pair_id not in eligible[model_name] or not all((candidate / x).is_file() for x in names):
                continue
            group_index = len(manifest) + 1
            dst = visual_root / f"{group_index:03d}_{model_name}_{pair_id}"
            dst.mkdir(parents=True, exist_ok=True)
            for name in names:
                shutil.copy2(candidate / name, dst / name)
            _make_grid([dst / x for x in names], dst / "grid_5class.png")
            ours = row_index.get((model_name, "ours", pair_id), {})
            manifest.append({
                "visual_index": group_index,
                "swap_model": model_name,
                "pair_id": pair_id,
                "source_identity": pair["source_identity"],
                "target_identity": pair["target_identity"],
                "vggface_acc1": ours.get("vggface_acc1"),
                "vggface_acc5": ours.get("vggface_acc5"),
                "sface_acc1": ours.get("sface_acc1"),
                "sface_acc5": ours.get("sface_acc5"),
                "directory": str(dst),
            })
            accepted += 1
    if len(manifest) != requested:
        raise RuntimeError(f"Could export only {len(manifest)} of {requested} required visual groups.")
    write_json(visual_root / "visual70_manifest.json", manifest)
    write_csv(visual_root / "visual70_manifest.csv", manifest)
    return manifest


def run(config_path):
    cfg = read_json(config_path)
    project_root = Path(cfg["project_root"])
    import sys
    src = str(project_root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "frozen_main_config.json", cfg)
    preflight(cfg, out_dir)

    protocol_path = out_dir / "protocol_manifest.json"
    if protocol_path.is_file():
        protocol = read_json(protocol_path)
    else:
        protocol = build_protocol(cfg)
        write_json(protocol_path, protocol)

    smoke_components(cfg, protocol, out_dir)

    protections = generate_protections(cfg, protocol, out_dir)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rows = run_swaps_and_evaluation(cfg, protocol, protections, out_dir)
    table, eligible = summarize(cfg, protocol, protections, rows, out_dir)
    visual = export_visual70(cfg, protocol, rows, eligible, out_dir)
    summary = {
        "status": "COMPLETE",
        "pair_count_per_model": len(protocol["pairs"]),
        "methods": cfg["methods"],
        "swap_models": cfg["swap_models"],
        "held_out_evaluators": ["vggface", "sface"],
        "eligible_pairs": {k: len(v) for k, v in eligible.items()},
        "visual_groups": len(visual),
        "main_table": str(out_dir / "results" / "main_table.csv"),
        "output_dir": str(out_dir),
    }
    write_json(out_dir / "summary.json", summary)
    return summary
