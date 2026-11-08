from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import torch


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="NullSwap-floor rerun preflight")
    parser.add_argument(
        "--config",
        default=str(root / "configs" / "nullswap_floor_rerun.json"),
    )
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    main_dir = Path(cfg["baseline_main_dir"])
    required = (
        main_dir / "frozen_main_config.json",
        main_dir / "protocol_manifest.json",
        main_dir / "protections" / "manifest.jsonl",
        main_dir / "results" / "per_pair.jsonl",
        main_dir / "four_evaluator_supplement"
        / "per_pair_four_evaluators.jsonl",
        main_dir / "robustness_v07" / "per_pair_transformed.jsonl",
    )
    for path in required:
        _require_file(path)

    base_cfg = json.loads(required[0].read_text(encoding="utf-8"))
    protocol = json.loads(required[1].read_text(encoding="utf-8"))
    expected_models = ["simswap", "blendswap", "faceshifter"]
    if base_cfg.get("swap_models") != expected_models:
        raise RuntimeError(
            f"Frozen swap models changed: {base_cfg.get('swap_models')}"
        )
    if "nullswap" not in base_cfg.get("methods", []):
        raise RuntimeError("Frozen main config does not contain NullSwap")
    if len(protocol.get("source_identities", [])) != 200:
        raise RuntimeError("Expected 200 frozen source identities")
    if len(protocol.get("pairs", [])) != 1000:
        raise RuntimeError("Expected 1000 frozen source-target pairs")
    per_source = Counter(
        int(row["source_identity"]) for row in protocol["pairs"]
    )
    if set(per_source.values()) != {5}:
        raise RuntimeError("Frozen protocol is not five targets per source")
    if sum(int(row["target_index"]) == 0 for row in protocol["pairs"]) != 200:
        raise RuntimeError("Frozen target-index-0 robustness subset is invalid")

    project_src = Path(base_cfg["project_root"]) / "src"
    if not project_src.is_dir():
        raise FileNotFoundError(project_src)
    sys.path.insert(0, str(project_src))

    from paper_exp.nullswap_floor_refinement import (
        RefinementConfig,
        VisualMetrics,
        candidate_dominates_anchor,
        project_nonincreasing_l2_direction,
        quantize_png01,
    )
    from reliable_face_defense_rpp.propagation import (
        DifferentiableTransformRegistry,
    )

    refinement = RefinementConfig.from_dict(cfg["refinement"])
    refinement.validate()
    registry = DifferentiableTransformRegistry.from_03bc_hard_red_pool_with_ste()
    missing_environments = set(refinement.environments) - set(registry.mapping)
    if missing_environments:
        raise RuntimeError(
            f"Missing differentiable transforms: {sorted(missing_environments)}"
        )

    x = torch.tensor([0.0, 0.1, 0.5001, 1.0]).view(1, 1, 2, 2)
    if not torch.equal(quantize_png01(x), quantize_png01(quantize_png01(x))):
        raise AssertionError("8-bit quantization is not idempotent")
    clean = torch.zeros(1, 1, 2, 2)
    current = torch.ones_like(clean)
    projected = project_nonincreasing_l2_direction(
        torch.ones_like(clean), current, clean
    )
    if float(((current - clean) * projected).sum()) > 1e-7:
        raise AssertionError("L2 half-space projection points outward")

    anchor_visual = VisualMetrics(30.0, 0.95, 0.02, 100.0)
    accepted, checks = candidate_dominates_anchor(
        anchor_visual=anchor_visual,
        candidate_visual=VisualMetrics(30.1, 0.951, 0.019, 99.0),
        anchor_signature={"arcface::clean": 0.20, "facenet::clean": 0.18},
        candidate_signature={"arcface::clean": 0.19, "facenet::clean": 0.17},
        cfg=refinement,
    )
    if not accepted or not checks["conditionwise_no_regression"]:
        raise AssertionError("Strict NullSwap-floor gate rejected valid fixture")

    if base_cfg.get("device") == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Frozen rerun requires CUDA, but CUDA is unavailable")
    print(
        "NULLSWAP_FLOOR_SELFCHECK=PASS "
        f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
        "sources=200 pairs=1000",
        flush=True,
    )


if __name__ == "__main__":
    main()
