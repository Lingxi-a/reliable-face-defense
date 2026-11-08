from __future__ import annotations

import csv

from paper_exp.four_evaluator_supplement import (
    ALL_EVALUATORS,
    _copy_non_replay_row,
    summarize_four_evaluators,
    write_nullswap_style_tables,
)


def _original(status, detected):
    return {
        "swap_model": "simswap",
        "method": "ours",
        "pair_id": "p0",
        "status": status,
        "detection_success": detected,
        "vggface_acc1": False,
        "vggface_acc5": False,
        "sface_acc1": False,
        "sface_acc5": False,
    }


def test_non_replay_failure_policy():
    row = _copy_non_replay_row(_original("SOURCE_DETECTION_FAILED", False))
    assert row["arcface_acc1"] is False
    assert row["facenet_acc5"] is False

    row = _copy_non_replay_row(_original("ENGINEERING_ERROR", None))
    assert row["arcface_acc1"] is True
    assert row["facenet_acc5"] is True


def test_four_evaluator_table_keeps_models_and_visual_metrics(tmp_path):
    cfg = {
        "swap_models": ["simswap", "blendswap", "faceshifter"],
        "methods": ["no_protection", "ours"],
    }
    rows = []
    for model in cfg["swap_models"]:
        for method in cfg["methods"]:
            row = {
                "swap_model": model,
                "method": method,
                "pair_id": "p0",
                "status": "SWAP_OK",
                "detection_success": True,
            }
            for evaluator in ALL_EVALUATORS:
                row[f"{evaluator}_acc1"] = method == "no_protection"
                row[f"{evaluator}_acc5"] = True
            rows.append(row)
    protections = [
        {
            "method": method,
            "status": "OK",
            "psnr": 30.0 if method == "ours" else float("inf"),
            "ssim": 0.95 if method == "ours" else 1.0,
            "lpips": 0.02 if method == "ours" else 0.0,
        }
        for method in cfg["methods"]
    ]

    table, eligible = summarize_four_evaluators(
        cfg, protections, rows, tmp_path
    )
    assert len(table) == 3 * 4 * 2 + 4 * 2
    assert {x["swap_model"] for x in table} == {
        "simswap", "blendswap", "faceshifter", "平均"
    }
    assert all(len(ids) == 1 for ids in eligible.values())
    ours = next(
        x for x in table
        if x["swap_model"] == "simswap"
        and x["evaluator"] == "arcface"
        and x["method"] == "ours"
    )
    assert ours["acc1"] == 0.0
    assert ours["acc5"] == 1.0
    assert ours["psnr"] == 30.0
    assert ours["ssim"] == 0.95
    assert ours["lpips"] == 0.02

    with (tmp_path / "main_table_four_evaluators.csv").open(
        encoding="utf-8-sig"
    ) as f:
        disk_rows = list(csv.DictReader(f))
    assert len(disk_rows) == len(table)

    source_rows = []
    for method in cfg["methods"]:
        row = {"method": method, "source_identity": 1}
        for evaluator in ALL_EVALUATORS:
            row[f"{evaluator}_acc1"] = method == "no_protection"
            row[f"{evaluator}_acc5"] = True
        source_rows.append(row)
    paper = write_nullswap_style_tables(
        cfg, protections, source_rows, rows, eligible, tmp_path
    )
    assert len(paper["visual_quality"]) == 2
    assert len(paper["source_cloaking"]) == 5
    assert len(paper["face_swap_acc1"]) == 4
    assert len(paper["face_swap_acc5"]) == 4
    assert (tmp_path / "paper_table_1_visual_quality.csv").is_file()
    assert (tmp_path / "paper_table_2_source_cloaking.csv").is_file()
    assert (tmp_path / "paper_table_3_face_swap_acc1.csv").is_file()
