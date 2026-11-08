from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Add ArcFace/FaceNet metrics without overwriting the main run."
    )
    parser.add_argument(
        "--main-dir",
        default="/root/autodl-tmp/reliable_paper_main_v07",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "configs/main_experiment.json"),
        help="Fallback only when the completed run lacks frozen_main_config.json.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    from paper_exp.four_evaluator_supplement import run

    print(json.dumps(run(args.main_dir, args.config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
