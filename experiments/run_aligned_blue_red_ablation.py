from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Protocol-aligned Blue-only versus Blue+Red ablation"
    )
    parser.add_argument(
        "--main-output",
        default="/root/autodl-tmp/reliable_paper_main_v07",
    )
    parser.add_argument(
        "--final-output",
        default="/root/autodl-tmp/reliable_paper_v07_nullswap_floor_rerun",
    )
    parser.add_argument(
        "--ablation-output",
        default="/root/autodl-tmp/aligned_blue_red_ablation_v01",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from paper_exp.aligned_blue_red_ablation import run

    result = run(args.main_output, args.final_output, args.ablation_output)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
