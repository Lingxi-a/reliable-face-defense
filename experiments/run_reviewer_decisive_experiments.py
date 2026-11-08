from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the reviewer-decisive experiments under the frozen paper protocol"
    )
    parser.add_argument("--main-output", default="/root/autodl-tmp/reliable_paper_main_v07")
    parser.add_argument(
        "--final-output",
        default="/root/autodl-tmp/reliable_paper_v07_nullswap_floor_rerun",
    )
    parser.add_argument(
        "--four-evaluator-output",
        default="/root/autodl-tmp/four_evaluator_supplement_v08",
    )
    parser.add_argument(
        "--output",
        default="/root/autodl-tmp/reviewer_decisive_experiments_v04",
    )
    parser.add_argument(
        "--stage",
        choices=("preflight", "protections", "all"),
        default="all",
    )
    parser.add_argument("--source-limit", type=int, default=None)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from paper_exp.reviewer_decisive_experiments import run

    result = run(
        main_output=args.main_output,
        final_output=args.final_output,
        four_evaluator_output=args.four_evaluator_output,
        output=args.output,
        source_limit=args.source_limit,
        stage=args.stage,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

