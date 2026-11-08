from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="NullSwap-floor candidate generation and frozen-protocol rerun"
    )
    parser.add_argument(
        "--config",
        default=str(root / "configs" / "nullswap_floor_rerun.json"),
    )
    parser.add_argument(
        "--stage",
        choices=("protections", "main", "all"),
        default="all",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.output_dir is not None:
        cfg["output_dir"] = str(args.output_dir)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")

    sys.path.insert(0, str(root))
    from paper_exp.nullswap_floor_rerun import run

    result = run(cfg, stage=args.stage, limit=args.limit)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

