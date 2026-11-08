from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "configs" / "main_experiment.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    from paper_exp.runner import run
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
