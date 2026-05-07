from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.runner import run_experiment_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an Etsin hyperspectral experiment from a config file.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML or JSON experiment configuration file.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_experiment_from_config(Path(args.config))
    print("Experiment completed.")
    print(summary)


if __name__ == "__main__":
    main()
