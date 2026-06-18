#!/usr/bin/env python3
"""
Quick-start script: run the baseline experiment with default config.

Usage:
    python scripts/run_baseline.py
    python scripts/run_baseline.py --config config/my_custom.yaml
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from model.src.experiments.runner import load_config, run_experiment, main

if __name__ == "__main__":
    main()
