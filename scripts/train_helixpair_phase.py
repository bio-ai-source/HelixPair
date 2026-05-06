from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from helixpair.config import load_config
from helixpair.training import train_phase


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one HelixPair staged phase.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", required=True, choices=["phase1", "phase2", "phase3", "phase4", "phase5"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device is not None:
        config["runtime"]["device"] = args.device
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)
    if args.seed is not None:
        config["runtime"]["seed"] = int(args.seed)

    metrics = train_phase(config, phase=args.phase)
    print(metrics)


if __name__ == "__main__":
    main()
