from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str((ROOT / "src").resolve()))

from helixpair.external_functional_benchmarks import build_external_functional_benchmarks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build benchmark-ready MPRA panels and endogenous confirmatory registries from external functional sources."
    )
    parser.add_argument("--positive-quantile", type=float, default=0.9)
    parser.add_argument("--negative-quantile", type=float, default=0.1)
    parser.add_argument("--min-replicates", type=int, default=3)
    parser.add_argument("--min-obs-bc", type=int, default=10)
    parser.add_argument("--max-log2-std", type=float, default=0.75)
    parser.add_argument("--max-orientation-gap", type=float, default=1.0)
    parser.add_argument("--hydrate-sequences", action="store_true")
    parser.add_argument("--window-length", type=int, default=96)
    parser.add_argument("--sequence-limit", type=int, default=None)
    args = parser.parse_args()

    manifest = build_external_functional_benchmarks(
        ROOT,
        positive_quantile=float(args.positive_quantile),
        negative_quantile=float(args.negative_quantile),
        min_replicates=int(args.min_replicates),
        min_obs_bc=int(args.min_obs_bc),
        max_log2_std=float(args.max_log2_std),
        max_orientation_gap=float(args.max_orientation_gap),
        hydrate_sequences=bool(args.hydrate_sequences),
        window_length=int(args.window_length),
        sequence_limit=args.sequence_limit,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
