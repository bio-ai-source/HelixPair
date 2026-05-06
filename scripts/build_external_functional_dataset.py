from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str((ROOT / "src").resolve()))

from helixpair.external_functional import build_external_functional_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canonicalize MPRA and endogenous perturbation sources into a shared external functional panel."
    )
    parser.add_argument("--window-length", type=int, default=96)
    parser.add_argument(
        "--hydrate-sequences-for",
        choices=["none", "mpra", "endogenous", "all"],
        default="none",
    )
    parser.add_argument(
        "--sequence-limit",
        type=int,
        default=None,
        help="Optional cap on UCSC sequence hydration requests.",
    )
    parser.add_argument("--resolve-flowfish-files", action="store_true")
    args = parser.parse_args()

    manifest = build_external_functional_dataset(
        ROOT,
        window_length=int(args.window_length),
        hydrate_sequences_for=str(args.hydrate_sequences_for),
        sequence_limit=args.sequence_limit,
        resolve_flowfish_files=bool(args.resolve_flowfish_files),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
