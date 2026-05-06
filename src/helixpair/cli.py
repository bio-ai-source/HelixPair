from __future__ import annotations

import argparse
from pathlib import Path

from helixpair.bundles import build_tensor_bundles, build_windows_and_candidates
from helixpair.config import load_config
from helixpair.features import build_baseline_feature_tables
from helixpair.harmonize import harmonize_cap_selex
from helixpair.ingest import build_fallback_tf_catalog, materialize_motif_sources
from helixpair.io_utils import write_table
from helixpair.splits import create_all_splits
from helixpair.state_data import prepare_real_state_data
from helixpair.synthetic import prepare_synthetic_scenario
from helixpair.training import train_phase


def _parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True)
    return parser


def build_tf_master_table_main() -> None:
    args = _parser("Build unified TF master table").parse_args()
    config = load_config(args.config)
    tf_master = build_fallback_tf_catalog(config["paths"]["project_root"])
    write_table(tf_master, config["tf_master"]["output"])


def download_motif_sources_main() -> None:
    args = _parser("Materialize motif resources from configured local/downloaded inputs").parse_args()
    config = load_config(args.config)
    materialize_motif_sources(config["paths"]["project_root"])


def harmonize_cap_selex_main() -> None:
    parser = _parser("Harmonize CAP-SELEX usable subset")
    parser.add_argument("--sample-limit", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    acquisition_cfg = config.get("data", {}).get("acquisition", {}).get("cap_selex", {})
    sample_limit = args.sample_limit if args.sample_limit is not None else acquisition_cfg.get("max_sequences_per_file")
    harmonize_cap_selex(
        config["paths"]["project_root"],
        sample_limit_per_file=sample_limit,
        minimum_pair_edges=int(acquisition_cfg.get("minimum_pair_edges", 1)),
        minimum_pairs=int(acquisition_cfg.get("minimum_pairs", 1)),
    )


def prepare_synthetic_main() -> None:
    parser = _parser("Prepare synthetic HelixPair scenario")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    prepare_synthetic_scenario(
        config["paths"]["project_root"],
        window_length=int(config["data"].get("window_length", 96)),
        availability_dim=int(config["model"]["availability_dim"]),
        state_dim=int(config["model"]["state_dim"]),
        seed=args.seed,
        scenario=args.scenario or config["runtime"]["scenario"],
    )


def prepare_real_state_data_main() -> None:
    parser = _parser("Prepare real state-layer scenario inputs")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--allow-proxy", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    acquisition_cfg = config.get("data", {}).get("acquisition", {}).get("state_layer", {})
    prepare_real_state_data(
        config["paths"]["project_root"],
        availability_dim=int(config["model"]["availability_dim"]),
        state_dim=int(config["model"]["state_dim"]),
        max_examples=args.max_examples if args.max_examples is not None else acquisition_cfg.get("max_examples"),
        allow_public_portal_build=bool(acquisition_cfg.get("allow_public_portal_build", True)),
        allow_proxy_from_get=bool(args.allow_proxy) or bool(acquisition_cfg.get("allow_proxy_from_get", False)),
    )


def build_windows_and_candidates_main() -> None:
    args = _parser("Build candidate windows and TF-pair anchors").parse_args()
    config = load_config(args.config)
    build_windows_and_candidates(
        Path(config["paths"]["project_root"]),
        scenario=config["runtime"]["scenario"],
        window_length=int(config["data"].get("window_length", 96)),
        top_k_anchors=int(config["data"].get("top_k_anchors", 24)),
        top_k_pairs=int(config["data"].get("top_k_pairs", 64)),
    )


def make_splits_main() -> None:
    args = _parser("Create HelixPair train/valid/test splits").parse_args()
    config = load_config(args.config)
    project_root = Path(config["paths"]["project_root"])
    scenario = config["runtime"]["scenario"]
    create_all_splits(
        window_path=project_root / "data_intermediate" / scenario / "window_table.parquet",
        pair_path=project_root / "data_intermediate" / scenario / "pairs.parquet",
        tf_master_path=config["splits"]["tf_master_table"],
        output_dir=project_root / "splits" / scenario,
    )


def build_tensor_bundles_main() -> None:
    parser = _parser("Build tensor bundles for HelixPair staged training")
    parser.add_argument("--phase", action="append", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    project_root = Path(config["paths"]["project_root"])
    build_tensor_bundles(
        project_root,
        scenario=config["runtime"]["scenario"],
        window_length=int(config["data"].get("window_length", 96)),
        availability_dim=int(config["model"]["availability_dim"]),
        state_dim=int(config["model"]["state_dim"]),
        split_name=str(config["runtime"].get("split_name", "default")),
        phases=args.phase,
    )


def build_feature_tables_main() -> None:
    args = _parser("Build feature tables used for HelixPair diagnostics").parse_args()
    config = load_config(args.config)
    build_baseline_feature_tables(
        Path(config["paths"]["project_root"]),
        scenario=config["runtime"]["scenario"],
        split_name=str(config["runtime"].get("split_name", "default")),
    )


def train_phase_main(phase: str) -> None:
    args = _parser(f"Train HelixPair {phase}").parse_args()
    config = load_config(args.config)
    metrics = train_phase(config, phase=phase)
    print(metrics)


def train_phase1_main() -> None:
    train_phase_main("phase1")


def train_phase2_main() -> None:
    train_phase_main("phase2")


def train_phase3_main() -> None:
    train_phase_main("phase3")


def train_phase4_main() -> None:
    train_phase_main("phase4")


def train_phase5_main() -> None:
    train_phase_main("phase5")
