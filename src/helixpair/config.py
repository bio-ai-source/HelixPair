from __future__ import annotations

import copy
import os
import string
from pathlib import Path
from typing import Any

from helixpair.constants import PROJECT_ROOT


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded or {}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def _resolve_default_path(config_dir: Path, relative_default: str) -> Path:
    candidate = Path(relative_default)
    if candidate.is_absolute():
        return candidate
    local = config_dir / candidate
    if local.exists():
        return local
    return PROJECT_ROOT / "configs" / candidate


def _normalize_paths(config: dict[str, Any]) -> dict[str, Any]:
    paths = config.setdefault("paths", {})
    raw_project_root = paths.get("project_root", PROJECT_ROOT)
    project_root = Path(raw_project_root)
    if not project_root.is_absolute():
        project_root = PROJECT_ROOT if str(project_root) in {"", "."} else (PROJECT_ROOT / project_root).resolve()
    paths["project_root"] = str(project_root)
    paths.setdefault("data_raw", str(project_root / "data_raw"))
    paths.setdefault("data_intermediate", str(project_root / "data_intermediate"))
    paths.setdefault("data_processed", str(project_root / "data_processed"))
    paths.setdefault("splits", str(project_root / "splits"))
    paths.setdefault("checkpoints", str(project_root / "checkpoints"))
    paths.setdefault("results", str(project_root / "results"))
    paths.setdefault("figures", str(project_root / "figures"))
    paths.setdefault("reports", str(project_root / "reports"))
    paths.setdefault("external", str(project_root / "external"))
    return config


def _normalize_runtime(config: dict[str, Any]) -> dict[str, Any]:
    runtime = config.setdefault("runtime", {})
    runtime.setdefault("device", "cuda")
    runtime.setdefault("amp", True)
    runtime.setdefault("scenario", "real")
    runtime.setdefault("split_name", "default")
    runtime.setdefault("seed", 11)
    return config


def _normalize_results(config: dict[str, Any]) -> dict[str, Any]:
    results = config.setdefault("results", {})
    root = Path(config["paths"]["results"])
    results.setdefault("root", str(root))
    results.setdefault("ledger_path", str(root / "run_ledger.jsonl"))
    results.setdefault("summary_path", str(root / "main_tables" / "main_metrics.json"))
    return config


def _normalize_training(config: dict[str, Any]) -> dict[str, Any]:
    training = config.setdefault("training", {})
    training.setdefault("batch_size", 128)
    training.setdefault("lr", 1e-4)
    training.setdefault("weight_decay", 1e-2)
    training.setdefault("epochs", 40)
    training.setdefault("early_stop_patience", 8)
    training.setdefault("gradient_clip", 1.0)
    training.setdefault("num_workers", 0)
    training.setdefault("pin_memory", True)
    training.setdefault("gradient_accumulation", 1)
    training.setdefault("init_from_previous_phase", True)
    training.setdefault("init_checkpoint", None)
    training.setdefault("allow_joint_residual_tuning_if_identifiable", False)
    training.setdefault("joint_residual_modules", ["bridge_head.bridge_scale"])
    training.setdefault(
        "freeze_schedule",
        {
            "phase1": ["embedding", "monomer_head"],
            "phase2": ["geometry_head", "bridge_head"],
            "phase3": ["chemical_potential_head"],
            "phase4": ["state_gate_head", "partition_head"],
            "phase5": ["state_gate_head", "bridge_head.bridge_scale"],
        },
    )
    return config


def infer_geometry_dim(helical_order: int, spline_bins: int) -> int:
    return 3 + 4 + (2 * int(helical_order)) + int(spline_bins)


def _normalize_model(config: dict[str, Any]) -> dict[str, Any]:
    model = config.setdefault("model", {})
    model.setdefault("num_families", 2048)
    model.setdefault("num_subfamilies", 2048)
    model.setdefault("num_paralogs", 2048)
    model.setdefault("num_tfs", 4096)
    model.setdefault("availability_dim", 16)
    model.setdefault("state_dim", 32)
    model.setdefault("sequence_channels", 9)
    model.setdefault("embedding_dim", 32)
    model.setdefault("rank", 8)
    model.setdefault("interface_channels", 13)
    model.setdefault("num_gap_bins", 25)
    model.setdefault("helical_order", 2)
    model.setdefault("spline_bins", 8)
    model.setdefault("bridge_kernel_size", 7)
    model.setdefault("bridge_hidden_dim", 64)
    model.setdefault("monomer_hidden_dim", 64)
    model.setdefault("state_gate_dropout_p", 0.1)
    model["geometry_dim"] = infer_geometry_dim(model["helical_order"], model["spline_bins"])
    return config


def _normalize_data_policy(config: dict[str, Any]) -> dict[str, Any]:
    data = config.setdefault("data", {})
    data.setdefault("window_length", 96)
    data.setdefault("top_k_anchors", 24)
    data.setdefault("top_k_pairs", 64)
    data.setdefault("offset_margin", 2)
    data.setdefault("interface_flank", 4)
    data.setdefault("split_manifest", "")
    acquisition = data.setdefault("acquisition", {})
    acquisition.setdefault(
        "cap_selex",
        {
            "max_sequences_per_file": None,
            "minimum_pair_edges": 8,
            "minimum_pairs": 16,
            "formal_phase1_per_tf": 1024,
            "formal_phase2_per_pair_label": 2048,
        },
    )
    acquisition.setdefault(
            "state_layer",
            {
                "require_candidate_sequences": True,
                "max_examples": None,
                "allow_public_portal_build": True,
                "allow_proxy_from_get": False,
                "proxy_examples": 512,
                "max_fragment_files": 4,
                "max_candidate_regions": 4096,
                "max_states": 256,
                "state_selection_multiplier": 12,
                "max_pairs_per_state": 2,
                "max_monomers_per_state": 1,
                "max_fragment_lines": None,
                "impute_missing_monomers": True,
                "impute_missing_pairs": True,
                "exclude_imputed_examples": True,
                "select_best_fragment_file_count": True,
                "formal_only": True,
            },
        )
    return config


def _normalize_ablations(config: dict[str, Any]) -> dict[str, Any]:
    ablations = config.setdefault("ablations", {})
    ablations.setdefault("disable_hierarchy_embedding", False)
    ablations.setdefault("disable_anchor_refinement", False)
    ablations.setdefault("disable_geometry_head", False)
    ablations.setdefault("disable_bridge_head", False)
    ablations.setdefault("disable_state_gate", False)
    ablations.setdefault("disable_partition_head", False)
    ablations.setdefault("disable_availability", False)
    ablations.setdefault("disable_helical_basis", False)
    ablations.setdefault("disable_shape_channels", False)
    ablations.setdefault("disable_uncertainty_ensemble", False)
    ablations.setdefault("allow_bridge_core_leakage", False)
    ablations.setdefault("enable_signed_gain_diagnostic", False)
    return config


def _normalize_split_policy(config: dict[str, Any]) -> dict[str, Any]:
    splits = config.setdefault("splits", {})
    splits.setdefault("tf_master_table", str(Path(config["paths"]["data_intermediate"]) / "tf_master_table.tsv"))
    splits.setdefault(
        "policy",
        {
            "paralog_aware": True,
            "family_level": "family",
            "subfamily_level": "subfamily",
            "state_group_column": "state_label",
            "sequence_group_column": "chromosome",
        },
    )
    return config


def _normalize_audits(config: dict[str, Any]) -> dict[str, Any]:
    audits = config.setdefault("audits", {})
    audits.setdefault("bridge", {"min_interface_drop": 0.05, "max_core_sensitivity": 0.1})
    audits.setdefault("identifiability", {"require_non_bipartite": True})
    audits.setdefault("availability_gap", {"matched_gain_floor": 0.0})
    return config


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    config = _normalize_paths(config)
    config = _normalize_runtime(config)
    config = _normalize_results(config)
    config = _normalize_model(config)
    config = _normalize_training(config)
    config = _normalize_data_policy(config)
    config = _normalize_ablations(config)
    config = _normalize_split_policy(config)
    config = _normalize_audits(config)
    config = _render_templates(config)
    return config


class _SafeTemplateDict(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _template_context(config: dict[str, Any]) -> dict[str, str]:
    paths = config.get("paths", {})
    runtime = config.get("runtime", {})
    return {
        "project_root": str(paths.get("project_root", PROJECT_ROOT)),
        "data_raw": str(paths.get("data_raw", Path(paths.get("project_root", PROJECT_ROOT)) / "data_raw")),
        "data_intermediate": str(paths.get("data_intermediate", Path(paths.get("project_root", PROJECT_ROOT)) / "data_intermediate")),
        "data_processed": str(paths.get("data_processed", Path(paths.get("project_root", PROJECT_ROOT)) / "data_processed")),
        "splits_root": str(paths.get("splits", Path(paths.get("project_root", PROJECT_ROOT)) / "splits")),
        "checkpoints": str(paths.get("checkpoints", Path(paths.get("project_root", PROJECT_ROOT)) / "checkpoints")),
        "results": str(paths.get("results", Path(paths.get("project_root", PROJECT_ROOT)) / "results")),
        "figures": str(paths.get("figures", Path(paths.get("project_root", PROJECT_ROOT)) / "figures")),
        "reports": str(paths.get("reports", Path(paths.get("project_root", PROJECT_ROOT)) / "reports")),
        "external": str(paths.get("external", Path(paths.get("project_root", PROJECT_ROOT)) / "external")),
        "scenario": str(runtime.get("scenario", "real")),
        "split_name": str(runtime.get("split_name", "default")),
        "seed": str(runtime.get("seed", 11)),
    }


def _render_templates(value: Any, context: dict[str, str] | None = None) -> Any:
    if context is None and isinstance(value, dict):
        context = _template_context(value)
    if context is None:
        context = {}
    if isinstance(value, dict):
        return {key: _render_templates(item, context=context) for key, item in value.items()}
    if isinstance(value, list):
        return [_render_templates(item, context=context) for item in value]
    if isinstance(value, str):
        formatter = string.Formatter()
        field_names = [field_name for _, field_name, _, _ in formatter.parse(value) if field_name]
        if not field_names:
            return value
        return value.format_map(_SafeTemplateDict(context))
    return value


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    defaults = config.pop("defaults", [])
    merged: dict[str, Any] = {}
    for relative_default in defaults:
        merged = deep_update(merged, load_config(_resolve_default_path(config_path.parent, relative_default)))
    return normalize_config(_expand_env(deep_update(merged, config)))


def dump_config(config: dict[str, Any], path: str | Path) -> None:
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
