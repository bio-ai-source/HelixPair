from __future__ import annotations

import json
from pathlib import Path

from helixpair.io_utils import read_table, resolve_path


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _count_reference_edges(project_root: Path, acceptance: dict) -> int:
    if "reference_edges" in acceptance:
        return int(acceptance["reference_edges"])
    reference_graph_path = project_root / "data_intermediate" / "reference_graph.parquet"
    if reference_graph_path.exists():
        return int(len(read_table(reference_graph_path)))
    return 0


def _count_usable_pairs(project_root: Path, acceptance: dict) -> int:
    if "usable_pair_rows" in acceptance:
        return int(acceptance["usable_pair_rows"])
    inventory_path = project_root / "data_intermediate" / "cap_selex_pair_inventory_usable.tsv"
    if inventory_path.exists():
        inventory = read_table(inventory_path)
        if "pair_usable" in inventory.columns:
            return int(inventory["pair_usable"].fillna(False).astype(bool).sum())
        return int(len(inventory))
    return 0


def assert_cap_selex_ready(
    project_root: str | Path,
    minimum_pair_edges: int = 1,
    minimum_pairs: int = 1,
) -> None:
    project_root = resolve_path(project_root)
    acceptance = _load_json(project_root / "reports" / "data_acceptance" / "cap_selex_acceptance.json")
    reference_edges = _count_reference_edges(project_root, acceptance)
    usable_pair_rows = _count_usable_pairs(project_root, acceptance)
    if reference_edges < minimum_pair_edges or usable_pair_rows < minimum_pairs:
        raise RuntimeError(
            "CAP-SELEX assets are insufficient for formal training: "
            f"reference_edges={reference_edges} (need >= {minimum_pair_edges}), "
            f"usable_pair_rows={usable_pair_rows} (need >= {minimum_pairs})."
        )


def assert_state_layer_ready(project_root: str | Path, allow_proxy: bool = False) -> None:
    project_root = resolve_path(project_root)
    inventory_path = project_root / "data_intermediate" / "real" / "state_inventory.json"
    inventory = _load_json(inventory_path)
    if not inventory:
        raise FileNotFoundError(f"Missing state inventory: {inventory_path}")
    status = str(inventory.get("status", ""))
    if status == "proxy_prepared" and not allow_proxy:
        raise RuntimeError(
            "Real state-layer assets are still in GET_proxy mode. "
            "Formal Phase III/IV/V runs require real candidate_sequences/state_features tables."
        )
    if status not in {"prepared", "proxy_prepared"}:
        raise RuntimeError(f"State-layer assets are not ready: status={status or 'missing'}.")
    if int(inventory.get("num_sequences", 0)) <= 0 or int(inventory.get("num_states", 0)) <= 0:
        raise RuntimeError("State-layer inventory is empty; formal Phase III/IV/V runs cannot proceed.")


def validate_phase_readiness(config: dict, phase: str) -> None:
    if str(config.get("runtime", {}).get("scenario", "")) != "real":
        return
    project_root = config["paths"]["project_root"]
    acquisition = config.get("data", {}).get("acquisition", {})
    cap_selex_cfg = acquisition.get("cap_selex", {})
    state_cfg = acquisition.get("state_layer", {})

    if phase in {"phase2", "phase4", "phase5"}:
        assert_cap_selex_ready(
            project_root,
            minimum_pair_edges=int(cap_selex_cfg.get("minimum_pair_edges", 1)),
            minimum_pairs=int(cap_selex_cfg.get("minimum_pairs", 1)),
        )
    if phase in {"phase3", "phase4", "phase5"}:
        assert_state_layer_ready(
            project_root,
            allow_proxy=bool(state_cfg.get("allow_proxy_from_get", False)),
        )
