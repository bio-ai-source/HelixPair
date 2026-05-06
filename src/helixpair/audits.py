from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from helixpair.io_utils import write_json


def bridge_leakage_audit(predictions_frame) -> dict[str, float]:
    required = {"bridge_score", "bridge_score_interface_shuffle", "bridge_score_core_rerandomized"}
    missing = required.difference(predictions_frame.columns)
    if missing:
        raise ValueError(f"Missing audit columns: {sorted(missing)}")
    audit_frame = predictions_frame[list(required)].dropna()
    if audit_frame.empty:
        return {
            "bridge_leakage_index": 0.0,
            "exclusive_core_sensitivity": 0.0,
        }
    interface_drop = float((audit_frame["bridge_score"] - audit_frame["bridge_score_interface_shuffle"]).mean())
    core_delta = float((audit_frame["bridge_score"] - audit_frame["bridge_score_core_rerandomized"]).abs().mean())
    return {
        "bridge_leakage_index": interface_drop,
        "exclusive_core_sensitivity": core_delta,
    }


def availability_gap_test(predictions_frame, score_col: str = "full_score", baseline_col: str = "availability_only_score") -> dict[str, float]:
    required = {score_col, baseline_col, "availability_stratum", "label"}
    missing = required.difference(predictions_frame.columns)
    if missing:
        raise ValueError(f"Missing availability-gap columns: {sorted(missing)}")
    grouped = predictions_frame.groupby("availability_stratum")
    residual_gains = [float(frame[score_col].mean() - frame[baseline_col].mean()) for _, frame in grouped]
    return {
        "matched_availability_gain_mean": float(np.mean(residual_gains)) if residual_gains else 0.0,
        "matched_availability_gain_std": float(np.std(residual_gains)) if residual_gains else 0.0,
    }


def helical_recovery_score(spacing_emd: float, spacing_kl: float) -> float:
    return float(1.0 / (1.0 + max(float(spacing_emd), 0.0) + max(float(spacing_kl), 0.0)))


def identifiability_audit(reference_frame) -> dict[str, object]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for row in reference_frame.itertuples(index=False):
        left = str(row.left_tf)
        right = str(row.right_tf)
        adjacency[left].add(right)
        adjacency[right].add(left)
    visited = set()
    components = []
    non_bipartite_components = []
    for node in adjacency:
        if node in visited:
            continue
        queue = deque([(node, 0)])
        colors = {node: 0}
        component_nodes = {node}
        bipartite = True
        visited.add(node)
        while queue:
            current, color = queue.popleft()
            for neighbor in adjacency[current]:
                component_nodes.add(neighbor)
                if neighbor not in colors:
                    colors[neighbor] = 1 - color
                    queue.append((neighbor, 1 - color))
                    visited.add(neighbor)
                elif colors[neighbor] == color:
                    bipartite = False
        components.append(sorted(component_nodes))
        if not bipartite:
            non_bipartite_components.append(sorted(component_nodes))
    return {
        "num_components": len(components),
        "num_non_bipartite_components": len(non_bipartite_components),
        "components": components,
        "non_bipartite_components": non_bipartite_components,
        "joint_finetune_allowed": len(components) == len(non_bipartite_components) if components else False,
    }


def write_audit_report(report: dict, output_path: str | Path) -> None:
    write_json(output_path, report)
