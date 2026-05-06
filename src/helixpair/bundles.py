from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from helixpair.constants import DEFAULT_ANCHOR_WINDOW, DEFAULT_GAP_BINS, DEFAULT_OFFSETS, ORIENTATION_TO_INDEX
from helixpair.data import save_tensor_dict
from helixpair.features import generate_pair_table
from helixpair.io_utils import ensure_dir, read_json, read_table, resolve_path, split_token, write_json, write_table
from helixpair.sequence import (
    build_pair_record,
    build_interface_tensor,
    build_offset_anchor_tensor,
    geometry_features,
    one_hot_encode,
    pfm_to_pwm,
    rerandomize_exclusive_cores,
    scan_pwm,
    shuffle_interface_region,
)
from helixpair.splits import assign_split


def _scenario_root(project_root: Path, scenario: str, subdir: str) -> Path:
    return ensure_dir(project_root / subdir / scenario)


def _pad_or_crop(sequence: str, window_length: int = 96) -> str:
    sequence = sequence.upper()
    if len(sequence) == window_length:
        return sequence
    if len(sequence) > window_length:
        start = max((len(sequence) - window_length) // 2, 0)
        return sequence[start : start + window_length]
    pad_total = window_length - len(sequence)
    left = pad_total // 2
    right = pad_total - left
    return ("N" * left) + sequence + ("N" * right)


def build_windows(project_root: str | Path, scenario: str = "real", window_length: int = 96, source_name: str = "sequences.parquet"):
    project_root = resolve_path(project_root)
    scenario_root = _scenario_root(project_root, scenario, "data_intermediate")
    source_path = scenario_root / source_name
    if not source_path.exists():
        raise FileNotFoundError(f"Missing scenario sequence table: {source_path}")
    sequences = read_table(source_path).copy()
    sequences["sequence"] = sequences["sequence"].astype(str).map(lambda seq: _pad_or_crop(seq, window_length=window_length))
    sequences["center"] = window_length // 2
    if "chromosome" not in sequences.columns:
        sequences["chromosome"] = scenario
    output_path = scenario_root / "window_table.parquet"
    write_table(sequences, output_path)
    return sequences


def _load_primary_motif_library(project_root: str | Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    tf_master = read_table(resolve_path(project_root) / "data_intermediate" / "tf_master_table.tsv")
    motif_positions = read_table(resolve_path(project_root) / "data_intermediate" / "motif_matrix_positions.tsv")
    library: dict[str, np.ndarray] = {}
    tf_to_gene = {}
    for row in tf_master.itertuples(index=False):
        motif_id = str(getattr(row, "motif_id", ""))
        if not motif_id:
            continue
        motif_rows = motif_positions[motif_positions["motif_id"] == motif_id].sort_values("position")
        if motif_rows.empty:
            continue
        pfm = motif_rows[["A", "C", "G", "T"]].to_numpy(dtype=np.float32).T
        library[str(row.tf_id)] = pfm_to_pwm(pfm)
        tf_to_gene[str(row.tf_id)] = str(row.gene_symbol)
    return library, tf_to_gene


def _index_maps(tf_master):
    tf_values = sorted(tf_master["tf_id"].dropna().astype(str).unique())
    family_values = sorted(tf_master["family"].fillna("").astype(str).unique())
    subfamily_values = sorted(tf_master["subfamily"].fillna("").astype(str).unique())
    paralog_values = sorted(tf_master["paralog_group"].fillna("").astype(str).unique())
    return {
        "tf_id": {value: index for index, value in enumerate(tf_values)},
        "family": {value: index for index, value in enumerate(family_values)},
        "subfamily": {value: index for index, value in enumerate(subfamily_values)},
        "paralog_group": {value: index for index, value in enumerate(paralog_values)},
    }


def _tf_row(tf_master, gene_symbol: str):
    rows = tf_master[tf_master["gene_symbol"] == gene_symbol]
    if rows.empty:
        return None
    return rows.iloc[0]


def _vector_from_columns(row, prefix: str, width: int) -> np.ndarray:
    values = []
    for index in range(width):
        value = getattr(row, f"{prefix}_{index}", 0.0)
        values.append(float(value))
    return np.asarray(values, dtype=np.float32)


def _gap_one_hot(edge_gap: float, bins: Sequence[int] = DEFAULT_GAP_BINS) -> np.ndarray:
    gap = int(round(edge_gap))
    gap = min(max(gap, int(bins[0])), int(bins[-1]))
    index = int(gap - int(bins[0]))
    vector = np.zeros((len(bins),), dtype=np.float32)
    vector[index] = 1.0
    return vector


def build_windows_and_candidates(
    project_root: str | Path,
    scenario: str = "real",
    window_length: int = 96,
    top_k_anchors: int = 24,
    top_k_pairs: int = 64,
    cutoff: float = -20.0,
):
    import pandas as pd

    project_root = resolve_path(project_root)
    scenario_root = _scenario_root(project_root, scenario, "data_intermediate")
    windows = build_windows(project_root, scenario=scenario, window_length=window_length)
    tf_master = read_table(project_root / "data_intermediate" / "tf_master_table.tsv")
    pwm_library, _ = _load_primary_motif_library(project_root)
    gene_to_tf_id = {row.gene_symbol: row.tf_id for row in tf_master.itertuples(index=False)}

    anchor_records = []
    direct_pair_records = []
    target_anchor_limit = min(int(top_k_anchors), 2)
    for row in windows.itertuples(index=False):
        target_tfs = [token for token in str(getattr(row, "tf_pair_label", "")).split("::") if token]
        allowed = list(
            dict.fromkeys(
                [gene_to_tf_id[token] for token in target_tfs if token in gene_to_tf_id and gene_to_tf_id[token] in pwm_library]
            )
        )
        anchors_by_tf: dict[str, list] = {}
        for tf_id in allowed:
            scanned = scan_pwm(row.sequence, pwm_library[tf_id], tf_id=tf_id, top_k=top_k_anchors, cutoff=cutoff)
            anchors_by_tf[tf_id] = scanned
            for anchor in scanned:
                anchor_records.append(
                    {
                        "seq_id": row.seq_id,
                        "tf_id": anchor.tf_id,
                        "start": anchor.start,
                        "end": anchor.end,
                        "strand": anchor.strand,
                        "score": anchor.score,
                        "offset_margin": anchor.offset_margin,
                    }
                )
        if (
            hasattr(row, "left_anchor_start")
            and hasattr(row, "right_anchor_start")
            and pd.notna(getattr(row, "left_anchor_start"))
            and pd.notna(getattr(row, "left_anchor_end"))
            and pd.notna(getattr(row, "right_anchor_start"))
            and pd.notna(getattr(row, "right_anchor_end"))
        ):
            direct_pair_records.append(
                {
                    "seq_id": row.seq_id,
                    "pair_id": f"{row.seq_id}::direct",
                    "left_tf": gene_to_tf_id.get(str(getattr(row, "left_tf", "")), ""),
                    "right_tf": gene_to_tf_id.get(str(getattr(row, "right_tf", "")), ""),
                    "left_start": int(getattr(row, "left_anchor_start")),
                    "left_end": int(getattr(row, "left_anchor_end")),
                    "right_start": int(getattr(row, "right_anchor_start")),
                    "right_end": int(getattr(row, "right_anchor_end")),
                    "orientation": str(getattr(row, "orientation", "++")),
                    "center_distance": float(getattr(row, "center_distance")),
                    "edge_gap": float(getattr(row, "edge_gap")),
                    "overlap_len": float(getattr(row, "overlap_len")),
                    "coarse_additive_score": float(getattr(row, "coarse_additive_score", 0.0)),
                }
            )
            continue
        if len(target_tfs) >= 2:
            left_tf_id = gene_to_tf_id.get(target_tfs[0], "")
            right_tf_id = gene_to_tf_id.get(target_tfs[-1], "")
            left_candidates = anchors_by_tf.get(left_tf_id, [])
            right_candidates = anchors_by_tf.get(right_tf_id, [])
            targeted_pairs = []
            if left_tf_id and right_tf_id:
                if left_tf_id == right_tf_id and left_candidates:
                    best_anchor = left_candidates[0]
                    targeted_pairs.append(
                        {
                            "seq_id": row.seq_id,
                            "pair_id": f"{row.seq_id}::target::0",
                            "left_tf": left_tf_id,
                            "right_tf": right_tf_id,
                            "left_start": int(best_anchor.start),
                            "left_end": int(best_anchor.end),
                            "right_start": int(best_anchor.start),
                            "right_end": int(best_anchor.end),
                            "orientation": f"{best_anchor.strand}{best_anchor.strand}",
                            "center_distance": 0.0,
                            "edge_gap": float(-(best_anchor.end - best_anchor.start)),
                            "overlap_len": float(best_anchor.end - best_anchor.start),
                            "coarse_additive_score": float(2.0 * best_anchor.score),
                        }
                    )
                elif left_candidates and right_candidates:
                    for left_anchor in left_candidates[:target_anchor_limit]:
                        for right_anchor in right_candidates[:target_anchor_limit]:
                            pair = build_pair_record(str(row.seq_id), left_anchor, right_anchor)
                            targeted_pairs.append(
                                {
                                    "seq_id": pair.seq_id,
                                    "pair_id": "",
                                    "left_tf": pair.left_tf,
                                    "right_tf": pair.right_tf,
                                    "left_start": pair.left_start,
                                    "left_end": pair.left_end,
                                    "right_start": pair.right_start,
                                    "right_end": pair.right_end,
                                    "orientation": pair.orientation,
                                    "center_distance": pair.center_distance,
                                    "edge_gap": pair.edge_gap,
                                    "overlap_len": pair.overlap_len,
                                    "coarse_additive_score": pair.coarse_additive_score,
                                }
                            )
                    targeted_pairs = sorted(targeted_pairs, key=lambda item: float(item["coarse_additive_score"]), reverse=True)[:top_k_pairs]
                    for pair_index, record in enumerate(targeted_pairs):
                        record["pair_id"] = f"{row.seq_id}::target::{pair_index}"
            if targeted_pairs:
                direct_pair_records.extend(targeted_pairs)
                continue
    anchors = pd.DataFrame.from_records(anchor_records)
    pairs = pd.DataFrame.from_records(direct_pair_records)
    scanned_pairs = pd.DataFrame(columns=["seq_id"])
    if not anchors.empty:
        targeted_seq_ids = {str(record["seq_id"]) for record in direct_pair_records}
        fallback_anchors = anchors[~anchors["seq_id"].astype(str).isin(targeted_seq_ids)].copy()
        if not fallback_anchors.empty:
            scanned_pairs = generate_pair_table(fallback_anchors, top_k_pairs=top_k_pairs)
    if pairs.empty:
        pairs = scanned_pairs
    elif not scanned_pairs.empty:
        pairs = pd.concat([pairs, scanned_pairs], ignore_index=True, sort=False).drop_duplicates(
            ["seq_id", "left_tf", "right_tf", "left_start", "right_start", "orientation"]
        )
    write_table(anchors, scenario_root / "anchors.parquet")
    write_table(pairs, scenario_root / "pairs.parquet")
    return windows, anchors, pairs


def _flatten_split_groups(grouped_examples: dict[str, list[tuple[str, list[dict]]]]) -> dict[str, list[dict]]:
    return {
        split: [example for _group, items in grouped_examples[split] for example in items]
        for split in ["train", "valid", "test"]
    }


def _move_group(grouped_examples: dict[str, list[tuple[str, list[dict]]]], source: str, target: str) -> bool:
    if source == target or not grouped_examples[source]:
        return False
    grouped_examples[target].append(grouped_examples[source].pop())
    return True


def _split_examples(
    examples: list[dict],
    split_groups: list[str],
    preserve_group_competition: bool = False,
) -> dict[str, list[dict]]:
    if preserve_group_competition:
        grouped: dict[str, list[dict]] = {}
        for example, group in zip(examples, split_groups):
            grouped.setdefault(str(group), []).append(example)
        grouped_examples: dict[str, list[tuple[str, list[dict]]]] = {"train": [], "valid": [], "test": []}
        for group, items in grouped.items():
            split = assign_split(str(group), train_fraction=0.7, valid_fraction=0.15)
            grouped_examples[split].append((group, items))
        if not grouped_examples["valid"] and grouped_examples["train"]:
            _move_group(grouped_examples, "train", "valid")
        if not grouped_examples["test"] and len(grouped_examples["train"]) > 1:
            _move_group(grouped_examples, "train", "test")
        if not grouped_examples["train"]:
            for fallback in ["valid", "test"]:
                if _move_group(grouped_examples, fallback, "train"):
                    break
        if not grouped_examples["valid"] and grouped_examples["train"]:
            _move_group(grouped_examples, "train", "valid")
        return _flatten_split_groups(grouped_examples)
    split_examples = {"train": [], "valid": [], "test": []}
    for example, group in zip(examples, split_groups):
        label_token = example.get("_metadata", {}).get("label")
        group_key = f"{group}::label::{label_token}" if label_token is not None else str(group)
        split = assign_split(group_key, train_fraction=0.7, valid_fraction=0.15)
        split_examples[split].append(example)
    if not split_examples["valid"] and len(split_examples["train"]) > 2:
        split_examples["valid"].append(split_examples["train"].pop())
    if not split_examples["test"] and len(split_examples["train"]) > 3:
        split_examples["test"].append(split_examples["train"].pop())
    if not split_examples["train"]:
        for fallback in ["valid", "test"]:
            if split_examples[fallback]:
                split_examples["train"].append(split_examples[fallback].pop())
                break
    if not split_examples["valid"] and len(split_examples["train"]) > 1:
        split_examples["valid"].append(split_examples["train"].pop())
    return _ensure_binary_label_support(split_examples)


def _example_label(example: dict) -> float | None:
    label = example.get("_metadata", {}).get("label")
    if label is None:
        return None
    return float(label)


def _move_label_example(split_examples: dict[str, list[dict]], source: str, target: str, label: float) -> bool:
    if source == target or len(split_examples[source]) <= 1:
        return False
    label_indices = [index for index, example in enumerate(split_examples[source]) if _example_label(example) == label]
    if len(label_indices) <= 1:
        return False
    split_examples[target].append(split_examples[source].pop(label_indices[0]))
    return True


def _ensure_binary_label_support(split_examples: dict[str, list[dict]]) -> dict[str, list[dict]]:
    labels = sorted({_example_label(example) for values in split_examples.values() for example in values if _example_label(example) is not None})
    if len(labels) < 2:
        return split_examples
    for target in ["valid", "train", "test"]:
        present = {_example_label(example) for example in split_examples[target]}
        for label in labels:
            if label in present:
                continue
            for source in ["train", "test", "valid"]:
                if _move_label_example(split_examples, source, target, label):
                    present.add(label)
                    break
    return split_examples


def _bundle_split_group(row, scenario: str, phase: str) -> str:
    if scenario in {"synthetic", "ablation"}:
        return str(getattr(row, "seq_id"))
    if phase == "phase4" and hasattr(row, "state_label") and pd.notna(getattr(row, "state_label")):
        return str(getattr(row, "state_label"))
    return str(getattr(row, "split_group", getattr(row, "tf_pair_label", row.seq_id)))


def _resolve_split_manifest(
    project_root: Path,
    scenario: str,
    split_name: str,
    explicit_split_manifest: str | Path | None,
) -> Path | None:
    if explicit_split_manifest:
        candidate = resolve_path(explicit_split_manifest)
        return candidate if candidate.exists() else None
    if split_token(split_name) == "default":
        return None
    candidate = project_root / "splits" / scenario / f"{split_name}.parquet"
    return candidate if candidate.exists() else None


def _load_split_lookup(split_manifest: Path | None) -> dict[str, dict[str, str]]:
    if split_manifest is None:
        return {"seq_id": {}, "group_value": {}}
    frame = read_table(split_manifest)
    if "split" not in frame.columns:
        raise ValueError(f"Split manifest must contain seq_id/split columns: {split_manifest}")
    seq_lookup = {}
    if "seq_id" in frame.columns:
        seq_frame = frame.drop_duplicates("seq_id", keep="first")
        seq_lookup = {str(row.seq_id): str(row.split) for row in seq_frame.itertuples(index=False)}
    group_lookup = {}
    if "group_value" in frame.columns:
        group_frame = frame.drop_duplicates("group_value", keep="first")
        group_lookup = {str(row.group_value): str(row.split) for row in group_frame.itertuples(index=False)}
    return {"seq_id": seq_lookup, "group_value": group_lookup}


def build_tensor_bundles(
    project_root: str | Path,
    scenario: str = "real",
    window_length: int = 96,
    anchor_window: int = DEFAULT_ANCHOR_WINDOW,
    offsets: Sequence[int] = DEFAULT_OFFSETS,
    availability_dim: int = 16,
    state_dim: int = 32,
    split_name: str = "default",
    split_manifest: str | Path | None = None,
    phases: Sequence[str] | None = None,
    helical_order: int = 2,
    spline_bins: int = 8,
    interface_flank: int = 4,
    include_shape_channels: bool = True,
    allow_bridge_core_leakage: bool = False,
) -> dict[str, str]:
    project_root = resolve_path(project_root)
    scenario_root = _scenario_root(project_root, scenario, "data_intermediate")
    processed_root = ensure_dir(project_root / "data_processed" / scenario)
    windows = read_table(scenario_root / "window_table.parquet")
    pairs = read_table(scenario_root / "pairs.parquet")
    tf_master = read_table(project_root / "data_intermediate" / "tf_master_table.tsv")
    index_maps = _index_maps(tf_master)
    split_lookup = _load_split_lookup(_resolve_split_manifest(project_root, scenario, split_name, split_manifest))
    split_dir = split_token(split_name)

    pair_lookup = {}
    if not pairs.empty:
        for seq_id, frame in pairs.groupby("seq_id"):
            pair_lookup[str(seq_id)] = frame.sort_values("coarse_additive_score", ascending=False)

    outputs: dict[str, str] = {}
    bundle_manifest: dict[str, dict[str, int]] = {}
    selected_phases = set(str(phase) for phase in phases) if phases is not None else None
    phase_values = sorted(windows["phase"].astype(str).unique())
    for phase in phase_values:
        if selected_phases is not None and phase not in selected_phases:
            continue
        phase_frame = windows[windows["phase"].astype(str) == phase].copy()
        state_group_lookup = {
            state_label: index for index, state_label in enumerate(sorted(phase_frame["state_label"].astype(str).unique()))
        } if "state_label" in phase_frame.columns else {}
        examples: list[dict] = []
        split_groups: list[str] = []
        for row in phase_frame.itertuples(index=False):
            target_tokens = [token for token in str(getattr(row, "tf_pair_label", "")).split("::") if token]
            if len(target_tokens) < 2:
                continue
            left_gene, right_gene = target_tokens[0], target_tokens[-1]
            left_tf = _tf_row(tf_master, left_gene)
            right_tf = _tf_row(tf_master, right_gene)
            if left_tf is None or right_tf is None:
                continue
            pair_frame = pair_lookup.get(str(row.seq_id))
            pair = None
            if pair_frame is not None and not pair_frame.empty:
                target_pair = pair_frame[
                    ((pair_frame["left_tf"] == left_tf.tf_id) & (pair_frame["right_tf"] == right_tf.tf_id))
                    | ((pair_frame["left_tf"] == right_tf.tf_id) & (pair_frame["right_tf"] == left_tf.tf_id))
                ]
                if not target_pair.empty:
                    pair = target_pair.sort_values("coarse_additive_score", ascending=False).iloc[0]
            if (
                pair is None
                and hasattr(row, "left_anchor_start")
                and pd.notna(getattr(row, "left_anchor_start"))
                and pd.notna(getattr(row, "left_anchor_end"))
                and pd.notna(getattr(row, "right_anchor_start"))
                and pd.notna(getattr(row, "right_anchor_end"))
            ):
                pair = row
            if pair is None:
                continue

            left_start = int(getattr(pair, "left_start", getattr(row, "left_anchor_start", 0)))
            left_end = int(getattr(pair, "left_end", getattr(row, "left_anchor_end", left_start + 1)))
            right_start = int(getattr(pair, "right_start", getattr(row, "right_anchor_start", left_start)))
            right_end = int(getattr(pair, "right_end", getattr(row, "right_anchor_end", right_start + 1)))
            orientation = str(getattr(pair, "orientation", getattr(row, "orientation", "++")))
            center_distance = float(getattr(pair, "center_distance"))
            edge_gap = float(getattr(pair, "edge_gap"))
            overlap_len = float(getattr(pair, "overlap_len"))

            left_anchor_offsets = build_offset_anchor_tensor(
                row.sequence,
                left_start,
                left_end,
                offsets=offsets,
                anchor_window=anchor_window,
                include_shape_channels=include_shape_channels,
            )
            right_anchor_offsets = build_offset_anchor_tensor(
                row.sequence,
                right_start,
                right_end,
                offsets=offsets,
                anchor_window=anchor_window,
                include_shape_channels=include_shape_channels,
            )
            interface = build_interface_tensor(
                row.sequence,
                left_start,
                left_end,
                right_start,
                right_end,
                flank=interface_flank,
                use_shape_channels=include_shape_channels,
                mask_exclusive_anchor_core=not allow_bridge_core_leakage,
            ).astype(np.float32)
            shuffled_sequence = shuffle_interface_region(row.sequence, left_start, left_end, right_start, right_end)
            core_rerandomized_sequence = rerandomize_exclusive_cores(row.sequence, left_start, left_end, right_start, right_end)
            interface_shuffle = build_interface_tensor(
                shuffled_sequence,
                left_start,
                left_end,
                right_start,
                right_end,
                flank=interface_flank,
                use_shape_channels=include_shape_channels,
                mask_exclusive_anchor_core=not allow_bridge_core_leakage,
            ).astype(np.float32)
            interface_core_rerandomized = build_interface_tensor(
                core_rerandomized_sequence,
                left_start,
                left_end,
                right_start,
                right_end,
                flank=interface_flank,
                use_shape_channels=include_shape_channels,
                mask_exclusive_anchor_core=not allow_bridge_core_leakage,
            ).astype(np.float32)
            availability = _vector_from_columns(row, "availability", availability_dim)
            state_context = _vector_from_columns(row, "state", state_dim)
            example = {
                "window_sequence": one_hot_encode(row.sequence).astype(np.float32),
                "left_anchor_offsets": left_anchor_offsets.astype(np.float32),
                "right_anchor_offsets": right_anchor_offsets.astype(np.float32),
                "geometry_features": geometry_features(
                    center_distance,
                    edge_gap,
                    overlap_len,
                    orientation,
                    order=helical_order,
                    bins=spline_bins,
                ).astype(np.float32),
                "interface_tensor": interface,
                "interface_tensor_shuffle": interface_shuffle,
                "interface_tensor_core_rerandomized": interface_core_rerandomized,
                "spacing_target": _gap_one_hot(edge_gap),
                "orientation_target": np.eye(len(ORIENTATION_TO_INDEX), dtype=np.float32)[ORIENTATION_TO_INDEX[orientation]],
                "composite_target": np.asarray(float(getattr(row, "composite_label", overlap_len > 0)), dtype=np.float32),
                "labels": np.asarray(float(getattr(row, "label")), dtype=np.float32),
                "compatibility": np.asarray(float(np.exp(-max(0.0, overlap_len - 4.0))), dtype=np.float32),
                "availability": availability,
                "state_context": state_context,
                "state_group_id": np.asarray(state_group_lookup.get(str(getattr(row, "state_label", row.seq_id)), -1), dtype=np.int64),
                "left_family_id": int(index_maps["family"][str(left_tf.family)]),
                "left_subfamily_id": int(index_maps["subfamily"][str(left_tf.subfamily)]),
                "left_paralog_id": int(index_maps["paralog_group"][str(left_tf.paralog_group)]),
                "left_tf_id": int(index_maps["tf_id"][str(left_tf.tf_id)]),
                "right_family_id": int(index_maps["family"][str(right_tf.family)]),
                "right_subfamily_id": int(index_maps["subfamily"][str(right_tf.subfamily)]),
                "right_paralog_id": int(index_maps["paralog_group"][str(right_tf.paralog_group)]),
                "right_tf_id": int(index_maps["tf_id"][str(right_tf.tf_id)]),
                "scenario_id": np.asarray({"synthetic": 0, "ablation": 1, "real": 2}.get(scenario, 0), dtype=np.int64),
                "phase_id": np.asarray({"phase1": 1, "phase2": 2, "phase3": 3, "phase4": 4, "phase5": 5}.get(phase, 0), dtype=np.int64),
                "_metadata": {
                    "seq_id": str(row.seq_id),
                    "phase": phase,
                    "scenario": scenario,
                    "split_name": split_name,
                    "left_tf": left_gene,
                    "right_tf": right_gene,
                    "label": float(getattr(row, "label")),
                    "state_label": str(getattr(row, "state_label", "")),
                    "source_dataset": str(getattr(row, "source_dataset", scenario)),
                    "construction_mode": str(getattr(row, "construction_mode", "")),
                    "phase4_label_source": str(getattr(row, "phase4_label_source", "")),
                    "phase4_evidence_supported": float(getattr(row, "phase4_evidence_supported", float("nan"))),
                    "orientation": orientation,
                    "edge_gap": edge_gap,
                    "overlap_len": overlap_len,
                    "composite_label": float(getattr(row, "composite_label", overlap_len > 0)),
                },
            }
            examples.append(example)
            split_groups.append(_bundle_split_group(row, scenario, phase))
        if not examples:
            continue
        if split_lookup["seq_id"] or split_lookup["group_value"]:
            split_examples = {"train": [], "valid": [], "test": []}
            for example, fallback_group in zip(examples, split_groups):
                split = split_lookup["seq_id"].get(str(example["_metadata"]["seq_id"]))
                if split not in split_examples:
                    split = split_lookup["group_value"].get(str(fallback_group))
                if split not in split_examples:
                    label_token = example.get("_metadata", {}).get("label")
                    group_key = f"{fallback_group}::label::{label_token}" if label_token is not None else str(fallback_group)
                    split = assign_split(group_key, train_fraction=0.7, valid_fraction=0.15)
                split_examples[split].append(example)
            if not split_examples["train"] or not split_examples["valid"]:
                split_examples = _split_examples(examples, split_groups, preserve_group_competition=(phase == "phase4"))
        else:
            split_examples = _split_examples(examples, split_groups, preserve_group_competition=(phase == "phase4"))
        bundle_manifest[phase] = {name: len(values) for name, values in split_examples.items()}
        phase_root = ensure_dir(processed_root / phase / split_dir)
        for split_partition, values in split_examples.items():
            if not values:
                continue
            tensor_dict = {}
            metadata_rows = []
            for key in values[0]:
                if key == "_metadata":
                    metadata_rows = [item[key] for item in values]
                    continue
                stacked = np.stack([np.asarray(item[key]) for item in values])
                dtype = torch.long if stacked.dtype.kind in {"i", "u"} else torch.float32
                tensor_dict[key] = torch.tensor(stacked, dtype=dtype)
            out_path = phase_root / f"{split_partition}_bundle.pt"
            save_tensor_dict(out_path, tensor_dict)
            if metadata_rows:
                write_table(pd.DataFrame.from_records(metadata_rows), phase_root / f"{split_partition}_metadata.parquet")
            outputs[f"{scenario}_{phase}_{split_dir}_{split_partition}"] = str(out_path)
    reports_root = ensure_dir(project_root / "reports")
    manifest_path = reports_root / f"bundle_manifest_{scenario}_{split_dir}.json"
    if manifest_path.exists():
        existing_manifest = read_json(manifest_path)
    else:
        existing_manifest = {}
    existing_manifest.update(bundle_manifest)
    write_json(manifest_path, existing_manifest)
    return outputs
