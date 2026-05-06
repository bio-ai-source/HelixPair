from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from helixpair.bundles import _gap_one_hot, _index_maps, _tf_row, _vector_from_columns
from helixpair.config import load_config
from helixpair.inference import load_model_for_inference
from helixpair.io_utils import ensure_dir, read_table, write_table, write_text
from helixpair.sequence import (
    build_interface_tensor,
    build_offset_anchor_tensor,
    geometry_features,
    one_hot_encode,
)
from helixpair.training import _forward_model


REPORTER_CONFIG = "configs/phase5_functional/reporter_pairranked_tune_partition_lr1e4_pen1.yaml"
REPORTER_PREDICTIONS = "results/per_example_predictions/phase5/real/reporter_pairranked_tune_partition_lr1e4_pen1/seed_11.parquet"
REPORTER_SCENARIO_ROOT = "data_intermediate/phase5_reporter_pairranked"


def _load_reporter_assets(project_root: Path) -> dict[str, object]:
    scenario_root = project_root / REPORTER_SCENARIO_ROOT
    window_table = read_table(scenario_root / "window_table.parquet")
    pair_table = read_table(scenario_root / "pairs.parquet")
    tf_master = read_table(project_root / "data_intermediate" / "tf_master_table.tsv")
    predictions = read_table(project_root / REPORTER_PREDICTIONS)
    checkpoint_path = Path(str(predictions["checkpoint_path"].iloc[0]))
    config = load_config(project_root / REPORTER_CONFIG)
    index_maps = _index_maps(tf_master)
    return {
        "config": config,
        "checkpoint_path": checkpoint_path,
        "window_table": window_table,
        "pair_table": pair_table,
        "tf_master": tf_master,
        "index_maps": index_maps,
    }


def _pick_row(window_table: pd.DataFrame, seq_id: str) -> pd.Series:
    frame = window_table[window_table["seq_id"].astype(str) == seq_id]
    if frame.empty:
        raise KeyError(f"Missing reporter window row for {seq_id}")
    return frame.iloc[0]


def _pick_pair(pair_table: pd.DataFrame, seq_id: str, orientation: str, edge_gap: float) -> pd.Series:
    frame = pair_table[pair_table["seq_id"].astype(str) == seq_id].copy()
    if frame.empty:
        raise KeyError(f"Missing candidate pair rows for {seq_id}")
    exact = frame[
        (frame["orientation"].astype(str) == str(orientation))
        & (frame["edge_gap"].astype(float) == float(edge_gap))
    ]
    if not exact.empty:
        return exact.sort_values("coarse_additive_score", ascending=False).iloc[0]
    return frame.sort_values("coarse_additive_score", ascending=False).iloc[0]


def _randomize_region(sequence: str, start: int, end: int, rng: np.random.Generator) -> str:
    if end <= start:
        return sequence
    gc = (sequence.count("G") + sequence.count("C")) / max(len(sequence), 1)
    alphabet = np.array(list("ACGT"))
    at = (1.0 - gc) / 2.0
    gc_prob = gc / 2.0
    probs = np.array([at, gc_prob, gc_prob, at], dtype=np.float64)
    replacement = "".join(rng.choice(alphabet, size=end - start, p=probs))
    return sequence[:start] + replacement + sequence[end:]


def _embed(sequence: str, insert: str, start: int) -> str:
    return sequence[:start] + insert + sequence[start + len(insert) :]


def _build_example(
    row: pd.Series,
    pair: pd.Series,
    tf_master: pd.DataFrame,
    index_maps: dict,
    availability_dim: int,
    state_dim: int,
    scenario: str,
    split_name: str,
    sequence_override: str | None = None,
    left_start_override: int | None = None,
    right_start_override: int | None = None,
) -> dict[str, torch.Tensor]:
    sequence = str(sequence_override if sequence_override is not None else row["sequence"])
    left_gene, right_gene = str(row["tf_pair_label"]).split("::", 1)
    left_tf = _tf_row(tf_master, left_gene)
    right_tf = _tf_row(tf_master, right_gene)
    if left_tf is None or right_tf is None:
        raise KeyError(f"Unable to resolve TF rows for {row['tf_pair_label']}")

    left_start = int(left_start_override if left_start_override is not None else pair["left_start"])
    left_end = left_start + int(pair["left_end"]) - int(pair["left_start"])
    right_start = int(right_start_override if right_start_override is not None else pair["right_start"])
    right_end = right_start + int(pair["right_end"]) - int(pair["right_start"])
    orientation = str(pair["orientation"])
    overlap_len = float(max(0, min(left_end, right_end) - max(left_start, right_start)))
    if right_start >= left_start:
        edge_gap = float(right_start - left_end)
    else:
        edge_gap = float(right_start - left_end)
    center_distance = float((right_start + right_end) / 2.0 - (left_start + left_end) / 2.0)

    availability = _vector_from_columns(row, "availability", int(availability_dim))
    state_context = _vector_from_columns(row, "state", int(state_dim))

    example = {
        "window_sequence": torch.tensor(one_hot_encode(sequence).astype(np.float32)).unsqueeze(0),
        "left_anchor_offsets": torch.tensor(
            build_offset_anchor_tensor(sequence, left_start, left_end, include_shape_channels=True).astype(np.float32)
        ).unsqueeze(0),
        "right_anchor_offsets": torch.tensor(
            build_offset_anchor_tensor(sequence, right_start, right_end, include_shape_channels=True).astype(np.float32)
        ).unsqueeze(0),
        "geometry_features": torch.tensor(
            geometry_features(center_distance, edge_gap, overlap_len, orientation, order=2, bins=8).astype(np.float32)
        ).unsqueeze(0),
        "interface_tensor": torch.tensor(
            build_interface_tensor(sequence, left_start, left_end, right_start, right_end, flank=4, use_shape_channels=True).astype(np.float32)
        ).unsqueeze(0),
        "labels": torch.tensor([float(row["label"])], dtype=torch.float32),
        "availability": torch.tensor(availability.astype(np.float32)).unsqueeze(0),
        "state_context": torch.tensor(state_context.astype(np.float32)).unsqueeze(0),
        "state_group_id": torch.tensor([0], dtype=torch.int64),
        "left_family_id": torch.tensor([int(index_maps["family"][str(left_tf.family)])], dtype=torch.int64),
        "left_subfamily_id": torch.tensor([int(index_maps["subfamily"][str(left_tf.subfamily)])], dtype=torch.int64),
        "left_paralog_id": torch.tensor([int(index_maps["paralog_group"][str(left_tf.paralog_group)])], dtype=torch.int64),
        "left_tf_id": torch.tensor([int(index_maps["tf_id"][str(left_tf.tf_id)])], dtype=torch.int64),
        "right_family_id": torch.tensor([int(index_maps["family"][str(right_tf.family)])], dtype=torch.int64),
        "right_subfamily_id": torch.tensor([int(index_maps["subfamily"][str(right_tf.subfamily)])], dtype=torch.int64),
        "right_paralog_id": torch.tensor([int(index_maps["paralog_group"][str(right_tf.paralog_group)])], dtype=torch.int64),
        "right_tf_id": torch.tensor([int(index_maps["tf_id"][str(right_tf.tf_id)])], dtype=torch.int64),
        "scenario_id": torch.tensor([{"synthetic": 0, "ablation": 1, "real": 2}.get(scenario, 0)], dtype=torch.int64),
        "phase_id": torch.tensor([5], dtype=torch.int64),
        "spacing_target": torch.tensor(_gap_one_hot(edge_gap).astype(np.float32)).unsqueeze(0),
        "orientation_target": torch.tensor([[1.0 if token == orientation else 0.0 for token in ["++", "+-", "-+", "--"]]], dtype=torch.float32),
        "composite_target": torch.tensor([float(row.get("composite_label", 0.0))], dtype=torch.float32),
        "compatibility": torch.tensor([float(math.exp(-max(0.0, overlap_len - 4.0)))], dtype=torch.float32),
    }
    example["_metadata"] = {
        "seq_id": str(row["seq_id"]),
        "split_name": split_name,
        "orientation": orientation,
        "edge_gap": edge_gap,
        "overlap_len": overlap_len,
        "sequence": sequence,
    }
    return example


def _score_example(model, device: torch.device, example: dict[str, torch.Tensor]) -> dict[str, float]:
    batch = {key: value.to(device) for key, value in example.items() if not key.startswith("_")}
    with torch.no_grad():
        outputs = _forward_model(model, batch)
        full_score = outputs.usage_probability if outputs.usage_probability is not None else torch.sigmoid(
            -(outputs.intrinsic_monomer_energy + outputs.biochemical_residual)
        )
        availability = outputs.availability_only_probability
        result = {
            "full_score": float(full_score[0].detach().cpu()),
            "geometry_residual": float(outputs.geometry_residual[0].detach().cpu()),
            "bridge_residual": float(outputs.bridge_residual[0].detach().cpu()),
            "biochemical_residual": float(outputs.biochemical_residual[0].detach().cpu()),
        }
        if availability is not None:
            result["availability_only_score"] = float(availability[0].detach().cpu())
        if outputs.state_gate is not None:
            result["state_gate"] = float(outputs.state_gate[0].detach().cpu())
            result["state_correction"] = float(outputs.state_correction[0].detach().cpu())
        return result


def _geometry_counterfactual(
    model,
    device: torch.device,
    assets: dict[str, object],
    high_seq_id: str,
    low_seq_id: str,
) -> pd.DataFrame:
    window_table = assets["window_table"]
    pair_table = assets["pair_table"]
    tf_master = assets["tf_master"]
    index_maps = assets["index_maps"]
    config = assets["config"]
    availability_dim = int(config["model"]["availability_dim"])
    state_dim = int(config["model"]["state_dim"])
    high_row = _pick_row(window_table, high_seq_id)
    low_row = _pick_row(window_table, low_seq_id)
    pair = _pick_pair(pair_table, high_seq_id, "++", 11.0)
    left_start = int(pair["left_start"])
    left_end = int(pair["left_end"])
    right_start = int(pair["right_start"])
    right_end = int(pair["right_end"])
    left_motif = str(high_row["sequence"])[left_start:left_end]
    right_motif = str(high_row["sequence"])[right_start:right_end]
    grid = [-3, 1, 5, 9, 11, 15, 19, 23]
    records: list[dict[str, object]] = []
    for gap in grid:
        new_right_start = left_end + int(gap)
        if new_right_start < left_end + 1 or new_right_start + len(right_motif) > len(str(high_row["sequence"])) - 1:
            continue
        rng = np.random.default_rng(1000 + int(gap))
        edited = _randomize_region(str(high_row["sequence"]), left_start, left_end, rng)
        edited = _randomize_region(edited, right_start, right_end, rng)
        edited = _embed(edited, left_motif, left_start)
        edited = _embed(edited, right_motif, new_right_start)
        for state_name, row in [("high", high_row), ("low", low_row)]:
            example = _build_example(
                row,
                pair,
                tf_master,
                index_maps,
                availability_dim,
                state_dim,
                scenario="real",
                split_name="counterfactual_geometry",
                sequence_override=edited,
                left_start_override=left_start,
                right_start_override=new_right_start,
            )
            score = _score_example(model, device, example)
            records.append(
                {
                    "variant": "geometry_counterfactual",
                    "seq_id": str(row["seq_id"]),
                    "state_mode": state_name,
                    "gap": int(gap),
                    "orientation": "++",
                    "sequence": edited,
                    **score,
                }
            )
    return pd.DataFrame.from_records(records)


def _interface_minimal_edit(
    model,
    device: torch.device,
    assets: dict[str, object],
    high_seq_id: str,
    low_seq_id: str,
    orientation: str,
    edge_gap: float,
) -> pd.DataFrame:
    window_table = assets["window_table"]
    pair_table = assets["pair_table"]
    tf_master = assets["tf_master"]
    index_maps = assets["index_maps"]
    config = assets["config"]
    availability_dim = int(config["model"]["availability_dim"])
    state_dim = int(config["model"]["state_dim"])
    high_row = _pick_row(window_table, high_seq_id)
    low_row = _pick_row(window_table, low_seq_id)
    pair = _pick_pair(pair_table, high_seq_id, orientation, edge_gap)
    left_end = int(pair["left_end"])
    right_start = int(pair["right_start"])
    base_sequence = str(high_row["sequence"])
    editable_positions = list(range(min(left_end, right_start), max(left_end, right_start)))
    records: list[dict[str, object]] = []
    for position in editable_positions:
        original = base_sequence[position]
        for base in "ACGT":
            if base == original:
                continue
            edited = base_sequence[:position] + base + base_sequence[position + 1 :]
            pair_scores = []
            for state_name, row in [("high", high_row), ("low", low_row)]:
                example = _build_example(
                    row,
                    pair,
                    tf_master,
                    index_maps,
                    availability_dim,
                    state_dim,
                    scenario="real",
                    split_name="counterfactual_interface",
                    sequence_override=edited,
                )
                score = _score_example(model, device, example)
                pair_scores.append({"state_mode": state_name, **score})
            margin = pair_scores[0]["full_score"] - pair_scores[1]["full_score"]
            records.append(
                {
                    "variant": "interface_single_edit",
                    "position": int(position),
                    "ref_base": original,
                    "alt_base": base,
                    "edited_sequence": edited,
                    "high_full_score": pair_scores[0]["full_score"],
                    "low_full_score": pair_scores[1]["full_score"],
                    "high_low_margin": margin,
                    "high_bridge_residual": pair_scores[0]["bridge_residual"],
                    "low_bridge_residual": pair_scores[1]["bridge_residual"],
                    "pair_orientation": str(pair["orientation"]),
                    "pair_gap": float(pair["edge_gap"]),
                }
            )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    baseline_margin = float(
        _score_example(
            model,
            device,
            _build_example(
                high_row,
                pair,
                tf_master,
                index_maps,
                availability_dim,
                state_dim,
                scenario="real",
                split_name="counterfactual_interface",
            ),
        )["full_score"]
        - _score_example(
            model,
            device,
            _build_example(
                low_row,
                pair,
                tf_master,
                index_maps,
                availability_dim,
                state_dim,
                scenario="real",
                split_name="counterfactual_interface",
            ),
        )["full_score"]
    )
    frame["margin_drop"] = baseline_margin - frame["high_low_margin"].astype(float)
    frame = frame.sort_values(["margin_drop", "high_low_margin"], ascending=[False, True]).reset_index(drop=True)
    return frame


def _ensemble_uncertainty(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensemble_paths = [
        project_root / "results" / "per_example_predictions" / "phase5" / "real" / "reporter_pairranked_tune_partition_lr1e4_pen1" / "seed_11.parquet",
        project_root / "results" / "per_example_predictions" / "phase5" / "real" / "reporter_pairranked_tune_partition_lr1e4_pen1_seed17" / "seed_17.parquet",
        project_root / "results" / "per_example_predictions" / "phase5" / "real" / "reporter_pairranked_tune_partition_lr1e4_pen1_seed23" / "seed_23.parquet",
        project_root / "results" / "per_example_predictions" / "phase5" / "real" / "reporter_pairranked_tune_partition_lr1e4_pen1_seed37" / "seed_37.parquet",
    ]
    merged = None
    for seed_index, path in enumerate(ensemble_paths, start=1):
        frame = read_table(path)[["seq_id", "label", "left_tf", "right_tf", "full_score"]].rename(columns={"full_score": f"score_seed_{seed_index}"})
        merged = frame if merged is None else merged.merge(frame, on=["seq_id", "label", "left_tf", "right_tf"], how="inner")
    score_columns = [column for column in merged.columns if column.startswith("score_seed_")]
    merged["ensemble_mean"] = merged[score_columns].mean(axis=1)
    merged["ensemble_std"] = merged[score_columns].std(axis=1, ddof=0)
    merged["predicted_label"] = (merged["ensemble_mean"].astype(float) >= 0.5).astype(float)
    merged["correct"] = (merged["predicted_label"] == merged["label"].astype(float)).astype(float)
    selective = merged.sort_values("ensemble_std").reset_index(drop=True)
    coverage_records = []
    for keep in range(1, len(selective) + 1):
        kept = selective.iloc[:keep]
        accuracy = float(kept["correct"].mean())
        coverage_records.append(
            {
                "coverage": keep / len(selective),
                "accuracy": accuracy,
                "selective_risk": 1.0 - accuracy,
                "kept_examples": int(keep),
            }
        )
    return merged, pd.DataFrame.from_records(coverage_records)


def run_reporter_counterfactuals(project_root: str | Path) -> dict[str, str]:
    project_root = Path(project_root)
    assets = _load_reporter_assets(project_root)
    config = assets["config"]
    checkpoint_path = assets["checkpoint_path"]
    model, device = load_model_for_inference(config, checkpoint_path)

    output_dir = ensure_dir(project_root / "reports" / "counterfactuals")
    geometry = _geometry_counterfactual(model, device, assets, "pairranked::sequence__4::high", "pairranked::sequence__4::low")
    interface = _interface_minimal_edit(
        model,
        device,
        assets,
        "pairranked::sequence__5::high",
        "pairranked::sequence__5::low",
        "-+",
        -29.0,
    )
    uncertainty, selective = _ensemble_uncertainty(project_root)

    geometry_path = output_dir / "geometry_counterfactual.csv"
    interface_path = output_dir / "interface_minimal_edit.csv"
    uncertainty_path = output_dir / "ensemble_uncertainty.csv"
    selective_path = output_dir / "selective_risk_curve.csv"
    write_table(geometry, geometry_path)
    write_table(interface, interface_path)
    write_table(uncertainty, uncertainty_path)
    write_table(selective, selective_path)

    summary_lines = [
        "# Reporter Counterfactual Summary",
        "",
        f"- Geometry rows: {len(geometry)}",
        f"- Interface single-edit candidates: {len(interface)}",
        f"- Ensemble shared examples: {len(uncertainty)}",
    ]
    if not interface.empty:
        top = interface.iloc[0]
        summary_lines.append(
            f"- Best single-base interface edit: pos {int(top['position'])} {top['ref_base']}->{top['alt_base']} with margin drop {float(top['margin_drop']):.4f}"
        )
    summary_path = output_dir / "counterfactual_summary.md"
    write_text(summary_path, "\n".join(summary_lines))
    return {
        "geometry_counterfactual": str(geometry_path),
        "interface_minimal_edit": str(interface_path),
        "ensemble_uncertainty": str(uncertainty_path),
        "selective_risk_curve": str(selective_path),
        "summary": str(summary_path),
    }
