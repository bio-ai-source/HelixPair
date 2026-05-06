from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from helixpair.io_utils import ensure_dir, read_table, resolve_path, write_json, write_table
from helixpair.sequence import embed_sequence, random_background


def _candidate_paths(project_root: Path, file_names: list[str]) -> list[Path]:
    roots = [
        project_root / "data_raw" / "state_layer",
        project_root / "data_raw" / "encode" / "screen",
        project_root / "data_intermediate" / "real",
    ]
    paths: list[Path] = []
    for root in roots:
        for file_name in file_names:
            paths.append(root / file_name)
    return paths


def _resolve_first(project_root: Path, file_names: list[str]) -> Path | None:
    for path in _candidate_paths(project_root, file_names):
        if path.exists():
            return path
    return None


def _resource_inventory(project_root: Path) -> dict[str, list[str]]:
    inventories: dict[str, list[str]] = {}
    for name in ["get_resource_inventory.tsv", "catlas_resource_inventory.tsv", "hca_resource_inventory.tsv", "encode_resource_inventory.tsv"]:
        path = project_root / "data_intermediate" / name
        if path.exists():
            frame = read_table(path)
            rel_col = "relative_path" if "relative_path" in frame.columns else frame.columns[0]
            inventories[name] = frame[rel_col].astype(str).head(50).tolist()
    return inventories


def _write_acceptance(scenario_root: Path, payload: dict) -> None:
    acceptance_root = ensure_dir(scenario_root.parent.parent / "reports" / "data_acceptance")
    (acceptance_root / "state_layer_acceptance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _ensure_vectors(frame: pd.DataFrame, prefix: str, width: int) -> pd.DataFrame:
    for index in range(width):
        column = f"{prefix}_{index}"
        if column not in frame.columns:
            frame[column] = 0.0
    return frame


def _load_consensus_map(project_root: Path) -> dict[str, str]:
    tf_master = read_table(project_root / "data_intermediate" / "tf_master_table.tsv")
    motif_positions = read_table(project_root / "data_intermediate" / "motif_matrix_positions.tsv")
    consensus_by_gene: dict[str, str] = {}
    for row in tf_master.itertuples(index=False):
        motif_id = str(getattr(row, "motif_id", ""))
        if not motif_id:
            continue
        positions = motif_positions[motif_positions["motif_id"] == motif_id].sort_values("position")
        if positions.empty:
            continue
        sequence = "".join(positions[["A", "C", "G", "T"]].idxmax(axis=1).tolist())
        consensus_by_gene[str(row.gene_symbol)] = sequence
    return consensus_by_gene


def _normalize_series(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    if values.empty:
        return values
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum - minimum < 1e-8:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return (values - minimum) / (maximum - minimum)


def _load_proxy_reference_pairs(project_root: Path, consensus_by_gene: dict[str, str]) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    reference_graph_path = project_root / "data_intermediate" / "reference_graph.parquet"
    if reference_graph_path.exists():
        frame = read_table(reference_graph_path)
        for row in frame.itertuples(index=False):
            left_tf = str(getattr(row, "left_tf", ""))
            right_tf = str(getattr(row, "right_tf", ""))
            if left_tf in consensus_by_gene and right_tf in consensus_by_gene:
                candidates.append((left_tf, right_tf))
    if candidates:
        return candidates
    inventory_path = project_root / "data_intermediate" / "cap_selex_pair_inventory_usable.tsv"
    if inventory_path.exists():
        frame = read_table(inventory_path)
        usable = frame[frame["pair_usable"].fillna(False).astype(bool)]
        for row in usable.itertuples(index=False):
            left_tf = str(getattr(row, "left_tf", ""))
            right_tf = str(getattr(row, "right_tf", ""))
            if left_tf in consensus_by_gene and right_tf in consensus_by_gene:
                candidates.append((left_tf, right_tf))
    deduped = []
    seen = set()
    for pair in candidates:
        if pair not in seen:
            deduped.append(pair)
            seen.add(pair)
    return deduped


def _build_get_proxy_inputs(
    project_root: Path,
    availability_dim: int,
    state_dim: int,
    max_examples: int | None,
    proxy_examples: int,
    window_length: int = 96,
    seed: int = 11,
) -> tuple[Path, Path]:
    get_root = project_root / "data_raw" / "get_model" / "data"
    ctcf_path = get_root / "ctcf_motif_count.num_celltype_gt_5.feather"
    if not ctcf_path.exists():
        raise FileNotFoundError("GET proxy mode requires ctcf_motif_count.num_celltype_gt_5.feather under data_raw/get_model/data.")
    consensus_by_gene = _load_consensus_map(project_root)
    reference_pairs = _load_proxy_reference_pairs(project_root, consensus_by_gene)
    if not reference_pairs:
        raise FileNotFoundError("GET proxy mode requires at least one reference TF pair with motif consensuses.")

    ctcf = pd.read_feather(ctcf_path).sort_values("num_celltype", ascending=False).reset_index(drop=True)
    example_count = int(max_examples if max_examples is not None else proxy_examples)
    example_count = max(example_count, len(reference_pairs) * 4)
    state_count = min(max(example_count // max(len(reference_pairs), 1), 8), len(ctcf))
    states = ctcf.head(state_count).copy()
    states["state_label"] = states.apply(
        lambda row: f"proxy::{row['Chromosome']}:{int(row['Start'])}-{int(row['End'])}",
        axis=1,
    )
    states["availability_norm"] = _normalize_series(states["num_celltype"])
    states["strand_pos_norm"] = _normalize_series(states["strand_positive"])
    states["strand_neg_norm"] = _normalize_series(states["strand_negative"])
    widths = states["End"].astype(float) - states["Start"].astype(float)
    states["width_norm"] = _normalize_series(widths)
    state_features = pd.DataFrame({"state_label": states["state_label"]})
    for index in range(availability_dim):
        state_features[f"availability_{index}"] = 0.0
    for index in range(state_dim):
        state_features[f"state_{index}"] = 0.0
    state_features["availability_0"] = states["availability_norm"].to_numpy()
    if availability_dim > 1:
        state_features["availability_1"] = states["strand_pos_norm"].to_numpy()
    if availability_dim > 2:
        state_features["availability_2"] = states["strand_neg_norm"].to_numpy()
    state_features["state_0"] = states["width_norm"].to_numpy()
    if state_dim > 1:
        state_features["state_1"] = np.log1p(states["Start"].astype(float)).to_numpy() / np.log1p(float(states["Start"].max()))

    rng = np.random.default_rng(seed)
    threshold = float(states["availability_norm"].median())
    sequence_rows: list[dict] = []
    for index in range(example_count):
        state_row = states.iloc[index % len(states)]
        left_tf, right_tf = reference_pairs[index % len(reference_pairs)]
        pair_mode = index % 3 != 0
        if pair_mode:
            left_motif = consensus_by_gene[left_tf]
            right_motif = consensus_by_gene[right_tf]
            gap = 4 + int(round(8.0 * float(state_row["width_norm"])))
            left_start = 18
            right_start = min(left_start + len(left_motif) + gap, window_length - len(right_motif) - 8)
            sequence = random_background(window_length, rng)
            sequence = embed_sequence(sequence, left_motif, left_start)
            sequence = embed_sequence(sequence, right_motif, right_start)
            label = float(float(state_row["availability_norm"]) >= threshold)
            seq_id = f"real_proxy::phase4::{left_tf}::{right_tf}::{index:05d}"
            sequence_rows.append(
                {
                    "seq_id": seq_id,
                    "sequence": sequence,
                    "state_label": state_row["state_label"],
                    "tf_pair_label": f"{left_tf}::{right_tf}",
                    "left_tf": left_tf,
                    "right_tf": right_tf,
                    "left_anchor_start": left_start,
                    "left_anchor_end": left_start + len(left_motif),
                    "right_anchor_start": right_start,
                    "right_anchor_end": right_start + len(right_motif),
                    "orientation": "++",
                    "center_distance": float((right_start + len(right_motif) / 2) - (left_start + len(left_motif) / 2)),
                    "edge_gap": float(right_start - (left_start + len(left_motif))),
                    "overlap_len": float(max(0, min(left_start + len(left_motif), right_start + len(right_motif)) - max(left_start, right_start))),
                    "coarse_additive_score": float(len(left_motif) + len(right_motif) + label),
                    "usage_label": label,
                    "label": label,
                    "phase": "phase4",
                    "composite_label": 0.0,
                    "element_type": "state_usage_proxy",
                    "source_dataset": "GET_proxy",
                    "phase4_label_version": "proxy_availability_v1",
                    "phase4_label_source": "proxy_state_availability",
                    "phase4_evidence_score": float(label),
                    "phase4_evidence_rank": 1.0,
                    "phase4_evidence_supported": float(label),
                    "split_group": str(state_row["Chromosome"]),
                    "chromosome": str(state_row["Chromosome"]),
                }
            )
        else:
            tf = left_tf if index % 2 == 0 else right_tf
            motif = consensus_by_gene[tf]
            start = 24
            sequence = random_background(window_length, rng)
            sequence = embed_sequence(sequence, motif, start)
            label = float(float(state_row["availability_norm"]) >= threshold)
            seq_id = f"real_proxy::phase3::{tf}::{index:05d}"
            sequence_rows.append(
                {
                    "seq_id": seq_id,
                    "sequence": sequence,
                    "state_label": state_row["state_label"],
                    "tf_pair_label": f"{tf}::{tf}",
                    "left_tf": tf,
                    "right_tf": tf,
                    "left_anchor_start": start,
                    "left_anchor_end": start + len(motif),
                    "right_anchor_start": start,
                    "right_anchor_end": start + len(motif),
                    "orientation": "++",
                    "center_distance": 0.0,
                    "edge_gap": float(-len(motif)),
                    "overlap_len": float(len(motif)),
                    "coarse_additive_score": float(len(motif) + label),
                    "usage_label": label,
                    "label": label,
                    "phase": "phase3",
                    "composite_label": 0.0,
                    "element_type": "monomer_proxy",
                    "source_dataset": "GET_proxy",
                    "phase4_label_version": "",
                    "phase4_label_source": "",
                    "phase4_evidence_score": float("nan"),
                    "phase4_evidence_rank": float("nan"),
                    "phase4_evidence_supported": float("nan"),
                    "split_group": str(state_row["Chromosome"]),
                    "chromosome": str(state_row["Chromosome"]),
                }
            )
    sequence_frame = pd.DataFrame.from_records(sequence_rows)
    real_root = ensure_dir(project_root / "data_intermediate" / "real")
    sequence_path = real_root / "candidate_sequences.parquet"
    state_path = real_root / "state_features.parquet"
    write_table(sequence_frame, sequence_path)
    write_table(state_features, state_path)
    return sequence_path, state_path


def _derive_labels(frame: pd.DataFrame) -> pd.DataFrame:
    if "label" not in frame.columns:
        for candidate in ["usage_label", "occupancy_label", "functional_label"]:
            if candidate in frame.columns:
                frame["label"] = frame[candidate].astype(float)
                break
    if "label" not in frame.columns:
        raise ValueError(
            "State sequence table must include `label` or one of `usage_label`, `occupancy_label`, `functional_label`."
        )
    if "phase" not in frame.columns:
        phase = np.full((len(frame),), "phase4", dtype=object)
        if "functional_label" in frame.columns:
            phase = np.where(frame["functional_label"].astype(float) > 0, "phase5", phase)
        frame["phase"] = phase
    if "split_group" not in frame.columns:
        fallback = "tf_pair_label" if "tf_pair_label" in frame.columns else "seq_id"
        frame["split_group"] = frame[fallback].astype(str)
    if "composite_label" not in frame.columns:
        frame["composite_label"] = 0.0
    if "source_dataset" not in frame.columns:
        frame["source_dataset"] = "state_layer"
    if "element_type" not in frame.columns:
        frame["element_type"] = "state_usage"
    return frame


def _build_monomer_calibration_set(frame: pd.DataFrame) -> pd.DataFrame:
    calibration = frame.copy()
    if "element_type" in calibration.columns:
        calibration = calibration[calibration["element_type"].astype(str).str.contains("monomer|state_usage", case=False, regex=True)]
    if "composite_label" in calibration.columns:
        calibration = calibration[calibration["composite_label"].fillna(0.0).astype(float) <= 0.0]
    if "tf_pair_label" in calibration.columns:
        calibration = calibration[
            calibration["tf_pair_label"].astype(str).map(
                lambda value: len({token for token in value.split("::") if token}) <= 1
            )
            | (calibration["label"].astype(float) <= 0.0)
        ]
    return calibration


def _exclude_imputed_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    if "construction_mode" not in frame.columns:
        return frame, {"excluded_rows": 0, "excluded_phase3_rows": 0, "excluded_phase4_rows": 0}
    construction_mode = frame["construction_mode"].astype(str)
    keep_mask = ~construction_mode.str.contains("imputed", case=False, regex=False)
    filtered = frame.loc[keep_mask].copy()
    excluded = frame.loc[~keep_mask].copy()
    if excluded.empty:
        return filtered, {"excluded_rows": 0, "excluded_phase3_rows": 0, "excluded_phase4_rows": 0}
    phase_counts = excluded["phase"].astype(str).value_counts().to_dict() if "phase" in excluded.columns else {}
    return filtered, {
        "excluded_rows": int(len(excluded)),
        "excluded_phase3_rows": int(phase_counts.get("phase3", 0)),
        "excluded_phase4_rows": int(phase_counts.get("phase4", 0)),
    }


def _can_upgrade_public_state_sequences(frame: pd.DataFrame) -> bool:
    required_columns = {
        "phase",
        "construction_mode",
        "state_label",
        "source_dataset",
        "pair_reference_supported",
    }
    return required_columns.issubset(frame.columns)


def _upgrade_public_state_sequences(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not _can_upgrade_public_state_sequences(frame):
        return frame
    phase4 = frame["phase"].astype(str) == "phase4"
    public_rows = frame["source_dataset"].astype(str).str.contains("HCA_SCREEN_public", na=False)
    missing_new_cols = any(
        column not in frame.columns
        for column in [
            "phase4_label_version",
            "phase4_label_source",
            "phase4_evidence_score",
            "phase4_evidence_rank",
            "phase4_evidence_supported",
        ]
    )
    if not bool((phase4 & public_rows).any()) or not missing_new_cols:
        return frame
    from helixpair.public_state import _assign_phase4_usage_labels

    return _assign_phase4_usage_labels(frame.copy())


def _retain_evidence_qualified_phase4_states(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    phase4_mask = frame["phase"].astype(str) == "phase4"
    if not bool(phase4_mask.any()):
        return frame, {"excluded_phase4_rows_without_evidence": 0, "excluded_phase4_states_without_evidence": 0}

    required_columns = {
        "construction_mode",
        "state_label",
        "label",
        "phase4_label_version",
        "phase4_label_source",
        "phase4_evidence_score",
        "phase4_evidence_rank",
        "phase4_evidence_supported",
    }
    missing_columns = sorted(column for column in required_columns if column not in frame.columns)
    if missing_columns:
        raise ValueError(
            "Phase4 rows require evidence-qualified metadata columns: "
            + ", ".join(missing_columns)
        )

    observed = frame.loc[phase4_mask & frame["construction_mode"].astype(str).eq("observed_pair")].copy()
    if observed.empty:
        return frame.loc[~phase4_mask].copy(), {
            "excluded_phase4_rows_without_evidence": int(phase4_mask.sum()),
            "excluded_phase4_states_without_evidence": int(frame.loc[phase4_mask, "state_label"].astype(str).nunique()),
        }

    qualified_states: set[str] = set()
    for state_label, state_frame in observed.groupby("state_label", sort=False):
        labels = state_frame["label"].astype(float)
        evidence_supported = state_frame["phase4_evidence_supported"].astype(float)
        if (
            len(state_frame) >= 2
            and bool((labels > 0.0).any())
            and bool((labels <= 0.0).any())
            and evidence_supported.nunique(dropna=False) > 1
        ):
            qualified_states.add(str(state_label))

    keep_mask = (~phase4_mask) | frame["state_label"].astype(str).isin(qualified_states)
    excluded_phase4 = frame.loc[phase4_mask & ~frame["state_label"].astype(str).isin(qualified_states)].copy()
    return frame.loc[keep_mask].copy(), {
        "excluded_phase4_rows_without_evidence": int(len(excluded_phase4)),
        "excluded_phase4_states_without_evidence": int(excluded_phase4["state_label"].astype(str).nunique()) if not excluded_phase4.empty else 0,
    }


def _public_state_inputs_require_refresh(sequence_path: Path | None, state_path: Path | None) -> tuple[bool, str]:
    if sequence_path is None or state_path is None:
        return False, ""
    try:
        sequences = read_table(sequence_path)
        states = read_table(state_path)
    except Exception as exc:
        return True, f"unreadable:{exc!r}"
    if "source_dataset" not in sequences.columns:
        return False, ""
    public_rows = sequences["source_dataset"].astype(str).str.contains("HCA_SCREEN_public", na=False)
    if not bool(public_rows.any()):
        return False, ""
    required_sequence_columns = {
        "phase",
        "construction_mode",
        "phase4_label_version",
        "phase4_label_source",
        "phase4_evidence_score",
        "phase4_evidence_rank",
        "phase4_evidence_supported",
    }
    missing_columns = sorted(column for column in required_sequence_columns if column not in sequences.columns)
    if missing_columns:
        if _can_upgrade_public_state_sequences(sequences):
            return False, ""
        return True, f"missing_columns:{','.join(missing_columns)}"
    if "state_label" not in states.columns:
        return True, "missing_state_label"
    phase4 = sequences[sequences["phase"].astype(str) == "phase4"].copy()
    observed = phase4[phase4["construction_mode"].astype(str) == "observed_pair"].copy()
    if observed.empty:
        return True, "missing_phase4_observed_pairs"
    rows_per_state = observed["state_label"].astype(str).value_counts()
    if int((rows_per_state >= 2).sum()) == 0:
        return True, "no_state_pair_competition"
    if "label" not in observed.columns:
        return True, "missing_phase4_labels"
    if observed["phase4_label_version"].replace("", np.nan).isna().any():
        return True, "missing_phase4_label_version"
    if observed["phase4_label_source"].replace("", np.nan).isna().any():
        return True, "missing_phase4_label_source"
    competition_labels = observed.groupby(observed["state_label"].astype(str))["label"].apply(
        lambda values: bool((values.astype(float) > 0).any()) and bool((values.astype(float) <= 0).any())
    )
    if not bool(competition_labels.all()):
        return True, "phase4_state_without_competition_labels"
    evidence_supported = observed.groupby(observed["state_label"].astype(str))["phase4_evidence_supported"].apply(
        lambda values: values.astype(float).nunique(dropna=False) > 1
    )
    if not bool(evidence_supported.all()):
        return True, "phase4_state_without_evidence_support"
    return False, ""


def _stable_hex(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _ranked_head(frame: pd.DataFrame, group_cols: list[str], limit: int) -> pd.DataFrame:
    if frame.empty or limit <= 0:
        return frame.iloc[0:0].copy()
    ranked = frame.copy()
    ranked["_stable_rank"] = ranked["seq_id"].astype(str).map(_stable_hex)
    ranked = ranked.sort_values(group_cols + ["_stable_rank"])
    ranked = ranked.groupby(group_cols, group_keys=False).head(limit).copy()
    return ranked.drop(columns="_stable_rank")


def _formal_cap_selex_subset(
    project_root: Path,
    max_phase1_per_tf: int,
    max_phase2_per_pair_label: int,
) -> pd.DataFrame:
    cap_sequences_path = project_root / "data_intermediate" / "cap_selex_sequences.parquet"
    inventory_path = project_root / "data_intermediate" / "cap_selex_pair_inventory_usable.tsv"
    if not cap_sequences_path.exists() or not inventory_path.exists():
        return pd.DataFrame()
    sequences = read_table(cap_sequences_path)
    inventory = read_table(inventory_path)
    monomer_tfs = set(inventory.loc[inventory["monomer_usable"].fillna(False).astype(bool), "left_tf"].astype(str))
    pair_labels = set(
        inventory.loc[inventory["pair_usable"].fillna(False).astype(bool), ["left_tf", "right_tf"]]
        .astype(str)
        .agg("::".join, axis=1)
        .tolist()
    )
    phase1 = sequences[(sequences["phase"].astype(str) == "phase1") & (sequences["tf_label"].astype(str).isin(monomer_tfs))].copy()
    phase2 = sequences[(sequences["phase"].astype(str) == "phase2") & (sequences["tf_pair_label"].astype(str).isin(pair_labels))].copy()
    phase1 = _ranked_head(phase1, ["tf_label", "label"], max_phase1_per_tf)
    phase2 = _ranked_head(phase2, ["tf_pair_label", "label"], max_phase2_per_pair_label)
    cap = pd.concat([phase1, phase2], ignore_index=True, sort=False)
    if cap.empty:
        return cap
    cap["source_dataset"] = cap["source_dataset"].fillna("CAP_SELEX")
    cap["state_label"] = cap["state_label"].fillna("in_vitro")
    cap["chromosome"] = cap.get("chromosome", pd.Series(index=cap.index, dtype=object)).fillna("in_vitro")
    cap["split_group"] = cap.get("split_group", pd.Series(index=cap.index, dtype=object)).fillna(cap["tf_pair_label"].astype(str))
    cap["construction_mode"] = "cap_selex_formal_subset"
    cap["composite_label"] = cap.get("composite_label", pd.Series(index=cap.index, dtype=float)).fillna(0.0)
    return cap


def _build_formal_real_sequences(
    project_root: Path,
    state_sequences: pd.DataFrame,
    max_phase1_per_tf: int,
    max_phase2_per_pair_label: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    cap = _formal_cap_selex_subset(
        project_root,
        max_phase1_per_tf=max_phase1_per_tf,
        max_phase2_per_pair_label=max_phase2_per_pair_label,
    )
    combined = pd.concat([cap, state_sequences], ignore_index=True, sort=False)
    if "seq_id" not in combined.columns:
        combined["seq_id"] = [f"real::{index:08d}" for index in range(len(combined))]
    if "chromosome" not in combined.columns:
        combined["chromosome"] = "real"
    if "split_group" not in combined.columns:
        combined["split_group"] = combined["tf_pair_label"].fillna(combined["seq_id"]).astype(str)
    counts = {
        "phase1_rows": int((combined["phase"].astype(str) == "phase1").sum()),
        "phase2_rows": int((combined["phase"].astype(str) == "phase2").sum()),
        "phase3_rows": int((combined["phase"].astype(str) == "phase3").sum()),
        "phase4_rows": int((combined["phase"].astype(str) == "phase4").sum()),
    }
    return combined, counts


def prepare_real_state_data(
    project_root: str | Path,
    availability_dim: int = 16,
    state_dim: int = 32,
    max_examples: int | None = None,
    allow_public_portal_build: bool = True,
    allow_proxy_from_get: bool = False,
    proxy_examples: int = 512,
    max_fragment_files: int = 4,
    max_candidate_regions: int = 4096,
    max_states: int = 256,
    state_selection_multiplier: int = 4,
    max_pairs_per_state: int = 2,
    max_monomers_per_state: int = 1,
    max_fragment_lines: int | None = None,
    impute_missing_monomers: bool = True,
    impute_missing_pairs: bool = True,
    select_best_fragment_file_count: bool = True,
    exclude_imputed_examples: bool = True,
    max_phase1_per_tf: int = 1024,
    max_phase2_per_pair_label: int = 2048,
    formal_only: bool = False,
) -> pd.DataFrame:
    project_root = resolve_path(project_root)
    scenario_root = ensure_dir(project_root / "data_intermediate" / "real")
    sequence_path = _resolve_first(project_root, ["candidate_sequences.parquet", "candidate_sequences.tsv", "sequences.parquet", "sequences.tsv"])
    state_path = _resolve_first(project_root, ["state_features.parquet", "state_features.tsv"])
    inventory_payload = {
        "sequence_candidates": [str(path) for path in _candidate_paths(project_root, ["candidate_sequences.parquet", "candidate_sequences.tsv", "sequences.parquet", "sequences.tsv"])],
        "state_candidates": [str(path) for path in _candidate_paths(project_root, ["state_features.parquet", "state_features.tsv"])],
        "found_sequence_path": str(sequence_path) if sequence_path else "",
        "found_state_path": str(state_path) if state_path else "",
        "resource_inventory": _resource_inventory(project_root),
    }
    if allow_public_portal_build and sequence_path is not None and state_path is not None:
        needs_refresh, refresh_reason = _public_state_inputs_require_refresh(sequence_path, state_path)
        if needs_refresh:
            inventory_payload["stale_public_portal_build"] = {
                "sequence_path": str(sequence_path),
                "state_path": str(state_path),
                "reason": refresh_reason,
            }
            sequence_path = None
            state_path = None
            inventory_payload["found_sequence_path"] = ""
            inventory_payload["found_state_path"] = ""

    if formal_only and allow_proxy_from_get:
        allow_proxy_from_get = False

    if sequence_path is None or state_path is None:
        if allow_public_portal_build:
            try:
                from helixpair.public_state import build_public_state_layer_inputs

                sequence_path, state_path, public_manifest = build_public_state_layer_inputs(
                    project_root,
                    availability_dim=availability_dim,
                    state_dim=state_dim,
                    window_length=96,
                    max_fragment_files=max_fragment_files,
                    max_candidate_regions=max_candidate_regions,
                    max_states=max_states,
                    state_selection_multiplier=state_selection_multiplier,
                    max_examples=max_examples,
                    max_pairs_per_state=max_pairs_per_state,
                    max_monomers_per_state=max_monomers_per_state,
                    max_fragment_lines=max_fragment_lines,
                    impute_missing_monomers=impute_missing_monomers,
                    impute_missing_pairs=impute_missing_pairs,
                    select_best_fragment_file_count=select_best_fragment_file_count,
                )
                inventory_payload["found_sequence_path"] = str(sequence_path)
                inventory_payload["found_state_path"] = str(state_path)
                inventory_payload["public_portal_build"] = public_manifest
            except Exception as exc:
                inventory_payload["public_portal_build_error"] = repr(exc)
        if sequence_path is None or state_path is None:
            if allow_proxy_from_get:
                try:
                    sequence_path, state_path = _build_get_proxy_inputs(
                        project_root,
                        availability_dim=availability_dim,
                        state_dim=state_dim,
                        max_examples=max_examples,
                        proxy_examples=proxy_examples,
                    )
                    inventory_payload["found_sequence_path"] = str(sequence_path)
                    inventory_payload["found_state_path"] = str(state_path)
                    inventory_payload["proxy_mode"] = "GET_proxy"
                except Exception as exc:
                    inventory_payload["proxy_mode"] = "failed"
                    inventory_payload["proxy_error"] = repr(exc)
                    write_json(scenario_root / "state_inventory.json", {"status": "inputs_missing", **inventory_payload})
                    _write_acceptance(
                        scenario_root,
                        {
                            "dataset": "state_layer",
                            "status": "insufficient",
                            **inventory_payload,
                        },
                    )
                    raise
            else:
                write_json(scenario_root / "state_inventory.json", {"status": "inputs_missing", **inventory_payload})
                _write_acceptance(
                    scenario_root,
                    {
                        "dataset": "state_layer",
                        "status": "insufficient",
                        **inventory_payload,
                    },
                )
                raise FileNotFoundError(
                    "Real state preparation requires candidate/state tables under data_raw/state_layer, data_raw/encode/screen, or data_intermediate/real."
                )

    sequences = _upgrade_public_state_sequences(read_table(sequence_path))
    states = read_table(state_path).drop_duplicates("state_label")
    merged = sequences.merge(states, on="state_label", how="inner")
    if max_examples is not None and len(merged) > max_examples:
        merged = merged.head(max_examples).copy()

    merged = _derive_labels(merged)
    if formal_only and "construction_mode" in merged.columns:
        observed_mask = merged["construction_mode"].astype(str).str.startswith("observed")
        merged = merged.loc[observed_mask].copy()
        if merged.empty:
            raise RuntimeError("Formal real-state preparation retained no observed rows after strict filtering.")
    merged = _ensure_vectors(merged, "availability", availability_dim)
    merged = _ensure_vectors(merged, "state", state_dim)
    exclusion_summary = {"excluded_rows": 0, "excluded_phase3_rows": 0, "excluded_phase4_rows": 0}
    if exclude_imputed_examples:
        merged, exclusion_summary = _exclude_imputed_rows(merged)
        if merged.empty:
            raise RuntimeError("All real state-layer candidates were excluded after removing imputed constructions.")
    if str(inventory_payload.get("proxy_mode", "")) != "GET_proxy":
        merged, phase4_evidence_summary = _retain_evidence_qualified_phase4_states(merged)
        exclusion_summary.update(phase4_evidence_summary)

    if "seq_id" not in merged.columns:
        merged["seq_id"] = [f"real::{index:08d}" for index in range(len(merged))]
    if "chromosome" not in merged.columns:
        merged["chromosome"] = "real"

    formal_sequences, phase_counts = _build_formal_real_sequences(
        project_root,
        merged,
        max_phase1_per_tf=int(max_phase1_per_tf),
        max_phase2_per_pair_label=int(max_phase2_per_pair_label),
    )
    formal_sequences = _ensure_vectors(formal_sequences, "availability", availability_dim)
    formal_sequences = _ensure_vectors(formal_sequences, "state", state_dim)
    calibration = _build_monomer_calibration_set(formal_sequences)

    write_table(merged, scenario_root / "candidate_sequences.parquet")
    write_table(states, scenario_root / "state_features.parquet")
    write_table(formal_sequences, scenario_root / "sequences.parquet")
    write_table(calibration, scenario_root / "monomer_calibration_set.parquet")
    acceptance_payload = {
        "dataset": "state_layer",
        "status": "proxy_prepared" if str(inventory_payload.get("proxy_mode", "")) == "GET_proxy" else "prepared",
        "num_sequences": int(len(merged)),
        "num_formal_sequences": int(len(formal_sequences)),
        "num_states": int(merged["state_label"].nunique()),
        "num_calibration_sequences": int(len(calibration)),
        "sequence_source": str(sequence_path),
        "state_source": str(state_path),
        **phase_counts,
        **exclusion_summary,
    }
    for key in ["stale_public_portal_build", "public_portal_build"]:
        if key in inventory_payload:
            acceptance_payload[key] = inventory_payload[key]
    if inventory_payload.get("proxy_mode"):
        acceptance_payload["proxy_mode"] = inventory_payload["proxy_mode"]
    write_json(scenario_root / "state_inventory.json", acceptance_payload)
    _write_acceptance(scenario_root, acceptance_payload)
    return formal_sequences
