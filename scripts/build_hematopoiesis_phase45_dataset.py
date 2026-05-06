from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str((ROOT / "src").resolve()))

from helixpair.bundles import build_tensor_bundles, build_windows_and_candidates  # noqa: E402
from helixpair.io_utils import ensure_dir, write_json, write_table  # noqa: E402
from helixpair.splits import assign_split  # noqa: E402


RAW_RDA = ROOT / "data_raw" / "external_phase45" / "hematopoiesis_mpra_data_updated2.rda"
CODA_RAW = ROOT / "data_raw" / "external_phase45" / "coda_mpra_table_s2.txt"
R_EXPORT_SCRIPT = ROOT / "scripts" / "data" / "export_hematopoiesis_pair_tables.R"
EXPORT_ROOT = ROOT / "data_intermediate" / "external_phase45"
EXPORTED_COMBINED = EXPORT_ROOT / "hematopoiesis_pair_libs_combined.csv"
SCENARIO = "phase45_hematopoiesis"
WINDOW_LENGTH = 96
AVAILABILITY_DIM = 16
STATE_DIM = 32
PHASE4_LABEL_VERSION = "hematopoiesis_pair_activity_v1"
PHASE4_LABEL_SOURCE = "hematopoiesis_synthetic_mpra"


def _run_r_export() -> None:
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    if EXPORTED_COMBINED.exists() and EXPORTED_COMBINED.stat().st_mtime >= RAW_RDA.stat().st_mtime:
        return
    subprocess.run(
        ["Rscript", str(R_EXPORT_SCRIPT), str(RAW_RDA), str(EXPORT_ROOT)],
        cwd=ROOT,
        check=True,
    )


def _load_tf_lookup() -> dict[str, str]:
    tf_master = pd.read_csv(ROOT / "data_intermediate" / "tf_master_table.tsv", sep="\t")
    lookup = {
        str(gene).upper(): str(gene)
        for gene in tf_master["gene_symbol"].dropna().astype(str)
    }
    if "TP53" in lookup:
        lookup["TRP53"] = "TP53"
    return lookup


def _clean_sequence(raw_sequence: str, window_length: int = WINDOW_LENGTH) -> str:
    raw_sequence = str(raw_sequence or "").strip()
    if not raw_sequence:
        return "N" * window_length
    sequence = "".join(character if character.upper() in {"A", "C", "G", "T", "N"} else "N" for character in raw_sequence)
    uppercase_positions = [
        index
        for index, character in enumerate(sequence)
        if character.upper() in {"A", "C", "G", "T", "N"} and character == character.upper()
    ]
    if uppercase_positions:
        span_start = min(uppercase_positions)
        span_end = max(uppercase_positions) + 1
        center = (span_start + span_end) // 2
        start = max(0, center - (window_length // 2))
    else:
        start = max(0, (len(sequence) - window_length) // 2)
    end = start + window_length
    if end > len(sequence):
        end = len(sequence)
        start = max(0, end - window_length)
    window = sequence[start:end].upper()
    if len(window) < window_length:
        pad_total = window_length - len(window)
        left_pad = pad_total // 2
        right_pad = pad_total - left_pad
        window = ("N" * left_pad) + window + ("N" * right_pad)
    return window


def _ordered_pair_label(label: str) -> str:
    tokens = [token for token in str(label).split("::") if token]
    if len(tokens) < 2:
        return str(label)
    ordered = sorted([tokens[0], tokens[-1]])
    return "::".join(ordered)


def _actual_state_vectors(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    states = sorted(frame["actual_state"].astype(str).unique().tolist())
    state_to_index = {state: index for index, state in enumerate(states)}
    for index in range(AVAILABILITY_DIM):
        frame[f"availability_{index}"] = 0.0
    for index in range(STATE_DIM):
        frame[f"state_{index}"] = 0.0
    for state, index in state_to_index.items():
        if index < AVAILABILITY_DIM:
            frame.loc[frame["actual_state"].astype(str) == state, f"availability_{index}"] = 1.0
        if index < STATE_DIM:
            frame.loc[frame["actual_state"].astype(str) == state, f"state_{index}"] = 1.0
    return frame, {state: state_to_index[state] for state in states}


def _prepare_pair_frame() -> tuple[pd.DataFrame, dict[str, int]]:
    _run_r_export()
    frame = pd.read_csv(EXPORTED_COMBINED).copy()
    tf_lookup = _load_tf_lookup()

    frame["left_tf"] = frame["TF1.name"].astype(str).str.upper().map(tf_lookup)
    frame["right_tf"] = frame["TF2.name"].astype(str).str.upper().map(tf_lookup)
    frame["activity_score"] = pd.to_numeric(frame["mean.norm.adj"], errors="coerce")
    frame["spacer"] = pd.to_numeric(frame.get("spacer"), errors="coerce")
    frame["sequence"] = frame["Seq"].astype(str).map(_clean_sequence)
    frame["actual_state"] = frame["actual_state"].astype(str)
    frame = frame[
        frame["left_tf"].notna()
        & frame["right_tf"].notna()
        & frame["activity_score"].notna()
        & frame["sequence"].notna()
    ].copy()
    frame["tf_pair_label"] = frame["left_tf"].astype(str) + "::" + frame["right_tf"].astype(str)
    frame["pair_transfer_group"] = frame["tf_pair_label"].astype(str).map(_ordered_pair_label)
    frame["competition_group"] = (
        frame["library_name"].astype(str)
        + "::"
        + frame["actual_state"].astype(str)
        + "::"
        + frame["pair_transfer_group"].astype(str)
    )
    frame["raw_seq_id"] = (
        frame["library_name"].astype(str)
        + "::"
        + frame["actual_state"].astype(str)
        + "::"
        + frame["CRS"].astype(str)
    )
    frame = frame.sort_values(
        ["competition_group", "activity_score", "CRS"],
        ascending=[True, False, True],
    ).copy()
    frame["group_size"] = frame.groupby("competition_group")["CRS"].transform("size").astype(int)
    frame["activity_rank_desc"] = frame.groupby("competition_group").cumcount().add(1).astype(int)
    frame["activity_rank_asc"] = frame.groupby("competition_group")["activity_score"].rank(method="first", ascending=True).astype(int)
    frame, state_index = _actual_state_vectors(frame)
    return frame, state_index


def _select_extrema_rows(group: pd.DataFrame, positives: int, negatives: int) -> pd.DataFrame:
    ordered = group.sort_values(["activity_score", "CRS"], ascending=[False, True]).copy()
    if ordered.empty:
        return ordered
    if float(ordered["activity_score"].iloc[0]) <= float(ordered["activity_score"].iloc[-1]):
        return ordered.iloc[0:0].copy()
    positive_rows = ordered.head(min(positives, len(ordered))).copy()
    negative_rows = ordered.tail(min(negatives, max(len(ordered) - len(positive_rows), 1))).copy()
    negative_rows = negative_rows.loc[~negative_rows["CRS"].astype(str).isin(positive_rows["CRS"].astype(str))].copy()
    selected = pd.concat([positive_rows, negative_rows], ignore_index=True, sort=False)
    selected = selected.drop_duplicates("CRS", keep="first").copy()
    if selected["CRS"].nunique() < 2:
        return ordered.iloc[0:0].copy()
    selected["derived_label"] = 0.0
    selected.loc[selected["CRS"].astype(str).isin(positive_rows["CRS"].astype(str)), "derived_label"] = 1.0
    return selected


def _build_phase4_rows(frame: pd.DataFrame) -> pd.DataFrame:
    selected_groups = []
    for _, group in frame.groupby("competition_group", sort=False):
        if len(group) < 2:
            continue
        selected = _select_extrema_rows(group, positives=1, negatives=1)
        if selected.empty:
            continue
        selected_groups.append(selected)
    phase4 = pd.concat(selected_groups, ignore_index=True, sort=False) if selected_groups else pd.DataFrame(columns=frame.columns.tolist() + ["derived_label"])
    if phase4.empty:
        return phase4
    phase4 = phase4.sort_values(["competition_group", "activity_score", "CRS"], ascending=[True, False, True]).copy()
    phase4["state_label"] = phase4["competition_group"].astype(str)
    phase4["seq_id"] = (
        "hematopoiesis::phase4::"
        + phase4["library_name"].astype(str).str.replace(".", "_", regex=False)
        + "::"
        + phase4["actual_state"].astype(str)
        + "::"
        + phase4["CRS"].astype(str)
    )
    phase4["label"] = phase4["derived_label"].astype(float)
    phase4["usage_label"] = phase4["label"]
    phase4["functional_label"] = np.nan
    phase4["phase"] = "phase4"
    phase4["split_group"] = phase4["competition_group"].astype(str)
    phase4["composite_label"] = 1.0
    phase4["element_type"] = "synthetic_pair_extrema"
    phase4["source_dataset"] = "hematopoiesis_synthetic_mpra"
    phase4["construction_mode"] = "observed_pair"
    phase4["chromosome"] = phase4["actual_state"].astype(str)
    phase4["phase4_label_version"] = PHASE4_LABEL_VERSION
    phase4["phase4_label_source"] = PHASE4_LABEL_SOURCE
    phase4["phase4_evidence_score"] = phase4["activity_score"].astype(float)
    phase4["phase4_evidence_rank"] = phase4.groupby("competition_group").cumcount().add(1).astype(float)
    phase4["phase4_evidence_supported"] = phase4["label"].astype(float)
    return phase4


def _build_phase5_rows(frame: pd.DataFrame) -> pd.DataFrame:
    selected_groups = []
    for _, group in frame.groupby("competition_group", sort=False):
        if len(group) < 2:
            continue
        positives = 2 if len(group) >= 4 else 1
        negatives = 2 if len(group) >= 4 else 1
        selected = _select_extrema_rows(group, positives=positives, negatives=negatives)
        if selected.empty:
            continue
        selected_groups.append(selected)
    phase5 = pd.concat(selected_groups, ignore_index=True, sort=False) if selected_groups else pd.DataFrame(columns=frame.columns.tolist() + ["derived_label"])
    if phase5.empty:
        return phase5
    phase5["state_label"] = "hematopoiesis::" + phase5["actual_state"].astype(str)
    phase5["seq_id"] = (
        "hematopoiesis::phase5::"
        + phase5["library_name"].astype(str).str.replace(".", "_", regex=False)
        + "::"
        + phase5["actual_state"].astype(str)
        + "::"
        + phase5["CRS"].astype(str)
    )
    phase5["label"] = phase5["derived_label"].astype(float)
    phase5["functional_label"] = phase5["label"]
    phase5["usage_label"] = phase5["label"]
    phase5["phase"] = "phase5"
    phase5["split_group"] = phase5["competition_group"].astype(str)
    phase5["composite_label"] = 1.0
    phase5["element_type"] = "synthetic_pair_functional_extrema"
    phase5["source_dataset"] = "hematopoiesis_synthetic_mpra"
    phase5["construction_mode"] = "observed_pair_functional"
    phase5["chromosome"] = phase5["actual_state"].astype(str)
    phase5["phase4_label_version"] = ""
    phase5["phase4_label_source"] = ""
    phase5["phase4_evidence_score"] = np.nan
    phase5["phase4_evidence_rank"] = np.nan
    phase5["phase4_evidence_supported"] = np.nan
    return phase5


def _build_split_manifest(frame: pd.DataFrame, split_name: str, key_column: str, output_path: Path) -> pd.DataFrame:
    local = frame[["seq_id", key_column]].copy()
    local["group_value"] = local[key_column].astype(str)
    local["split_name"] = split_name
    local["split"] = local["group_value"].map(assign_split)
    manifest = local[["seq_id", "split_name", "split", "group_value"]].copy()
    write_table(manifest, output_path)
    return manifest


def _phase_summary(frame: pd.DataFrame, phase: str) -> dict[str, object]:
    phase_frame = frame[frame["phase"].astype(str) == phase].copy()
    return {
        "rows": int(len(phase_frame)),
        "label_counts": {str(key): int(value) for key, value in phase_frame["label"].value_counts().sort_index().items()},
        "actual_states": sorted(phase_frame["actual_state"].astype(str).unique().tolist()),
        "pairs": int(phase_frame["pair_transfer_group"].astype(str).nunique()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build external hematopoiesis sidecar datasets for phase4 and phase5.")
    parser.add_argument("--window-length", type=int, default=WINDOW_LENGTH)
    parser.add_argument("--top-k-anchors", type=int, default=8)
    parser.add_argument("--top-k-pairs", type=int, default=8)
    args = parser.parse_args()

    if not RAW_RDA.exists():
        raise FileNotFoundError(f"Missing raw hematopoiesis archive: {RAW_RDA}")

    pair_frame, state_index = _prepare_pair_frame()
    phase4 = _build_phase4_rows(pair_frame)
    phase5 = _build_phase5_rows(pair_frame)
    if phase4.empty or phase5.empty:
        raise RuntimeError("Failed to derive non-empty phase4/phase5 external datasets from hematopoiesis MPRA.")

    scenario_root = ensure_dir(ROOT / "data_intermediate" / SCENARIO)
    split_root = ensure_dir(ROOT / "splits" / SCENARIO)
    report_root = ensure_dir(ROOT / "reports" / "external_phase45")

    combined = pd.concat([phase4, phase5], ignore_index=True, sort=False).copy()
    combined["sequence"] = combined["sequence"].astype(str)

    write_table(combined, scenario_root / "sequences.parquet")
    write_table(phase4, report_root / "phase4_hematopoiesis_sequences.csv")
    write_table(phase5, report_root / "phase5_hematopoiesis_sequences.csv")

    pair_group_summary = (
        pair_frame.groupby(["library_name", "actual_state", "pair_transfer_group"], as_index=False)
        .agg(
            group_size=("CRS", "size"),
            top_activity=("activity_score", "max"),
            bottom_activity=("activity_score", "min"),
        )
    )
    pair_group_summary["activity_margin"] = pair_group_summary["top_activity"] - pair_group_summary["bottom_activity"]
    write_table(pair_group_summary, report_root / "hematopoiesis_pair_group_summary.csv")

    unseen_pair_manifest = _build_split_manifest(combined, "unseen_pair", "pair_transfer_group", split_root / "unseen_pair.parquet")
    unseen_state_manifest = _build_split_manifest(combined, "unseen_state", "actual_state", split_root / "unseen_state.parquet")
    write_table(unseen_pair_manifest, report_root / "phase45_hematopoiesis_unseen_pair_manifest.csv")
    write_table(unseen_state_manifest, report_root / "phase45_hematopoiesis_unseen_state_manifest.csv")

    build_windows_and_candidates(
        ROOT,
        scenario=SCENARIO,
        window_length=int(args.window_length),
        top_k_anchors=int(args.top_k_anchors),
        top_k_pairs=int(args.top_k_pairs),
    )

    output_default = build_tensor_bundles(
        ROOT,
        scenario=SCENARIO,
        window_length=int(args.window_length),
        split_name="default",
        phases=["phase4", "phase5"],
    )
    output_unseen_pair = build_tensor_bundles(
        ROOT,
        scenario=SCENARIO,
        window_length=int(args.window_length),
        split_name="unseen_pair",
        split_manifest=split_root / "unseen_pair.parquet",
        phases=["phase4", "phase5"],
    )
    output_unseen_state = build_tensor_bundles(
        ROOT,
        scenario=SCENARIO,
        window_length=int(args.window_length),
        split_name="unseen_state",
        split_manifest=split_root / "unseen_state.parquet",
        phases=["phase4", "phase5"],
    )

    coda_rows = 0
    if CODA_RAW.exists():
        with CODA_RAW.open("r", encoding="utf-8") as handle:
            coda_rows = max(sum(1 for _ in handle) - 1, 0)

    manifest = {
        "scenario": SCENARIO,
        "raw_source": str(RAW_RDA),
        "raw_coda_source": str(CODA_RAW) if CODA_RAW.exists() else "",
        "window_length": int(args.window_length),
        "top_k_anchors": int(args.top_k_anchors),
        "top_k_pairs": int(args.top_k_pairs),
        "state_index": state_index,
        "raw_rows_after_tf_filter": int(len(pair_frame)),
        "phase4": _phase_summary(combined, "phase4"),
        "phase5": _phase_summary(combined, "phase5"),
        "split_counts": {
            "unseen_pair": unseen_pair_manifest["split"].value_counts().sort_index().to_dict(),
            "unseen_state": unseen_state_manifest["split"].value_counts().sort_index().to_dict(),
        },
        "outputs": {
            "default": output_default,
            "unseen_pair": output_unseen_pair,
            "unseen_state": output_unseen_state,
        },
        "coda_flat_table_rows": coda_rows,
    }
    write_json(report_root / "phase45_hematopoiesis_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
