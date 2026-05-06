from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str((ROOT / "src").resolve()))

from helixpair.bundles import build_tensor_bundles, build_windows_and_candidates  # noqa: E402
from helixpair.io_utils import ensure_dir, write_json, write_table  # noqa: E402


SUPPLEMENT_PATH = ROOT / "data_raw" / "cap_selex" / "repository" / "unzipped" / "41586_2025_8844_MOESM10_ESM.xlsx"
STATE_FEATURE_PATH = ROOT / "data_raw" / "state_layer" / "state_features.parquet"
SCENARIO = "phase5_reporter"
SPLIT_NAME = "reporter_default"


CONSTRUCTS = {
    "Sequence # 4": {
        "pair": "PROX1::HOXA2",
        "positive_in_high_state": 1.0,
        "composite_label": 1.0,
        "evidence": "PROX-HOX2 composite motif drove reproducible expression in enSERT embryos.",
    },
    "Sequence # 5": {
        "pair": "GLI3::RFX3",
        "positive_in_high_state": 1.0,
        "composite_label": 1.0,
        "evidence": "GLI3-RFX3 composite motif drove reproducible expression in enSERT embryos.",
    },
    "Sequence # 6": {
        "pair": "GLI3::FLI1",
        "positive_in_high_state": 0.0,
        "composite_label": 0.0,
        "evidence": "FLI1-GLI3 reporter was described as silenced and used as a negative control.",
    },
    "Sequence # 7": {
        "pair": "GLI3::RFX3",
        "positive_in_high_state": 1.0,
        "composite_label": 1.0,
        "evidence": "GLI3-RFX3 composite motif variant used in enSERT assay; treated as active composite motif.",
    },
    "Sequence # 8": {
        "pair": "GLI3::RFX3",
        "positive_in_high_state": 1.0,
        "composite_label": 1.0,
        "evidence": "GLI3-RFX3 composite motif variant used in enSERT assay; treated as active composite motif.",
    },
}


VALID_SPLIT = {
    "Sequence # 4::high": "valid",
    "Sequence # 8::high": "valid",
    "Sequence # 6::high": "valid",
    "Sequence # 7::low": "valid",
}


def _load_construct_table() -> pd.DataFrame:
    table = pd.read_excel(SUPPLEMENT_PATH, sheet_name="Supplementary Table S8", header=7)
    table = table.rename(
        columns={
            "Sequence name": "sequence_name",
            "TF pair": "tf_pair_raw",
            "Core sequence": "core_sequence",
            "5' Flanking sequence": "flank_5p",
            "3' Flanking sequence": "flank_3p",
            "Interval sequence": "interval_sequence",
            "Final sequences": "final_sequence",
        }
    )
    table = table[table["sequence_name"].isin(CONSTRUCTS)].copy()
    if len(table) != len(CONSTRUCTS):
        missing = sorted(set(CONSTRUCTS) - set(table["sequence_name"].astype(str)))
        raise RuntimeError(f"Missing reporter constructs in Supplementary Table S8: {missing}")
    table["final_sequence"] = table["final_sequence"].astype(str).str.upper()
    return table


def _select_real_states() -> tuple[str, str, pd.DataFrame]:
    state_frame = pd.read_parquet(STATE_FEATURE_PATH).copy()
    availability_cols = [col for col in state_frame.columns if col.startswith("availability_")]
    state_cols = [col for col in state_frame.columns if col.startswith("state_") and col != "state_label"]
    for column in availability_cols + state_cols:
        state_frame[column] = pd.to_numeric(state_frame[column], errors="coerce").fillna(0.0)
    state_frame["availability_sum"] = state_frame[availability_cols].sum(axis=1)
    state_frame["state_sum"] = state_frame[state_cols].sum(axis=1)
    high_row = state_frame.sort_values(["availability_sum", "state_sum"], ascending=[False, False]).iloc[0]
    low_row = state_frame.sort_values(["availability_sum", "state_sum"], ascending=[True, True]).iloc[0]
    selected = state_frame[state_frame["state_label"].isin([high_row.state_label, low_row.state_label])].copy()
    return str(high_row.state_label), str(low_row.state_label), selected


def _build_phase5_sequences(high_state: str, low_state: str, selected_states: pd.DataFrame) -> pd.DataFrame:
    construct_table = _load_construct_table()
    records: list[dict[str, object]] = []
    for row in construct_table.itertuples(index=False):
        metadata = CONSTRUCTS[str(row.sequence_name)]
        for state_mode, state_label in [("high", high_state), ("low", low_state)]:
            functional_label = float(metadata["positive_in_high_state"]) if state_mode == "high" else 0.0
            seq_id = f"reporter::{str(row.sequence_name).replace(' ', '_').replace('#', '').lower()}::{state_mode}"
            records.append(
                {
                    "seq_id": seq_id,
                    "sequence_name": str(row.sequence_name),
                    "sequence": str(row.final_sequence),
                    "state_label": state_label,
                    "state_mode": state_mode,
                    "tf_pair_label": str(metadata["pair"]),
                    "functional_label": functional_label,
                    "usage_label": functional_label,
                    "label": functional_label,
                    "phase": "phase5",
                    "composite_label": float(metadata["composite_label"]),
                    "element_type": "reporter_construct",
                    "source_dataset": "CAP_SELEX_enSERT_reporter",
                    "construction_mode": "observed_reporter",
                    "split_group": seq_id,
                    "chromosome": "reporter",
                    "evidence_note": str(metadata["evidence"]),
                    "core_sequence": str(row.core_sequence),
                    "flank_5p": str(row.flank_5p),
                    "flank_3p": str(row.flank_3p),
                    "interval_sequence": str(row.interval_sequence),
                }
            )
    frame = pd.DataFrame.from_records(records)
    merged = frame.merge(selected_states, on="state_label", how="left", validate="many_to_one")
    if merged.isna().any().any():
        missing_state_rows = merged[merged["availability_0"].isna()] if "availability_0" in merged.columns else merged[merged.isna().any(axis=1)]
        if not missing_state_rows.empty:
            raise RuntimeError(f"Failed to attach state vectors to reporter rows: {missing_state_rows['seq_id'].tolist()}")
    return merged


def _build_split_manifest(frame: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    split_records: list[dict[str, str]] = []
    for row in frame.itertuples(index=False):
        key = f"{row.sequence_name}::{row.state_mode}"
        split = VALID_SPLIT.get(key, "train")
        split_records.append(
            {
                "seq_id": str(row.seq_id),
                "split_name": SPLIT_NAME,
                "split": split,
                "group_value": str(row.seq_id),
            }
        )
    split_frame = pd.DataFrame.from_records(split_records)
    write_table(split_frame, output_path)
    return split_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sidecar Phase V reporter bundles from Supplementary Table S8.")
    parser.add_argument("--window-length", type=int, default=96)
    parser.add_argument("--top-k-anchors", type=int, default=24)
    parser.add_argument("--top-k-pairs", type=int, default=64)
    args = parser.parse_args()

    high_state, low_state, selected_states = _select_real_states()

    scenario_root = ensure_dir(ROOT / "data_intermediate" / SCENARIO)
    split_root = ensure_dir(ROOT / "splits" / SCENARIO)
    report_root = ensure_dir(ROOT / "reports" / "phase5_reporter")

    sequence_frame = _build_phase5_sequences(high_state, low_state, selected_states)
    write_table(sequence_frame, scenario_root / "sequences.parquet")
    write_table(sequence_frame, report_root / "phase5_reporter_sequences.csv")

    split_frame = _build_split_manifest(sequence_frame, split_root / f"{SPLIT_NAME}.parquet")
    write_table(split_frame, report_root / "phase5_reporter_split_manifest.csv")

    build_windows_and_candidates(
        ROOT,
        scenario=SCENARIO,
        window_length=int(args.window_length),
        top_k_anchors=int(args.top_k_anchors),
        top_k_pairs=int(args.top_k_pairs),
    )
    outputs = build_tensor_bundles(
        ROOT,
        scenario=SCENARIO,
        window_length=int(args.window_length),
        split_name=SPLIT_NAME,
        split_manifest=split_root / f"{SPLIT_NAME}.parquet",
        phases=["phase5"],
    )

    pairs = pd.read_parquet(scenario_root / "pairs.parquet")
    observed_pair_counts = (
        pairs.groupby("seq_id", as_index=False).size().rename(columns={"size": "candidate_pairs"}) if not pairs.empty else pd.DataFrame(columns=["seq_id", "candidate_pairs"])
    )
    pair_audit = sequence_frame[["seq_id", "sequence_name", "tf_pair_label", "state_mode", "label"]].merge(
        observed_pair_counts, on="seq_id", how="left"
    )
    pair_audit["candidate_pairs"] = pair_audit["candidate_pairs"].fillna(0).astype(int)
    write_table(pair_audit, report_root / "phase5_reporter_pair_audit.csv")
    if (pair_audit["candidate_pairs"] <= 0).any():
        failing = pair_audit.loc[pair_audit["candidate_pairs"] <= 0, "seq_id"].tolist()
        raise RuntimeError(f"No candidate pairs were recovered for reporter rows: {failing}")

    manifest = {
        "scenario": SCENARIO,
        "split_name": SPLIT_NAME,
        "source": str(SUPPLEMENT_PATH),
        "selected_states": {
            "high": high_state,
            "low": low_state,
        },
        "num_sequences": int(len(sequence_frame)),
        "label_counts": {str(key): int(value) for key, value in sequence_frame["label"].value_counts().sort_index().items()},
        "split_counts": {str(key): int(value) for key, value in split_frame["split"].value_counts().sort_index().items()},
        "bundle_outputs": outputs,
    }
    write_json(report_root / "phase5_reporter_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
