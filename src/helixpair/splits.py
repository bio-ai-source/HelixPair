from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import pandas as pd

from helixpair.io_utils import ensure_dir, read_table, resolve_path, split_token, write_json, write_text, write_table

PAIR_TRANSFER_SPLITS = {"unseen_pair", "unseen_family", "unseen_subfamily"}
STATE_TRANSFER_SPLITS = {"unseen_state"}


def stable_bucket(value: str, num_buckets: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % num_buckets


def assign_split(value: str, train_fraction: float = 0.7, valid_fraction: float = 0.15) -> str:
    bucket = stable_bucket(value, 1000) / 1000.0
    if bucket < train_fraction:
        return "train"
    if bucket < train_fraction + valid_fraction:
        return "valid"
    return "test"


def _ordered_pair(left: str, right: str) -> str:
    ordered = sorted([str(left), str(right)])
    return "::".join(ordered)


def phase_split_name(split_name: str, phase: str) -> str:
    requested = split_token(split_name)
    if requested in PAIR_TRANSFER_SPLITS:
        return requested if phase in {"phase2", "phase4", "phase5"} else "default"
    if requested in STATE_TRANSFER_SPLITS:
        return requested if phase in {"phase3", "phase4", "phase5"} else "default"
    return requested


def _sequence_group(frame: pd.DataFrame) -> pd.Series:
    if "chromosome" in frame.columns:
        return frame["chromosome"].astype(str)
    if "sequence" in frame.columns:
        return frame["sequence"].astype(str).map(lambda seq: f"kmer::{hashlib.sha256(seq[:32].encode('utf-8')).hexdigest()[:12]}")
    return frame["seq_id"].astype(str)


def _pair_family_blocks(pair_frame: pd.DataFrame, tf_master: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    family_lookup = tf_master.set_index("gene_symbol")["family"].astype(str).to_dict()
    subfamily_lookup = tf_master.set_index("gene_symbol")["subfamily"].astype(str).to_dict()
    paralog_lookup = tf_master.set_index("gene_symbol")["paralog_group"].astype(str).to_dict()

    def _block(row, lookup):
        return _ordered_pair(lookup.get(str(row.left_tf), ""), lookup.get(str(row.right_tf), ""))

    family_block = pair_frame.apply(lambda row: _block(row, family_lookup), axis=1)
    subfamily_block = pair_frame.apply(lambda row: _block(row, subfamily_lookup), axis=1)
    paralog_block = pair_frame.apply(lambda row: _block(row, paralog_lookup), axis=1)
    combined = family_block.where(paralog_block == "::", family_block + "::paralog::" + paralog_block)
    return combined, subfamily_block


def _pair_target_examples(window_frame: pd.DataFrame) -> pd.DataFrame:
    if "phase" not in window_frame.columns or "tf_pair_label" not in window_frame.columns:
        return pd.DataFrame(columns=list(window_frame.columns) + ["left_tf", "right_tf"])
    phases = {"phase2", "phase4", "phase5"}
    examples = window_frame[window_frame["phase"].astype(str).isin(phases)].copy()
    if examples.empty:
        return examples
    left_right = examples["tf_pair_label"].astype(str).str.split("::", expand=True)
    examples["left_tf"] = left_right[0].astype(str)
    examples["right_tf"] = left_right[left_right.columns[-1]].astype(str)
    return examples


def _state_target_examples(window_frame: pd.DataFrame) -> pd.DataFrame:
    if "phase" not in window_frame.columns:
        return window_frame.copy()
    phases = {"phase3", "phase4", "phase5"}
    examples = window_frame[window_frame["phase"].astype(str).isin(phases)].copy()
    return examples if not examples.empty else window_frame.copy()


def make_group_splits(frame: pd.DataFrame, group_series: pd.Series, split_name: str, output_dir: str | Path) -> pd.DataFrame:
    records = []
    local = frame.copy()
    local["_group_value"] = group_series.astype(str).values
    for group_value, group_frame in local.groupby("_group_value"):
        split = assign_split(str(group_value))
        for row in group_frame.itertuples(index=False):
            payload = row._asdict()
            payload.pop("_group_value", None)
            payload["split_name"] = split_name
            payload["split"] = split
            payload["group_value"] = group_value
            records.append(payload)
    split_frame = pd.DataFrame.from_records(records)
    output_path = Path(output_dir) / f"{split_name}.parquet"
    write_table(split_frame, output_path)
    return split_frame


def create_all_splits(window_path: str | Path, pair_path: str | Path, tf_master_path: str | Path, output_dir: str | Path) -> dict[str, str]:
    window_frame = read_table(window_path)
    pair_frame = read_table(pair_path)
    tf_master = read_table(tf_master_path)
    output_dir = ensure_dir(output_dir)

    pair_examples = _pair_target_examples(window_frame)
    state_examples = _state_target_examples(window_frame)
    seen_source = pair_examples if not pair_examples.empty else pair_frame if not pair_frame.empty else window_frame
    seen = make_group_splits(seen_source, seen_source["seq_id"].astype(str), "seen", output_dir)
    pair_group = (
        pair_examples.apply(lambda row: _ordered_pair(row.left_tf, row.right_tf), axis=1)
        if not pair_examples.empty
        else pair_frame.apply(lambda row: _ordered_pair(row.left_tf, row.right_tf), axis=1)
    )
    unseen_pair = make_group_splits(pair_examples if not pair_examples.empty else pair_frame, pair_group, "unseen_pair", output_dir)
    family_block, subfamily_block = _pair_family_blocks(pair_examples if not pair_examples.empty else pair_frame, tf_master)
    unseen_family = make_group_splits(pair_examples if not pair_examples.empty else pair_frame, family_block, "unseen_family", output_dir)
    unseen_subfamily = make_group_splits(
        pair_examples if not pair_examples.empty else pair_frame,
        subfamily_block,
        "unseen_subfamily",
        output_dir,
    )
    state_group = (
        state_examples["state_label"].astype(str)
        if "state_label" in state_examples.columns
        else pd.Series(["unknown"] * len(state_examples))
    )
    unseen_state = make_group_splits(state_examples, state_group, "unseen_state", output_dir)
    locus = make_group_splits(window_frame, _sequence_group(window_frame), "sequence_locus", output_dir)

    manifest = {
        "seen": str(output_dir / "seen.parquet"),
        "unseen_pair": str(output_dir / "unseen_pair.parquet"),
        "unseen_family": str(output_dir / "unseen_family.parquet"),
        "unseen_subfamily": str(output_dir / "unseen_subfamily.parquet"),
        "unseen_state": str(output_dir / "unseen_state.parquet"),
        "sequence_locus": str(output_dir / "sequence_locus.parquet"),
        "row_counts": {
            "seen": int(len(seen)),
            "unseen_pair": int(len(unseen_pair)),
            "unseen_family": int(len(unseen_family)),
            "unseen_subfamily": int(len(unseen_subfamily)),
            "unseen_state": int(len(unseen_state)),
            "sequence_locus": int(len(locus)),
        },
        "unique_groups": {
            "pair": int(pair_group.nunique()),
            "family_block": int(family_block.nunique()),
            "subfamily_block": int(subfamily_block.nunique()),
            "state": int(state_group.nunique()),
            "locus": int(_sequence_group(window_frame).nunique()),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    write_text(
        output_dir / "split_change_log.md",
        "\n".join(
            [
                "# Split Change Log",
                "",
                "- Formal split manifests generated with pair-, family-, subfamily-, state-, and sequence/locus-level grouping.",
                "- Family blocks incorporate paralog-group information when available.",
            ]
        ),
    )
    return manifest


def assert_split_disjoint(split_frame: pd.DataFrame, key_col: str = "seq_id") -> None:
    observed = defaultdict(set)
    for row in split_frame.itertuples(index=False):
        observed[getattr(row, key_col)].add(row.split)
    collisions = {key: values for key, values in observed.items() if len(values) > 1}
    if collisions:
        raise ValueError(f"Detected split leakage for {len(collisions)} groups in {key_col}")
