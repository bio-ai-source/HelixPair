from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from helixpair.constants import DEFAULT_GAP_BINS, DEFAULT_HELICAL_PERIOD, ORIENTATION_TO_INDEX
from helixpair.io_utils import ensure_dir, read_table, resolve_path, write_json, write_table
from helixpair.sequence import embed_sequence, random_background, reverse_complement


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


def _mutate_motif(motif: str, rng: np.random.Generator, num_changes: int | None = None) -> str:
    bases = np.asarray(list("ACGT"))
    motif_chars = list(motif)
    if num_changes is None:
        num_changes = max(3, int(np.ceil(len(motif_chars) * 0.4)))
    change_count = min(max(num_changes, 1), len(motif_chars))
    for position in rng.choice(np.arange(len(motif_chars)), size=change_count, replace=False):
        current = motif_chars[int(position)]
        choices = bases[bases != current]
        motif_chars[int(position)] = str(rng.choice(choices))
    return "".join(motif_chars)


def _one_hot_state(index: int, width: int) -> list[float]:
    values = [0.0] * width
    values[index % width] = 1.0
    return values


def _state_vector(lineage_index: int, state_level: float, width: int, rng: np.random.Generator) -> list[float]:
    vec = rng.normal(0.0, 0.08, size=width).astype(float)
    vec[:4] = 0.0
    vec[0] = state_level
    vec[1] = float(lineage_index) / max(width, 1)
    if width > 2:
        vec[2] = float(lineage_index % 2)
    if width > 3:
        vec[3] = float(lineage_index % 3) / 2.0
    return vec.tolist()


def _availability_vector(level: float, lineage_index: int, width: int, rng: np.random.Generator) -> list[float]:
    vec = rng.normal(0.0, 0.05, size=width).astype(float)
    vec[:] = np.clip(vec, -0.1, 0.1)
    vec[0] = np.clip(level, 0.0, 1.2)
    if width > 1:
        vec[1] = np.clip(0.5 * level + 0.1 * (lineage_index % 2), 0.0, 1.0)
    if width > 2:
        vec[2] = np.clip(0.35 + 0.15 * (lineage_index % 3), 0.0, 1.0)
    return vec.tolist()


def _helical_alignment(gap: int, preferred_gap: int) -> float:
    phase = 2.0 * np.pi * (gap - preferred_gap) / DEFAULT_HELICAL_PERIOD
    return float(np.cos(phase))


def _select_diverse_tfs(project_root: Path, max_families: int = 6, max_per_family: int = 2) -> pd.DataFrame:
    tf_master = read_table(project_root / "data_intermediate" / "tf_master_table.tsv")
    consensus_by_gene = _load_consensus_map(project_root)
    tf_frame = tf_master[tf_master["gene_symbol"].isin(consensus_by_gene.keys())].copy()
    tf_frame = tf_frame.drop_duplicates("gene_symbol")
    if tf_frame.empty:
        raise ValueError("Synthetic scenario requires TFs with motif consensuses.")
    selected_rows = []
    for family, group in tf_frame.sort_values(["family", "gene_symbol"]).groupby("family", sort=True):
        if not str(family):
            continue
        for row in group.head(max_per_family).itertuples(index=False):
            selected_rows.append(row._asdict())
        if len({item["family"] for item in selected_rows}) >= max_families:
            break
    selected = pd.DataFrame.from_records(selected_rows)
    if len(selected) < 6:
        selected = tf_frame.head(8).copy()
    return selected.reset_index(drop=True)


def _pair_specs(selected: pd.DataFrame, rng: np.random.Generator, scenario: str) -> list[dict[str, object]]:
    family_rows: list[tuple[str, list[str]]] = []
    for family, group in selected.groupby("family", sort=True):
        family_rows.append((str(family), group["gene_symbol"].astype(str).tolist()))
    preferred_by_family = {
        family: int(DEFAULT_GAP_BINS[(index * 3) % len(DEFAULT_GAP_BINS)])
        for index, (family, _genes) in enumerate(family_rows)
    }
    specs: list[dict[str, object]] = []
    orientations = list(ORIENTATION_TO_INDEX.keys())
    token_pool = [
        "CGT", "GCA", "TGC", "AGC", "CAT", "GAT", "TCG", "CGA", "ATG", "GCT",
    ]
    for family_index, (family, genes) in enumerate(family_rows):
        if len(genes) >= 2:
            specs.append(
                {
                    "left_tf": genes[0],
                    "right_tf": genes[1],
                    "family_key": family,
                    "preferred_gap": preferred_by_family[family],
                    "orientation": orientations[family_index % len(orientations)],
                    "interface_token": token_pool[(2 * family_index) % len(token_pool)],
                    "decoy_token": token_pool[(2 * family_index + 1) % len(token_pool)],
                    "composite": True if scenario == "ablation" else bool(family_index % 2 == 0),
                    "deployment_axis": family_index,
                    "state_gain": 1.25 + 0.15 * family_index,
                }
            )
        if family_index + 1 < len(family_rows):
            next_family, next_genes = family_rows[family_index + 1]
            specs.append(
                {
                    "left_tf": genes[0],
                    "right_tf": next_genes[0],
                    "family_key": f"{family}::{next_family}",
                    "preferred_gap": preferred_by_family[family],
                    "orientation": orientations[(family_index + 1) % len(orientations)],
                    "interface_token": token_pool[(3 * family_index + 3) % len(token_pool)],
                    "decoy_token": token_pool[(3 * family_index + 4) % len(token_pool)],
                    "composite": scenario == "ablation",
                    "deployment_axis": family_index + 1,
                    "state_gain": 1.1 + 0.1 * family_index,
                }
            )
    rng.shuffle(specs)
    return specs[: max(8, min(len(specs), 14))]


def _embed_pair_sequence(
    window_length: int,
    left_motif: str,
    right_motif: str,
    left_start: int,
    gap: int,
    orientation: str,
    interface_token: str,
    composite: bool,
    rng: np.random.Generator,
) -> tuple[str, int, int, int, int]:
    sequence = random_background(window_length, rng, gc=0.55)
    left_insert = left_motif if orientation[0] == "+" else reverse_complement(left_motif)
    right_insert = right_motif if orientation[1] == "+" else reverse_complement(right_motif)
    right_start = int(np.clip(left_start + len(left_insert) + gap, 8, window_length - len(right_insert) - 8))
    sequence = embed_sequence(sequence, left_insert, left_start)
    sequence = embed_sequence(sequence, right_insert, right_start)
    interface_start = min(left_start + len(left_insert), right_start)
    overlap_len = max(0, min(left_start + len(left_insert), right_start + len(right_insert)) - max(left_start, right_start))
    if composite and overlap_len == 0:
        token_start = max(interface_start - 1, left_start + len(left_insert) - 2)
    else:
        token_start = max(interface_start, 0)
    if token_start + len(interface_token) < window_length:
        sequence = embed_sequence(sequence, interface_token, token_start)
    return sequence, right_start, overlap_len, token_start, interface_start


def prepare_synthetic_scenario(
    project_root: str | Path,
    window_length: int = 96,
    num_phase1_per_tf: int = 96,
    num_phase2_per_pair: int = 192,
    num_state_examples_per_pair: int = 64,
    availability_dim: int = 16,
    state_dim: int = 32,
    seed: int = 11,
    scenario: str = "synthetic",
) -> pd.DataFrame:
    project_root = resolve_path(project_root)
    scenario_root = ensure_dir(project_root / "data_intermediate" / scenario)
    rng = np.random.default_rng(seed)
    selected = _select_diverse_tfs(project_root)
    consensus_by_gene = _load_consensus_map(project_root)
    specs = _pair_specs(selected, rng, scenario=scenario)

    rows: list[dict[str, object]] = []

    for tf in selected.itertuples(index=False):
        motif = consensus_by_gene[str(tf.gene_symbol)]
        for replicate in range(num_phase1_per_tf):
            positive = random_background(window_length, rng, gc=0.52)
            start = int(rng.integers(12, window_length - len(motif) - 12))
            positive = embed_sequence(positive, motif, start)
            hard_negative = random_background(window_length, rng, gc=0.52)
            hard_negative = embed_sequence(hard_negative, _mutate_motif(motif, rng), start)
            for label, sequence, role in [(1.0, positive, "positive"), (0.0, hard_negative, "hard_negative")]:
                rows.append(
                    {
                        "seq_id": f"{scenario}::phase1::{tf.gene_symbol}::{role}::{replicate}",
                        "phase": "phase1",
                        "sequence": sequence,
                        "label": label,
                        "tf_pair_label": f"{tf.gene_symbol}::{tf.gene_symbol}",
                        "left_tf": tf.gene_symbol,
                        "right_tf": tf.gene_symbol,
                        "left_anchor_start": start,
                        "left_anchor_end": start + len(motif),
                        "right_anchor_start": start,
                        "right_anchor_end": start + len(motif),
                        "orientation": "++",
                        "center_distance": 0.0,
                        "edge_gap": float(-len(motif)),
                        "overlap_len": float(len(motif)),
                        "coarse_additive_score": float(len(motif) + 0.25 * label),
                        "source_dataset": scenario,
                        "state_label": f"{scenario}_in_vitro",
                        "split_group": tf.gene_symbol,
                        "chromosome": f"{scenario}_chr{(replicate % 11) + 1}",
                        **{f"availability_{index}": 0.0 for index in range(availability_dim)},
                        **{f"state_{index}": 0.0 for index in range(state_dim)},
                    }
                )

    for pair_index, spec in enumerate(specs):
        left_motif = consensus_by_gene[str(spec["left_tf"])]
        right_motif = consensus_by_gene[str(spec["right_tf"])]
        preferred_gap = int(spec["preferred_gap"])
        positive_orientation = str(spec["orientation"])
        negative_orientation = list(ORIENTATION_TO_INDEX.keys())[(ORIENTATION_TO_INDEX[positive_orientation] + 1) % len(ORIENTATION_TO_INDEX)]
        for replicate in range(num_phase2_per_pair):
            positive = replicate % 2 == 0
            left_start = int(rng.integers(10, 22))
            gap = preferred_gap + int(rng.choice([-2, -1, 0, 1, 2]))
            interface_token = str(spec["interface_token"] if positive else spec["decoy_token"])
            orientation = positive_orientation if positive or replicate % 3 != 0 else negative_orientation
            sequence, right_start, overlap_len, _token_start, _interface_start = _embed_pair_sequence(
                window_length=window_length,
                left_motif=left_motif,
                right_motif=right_motif,
                left_start=left_start,
                gap=gap,
                orientation=orientation,
                interface_token=interface_token,
                composite=bool(spec["composite"]),
                rng=rng,
            )
            edge_gap = float(right_start - (left_start + len(left_motif)))
            center_distance = float((right_start + len(right_motif) / 2) - (left_start + len(left_motif) / 2))
            helical_bonus = 0.75 * _helical_alignment(int(round(edge_gap)), preferred_gap)
            interface_bonus = 1.5 if interface_token == str(spec["interface_token"]) else -1.2
            orientation_bonus = 0.6 if orientation == positive_orientation else -0.5
            pair_score = helical_bonus + interface_bonus + orientation_bonus - 0.6
            label = float(pair_score > 0.0)
            rows.append(
                {
                    "seq_id": f"{scenario}::phase2::{spec['left_tf']}::{spec['right_tf']}::{int(label)}::{replicate}",
                    "phase": "phase2",
                    "sequence": sequence,
                    "label": label,
                    "composite_label": float(bool(spec["composite"]) and label > 0.0),
                    "tf_pair_label": f"{spec['left_tf']}::{spec['right_tf']}",
                    "left_tf": spec["left_tf"],
                    "right_tf": spec["right_tf"],
                    "left_anchor_start": left_start,
                    "left_anchor_end": left_start + len(left_motif),
                    "right_anchor_start": right_start,
                    "right_anchor_end": right_start + len(right_motif),
                    "orientation": orientation,
                    "center_distance": center_distance,
                    "edge_gap": edge_gap,
                    "overlap_len": float(overlap_len),
                    "coarse_additive_score": float(len(left_motif) + len(right_motif) + pair_score),
                    "source_dataset": scenario,
                    "state_label": f"{scenario}_in_vitro",
                    "split_group": f"{spec['left_tf']}::{spec['right_tf']}",
                    "chromosome": f"{scenario}_pairchr{(pair_index % 13) + 1}",
                    **{f"availability_{index}": 0.0 for index in range(availability_dim)},
                    **{f"state_{index}": 0.0 for index in range(state_dim)},
                }
            )

    state_labels = [f"{scenario}_state_{index:02d}" for index in range(24)]
    state_payloads = []
    for state_index, state_label in enumerate(state_labels):
        lineage_index = state_index % 6
        state_level = 1.0 if state_index % 3 != 0 else -0.4
        availability_level = 0.85 if state_index % 4 != 0 else 0.25
        state_payloads.append(
            {
                "state_label": state_label,
                "lineage_index": lineage_index,
                "state_level": state_level,
                "availability_level": availability_level,
                "state_values": _state_vector(lineage_index, state_level, state_dim, rng),
                "availability_values": _availability_vector(availability_level, lineage_index, availability_dim, rng),
            }
        )

    for tf in selected.itertuples(index=False):
        motif = consensus_by_gene[str(tf.gene_symbol)]
        for state_index, payload in enumerate(state_payloads):
            sequence = random_background(window_length, rng, gc=0.54)
            start = 24
            sequence = embed_sequence(sequence, motif, start)
            availability_level = float(payload["availability_level"])
            state_level = float(payload["state_level"])
            monomer_score = 1.4 * availability_level + 0.35 * max(state_level, 0.0) - 0.8
            label = float(monomer_score > 0.0)
            rows.append(
                {
                    "seq_id": f"{scenario}::phase3::{tf.gene_symbol}::{state_index:03d}",
                    "phase": "phase3",
                    "sequence": sequence,
                    "label": label,
                    "composite_label": 0.0,
                    "tf_pair_label": f"{tf.gene_symbol}::{tf.gene_symbol}",
                    "left_tf": tf.gene_symbol,
                    "right_tf": tf.gene_symbol,
                    "left_anchor_start": start,
                    "left_anchor_end": start + len(motif),
                    "right_anchor_start": start,
                    "right_anchor_end": start + len(motif),
                    "orientation": "++",
                    "center_distance": 0.0,
                    "edge_gap": float(-len(motif)),
                    "overlap_len": float(len(motif)),
                    "coarse_additive_score": float(len(motif) + monomer_score),
                    "source_dataset": scenario,
                    "state_label": payload["state_label"],
                    "split_group": payload["state_label"],
                    "chromosome": f"{scenario}_statechr{(state_index % 17) + 1}",
                    **{f"availability_{index}": payload["availability_values"][index] for index in range(availability_dim)},
                    **{f"state_{index}": payload["state_values"][index] for index in range(state_dim)},
                }
            )

    helical_scale = 1.75 if scenario == "ablation" else 1.35
    bridge_scale = 1.45 if scenario == "ablation" else 1.1
    gate_scale = 1.35 if scenario == "ablation" else 1.0
    interface_match_rate = 0.65 if scenario == "ablation" else 0.5
    score_bias = -1.45 if scenario == "ablation" else -0.95
    monomer_scale = 0.08 if scenario == "ablation" else 0.25
    geometry_match_offsets = [-2, -1, 0, 1, 2]
    geometry_mismatch_offsets = [-7, -6, -5, 5, 6, 7]
    for pair_index, spec in enumerate(specs):
        left_motif = consensus_by_gene[str(spec["left_tf"])]
        right_motif = consensus_by_gene[str(spec["right_tf"])]
        preferred_gap = int(spec["preferred_gap"])
        orientation = str(spec["orientation"])
        for state_index, payload in enumerate(state_payloads):
            for replicate in range(num_state_examples_per_pair):
                positive_geometry = bool(rng.random() > 0.35)
                # Keep ablation strongly signal-driven without collapsing phase4 into a single positive class.
                interface_match = bool(rng.random() < interface_match_rate)
                left_start = int(rng.integers(12, 18))
                gap_choices = geometry_match_offsets if positive_geometry else geometry_mismatch_offsets
                gap = preferred_gap + int(rng.choice(gap_choices))
                token = str(spec["interface_token"] if interface_match else spec["decoy_token"])
                sequence, right_start, overlap_len, _token_start, _interface_start = _embed_pair_sequence(
                    window_length=window_length,
                    left_motif=left_motif,
                    right_motif=right_motif,
                    left_start=left_start,
                    gap=gap,
                    orientation=orientation,
                    interface_token=token,
                    composite=bool(spec["composite"]),
                    rng=rng,
                )
                edge_gap = float(right_start - (left_start + len(left_motif)))
                center_distance = float((right_start + len(right_motif) / 2) - (left_start + len(left_motif) / 2))
                availability_level = float(payload["availability_level"])
                state_level = float(payload["state_level"])
                helical_bonus = helical_scale * _helical_alignment(int(round(edge_gap)), preferred_gap)
                bridge_bonus = 1.7 * bridge_scale if interface_match else -1.5 * bridge_scale
                lineage_active = payload["lineage_index"] == int(spec["deployment_axis"]) % 6
                lineage_factor = 1.0 if lineage_active else 0.2
                availability_factor = 0.3 + 0.9 * availability_level
                interaction_gate = availability_factor * (
                    0.35 + gate_scale * float(spec["state_gain"]) * max(state_level, 0.0) * lineage_factor
                )
                interaction_base = 0.8 * helical_bonus + 0.9 * bridge_bonus
                state_correction = 0.25 * gate_scale * max(state_level, 0.0) * (1.0 if lineage_active else -0.4)
                composite_bonus = 0.5 if bool(spec["composite"]) and overlap_len > 0 else -0.1
                left_effective_energy = 1.45 - 2.0 * availability_level - 0.15 * max(state_level, 0.0)
                right_effective_energy = 1.35 - 1.9 * availability_level - 0.15 * max(state_level, 0.0)
                state_residual = -(1.1 * interaction_gate * interaction_base + 0.5 * state_correction)
                compatibility = float(np.exp(-max(0.0, overlap_len - 4.0)))
                log_z_mono = np.log1p(np.exp(-left_effective_energy) + np.exp(-right_effective_energy))
                log_z_pair = np.log(
                    np.exp(log_z_mono)
                    + compatibility * np.exp(-(left_effective_energy + right_effective_energy + state_residual))
                )
                cooperative_gain = log_z_pair - log_z_mono
                state_bias = 0.2 * max(state_level, 0.0) * (1.0 if lineage_active else -0.35)
                monomer_term = monomer_scale * availability_level
                total_score = (
                    0.35 * log_z_mono
                    + 1.75 * cooperative_gain
                    + 0.3 * monomer_term
                    + state_bias
                    + composite_bonus
                    + score_bias
                )
                label = float(total_score > 0.0)
                rows.append(
                    {
                        "seq_id": f"{scenario}::phase4::{spec['left_tf']}::{spec['right_tf']}::{state_index:03d}::{replicate:03d}",
                        "phase": "phase4",
                        "sequence": sequence,
                        "label": label,
                        "composite_label": float(bool(spec["composite"]) and interface_match),
                        "tf_pair_label": f"{spec['left_tf']}::{spec['right_tf']}",
                        "left_tf": spec["left_tf"],
                        "right_tf": spec["right_tf"],
                        "left_anchor_start": left_start,
                        "left_anchor_end": left_start + len(left_motif),
                        "right_anchor_start": right_start,
                        "right_anchor_end": right_start + len(right_motif),
                        "orientation": orientation,
                        "center_distance": center_distance,
                        "edge_gap": edge_gap,
                        "overlap_len": float(overlap_len),
                        "coarse_additive_score": float(len(left_motif) + len(right_motif) + total_score),
                        "source_dataset": scenario,
                        "state_label": payload["state_label"],
                        "split_group": payload["state_label"],
                        "chromosome": f"{scenario}_statechr{(state_index % 17) + 1}",
                        **{f"availability_{index}": payload["availability_values"][index] for index in range(availability_dim)},
                        **{f"state_{index}": payload["state_values"][index] for index in range(state_dim)},
                    }
                )

    frame = pd.DataFrame.from_records(rows)
    write_table(frame, scenario_root / "sequences.parquet")
    write_json(
        scenario_root / "scenario_manifest.json",
        {
            "seed": seed,
            "scenario": scenario,
            "num_rows": int(len(frame)),
            "num_pairs": int(len(specs)),
            "num_tfs": int(len(selected)),
            "num_states": int(len(state_labels)),
        },
    )
    return frame
