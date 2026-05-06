from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from helixpair.external_functional import ENCODE_MPRA_SPECS, hydrate_region_sequences
from helixpair.public_state import (
    _find_hit_lists,
    _load_consensus_map,
    _load_pwm_map,
    _load_reference_pairs,
)
from helixpair.bundles import build_tensor_bundles, build_windows_and_candidates
from helixpair.io_utils import ensure_dir, resolve_path, write_json, write_table
from helixpair.splits import assign_split


SCENARIO = "external_functional_mpra_benchmark"
ENDOGENOUS_PRIORITY_BIOSAMPLES = ("K562", "HepG2", "WTC11")
AVAILABILITY_DIM = 16
STATE_DIM = 32


def _strip_reverse_suffix(value: str) -> str:
    return str(value).replace("_Reversed:", "")


def _is_control_element(value: str) -> bool:
    token = str(value)
    return token == "no_BC" or token.startswith("ENSG")


def _aggregate_mpra_bed(frame: pd.DataFrame) -> pd.DataFrame:
    local = frame.copy()
    local["element_group"] = local["name"].astype(str).map(_strip_reverse_suffix)
    local["is_reverse"] = local["name"].astype(str).str.endswith("_Reversed:")
    aggregated = (
        local.groupby("element_group", as_index=False)
        .agg(
            chromosome=("chromosome", "first"),
            region_start=("region_start", "min"),
            region_end=("region_end", "max"),
            bed_rows=("name", "size"),
            bed_reverse_rows=("is_reverse", "sum"),
            mean_bed_activity=("activity_score", "mean"),
            std_bed_activity=("activity_score", "std"),
        )
        .reset_index(drop=True)
    )
    aggregated["bed_forward_rows"] = aggregated["bed_rows"] - aggregated["bed_reverse_rows"]
    aggregated["element_width"] = aggregated["region_end"].astype(int) - aggregated["region_start"].astype(int)
    return aggregated


def _aggregate_mpra_tsv(frame: pd.DataFrame) -> pd.DataFrame:
    local = frame.copy()
    local["element_group"] = local["name"].astype(str).map(_strip_reverse_suffix)
    local["is_reverse"] = local["name"].astype(str).str.endswith("_Reversed:")
    grouped = local.groupby("element_group", as_index=False)
    aggregated = grouped.agg(
        tsv_rows=("name", "size"),
        replicate_count=("replicate", "nunique"),
        replicate_min=("replicate", "min"),
        replicate_max=("replicate", "max"),
        mean_log2_activity=("log2", "mean"),
        std_log2_activity=("log2", "std"),
        mean_ratio=("ratio", "mean"),
        std_ratio=("ratio", "std"),
        min_obs_bc=("n_obs_bc", "min"),
        mean_obs_bc=("n_obs_bc", "mean"),
        reverse_rows=("is_reverse", "sum"),
    )
    aggregated["forward_rows"] = aggregated["tsv_rows"] - aggregated["reverse_rows"]

    orientation_means = (
        local.groupby(["element_group", "is_reverse"], as_index=False)["log2"]
        .mean()
        .pivot(index="element_group", columns="is_reverse", values="log2")
        .rename(columns={False: "forward_mean_log2", True: "reverse_mean_log2"})
        .reset_index()
    )
    aggregated = aggregated.merge(orientation_means, on="element_group", how="left")
    aggregated["orientation_gap_log2"] = (
        aggregated["forward_mean_log2"].astype(float) - aggregated["reverse_mean_log2"].astype(float)
    ).abs()
    return aggregated


def summarize_mpra_elements(
    bed_frame: pd.DataFrame,
    tsv_frame: pd.DataFrame,
    *,
    biosample: str,
    source_dataset: str,
    positive_quantile: float = 0.9,
    negative_quantile: float = 0.1,
    min_replicates: int = 3,
    min_obs_bc: int = 10,
    max_log2_std: float = 0.75,
    max_orientation_gap: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    bed_summary = _aggregate_mpra_bed(bed_frame)
    tsv_summary = _aggregate_mpra_tsv(tsv_frame)
    merged = bed_summary.merge(tsv_summary, on="element_group", how="inner", validate="one_to_one")
    merged["biosample_label"] = biosample
    merged["state_label"] = biosample
    merged["source_dataset"] = source_dataset
    merged["assay_family"] = "lentiMPRA"
    merged["is_control"] = merged["element_group"].astype(str).map(_is_control_element)
    merged["passes_quality"] = (
        ~merged["is_control"]
        & merged["replicate_count"].astype(int).ge(int(min_replicates))
        & merged["min_obs_bc"].astype(float).ge(float(min_obs_bc))
        & merged["std_log2_activity"].fillna(0.0).astype(float).le(float(max_log2_std))
        & merged["orientation_gap_log2"].fillna(0.0).astype(float).le(float(max_orientation_gap))
    )

    filtered = merged.loc[merged["passes_quality"]].copy()
    if filtered.empty:
        merged["functional_label"] = pd.NA
        merged["usage_label"] = pd.NA
        merged["label"] = pd.NA
        merged["quantile_low"] = pd.NA
        merged["quantile_high"] = pd.NA
        merged["rank_desc"] = pd.NA
        return merged, {
            "biosample": biosample,
            "source_dataset": source_dataset,
            "rows_total": int(len(merged)),
            "rows_quality": 0,
            "rows_labeled": 0,
        }

    low = float(filtered["mean_log2_activity"].quantile(float(negative_quantile)))
    high = float(filtered["mean_log2_activity"].quantile(float(positive_quantile)))

    merged["quantile_low"] = low
    merged["quantile_high"] = high
    merged["label"] = pd.NA
    merged.loc[merged["passes_quality"] & merged["mean_log2_activity"].ge(high), "label"] = 1.0
    merged.loc[merged["passes_quality"] & merged["mean_log2_activity"].le(low), "label"] = 0.0
    merged["functional_label"] = merged["label"]
    merged["usage_label"] = merged["label"]
    merged["rank_desc"] = merged["mean_log2_activity"].rank(method="first", ascending=False)
    merged["seq_id"] = (
        "external_mpra::"
        + merged["biosample_label"].astype(str).str.lower()
        + "::"
        + merged["element_group"].astype(str)
    )
    merged["split_group"] = merged["element_group"].astype(str)
    merged["phase"] = "phase5"
    merged["construction_mode"] = "external_functional_extrema"
    merged["element_type"] = "mpra_candidate"
    merged["composite_label"] = 1.0

    labeled_rows = merged["label"].notna()
    return merged, {
        "biosample": biosample,
        "source_dataset": source_dataset,
        "rows_total": int(len(merged)),
        "rows_quality": int(merged["passes_quality"].sum()),
        "rows_labeled": int(labeled_rows.sum()),
        "positive_rows": int((merged["label"] == 1.0).sum()),
        "negative_rows": int((merged["label"] == 0.0).sum()),
        "quantile_low_value": low,
        "quantile_high_value": high,
    }


def _build_default_split_manifest(frame: pd.DataFrame, *, split_name: str = "default") -> pd.DataFrame:
    records: list[dict[str, str]] = []
    for row in frame.itertuples(index=False):
        group_value = str(row.split_group)
        split = assign_split(group_value)
        records.append(
            {
                "seq_id": str(row.seq_id),
                "split_name": split_name,
                "split": split,
                "group_value": group_value,
            }
        )
    return pd.DataFrame.from_records(records)


def _build_cross_biosample_split_manifest(frame: pd.DataFrame, *, heldout_biosample: str) -> pd.DataFrame:
    split_name = f"cross_biosample_test_{heldout_biosample.lower()}"
    records: list[dict[str, str]] = []
    for row in frame.itertuples(index=False):
        if str(row.biosample_label) == heldout_biosample:
            split = "test"
        else:
            split = assign_split(str(row.split_group), train_fraction=0.85, valid_fraction=0.15)
            if split == "test":
                split = "train"
        records.append(
            {
                "seq_id": str(row.seq_id),
                "split_name": split_name,
                "split": split,
                "group_value": str(row.split_group),
            }
        )
    return pd.DataFrame.from_records(records)


def _rebalance_filtered_split_manifest(
    manifest: pd.DataFrame,
    candidate_rows: pd.DataFrame,
    *,
    preserve_test: bool,
) -> pd.DataFrame:
    if manifest.empty:
        return manifest
    label_lookup = (
        candidate_rows[["seq_id", "label"]]
        .drop_duplicates("seq_id", keep="last")
        .assign(seq_id=lambda frame: frame["seq_id"].astype(str))
        .set_index("seq_id")["label"]
        .to_dict()
    )
    local = manifest.copy()
    local["seq_id"] = local["seq_id"].astype(str)
    local["label"] = local["seq_id"].map(label_lookup)
    movable_sources = ["train", "valid"] if preserve_test else ["train", "valid", "test"]
    targets = ["train", "valid"] if preserve_test else ["train", "valid", "test"]

    def _label_presence(split_name: str) -> set[float]:
        return set(local.loc[local["split"] == split_name, "label"].dropna().astype(float).tolist())

    for target in targets:
        present = _label_presence(target)
        for label in [0.0, 1.0]:
            if label in present:
                continue
            for source in movable_sources:
                source_rows = local.loc[(local["split"] == source) & (local["label"].astype(float) == label)].copy()
                if len(source_rows) <= 1 and source == target:
                    continue
                if source != target and source_rows.empty:
                    continue
                if source == target:
                    continue
                if len(local.loc[local["split"] == source]) <= 1:
                    continue
                move_index = source_rows.index[0]
                local.loc[move_index, "split"] = target
                present.add(label)
                break
    return local.drop(columns="label")


def canonicalize_endogenous_registry(frame: pd.DataFrame) -> pd.DataFrame:
    local = frame.copy()
    biosample = local["biosample_label"].astype(str)
    local["canonical_biosample"] = "other"
    local.loc[biosample.str.contains("K562", case=False, na=False), "canonical_biosample"] = "K562"
    local.loc[biosample.str.contains("HepG2", case=False, na=False), "canonical_biosample"] = "HepG2"
    local.loc[biosample.str.contains("WTC11", case=False, na=False), "canonical_biosample"] = "WTC11"
    local["is_priority_biosample"] = local["canonical_biosample"].isin(ENDOGENOUS_PRIORITY_BIOSAMPLES)
    local["registry_id"] = (
        local["canonical_biosample"].astype(str)
        + "::"
        + local["enhancer_id"].astype(str)
        + "::"
        + local["putative_target_gene"].fillna("").astype(str)
        + "::"
        + local["perturbation_platform"].astype(str)
    )
    return local


def build_mpra_state_features(
    biosamples: list[str],
    *,
    availability_dim: int = AVAILABILITY_DIM,
    state_dim: int = STATE_DIM,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    ordered = sorted([str(item) for item in biosamples])
    for index, biosample in enumerate(ordered):
        row: dict[str, Any] = {"state_label": biosample}
        for feature_index in range(int(availability_dim)):
            row[f"availability_{feature_index}"] = 0.0
        for feature_index in range(int(state_dim)):
            row[f"state_{feature_index}"] = 0.0
        if index < availability_dim:
            row[f"availability_{index}"] = 1.0
        if index < state_dim:
            row[f"state_{index}"] = 1.0
        row["availability_0"] = max(float(row["availability_0"]), 1.0)
        if state_dim > 30:
            row["state_30"] = float(index)
        if state_dim > 31:
            row["state_31"] = float(len(ordered))
        records.append(row)
    return pd.DataFrame.from_records(records)


def select_mpra_candidate_pair(
    hit_lists: dict[str, list[dict[str, Any]]],
    *,
    reference_pair_set: set[tuple[str, str]] | None = None,
    prefer_gap: float = 6.0,
) -> dict[str, Any] | None:
    reference_pair_set = reference_pair_set or set()
    genes = sorted(str(gene) for gene in hit_lists)
    pair_candidates: list[dict[str, Any]] = []
    for left_rank, left_tf in enumerate(genes):
        left_hits = hit_lists.get(left_tf, [])
        for right_tf in genes[left_rank + 1 :]:
            right_hits = hit_lists.get(right_tf, [])
            reference_supported = tuple(sorted((left_tf, right_tf))) in reference_pair_set
            for left_hit_rank, left_hit in enumerate(left_hits):
                for right_hit_rank, right_hit in enumerate(right_hits):
                    center_distance = float(
                        (float(right_hit["start"]) + float(right_hit["end"])) / 2.0
                        - (float(left_hit["start"]) + float(left_hit["end"])) / 2.0
                    )
                    edge_gap = float(float(right_hit["start"]) - float(left_hit["end"]))
                    overlap_len = float(
                        max(
                            0.0,
                            min(float(left_hit["end"]), float(right_hit["end"]))
                            - max(float(left_hit["start"]), float(right_hit["start"])),
                        )
                    )
                    width_sum = int(left_hit["width"]) + int(right_hit["width"])
                    gap_penalty = abs(edge_gap - float(prefer_gap))
                    overlap_penalty = overlap_len * 0.5
                    anchor_rank_penalty = left_hit_rank + right_hit_rank
                    pair_candidates.append(
                        {
                            "left_tf": left_tf,
                            "right_tf": right_tf,
                            "left_hit": left_hit,
                            "right_hit": right_hit,
                            "center_distance": center_distance,
                            "edge_gap": edge_gap,
                            "overlap_len": overlap_len,
                            "score": width_sum,
                            "reference_supported": reference_supported,
                            "pair_mode": "heterotypic_reference" if reference_supported else "heterotypic_fallback",
                            "candidate_priority": (
                                1 if reference_supported else 0,
                                width_sum,
                                -gap_penalty,
                                -overlap_penalty,
                                -anchor_rank_penalty,
                            ),
                            "candidate_key": f"{left_tf}::{right_tf}::{left_hit_rank}::{right_hit_rank}",
                        }
                    )

    if pair_candidates:
        pair_candidates.sort(
            key=lambda item: (
                -int(item["candidate_priority"][0]),
                -int(item["candidate_priority"][1]),
                float(-item["candidate_priority"][2]),
                float(-item["candidate_priority"][3]),
                float(-item["candidate_priority"][4]),
                str(item["left_tf"]),
                str(item["right_tf"]),
            )
        )
        return pair_candidates[0]

    homotypic_candidates: list[dict[str, Any]] = []
    for gene in genes:
        hits = hit_lists.get(gene, [])
        if len(hits) < 2:
            continue
        for left_hit_rank, left_hit in enumerate(hits[:-1]):
            for right_hit_rank, right_hit in enumerate(hits[left_hit_rank + 1 :], start=left_hit_rank + 1):
                center_distance = float(
                    (float(right_hit["start"]) + float(right_hit["end"])) / 2.0
                    - (float(left_hit["start"]) + float(left_hit["end"])) / 2.0
                )
                edge_gap = float(float(right_hit["start"]) - float(left_hit["end"]))
                overlap_len = float(
                    max(
                        0.0,
                        min(float(left_hit["end"]), float(right_hit["end"]))
                        - max(float(left_hit["start"]), float(right_hit["start"])),
                    )
                )
                width_sum = int(left_hit["width"]) + int(right_hit["width"])
                gap_penalty = abs(edge_gap - float(prefer_gap))
                overlap_penalty = overlap_len * 0.5
                anchor_rank_penalty = left_hit_rank + right_hit_rank
                homotypic_candidates.append(
                    {
                        "left_tf": gene,
                        "right_tf": gene,
                        "left_hit": left_hit,
                        "right_hit": right_hit,
                        "center_distance": center_distance,
                        "edge_gap": edge_gap,
                        "overlap_len": overlap_len,
                        "score": width_sum,
                        "reference_supported": False,
                        "pair_mode": "homotypic_fallback",
                        "candidate_priority": (
                            width_sum,
                            -gap_penalty,
                            -overlap_penalty,
                            -anchor_rank_penalty,
                        ),
                        "candidate_key": f"{gene}::{gene}::{left_hit_rank}::{right_hit_rank}",
                    }
                )
    if not homotypic_candidates:
        return None
    homotypic_candidates.sort(
        key=lambda item: (
            -int(item["candidate_priority"][0]),
            float(-item["candidate_priority"][1]),
            float(-item["candidate_priority"][2]),
            float(-item["candidate_priority"][3]),
            str(item["left_tf"]),
        )
    )
    return homotypic_candidates[0]


def build_mpra_phase5_candidate_rows(
    hydrated_panel: pd.DataFrame,
    *,
    reference_pair_set: set[tuple[str, str]],
    pwm_map: dict[str, dict[str, Any]],
    candidate_genes: list[str],
    top_k_hits: int = 2,
    existing_status: pd.DataFrame | None = None,
    max_pending_rows: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    records: list[dict[str, Any]] = []
    status_records: list[dict[str, Any]] = []
    processed_ids: set[str] = set()
    missing_pair_rows = 0
    pair_mode_counts: dict[str, int] = {}
    if existing_status is not None and not existing_status.empty:
        processed_ids = set(existing_status["seq_id"].astype(str).tolist())
        missing_pair_rows = int((~existing_status["has_candidate"].astype(bool)).sum())
        cached_pair_modes = (
            existing_status.loc[existing_status["has_candidate"].astype(bool), "pair_mode"]
            .astype(str)
            .value_counts()
            .to_dict()
        )
        for mode, count in cached_pair_modes.items():
            pair_mode_counts[str(mode)] = int(count)

    pending = hydrated_panel.loc[~hydrated_panel["seq_id"].astype(str).isin(processed_ids)].copy()
    if max_pending_rows is not None:
        pending = pending.head(int(max_pending_rows)).copy()

    for row in pending.itertuples(index=False):
        hit_lists = _find_hit_lists(str(row.sequence), pwm_map, candidate_genes, top_k=int(top_k_hits))
        candidate = select_mpra_candidate_pair(hit_lists, reference_pair_set=reference_pair_set)
        if candidate is None:
            missing_pair_rows += 1
            status_records.append(
                {
                    "seq_id": str(row.seq_id),
                    "has_candidate": False,
                    "pair_mode": "",
                    "pair_reference_supported": 0.0,
                }
            )
            continue
        pair_mode = str(candidate["pair_mode"])
        pair_mode_counts[pair_mode] = int(pair_mode_counts.get(pair_mode, 0)) + 1
        status_records.append(
            {
                "seq_id": str(row.seq_id),
                "has_candidate": True,
                "pair_mode": pair_mode,
                "pair_reference_supported": float(candidate["reference_supported"]),
            }
        )
        records.append(
            {
                "seq_id": str(row.seq_id),
                "sequence": str(row.sequence),
                "chromosome": str(row.chromosome),
                "region_start": int(row.region_start),
                "region_end": int(row.region_end),
                "window_start": int(getattr(row, "window_start", row.region_start)),
                "window_end": int(getattr(row, "window_end", row.region_end)),
                "state_label": str(row.state_label),
                "biosample_label": str(row.biosample_label),
                "tf_pair_label": f"{candidate['left_tf']}::{candidate['right_tf']}",
                "left_tf": str(candidate["left_tf"]),
                "right_tf": str(candidate["right_tf"]),
                "left_anchor_start": int(candidate["left_hit"]["start"]),
                "left_anchor_end": int(candidate["left_hit"]["end"]),
                "right_anchor_start": int(candidate["right_hit"]["start"]),
                "right_anchor_end": int(candidate["right_hit"]["end"]),
                "orientation": f"{candidate['left_hit']['strand']}{candidate['right_hit']['strand']}",
                "center_distance": float(candidate["center_distance"]),
                "edge_gap": float(candidate["edge_gap"]),
                "overlap_len": float(candidate["overlap_len"]),
                "coarse_additive_score": float(candidate["score"]),
                "label": float(row.label),
                "functional_label": float(row.functional_label),
                "usage_label": float(row.usage_label),
                "phase": "phase5",
                "split_group": str(row.split_group),
                "composite_label": float(row.composite_label),
                "element_group": str(row.element_group),
                "source_dataset": str(row.source_dataset),
                "assay_family": str(row.assay_family),
                "construction_mode": "external_functional_observed_pair_candidate",
                "element_type": "mpra_candidate_pair",
                "pair_mode": pair_mode,
                "pair_reference_supported": float(candidate["reference_supported"]),
                "pair_candidate_key": str(candidate["candidate_key"]),
                "pair_hit_gene_count": int(len(hit_lists)),
                "priority_rank_within_biosample": int(row.priority_rank_within_biosample),
                "mean_log2_activity": float(row.mean_log2_activity),
                "std_log2_activity": float(row.std_log2_activity),
                "orientation_gap_log2": float(row.orientation_gap_log2),
                "left_anchor_score": float(candidate["left_hit"]["score"]),
                "right_anchor_score": float(candidate["right_hit"]["score"]),
                "left_anchor_width": int(candidate["left_hit"]["width"]),
                "right_anchor_width": int(candidate["right_hit"]["width"]),
            }
        )

    frame = pd.DataFrame.from_records(records)
    status_frame = pd.DataFrame.from_records(status_records)
    if existing_status is not None and not existing_status.empty:
        status_frame = pd.concat([existing_status, status_frame], ignore_index=True, sort=False)
        status_frame = status_frame.drop_duplicates("seq_id", keep="last").copy()
    manifest = {
        "hydrated_rows": int(len(hydrated_panel)),
        "pending_rows": int(len(pending)),
        "processed_rows": int(len(status_frame)) if not status_frame.empty else int(len(frame)),
        "candidate_rows": int(status_frame["has_candidate"].astype(bool).sum()) if not status_frame.empty else int(len(frame)),
        "rows_without_candidate_pair": int(missing_pair_rows),
        "pair_mode_counts": {str(key): int(value) for key, value in sorted(pair_mode_counts.items())},
        "reference_supported_rows": int(frame["pair_reference_supported"].astype(float).gt(0).sum()) if not frame.empty else 0,
    }
    return frame, manifest, status_frame


def _hydrate_mpra_sequence_windows(
    project_root: Path,
    mpra_panel: pd.DataFrame,
    *,
    window_length: int,
    max_elements: int | None = None,
    max_missing_rows: int | None = None,
) -> pd.DataFrame:
    scenario_root = ensure_dir(project_root / "data_intermediate" / SCENARIO)
    cache_path = scenario_root / "mpra_panel_sequence_windows.parquet"

    if max_elements is None:
        target = mpra_panel.copy()
    else:
        grouped = [
            frame.sort_values(
                ["priority_rank_within_biosample", "element_group"],
                ascending=[True, True],
            ).reset_index(drop=True)
            for _, frame in mpra_panel.groupby(["biosample_label", "label"], sort=True)
        ]
        picked: list[pd.Series] = []
        cursor = 0
        while len(picked) < int(max_elements):
            advanced = False
            for frame in grouped:
                if cursor < len(frame):
                    picked.append(frame.iloc[cursor])
                    advanced = True
                    if len(picked) >= int(max_elements):
                        break
            if not advanced:
                break
            cursor += 1
        target = pd.DataFrame(picked).reset_index(drop=True).copy()
    target_ids = target["seq_id"].astype(str).tolist()

    existing = pd.DataFrame()
    if cache_path.exists():
        existing = pd.read_parquet(cache_path)
        existing = existing.drop_duplicates("seq_id", keep="last").copy()
    existing_ids = set(existing["seq_id"].astype(str).tolist()) if not existing.empty else set()
    missing = target.loc[~target["seq_id"].astype(str).isin(existing_ids)].copy()
    if max_missing_rows is not None:
        missing = missing.head(int(max_missing_rows)).copy()
    if not missing.empty:
        batch_rows = 256
        for start in range(0, len(missing), batch_rows):
            hydrated_missing = hydrate_region_sequences(
                missing.iloc[start : start + batch_rows].copy(),
                window_length=int(window_length),
                max_workers=8,
            )
            combined = pd.concat([existing, hydrated_missing], ignore_index=True, sort=False)
            combined = combined.drop_duplicates("seq_id", keep="last").copy()
            write_table(combined, cache_path)
            existing = combined
    if existing.empty:
        return existing
    lookup = existing.set_index(existing["seq_id"].astype(str), drop=False)
    available_target_ids = [seq_id for seq_id in target_ids if seq_id in lookup.index]
    hydrated = lookup.loc[available_target_ids].reset_index(drop=True).copy()
    return hydrated


def build_external_functional_mpra_phase5_scenario(
    project_root: str | Path,
    *,
    split_names: tuple[str, ...] = ("default", "cross_biosample_test_hepg2", "cross_biosample_test_k562"),
    window_length: int = 96,
    max_elements: int | None = None,
    top_k_hits: int = 2,
    max_pending_rows: int | None = None,
    max_hydrate_rows: int | None = None,
) -> dict[str, Any]:
    project_root = resolve_path(project_root)
    scenario_root = ensure_dir(project_root / "data_intermediate" / SCENARIO)
    split_root = ensure_dir(project_root / "splits" / SCENARIO)
    report_root = ensure_dir(project_root / "reports" / SCENARIO)

    mpra_panel = pd.read_parquet(scenario_root / "mpra_panel.parquet")
    state_features = pd.read_parquet(scenario_root / "state_features.parquet")
    hydrated = _hydrate_mpra_sequence_windows(
        project_root,
        mpra_panel,
        window_length=int(window_length),
        max_elements=max_elements,
        max_missing_rows=max_hydrate_rows,
    )
    target_hydrated_rows = int(len(mpra_panel)) if max_elements is None else int(max_elements)
    hydrate_pending_rows = max(0, target_hydrated_rows - int(len(hydrated)))
    if hydrate_pending_rows > 0:
        if max_hydrate_rows is None:
            raise RuntimeError(
                f"External functional MPRA hydration incomplete: {len(hydrated)} / {target_hydrated_rows} rows cached."
            )
        return {
            "scenario": SCENARIO,
            "status": "hydrate_incomplete",
            "window_length": int(window_length),
            "hydrated_rows": int(len(hydrated)),
            "target_hydrated_rows": int(target_hydrated_rows),
            "hydrate_pending_rows": int(hydrate_pending_rows),
            "max_hydrate_rows": int(max_hydrate_rows),
            "cache_path": str(scenario_root / "mpra_panel_sequence_windows.parquet"),
        }
    consensus_map = _load_consensus_map(project_root)
    reference_pairs = _load_reference_pairs(project_root, consensus_map)
    candidate_genes = sorted({gene for pair in reference_pairs for gene in pair})
    pwm_map = _load_pwm_map(project_root, candidate_genes)
    reference_pair_set = {tuple(sorted(pair)) for pair in reference_pairs}
    candidate_cache_path = scenario_root / "candidate_sequences.parquet"
    status_cache_path = scenario_root / "candidate_scan_status.parquet"
    existing_candidate_rows = pd.DataFrame()
    existing_status = pd.DataFrame()
    if candidate_cache_path.exists():
        existing_candidate_rows = pd.read_parquet(candidate_cache_path).drop_duplicates("seq_id", keep="last").copy()
    if status_cache_path.exists():
        existing_status = pd.read_parquet(status_cache_path).drop_duplicates("seq_id", keep="last").copy()

    new_candidate_rows, candidate_manifest, status_frame = build_mpra_phase5_candidate_rows(
        hydrated,
        reference_pair_set=reference_pair_set,
        pwm_map=pwm_map,
        candidate_genes=[gene for gene in candidate_genes if gene in pwm_map],
        top_k_hits=int(top_k_hits),
        existing_status=existing_status,
        max_pending_rows=max_pending_rows,
    )
    candidate_rows = pd.concat([existing_candidate_rows, new_candidate_rows], ignore_index=True, sort=False)
    candidate_rows = candidate_rows.drop_duplicates("seq_id", keep="last").copy()
    candidate_rows = candidate_rows.loc[candidate_rows["seq_id"].astype(str).isin(hydrated["seq_id"].astype(str))].copy()
    if candidate_rows.empty:
        raise RuntimeError("External functional MPRA scenario builder found no pair-aware candidate rows.")

    write_table(candidate_rows, candidate_cache_path)
    write_table(candidate_rows, scenario_root / "sequences.parquet")
    write_table(candidate_rows, report_root / "phase5_candidate_sequences.csv")
    write_table(status_frame, status_cache_path)
    write_table(status_frame, report_root / "phase5_candidate_scan_status.csv")
    write_table(state_features, scenario_root / "state_features.parquet")

    build_windows_and_candidates(
        project_root,
        scenario=SCENARIO,
        window_length=int(window_length),
        top_k_anchors=8,
        top_k_pairs=8,
    )

    bundle_outputs: dict[str, dict[str, str]] = {}
    split_counts: dict[str, dict[str, int]] = {}
    for split_name in split_names:
        split_manifest_path = split_root / f"{split_name}.parquet"
        if not split_manifest_path.exists():
            raise FileNotFoundError(f"Missing split manifest for external functional scenario: {split_manifest_path}")
        split_manifest = pd.read_parquet(split_manifest_path)
        filtered_manifest = split_manifest.loc[
            split_manifest["seq_id"].astype(str).isin(candidate_rows["seq_id"].astype(str))
        ].copy()
        if filtered_manifest.empty:
            raise RuntimeError(f"Split manifest {split_name} retained no candidate rows.")
        filtered_manifest = _rebalance_filtered_split_manifest(
            filtered_manifest,
            candidate_rows,
            preserve_test=split_name.startswith("cross_biosample_test_"),
        )
        write_table(filtered_manifest, report_root / f"{split_name}_phase5_candidate_manifest.csv")
        split_counts[split_name] = {
            str(key): int(value) for key, value in filtered_manifest["split"].value_counts().sort_index().items()
        }
        bundle_outputs[split_name] = build_tensor_bundles(
            project_root,
            scenario=SCENARIO,
            window_length=int(window_length),
            split_name=split_name,
            split_manifest=split_manifest_path,
            phases=["phase5"],
        )

    manifest = {
        "scenario": SCENARIO,
        "window_length": int(window_length),
        "top_k_hits": int(top_k_hits),
        "max_elements": None if max_elements is None else int(max_elements),
        "max_pending_rows": None if max_pending_rows is None else int(max_pending_rows),
        "reference_pair_count": int(len(reference_pairs)),
        "reference_gene_count": int(len(candidate_genes)),
        "hydrated_sequence_cache": str(scenario_root / "mpra_panel_sequence_windows.parquet"),
        "outputs": {
            "candidate_sequences": str(candidate_cache_path),
            "candidate_scan_status": str(status_cache_path),
            "sequences": str(scenario_root / "sequences.parquet"),
            "state_features": str(scenario_root / "state_features.parquet"),
            **{f"bundle_{name}": values for name, values in bundle_outputs.items()},
        },
        "split_counts": split_counts,
        **candidate_manifest,
    }
    write_json(report_root / "phase5_scenario_manifest.json", manifest)
    return manifest


def build_external_functional_benchmarks(
    project_root: str | Path,
    *,
    positive_quantile: float = 0.9,
    negative_quantile: float = 0.1,
    min_replicates: int = 3,
    min_obs_bc: int = 10,
    max_log2_std: float = 0.75,
    max_orientation_gap: float = 1.0,
    hydrate_sequences: bool = False,
    window_length: int = 96,
    sequence_limit: int | None = None,
) -> dict[str, Any]:
    project_root = resolve_path(project_root)
    raw_root = ensure_dir(project_root / "data_raw" / "external_functional")
    source_root = ensure_dir(project_root / "data_intermediate" / "external_functional")
    scenario_root = ensure_dir(project_root / "data_intermediate" / SCENARIO)
    split_root = ensure_dir(project_root / "splits" / SCENARIO)
    report_root = ensure_dir(project_root / "reports" / SCENARIO)

    mpra_tables: list[pd.DataFrame] = []
    mpra_stats: list[dict[str, Any]] = []
    for spec in ENCODE_MPRA_SPECS:
        dataset_root = raw_root / spec["dataset_id"]
        bed_candidates = sorted(dataset_root.glob("ActivityRatios.*_full.bed.gz"))
        tsv_candidates = sorted(dataset_root.glob("*_full.ActivityRatios.tsv"))
        if not bed_candidates or not tsv_candidates:
            raise FileNotFoundError(f"Missing MPRA raw files under {dataset_root}")
        bed_frame = pd.read_csv(
            bed_candidates[0],
            sep="\t",
            header=None,
            names=[
                "chromosome",
                "region_start",
                "region_end",
                "name",
                "bed_score",
                "strand",
                "activity_score",
                "activity_aux_1",
                "activity_aux_2",
                "activity_aux_3",
                "activity_aux_4",
            ],
            compression="gzip",
        )
        tsv_frame = pd.read_csv(tsv_candidates[0], sep="\t")
        summary, stats = summarize_mpra_elements(
            bed_frame,
            tsv_frame,
            biosample=str(spec["biosample"]),
            source_dataset=str(spec["dataset_id"]),
            positive_quantile=float(positive_quantile),
            negative_quantile=float(negative_quantile),
            min_replicates=int(min_replicates),
            min_obs_bc=int(min_obs_bc),
            max_log2_std=float(max_log2_std),
            max_orientation_gap=float(max_orientation_gap),
        )
        mpra_tables.append(summary)
        mpra_stats.append(stats)

    mpra_panel = pd.concat(mpra_tables, ignore_index=True, sort=False)
    mpra_panel = mpra_panel.loc[mpra_panel["label"].notna()].copy()
    mpra_panel["label"] = mpra_panel["label"].astype(float)
    mpra_panel["functional_label"] = mpra_panel["functional_label"].astype(float)
    mpra_panel["usage_label"] = mpra_panel["usage_label"].astype(float)
    mpra_panel["priority_rank_within_biosample"] = (
        mpra_panel.groupby("biosample_label")["mean_log2_activity"].rank(method="first", ascending=False).astype(int)
    )
    write_table(mpra_panel, scenario_root / "mpra_panel.parquet")
    write_table(mpra_panel, report_root / "mpra_panel.csv")

    state_features = build_mpra_state_features(
        sorted(mpra_panel["biosample_label"].dropna().astype(str).unique().tolist())
    )
    write_table(state_features, scenario_root / "state_features.parquet")

    sequence_scaffold = mpra_panel.merge(state_features, on="state_label", how="left", validate="many_to_one").copy()
    write_table(sequence_scaffold, scenario_root / "sequence_scaffold.parquet")

    default_split = _build_default_split_manifest(mpra_panel, split_name="default")
    write_table(default_split, split_root / "default.parquet")
    write_table(default_split, report_root / "default_split_manifest.csv")

    cross_outputs: dict[str, str] = {}
    cross_counts: dict[str, dict[str, int]] = {}
    for biosample in sorted(mpra_panel["biosample_label"].dropna().astype(str).unique().tolist()):
        manifest = _build_cross_biosample_split_manifest(mpra_panel, heldout_biosample=biosample)
        split_name = f"cross_biosample_test_{biosample.lower()}"
        output_path = split_root / f"{split_name}.parquet"
        write_table(manifest, output_path)
        write_table(manifest, report_root / f"{split_name}_split_manifest.csv")
        cross_outputs[split_name] = str(output_path)
        cross_counts[split_name] = {
            str(key): int(value) for key, value in manifest["split"].value_counts().sort_index().items()
        }

    hydrated_outputs: dict[str, str] = {}
    if hydrate_sequences:
        hydrated = hydrate_region_sequences(
            mpra_panel,
            window_length=int(window_length),
            max_rows=sequence_limit,
        )
        output_path = scenario_root / "mpra_panel_sequence_windows.parquet"
        write_table(hydrated, output_path)
        hydrated_outputs["mpra_panel_sequence_windows"] = str(output_path)

    endogenous_frame = pd.read_parquet(source_root / "endogenous_perturbation_observations.parquet")
    endogenous_registry = canonicalize_endogenous_registry(endogenous_frame)
    priority_registry = endogenous_registry.loc[endogenous_registry["is_priority_biosample"]].copy()
    write_table(endogenous_registry, report_root / "endogenous_registry.csv")
    write_table(priority_registry, report_root / "endogenous_priority_registry.csv")

    flowfish_dataset_manifest_path = project_root / "reports" / "external_functional" / "flowfish_dataset_manifest.csv"
    flowfish_summary: list[dict[str, Any]] = []
    if flowfish_dataset_manifest_path.exists():
        flowfish_summary = pd.read_csv(flowfish_dataset_manifest_path).to_dict(orient="records")

    manifest = {
        "scenario": SCENARIO,
        "mpra_stats": mpra_stats,
        "mpra_rows": int(len(mpra_panel)),
        "mpra_label_counts": {str(key): int(value) for key, value in mpra_panel["label"].value_counts().sort_index().items()},
        "default_split_counts": {str(key): int(value) for key, value in default_split["split"].value_counts().sort_index().items()},
        "cross_biosample_split_counts": cross_counts,
        "priority_endogenous_rows": int(len(priority_registry)),
        "priority_endogenous_counts": {
            str(key): int(value)
            for key, value in priority_registry["canonical_biosample"].value_counts().sort_index().items()
        },
        "flowfish_dataset_manifest": flowfish_summary,
        "outputs": {
            "mpra_panel": str(scenario_root / "mpra_panel.parquet"),
            "state_features": str(scenario_root / "state_features.parquet"),
            "sequence_scaffold": str(scenario_root / "sequence_scaffold.parquet"),
            "default_split": str(split_root / "default.parquet"),
            "endogenous_registry": str(report_root / "endogenous_registry.csv"),
            "endogenous_priority_registry": str(report_root / "endogenous_priority_registry.csv"),
            **cross_outputs,
            **hydrated_outputs,
        },
    }
    write_json(report_root / "benchmark_manifest.json", manifest)
    return manifest
