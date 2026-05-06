from __future__ import annotations

import functools
import gzip
import hashlib
import heapq
import itertools
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from helixpair.io_utils import ensure_dir, read_table, resolve_path, write_json, write_table
from helixpair.sequence import _prepare_scan_tracks, _scan_pwm_with_tracks, embed_sequence, pfm_to_pwm, scan_pwm

STANDARD_CHROMS = {f"chr{index}" for index in range(1, 23)} | {"chrX", "chrY"}
ELEMENT_PRIORITY = {
    "pELS": 0,
    "dELS": 1,
    "CA-TF": 2,
    "TF": 3,
    "CA-CTCF": 4,
    "CA": 5,
}
PHASE4_LABEL_VERSION = "reference_pair_support_v1"
PHASE4_LABEL_SOURCE = "reference_pair_support"
_UCSC_SESSION_LOCAL = threading.local()


def _stable_rank(*parts: object) -> int:
    key = "::".join(map(str, parts))
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:15], 16)


def _download_stream(url: str, destination: Path) -> Path:
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
    return destination


def _select_hca_fragments(manifest_path: Path, max_files: int) -> pd.DataFrame:
    manifest = read_table(manifest_path)
    fragments = manifest[manifest["format"].astype(str).str.contains("bed.gz", na=False)].copy()
    if fragments.empty:
        raise FileNotFoundError(f"No HCA fragment bed.gz entries found in {manifest_path}")
    fragments = fragments[fragments["azul_url"].astype(str).str.len() > 0].copy()
    fragments = fragments.sort_values(["size", "name"]).drop_duplicates("name")
    selected_rows: list[dict[str, Any]] = []
    used_organs: set[str] = set()
    for row in fragments.itertuples(index=False):
        organ = str(getattr(row, "organ", ""))
        if organ in used_organs:
            continue
        selected_rows.append(row._asdict())
        used_organs.add(organ)
        if len(selected_rows) >= max_files:
            break
    if len(selected_rows) < max_files:
        selected_names = {str(item["name"]) for item in selected_rows}
        for row in fragments.itertuples(index=False):
            if str(row.name) in selected_names:
                continue
            selected_rows.append(row._asdict())
            if len(selected_rows) >= max_files:
                break
    return pd.DataFrame.from_records(selected_rows).reset_index(drop=True)


def _sample_screen_regions(screen_path: Path, max_regions: int) -> pd.DataFrame:
    heap: list[tuple[int, int, dict[str, Any]]] = []
    with screen_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.rstrip().split("\t")
            if len(parts) < 6:
                continue
            chrom = parts[0]
            if chrom not in STANDARD_CHROMS:
                continue
            start = int(parts[1])
            end = int(parts[2])
            if end <= start:
                continue
            element_type = parts[5]
            priority = ELEMENT_PRIORITY.get(element_type, 9)
            rank = priority * (10**15) + _stable_rank(parts[3], parts[4], chrom, start, end)
            record = {
                "chromosome": chrom,
                "region_start": start,
                "region_end": end,
                "state_label": f"{chrom}:{start}-{end}",
                "screen_id": parts[3],
                "encode_id": parts[4],
                "element_type": element_type,
                "width": end - start,
            }
            item = (-rank, _stable_rank(chrom, start, end, element_type), record)
            if len(heap) < max_regions:
                heapq.heappush(heap, item)
                continue
            if item > heap[0]:
                heapq.heapreplace(heap, item)
    if not heap:
        raise FileNotFoundError(f"No candidate SCREEN regions were retained from {screen_path}")
    records = [item[2] for item in heap]
    frame = pd.DataFrame.from_records(records)
    return frame.sort_values(["chromosome", "region_start", "region_end"]).reset_index(drop=True)


def _build_region_bin_index(regions: pd.DataFrame, bin_size: int) -> dict[str, dict[int, list[int]]]:
    index: dict[str, dict[int, list[int]]] = {}
    for row in regions.itertuples():
        chrom_bins = index.setdefault(str(row.chromosome), {})
        start_bin = int(row.region_start) // bin_size
        end_bin = max(int(row.region_end) - 1, int(row.region_start)) // bin_size
        for bin_id in range(start_bin, end_bin + 1):
            chrom_bins.setdefault(bin_id, []).append(int(row.Index))
    return index


def _count_fragment_overlaps(
    fragment_path: Path,
    regions: pd.DataFrame,
    bin_index: dict[str, dict[int, list[int]]],
    bin_size: int,
    max_lines: int | None = None,
) -> tuple[np.ndarray, int]:
    starts = regions["region_start"].to_numpy(dtype=np.int64)
    ends = regions["region_end"].to_numpy(dtype=np.int64)
    counts = np.zeros((len(regions),), dtype=np.int64)
    processed = 0
    with gzip.open(fragment_path, "rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if max_lines is not None and processed >= max_lines:
                break
            parts = line.rstrip().split("\t")
            if len(parts) < 3:
                continue
            chrom = parts[0]
            if chrom not in bin_index:
                continue
            start = int(parts[1])
            end = int(parts[2])
            if end <= start:
                continue
            processed += 1
            candidate_ids: set[int] = set()
            start_bin = start // bin_size
            end_bin = max(end - 1, start) // bin_size
            chrom_bins = bin_index[chrom]
            for bin_id in range(start_bin, end_bin + 1):
                candidate_ids.update(chrom_bins.get(bin_id, []))
            for region_index in candidate_ids:
                if starts[region_index] < end and ends[region_index] > start:
                    counts[region_index] += 1
    return counts, processed


@functools.lru_cache(maxsize=32768)
def _fetch_ucsc_sequence(chrom: str, start: int, end: int) -> str:
    import requests

    session = getattr(_UCSC_SESSION_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        _UCSC_SESSION_LOCAL.session = session
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = session.get(
                "https://api.genome.ucsc.edu/getData/sequence",
                params={"genome": "hg38", "chrom": chrom, "start": start, "end": end},
                timeout=120,
            )
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "").strip()
                wait_seconds = float(retry_after) if retry_after else float(5 * (attempt + 1))
                time.sleep(max(wait_seconds, 1.0))
                continue
            response.raise_for_status()
            payload = response.json()
            return str(payload.get("dna", "")).upper()
        except Exception as exc:  # network retry path
            last_error = exc
            time.sleep(1.0 + attempt)
    if last_error is None:
        raise RuntimeError("UCSC sequence fetch failed without an exception.")
    raise last_error


def _centered_window(chrom: str, start: int, end: int, window_length: int) -> tuple[int, int]:
    center = (int(start) + int(end)) // 2
    left = max(center - window_length // 2, 0)
    return left, left + window_length


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
        consensus_by_gene[str(row.gene_symbol)] = "".join(positions[["A", "C", "G", "T"]].idxmax(axis=1).tolist())
    return consensus_by_gene


def _load_pwm_map(project_root: Path, genes: list[str]) -> dict[str, dict[str, Any]]:
    tf_master = read_table(project_root / "data_intermediate" / "tf_master_table.tsv")
    motif_positions = read_table(project_root / "data_intermediate" / "motif_matrix_positions.tsv")
    pwm_map: dict[str, dict[str, Any]] = {}
    target_genes = set(genes)
    for row in tf_master.itertuples(index=False):
        gene_symbol = str(getattr(row, "gene_symbol", ""))
        if gene_symbol not in target_genes:
            continue
        motif_id = str(getattr(row, "motif_id", ""))
        if not motif_id:
            continue
        positions = motif_positions[motif_positions["motif_id"] == motif_id].sort_values("position")
        if positions.empty:
            continue
        pfm = positions[["A", "C", "G", "T"]].to_numpy(dtype=np.float32).T
        pwm = pfm_to_pwm(pfm)
        consensus_score = float(np.max(pwm, axis=0).sum())
        pwm_map[gene_symbol] = {
            "pwm": pwm,
            "motif_len": int(pwm.shape[1]),
            "consensus_score": consensus_score,
            "cutoff": max(1.5, 0.55 * consensus_score),
        }
    return pwm_map


def _normalize_series(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    if values.empty:
        return values
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum - minimum < 1e-8:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return (values - minimum) / (maximum - minimum)


def _phase4_state_support_score(frame: pd.DataFrame) -> pd.Series:
    support = 0.7 * _normalize_series(frame["state_total_fragment_count"]) + 0.3 * _normalize_series(frame["state_supporting_files"])
    return support.clip(lower=0.0, upper=1.0)


def _assign_phase4_usage_labels(candidate_sequences: pd.DataFrame) -> pd.DataFrame:
    if candidate_sequences.empty or "phase" not in candidate_sequences.columns:
        return candidate_sequences
    frame = candidate_sequences.copy()
    for column in ["pair_grammar_score", "pair_usage_score", "state_support_score", "pair_rank_in_state"]:
        if column not in frame.columns:
            frame[column] = np.nan
    if "phase4_label_version" not in frame.columns:
        frame["phase4_label_version"] = ""
    if "phase4_label_source" not in frame.columns:
        frame["phase4_label_source"] = ""
    if "phase4_evidence_score" not in frame.columns:
        frame["phase4_evidence_score"] = np.nan
    if "phase4_evidence_rank" not in frame.columns:
        frame["phase4_evidence_rank"] = np.nan
    if "phase4_evidence_supported" not in frame.columns:
        frame["phase4_evidence_supported"] = np.nan

    phase4_mask = frame["phase"].astype(str) == "phase4"
    if not phase4_mask.any():
        return frame

    phase4_frame = frame.loc[phase4_mask].copy()
    if {"state_total_fragment_count", "state_supporting_files"}.issubset(phase4_frame.columns):
        phase4_state_support = _phase4_state_support_score(phase4_frame)
        frame.loc[phase4_frame.index, "state_support_score"] = phase4_state_support.to_numpy()

    observed_mask = phase4_frame["construction_mode"].astype(str) == "observed_pair"
    if observed_mask.any():
        observed = phase4_frame.loc[observed_mask].copy()
        if "pair_reference_supported" in observed.columns:
            evidence_supported = observed["pair_reference_supported"].astype(float).fillna(0.0)
            evidence_score = evidence_supported.copy()
            label_source = pd.Series(PHASE4_LABEL_SOURCE, index=observed.index, dtype=object)
            label_version = pd.Series(PHASE4_LABEL_VERSION, index=observed.index, dtype=object)
        else:
            evidence_supported = observed["phase4_evidence_supported"].astype(float).fillna(0.0)
            evidence_score = observed["phase4_evidence_score"].astype(float).fillna(evidence_supported)
            label_source = observed["phase4_label_source"].replace("", PHASE4_LABEL_SOURCE)
            label_version = observed["phase4_label_version"].replace("", PHASE4_LABEL_VERSION)

        observed["pair_grammar_score"] = np.nan
        observed["pair_usage_score"] = np.nan
        observed["pair_rank_in_state"] = np.nan
        observed["phase4_label_source"] = label_source
        observed["phase4_label_version"] = label_version
        observed["phase4_evidence_supported"] = evidence_supported
        observed["phase4_evidence_score"] = evidence_score
        observed = observed.sort_values(
            ["state_label", "phase4_evidence_score", "seq_id"],
            ascending=[True, False, True],
        ).copy()
        observed["phase4_evidence_rank"] = observed.groupby("state_label").cumcount().add(1).astype(float)
        observed["usage_label"] = 0.0
        observed["label"] = 0.0

        qualified_states: set[str] = set()
        for state_label, state_frame in observed.groupby("state_label", sort=False):
            scores = state_frame["phase4_evidence_score"].astype(float)
            top_score = float(scores.max())
            bottom_score = float(scores.min())
            unique_top = int((scores == top_score).sum()) == 1
            has_supported = bool((state_frame["phase4_evidence_supported"].astype(float) > 0.0).any())
            has_unsupported = bool((state_frame["phase4_evidence_supported"].astype(float) <= 0.0).any())
            if len(state_frame) >= 2 and unique_top and top_score > bottom_score and has_supported and has_unsupported:
                qualified_states.add(str(state_label))

        if qualified_states:
            qualified_mask = observed["state_label"].astype(str).isin(qualified_states)
            winners = observed.loc[qualified_mask].groupby("state_label")["phase4_evidence_rank"].idxmin()
            observed.loc[winners, "usage_label"] = 1.0
            observed.loc[winners, "label"] = 1.0

        frame.loc[
            observed.index,
            [
                "pair_grammar_score",
                "pair_usage_score",
                "pair_rank_in_state",
                "phase4_label_source",
                "phase4_label_version",
                "phase4_evidence_supported",
                "phase4_evidence_score",
                "phase4_evidence_rank",
                "usage_label",
                "label",
            ],
        ] = observed[
            [
                "pair_grammar_score",
                "pair_usage_score",
                "pair_rank_in_state",
                "phase4_label_source",
                "phase4_label_version",
                "phase4_evidence_supported",
                "phase4_evidence_score",
                "phase4_evidence_rank",
                "usage_label",
                "label",
            ]
        ].to_numpy()
        keep_mask = (~phase4_mask) | frame["state_label"].astype(str).isin(qualified_states)
        frame = frame.loc[keep_mask].copy()
        phase4_mask = frame["phase"].astype(str) == "phase4"

    imputed_mask = phase4_frame["construction_mode"].astype(str) == "imputed_pair"
    if imputed_mask.any():
        imputed_index = phase4_frame.loc[imputed_mask].index
        imputed_index = [index for index in imputed_index if index in frame.index]
        frame.loc[imputed_index, "pair_grammar_score"] = np.nan
        frame.loc[imputed_index, "pair_usage_score"] = np.nan
        frame.loc[imputed_index, "pair_rank_in_state"] = np.nan
        frame.loc[imputed_index, "phase4_label_source"] = PHASE4_LABEL_SOURCE
        frame.loc[imputed_index, "phase4_label_version"] = PHASE4_LABEL_VERSION
        frame.loc[imputed_index, "phase4_evidence_supported"] = 0.0
        frame.loc[imputed_index, "phase4_evidence_score"] = 0.0
        frame.loc[imputed_index, "phase4_evidence_rank"] = np.nan
        frame.loc[imputed_index, "usage_label"] = 0.0
        frame.loc[imputed_index, "label"] = 0.0
    return frame


def _load_reference_pairs(project_root: Path, consensus_by_gene: dict[str, str]) -> list[tuple[str, str]]:
    reference_graph_path = project_root / "data_intermediate" / "reference_graph.parquet"
    if not reference_graph_path.exists():
        return []
    reference_graph = read_table(reference_graph_path)
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in reference_graph.itertuples(index=False):
        pair = (str(row.left_tf), str(row.right_tf))
        if pair[0] not in consensus_by_gene or pair[1] not in consensus_by_gene:
            continue
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
    return pairs


def _anchor_payload(anchor, pwm_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "start": int(anchor.start),
        "end": int(anchor.end),
        "strand": str(anchor.strand),
        "width": int(anchor.end - anchor.start),
        "score": float(anchor.score),
        "cutoff": float(pwm_payload["cutoff"]),
        "consensus_score": float(pwm_payload["consensus_score"]),
    }


def _find_hit_lists(
    sequence: str,
    pwm_map: dict[str, dict[str, Any]],
    genes: list[str],
    top_k: int = 2,
) -> dict[str, list[dict[str, Any]]]:
    hits: dict[str, list[dict[str, Any]]] = {}
    forward_track, reverse_track = _prepare_scan_tracks(sequence)
    score_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for gene in genes:
        pwm_payload = pwm_map.get(gene)
        if not pwm_payload:
            continue
        anchors = _scan_pwm_with_tracks(
            sequence,
            pwm_payload["pwm"],
            tf_id=gene,
            top_k=top_k,
            cutoff=float(pwm_payload["cutoff"]),
            forward_track=forward_track,
            reverse_track=reverse_track,
            score_cache=score_cache,
        )
        if not anchors:
            continue
        hits[gene] = [_anchor_payload(anchor, pwm_payload) for anchor in anchors]
    return hits


def _audit_public_state_layer(
    candidate_sequences: pd.DataFrame,
    state_features: pd.DataFrame,
    window_length: int,
) -> dict[str, Any]:
    sequence_lengths = candidate_sequences["sequence"].astype(str).map(len)
    unique_states = set(state_features["state_label"].astype(str))
    represented_states = candidate_sequences["state_label"].astype(str).nunique()
    rows_per_state = candidate_sequences["state_label"].astype(str).value_counts()
    anchor_ok = (
        (candidate_sequences["left_anchor_start"].astype(int) >= 0)
        & (candidate_sequences["left_anchor_end"].astype(int) <= window_length)
        & (candidate_sequences["right_anchor_start"].astype(int) >= 0)
        & (candidate_sequences["right_anchor_end"].astype(int) <= window_length)
        & (candidate_sequences["left_anchor_start"].astype(int) < candidate_sequences["left_anchor_end"].astype(int))
        & (candidate_sequences["right_anchor_start"].astype(int) < candidate_sequences["right_anchor_end"].astype(int))
    )
    recalculated_overlap = (
        candidate_sequences[["left_anchor_end", "right_anchor_end"]].min(axis=1)
        - candidate_sequences[["left_anchor_start", "right_anchor_start"]].max(axis=1)
    ).clip(lower=0)
    recalculated_gap = candidate_sequences["right_anchor_start"].astype(int) - candidate_sequences["left_anchor_end"].astype(int)
    recalculated_distance = (
        (candidate_sequences["right_anchor_start"].astype(float) + candidate_sequences["right_anchor_end"].astype(float)) / 2.0
        - (candidate_sequences["left_anchor_start"].astype(float) + candidate_sequences["left_anchor_end"].astype(float)) / 2.0
    )
    audit = {
        "num_sequences": int(len(candidate_sequences)),
        "num_states": int(len(state_features)),
        "unique_seq_ids": int(candidate_sequences["seq_id"].astype(str).nunique()),
        "duplicate_seq_ids": int(len(candidate_sequences) - candidate_sequences["seq_id"].astype(str).nunique()),
        "sequence_length_min": int(sequence_lengths.min()) if not sequence_lengths.empty else 0,
        "sequence_length_max": int(sequence_lengths.max()) if not sequence_lengths.empty else 0,
        "missing_state_labels": int((~candidate_sequences["state_label"].astype(str).isin(unique_states)).sum()),
        "represented_states": int(represented_states),
        "states_without_sequences": int(len(state_features) - represented_states),
        "rows_per_state_min": int(rows_per_state.min()) if not rows_per_state.empty else 0,
        "rows_per_state_max": int(rows_per_state.max()) if not rows_per_state.empty else 0,
        "anchors_in_bounds": bool(anchor_ok.all()) if len(anchor_ok) else True,
        "overlap_consistency_max_abs_diff": float(
            (candidate_sequences["overlap_len"].astype(float) - recalculated_overlap.astype(float)).abs().max()
        )
        if len(candidate_sequences)
        else 0.0,
        "gap_consistency_max_abs_diff": float(
            (candidate_sequences["edge_gap"].astype(float) - recalculated_gap.astype(float)).abs().max()
        )
        if len(candidate_sequences)
        else 0.0,
        "center_distance_consistency_max_abs_diff": float(
            (candidate_sequences["center_distance"].astype(float) - recalculated_distance.astype(float)).abs().max()
        )
        if len(candidate_sequences)
        else 0.0,
        "phase_counts": {str(key): int(value) for key, value in candidate_sequences["phase"].astype(str).value_counts().items()},
        "construction_counts": {
            str(key): int(value) for key, value in candidate_sequences["construction_mode"].astype(str).value_counts().items()
        },
        "label_counts": {
            str(key): int(value) for key, value in candidate_sequences["label"].astype(float).round(3).value_counts().items()
        },
    }
    return audit


def _public_state_manifest_score(manifest: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return (
        int(manifest.get("observed_pair_rows", 0)),
        int(manifest.get("observed_monomer_rows", 0)),
        -int(manifest.get("imputed_pair_rows", 0)),
        -int(manifest.get("imputed_monomer_rows", 0)),
        int(manifest.get("num_candidate_sequences", 0)),
    )


def _build_state_feature_frame(states: pd.DataFrame, availability_dim: int, state_dim: int) -> pd.DataFrame:
    features = pd.DataFrame({"state_label": states["state_label"]})
    for index in range(availability_dim):
        features[f"availability_{index}"] = 0.0
    for index in range(state_dim):
        features[f"state_{index}"] = 0.0
    features["availability_0"] = _normalize_series(states["total_fragment_count"]).to_numpy()
    if availability_dim > 1:
        features["availability_1"] = _normalize_series(states["supporting_files"]).to_numpy()
    if availability_dim > 2 and "fragment_count_0" in states.columns:
        features["availability_2"] = _normalize_series(states["fragment_count_0"]).to_numpy()
    if availability_dim > 3 and "fragment_count_1" in states.columns:
        features["availability_3"] = _normalize_series(states["fragment_count_1"]).to_numpy()
    features["state_0"] = _normalize_series(states["width"]).to_numpy()
    if state_dim > 1:
        features["state_1"] = states["gc_content"].astype(float).to_numpy()
    if state_dim > 2:
        features["state_2"] = states["chromosome"].astype(str).map(lambda value: 1.0 if value in {"chrX", "chrY"} else 0.0).to_numpy()
    for offset, element_type in enumerate(["pELS", "dELS", "CA", "CA-CTCF", "CA-TF", "TF"], start=3):
        if offset >= state_dim:
            break
        features[f"state_{offset}"] = states["element_type"].astype(str).eq(element_type).astype(float).to_numpy()
    return features


def _build_public_state_layer_result(
    project_root: str | Path,
    availability_dim: int = 16,
    state_dim: int = 32,
    window_length: int = 96,
    max_fragment_files: int = 1,
    max_candidate_regions: int = 4096,
    max_states: int = 128,
    state_selection_multiplier: int = 4,
    max_examples: int | None = None,
    max_pairs_per_state: int = 1,
    max_monomers_per_state: int = 1,
    max_fragment_lines: int | None = None,
    impute_missing_monomers: bool = True,
    impute_missing_pairs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    project_root = resolve_path(project_root)
    screen_path = project_root / "data_raw" / "encode" / "screen" / "GRCh38-cCREs.bed"
    manifest_path = project_root / "data_intermediate" / "hca_fragment_manifest.tsv"
    if not screen_path.exists():
        raise FileNotFoundError(f"Missing ENCODE SCREEN bed file: {screen_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing HCA fragment manifest: {manifest_path}")

    fragment_rows = _select_hca_fragments(manifest_path, max_files=max_fragment_files)
    fragment_root = ensure_dir(project_root / "data_raw" / "hca" / "fragments")
    fragment_paths: list[Path] = []
    for row in fragment_rows.itertuples(index=False):
        fragment_paths.append(_download_stream(str(row.azul_url), fragment_root / str(row.name)))

    sampled_regions = _sample_screen_regions(screen_path, max_regions=max_candidate_regions)
    bin_size = 100_000
    bin_index = _build_region_bin_index(sampled_regions, bin_size=bin_size)
    processed_lines: dict[str, int] = {}
    for file_index, fragment_path in enumerate(fragment_paths):
        counts, processed = _count_fragment_overlaps(
            fragment_path,
            sampled_regions,
            bin_index,
            bin_size=bin_size,
            max_lines=max_fragment_lines,
        )
        sampled_regions[f"fragment_count_{file_index}"] = counts
        processed_lines[fragment_path.name] = processed

    count_columns = [column for column in sampled_regions.columns if column.startswith("fragment_count_")]
    sampled_regions["total_fragment_count"] = sampled_regions[count_columns].sum(axis=1)
    sampled_regions["supporting_files"] = (sampled_regions[count_columns] > 0).sum(axis=1)
    candidate_states = sampled_regions[sampled_regions["total_fragment_count"] > 0].copy()
    if candidate_states.empty:
        raise RuntimeError("HCA fragments did not overlap any sampled SCREEN regions.")
    candidate_states = candidate_states.sort_values(
        ["total_fragment_count", "supporting_files", "chromosome", "region_start"],
        ascending=[False, False, True, True],
    ).head(max(int(max_states) * max(int(state_selection_multiplier), 1), int(max_states)))

    sequences = []
    for row in candidate_states.itertuples(index=False):
        window_start, window_end = _centered_window(str(row.chromosome), int(row.region_start), int(row.region_end), window_length)
        sequence = _fetch_ucsc_sequence(str(row.chromosome), window_start, window_end)
        if len(sequence) < window_length:
            sequence = sequence + ("N" * (window_length - len(sequence)))
        if len(sequence) != window_length:
            continue
        sequences.append(
            {
                "state_label": str(row.state_label),
                "chromosome": str(row.chromosome),
                "window_start": int(window_start),
                "window_end": int(window_end),
                "sequence": sequence.upper(),
                "gc_content": float((sequence.upper().count("G") + sequence.upper().count("C")) / max(len(sequence), 1)),
            }
        )
    sequence_frame = pd.DataFrame.from_records(sequences)
    if sequence_frame.empty:
        raise RuntimeError("Failed to fetch any UCSC genomic windows for selected states.")
    consensus_by_gene = _load_consensus_map(project_root)
    reference_pairs = _load_reference_pairs(project_root, consensus_by_gene)
    if not reference_pairs:
        raise RuntimeError("Public state builder requires a reference TF-pair graph with motif consensuses.")

    genes = sorted({gene for pair in reference_pairs for gene in pair})
    pwm_map = _load_pwm_map(project_root, genes)
    if not pwm_map:
        raise RuntimeError("Public state builder requires at least one PWM-backed TF motif.")
    selected_states = candidate_states.merge(sequence_frame, on=["state_label", "chromosome"], how="inner")
    if selected_states.empty:
        raise RuntimeError("Selected states were lost during UCSC sequence resolution.")
    reference_pair_set = {tuple(sorted(pair)) for pair in reference_pairs}
    hit_cache: dict[str, dict[str, dict[str, Any]]] = {}
    hit_list_cache: dict[str, dict[str, list[dict[str, Any]]]] = {}
    monomer_hit_counts: list[int] = []
    pair_hit_counts: list[int] = []
    hit_top_k = max(2, int(max_pairs_per_state))
    for row in selected_states.itertuples(index=False):
        hit_lists = _find_hit_lists(str(row.sequence), pwm_map, genes, top_k=hit_top_k)
        best_hits = {gene: anchors[0] for gene, anchors in hit_lists.items() if anchors}
        hit_list_cache[str(row.state_label)] = hit_lists
        hit_cache[str(row.state_label)] = best_hits
        monomer_hit_counts.append(int(len(best_hits)))
        pair_hit_count = 0
        observed_genes = sorted(hit_lists)
        for left_tf, right_tf in itertools.combinations(observed_genes, 2):
            anchor_pairs = len(hit_lists[left_tf]) * len(hit_lists[right_tf])
            if tuple(sorted((left_tf, right_tf))) in reference_pair_set:
                pair_hit_count += anchor_pairs
            else:
                pair_hit_count += min(anchor_pairs, 1)
        pair_hit_counts.append(pair_hit_count)
    selected_states["observed_monomer_candidate_count"] = monomer_hit_counts
    selected_states["observed_pair_candidate_count"] = pair_hit_counts
    selected_states = selected_states.sort_values(
        [
            "observed_pair_candidate_count",
            "observed_monomer_candidate_count",
            "total_fragment_count",
            "supporting_files",
            "chromosome",
            "region_start",
        ],
        ascending=[False, False, False, False, True, True],
    ).head(max_states).reset_index(drop=True)
    threshold = float(selected_states["total_fragment_count"].median())
    state_features = _build_state_feature_frame(selected_states, availability_dim=availability_dim, state_dim=state_dim)
    sequence_rows: list[dict[str, Any]] = []
    for state_index, row in enumerate(selected_states.itertuples(index=False)):
        hits = hit_cache.get(str(row.state_label), {})
        label = float(float(row.total_fragment_count) >= threshold)

        matched_genes = sorted(hits.keys(), key=lambda gene: (-int(hits[gene]["width"]), gene))
        for monomer_gene in matched_genes[:max_monomers_per_state]:
            hit = hits[monomer_gene]
            seq_id = f"public::phase3::{monomer_gene}::{state_index:05d}"
            sequence_rows.append(
                {
                    "seq_id": seq_id,
                    "sequence": str(row.sequence),
                    "state_label": str(row.state_label),
                    "tf_pair_label": f"{monomer_gene}::{monomer_gene}",
                    "left_tf": monomer_gene,
                    "right_tf": monomer_gene,
                    "left_anchor_start": int(hit["start"]),
                    "left_anchor_end": int(hit["end"]),
                    "right_anchor_start": int(hit["start"]),
                    "right_anchor_end": int(hit["end"]),
                    "orientation": str(hit["strand"]) * 2,
                    "center_distance": 0.0,
                    "edge_gap": float(-int(hit["width"])),
                    "overlap_len": float(int(hit["width"])),
                    "coarse_additive_score": float(int(hit["width"]) + label),
                    "usage_label": label,
                    "label": label,
                    "phase": "phase3",
                    "composite_label": 0.0,
                    "element_type": "monomer_observed_public",
                    "source_dataset": "HCA_SCREEN_public",
                    "construction_mode": "observed_monomer",
                    "anchor_score": float(hit["score"]),
                    "anchor_cutoff": float(hit["cutoff"]),
                    "anchor_consensus_score": float(hit["consensus_score"]),
                    "anchor_width": int(hit["width"]),
                    "state_total_fragment_count": float(row.total_fragment_count),
                    "state_supporting_files": float(row.supporting_files),
                    "observed_monomer_candidate_count": int(row.observed_monomer_candidate_count),
                    "observed_pair_candidate_count": int(row.observed_pair_candidate_count),
                    "split_group": str(row.chromosome),
                    "chromosome": str(row.chromosome),
                }
            )

        if not matched_genes and impute_missing_monomers:
            monomer_gene = reference_pairs[state_index % len(reference_pairs)][0]
            monomer_motif = consensus_by_gene[monomer_gene]
            monomer_start = min(24, max(0, window_length - len(monomer_motif) - 1))
            imputed_sequence = embed_sequence(str(row.sequence), monomer_motif, monomer_start)
            seq_id = f"public::phase3::{monomer_gene}::imputed::{state_index:05d}"
            sequence_rows.append(
                {
                    "seq_id": seq_id,
                    "sequence": imputed_sequence,
                    "state_label": str(row.state_label),
                    "tf_pair_label": f"{monomer_gene}::{monomer_gene}",
                    "left_tf": monomer_gene,
                    "right_tf": monomer_gene,
                    "left_anchor_start": int(monomer_start),
                    "left_anchor_end": int(monomer_start + len(monomer_motif)),
                    "right_anchor_start": int(monomer_start),
                    "right_anchor_end": int(monomer_start + len(monomer_motif)),
                    "orientation": "++",
                    "center_distance": 0.0,
                    "edge_gap": float(-len(monomer_motif)),
                    "overlap_len": float(len(monomer_motif)),
                    "coarse_additive_score": float(len(monomer_motif) + label),
                    "usage_label": label,
                    "label": label,
                    "phase": "phase3",
                    "composite_label": 0.0,
                    "element_type": "monomer_imputed_public",
                    "source_dataset": "HCA_SCREEN_public",
                    "construction_mode": "imputed_monomer",
                    "anchor_score": float("nan"),
                    "anchor_cutoff": float("nan"),
                    "anchor_consensus_score": float(len(monomer_motif)),
                    "anchor_width": int(len(monomer_motif)),
                    "state_total_fragment_count": float(row.total_fragment_count),
                    "state_supporting_files": float(row.supporting_files),
                    "observed_monomer_candidate_count": int(row.observed_monomer_candidate_count),
                    "observed_pair_candidate_count": int(row.observed_pair_candidate_count),
                    "split_group": str(row.chromosome),
                    "chromosome": str(row.chromosome),
                }
            )

        pair_candidates: list[dict[str, Any]] = []
        hit_lists = hit_list_cache.get(str(row.state_label), {})
        observed_genes = sorted(hit_lists)
        target_pair_budget = max(int(max_pairs_per_state), 2)
        for left_tf, right_tf in itertools.combinations(observed_genes, 2):
            reference_supported = tuple(sorted((left_tf, right_tf))) in reference_pair_set
            for left_rank, left_hit in enumerate(hit_lists[left_tf]):
                for right_rank, right_hit in enumerate(hit_lists[right_tf]):
                    center_distance = float((right_hit["start"] + right_hit["end"]) / 2 - (left_hit["start"] + left_hit["end"]) / 2)
                    edge_gap = float(int(right_hit["start"]) - int(left_hit["end"]))
                    overlap_len = float(
                        max(0, min(int(left_hit["end"]), int(right_hit["end"])) - max(int(left_hit["start"]), int(right_hit["start"])))
                    )
                    width_sum = int(left_hit["width"]) + int(right_hit["width"])
                    gap_penalty = abs(edge_gap - 6.0)
                    overlap_penalty = overlap_len * 0.5
                    anchor_rank_penalty = left_rank + right_rank
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
                            "candidate_priority": (
                                1 if reference_supported else 0,
                                width_sum,
                                -gap_penalty,
                                -overlap_penalty,
                                -anchor_rank_penalty,
                            ),
                            "candidate_key": f"{left_tf}::{right_tf}::{left_rank}::{right_rank}",
                        }
                    )
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
        pair_candidates = pair_candidates[:target_pair_budget]
        for pair_rank, pair in enumerate(pair_candidates):
            seq_id = f"public::phase4::{pair['left_tf']}::{pair['right_tf']}::{state_index:05d}::{pair_rank:02d}"
            sequence_rows.append(
                {
                    "seq_id": seq_id,
                    "sequence": str(row.sequence),
                    "state_label": str(row.state_label),
                    "tf_pair_label": f"{pair['left_tf']}::{pair['right_tf']}",
                    "left_tf": pair["left_tf"],
                    "right_tf": pair["right_tf"],
                    "left_anchor_start": int(pair["left_hit"]["start"]),
                    "left_anchor_end": int(pair["left_hit"]["end"]),
                    "right_anchor_start": int(pair["right_hit"]["start"]),
                    "right_anchor_end": int(pair["right_hit"]["end"]),
                    "orientation": f"{pair['left_hit']['strand']}{pair['right_hit']['strand']}",
                    "center_distance": float(pair["center_distance"]),
                    "edge_gap": float(pair["edge_gap"]),
                    "overlap_len": float(pair["overlap_len"]),
                    "coarse_additive_score": float(pair["score"] + label),
                    "usage_label": label,
                    "label": label,
                    "phase": "phase4",
                    "composite_label": 0.0,
                    "element_type": "state_usage_observed_public",
                    "source_dataset": "HCA_SCREEN_public",
                    "construction_mode": "observed_pair",
                    "left_anchor_score": float(pair["left_hit"]["score"]),
                    "right_anchor_score": float(pair["right_hit"]["score"]),
                    "left_anchor_consensus_score": float(pair["left_hit"]["consensus_score"]),
                    "right_anchor_consensus_score": float(pair["right_hit"]["consensus_score"]),
                    "left_anchor_width": int(pair["left_hit"]["width"]),
                    "right_anchor_width": int(pair["right_hit"]["width"]),
                    "state_total_fragment_count": float(row.total_fragment_count),
                    "state_supporting_files": float(row.supporting_files),
                    "observed_pair_candidate_count": int(len(pair_candidates)),
                    "pair_candidate_rank": int(pair_rank),
                    "pair_reference_supported": float(pair["reference_supported"]),
                    "pair_candidate_key": str(pair["candidate_key"]),
                    "split_group": str(row.chromosome),
                    "chromosome": str(row.chromosome),
                }
            )

        if pair_candidates or not impute_missing_pairs:
            continue
        left_tf, right_tf = reference_pairs[state_index % len(reference_pairs)]
        left_motif = consensus_by_gene[left_tf]
        right_motif = consensus_by_gene[right_tf]
        left_start = 16
        right_start = min(left_start + len(left_motif) + 8, window_length - len(right_motif) - 8)
        imputed_sequence = embed_sequence(str(row.sequence), left_motif, left_start)
        imputed_sequence = embed_sequence(imputed_sequence, right_motif, right_start)
        seq_id = f"public::phase4::{left_tf}::{right_tf}::imputed::{state_index:05d}"
        sequence_rows.append(
            {
                "seq_id": seq_id,
                "sequence": imputed_sequence,
                "state_label": str(row.state_label),
                "tf_pair_label": f"{left_tf}::{right_tf}",
                "left_tf": left_tf,
                "right_tf": right_tf,
                "left_anchor_start": int(left_start),
                "left_anchor_end": int(left_start + len(left_motif)),
                "right_anchor_start": int(right_start),
                "right_anchor_end": int(right_start + len(right_motif)),
                "orientation": "++",
                "center_distance": float((right_start + len(right_motif) / 2) - (left_start + len(left_motif) / 2)),
                "edge_gap": float(right_start - (left_start + len(left_motif))),
                "overlap_len": float(0.0),
                "coarse_additive_score": float(len(left_motif) + len(right_motif) + label),
                "usage_label": label,
                "label": label,
                "phase": "phase4",
                "composite_label": 0.0,
                "element_type": "state_usage_imputed_public",
                "source_dataset": "HCA_SCREEN_public",
                "construction_mode": "imputed_pair",
                "left_anchor_score": float("nan"),
                "right_anchor_score": float("nan"),
                "left_anchor_consensus_score": float(len(left_motif)),
                "right_anchor_consensus_score": float(len(right_motif)),
                "left_anchor_width": int(len(left_motif)),
                "right_anchor_width": int(len(right_motif)),
                "state_total_fragment_count": float(row.total_fragment_count),
                "state_supporting_files": float(row.supporting_files),
                "observed_pair_candidate_count": int(row.observed_pair_candidate_count),
                "split_group": str(row.chromosome),
                "chromosome": str(row.chromosome),
            }
        )

    candidate_sequences = pd.DataFrame.from_records(sequence_rows)
    if candidate_sequences.empty:
        raise RuntimeError("Public state builder did not materialize any phase3/phase4 candidate sequences.")
    candidate_sequences = _assign_phase4_usage_labels(candidate_sequences)
    if max_examples is not None and len(candidate_sequences) > max_examples:
        candidate_sequences = candidate_sequences.head(max_examples).copy()

    audit = _audit_public_state_layer(candidate_sequences, state_features, window_length=window_length)
    construction_counts = candidate_sequences["construction_mode"].astype(str).value_counts().to_dict()

    manifest = {
        "builder": "public_state_layer",
        "sequence_source": "UCSC_hg38_api",
        "state_source": "HCA_fragments + ENCODE_SCREEN",
        "fragment_files": [str(path) for path in fragment_paths],
        "fragment_processed_lines": processed_lines,
        "num_sampled_regions": int(len(sampled_regions)),
        "num_candidate_states": int(len(candidate_states)),
        "num_selected_states": int(len(selected_states)),
        "num_state_rows": int(len(state_features)),
        "num_candidate_sequences": int(len(candidate_sequences)),
        "observed_monomer_rows": int(construction_counts.get("observed_monomer", 0)),
        "imputed_monomer_rows": int(construction_counts.get("imputed_monomer", 0)),
        "observed_pair_rows": int(construction_counts.get("observed_pair", 0)),
        "imputed_pair_rows": int(construction_counts.get("imputed_pair", 0)),
        "max_fragment_files": int(max_fragment_files),
        "max_candidate_regions": int(max_candidate_regions),
        "max_states": int(max_states),
        "state_selection_multiplier": int(state_selection_multiplier),
        "window_length": int(window_length),
        "audit": audit,
    }
    return candidate_sequences, state_features, manifest


def build_public_state_layer_inputs(
    project_root: str | Path,
    availability_dim: int = 16,
    state_dim: int = 32,
    window_length: int = 96,
    max_fragment_files: int = 1,
    max_candidate_regions: int = 4096,
    max_states: int = 128,
    state_selection_multiplier: int = 4,
    max_examples: int | None = None,
    max_pairs_per_state: int = 1,
    max_monomers_per_state: int = 1,
    max_fragment_lines: int | None = None,
    impute_missing_monomers: bool = True,
    impute_missing_pairs: bool = True,
    select_best_fragment_file_count: bool = True,
) -> tuple[Path, Path, dict[str, Any]]:
    project_root = resolve_path(project_root)
    requested_max_fragment_files = max(int(max_fragment_files), 1)
    fragment_file_counts = [requested_max_fragment_files]
    selection_mode = "fixed"
    if select_best_fragment_file_count and requested_max_fragment_files > 1:
        fragment_file_counts = list(range(1, requested_max_fragment_files + 1))
        selection_mode = "best_prefix"

    best_result: tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]] | None = None
    best_score: tuple[int, int, int, int, int] | None = None
    best_fragment_file_count: int | None = None
    evaluations: list[dict[str, Any]] = []
    for fragment_file_count in fragment_file_counts:
        candidate_sequences, state_features, manifest = _build_public_state_layer_result(
            project_root,
            availability_dim=availability_dim,
            state_dim=state_dim,
            window_length=window_length,
            max_fragment_files=fragment_file_count,
            max_candidate_regions=max_candidate_regions,
            max_states=max_states,
            state_selection_multiplier=state_selection_multiplier,
            max_examples=max_examples,
            max_pairs_per_state=max_pairs_per_state,
            max_monomers_per_state=max_monomers_per_state,
            max_fragment_lines=max_fragment_lines,
            impute_missing_monomers=impute_missing_monomers,
            impute_missing_pairs=impute_missing_pairs,
        )
        score = _public_state_manifest_score(manifest)
        evaluations.append(
            {
                "fragment_file_count": int(fragment_file_count),
                "fragment_files": list(manifest["fragment_files"]),
                "observed_monomer_rows": int(manifest["observed_monomer_rows"]),
                "observed_pair_rows": int(manifest["observed_pair_rows"]),
                "imputed_monomer_rows": int(manifest["imputed_monomer_rows"]),
                "imputed_pair_rows": int(manifest["imputed_pair_rows"]),
                "num_candidate_states": int(manifest["num_candidate_states"]),
                "num_selected_states": int(manifest["num_selected_states"]),
                "num_candidate_sequences": int(manifest["num_candidate_sequences"]),
            }
        )
        if best_score is None or score > best_score or (score == best_score and fragment_file_count < int(best_fragment_file_count)):
            best_result = (candidate_sequences, state_features, manifest)
            best_score = score
            best_fragment_file_count = int(fragment_file_count)

    if best_result is None or best_fragment_file_count is None:
        raise RuntimeError("Public state builder did not evaluate any fragment-file configurations.")

    candidate_sequences, state_features, manifest = best_result
    output_root = ensure_dir(project_root / "data_raw" / "state_layer")
    sequence_path = output_root / "candidate_sequences.parquet"
    state_path = output_root / "state_features.parquet"
    write_table(candidate_sequences, sequence_path)
    write_table(state_features, state_path)
    write_json(output_root / "public_state_audit.json", manifest["audit"])
    manifest.update(
        {
            "max_fragment_files": int(requested_max_fragment_files),
            "selected_fragment_file_count": int(best_fragment_file_count),
            "fragment_count_selection_mode": selection_mode,
            "fragment_count_evaluations": evaluations,
        }
    )
    write_json(output_root / "public_state_manifest.json", manifest)
    return sequence_path, state_path, manifest
