from __future__ import annotations

import gzip
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from scipy.stats import fisher_exact, mannwhitneyu

from helixpair.bundles import build_tensor_bundles, build_windows_and_candidates
from helixpair.config import load_config
from helixpair.data import make_loader
from helixpair.inference import load_model_for_inference
from helixpair.io_utils import ensure_dir, read_table, resolve_path, write_json, write_table, write_text
from helixpair.public_state import _assign_phase4_usage_labels, _find_hit_lists, _load_consensus_map, _load_pwm_map, _load_reference_pairs
from helixpair.public_state import _build_public_state_layer_result, _public_state_manifest_score
from helixpair.training import _forward_model, _scores_for_phase


ENCODE_SEARCH_URL = "https://www.encodeproject.org/search/"
ENCODE_HEADERS = {"accept": "application/json"}
ENCODE_ASSAYS = ("chip-seq", "cut&run")
PEAK_OUTPUT_TOKENS = ("idr thresholded peaks", "pseudoreplicated peaks")
ENHANCER_LIKE_TYPES = {"pELS", "dELS"}


def _phase2_reference_paths(project_root: Path) -> tuple[Path, Path]:
    prediction_path = project_root / "results" / "per_example_predictions" / "phase2" / "real" / "default" / "seed_11.parquet"
    prediction_frame = read_table(prediction_path)
    checkpoint_path = Path(str(prediction_frame["checkpoint_path"].iloc[0]))
    config_path = checkpoint_path.parent / "config.yaml"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing phase2 checkpoint for orthogonal scoring: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing phase2 config for orthogonal scoring: {config_path}")
    return config_path, checkpoint_path


def _screen_registry(project_root: Path) -> pd.DataFrame:
    screen_path = project_root / "data_raw" / "encode" / "screen" / "GRCh38-cCREs.bed"
    frame = pd.read_csv(
        screen_path,
        sep="\t",
        header=None,
        usecols=[0, 1, 2, 3, 4, 5],
        names=["chromosome", "region_start", "region_end", "screen_id", "encode_id", "element_type"],
    )
    frame["state_label"] = (
        frame["chromosome"].astype(str)
        + ":"
        + frame["region_start"].astype(int).astype(str)
        + "-"
        + frame["region_end"].astype(int).astype(str)
    )
    return frame.drop_duplicates("state_label", keep="first").reset_index(drop=True)


def _parse_state_coordinates(state_label: str) -> tuple[str, int, int]:
    chrom, coords = str(state_label).split(":", 1)
    start, end = coords.split("-", 1)
    return chrom, int(start), int(end)


def _safe_probability(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    return float(values.mean())


def _binary_summary(frame: pd.DataFrame, mask: pd.Series, label_col: str) -> dict[str, float]:
    subset = frame.loc[mask].copy()
    if subset.empty:
        return {
            "n": 0.0,
            "positive_n": 0.0,
            "negative_n": 0.0,
            "positive_rate": float("nan"),
        }
    labels = subset[label_col].astype(float)
    return {
        "n": float(len(subset)),
        "positive_n": float((labels > 0.5).sum()),
        "negative_n": float((labels <= 0.5).sum()),
        "positive_rate": float((labels > 0.5).mean()),
    }


def _fisher_payload(frame: pd.DataFrame, row_mask: pd.Series, col_mask: pd.Series) -> dict[str, float]:
    a = int((row_mask & col_mask).sum())
    b = int((row_mask & ~col_mask).sum())
    c = int((~row_mask & col_mask).sum())
    d = int((~row_mask & ~col_mask).sum())
    odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
    return {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "d": float(d),
        "odds_ratio": float(odds_ratio) if math.isfinite(odds_ratio) else float("inf"),
        "p_value": float(p_value),
    }


def _score_phase2_bundle(
    config: dict,
    checkpoint_path: Path,
    bundle_path: Path,
    metadata_path: Path,
) -> pd.DataFrame:
    model, device = load_model_for_inference(config, checkpoint_path)
    loader = make_loader(bundle_path, batch_size=int(config["training"].get("batch_size", 128)), shuffle=False)
    metadata = read_table(metadata_path).reset_index(drop=True)
    rows: list[dict[str, float | str]] = []
    index = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = _forward_model(model, batch)
            full_score, _availability_only, additive_null = _scores_for_phase("phase2", outputs)
            biochemical_probability = torch.sigmoid(-outputs.biochemical_residual)
            for local_index in range(full_score.shape[0]):
                row = {
                    "bundle_index": float(index),
                    "label": float(batch["labels"][local_index].detach().cpu()),
                    "phase2_pair_score": float(full_score[local_index].detach().cpu()),
                    "phase2_additive_null_score": float(additive_null[local_index].detach().cpu()),
                    "phase2_biochemical_probability": float(biochemical_probability[local_index].detach().cpu()),
                    "phase2_geometry_residual": float(outputs.geometry_residual[local_index].detach().cpu()),
                    "phase2_bridge_residual": float(outputs.bridge_residual[local_index].detach().cpu()),
                    "phase2_biochemical_residual": float(outputs.biochemical_residual[local_index].detach().cpu()),
                }
                if outputs.monomer_free_energy is not None:
                    row["phase2_monomer_free_energy"] = float(outputs.monomer_free_energy[local_index].detach().cpu())
                if index < len(metadata):
                    row.update(metadata.iloc[index].to_dict())
                rows.append(row)
                index += 1
    return pd.DataFrame.from_records(rows)


def _score_public_ccre_phase2(project_root: Path, output_root: Path) -> dict[str, Path]:
    return _score_public_ccre_phase2_for_scenario(project_root, "real", "orthogonal_phase4_all", output_root)


def _score_public_ccre_phase2_for_scenario(
    project_root: Path,
    scenario_name: str,
    split_name: str,
    output_root: Path,
) -> dict[str, Path]:
    config_path, checkpoint_path = _phase2_reference_paths(project_root)
    config = load_config(config_path)
    bundle_root = project_root / "data_processed" / scenario_name / "phase4" / split_name
    partitions = []
    for split_partition in ("train", "valid", "test"):
        bundle_path = bundle_root / f"{split_partition}_bundle.pt"
        metadata_path = bundle_root / f"{split_partition}_metadata.parquet"
        if not bundle_path.exists() or not metadata_path.exists():
            continue
        scored = _score_phase2_bundle(config, checkpoint_path, bundle_path, metadata_path)
        scored["bundle_partition"] = split_partition
        partitions.append(scored)
    if not partitions:
        raise FileNotFoundError(f"No orthogonal phase4 bundles found under {bundle_root}")

    scored = pd.concat(partitions, ignore_index=True, sort=False)
    registry = _screen_registry(project_root)
    scored = scored.merge(registry, on="state_label", how="left")
    scored["ccre_group"] = np.where(scored["element_type"].astype(str).isin(ENHANCER_LIKE_TYPES), "enhancer_like", "other")
    scored["phase2_state_rank"] = (
        scored.groupby("state_label")["phase2_biochemical_probability"].rank(method="first", ascending=False).astype(int)
    )
    scored["phase2_state_top"] = scored["phase2_state_rank"].eq(1)
    scored["phase2_state_margin"] = scored.groupby("state_label")["phase2_biochemical_probability"].transform(
        lambda values: float(values.max() - values.min()) if len(values) else float("nan")
    )
    scored["state_level_correct"] = scored["phase2_state_top"] & scored["label"].astype(float).gt(0.5)

    winners = scored.loc[scored["phase2_state_top"]].copy()
    winners["high_confidence_winner"] = winners["phase2_biochemical_probability"].ge(
        winners["phase2_biochemical_probability"].quantile(0.75)
    )
    state_accuracy = float(winners["label"].astype(float).gt(0.5).mean()) if not winners.empty else float("nan")
    element_types = set(winners["element_type"].dropna().astype(str))
    if len(set(winners["ccre_group"].astype(str))) >= 2:
        contrast_mask = winners["ccre_group"].astype(str).eq("enhancer_like")
        contrast_name = "enhancer_like"
        contrast_metric = "high_confidence_winner_enhancer_like_or"
    elif "pELS" in element_types and len(element_types) >= 2:
        contrast_mask = winners["element_type"].astype(str).eq("pELS")
        contrast_name = "pELS"
        contrast_metric = "high_confidence_winner_pels_or"
    elif len(element_types) >= 2:
        dominant_type = winners["element_type"].astype(str).value_counts().index[0]
        contrast_mask = winners["element_type"].astype(str).eq(dominant_type)
        contrast_name = str(dominant_type)
        contrast_metric = f"high_confidence_winner_{str(dominant_type).lower()}_or"
    else:
        contrast_mask = pd.Series(False, index=winners.index)
        contrast_name = "single_class_only"
        contrast_metric = "high_confidence_winner_single_class_only"

    overall_fisher = _fisher_payload(
        winners,
        winners["high_confidence_winner"].astype(bool),
        contrast_mask,
    )

    per_class = (
        winners.groupby("element_type", dropna=False)
        .agg(
            n_states=("state_label", "nunique"),
            mean_phase2_biochemical_probability=("phase2_biochemical_probability", "mean"),
            median_phase2_biochemical_probability=("phase2_biochemical_probability", "median"),
            mean_state_margin=("phase2_state_margin", "mean"),
            correct_rate=("label", lambda values: float(pd.Series(values).astype(float).gt(0.5).mean())),
        )
        .reset_index()
        .sort_values(["mean_phase2_biochemical_probability", "n_states"], ascending=[False, False])
    )

    summary = pd.DataFrame.from_records(
        [
            {
                "metric": "state_winner_accuracy",
                "value": state_accuracy,
                "n_states": float(winners["state_label"].nunique()),
            },
            {
                "metric": contrast_metric,
                "value": overall_fisher["odds_ratio"],
                "p_value": overall_fisher["p_value"],
                "contrast_class": contrast_name,
                "a": overall_fisher["a"],
                "b": overall_fisher["b"],
                "c": overall_fisher["c"],
                "d": overall_fisher["d"],
            },
        ]
    )

    all_rows_summary = pd.DataFrame.from_records(
        [
            {
                "metric": "all_rows_mean_phase2_biochemical_probability",
                "rows": float(len(scored)),
                "pels_mean": float(scored.loc[scored["element_type"].astype(str) == "pELS", "phase2_biochemical_probability"].mean()),
                "dels_mean": float(scored.loc[scored["element_type"].astype(str) == "dELS", "phase2_biochemical_probability"].mean()),
            }
        ]
    )

    scored_path = output_root / "public_ccre_phase2_scored.parquet"
    winners_path = output_root / "public_ccre_phase2_state_winners.csv"
    per_class_path = output_root / "ccre_enrichment_by_class.csv"
    summary_path = output_root / "ccre_enrichment_summary.csv"
    all_rows_summary_path = output_root / "ccre_all_rows_summary.csv"
    write_table(scored, scored_path)
    write_table(winners, winners_path)
    write_table(per_class, per_class_path)
    write_table(summary, summary_path)
    write_table(all_rows_summary, all_rows_summary_path)

    summary_lines = [
        "# cCRE Enrichment Summary",
        "",
        f"- Phase2 checkpoint: `{checkpoint_path}`",
        f"- Scored public phase4 candidate rows: `{len(scored)}`",
        f"- Public states: `{winners['state_label'].nunique()}`",
        f"- State winner accuracy: `{state_accuracy:.3f}`",
        f"- High-confidence winner `{contrast_name}` enrichment OR: `{overall_fisher['odds_ratio']:.3f}`",
        f"- Fisher exact one-sided p-value: `{overall_fisher['p_value']:.4g}`",
    ]
    write_text(output_root / "ccre_enrichment_summary.md", "\n".join(summary_lines))
    return {
        "scored_path": scored_path,
        "winners_path": winners_path,
        "per_class_path": per_class_path,
        "summary_path": summary_path,
        "all_rows_summary_path": all_rows_summary_path,
    }


def summarize_ccre_state_blocked_effect(
    scored_path: str | Path,
    output_root: str | Path,
    positive_class: str = "pELS",
    negative_class: str = "dELS",
    n_bootstrap: int = 10000,
    n_permutations: int = 10000,
    seed: int = 11,
) -> dict[str, str]:
    scored = read_table(scored_path)
    output_root = ensure_dir(output_root)
    required_columns = {"state_label", "element_type", "phase2_biochemical_probability"}
    missing = required_columns - set(scored.columns)
    if missing:
        raise ValueError(f"Cannot summarize cCRE state-blocked effect; missing columns: {sorted(missing)}")

    state_frame = (
        scored.groupby("state_label", as_index=False)
        .agg(
            element_type=("element_type", lambda values: str(pd.Series(values).dropna().astype(str).iloc[0])),
            chromosome=("chromosome", lambda values: str(pd.Series(values).dropna().astype(str).iloc[0]))
            if "chromosome" in scored.columns
            else ("state_label", lambda values: ""),
            n_rows=("phase2_biochemical_probability", "size"),
            mean_phase2_biochemical_probability=("phase2_biochemical_probability", "mean"),
            median_phase2_biochemical_probability=("phase2_biochemical_probability", "median"),
        )
        .sort_values(["element_type", "state_label"])
        .reset_index(drop=True)
    )
    positive_values = state_frame.loc[
        state_frame["element_type"].astype(str).eq(positive_class),
        "mean_phase2_biochemical_probability",
    ].astype(float)
    negative_values = state_frame.loc[
        state_frame["element_type"].astype(str).eq(negative_class),
        "mean_phase2_biochemical_probability",
    ].astype(float)
    if positive_values.empty or negative_values.empty:
        raise ValueError(
            f"Cannot summarize cCRE state-blocked effect; need both {positive_class} and {negative_class} states."
        )

    positive_array = positive_values.to_numpy(dtype=float)
    negative_array = negative_values.to_numpy(dtype=float)
    observed_diff = float(positive_array.mean() - negative_array.mean())
    observed_ratio = float(positive_array.mean() / negative_array.mean()) if float(negative_array.mean()) != 0.0 else float("inf")
    mw_p_value = float(mannwhitneyu(positive_array, negative_array, alternative="greater").pvalue)

    rng = np.random.default_rng(seed)
    bootstrap_diffs = np.empty(int(n_bootstrap), dtype=float)
    for index in range(int(n_bootstrap)):
        positive_sample = rng.choice(positive_array, size=len(positive_array), replace=True)
        negative_sample = rng.choice(negative_array, size=len(negative_array), replace=True)
        bootstrap_diffs[index] = float(positive_sample.mean() - negative_sample.mean())
    ci_low, ci_high = np.quantile(bootstrap_diffs, [0.025, 0.975])

    combined = np.concatenate([positive_array, negative_array])
    positive_n = len(positive_array)
    permutation_diffs = np.empty(int(n_permutations), dtype=float)
    for index in range(int(n_permutations)):
        shuffled = rng.permutation(combined)
        permutation_diffs[index] = float(shuffled[:positive_n].mean() - shuffled[positive_n:].mean())
    permutation_p_value = float((np.sum(permutation_diffs >= observed_diff) + 1.0) / (len(permutation_diffs) + 1.0))
    jackknife_rows: list[dict[str, float | str]] = []
    if "chromosome" in state_frame.columns and state_frame["chromosome"].astype(str).ne("").any():
        for chromosome in sorted(state_frame["chromosome"].dropna().astype(str).unique()):
            retained = state_frame.loc[~state_frame["chromosome"].astype(str).eq(chromosome)].copy()
            retained_positive = retained.loc[
                retained["element_type"].astype(str).eq(positive_class),
                "mean_phase2_biochemical_probability",
            ].astype(float)
            retained_negative = retained.loc[
                retained["element_type"].astype(str).eq(negative_class),
                "mean_phase2_biochemical_probability",
            ].astype(float)
            if retained_positive.empty or retained_negative.empty:
                continue
            retained_effect = float(retained_positive.mean() - retained_negative.mean())
            jackknife_rows.append(
                {
                    "left_out_chromosome": chromosome,
                    "positive_states": float(len(retained_positive)),
                    "negative_states": float(len(retained_negative)),
                    "positive_mean": float(retained_positive.mean()),
                    "negative_mean": float(retained_negative.mean()),
                    "effect": retained_effect,
                }
            )
    jackknife = pd.DataFrame.from_records(
        jackknife_rows,
        columns=[
            "left_out_chromosome",
            "positive_states",
            "negative_states",
            "positive_mean",
            "negative_mean",
            "effect",
        ],
    )
    jackknife_min_effect = float(jackknife["effect"].min()) if not jackknife.empty else float("nan")
    jackknife_positive_fraction = float(jackknife["effect"].gt(0.0).mean()) if not jackknife.empty else float("nan")

    summary = pd.DataFrame.from_records(
        [
            {
                "metric": "state_blocked_mean_difference",
                "positive_class": positive_class,
                "negative_class": negative_class,
                "positive_states": float(len(positive_array)),
                "negative_states": float(len(negative_array)),
                "positive_mean": float(positive_array.mean()),
                "negative_mean": float(negative_array.mean()),
                "effect": observed_diff,
                "effect_ratio": observed_ratio,
                "bootstrap_ci_low": float(ci_low),
                "bootstrap_ci_high": float(ci_high),
                "bootstrap_iterations": float(n_bootstrap),
                "permutation_p_value_greater": permutation_p_value,
                "permutation_iterations": float(n_permutations),
                "state_level_mannwhitney_p_value_greater": mw_p_value,
                "chromosome_jackknife_min_effect": jackknife_min_effect,
                "chromosome_jackknife_positive_fraction": jackknife_positive_fraction,
                "seed": float(seed),
            }
        ]
    )
    state_summary_path = output_root / "ccre_state_level_summary.csv"
    robustness_path = output_root / "ccre_state_blocked_robustness.csv"
    jackknife_path = output_root / "ccre_chromosome_jackknife.csv"
    robustness_md_path = output_root / "ccre_state_blocked_robustness.md"
    write_table(state_frame, state_summary_path)
    write_table(summary, robustness_path)
    write_table(jackknife, jackknife_path)
    lines = [
        "# cCRE State-Blocked Robustness",
        "",
        f"- Comparison: `{positive_class}` vs `{negative_class}`",
        f"- State-level N: `{len(positive_array)}` vs `{len(negative_array)}`",
        f"- State-level mean Phase2 biochemical probability: `{positive_array.mean():.4f}` vs `{negative_array.mean():.4f}`",
        f"- Mean difference: `{observed_diff:.4f}`",
        f"- Bootstrap 95% CI: `[{float(ci_low):.4f}, {float(ci_high):.4f}]`",
        f"- One-sided state-label permutation p-value: `{permutation_p_value:.4g}`",
        f"- State-level Mann-Whitney one-sided p-value: `{mw_p_value:.4g}`",
        f"- Chromosome jackknife minimum effect: `{jackknife_min_effect:.4f}`",
        f"- Chromosome jackknife positive fraction: `{jackknife_positive_fraction:.3f}`",
    ]
    write_text(robustness_md_path, "\n".join(lines))
    return {
        "state_summary_path": str(state_summary_path),
        "robustness_path": str(robustness_path),
        "jackknife_path": str(jackknife_path),
        "robustness_md_path": str(robustness_md_path),
    }


def _expanded_public_phase4_rows(project_root: Path, max_pairs_per_state: int) -> pd.DataFrame:
    current = read_table(project_root / "data_intermediate" / "real" / "candidate_sequences.parquet")
    state_rows = current.sort_values(["state_label", "phase"]).drop_duplicates("state_label", keep="first").copy()
    state_rows = state_rows[["state_label", "sequence", "state_total_fragment_count", "state_supporting_files", "chromosome"]].copy()

    consensus = _load_consensus_map(project_root)
    reference_pairs = _load_reference_pairs(project_root, consensus)
    reference_pair_set = {tuple(sorted(pair)) for pair in reference_pairs}
    genes = sorted({gene for pair in reference_pairs for gene in pair})
    pwm_map = _load_pwm_map(project_root, genes)

    rows: list[dict[str, object]] = []
    top_k = max(2, int(max_pairs_per_state))
    pair_budget = max(int(max_pairs_per_state), 2)
    for state_index, row in enumerate(state_rows.itertuples(index=False)):
        hit_lists = _find_hit_lists(str(row.sequence), pwm_map, genes, top_k=top_k)
        observed_genes = sorted(hit_lists)
        pair_candidates: list[dict[str, object]] = []
        for left_tf, right_tf in itertools.combinations(observed_genes, 2):
            reference_supported = tuple(sorted((left_tf, right_tf))) in reference_pair_set
            for left_rank, left_hit in enumerate(hit_lists[left_tf]):
                for right_rank, right_hit in enumerate(hit_lists[right_tf]):
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
                            "center_distance": float((right_hit["start"] + right_hit["end"]) / 2 - (left_hit["start"] + left_hit["end"]) / 2),
                            "edge_gap": edge_gap,
                            "overlap_len": overlap_len,
                            "score": width_sum,
                            "reference_supported": reference_supported,
                            "candidate_key": f"{left_tf}::{right_tf}::{left_rank}::{right_rank}",
                            "candidate_priority": (
                                1 if reference_supported else 0,
                                width_sum,
                                -gap_penalty,
                                -overlap_penalty,
                                -anchor_rank_penalty,
                            ),
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
        pair_candidates = pair_candidates[:pair_budget]
        for pair_rank, pair in enumerate(pair_candidates):
            rows.append(
                {
                    "seq_id": f"expanded::phase4::{pair['left_tf']}::{pair['right_tf']}::{state_index:05d}::{pair_rank:02d}",
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
                    "coarse_additive_score": float(pair["score"]),
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
                    "state_total_fragment_count": float(row.state_total_fragment_count),
                    "state_supporting_files": float(row.state_supporting_files),
                    "observed_pair_candidate_count": int(len(pair_candidates)),
                    "pair_candidate_rank": int(pair_rank),
                    "pair_reference_supported": float(pair["reference_supported"]),
                    "pair_candidate_key": str(pair["candidate_key"]),
                    "split_group": str(row.chromosome),
                    "chromosome": str(row.chromosome),
                }
            )
    frame = pd.DataFrame.from_records(rows)
    return _assign_phase4_usage_labels(frame)


def build_expanded_public_phase4_scenario(
    project_root: str | Path,
    scenario_name: str = "orthogonal_public_expanded",
    max_pairs_per_state: int = 5,
) -> dict[str, str]:
    project_root = resolve_path(project_root)
    frame = _expanded_public_phase4_rows(project_root, max_pairs_per_state=max_pairs_per_state)
    scenario_root = ensure_dir(project_root / "data_intermediate" / scenario_name)
    sequences_path = scenario_root / "sequences.parquet"
    write_table(frame, sequences_path)
    build_windows_and_candidates(project_root, scenario=scenario_name, window_length=96, top_k_anchors=24, top_k_pairs=64)
    build_tensor_bundles(
        project_root,
        scenario=scenario_name,
        window_length=96,
        availability_dim=16,
        state_dim=32,
        split_name="default",
        phases=["phase4"],
    )
    manifest = {
        "scenario_name": scenario_name,
        "max_pairs_per_state": int(max_pairs_per_state),
        "num_rows": int(len(frame)),
        "num_states": int(frame["state_label"].astype(str).nunique()),
        "positives": int((frame["label"].astype(float) > 0.5).sum()),
        "negatives": int((frame["label"].astype(float) <= 0.5).sum()),
        "sequences_path": str(sequences_path),
    }
    write_json(project_root / "reports" / f"{scenario_name}_manifest.json", manifest)
    return manifest


def build_public_panel_orthogonal_scenario(
    project_root: str | Path,
    scenario_name: str = "orthogonal_public_panel",
    max_pairs_per_state: int = 4,
    availability_dim: int = 16,
    state_dim: int = 32,
    window_length: int = 96,
    max_fragment_files: int = 4,
    max_candidate_regions: int = 4096,
    max_states: int = 256,
    state_selection_multiplier: int = 12,
    max_examples: int | None = None,
    max_monomers_per_state: int = 1,
    max_fragment_lines: int | None = None,
    impute_missing_monomers: bool = True,
    impute_missing_pairs: bool = True,
    select_best_fragment_file_count: bool = True,
) -> dict[str, str | int | float]:
    project_root = resolve_path(project_root)
    requested_max_fragment_files = max(int(max_fragment_files), 1)
    fragment_file_counts = [requested_max_fragment_files]
    selection_mode = "fixed"
    if select_best_fragment_file_count and requested_max_fragment_files > 1:
        fragment_file_counts = list(range(1, requested_max_fragment_files + 1))
        selection_mode = "best_prefix"

    best_result: tuple[pd.DataFrame, pd.DataFrame, dict[str, object]] | None = None
    best_score: tuple[int, int, int, int, int] | None = None
    best_fragment_file_count: int | None = None
    evaluations: list[dict[str, object]] = []
    for fragment_file_count in fragment_file_counts:
        candidate_sequences, state_features, panel_manifest = _build_public_state_layer_result(
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
        score = _public_state_manifest_score(panel_manifest)
        evaluations.append(
            {
                "fragment_file_count": int(fragment_file_count),
                "fragment_files": list(panel_manifest["fragment_files"]),
                "observed_monomer_rows": int(panel_manifest["observed_monomer_rows"]),
                "observed_pair_rows": int(panel_manifest["observed_pair_rows"]),
                "imputed_monomer_rows": int(panel_manifest["imputed_monomer_rows"]),
                "imputed_pair_rows": int(panel_manifest["imputed_pair_rows"]),
                "num_candidate_states": int(panel_manifest["num_candidate_states"]),
                "num_selected_states": int(panel_manifest["num_selected_states"]),
                "num_candidate_sequences": int(panel_manifest["num_candidate_sequences"]),
            }
        )
        if best_score is None or score > best_score or (score == best_score and fragment_file_count < int(best_fragment_file_count)):
            best_result = (candidate_sequences, state_features, panel_manifest)
            best_score = score
            best_fragment_file_count = int(fragment_file_count)

    if best_result is None or best_fragment_file_count is None:
        raise RuntimeError("Public panel orthogonal builder did not evaluate any fragment-file configurations.")

    candidate_sequences, state_features, panel_manifest = best_result
    phase4_sequences = candidate_sequences[candidate_sequences["phase"].astype(str) == "phase4"].copy()
    if phase4_sequences.empty:
        raise RuntimeError("Public panel orthogonal builder produced no phase4 rows.")
    phase4_sequences = phase4_sequences[phase4_sequences["construction_mode"].astype(str) == "observed_pair"].copy()
    if phase4_sequences.empty:
        raise RuntimeError("Public panel orthogonal builder retained no observed phase4 rows.")

    represented_states = set(phase4_sequences["state_label"].astype(str))
    filtered_states = state_features[state_features["state_label"].astype(str).isin(represented_states)].drop_duplicates("state_label").copy()
    if filtered_states.empty:
        raise RuntimeError("Public panel orthogonal builder retained no state features for represented phase4 rows.")

    merged = phase4_sequences.merge(filtered_states, on="state_label", how="inner")
    if merged.empty:
        raise RuntimeError("Public panel orthogonal scenario lost all rows during state-feature merge.")

    scenario_root = ensure_dir(project_root / "data_intermediate" / scenario_name)
    sequences_path = scenario_root / "sequences.parquet"
    state_features_path = scenario_root / "state_features.parquet"
    write_table(merged, sequences_path)
    write_table(filtered_states, state_features_path)
    build_windows_and_candidates(project_root, scenario=scenario_name, window_length=window_length, top_k_anchors=24, top_k_pairs=64)
    build_tensor_bundles(
        project_root,
        scenario=scenario_name,
        window_length=window_length,
        availability_dim=availability_dim,
        state_dim=state_dim,
        split_name="default",
        phases=["phase4"],
    )

    report_root = ensure_dir(project_root / "reports" / scenario_name)
    manifest = {
        "scenario_name": scenario_name,
        "scenario_mode": "public_panel",
        "phase_scope": "phase4_only",
        "max_pairs_per_state": int(max_pairs_per_state),
        "max_fragment_files": int(requested_max_fragment_files),
        "selected_fragment_file_count": int(best_fragment_file_count),
        "fragment_count_selection_mode": selection_mode,
        "fragment_count_evaluations": evaluations,
        "max_candidate_regions": int(max_candidate_regions),
        "max_states": int(max_states),
        "state_selection_multiplier": int(state_selection_multiplier),
        "num_rows": int(len(merged)),
        "num_states": int(merged["state_label"].astype(str).nunique()),
        "positives": int((merged["label"].astype(float) > 0.5).sum()),
        "negatives": int((merged["label"].astype(float) <= 0.5).sum()),
        "sequences_path": str(sequences_path),
        "state_features_path": str(state_features_path),
        "public_panel_manifest": panel_manifest,
    }
    write_json(report_root / "public_panel_manifest.json", manifest)
    return manifest


def _encode_search(target_gene: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    response = requests.get(
        ENCODE_SEARCH_URL,
        params={
            "type": "File",
            "target.label": target_gene,
            "assembly": "GRCh38",
            "status": "released",
            "file_format": "bed",
            "format": "json",
            "limit": "all",
        },
        headers=ENCODE_HEADERS,
        timeout=180,
    )
    if response.status_code == 404:
        return []
    response.raise_for_status()
    payload = response.json()
    assay_allow = tuple(token.lower() for token in ENCODE_ASSAYS)
    for item in payload.get("@graph", []):
        assay_title = str(item.get("assay_term_name", item.get("assay_title", ""))).strip()
        if not any(token in assay_title.lower() for token in assay_allow):
            continue
        output_type = str(item.get("output_type", "")).lower()
        if not any(token in output_type for token in PEAK_OUTPUT_TOKENS):
            continue
        href = str(item.get("href", "")).strip()
        accession = str(item.get("accession", "")).strip()
        if not href or not accession:
            continue
        records.append(
            {
                "target_gene": target_gene,
                "assay_title": assay_title,
                "accession": accession,
                "href": href,
                "output_type": str(item.get("output_type", "")),
                "biosample": str(item.get("simple_biosample_summary", "")),
                "dataset": str(item.get("dataset", "")),
            }
        )
    deduped = pd.DataFrame.from_records(records)
    if deduped.empty:
        return []
    deduped = deduped.drop_duplicates("accession", keep="first").sort_values(["assay_title", "accession"])
    return deduped.to_dict("records")


def _download_encode_peak(record: dict[str, str], cache_root: Path) -> Path:
    gene_root = ensure_dir(cache_root / record["target_gene"])
    suffix = ".bed.gz" if str(record["href"]).endswith(".bed.gz") else ".bed"
    target_path = gene_root / f"{record['accession']}{suffix}"
    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path
    response = requests.get(f"https://www.encodeproject.org{record['href']}", timeout=300)
    response.raise_for_status()
    target_path.write_bytes(response.content)
    return target_path


def _read_bed_intervals(path: Path) -> list[tuple[str, int, int]]:
    opener = gzip.open if path.suffix == ".gz" else open
    intervals: list[tuple[str, int, int]] = []
    with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 3:
                continue
            chrom = str(parts[0])
            start = int(parts[1])
            end = int(parts[2])
            if end <= start:
                continue
            intervals.append((chrom, start, end))
    return intervals


def _merge_intervals(intervals: list[tuple[str, int, int]]) -> dict[str, list[tuple[int, int]]]:
    merged: dict[str, list[tuple[int, int]]] = {}
    if not intervals:
        return merged
    intervals = sorted(intervals, key=lambda item: (item[0], item[1], item[2]))
    current_chrom = ""
    current_start = -1
    current_end = -1
    for chrom, start, end in intervals:
        if chrom != current_chrom or start > current_end:
            if current_chrom:
                merged.setdefault(current_chrom, []).append((current_start, current_end))
            current_chrom = chrom
            current_start = start
            current_end = end
            continue
        current_end = max(current_end, end)
    if current_chrom:
        merged.setdefault(current_chrom, []).append((current_start, current_end))
    return merged


def _has_overlap(region: tuple[str, int, int], merged_intervals: dict[str, list[tuple[int, int]]]) -> bool:
    chrom, start, end = region
    for interval_start, interval_end in merged_intervals.get(chrom, []):
        if interval_end <= start:
            continue
        if interval_start >= end:
            break
        return True
    return False


def _compute_encode_overlap(project_root: Path, scored_path: Path, output_root: Path) -> dict[str, Path]:
    scored = read_table(scored_path)
    phase4 = scored.copy()
    phase4["region_tuple"] = phase4["state_label"].astype(str).map(_parse_state_coordinates)
    target_genes = sorted(set(phase4["left_tf"].astype(str)) | set(phase4["right_tf"].astype(str)))
    cache_root = ensure_dir(project_root / "data_raw" / "encode" / "orthogonal_tf_peaks")

    search_rows: list[dict[str, str | float]] = []
    merged_by_gene: dict[str, dict[str, list[tuple[int, int]]]] = {}
    for target_gene in target_genes:
        files = _encode_search(target_gene)
        search_rows.extend(files)
        if not files:
            continue
        intervals: list[tuple[str, int, int]] = []
        for record in files:
            peak_path = _download_encode_peak(record, cache_root)
            intervals.extend(_read_bed_intervals(peak_path))
        merged_by_gene[target_gene] = _merge_intervals(intervals)

    overlap_rows = phase4.copy()
    overlap_rows["left_tf_covered"] = overlap_rows["left_tf"].astype(str).isin(merged_by_gene)
    overlap_rows["right_tf_covered"] = overlap_rows["right_tf"].astype(str).isin(merged_by_gene)
    overlap_rows["left_tf_overlap"] = overlap_rows.apply(
        lambda row: _has_overlap(row["region_tuple"], merged_by_gene.get(str(row["left_tf"]), {})),
        axis=1,
    )
    overlap_rows["right_tf_overlap"] = overlap_rows.apply(
        lambda row: _has_overlap(row["region_tuple"], merged_by_gene.get(str(row["right_tf"]), {})),
        axis=1,
    )
    overlap_rows["num_covered_tfs"] = overlap_rows["left_tf_covered"].astype(int) + overlap_rows["right_tf_covered"].astype(int)
    overlap_rows["any_tf_overlap"] = overlap_rows["left_tf_overlap"] | overlap_rows["right_tf_overlap"]
    overlap_rows["both_tfs_overlap"] = overlap_rows["left_tf_overlap"] & overlap_rows["right_tf_overlap"]

    any_covered = overlap_rows["num_covered_tfs"].ge(1)
    both_covered = overlap_rows["num_covered_tfs"].ge(2)
    positive = overlap_rows["label"].astype(float).gt(0.5)
    any_payload = _fisher_payload(overlap_rows.loc[any_covered], positive.loc[any_covered], overlap_rows.loc[any_covered, "any_tf_overlap"])
    both_payload = _fisher_payload(overlap_rows.loc[both_covered], positive.loc[both_covered], overlap_rows.loc[both_covered, "both_tfs_overlap"])

    summary = pd.DataFrame.from_records(
        [
            {
                "metric": "any_tf_overlap_enrichment_positive_vs_negative",
                "n_rows": float(any_covered.sum()),
                "odds_ratio": any_payload["odds_ratio"],
                "p_value": any_payload["p_value"],
                "positive_overlap_rate": _safe_probability(overlap_rows.loc[any_covered & positive, "any_tf_overlap"].astype(float)),
                "negative_overlap_rate": _safe_probability(overlap_rows.loc[any_covered & ~positive, "any_tf_overlap"].astype(float)),
            },
            {
                "metric": "both_tf_overlap_enrichment_positive_vs_negative",
                "n_rows": float(both_covered.sum()),
                "odds_ratio": both_payload["odds_ratio"],
                "p_value": both_payload["p_value"],
                "positive_overlap_rate": _safe_probability(overlap_rows.loc[both_covered & positive, "both_tfs_overlap"].astype(float)),
                "negative_overlap_rate": _safe_probability(overlap_rows.loc[both_covered & ~positive, "both_tfs_overlap"].astype(float)),
            },
        ]
    )

    search_manifest = pd.DataFrame.from_records(search_rows)
    if search_manifest.empty:
        search_manifest = pd.DataFrame(
            columns=["target_gene", "assay_title", "accession", "href", "output_type", "biosample", "dataset"]
        )
        target_manifest = pd.DataFrame(columns=["target_gene", "file_count", "assay_count"])
    else:
        target_manifest = (
            search_manifest.groupby("target_gene", dropna=False)
            .agg(
                file_count=("accession", "nunique"),
                assay_count=("assay_title", "nunique"),
            )
            .reset_index()
            .sort_values(["file_count", "target_gene"], ascending=[False, True])
        )

    search_manifest_path = output_root / "encode_peak_search_manifest.csv"
    overlap_path = output_root / "encode_overlap_candidates.parquet"
    summary_path = output_root / "encode_overlap_summary.csv"
    target_path = output_root / "encode_target_coverage.csv"
    write_table(search_manifest, search_manifest_path)
    write_table(overlap_rows.drop(columns="region_tuple"), overlap_path)
    write_table(summary, summary_path)
    write_table(target_manifest, target_path)

    summary_lines = [
        "# ENCODE Overlap Summary",
        "",
        f"- Queried target genes: `{len(target_genes)}`",
        f"- ENCODE-covered genes: `{len(merged_by_gene)}`",
        f"- Candidate rows with at least one covered TF: `{int(any_covered.sum())}`",
        f"- Candidate rows with both TFs covered: `{int(both_covered.sum())}`",
        f"- Positive vs negative any-TF overlap OR: `{any_payload['odds_ratio']:.3f}` (p=`{any_payload['p_value']:.4g}`)",
        f"- Positive vs negative both-TF overlap OR: `{both_payload['odds_ratio']:.3f}` (p=`{both_payload['p_value']:.4g}`)",
    ]
    write_text(output_root / "encode_overlap_summary.md", "\n".join(summary_lines))
    return {
        "search_manifest_path": search_manifest_path,
        "overlap_path": overlap_path,
        "summary_path": summary_path,
        "target_path": target_path,
    }


def run_orthogonal_validation(project_root: str | Path) -> dict[str, str]:
    project_root = resolve_path(project_root)
    output_root = ensure_dir(project_root / "reports" / "orthogonal_validation")
    ccre_outputs = _score_public_ccre_phase2(project_root, output_root)
    robustness_outputs = summarize_ccre_state_blocked_effect(ccre_outputs["scored_path"], output_root)
    encode_outputs = _compute_encode_overlap(project_root, ccre_outputs["scored_path"], output_root)
    manifest = {
        "ccre_scored": str(ccre_outputs["scored_path"]),
        "ccre_winners": str(ccre_outputs["winners_path"]),
        "ccre_by_class": str(ccre_outputs["per_class_path"]),
        "ccre_summary": str(ccre_outputs["summary_path"]),
        "ccre_state_summary": str(robustness_outputs["state_summary_path"]),
        "ccre_state_blocked_robustness": str(robustness_outputs["robustness_path"]),
        "ccre_chromosome_jackknife": str(robustness_outputs["jackknife_path"]),
        "encode_manifest": str(encode_outputs["search_manifest_path"]),
        "encode_overlap": str(encode_outputs["overlap_path"]),
        "encode_summary": str(encode_outputs["summary_path"]),
        "encode_targets": str(encode_outputs["target_path"]),
    }
    write_json(output_root / "orthogonal_validation_manifest.json", manifest)
    return manifest


def run_expanded_orthogonal_validation(
    project_root: str | Path,
    scenario_name: str = "orthogonal_public_expanded",
    max_pairs_per_state: int = 5,
) -> dict[str, str]:
    project_root = resolve_path(project_root)
    build_expanded_public_phase4_scenario(project_root, scenario_name=scenario_name, max_pairs_per_state=max_pairs_per_state)
    output_root = ensure_dir(project_root / "reports" / scenario_name)
    ccre_outputs = _score_public_ccre_phase2_for_scenario(project_root, scenario_name, "default", output_root)
    robustness_outputs = summarize_ccre_state_blocked_effect(ccre_outputs["scored_path"], output_root)
    encode_outputs = _compute_encode_overlap(project_root, ccre_outputs["scored_path"], output_root)
    manifest = {
        "scenario_name": scenario_name,
        "max_pairs_per_state": int(max_pairs_per_state),
        "ccre_scored": str(ccre_outputs["scored_path"]),
        "ccre_winners": str(ccre_outputs["winners_path"]),
        "ccre_by_class": str(ccre_outputs["per_class_path"]),
        "ccre_summary": str(ccre_outputs["summary_path"]),
        "ccre_all_rows_summary": str(ccre_outputs["all_rows_summary_path"]),
        "ccre_state_summary": str(robustness_outputs["state_summary_path"]),
        "ccre_state_blocked_robustness": str(robustness_outputs["robustness_path"]),
        "ccre_chromosome_jackknife": str(robustness_outputs["jackknife_path"]),
        "encode_manifest": str(encode_outputs["search_manifest_path"]),
        "encode_overlap": str(encode_outputs["overlap_path"]),
        "encode_summary": str(encode_outputs["summary_path"]),
        "encode_targets": str(encode_outputs["target_path"]),
    }
    write_json(output_root / "orthogonal_validation_manifest.json", manifest)
    return manifest


def run_public_panel_orthogonal_validation(
    project_root: str | Path,
    scenario_name: str = "orthogonal_public_panel",
    max_pairs_per_state: int = 4,
    availability_dim: int = 16,
    state_dim: int = 32,
    window_length: int = 96,
    max_fragment_files: int = 4,
    max_candidate_regions: int = 4096,
    max_states: int = 256,
    state_selection_multiplier: int = 12,
    max_examples: int | None = None,
    max_monomers_per_state: int = 1,
    max_fragment_lines: int | None = None,
    impute_missing_monomers: bool = True,
    impute_missing_pairs: bool = True,
    select_best_fragment_file_count: bool = True,
) -> dict[str, str | int | float]:
    project_root = resolve_path(project_root)
    panel_manifest = build_public_panel_orthogonal_scenario(
        project_root,
        scenario_name=scenario_name,
        max_pairs_per_state=max_pairs_per_state,
        availability_dim=availability_dim,
        state_dim=state_dim,
        window_length=window_length,
        max_fragment_files=max_fragment_files,
        max_candidate_regions=max_candidate_regions,
        max_states=max_states,
        state_selection_multiplier=state_selection_multiplier,
        max_examples=max_examples,
        max_monomers_per_state=max_monomers_per_state,
        max_fragment_lines=max_fragment_lines,
        impute_missing_monomers=impute_missing_monomers,
        impute_missing_pairs=impute_missing_pairs,
        select_best_fragment_file_count=select_best_fragment_file_count,
    )
    output_root = ensure_dir(project_root / "reports" / scenario_name)
    ccre_outputs = _score_public_ccre_phase2_for_scenario(project_root, scenario_name, "default", output_root)
    robustness_outputs = summarize_ccre_state_blocked_effect(ccre_outputs["scored_path"], output_root)
    encode_outputs = _compute_encode_overlap(project_root, ccre_outputs["scored_path"], output_root)
    manifest = {
        "scenario_name": scenario_name,
        "scenario_mode": "public_panel",
        "max_pairs_per_state": int(max_pairs_per_state),
        "max_fragment_files": int(max_fragment_files),
        "max_candidate_regions": int(max_candidate_regions),
        "max_states": int(max_states),
        "state_selection_multiplier": int(state_selection_multiplier),
        "panel_manifest_path": str(output_root / "public_panel_manifest.json"),
        "panel_rows": int(panel_manifest["num_rows"]),
        "panel_states": int(panel_manifest["num_states"]),
        "ccre_scored": str(ccre_outputs["scored_path"]),
        "ccre_winners": str(ccre_outputs["winners_path"]),
        "ccre_by_class": str(ccre_outputs["per_class_path"]),
        "ccre_summary": str(ccre_outputs["summary_path"]),
        "ccre_all_rows_summary": str(ccre_outputs["all_rows_summary_path"]),
        "ccre_state_summary": str(robustness_outputs["state_summary_path"]),
        "ccre_state_blocked_robustness": str(robustness_outputs["robustness_path"]),
        "ccre_chromosome_jackknife": str(robustness_outputs["jackknife_path"]),
        "encode_manifest": str(encode_outputs["search_manifest_path"]),
        "encode_overlap": str(encode_outputs["overlap_path"]),
        "encode_summary": str(encode_outputs["summary_path"]),
        "encode_targets": str(encode_outputs["target_path"]),
    }
    write_json(output_root / "orthogonal_validation_manifest.json", manifest)
    return manifest
