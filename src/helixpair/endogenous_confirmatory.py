from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from helixpair.bundles import build_tensor_bundles, build_windows_and_candidates
from helixpair.config import load_config
from helixpair.external_functional import hydrate_region_sequences
from helixpair.external_functional_benchmarks import build_mpra_phase5_candidate_rows
from helixpair.inference import predict_bundle
from helixpair.io_utils import ensure_dir, read_table, resolve_path, write_json, write_table
from helixpair.public_state import _load_consensus_map, _load_pwm_map, _load_reference_pairs


DEFAULT_BENCHMARK_SCENARIO = "external_functional_mpra_benchmark"
DEFAULT_BENCHMARK_SPLIT = "default"
CONFIRMATORY_SCENARIO = "external_functional_endogenous_confirmatory"
OUTPUT_BASENAME = "endogenous_confirmatory"


def _parse_effect_payload(value: str) -> tuple[float, float]:
    tokens = str(value).strip().rsplit("_", 2)
    if len(tokens) != 3:
        return float("nan"), float("nan")
    effect = pd.to_numeric(tokens[1], errors="coerce")
    p_value = pd.to_numeric(tokens[2], errors="coerce")
    return float(effect) if pd.notna(effect) else float("nan"), float(p_value) if pd.notna(p_value) else float("nan")


def _observed_direction(effect_size: float) -> str:
    if not math.isfinite(float(effect_size)):
        return ""
    if float(effect_size) < 0.0:
        return "activating"
    if float(effect_size) > 0.0:
        return "repressive"
    return ""


def _candidate_split_manifest(seq_ids: list[str], *, split_name: str) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for index, seq_id in enumerate(seq_ids):
        partition = "test"
        if index == 0:
            partition = "train"
        elif index == 1:
            partition = "valid"
        rows.append(
            {
                "seq_id": str(seq_id),
                "split_name": split_name,
                "split": partition,
                "group_value": str(seq_id),
            }
        )
    return pd.DataFrame.from_records(rows)


def _checkpoint_and_config_from_seed(
    project_root: Path,
    *,
    scenario: str,
    split_name: str,
    seed: int,
) -> tuple[Path, Path]:
    prediction_path = (
        project_root
        / "results"
        / "per_example_predictions"
        / "phase5"
        / scenario
        / split_name
        / f"seed_{seed}.parquet"
    )
    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing phase5 prediction file for seed {seed}: {prediction_path}")
    prediction_frame = read_table(prediction_path)
    if prediction_frame.empty or "checkpoint_path" not in prediction_frame.columns:
        raise ValueError(f"Prediction file lacks checkpoint metadata: {prediction_path}")
    checkpoint_path = Path(str(prediction_frame["checkpoint_path"].iloc[0]))
    config_path = checkpoint_path.parent / "config.yaml"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint resolved from predictions: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config adjacent to checkpoint: {config_path}")
    return checkpoint_path, config_path


def _score_partitions(
    config_path: Path,
    checkpoint_path: Path,
    bundle_root: Path,
) -> pd.DataFrame:
    config = load_config(config_path)
    frames: list[pd.DataFrame] = []
    for partition in ("train", "valid", "test"):
        bundle_path = bundle_root / f"{partition}_bundle.pt"
        metadata_path = bundle_root / f"{partition}_metadata.parquet"
        if not bundle_path.exists() or not metadata_path.exists():
            continue
        local = predict_bundle(config, checkpoint_path, bundle_path, metadata_path)
        local["bundle_partition"] = partition
        frames.append(local)
    if not frames:
        raise FileNotFoundError(f"No confirmatory bundles found under {bundle_root}")
    return pd.concat(frames, ignore_index=True, sort=False)


def _direction_auc(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {"direction_auprc": float("nan"), "direction_auroc": float("nan")}
    labels = frame["observed_direction"].astype(str).eq("activating").astype(float)
    if labels.nunique() < 2:
        return {"direction_auprc": float("nan"), "direction_auroc": float("nan")}
    scores = pd.to_numeric(frame["full_score_mean"], errors="coerce").astype(float)
    return {
        "direction_auprc": float(average_precision_score(labels, scores)),
        "direction_auroc": float(roc_auc_score(labels, scores)),
    }


def run_endogenous_confirmatory_evaluation(
    project_root: str | Path,
    *,
    benchmark_scenario: str = DEFAULT_BENCHMARK_SCENARIO,
    benchmark_split: str = DEFAULT_BENCHMARK_SPLIT,
    seeds: tuple[int, ...] = (11, 17, 23, 29, 47),
    score_threshold: float = 0.5,
    window_length: int = 96,
    top_k_hits: int = 2,
) -> dict[str, Any]:
    project_root = resolve_path(project_root)
    report_root = ensure_dir(project_root / "reports" / benchmark_scenario)
    output_root = ensure_dir(report_root / OUTPUT_BASENAME)
    registry_path = report_root / "endogenous_priority_registry.csv"
    state_features_path = project_root / "data_intermediate" / benchmark_scenario / "state_features.parquet"
    if not registry_path.exists():
        raise FileNotFoundError(f"Missing priority registry: {registry_path}")
    if not state_features_path.exists():
        raise FileNotFoundError(f"Missing benchmark state features: {state_features_path}")

    registry = pd.read_csv(registry_path).copy()
    state_features = read_table(state_features_path).copy()
    supported_state_labels = set(state_features["state_label"].astype(str).tolist())

    effects = registry["perturbation_descriptor"].astype(str).map(_parse_effect_payload)
    registry["effect_size"] = effects.map(lambda item: item[0])
    registry["effect_p_value"] = effects.map(lambda item: item[1])
    registry["observed_direction"] = registry["effect_size"].map(_observed_direction)
    registry["direction_evaluable"] = registry["observed_direction"].astype(str).ne("")
    registry["direction_significant"] = registry["effect_p_value"].fillna(1.0).astype(float).le(0.05)
    registry["supported_state_label"] = registry["canonical_biosample"].astype(str).isin(supported_state_labels)
    registry["state_label_model"] = registry["canonical_biosample"].astype(str)
    registry["seq_id"] = "endogenous_confirmatory::" + registry["registry_id"].astype(str)

    scoring_frame = registry.loc[registry["direction_evaluable"] & registry["supported_state_label"]].copy()
    scoring_frame["biosample_label"] = scoring_frame["canonical_biosample"].astype(str)
    scoring_frame["state_label"] = scoring_frame["canonical_biosample"].astype(str)
    scoring_frame["label"] = 0.0
    scoring_frame["functional_label"] = 0.0
    scoring_frame["usage_label"] = 0.0
    scoring_frame["phase"] = "phase5"
    scoring_frame["split_group"] = scoring_frame["enhancer_id"].astype(str)
    scoring_frame["composite_label"] = 1.0
    scoring_frame["element_group"] = scoring_frame["enhancer_id"].astype(str)
    scoring_frame["priority_rank_within_biosample"] = (
        scoring_frame.groupby("canonical_biosample").cumcount().astype(int) + 1
    )
    scoring_frame["mean_log2_activity"] = float("nan")
    scoring_frame["std_log2_activity"] = float("nan")
    scoring_frame["orientation_gap_log2"] = float("nan")

    hydrated = pd.DataFrame()
    candidate_rows = pd.DataFrame()
    status_frame = pd.DataFrame(columns=["seq_id", "has_candidate", "pair_mode", "pair_reference_supported"])
    candidate_manifest: dict[str, Any] = {
        "hydrated_rows": 0,
        "pending_rows": 0,
        "processed_rows": 0,
        "candidate_rows": 0,
        "rows_without_candidate_pair": 0,
        "pair_mode_counts": {},
        "reference_supported_rows": 0,
    }
    supported_direction_rows = int(len(scoring_frame))
    if not scoring_frame.empty:
        hydrated = hydrate_region_sequences(scoring_frame, window_length=int(window_length))
        write_table(hydrated, output_root / "hydrated_direction_evaluable_rows.parquet")

        consensus_map = _load_consensus_map(project_root)
        reference_pairs = _load_reference_pairs(project_root, consensus_map)
        candidate_genes = sorted({gene for pair in reference_pairs for gene in pair})
        pwm_map = _load_pwm_map(project_root, candidate_genes)
        reference_pair_set = {tuple(sorted(pair)) for pair in reference_pairs}
        candidate_rows, candidate_manifest, status_frame = build_mpra_phase5_candidate_rows(
            hydrated,
            reference_pair_set=reference_pair_set,
            pwm_map=pwm_map,
            candidate_genes=[gene for gene in candidate_genes if gene in pwm_map],
            top_k_hits=int(top_k_hits),
        )
        write_table(status_frame, output_root / "candidate_status.csv")
        if not candidate_rows.empty:
            confirmatory_state_features = state_features.loc[
                state_features["state_label"].astype(str).isin(candidate_rows["state_label"].astype(str))
            ].copy()
            scenario_root = ensure_dir(project_root / "data_intermediate" / CONFIRMATORY_SCENARIO)
            split_root = ensure_dir(project_root / "splits" / CONFIRMATORY_SCENARIO)
            split_name = "confirmatory"
            write_table(candidate_rows, scenario_root / "sequences.parquet")
            write_table(confirmatory_state_features, scenario_root / "state_features.parquet")
            write_table(_candidate_split_manifest(candidate_rows["seq_id"].astype(str).tolist(), split_name=split_name), split_root / f"{split_name}.parquet")
            build_windows_and_candidates(
                project_root,
                scenario=CONFIRMATORY_SCENARIO,
                window_length=int(window_length),
                top_k_anchors=8,
                top_k_pairs=8,
            )
            build_tensor_bundles(
                project_root,
                scenario=CONFIRMATORY_SCENARIO,
                window_length=int(window_length),
                split_name=split_name,
                split_manifest=split_root / f"{split_name}.parquet",
                phases=["phase5"],
            )

    seed_frames: list[pd.DataFrame] = []
    bundle_root = project_root / "data_processed" / CONFIRMATORY_SCENARIO / "phase5" / "confirmatory"
    used_seeds: list[int] = []
    if not candidate_rows.empty and bundle_root.exists():
        for seed in seeds:
            checkpoint_path, config_path = _checkpoint_and_config_from_seed(
                project_root,
                scenario=benchmark_scenario,
                split_name=benchmark_split,
                seed=int(seed),
            )
            scored = _score_partitions(config_path, checkpoint_path, bundle_root)
            selected_columns = [
                column
                for column in (
                    "seq_id",
                    "full_score",
                    "availability_only_score",
                    "state_gate",
                    "state_correction",
                    "cooperative_gain",
                )
                if column in scored.columns
            ]
            seed_frame = scored[selected_columns].copy()
            seed_frame = seed_frame.rename(
                columns={
                    "full_score": f"full_score_seed_{seed}",
                    "availability_only_score": f"availability_only_score_seed_{seed}",
                    "state_gate": f"state_gate_seed_{seed}",
                    "state_correction": f"state_correction_seed_{seed}",
                    "cooperative_gain": f"cooperative_gain_seed_{seed}",
                }
            )
            seed_frames.append(seed_frame)
            used_seeds.append(int(seed))

    ensemble_scores = pd.DataFrame(
        columns=[
            "seq_id",
            "full_score_mean",
            "full_score_std",
            "availability_only_score_mean",
            "availability_only_score_std",
            "state_gate_mean",
            "state_gate_std",
            "state_correction_mean",
            "state_correction_std",
            "cooperative_gain_mean",
            "cooperative_gain_std",
        ]
    )
    if seed_frames:
        ensemble_scores = seed_frames[0]
        for frame in seed_frames[1:]:
            ensemble_scores = ensemble_scores.merge(frame, on="seq_id", how="inner")
        ensemble_scores = ensemble_scores.drop_duplicates("seq_id", keep="last").copy()
        for prefix in ("full_score", "availability_only_score", "state_gate", "state_correction", "cooperative_gain"):
            seed_columns = [column for column in ensemble_scores.columns if column.startswith(f"{prefix}_seed_")]
            if not seed_columns:
                continue
            ensemble_scores[f"{prefix}_mean"] = ensemble_scores[seed_columns].mean(axis=1)
            ensemble_scores[f"{prefix}_std"] = (
                ensemble_scores[seed_columns].std(axis=1, ddof=1) if len(seed_columns) > 1 else 0.0
            )

    direction_frame = registry.loc[registry["direction_evaluable"]].copy()
    direction_frame = direction_frame.merge(
        status_frame[["seq_id", "has_candidate", "pair_mode", "pair_reference_supported"]],
        on="seq_id",
        how="left",
    )
    direction_frame = direction_frame.merge(ensemble_scores, on="seq_id", how="left")
    direction_frame["has_candidate"] = direction_frame["has_candidate"].fillna(False).astype(bool)
    direction_frame["scored"] = direction_frame["full_score_mean"].notna()
    direction_frame["predicted_direction"] = np.where(
        direction_frame["scored"],
        np.where(direction_frame["full_score_mean"].astype(float) >= float(score_threshold), "activating", "repressive"),
        "",
    )
    direction_frame["direction_consistent"] = np.where(
        direction_frame["scored"],
        direction_frame["predicted_direction"].astype(str) == direction_frame["observed_direction"].astype(str),
        pd.NA,
    )
    direction_frame["coverage_note"] = "scored"
    direction_frame.loc[~direction_frame["supported_state_label"].astype(bool), "coverage_note"] = "unsupported_state_label"
    direction_frame.loc[
        direction_frame["supported_state_label"].astype(bool) & ~direction_frame["has_candidate"].astype(bool),
        "coverage_note",
    ] = "no_candidate_pair"
    direction_frame.loc[
        direction_frame["supported_state_label"].astype(bool)
        & direction_frame["has_candidate"].astype(bool)
        & ~direction_frame["scored"].astype(bool),
        "coverage_note",
    ] = "missing_prediction"
    direction_frame = direction_frame.sort_values(
        ["canonical_biosample", "target_gene", "enhancer_id", "registry_id"]
    ).reset_index(drop=True)
    write_table(direction_frame, report_root / "endogenous_direction_consistency.csv")

    scored_direction = direction_frame.loc[direction_frame["scored"].astype(bool)].copy()
    target_rows: list[dict[str, Any]] = []
    for (biosample, target_gene), group in scored_direction.groupby(["canonical_biosample", "target_gene"], sort=True):
        mean_effect = float(group["effect_size"].mean())
        target_rows.append(
            {
                "canonical_biosample": str(biosample),
                "target_gene": str(target_gene),
                "n_rows": int(len(group)),
                "n_unique_enhancers": int(group["enhancer_id"].astype(str).nunique()),
                "mean_effect_size": mean_effect,
                "observed_target_direction": _observed_direction(mean_effect),
                "mean_full_score": float(group["full_score_mean"].mean()),
                "std_full_score": float(group["full_score_mean"].std(ddof=1)) if len(group) > 1 else 0.0,
                "predicted_target_direction": (
                    "activating" if float(group["full_score_mean"].mean()) >= float(score_threshold) else "repressive"
                ),
                "row_direction_consistency_rate": float(pd.Series(group["direction_consistent"]).astype(float).mean()),
                "target_consistent": (
                    ("activating" if float(group["full_score_mean"].mean()) >= float(score_threshold) else "repressive")
                    == _observed_direction(mean_effect)
                ),
                "pair_modes": ",".join(sorted(group["pair_mode"].dropna().astype(str).unique().tolist())),
            }
        )
    target_frame = pd.DataFrame.from_records(target_rows)
    if not target_frame.empty:
        target_frame = target_frame.sort_values(
            ["canonical_biosample", "n_rows", "target_gene"],
            ascending=[True, False, True],
        ).reset_index(drop=True)
    write_table(target_frame, report_root / "endogenous_target_consistency.csv")

    biosample_rows: list[dict[str, Any]] = []
    for biosample, group in registry.groupby("canonical_biosample", sort=True):
        local_direction = direction_frame.loc[direction_frame["canonical_biosample"].astype(str) == str(biosample)].copy()
        local_scored = local_direction.loc[local_direction["scored"].astype(bool)].copy()
        local_target = target_frame.loc[target_frame["canonical_biosample"].astype(str) == str(biosample)].copy()
        notes: list[str] = []
        if not bool(group["supported_state_label"].astype(bool).any()):
            notes.append("unsupported_state_label")
        if int(group["direction_evaluable"].astype(bool).sum()) == 0:
            notes.append("no_signed_effect_descriptor")
        elif int(local_scored.shape[0]) == 0:
            if int(local_direction["has_candidate"].astype(bool).sum()) == 0:
                notes.append("no_candidate_pair")
            else:
                notes.append("missing_prediction")
        else:
            notes.append("scored")
        biosample_metrics = _direction_auc(local_scored)
        biosample_rows.append(
            {
                "canonical_biosample": str(biosample),
                "registry_rows": int(len(group)),
                "unique_targets": int(group["target_gene"].astype(str).nunique()),
                "direction_evaluable_rows": int(group["direction_evaluable"].astype(bool).sum()),
                "supported_direction_rows": int(
                    (group["direction_evaluable"].astype(bool) & group["supported_state_label"].astype(bool)).sum()
                ),
                "candidate_rows": int(local_direction["has_candidate"].astype(bool).sum()) if not local_direction.empty else 0,
                "scored_rows": int(len(local_scored)),
                "direction_consistency_rate": (
                    float(pd.Series(local_scored["direction_consistent"]).astype(float).mean())
                    if not local_scored.empty
                    else float("nan")
                ),
                "target_consistency_rate": (
                    float(local_target["target_consistent"].astype(float).mean()) if not local_target.empty else float("nan")
                ),
                "mean_full_score": float(local_scored["full_score_mean"].mean()) if not local_scored.empty else float("nan"),
                "median_full_score": float(local_scored["full_score_mean"].median()) if not local_scored.empty else float("nan"),
                "direction_auprc": biosample_metrics["direction_auprc"],
                "direction_auroc": biosample_metrics["direction_auroc"],
                "notes": ",".join(notes),
            }
        )

    overall_metrics = _direction_auc(scored_direction)
    biosample_rows.append(
        {
            "canonical_biosample": "ALL",
            "registry_rows": int(len(registry)),
            "unique_targets": int(registry["target_gene"].astype(str).nunique()),
            "direction_evaluable_rows": int(registry["direction_evaluable"].astype(bool).sum()),
            "supported_direction_rows": int(
                (registry["direction_evaluable"].astype(bool) & registry["supported_state_label"].astype(bool)).sum()
            ),
            "candidate_rows": int(direction_frame["has_candidate"].astype(bool).sum()) if not direction_frame.empty else 0,
            "scored_rows": int(len(scored_direction)),
            "direction_consistency_rate": (
                float(pd.Series(scored_direction["direction_consistent"]).astype(float).mean())
                if not scored_direction.empty
                else float("nan")
            ),
            "target_consistency_rate": (
                float(target_frame["target_consistent"].astype(float).mean()) if not target_frame.empty else float("nan")
            ),
            "mean_full_score": float(scored_direction["full_score_mean"].mean()) if not scored_direction.empty else float("nan"),
            "median_full_score": float(scored_direction["full_score_mean"].median()) if not scored_direction.empty else float("nan"),
            "direction_auprc": overall_metrics["direction_auprc"],
            "direction_auroc": overall_metrics["direction_auroc"],
            "notes": "aggregate",
        }
    )
    biosample_frame = pd.DataFrame.from_records(biosample_rows).sort_values("canonical_biosample").reset_index(drop=True)
    write_table(biosample_frame, report_root / "endogenous_biosample_summary.csv")

    summary_lines = [
        "# Endogenous Confirmatory Evaluation Summary",
        "",
        f"- Benchmark source scenario: `{benchmark_scenario}`",
        f"- Benchmark source split: `{benchmark_split}`",
        f"- Seeds scored: `{', '.join(str(seed) for seed in used_seeds) if used_seeds else 'none'}`",
        f"- Score threshold for activating / repressive calls: `{score_threshold:.3f}`",
        f"- Priority registry rows: `{len(registry)}`",
        f"- Direction-evaluable rows: `{int(registry['direction_evaluable'].astype(bool).sum())}`",
        f"- Direction-evaluable rows with supported state labels: `{supported_direction_rows}`",
        f"- Candidate rows found: `{int(direction_frame['has_candidate'].astype(bool).sum()) if not direction_frame.empty else 0}`",
        f"- Rows scored by the multi-seed ensemble: `{int(len(scored_direction))}`",
        "",
        "## Biosample Summary",
        "",
        "| Biosample | Registry Rows | Direction-Evaluable | Candidate Rows | Scored Rows | Direction Consistency | Target Consistency | Notes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in biosample_frame.itertuples(index=False):
        summary_lines.append(
            f"| {row.canonical_biosample} | {row.registry_rows} | {row.direction_evaluable_rows} | {row.candidate_rows} | {row.scored_rows} | {row.direction_consistency_rate:.6f} | {row.target_consistency_rate:.6f} | {row.notes} |"
        )
    summary_path = report_root / "endogenous_confirmatory_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    manifest = {
        "benchmark_scenario": benchmark_scenario,
        "benchmark_split": benchmark_split,
        "seeds": [int(seed) for seed in used_seeds],
        "score_threshold": float(score_threshold),
        "window_length": int(window_length),
        "top_k_hits": int(top_k_hits),
        "registry_rows": int(len(registry)),
        "direction_evaluable_rows": int(registry["direction_evaluable"].astype(bool).sum()),
        "supported_direction_rows": int(supported_direction_rows),
        "candidate_manifest": candidate_manifest,
        "scored_rows": int(len(scored_direction)),
        "outputs": {
            "direction_consistency": str(report_root / "endogenous_direction_consistency.csv"),
            "target_consistency": str(report_root / "endogenous_target_consistency.csv"),
            "biosample_summary": str(report_root / "endogenous_biosample_summary.csv"),
            "summary_markdown": str(summary_path),
        },
    }
    write_json(report_root / "endogenous_confirmatory_manifest.json", manifest)
    return manifest
