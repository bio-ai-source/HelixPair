from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from helixpair.io_utils import ensure_dir, read_json, read_table, resolve_path, write_json, write_table


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if labels.size == 0:
        raise ValueError("Cannot score empty prediction vectors.")
    if np.unique(labels).size < 2:
        raise ValueError("Cannot score degenerate label vectors with fewer than 2 classes.")
    return {
        "auprc": float(average_precision_score(labels, scores)),
        "auroc": float(roc_auc_score(labels, scores)),
    }


def _safe_metric(metric_name: str, labels: np.ndarray, scores: np.ndarray) -> float:
    if metric_name == "auprc":
        return float(average_precision_score(labels, scores))
    if metric_name == "auroc":
        return float(roc_auc_score(labels, scores))
    raise ValueError(f"Unsupported metric: {metric_name}")


def paired_bootstrap_metric_diff(
    labels: np.ndarray,
    model_scores: np.ndarray,
    baseline_scores: np.ndarray,
    *,
    metric_name: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=float)
    model_scores = np.asarray(model_scores, dtype=float)
    baseline_scores = np.asarray(baseline_scores, dtype=float)
    observed_model = _safe_metric(metric_name, labels, model_scores)
    observed_baseline = _safe_metric(metric_name, labels, baseline_scores)
    observed_diff = float(observed_model - observed_baseline)
    rng = np.random.default_rng(seed)
    diffs = np.empty(int(n_bootstrap), dtype=float)
    n_rows = int(labels.shape[0])
    for index in range(int(n_bootstrap)):
        sampled = rng.integers(0, n_rows, size=n_rows)
        boot_labels = labels[sampled]
        if np.unique(boot_labels).size < 2:
            diffs[index] = observed_diff
            continue
        diffs[index] = float(
            _safe_metric(metric_name, boot_labels, model_scores[sampled])
            - _safe_metric(metric_name, boot_labels, baseline_scores[sampled])
        )
    ci_low, ci_high = np.quantile(diffs, [0.025, 0.975])
    bootstrap_p_value = float((np.sum(diffs <= 0.0) + 1.0) / (len(diffs) + 1.0))
    return {
        "metric_name": metric_name,
        "model_metric": observed_model,
        "baseline_metric": observed_baseline,
        "observed_diff": observed_diff,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "bootstrap_p_value": bootstrap_p_value,
    }


def paired_permutation_metric_diff(
    labels: np.ndarray,
    model_scores: np.ndarray,
    baseline_scores: np.ndarray,
    *,
    metric_name: str,
    n_permutations: int,
    seed: int,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=float)
    model_scores = np.asarray(model_scores, dtype=float)
    baseline_scores = np.asarray(baseline_scores, dtype=float)
    observed_diff = float(
        _safe_metric(metric_name, labels, model_scores) - _safe_metric(metric_name, labels, baseline_scores)
    )
    rng = np.random.default_rng(seed)
    diffs = np.empty(int(n_permutations), dtype=float)
    for index in range(int(n_permutations)):
        swap_mask = rng.random(size=labels.shape[0]) < 0.5
        perm_model = np.where(swap_mask, baseline_scores, model_scores)
        perm_baseline = np.where(swap_mask, model_scores, baseline_scores)
        diffs[index] = float(
            _safe_metric(metric_name, labels, perm_model) - _safe_metric(metric_name, labels, perm_baseline)
        )
    permutation_p_value = float((np.sum(diffs >= observed_diff) + 1.0) / (len(diffs) + 1.0))
    return {
        "metric_name": metric_name,
        "observed_diff": observed_diff,
        "permutation_p_value": permutation_p_value,
    }


def seed_inventory_for_scenario(
    project_root: str | Path,
    *,
    scenario: str,
    phases: tuple[str, ...] | None = None,
    splits: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    project_root = resolve_path(project_root)
    rows: list[dict[str, Any]] = []
    selected_phases = {str(value) for value in phases} if phases is not None else None
    selected_splits = {str(value) for value in splits} if splits is not None else None
    for phase in ["phase4", "phase5"]:
        if selected_phases is not None and phase not in selected_phases:
            continue
        phase_root = project_root / "checkpoints" / phase / scenario
        if not phase_root.exists():
            continue
        for split_dir in sorted([path for path in phase_root.iterdir() if path.is_dir()]):
            if selected_splits is not None and split_dir.name not in selected_splits:
                continue
            for seed_dir in sorted([path for path in split_dir.iterdir() if path.is_dir() and path.name.startswith("seed_")]):
                metrics_files = list(seed_dir.glob("*/metrics.json"))
                rows.append(
                    {
                        "phase": phase,
                        "scenario": scenario,
                        "split_name": split_dir.name,
                        "seed_dir": seed_dir.name,
                        "run_count": int(len(metrics_files)),
                    }
                )
    return pd.DataFrame.from_records(rows)


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    if p_values.size == 0:
        return np.asarray([], dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order] * float(len(p_values)) / np.arange(1, len(p_values) + 1, dtype=float)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    output = np.empty_like(adjusted)
    output[order] = adjusted
    return output


def _phase_prediction_path(project_root: Path, *, phase: str, scenario: str, split_name: str, seed: int) -> Path:
    return project_root / "results" / "per_example_predictions" / phase / scenario / split_name / f"seed_{seed}.parquet"


def _baseline_prediction_path(project_root: Path, *, scenario: str, split_name: str, baseline_name: str) -> Path:
    return project_root / "results" / "baselines" / scenario / split_name / baseline_name / "predictions.parquet"


def _load_helixpair_predictions(project_root: Path, *, phase: str, scenario: str, split_name: str, seed: int) -> pd.DataFrame:
    path = _phase_prediction_path(project_root, phase=phase, scenario=scenario, split_name=split_name, seed=seed)
    frame = read_table(path).copy()
    frame["helixpair_score"] = pd.to_numeric(frame["full_score"], errors="coerce")
    frame["availability_only_score"] = pd.to_numeric(frame["availability_only_score"], errors="coerce")
    frame["additive_null_score"] = pd.to_numeric(frame["additive_null_score"], errors="coerce")
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    return frame


def _collect_methods_from_helixpair_frame(
    helixpair: pd.DataFrame,
    project_root: Path,
    *,
    scenario: str,
    split_name: str,
    include_external_baselines: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    methods = ["HelixPair", "availability_only", "additive_null"]
    merged = helixpair[["seq_id", "label", "helixpair_score", "availability_only_score", "additive_null_score"]].rename(
        columns={
            "helixpair_score": "HelixPair",
            "availability_only_score": "availability_only",
            "additive_null_score": "additive_null",
        }
    )

    baseline_meta_path = project_root / "results" / "baselines" / scenario / split_name / "external_baselines.json"
    if include_external_baselines and baseline_meta_path.exists():
        baseline_meta = read_json(baseline_meta_path)
        for baseline_name, payload in sorted(baseline_meta.items()):
            if str(payload.get("status", "")) != "completed":
                continue
            pred_path = _baseline_prediction_path(project_root, scenario=scenario, split_name=split_name, baseline_name=baseline_name)
            if not pred_path.exists():
                continue
            baseline_frame = read_table(pred_path).copy()
            if "score" not in baseline_frame.columns:
                continue
            baseline_frame = baseline_frame[["seq_id", "score"]].rename(columns={"score": baseline_name})
            merged = merged.merge(baseline_frame, on="seq_id", how="inner")
            methods.append(str(baseline_name))
    merged = merged.dropna(subset=["label", *methods]).reset_index(drop=True)
    return merged, methods


def _collect_methods(
    project_root: Path,
    *,
    phase: str,
    scenario: str,
    split_name: str,
    seed: int,
    include_external_baselines: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    helixpair = _load_helixpair_predictions(project_root, phase=phase, scenario=scenario, split_name=split_name, seed=seed)
    return _collect_methods_from_helixpair_frame(
        helixpair,
        project_root,
        scenario=scenario,
        split_name=split_name,
        include_external_baselines=include_external_baselines,
    )


def _existing_split_names(
    project_root: Path,
    *,
    scenario: str,
    phases: tuple[str, ...],
    seeds: tuple[int, ...] | None = None,
    seed: int | None = 11,
) -> tuple[str, ...]:
    candidate_seeds = tuple(int(value) for value in (seeds or (() if seed is None else (seed,)))) or (11,)
    names: set[str] = set()
    for phase in phases:
        phase_root = project_root / "results" / "per_example_predictions" / phase / scenario
        if not phase_root.exists():
            continue
        for split_dir in sorted(path for path in phase_root.iterdir() if path.is_dir()):
            if any((split_dir / f"seed_{current_seed}.parquet").exists() for current_seed in candidate_seeds):
                names.add(split_dir.name)
    return tuple(sorted(names))


def _normalize_requested_seeds(seed: int, seeds: tuple[int, ...] | None) -> tuple[int, ...]:
    if seeds is None:
        return (int(seed),)
    normalized: list[int] = []
    seen: set[int] = set()
    for value in seeds:
        current = int(value)
        if current in seen:
            continue
        seen.add(current)
        normalized.append(current)
    return tuple(normalized) or (int(seed),)


def _aggregate_seed_prediction_frames(seed_frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    if not seed_frames:
        raise ValueError("Cannot aggregate empty seed prediction frames.")
    merged: pd.DataFrame | None = None
    score_columns = ["helixpair_score", "availability_only_score", "additive_null_score"]
    for current_seed, frame in sorted(seed_frames.items()):
        current = frame[["seq_id", "label", *score_columns]].copy()
        current = current.rename(columns={column: f"{column}__seed_{current_seed}" for column in score_columns})
        if merged is None:
            merged = current
        else:
            merged = merged.merge(current, on=["seq_id", "label"], how="inner")
    if merged is None:
        raise ValueError("Failed to merge seed prediction frames.")
    aggregated = merged[["seq_id", "label"]].copy()
    for column in score_columns:
        seed_columns = [name for name in merged.columns if name.startswith(f"{column}__seed_")]
        aggregated[column] = merged[seed_columns].mean(axis=1)
    return aggregated


def _stats_random_seed(seed_values: tuple[int, ...]) -> int:
    mixed = sum((index + 1) * int(value) for index, value in enumerate(seed_values))
    return max(int(mixed % (2**31 - 1)), 1)


def _feature_baseline_rows(
    project_root: Path,
    *,
    scenario: str,
    split_name: str,
    phase: str,
) -> list[dict[str, Any]]:
    path = project_root / "results" / "baselines" / scenario / split_name / "feature_baselines.json"
    if not path.exists():
        return []
    payload = read_json(path)
    phase_payload = payload.get(phase, {})
    if not isinstance(phase_payload, dict):
        return []

    rows: list[dict[str, Any]] = []
    for feature_group, methods in sorted(phase_payload.items()):
        if not isinstance(methods, dict):
            continue
        for method_name, metrics in sorted(methods.items()):
            if not isinstance(metrics, dict):
                continue
            if "auprc" not in metrics:
                continue
            rows.append(
                {
                    "scenario": scenario,
                    "split_name": split_name,
                    "phase": phase,
                    "feature_group": str(feature_group),
                    "estimator": str(method_name),
                    "method": f"{feature_group}/{method_name}",
                    "paired_stats_available": False,
                    "auprc": float(metrics.get("auprc", float("nan"))),
                    "auroc": float(metrics.get("auroc", float("nan"))),
                    "balanced_accuracy": float(metrics.get("balanced_accuracy", float("nan"))),
                    "brier": float(metrics.get("brier", float("nan"))),
                    "ece": float(metrics.get("ece", float("nan"))),
                    "nll": float(metrics.get("nll", float("nan"))),
                }
            )
    return rows


def summarize_benchmark_significance(
    project_root: str | Path,
    *,
    scenario: str,
    report_root: str | Path | None = None,
    output_prefix: str = "benchmark",
    summary_title: str | None = None,
    splits: tuple[str, ...] | None = None,
    phases: tuple[str, ...] = ("phase4", "phase5"),
    metrics: tuple[str, ...] = ("auprc", "auroc"),
    seed: int = 11,
    seeds: tuple[int, ...] | None = None,
    seed_aggregation: str = "none",
    n_bootstrap: int = 5000,
    n_permutations: int = 5000,
    preferred_external_baselines: tuple[str, ...] = ("BOM", "COBIND", "GET", "SATORI", "TF_COMB", "TIANA"),
    external_baseline_phases: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    project_root = resolve_path(project_root)
    report_root = ensure_dir(report_root or (project_root / "reports" / scenario))
    requested_seeds = _normalize_requested_seeds(seed, seeds)
    seed_aggregation_mode = str(seed_aggregation).strip().lower() or "none"
    aggregate_seeds = seed_aggregation_mode not in {"none", "single", "single_seed"} and len(requested_seeds) > 1
    resolved_splits = tuple(
        splits
        or _existing_split_names(project_root, scenario=scenario, phases=phases, seeds=requested_seeds, seed=seed)
    )
    title = summary_title or f"{scenario} Significance Summary"

    seed_inventory = seed_inventory_for_scenario(project_root, scenario=scenario, phases=phases, splits=resolved_splits)
    seed_inventory_path = report_root / f"{output_prefix}_seed_inventory.csv"
    write_table(seed_inventory, seed_inventory_path)

    metric_rows: list[dict[str, Any]] = []
    significance_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    missing_slices: list[dict[str, Any]] = []

    for split_name in resolved_splits:
        for phase in phases:
            active_seeds: tuple[int, ...]
            if aggregate_seeds:
                seed_frames: dict[int, pd.DataFrame] = {}
                for requested_seed in requested_seeds:
                    prediction_path = _phase_prediction_path(
                        project_root,
                        phase=phase,
                        scenario=scenario,
                        split_name=split_name,
                        seed=requested_seed,
                    )
                    if prediction_path.exists():
                        seed_frames[int(requested_seed)] = _load_helixpair_predictions(
                            project_root,
                            phase=phase,
                            scenario=scenario,
                            split_name=split_name,
                            seed=int(requested_seed),
                        )
                        continue
                    missing_slices.append(
                        {
                            "scenario": scenario,
                            "split_name": split_name,
                            "phase": phase,
                            "seed": int(requested_seed),
                            "reason": "missing_helixpair_predictions",
                            "path": str(prediction_path),
                        }
                    )
                if not seed_frames:
                    continue
                active_seeds = tuple(sorted(seed_frames))
                helixpair_frame = _aggregate_seed_prediction_frames(seed_frames)
                summary_seed_value: str | int = "aggregate"
            else:
                prediction_path = _phase_prediction_path(
                    project_root,
                    phase=phase,
                    scenario=scenario,
                    split_name=split_name,
                    seed=seed,
                )
                if not prediction_path.exists():
                    missing_slices.append(
                        {
                            "scenario": scenario,
                            "split_name": split_name,
                            "phase": phase,
                            "seed": seed,
                            "reason": "missing_helixpair_predictions",
                            "path": str(prediction_path),
                        }
                    )
                    continue
                active_seeds = (int(seed),)
                helixpair_frame = _load_helixpair_predictions(
                    project_root,
                    phase=phase,
                    scenario=scenario,
                    split_name=split_name,
                    seed=int(seed),
                )
                summary_seed_value = int(seed)

            merged, methods = _collect_methods_from_helixpair_frame(
                helixpair_frame,
                project_root,
                scenario=scenario,
                split_name=split_name,
                include_external_baselines=external_baseline_phases is None or phase in external_baseline_phases,
            )
            if merged.empty:
                missing_slices.append(
                    {
                        "scenario": scenario,
                        "split_name": split_name,
                        "phase": phase,
                        "seed": summary_seed_value,
                        "reason": "no_aligned_rows_after_merge",
                        "path": str(
                            _phase_prediction_path(
                                project_root,
                                phase=phase,
                                scenario=scenario,
                                split_name=split_name,
                                seed=int(active_seeds[0]),
                            )
                        ),
                    }
                )
                continue

            labels = merged["label"].to_numpy(dtype=float)
            seed_list_token = ",".join(str(value) for value in active_seeds)
            stats_seed = _stats_random_seed(active_seeds)
            for method in methods:
                method_metrics = binary_metrics(labels, merged[method].to_numpy(dtype=float))
                metric_rows.append(
                    {
                        "scenario": scenario,
                        "split_name": split_name,
                        "phase": phase,
                        "method": method,
                        "seed": summary_seed_value,
                        "seed_count": int(len(active_seeds)),
                        "seed_list": seed_list_token,
                        "seed_aggregation": seed_aggregation_mode if aggregate_seeds else "none",
                        "rows": int(len(merged)),
                        **method_metrics,
                    }
                )
            for baseline_name in [method for method in methods if method != "HelixPair"]:
                for metric_name in metrics:
                    bootstrap = paired_bootstrap_metric_diff(
                        labels,
                        merged["HelixPair"].to_numpy(dtype=float),
                        merged[baseline_name].to_numpy(dtype=float),
                        metric_name=metric_name,
                        n_bootstrap=int(n_bootstrap),
                        seed=stats_seed,
                    )
                    permutation = paired_permutation_metric_diff(
                        labels,
                        merged["HelixPair"].to_numpy(dtype=float),
                        merged[baseline_name].to_numpy(dtype=float),
                        metric_name=metric_name,
                        n_permutations=int(n_permutations),
                        seed=stats_seed,
                    )
                    significance_rows.append(
                        {
                            "scenario": scenario,
                            "split_name": split_name,
                            "phase": phase,
                            "seed": summary_seed_value,
                            "seed_count": int(len(active_seeds)),
                            "seed_list": seed_list_token,
                            "seed_aggregation": seed_aggregation_mode if aggregate_seeds else "none",
                            "comparison_method": baseline_name,
                            **bootstrap,
                            **{key: value for key, value in permutation.items() if key != "metric_name"},
                            "rows": int(len(merged)),
                        }
                    )
            feature_rows.extend(_feature_baseline_rows(project_root, scenario=scenario, split_name=split_name, phase=phase))

    metrics_frame = pd.DataFrame.from_records(metric_rows)
    if not metrics_frame.empty:
        metrics_frame = metrics_frame.sort_values(["split_name", "phase", "method"]).reset_index(drop=True)

    significance_frame = pd.DataFrame.from_records(significance_rows)
    if not significance_frame.empty:
        significance_frame = significance_frame.sort_values(
            ["split_name", "phase", "metric_name", "observed_diff"],
            ascending=[True, True, True, False],
        ).reset_index(drop=True)
        significance_frame["bh_fdr"] = _benjamini_hochberg(
            significance_frame["permutation_p_value"].to_numpy(dtype=float)
        )
        significance_frame["significant_win"] = (
            (significance_frame["ci_low"].astype(float) > 0.0)
            & (significance_frame["permutation_p_value"].astype(float) < 0.05)
        )
        significance_frame["significant_win_fdr"] = (
            (significance_frame["ci_low"].astype(float) > 0.0)
            & (significance_frame["bh_fdr"].astype(float) < 0.05)
        )

    feature_frame = pd.DataFrame.from_records(feature_rows)
    if not feature_frame.empty:
        feature_frame = feature_frame.sort_values(
            ["split_name", "phase", "auprc", "method"],
            ascending=[True, True, False, True],
        ).reset_index(drop=True)

    missing_frame = pd.DataFrame.from_records(missing_slices)
    if not missing_frame.empty:
        missing_frame = missing_frame.sort_values(["split_name", "phase", "reason"]).reset_index(drop=True)

    metrics_path = report_root / f"{output_prefix}_significance_metrics.csv"
    significance_path = report_root / f"{output_prefix}_significance_summary.csv"
    feature_path = report_root / f"{output_prefix}_feature_baseline_summary.csv"
    missing_path = report_root / f"{output_prefix}_missing_slices.csv"
    write_table(metrics_frame, metrics_path)
    write_table(significance_frame, significance_path)
    write_table(feature_frame, feature_path)
    write_table(missing_frame, missing_path)

    preferred_set = {str(name) for name in preferred_external_baselines}
    claim_rows: list[dict[str, Any]] = []
    if not significance_frame.empty:
        for (split_name, phase, metric_name), group in significance_frame.groupby(
            ["split_name", "phase", "metric_name"],
            sort=False,
        ):
            external_group = group[group["comparison_method"].astype(str).isin(preferred_set)].copy()
            candidate_group = external_group if not external_group.empty else group.copy()
            best_row = candidate_group.sort_values(
                ["baseline_metric", "comparison_method"],
                ascending=[False, True],
            ).iloc[0]
            claim_rows.append(
                {
                    "split_name": str(split_name),
                    "phase": str(phase),
                    "metric_name": str(metric_name),
                    "best_baseline": str(best_row.comparison_method),
                    "helixpair_metric": float(best_row.model_metric),
                    "baseline_metric": float(best_row.baseline_metric),
                    "lead": float(best_row.observed_diff),
                    "ci_low": float(best_row.ci_low),
                    "ci_high": float(best_row.ci_high),
                    "permutation_p_value": float(best_row.permutation_p_value),
                    "bootstrap_p_value": float(best_row.bootstrap_p_value),
                    "bh_fdr": float(best_row.bh_fdr),
                    "significant_win": bool(best_row.significant_win),
                    "significant_win_fdr": bool(best_row.significant_win_fdr),
                    "seed": best_row.seed,
                    "seed_count": int(best_row.seed_count),
                    "seed_list": str(best_row.seed_list),
                    "seed_aggregation": str(best_row.seed_aggregation),
                }
            )

    claim_frame = pd.DataFrame.from_records(claim_rows)
    claim_ready_path = report_root / f"{output_prefix}_claim_ready_summary.csv"
    if not claim_frame.empty:
        claim_frame = claim_frame.sort_values(["metric_name", "phase", "split_name"]).reset_index(drop=True)
    write_table(claim_frame, claim_ready_path)

    summary_md = [
        f"# {title}",
        "",
        f"- Scenario: `{scenario}`",
        f"- Splits summarized: `{', '.join(resolved_splits) if resolved_splits else 'none'}`",
        f"- Seeds available: `{', '.join(sorted(seed_inventory['seed_dir'].astype(str).unique().tolist())) if not seed_inventory.empty else 'none'}`",
        f"- Requested summary seeds: `{', '.join(str(value) for value in requested_seeds)}`",
        f"- Bootstrap rounds per comparison: `{n_bootstrap}`",
        f"- Permutation rounds per comparison: `{n_permutations}`",
        "- BH-FDR is applied to permutation p-values across all summarized pairwise comparisons.",
        "",
        "| Split | Phase | Metric | Baseline | HelixPair | Baseline | Diff | 95% CI | Permutation p | BH-FDR |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    if aggregate_seeds:
        summary_md.insert(5, f"- Summary seed aggregation: `example-wise {seed_aggregation_mode}` over the available seeds per slice.")
    else:
        summary_md.insert(5, f"- Current summary seed: `{seed}`")
    for row in significance_frame.itertuples(index=False):
        summary_md.append(
            f"| {row.split_name} | {row.phase} | {row.metric_name} | {row.comparison_method} | {row.model_metric:.6f} | {row.baseline_metric:.6f} | {row.observed_diff:.6f} | [{row.ci_low:.6f}, {row.ci_high:.6f}] | {row.permutation_p_value:.6g} | {row.bh_fdr:.6g} |"
        )

    if not feature_frame.empty:
        summary_md.extend(
            [
                "",
                "## Feature Baseline Snapshot",
                "",
                "| Split | Phase | Method | AUPRC | AUROC | Paired Stats |",
                "| --- | --- | --- | ---: | ---: | --- |",
            ]
        )
        for row in feature_frame.itertuples(index=False):
            summary_md.append(
                f"| {row.split_name} | {row.phase} | {row.method} | {row.auprc:.6f} | {row.auroc:.6f} | {'yes' if row.paired_stats_available else 'no'} |"
            )

    if not missing_frame.empty:
        summary_md.extend(
            [
                "",
                "## Missing Slices",
                "",
                "| Split | Phase | Seed | Reason | Path |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in missing_frame.itertuples(index=False):
            summary_md.append(f"| {row.split_name} | {row.phase} | {row.seed} | {row.reason} | {row.path} |")

    summary_markdown_path = report_root / f"{output_prefix}_significance_summary.md"
    summary_markdown_path.write_text("\n".join(summary_md) + "\n", encoding="utf-8")

    claim_md = [
        f"# {title.replace('Summary', 'Claim-Ready Summary')}",
        "",
        "| Split | Phase | Metric | Strongest Baseline | HelixPair | Baseline | Lead | 95% CI | Permutation p | BH-FDR | Significant Win |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for row in claim_frame.itertuples(index=False):
        claim_md.append(
            f"| {row.split_name} | {row.phase} | {row.metric_name} | {row.best_baseline} | {row.helixpair_metric:.6f} | {row.baseline_metric:.6f} | {row.lead:.6f} | [{row.ci_low:.6f}, {row.ci_high:.6f}] | {row.permutation_p_value:.6g} | {row.bh_fdr:.6g} | {'yes' if row.significant_win_fdr else 'no'} |"
        )
    claim_markdown_path = report_root / f"{output_prefix}_claim_ready_summary.md"
    claim_markdown_path.write_text("\n".join(claim_md) + "\n", encoding="utf-8")

    manifest = {
        "scenario": scenario,
        "summary_seed": None if aggregate_seeds else int(seed),
        "requested_seeds": [int(value) for value in requested_seeds],
        "seed_aggregation": seed_aggregation_mode if aggregate_seeds else "none",
        "available_seed_dirs": sorted(seed_inventory["seed_dir"].astype(str).unique().tolist())
        if not seed_inventory.empty
        else [],
        "splits": list(resolved_splits),
        "phases": list(phases),
        "metrics": list(metrics),
        "n_bootstrap": int(n_bootstrap),
        "n_permutations": int(n_permutations),
        "outputs": {
            "seed_inventory": str(seed_inventory_path),
            "metrics": str(metrics_path),
            "significance_summary": str(significance_path),
            "summary_markdown": str(summary_markdown_path),
            "claim_ready_summary": str(claim_ready_path),
            "claim_ready_markdown": str(claim_markdown_path),
            "feature_baseline_summary": str(feature_path),
            "missing_slices": str(missing_path),
        },
    }
    write_json(report_root / f"{output_prefix}_significance_manifest.json", manifest)
    return manifest


def summarize_phase45_significance(
    project_root: str | Path,
    *,
    scenario: str = "phase45_hematopoiesis",
    splits: tuple[str, ...] = ("default", "unseen_pair", "unseen_state"),
    phases: tuple[str, ...] = ("phase4", "phase5"),
    metrics: tuple[str, ...] = ("auprc", "auroc"),
    seed: int = 11,
    seeds: tuple[int, ...] | None = None,
    seed_aggregation: str = "none",
    n_bootstrap: int = 5000,
    n_permutations: int = 5000,
) -> dict[str, Any]:
    project_root = resolve_path(project_root)
    return summarize_benchmark_significance(
        project_root,
        scenario=scenario,
        report_root=project_root / "reports" / "external_phase45",
        output_prefix="phase45",
        summary_title="Phase45 Significance Summary",
        splits=splits,
        phases=phases,
        metrics=metrics,
        seed=seed,
        seeds=seeds,
        seed_aggregation=seed_aggregation,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        preferred_external_baselines=("BOM", "COBIND", "GET", "SATORI", "TF_COMB", "TIANA"),
        external_baseline_phases=("phase4",),
    )
