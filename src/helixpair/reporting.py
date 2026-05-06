from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from helixpair.io_utils import ensure_dir, read_json, write_table, write_json, write_text


def _latest_metric_file(phase_root: Path) -> Path | None:
    candidates = sorted(phase_root.rglob("metrics.json"))
    return candidates[-1] if candidates else None


def aggregate_main_metrics(project_root: str | Path, scenario: str = "synthetic") -> dict[str, str]:
    project_root = Path(project_root)
    results_root = ensure_dir(project_root / "results" / "main_tables")
    records = []
    baseline_root = project_root / "results" / "baselines" / scenario
    for metric_path in sorted(baseline_root.rglob("*_metrics.json")):
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        records.append(
            {
                "method": metric_path.stem.replace("_metrics", ""),
                "phase": "baseline",
                "scenario": scenario,
                "split_name": metric_path.parent.name if metric_path.parent != baseline_root else "default",
                "auprc": metrics.get("auprc", 0.0),
                "auroc": metrics.get("auroc", 0.0),
                "ece": metrics.get("ece", 0.0),
                "best_valid_loss": metrics.get("best_valid_loss", 0.0),
            }
        )
    ledger_path = project_root / "results" / "run_ledger.jsonl"
    if ledger_path.exists():
        latest_runs: dict[tuple[str, str, str, int], dict[str, object]] = {}
        with ledger_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("scenario") != scenario or "metrics" not in payload:
                    continue
                key = (
                    str(payload.get("phase", "unknown")),
                    str(payload.get("scenario", scenario)),
                    str(payload.get("split_name", "default")),
                    int(payload.get("seed", -1)),
                )
                latest_runs[key] = payload
        for payload in latest_runs.values():
            records.append(
                {
                    "method": payload.get("phase", "unknown"),
                    "phase": payload.get("phase", "unknown"),
                    "scenario": payload.get("scenario", scenario),
                    "split_name": payload.get("split_name", "default"),
                    "seed": payload.get("seed", -1),
                    "auprc": payload["metrics"].get("auprc", 0.0),
                    "auroc": payload["metrics"].get("auroc", 0.0),
                    "ece": payload["metrics"].get("ece", 0.0),
                    "best_valid_loss": payload["metrics"].get("best_valid_loss", 0.0),
                }
            )
    metrics_frame = pd.DataFrame.from_records(records)
    metrics_path = results_root / "main_metrics.json"
    metrics_frame.to_json(metrics_path, orient="records", indent=2)

    spacing_path = results_root / "spacing_landscape.parquet"
    prediction_candidates = sorted((project_root / "results" / "per_example_predictions" / "phase2" / scenario).rglob("seed_11.parquet"))
    if prediction_candidates:
        predictions = pd.read_parquet(prediction_candidates[-1])
        frame = predictions.copy()
        if "full_score" in frame.columns:
            frame["score"] = frame["full_score"]
        elif "score" not in frame.columns:
            frame["score"] = frame.iloc[:, -1]
        landscape = frame.groupby(["edge_gap", "orientation"], as_index=False)["score"].mean()
        landscape = landscape.rename(columns={"edge_gap": "gap"})
        landscape["method"] = "HelixPair"
        landscape["split_name"] = prediction_candidates[-1].parent.name
    else:
        landscape = pd.DataFrame({"gap": [], "orientation": [], "score": [], "method": [], "split_name": []})
    write_table(landscape, spacing_path)
    return {"metrics_table": str(metrics_path), "spacing_landscape": str(spacing_path)}


def summarize_external_functional_mpra_formal_phase5(
    project_root: str | Path,
    *,
    scenario: str = "external_functional_mpra_benchmark",
    report_root: str | Path | None = None,
    summary_path: str | Path | None = None,
    locked_config_path: str | Path | None = None,
) -> dict[str, str]:
    project_root = Path(project_root)
    report_root = Path(report_root) if report_root is not None else ensure_dir(project_root / "reports" / scenario / "formal_phase5_multiseed")
    summary_path = Path(summary_path) if summary_path is not None else report_root / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing formal summary: {summary_path}")

    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    if not rows:
        raise ValueError(f"Formal summary contains no rows: {summary_path}")

    frame = pd.DataFrame.from_records(rows).sort_values(["split_name", "seed"]).reset_index(drop=True)
    metric_columns = [
        "auprc",
        "auroc",
        "ece",
        "best_valid_loss",
        "relative_gain_vs_availability_auprc",
        "relative_gain_vs_additive_auprc",
    ]
    for column in metric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    seed_metrics_path = report_root / "seed_metrics.csv"
    write_table(frame, seed_metrics_path)

    grouped = frame.groupby("split_name", sort=True)
    summary_rows: list[dict[str, object]] = []
    for split_name, group in grouped:
        row: dict[str, object] = {
            "split_name": str(split_name),
            "n_seeds": int(group["seed"].nunique()),
            "seeds": ",".join(str(int(seed)) for seed in sorted(group["seed"].dropna().astype(int).tolist())),
            "best_seed_by_auprc": int(group.sort_values(["auprc", "seed"], ascending=[False, True]).iloc[0]["seed"]),
        }
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            row[f"{metric}_min"] = float(group[metric].min())
            row[f"{metric}_max"] = float(group[metric].max())
        summary_rows.append(row)

    summary_frame = pd.DataFrame.from_records(summary_rows).sort_values("split_name").reset_index(drop=True)
    split_summary_path = report_root / "split_summary.csv"
    split_summary_json_path = report_root / "split_summary.json"
    write_table(summary_frame, split_summary_path)
    write_json(split_summary_json_path, {"rows": summary_rows})

    locked_config_value = str(locked_config_path) if locked_config_path else ""
    lines = [
        "# External Functional MPRA Phase5 Formal Summary",
        "",
        f"- Scenario: `{scenario}`",
        f"- Source summary: `{summary_path}`",
        f"- Locked config: `{locked_config_value or 'not recorded'}`",
        f"- Splits summarized: `{summary_frame['split_name'].nunique()}`",
        f"- Total runs summarized: `{len(frame)}`",
        "",
        "## Split Aggregates",
        "",
    ]
    for row in summary_rows:
        lines.extend(
            [
                f"### {row['split_name']}",
                "",
                f"- Seeds: `{row['seeds']}`",
                f"- Best seed by AUPRC: `{row['best_seed_by_auprc']}`",
                f"- AUPRC: `{row['auprc_mean']:.6f} ± {row['auprc_std']:.6f}` (range `{row['auprc_min']:.6f}` to `{row['auprc_max']:.6f}`)",
                f"- AUROC: `{row['auroc_mean']:.6f} ± {row['auroc_std']:.6f}`",
                f"- ECE: `{row['ece_mean']:.6f} ± {row['ece_std']:.6f}`",
                f"- Gain vs availability AUPRC: `{row['relative_gain_vs_availability_auprc_mean']:.6f} ± {row['relative_gain_vs_availability_auprc_std']:.6f}`",
                f"- Gain vs additive AUPRC: `{row['relative_gain_vs_additive_auprc_mean']:.6f} ± {row['relative_gain_vs_additive_auprc_std']:.6f}`",
                "",
            ]
        )
    summary_markdown_path = report_root / "summary.md"
    write_text(summary_markdown_path, "\n".join(lines).rstrip() + "\n")
    return {
        "summary_json": str(summary_path),
        "seed_metrics_csv": str(seed_metrics_path),
        "split_summary_csv": str(split_summary_path),
        "split_summary_json": str(split_summary_json_path),
        "summary_markdown": str(summary_markdown_path),
    }
