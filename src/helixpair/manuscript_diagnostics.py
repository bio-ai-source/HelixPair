from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from helixpair.constants import DEFAULT_OFFSETS
from helixpair.io_utils import ensure_dir, read_table, write_json, write_table, write_text


def run_synthetic_identifiability_report(
    output_dir: str | Path,
    *,
    seed: int = 11,
    n_examples: int = 2048,
) -> dict[str, str]:
    output_dir = ensure_dir(output_dir)
    rng = np.random.default_rng(seed)
    rows = []
    for scenario in ("spacing_only", "composite_only", "mixed_grammar", "signed_suppression"):
        gap = rng.integers(-4, 21, size=n_examples)
        phase = np.cos(2.0 * np.pi * gap / 10.5)
        composite = rng.binomial(1, 0.35, size=n_examples)
        monomer = rng.normal(0.0, 1.0, size=n_examples)
        state = rng.normal(0.0, 1.0, size=n_examples)
        if scenario == "spacing_only":
            pair = 1.4 * phase
        elif scenario == "composite_only":
            pair = 1.8 * composite
        elif scenario == "mixed_grammar":
            pair = 0.9 * phase + 1.2 * composite
        else:
            pair = 1.0 * phase - 1.6 * (state > 0.75).astype(float)
        logits = 0.9 * monomer + pair + rng.normal(0.0, 0.35, size=n_examples)
        labels = (rng.random(n_examples) < 1.0 / (1.0 + np.exp(-logits))).astype(int)
        feature_frame = pd.DataFrame(
            {
                "monomer": monomer,
                "helical_cos": phase,
                "helical_sin": np.sin(2.0 * np.pi * gap / 10.5),
                "composite": composite,
                "state": state,
            }
        )
        train_x, test_x, train_y, test_y, train_pair, test_pair = train_test_split(
            feature_frame,
            labels,
            pair,
            test_size=0.35,
            random_state=seed,
            stratify=labels,
        )
        clf = LogisticRegression(max_iter=1000).fit(train_x, train_y)
        scores = clf.predict_proba(test_x)[:, 1]
        ridge = Ridge(alpha=1.0).fit(train_x[["helical_cos", "helical_sin", "composite", "state"]], train_pair)
        recovered = ridge.predict(test_x[["helical_cos", "helical_sin", "composite", "state"]])
        rows.append(
            {
                "scenario": scenario,
                "n_examples": int(n_examples),
                "prevalence": float(test_y.mean()),
                "auprc": float(average_precision_score(test_y, scores)),
                "auroc": float(roc_auc_score(test_y, scores)) if len(np.unique(test_y)) > 1 else 0.0,
                "pair_component_correlation": float(np.corrcoef(test_pair, recovered)[0, 1]),
            }
        )
    frame = pd.DataFrame.from_records(rows)
    csv_path = output_dir / "synthetic_identifiability.csv"
    json_path = output_dir / "synthetic_identifiability.json"
    md_path = output_dir / "synthetic_identifiability.md"
    write_table(frame, csv_path)
    write_json(json_path, {"rows": rows})
    write_text(md_path, "# Synthetic Identifiability Diagnostics\n\n" + _frame_to_markdown(frame) + "\n")
    return {"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)}


def run_state_control_report(prediction_table: str | Path, output_dir: str | Path, *, seed: int = 11) -> dict[str, str]:
    output_dir = ensure_dir(output_dir)
    frame = read_table(prediction_table).copy()
    if "label" not in frame or "full_score" not in frame:
        raise ValueError("State-control report requires label and full_score columns.")
    rows: list[dict[str, Any]] = []
    labels = frame["label"].astype(int).to_numpy()
    full = frame["full_score"].astype(float).to_numpy()
    availability = frame.get("availability_only_score", pd.Series(np.zeros(len(frame)))).astype(float).to_numpy()
    residual = full - availability
    rows.append(
        {
            "control": "full_vs_label",
            "metric": "auprc",
            "value": float(average_precision_score(labels, full)),
            "interpretation": "Observed score-label association.",
        }
    )
    rows.append(
        {
            "control": "availability_residual_vs_label",
            "metric": "auprc",
            "value": float(average_precision_score(labels, residual)),
            "interpretation": "Signal remaining after subtracting availability-only score.",
        }
    )
    rng = np.random.default_rng(seed)
    shuffled = full.copy()
    rng.shuffle(shuffled)
    rows.append(
        {
            "control": "state_shuffled_score_vs_label",
            "metric": "auprc",
            "value": float(average_precision_score(labels, shuffled)),
            "interpretation": "Permutation control for state-score alignment.",
        }
    )
    if "availability_stratum" in frame:
        gains = []
        for _stratum, group in frame.groupby("availability_stratum"):
            if len(group) >= 2:
                gains.append(float(group["full_score"].mean() - group.get("availability_only_score", 0).mean()))
        rows.append(
            {
                "control": "availability_matched_gain",
                "metric": "mean_delta_score",
                "value": float(np.mean(gains)) if gains else 0.0,
                "interpretation": "Mean full-minus-availability score within availability strata.",
            }
        )
    state_columns = [column for column in frame.columns if str(column).startswith("state_")]
    if state_columns:
        values = frame[state_columns].fillna(0.0).to_numpy()
        mi = mutual_info_classif(values, labels, discrete_features=False, random_state=seed)
        rows.append(
            {
                "control": "state_mutual_information",
                "metric": "mean_mi",
                "value": float(np.mean(mi)),
                "interpretation": "Average state-feature mutual information with labels.",
            }
        )
    result = pd.DataFrame.from_records(rows)
    csv_path = output_dir / "state_control_report.csv"
    json_path = output_dir / "state_control_report.json"
    md_path = output_dir / "state_control_report.md"
    write_table(result, csv_path)
    write_json(json_path, {"rows": rows})
    write_text(md_path, "# State-Control Diagnostics\n\n" + _frame_to_markdown(result) + "\n")
    return {"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)}


def write_sensitivity_summary(config: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    output_dir = ensure_dir(output_dir)
    data = config.get("data", {})
    model = config.get("model", {})
    ablations = config.get("ablations", {})
    rows = [
        {
            "component": "helical_harmonic_order",
            "value": int(model.get("helical_order", 2)),
            "status": "reported",
        },
        {"component": "window_length_bp", "value": int(data.get("window_length", 96)), "status": "reported"},
        {"component": "top_k_anchors", "value": int(data.get("top_k_anchors", 24)), "status": "reported"},
        {"component": "top_k_pairs", "value": int(data.get("top_k_pairs", 64)), "status": "reported"},
        {"component": "geometry_rank", "value": int(model.get("rank", 8)), "status": "reported"},
        {"component": "bridge_mask", "value": not bool(ablations.get("allow_bridge_core_leakage", False)), "status": "reported"},
        {"component": "dna_shape_channels", "value": not bool(ablations.get("disable_shape_channels", False)), "status": "reported"},
        {"component": "anchor_offsets", "value": ",".join(str(item) for item in DEFAULT_OFFSETS), "status": "reported"},
    ]
    frame = pd.DataFrame.from_records(rows)
    csv_path = output_dir / "sensitivity_summary.csv"
    md_path = output_dir / "sensitivity_summary.md"
    write_table(frame, csv_path)
    write_text(md_path, "# Sensitivity Summary\n\n" + _frame_to_markdown(frame) + "\n")
    return {"csv": str(csv_path), "markdown": str(md_path)}


def run_manuscript_diagnostics(config: dict[str, Any]) -> dict[str, str]:
    output_dir = ensure_dir(Path(config["paths"]["reports"]) / "manuscript_diagnostics")
    outputs = {}
    outputs.update({f"synthetic_{key}": value for key, value in run_synthetic_identifiability_report(output_dir).items()})
    outputs.update({f"sensitivity_{key}": value for key, value in write_sensitivity_summary(config, output_dir).items()})
    prediction_table = config.get("evaluation", {}).get("prediction_table")
    if prediction_table and Path(str(prediction_table)).exists():
        outputs.update({f"state_control_{key}": value for key, value in run_state_control_report(prediction_table, output_dir).items()})
    write_json(output_dir / "manifest.json", outputs)
    return outputs


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    safe = frame.copy().replace({np.nan: ""})
    header = "| " + " | ".join(str(column) for column in safe.columns) + " |"
    separator = "| " + " | ".join("---" for _ in safe.columns) + " |"
    rows = [
        "| " + " | ".join(_markdown_cell(value) for value in row) + " |"
        for row in safe.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _markdown_cell(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("\n", " ").replace("|", "\\|")
