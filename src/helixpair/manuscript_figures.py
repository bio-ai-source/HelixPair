from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches
from matplotlib.gridspec import GridSpecFromSubplotSpec
from sklearn.metrics import average_precision_score, roc_auc_score

from helixpair.io_utils import ensure_dir, read_table, write_text

COLORS = {
    "helixpair": "#1f3b5b",
    "availability": "#7aa95c",
    "additive": "#c98a4b",
    "external": "#4d8076",
    "accent": "#c8553d",
    "soft": "#d9e2ec",
    "mid": "#6b7c93",
    "dark": "#243b53",
    "gold": "#e0b04b",
}

ORIENTATION_ORDER = ["++", "+-", "-+", "--"]


def build_manuscript_figures(config: dict) -> dict[str, str]:
    _configure_theme()
    project_root = Path(config["paths"]["project_root"])
    figure_cfg = config.get("figures", {})
    main_dir = ensure_dir(figure_cfg.get("main_dir", project_root / "figures" / "main"))
    manifest_lines = [
        "# Manuscript Figure Manifest",
        "",
        "- Fig.1: conceptual synthesis of the planned method framing and identifiability rules.",
        "- Fig.2: phase2 biochemical results from real CAP-SELEX splits plus synthetic composite-vs-spacing recovery.",
        "- Fig.3: phase4 ablation behavior and a concrete reporter-sequence energy decomposition.",
        "- Fig.4: state-deployment benchmarks from phase45 hematopoiesis plus matched-availability analysis.",
        "- Fig.5: orthogonal validation panels built from real ENCODE overlap, SCREEN/cCRE enrichment, and held-out case studies.",
        "- Fig.6: geometry counterfactuals, interface minimal edits, ensemble consistency, and selective-risk analysis from reporter experiments.",
    ]
    outputs = {
        **_build_fig1(main_dir),
        **_build_fig2(project_root, main_dir),
        **_build_fig3(project_root, main_dir),
        **_build_fig4(project_root, main_dir),
        **_build_fig5(project_root, main_dir),
        **_build_fig6(project_root, main_dir),
    }
    manifest_path = main_dir / "figure_manifest.md"
    write_text(manifest_path, "\n".join(manifest_lines))
    outputs["manifest"] = str(manifest_path)
    return outputs


def _configure_theme() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#cbd2d9",
            "axes.labelcolor": COLORS["dark"],
            "xtick.color": COLORS["dark"],
            "ytick.color": COLORS["dark"],
            "grid.color": "#d9e2ec",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def _load_frame(path: str | Path) -> pd.DataFrame:
    frame = read_table(path)
    frame = frame.copy()
    frame.columns = [str(column).lstrip("\ufeff") for column in frame.columns]
    return frame


def _load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> dict[str, str]:
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, bbox_inches="tight", dpi=240)
    plt.close(figure)
    return {stem: str(pdf_path), f"{stem}_png": str(png_path)}


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
        ha="right",
        color=COLORS["dark"],
    )


def _box(ax: plt.Axes, x: float, y: float, w: float, h: float, text: str, facecolor: str) -> None:
    patch = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=COLORS["dark"],
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2.0,
        y + h / 2.0,
        textwrap.fill(text, width=18),
        ha="center",
        va="center",
        fontsize=9,
        color=COLORS["dark"],
    )


def _arrow(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float) -> None:
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops={"arrowstyle": "->", "lw": 1.4, "color": COLORS["dark"]},
    )


def _safe_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    mask = np.isfinite(labels) & np.isfinite(scores)
    labels = labels[mask]
    scores = scores[mask]
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    mask = np.isfinite(labels) & np.isfinite(scores)
    labels = labels[mask]
    scores = scores[mask]
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _bootstrap_metric(
    labels: np.ndarray,
    scores: np.ndarray,
    metric_name: str,
    bootstrap_rounds: int = 250,
    seed: int = 11,
) -> tuple[float, float, float]:
    metric_fn = _safe_auprc if metric_name == "AUPRC" else _safe_auroc
    point = metric_fn(labels, scores)
    if not math.isfinite(point):
        return point, float("nan"), float("nan")
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(bootstrap_rounds):
        indices = rng.integers(0, len(labels), len(labels))
        sample_labels = labels[indices]
        if np.unique(sample_labels).size < 2:
            continue
        boot.append(metric_fn(sample_labels, scores[indices]))
    if not boot:
        return point, float("nan"), float("nan")
    return point, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _best_runs(project_root: Path) -> pd.DataFrame:
    return _load_frame(project_root / "reports" / "experiment_docx_analysis" / "csv" / "derived" / "stage_best_runs.csv")


def _phase2_predictions(project_root: Path, split_name: str, scenario: str = "real") -> pd.DataFrame:
    return _load_frame(project_root / "results" / "per_example_predictions" / "phase2" / scenario / split_name / "seed_11.parquet")


def _pair_label(frame: pd.DataFrame) -> pd.Series:
    return frame["left_tf"].astype(str) + "::" + frame["right_tf"].astype(str)


def _state_mode(frame: pd.DataFrame) -> pd.Series:
    return frame["seq_id"].astype(str).str.split("::").str[-1]


def _build_fig1(output_dir: Path) -> dict[str, str]:
    figure = plt.figure(figsize=(14, 10))
    grid = figure.add_gridspec(2, 2, hspace=0.3, wspace=0.2)
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[0, 1])
    ax_c = figure.add_subplot(grid[1, 0])
    ax_d = figure.add_subplot(grid[1, 1])

    for axis in (ax_a, ax_b, ax_c, ax_d):
        axis.set_axis_off()
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)

    _panel_label(ax_a, "a")
    _box(ax_a, 0.04, 0.60, 0.24, 0.22, "In vitro biochemistry", "#e9f2f9")
    _box(ax_a, 0.38, 0.60, 0.24, 0.22, "In vivo state regulation", "#eef7e8")
    _box(ax_a, 0.72, 0.60, 0.24, 0.22, "Black-box interaction scoring", "#f9efe5")
    _box(ax_a, 0.22, 0.18, 0.56, 0.24, "HelixPair bridges zero-model calibration, pair grammar, and state deployment", "#f7e9e5")
    _arrow(ax_a, 0.16, 0.60, 0.40, 0.42)
    _arrow(ax_a, 0.50, 0.60, 0.50, 0.42)
    _arrow(ax_a, 0.84, 0.60, 0.60, 0.42)
    ax_a.text(0.5, 0.90, "Existing method landscape and HelixPair gap", ha="center", fontsize=12, fontweight="bold")

    _panel_label(ax_b, "b")
    modules = [
        ("Monomer calibration", 0.04, 0.64, "#e9f2f9"),
        ("Chemical potential", 0.28, 0.64, "#eef7e8"),
        ("Anchor refinement", 0.52, 0.64, "#f9efe5"),
        ("Geometry + bridge residuals", 0.04, 0.24, "#e6eef8"),
        ("Deployment gate", 0.42, 0.24, "#e9f5e7"),
        ("Partition / usage", 0.70, 0.24, "#faeee6"),
    ]
    for text, x, y, color in modules:
        _box(ax_b, x, y, 0.22, 0.16, text, color)
    _arrow(ax_b, 0.26, 0.72, 0.28, 0.72)
    _arrow(ax_b, 0.50, 0.72, 0.52, 0.72)
    _arrow(ax_b, 0.15, 0.64, 0.15, 0.40)
    _arrow(ax_b, 0.63, 0.64, 0.53, 0.40)
    _arrow(ax_b, 0.26, 0.32, 0.42, 0.32)
    _arrow(ax_b, 0.64, 0.32, 0.70, 0.32)
    ax_b.text(0.5, 0.90, "HelixPair module layout", ha="center", fontsize=12, fontweight="bold")

    _panel_label(ax_c, "c")
    _box(ax_c, 0.05, 0.64, 0.24, 0.18, "Null state\nE(seq) = 0", "#f3f4f6")
    _box(ax_c, 0.38, 0.64, 0.24, 0.18, "Monomer state\nE = E_left + E_right", "#e9f2f9")
    _box(ax_c, 0.71, 0.64, 0.24, 0.18, "Pair state\nE = monomer + geometry + bridge + state", "#eef7e8")
    _arrow(ax_c, 0.29, 0.73, 0.38, 0.73)
    _arrow(ax_c, 0.62, 0.73, 0.71, 0.73)
    ax_c.plot([0.12, 0.22], [0.30, 0.30], lw=6, color=COLORS["mid"])
    ax_c.plot([0.45, 0.55], [0.30, 0.30], lw=6, color=COLORS["helixpair"])
    ax_c.plot([0.78, 0.88], [0.30, 0.30], lw=6, color=COLORS["availability"])
    ax_c.text(0.17, 0.20, "Null occupancy", ha="center")
    ax_c.text(0.50, 0.20, "Intrinsic monomer energy", ha="center")
    ax_c.text(0.83, 0.20, "Residual pair grammar + deployment", ha="center")
    ax_c.text(0.5, 0.90, "Null / monomer / pair statistical states", ha="center", fontsize=12, fontweight="bold")

    _panel_label(ax_d, "d")
    _box(ax_d, 0.05, 0.62, 0.40, 0.22, "Reference-graph audit before any joint fitting", "#f9efe5")
    _box(ax_d, 0.55, 0.62, 0.40, 0.22, "Interface-masked bridge residuals", "#e9f2f9")
    _box(ax_d, 0.05, 0.20, 0.27, 0.20, "No odd cycles", "#eef7e8")
    _box(ax_d, 0.37, 0.20, 0.27, 0.20, "Freeze monomer if audit fails", "#fce8e6")
    _box(ax_d, 0.69, 0.20, 0.26, 0.20, "Leakage checks on bridge core", "#eef7e8")
    ax_d.plot([0.12, 0.20, 0.28], [0.52, 0.58, 0.52], color=COLORS["dark"], lw=1.8)
    ax_d.scatter([0.12, 0.20, 0.28], [0.52, 0.58, 0.52], s=55, color=COLORS["helixpair"])
    ax_d.plot([0.72, 0.83], [0.50, 0.50], color=COLORS["dark"], lw=6)
    ax_d.plot([0.77, 0.79], [0.50, 0.50], color=COLORS["accent"], lw=6)
    ax_d.text(0.78, 0.43, "interface-only residual window", ha="center", fontsize=9)
    ax_d.text(0.5, 0.90, "Leakage control and identifiability constraints", ha="center", fontsize=12, fontweight="bold")

    figure.suptitle("Fig.1 | Problem framing, architecture, and identifiability", fontsize=15, fontweight="bold", y=0.98)
    return _save_figure(figure, output_dir, "fig1_problem_architecture")


def _build_fig2(project_root: Path, output_dir: Path) -> dict[str, str]:
    figure = plt.figure(figsize=(15, 12))
    grid = figure.add_gridspec(3, 2, hspace=0.45, wspace=0.28)
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[0, 1])
    ax_c = figure.add_subplot(grid[1, 0])
    ax_d = figure.add_subplot(grid[1, 1])
    ax_e = figure.add_subplot(grid[2, 0])
    ax_f = figure.add_subplot(grid[2, 1])

    _panel_label(ax_a, "a")
    ax_a.set_axis_off()
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    _box(ax_a, 0.03, 0.60, 0.22, 0.18, "CAP-SELEX harmonization", "#e9f2f9")
    _box(ax_a, 0.30, 0.60, 0.22, 0.18, "Window / anchor candidates", "#eef7e8")
    _box(ax_a, 0.57, 0.60, 0.18, 0.18, "Phase I", "#f9efe5")
    _box(ax_a, 0.79, 0.60, 0.18, 0.18, "Phase II", "#f7e9e5")
    _box(ax_a, 0.18, 0.18, 0.64, 0.22, "Outputs: interaction discrimination, spacing/orientation recovery, leakage audits", "#f3f4f6")
    _arrow(ax_a, 0.25, 0.69, 0.30, 0.69)
    _arrow(ax_a, 0.52, 0.69, 0.57, 0.69)
    _arrow(ax_a, 0.75, 0.69, 0.79, 0.69)
    _arrow(ax_a, 0.66, 0.60, 0.50, 0.40)
    ax_a.text(0.5, 0.90, "CAP-SELEX data flow and training pipeline", ha="center", fontsize=12, fontweight="bold")

    interaction_records = []
    for split_name in ("default", "unseen_pair", "unseen_family"):
        frame = _phase2_predictions(project_root, split_name)
        labels = frame["label"].to_numpy(dtype=float)
        scores = frame["full_score"].to_numpy(dtype=float)
        for metric_name in ("AUPRC", "AUROC"):
            point, low, high = _bootstrap_metric(labels, scores, metric_name)
            interaction_records.append(
                {
                    "split_name": split_name,
                    "metric": metric_name,
                    "value": point,
                    "low": low,
                    "high": high,
                    "n": len(frame),
                }
            )
    interaction_df = pd.DataFrame(interaction_records)
    _panel_label(ax_b, "b")
    sns.barplot(
        data=interaction_df,
        x="split_name",
        y="value",
        hue="metric",
        palette=[COLORS["helixpair"], COLORS["gold"]],
        ax=ax_b,
    )
    for bar, (_, row) in zip(ax_b.patches, interaction_df.iterrows()):
        if math.isfinite(row["low"]) and math.isfinite(row["high"]):
            center_x = bar.get_x() + bar.get_width() / 2.0
            ax_b.plot([center_x, center_x], [row["low"], row["high"]], color=COLORS["dark"], lw=1.2)
    ax_b.set_ylim(0, 1.05)
    ax_b.set_xlabel("split")
    ax_b.set_ylabel("metric value")
    ax_b.set_title("Interaction discrimination with bootstrap CIs")
    ax_b.legend(title="")
    ax_b.set_xticklabels(
        [f"{tick.get_text()}\nN={int(interaction_df[interaction_df['split_name'] == tick.get_text()]['n'].iloc[0])}" for tick in ax_b.get_xticklabels()]
    )

    best_runs = _best_runs(project_root)
    spacing_records = []
    subset = best_runs[(best_runs["phase"] == "phase2") & (best_runs["scenario"] == "real")]
    subset = subset[subset["split_name"].isin(["default", "unseen_pair", "unseen_family"])]
    for _, row in subset.iterrows():
        metrics = _load_json(Path(row["output_dir"]) / "metrics.json")
        spacing_records.append({"split_name": row["split_name"], "metric": "Spacing EMD", "value": metrics.get("spacing_emd", np.nan)})
        spacing_records.append({"split_name": row["split_name"], "metric": "Spacing KL", "value": metrics.get("spacing_kl", np.nan)})
    spacing_df = pd.DataFrame(spacing_records)
    _panel_label(ax_c, "c")
    sns.barplot(
        data=spacing_df,
        x="split_name",
        y="value",
        hue="metric",
        palette=[COLORS["availability"], COLORS["accent"]],
        ax=ax_c,
    )
    ax_c.set_xlabel("split")
    ax_c.set_ylabel("distance")
    ax_c.set_title("Spacing / orientation recovery")
    ax_c.legend(title="")

    synthetic = _load_frame(project_root / "results" / "per_example_predictions" / "phase2" / "synthetic" / "default" / "seed_11.parquet")
    synthetic["grammar_type"] = np.where(synthetic["composite_target"].astype(float) > 0.5, "composite", "spacing")
    grammar_records = []
    for grammar_type, part in synthetic.groupby("grammar_type"):
        grammar_records.append(
            {
                "grammar_type": grammar_type,
                "auprc": _safe_auprc(part["label"].to_numpy(dtype=float), part["full_score"].to_numpy(dtype=float)),
                "n": len(part),
            }
        )
    grammar_df = pd.DataFrame(grammar_records)
    _panel_label(ax_d, "d")
    sns.barplot(data=grammar_df, x="grammar_type", y="auprc", palette=[COLORS["gold"], COLORS["helixpair"]], ax=ax_d)
    ax_d.set_ylim(0, 1.05)
    ax_d.set_xlabel("synthetic grammar subset")
    ax_d.set_ylabel("AUPRC")
    ax_d.set_title("Composite vs spacing grammar recovery")
    for _, row in grammar_df.iterrows():
        ax_d.text(
            list(grammar_df["grammar_type"]).index(row["grammar_type"]),
            min(row["auprc"] + 0.03, 1.02),
            f"N={row['n']}",
            ha="center",
            fontsize=9,
        )

    landscape_frame = _phase2_predictions(project_root, "default")
    left_tf, right_tf = _select_landscape_pair(landscape_frame)
    pair_subset = landscape_frame[(landscape_frame["left_tf"] == left_tf) & (landscape_frame["right_tf"] == right_tf)].copy()
    pair_subset = pair_subset.groupby(["edge_gap", "orientation"], as_index=False)["full_score"].mean()
    _panel_label(ax_e, "e")
    sns.lineplot(
        data=pair_subset,
        x="edge_gap",
        y="full_score",
        hue="orientation",
        hue_order=ORIENTATION_ORDER,
        palette="crest",
        marker="o",
        ax=ax_e,
    )
    ax_e.set_xlabel("edge gap")
    ax_e.set_ylabel("mean full score")
    ax_e.set_title(f"Representative pair landscape: {left_tf}::{right_tf}")
    ax_e.legend(title="orientation", ncol=2)

    audit_frame = landscape_frame[
        ["label", "bridge_score", "bridge_score_interface_shuffle", "bridge_score_core_rerandomized"]
    ].copy()
    audit_frame = audit_frame.rename(
        columns={
            "bridge_score": "observed",
            "bridge_score_interface_shuffle": "interface_shuffle",
            "bridge_score_core_rerandomized": "core_rerandomized",
        }
    )
    audit_long = audit_frame.melt(id_vars="label", var_name="audit_variant", value_name="bridge_value")
    audit_long["label_group"] = np.where(audit_long["label"].astype(float) > 0.5, "positive", "negative")
    summary = (
        audit_long.groupby(["audit_variant", "label_group"], as_index=False)
        .agg(mean=("bridge_value", "mean"), std=("bridge_value", "std"), count=("bridge_value", "count"))
    )
    summary["sem95"] = 1.96 * summary["std"] / np.sqrt(summary["count"].clip(lower=1))
    _panel_label(ax_f, "f")
    sns.pointplot(
        data=summary,
        x="audit_variant",
        y="mean",
        hue="label_group",
        palette=[COLORS["mid"], COLORS["accent"]],
        dodge=0.25,
        markers="o",
        linestyles="-",
        ax=ax_f,
    )
    for _, row in summary.iterrows():
        xpos = ["observed", "interface_shuffle", "core_rerandomized"].index(row["audit_variant"])
        xpos = xpos + (-0.08 if row["label_group"] == "negative" else 0.08)
        ax_f.plot([xpos, xpos], [row["mean"] - row["sem95"], row["mean"] + row["sem95"]], color=COLORS["dark"], lw=1.1)
    ax_f.set_xlabel("bridge audit condition")
    ax_f.set_ylabel("mean bridge score")
    ax_f.set_title("Bridge leakage audit")
    ax_f.legend(title="")

    figure.suptitle("Fig.2 | Biochemical-layer main results", fontsize=15, fontweight="bold", y=0.99)
    return _save_figure(figure, output_dir, "fig2_biochemical_results")


def _select_landscape_pair(frame: pd.DataFrame) -> tuple[str, str]:
    local = frame.copy()
    local["pair_label"] = _pair_label(local)
    ranked = []
    for pair_label, part in local.groupby("pair_label"):
        if len(part) < 100:
            continue
        mean_curve = part.groupby("edge_gap")["full_score"].mean()
        score_span = float(part["full_score"].max() - part["full_score"].min())
        ranked.append((score_span + float(mean_curve.std()), pair_label))
    if not ranked:
        pair_label = local["pair_label"].iloc[0]
    else:
        pair_label = max(ranked)[1]
    left_tf, right_tf = pair_label.split("::", 1)
    return left_tf, right_tf


def _first_harmonic_diagnostic(
    gaps: pd.Series | np.ndarray,
    values: pd.Series | np.ndarray,
    period: float = 10.5,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    x = np.asarray(gaps, dtype=float)
    y = np.asarray(values, dtype=float)
    centered = y - y.mean()
    design = np.column_stack(
        [
            np.cos(2.0 * np.pi * x / period),
            np.sin(2.0 * np.pi * x / period),
        ]
    )
    coef, *_ = np.linalg.lstsq(design, centered, rcond=None)
    fitted = design @ coef
    amplitude = float(np.sqrt((coef**2).sum()))
    denominator = max(float((centered**2).sum()), 1e-12)
    r2 = 1.0 - float(((centered - fitted) ** 2).sum()) / denominator
    return centered, fitted, amplitude, r2


def _build_fig3(project_root: Path, output_dir: Path) -> dict[str, str]:
    figure = plt.figure(figsize=(15, 10.5))
    grid = figure.add_gridspec(2, 2, hspace=0.38, wspace=0.25)
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[0, 1])
    ax_c = figure.add_subplot(grid[1, 0])
    ax_d = figure.add_subplot(grid[1, 1])

    full_frame = _load_frame(project_root / "results" / "per_example_predictions" / "phase4" / "ablation" / "unseen_pair" / "matched_full" / "seed_11.parquet")
    no_helical_frame = _load_frame(project_root / "results" / "per_example_predictions" / "phase4" / "ablation" / "unseen_pair" / "no_helical" / "seed_11.parquet")
    curve_records = []
    diagnostics: dict[str, tuple[float, float]] = {}
    for label, frame in (("full", full_frame), ("no_helical", no_helical_frame)):
        subset = frame[frame["label"].astype(float) > 0.5]
        if subset.empty:
            subset = frame
        grouped = subset.groupby("edge_gap", as_index=False)["geometry_residual"].mean().sort_values("edge_gap")
        centered, fitted, amplitude, r2 = _first_harmonic_diagnostic(grouped["edge_gap"], grouped["geometry_residual"])
        grouped["centered_geometry_residual"] = centered
        grouped["harmonic_fit"] = fitted
        grouped["variant"] = label
        curve_records.append(grouped)
        diagnostics[label] = (amplitude, r2)
    curve_df = pd.concat(curve_records, ignore_index=True)
    _panel_label(ax_a, "a")
    palette = {"full": COLORS["helixpair"], "no_helical": COLORS["gold"]}
    sns.lineplot(
        data=curve_df,
        x="edge_gap",
        y="centered_geometry_residual",
        hue="variant",
        palette=palette,
        marker="o",
        ax=ax_a,
    )
    for variant, color in palette.items():
        local = curve_df[curve_df["variant"] == variant].sort_values("edge_gap")
        ax_a.plot(local["edge_gap"], local["harmonic_fit"], linestyle="--", color=color, alpha=0.85, linewidth=1.4)
    ax_a.set_xlabel("edge gap")
    ax_a.set_ylabel("centered mean geometry residual")
    ax_a.set_title("Helical modulation diagnostic")
    ax_a.legend(title="")
    ax_a.text(
        0.03,
        0.04,
        (
            "1st-harmonic amplitude\n"
            f"full = {diagnostics['full'][0]:.3f} (R^2={diagnostics['full'][1]:.2f})\n"
            f"no_helical = {diagnostics['no_helical'][0]:.3f} (R^2={diagnostics['no_helical'][1]:.2f})"
        ),
        transform=ax_a.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#f8fafc", "edgecolor": "#d9e2ec"},
    )

    ablation = _load_frame(project_root / "reports" / "experiment_docx_analysis" / "csv" / "derived" / "phase4_ablation_latest_by_variant.csv")
    ablation = ablation[ablation["variant"].isin(["full", "no_helical", "geometry_only", "no_dna_shape", "no_availability", "no_state_gate"])].copy()
    full_auprc = float(ablation.loc[ablation["variant"] == "full", "auprc"].iloc[0])
    display_name = {
        "no_helical": "No helical basis",
        "geometry_only": "No bridge residual",
        "no_dna_shape": "No DNA shape",
        "no_availability": "No chemical potential",
        "no_state_gate": "No deployment gate",
    }
    ablation = ablation[ablation["variant"] != "full"].copy()
    ablation["delta"] = ablation["auprc"].astype(float) - full_auprc
    ablation["display"] = ablation["variant"].map(display_name)
    ablation = ablation.sort_values("delta")
    _panel_label(ax_b, "b")
    sns.barplot(data=ablation, x="delta", y="display", color=COLORS["accent"], ax=ax_b)
    ax_b.axvline(0.0, color=COLORS["dark"], lw=1.1)
    ax_b.set_xlabel("delta AUPRC vs full")
    ax_b.set_ylabel("")
    ax_b.set_title("Performance drop after module removal")

    _panel_label(ax_c, "c")
    ax_c.set_axis_off()
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, 1)
    _box(ax_c, 0.05, 0.65, 0.22, 0.15, "Family kernel", "#e9f2f9")
    _box(ax_c, 0.39, 0.65, 0.22, 0.15, "Subfamily kernel", "#eef7e8")
    _box(ax_c, 0.73, 0.65, 0.22, 0.15, "Paralog / TF specialization", "#f9efe5")
    _box(ax_c, 0.18, 0.25, 0.22, 0.15, "Shared prior", "#f3f4f6")
    _box(ax_c, 0.60, 0.25, 0.22, 0.15, "OOD transfer target", "#f3f4f6")
    _arrow(ax_c, 0.27, 0.65, 0.29, 0.40)
    _arrow(ax_c, 0.50, 0.65, 0.50, 0.40)
    _arrow(ax_c, 0.73, 0.65, 0.71, 0.40)
    _arrow(ax_c, 0.40, 0.33, 0.60, 0.33)
    ax_c.text(0.5, 0.90, "Family / subfamily / paralog kernel sharing", ha="center", fontsize=12, fontweight="bold")

    reporter = _load_frame(project_root / "results" / "per_example_predictions" / "phase5" / "real" / "reporter_pairranked" / "seed_11.parquet")
    target = reporter[(reporter["left_tf"] == "GLI3") & (reporter["right_tf"] == "RFX3") & (reporter["seq_id"].astype(str).str.endswith("high"))].copy()
    if target.empty:
        target = reporter.sort_values("full_score", ascending=False).head(1)
    point = target.iloc[0]
    decomp = pd.DataFrame(
        [
            {"component": "Monomer free energy", "value": float(point["monomer_free_energy"])},
            {"component": "Geometry residual", "value": float(point["geometry_residual"])},
            {"component": "Bridge residual", "value": float(point["bridge_score"])},
            {"component": "State correction", "value": float(point["state_correction"])},
        ]
    )
    decomp["sign"] = np.where(decomp["value"] >= 0.0, "positive", "negative")
    _panel_label(ax_d, "d")
    sns.barplot(
        data=decomp,
        x="value",
        y="component",
        hue="sign",
        palette={"positive": COLORS["helixpair"], "negative": COLORS["accent"]},
        dodge=False,
        ax=ax_d,
    )
    ax_d.axvline(0.0, color=COLORS["dark"], lw=1.0)
    ax_d.set_xlabel("component value")
    ax_d.set_ylabel("")
    ax_d.set_title(f"Single-sequence energy decomposition: {point['left_tf']}::{point['right_tf']}")
    ax_d.legend_.remove()
    ax_d.text(
        0.02,
        0.05,
        f"full score = {point['full_score']:.3f}\ndeployment gate = {point['state_gate']:.3f}",
        transform=ax_d.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#f8fafc", "edgecolor": "#d9e2ec"},
    )

    figure.suptitle("Fig.3 | Interpretability and module necessity", fontsize=15, fontweight="bold", y=0.99)
    return _save_figure(figure, output_dir, "fig3_interpretability_ablation")


def _build_fig4(project_root: Path, output_dir: Path) -> dict[str, str]:
    figure = plt.figure(figsize=(15, 10.5))
    grid = figure.add_gridspec(2, 2, hspace=0.40, wspace=0.30)
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[0, 1])
    ax_c = figure.add_subplot(grid[1, 0])
    ax_d = figure.add_subplot(grid[1, 1])

    _panel_label(ax_a, "a")
    ax_a.set_axis_off()
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    _box(ax_a, 0.04, 0.62, 0.22, 0.18, "State encoder", "#e9f2f9")
    _box(ax_a, 0.36, 0.62, 0.24, 0.18, "Chemical potential head", "#eef7e8")
    _box(ax_a, 0.70, 0.62, 0.24, 0.18, "Deployment gate", "#f9efe5")
    _box(ax_a, 0.20, 0.18, 0.60, 0.20, "Shared biochemical grammar deployed onto new states", "#f3f4f6")
    _arrow(ax_a, 0.26, 0.71, 0.36, 0.71)
    _arrow(ax_a, 0.60, 0.71, 0.70, 0.71)
    _arrow(ax_a, 0.82, 0.62, 0.62, 0.38)
    _arrow(ax_a, 0.48, 0.62, 0.42, 0.38)
    ax_a.text(0.5, 0.90, "State encoder + chemical potential + gate", ha="center", fontsize=12, fontweight="bold")

    benchmark = _load_frame(project_root / "reports" / "external_phase45" / "phase45_benchmark_summary.csv")
    benchmark = benchmark[(benchmark["phase"] == "phase4") & (benchmark["split_name"].isin(["default", "unseen_pair", "unseen_state"]))].copy()
    benchmark_records = []
    for _, row in benchmark.iterrows():
        n_rows = len(_load_frame(project_root / "results" / "per_example_predictions" / "phase4" / "phase45_hematopoiesis" / row["split_name"] / "seed_11.parquet"))
        benchmark_records.extend(
            [
                {"split_name": row["split_name"], "method": "HelixPair", "value": row["helixpair_auprc"], "n": n_rows},
                {"split_name": row["split_name"], "method": "Availability only", "value": row["availability_only_auprc"], "n": n_rows},
                {"split_name": row["split_name"], "method": "Formal ranking baseline", "value": row["best_measured_baseline_auprc"], "n": n_rows},
            ]
        )
    benchmark_df = pd.DataFrame(benchmark_records)
    _panel_label(ax_b, "b")
    sns.barplot(
        data=benchmark_df,
        x="split_name",
        y="value",
        hue="method",
        palette=[COLORS["helixpair"], COLORS["availability"], COLORS["external"]],
        ax=ax_b,
    )
    ax_b.set_ylim(0, 1.0)
    ax_b.set_xlabel("phase45 deployment split")
    ax_b.set_ylabel("AUPRC")
    ax_b.set_title("State deployment benchmark summary")
    ax_b.legend(title="")
    ax_b.set_xticklabels(
        [f"{tick.get_text()}\nN={int(benchmark_df[benchmark_df['split_name'] == tick.get_text()]['n'].iloc[0])}" for tick in ax_b.get_xticklabels()]
    )

    unseen_state = _load_frame(project_root / "results" / "per_example_predictions" / "phase4" / "phase45_hematopoiesis" / "unseen_state" / "seed_11.parquet")
    unseen_state["availability_stratum_num"] = pd.to_numeric(unseen_state["availability_stratum"], errors="coerce")
    strata_records = []
    for stratum, part in unseen_state.groupby("availability_stratum_num"):
        labels = part["label"].to_numpy(dtype=float)
        full = _safe_auprc(labels, part["full_score"].to_numpy(dtype=float))
        avail = _safe_auprc(labels, part["availability_only_score"].to_numpy(dtype=float))
        strata_records.extend(
            [
                {"stratum": stratum, "method": "HelixPair", "auprc": full, "n": len(part)},
                {"stratum": stratum, "method": "Availability only", "auprc": avail, "n": len(part)},
            ]
        )
    strata_df = pd.DataFrame(strata_records).sort_values(["stratum", "method"])
    _panel_label(ax_c, "c")
    sns.lineplot(
        data=strata_df,
        x="stratum",
        y="auprc",
        hue="method",
        palette=[COLORS["helixpair"], COLORS["availability"]],
        marker="o",
        ax=ax_c,
    )
    ax_c.set_xlabel("matched availability stratum")
    ax_c.set_ylabel("AUPRC")
    ax_c.set_title("Availability-matched comparison on unseen_state")
    ax_c.legend(title="")
    counts = strata_df.groupby("stratum")["n"].max()
    for stratum, count in counts.items():
        ax_c.text(stratum, ax_c.get_ylim()[0] + 0.02, f"N={int(count)}", ha="center", fontsize=8)

    default_phase45 = _load_frame(project_root / "results" / "per_example_predictions" / "phase4" / "phase45_hematopoiesis" / "default" / "seed_11.parquet")
    default_phase45["actual_state"] = default_phase45["state_label"].astype(str).str.split("::").str[1]
    default_phase45["pair_label"] = _pair_label(default_phase45)
    state_order = default_phase45["actual_state"].value_counts().head(6).index.tolist()
    pair_order = default_phase45["pair_label"].value_counts().head(8).index.tolist()
    heatmap = (
        default_phase45[default_phase45["actual_state"].isin(state_order) & default_phase45["pair_label"].isin(pair_order)]
        .groupby(["actual_state", "pair_label"], as_index=False)["full_score"]
        .mean()
        .pivot(index="actual_state", columns="pair_label", values="full_score")
        .reindex(index=state_order, columns=pair_order)
    )
    _panel_label(ax_d, "d")
    sns.heatmap(heatmap, cmap="crest", ax=ax_d, cbar_kws={"label": "mean full score"})
    ax_d.set_xlabel("TF pair")
    ax_d.set_ylabel("state")
    ax_d.set_title("Deployment heatmap across top states and pairs")

    figure.suptitle("Fig.4 | State deployment results", fontsize=15, fontweight="bold", y=0.99)
    return _save_figure(figure, output_dir, "fig4_state_deployment")


def _build_fig5(project_root: Path, output_dir: Path) -> dict[str, str]:
    figure = plt.figure(figsize=(15, 10.5))
    grid = figure.add_gridspec(2, 2, hspace=0.40, wspace=0.30)
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[0, 1])
    ax_c = figure.add_subplot(grid[1, 0])
    ax_d = figure.add_subplot(grid[1, 1])

    orthogonal_root = project_root / "reports" / "orthogonal_validation"
    encode_summary = _load_frame(orthogonal_root / "encode_overlap_summary.csv")
    overlap_display = {
        "any_tf_overlap_enrichment_positive_vs_negative": "Any TF peak overlap",
        "both_tf_overlap_enrichment_positive_vs_negative": "Both TF peaks overlap",
    }
    overlap_records = []
    for _, row in encode_summary.iterrows():
        label = overlap_display.get(str(row["metric"]), str(row["metric"]))
        overlap_records.extend(
            [
                {"metric": label, "group": "Supported", "rate": float(row["positive_overlap_rate"]), "n": float(row["n_rows"])},
                {"metric": label, "group": "Unsupported", "rate": float(row["negative_overlap_rate"]), "n": float(row["n_rows"])},
            ]
        )
    overlap_df = pd.DataFrame(overlap_records)
    _panel_label(ax_a, "a")
    sns.barplot(
        data=overlap_df,
        x="metric",
        y="rate",
        hue="group",
        palette=[COLORS["accent"], COLORS["mid"]],
        ax=ax_a,
    )
    ax_a.set_ylim(0, 0.35)
    ax_a.set_xlabel("")
    ax_a.set_ylabel("overlap rate")
    ax_a.set_title("ENCODE ChIP-seq overlap on covered public cCRE windows")
    ax_a.legend(title="")
    for index, row in encode_summary.reset_index(drop=True).iterrows():
        ax_a.text(index, 0.32, f"N={int(row['n_rows'])}", ha="center", fontsize=8, color=COLORS["dark"])

    ccre_class = _load_frame(orthogonal_root / "ccre_enrichment_by_class.csv")
    _panel_label(ax_b, "b")
    sns.barplot(
        data=ccre_class,
        x="element_type",
        y="mean_phase2_biochemical_probability",
        palette=[COLORS["gold"], COLORS["helixpair"]],
        ax=ax_b,
    )
    ax_b.set_ylim(0, 1.0)
    ax_b.set_xlabel("SCREEN cCRE class")
    ax_b.set_ylabel("mean Phase2 biochemical probability")
    ax_b.set_title("CAP-SELEX grammar transfer across cCRE classes")
    for index, row in ccre_class.reset_index(drop=True).iterrows():
        ax_b.text(
            index,
            float(row["mean_phase2_biochemical_probability"]) + 0.04,
            f"N={int(row['n_states'])}\nacc={float(row['correct_rate']):.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLORS["dark"],
        )

    pair_summary = _load_frame(project_root / "reports" / "external_phase45" / "hematopoiesis_pair_group_summary.csv")
    pair_summary["norm_pair"] = pair_summary["pair_transfer_group"].astype(str).map(lambda value: "::".join(sorted(value.split("::"))))
    default_phase45 = _load_frame(project_root / "results" / "per_example_predictions" / "phase4" / "phase45_hematopoiesis" / "default" / "seed_11.parquet")
    default_phase45["norm_pair"] = default_phase45.apply(
        lambda row: "::".join(sorted([str(row["left_tf"]), str(row["right_tf"])])),
        axis=1,
    )
    available_pairs = set(default_phase45["norm_pair"])
    pair_margin = (
        pair_summary[pair_summary["norm_pair"].isin(available_pairs)]
        .groupby("norm_pair", as_index=False)
        .agg(activity_margin=("activity_margin", "max"), group_size=("group_size", "max"))
        .sort_values("activity_margin", ascending=False)
        .head(10)
    )
    if pair_margin.empty:
        case_pair = default_phase45["norm_pair"].iloc[0]
    else:
        ranked_case = pair_margin[pair_margin["norm_pair"].map(lambda pair: default_phase45[default_phase45["norm_pair"] == pair]["edge_gap"].nunique() >= 4)]
        case_pair = (ranked_case if not ranked_case.empty else pair_margin).iloc[0]["norm_pair"]
    case_frame = default_phase45[default_phase45["norm_pair"] == case_pair].copy()
    case_frame["actual_state"] = case_frame["state_label"].astype(str).str.split("::").str[1]
    case_frame["delta_vs_availability"] = case_frame["full_score"].astype(float) - case_frame["availability_only_score"].astype(float)
    case_frame["label_group"] = np.where(case_frame["label"].astype(float) > 0.5, "supported", "unsupported")
    _panel_label(ax_c, "c")
    sns.scatterplot(
        data=case_frame,
        x="edge_gap",
        y="delta_vs_availability",
        hue="actual_state",
        style="label_group",
        s=80,
        ax=ax_c,
    )
    sns.lineplot(
        data=case_frame.sort_values("edge_gap"),
        x="edge_gap",
        y="delta_vs_availability",
        hue="actual_state",
        estimator=None,
        units="seq_id",
        legend=False,
        alpha=0.35,
        ax=ax_c,
    )
    ax_c.axhline(0.0, color=COLORS["dark"], lw=1.0)
    ax_c.set_xlabel("edge gap")
    ax_c.set_ylabel("full score - availability score")
    ax_c.set_title(f"Spacing/helical case study: {case_pair}")
    ax_c.legend(title="")

    reporter = _load_frame(project_root / "results" / "per_example_predictions" / "phase5" / "real" / "reporter_pairranked" / "seed_11.parquet")
    reporter = reporter[(reporter["left_tf"] == "GLI3") & (reporter["right_tf"] == "RFX3")].copy()
    reporter["state_mode"] = pd.Categorical(_state_mode(reporter), categories=["low", "high"], ordered=True)
    reporter["sequence_tag"] = reporter["seq_id"].astype(str).str.extract(r"sequence__(\d+)")[0].fillna(reporter["seq_id"].astype(str))
    _panel_label(ax_d, "d")
    sns.lineplot(
        data=reporter.sort_values(["sequence_tag", "state_mode"]),
        x="state_mode",
        y="full_score",
        hue="sequence_tag",
        marker="o",
        linewidth=2.0,
        palette="Set2",
        ax=ax_d,
    )
    ax_d.set_xlabel("reporter state")
    ax_d.set_ylabel("full score")
    ax_d.set_title("Interface/state-switch case study: GLI3::RFX3")
    ax_d.legend(title="sequence")
    for sequence_tag, part in reporter.groupby("sequence_tag"):
        if {"low", "high"}.issubset(set(part["state_mode"].astype(str))):
            high_value = float(part.loc[part["state_mode"].astype(str) == "high", "full_score"].iloc[0])
            low_value = float(part.loc[part["state_mode"].astype(str) == "low", "full_score"].iloc[0])
            ax_d.text(
                1.02,
                high_value,
                f"S{sequence_tag} Δ={high_value - low_value:.2f}",
                fontsize=8,
                color=COLORS["dark"],
            )

    figure.suptitle("Fig.5 | Orthogonal validation and reporter-state case studies", fontsize=15, fontweight="bold", y=0.99)
    return _save_figure(figure, output_dir, "fig5_orthogonal_validation")


def _build_fig6(project_root: Path, output_dir: Path) -> dict[str, str]:
    figure = plt.figure(figsize=(15, 10.5))
    grid = figure.add_gridspec(2, 2, hspace=0.42, wspace=0.30)
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[0, 1])
    ax_c = figure.add_subplot(grid[1, 0])
    ax_d = figure.add_subplot(grid[1, 1])

    counterfactual_root = project_root / "reports" / "counterfactuals"
    geometry = _load_frame(counterfactual_root / "geometry_counterfactual.csv")
    _panel_label(ax_a, "a")
    sns.lineplot(
        data=geometry.sort_values(["state_mode", "gap"]),
        x="gap",
        y="full_score",
        hue="state_mode",
        marker="o",
        palette=[COLORS["mid"], COLORS["accent"]],
        ax=ax_a,
    )
    ax_a.set_xlabel("edited edge gap")
    ax_a.set_ylabel("full score")
    ax_a.set_title("Geometry counterfactual across matched high/low states")
    ax_a.legend(title="state")

    interface = _load_frame(counterfactual_root / "interface_minimal_edit.csv").copy()
    interface["edit_label"] = (
        interface["ref_base"].astype(str)
        + interface["position"].astype(int).astype(str)
        + interface["alt_base"].astype(str)
    )
    interface_top = interface.sort_values("margin_drop", ascending=False).head(12).iloc[::-1]
    _panel_label(ax_b, "b")
    sns.barplot(data=interface_top, x="margin_drop", y="edit_label", color=COLORS["accent"], ax=ax_b)
    ax_b.set_xlabel("high-low margin drop")
    ax_b.set_ylabel("single-base edit")
    ax_b.set_title("Interface minimal-edit counterfactuals")

    ensemble = _load_frame(counterfactual_root / "ensemble_uncertainty.csv").copy()
    ensemble["sample_tag"] = (
        ensemble["left_tf"].astype(str)
        + "::"
        + ensemble["right_tf"].astype(str)
        + " "
        + ensemble["seq_id"].astype(str).str.extract(r"sequence__(\d+)")[0].fillna("?")
    )
    _panel_label(ax_c, "c")
    positions = np.arange(len(ensemble))
    colors = [COLORS["accent"] if bool(value) else COLORS["mid"] for value in ensemble["correct"].astype(bool)]
    ax_c.scatter(positions, ensemble["ensemble_mean"], s=90, c=colors)
    ax_c.errorbar(
        positions,
        ensemble["ensemble_mean"],
        yerr=ensemble["ensemble_std"],
        fmt="none",
        ecolor=COLORS["dark"],
        elinewidth=1.2,
        capsize=3,
    )
    ax_c.set_xticks(positions)
    ax_c.set_xticklabels(ensemble["sample_tag"], rotation=15, ha="right")
    ax_c.set_ylabel("ensemble mean score")
    ax_c.set_title("Ensemble consistency across shared reporter constructs")

    selective = _load_frame(counterfactual_root / "selective_risk_curve.csv")
    _panel_label(ax_d, "d")
    sns.lineplot(data=selective, x="coverage", y="accuracy", marker="o", color=COLORS["helixpair"], ax=ax_d)
    sns.lineplot(data=selective, x="coverage", y="selective_risk", marker="o", color=COLORS["accent"], ax=ax_d)
    ax_d.set_ylim(0, 1.05)
    ax_d.set_xlabel("coverage")
    ax_d.set_ylabel("metric value")
    ax_d.set_title("Accuracy-coverage and selective risk")
    ax_d.legend(["accuracy", "selective risk"], title="")

    figure.suptitle("Fig.6 | Counterfactuals, uncertainty, and selective risk", fontsize=15, fontweight="bold", y=0.99)
    return _save_figure(figure, output_dir, "fig6_geometry_uncertainty")
