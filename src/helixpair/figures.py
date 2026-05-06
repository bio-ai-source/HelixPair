from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from helixpair.io_utils import read_table


def make_main_results_figure(metrics_path: str | Path, output_path: str | Path) -> None:
    metrics = pd.read_json(metrics_path)
    if "method" not in metrics.columns:
        metrics = pd.DataFrame(metrics)
    melted = metrics.melt(id_vars=["method"], value_vars=["auprc", "auroc", "ece"], var_name="metric", value_name="value")
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(data=melted, x="metric", y="value", hue="method", ax=axis)
    axis.set_title("HelixPair Main Metrics")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def make_spacing_landscape_figure(landscape_path: str | Path, output_path: str | Path) -> None:
    landscape = read_table(landscape_path)
    sns.set_theme(style="ticks")
    figure, axis = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=landscape, x="gap", y="score", hue="method", style="orientation", ax=axis)
    axis.set_title("Spacing / Orientation Landscape")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def make_robustness_figure(metrics_path: str | Path, output_path: str | Path) -> None:
    frame = read_table(metrics_path)
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(9, 5))
    sns.barplot(data=frame, x="assay", y="auprc", ax=axis)
    axis.set_title("Robustness AUPRC")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
