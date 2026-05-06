from __future__ import annotations

from pathlib import Path

import pandas as pd

from helixpair.inference import predict_bundle
from helixpair.io_utils import ensure_dir, write_table

TARGET_CASE_PAIRS = {
    ("GLI3", "RFX3"),
    ("PROX1", "HOXA2"),
    ("PROX2", "HOXA2"),
}


def _target_case_pool(frame: pd.DataFrame) -> pd.DataFrame:
    if not {"left_tf", "right_tf"}.issubset(frame.columns):
        return frame
    local = frame.copy()
    pair_keys = list(zip(local["left_tf"].astype(str), local["right_tf"].astype(str)))
    mask = pd.Series([key in TARGET_CASE_PAIRS for key in pair_keys], index=local.index)
    targeted = local.loc[mask]
    return targeted if not targeted.empty else frame


def _select_spacing_dominant(frame: pd.DataFrame) -> pd.Series:
    frame = _target_case_pool(frame)
    ranked = frame.assign(spacing_score=frame["geometry_residual"].abs() - frame["bridge_residual"].abs())
    return ranked.sort_values("spacing_score", ascending=False).iloc[0]


def _select_composite_candidate(frame: pd.DataFrame) -> pd.Series:
    frame = _target_case_pool(frame)
    ranked = frame.assign(composite_score=frame["bridge_residual"].abs() + frame.get("composite_label", 0.0))
    return ranked.sort_values("composite_score", ascending=False).iloc[0]


def run_case_studies(
    config: dict,
    checkpoint_path: str | Path,
    bundle_path: str | Path,
    metadata_path: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    output_dir = ensure_dir(output_dir)
    frame = predict_bundle(config, checkpoint_path, bundle_path, metadata_path=metadata_path)
    spacing_case = _select_spacing_dominant(frame)
    composite_case = _select_composite_candidate(frame)

    spacing_report = "\n".join(
        [
            "# Case Study 1",
            "",
            f"- Left TF: {spacing_case.get('left_tf', '')}",
            f"- Right TF: {spacing_case.get('right_tf', '')}",
            f"- Geometry residual: {spacing_case['geometry_residual']:.4f}",
            f"- Bridge residual: {spacing_case['bridge_residual']:.4f}",
            f"- Gap: {spacing_case.get('edge_gap', 0.0)}",
            f"- Orientation: {spacing_case.get('orientation', '')}",
        ]
    )
    composite_report = "\n".join(
        [
            "# Case Study 2",
            "",
            f"- Left TF: {composite_case.get('left_tf', '')}",
            f"- Right TF: {composite_case.get('right_tf', '')}",
            f"- Geometry residual: {composite_case['geometry_residual']:.4f}",
            f"- Bridge residual: {composite_case['bridge_residual']:.4f}",
            f"- State gate: {composite_case.get('state_gate', float('nan'))}",
            f"- Composite label: {composite_case.get('composite_label', float('nan'))}",
        ]
    )
    (Path(output_dir) / "case_study_1.md").write_text(spacing_report, encoding="utf-8")
    (Path(output_dir) / "case_study_2.md").write_text(composite_report, encoding="utf-8")
    write_table(frame, Path(output_dir) / "case_study_predictions.parquet")
    return {
        "case_study_1": str(Path(output_dir) / "case_study_1.md"),
        "case_study_2": str(Path(output_dir) / "case_study_2.md"),
    }
