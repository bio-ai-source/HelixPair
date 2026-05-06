from __future__ import annotations

from pathlib import Path

import numpy as np

from helixpair.io_utils import read_table, split_token, write_table
from helixpair.sequence import build_pair_record, geometry_features, scan_pwm
from helixpair.splits import phase_split_name


def compute_helical_feature_frame(pair_frame, order: int = 2, bins: int = 8):
    import pandas as pd

    records = []
    for row in pair_frame.itertuples(index=False):
        features = geometry_features(
            center_distance=row.center_distance,
            edge_gap=row.edge_gap,
            overlap_len=row.overlap_len,
            orientation=row.orientation,
            order=order,
            bins=bins,
        )
        records.append(
            {
                "seq_id": row.seq_id,
                "pair_id": getattr(row, "pair_id", None),
                **{f"geom_{index}": float(value) for index, value in enumerate(features)},
            }
        )
    return pd.DataFrame.from_records(records)


def generate_anchor_table(window_frame, pwm_library: dict[str, np.ndarray], top_k: int = 24, cutoff: float = 1e-3):
    import pandas as pd

    anchors = []
    for row in window_frame.itertuples(index=False):
        for tf_id, pwm in pwm_library.items():
            for anchor in scan_pwm(row.sequence, pwm, tf_id=tf_id, top_k=top_k, cutoff=cutoff):
                anchors.append(
                    {
                        "seq_id": row.seq_id,
                        "tf_id": anchor.tf_id,
                        "start": anchor.start,
                        "end": anchor.end,
                        "strand": anchor.strand,
                        "score": anchor.score,
                        "offset_margin": anchor.offset_margin,
                    }
                )
    return pd.DataFrame.from_records(anchors)


def generate_pair_table(anchor_frame, top_k_pairs: int = 64):
    import pandas as pd

    pair_records = []
    for seq_id, group in anchor_frame.groupby("seq_id"):
        anchors = list(group.sort_values("score", ascending=False).itertuples(index=False))
        local = []
        for left_index, left in enumerate(anchors):
            for right in anchors[left_index + 1 :]:
                local.append(build_pair_record(seq_id, left, right))
        local.sort(key=lambda pair: pair.coarse_additive_score, reverse=True)
        for pair_id, pair in enumerate(local[:top_k_pairs]):
            pair_records.append(
                {
                    "seq_id": pair.seq_id,
                    "pair_id": f"{seq_id}::pair::{pair_id}",
                    "left_tf": pair.left_tf,
                    "right_tf": pair.right_tf,
                    "left_start": pair.left_start,
                    "left_end": pair.left_end,
                    "right_start": pair.right_start,
                    "right_end": pair.right_end,
                    "orientation": pair.orientation,
                    "center_distance": pair.center_distance,
                    "edge_gap": pair.edge_gap,
                    "overlap_len": pair.overlap_len,
                    "coarse_additive_score": pair.coarse_additive_score,
                }
            )
    return pd.DataFrame.from_records(pair_records)


def cache_geometry_features(pair_path: str | Path, output_path: str | Path, order: int = 2, bins: int = 8) -> None:
    pair_frame = read_table(pair_path)
    write_table(compute_helical_feature_frame(pair_frame, order=order, bins=bins), output_path)


def build_baseline_feature_tables(project_root: str | Path, scenario: str = "real", split_name: str = "default") -> dict[str, str]:
    project_root = Path(project_root)
    processed_root = project_root / "data_processed" / scenario
    requested_split_dir = split_token(split_name)
    output_root = project_root / "data_processed" / "baselines" / scenario / requested_split_dir
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for phase in ["phase2", "phase4", "phase5"]:
        effective_split_dir = split_token(phase_split_name(split_name, phase))
        phase_root = processed_root / phase / effective_split_dir
        for split_partition in ["train", "valid", "test"]:
            bundle_path = phase_root / f"{split_partition}_bundle.pt"
            if not bundle_path.exists():
                continue
            import torch
            import pandas as pd

            payload = torch.load(bundle_path, map_location="cpu", weights_only=False)
            frame = pd.DataFrame(
                {
                    "geom_0": payload["geometry_features"].float().numpy()[:, 0],
                    "geom_mean": payload["geometry_features"].float().mean(dim=1).numpy(),
                    "geom_std": payload["geometry_features"].float().std(dim=1).numpy(),
                    "interface_mass": payload["interface_tensor"].float().sum(dim=(1, 2)).numpy(),
                    "left_tf_id": payload["left_tf_id"].numpy(),
                    "right_tf_id": payload["right_tf_id"].numpy(),
                    "left_family_id": payload["left_family_id"].numpy(),
                    "right_family_id": payload["right_family_id"].numpy(),
                    "label": payload["labels"].numpy(),
                }
            )
            if "left_subfamily_id" in payload:
                frame["left_subfamily_id"] = payload["left_subfamily_id"].numpy()
            if "right_subfamily_id" in payload:
                frame["right_subfamily_id"] = payload["right_subfamily_id"].numpy()
            if "left_paralog_id" in payload:
                frame["left_paralog_id"] = payload["left_paralog_id"].numpy()
            if "right_paralog_id" in payload:
                frame["right_paralog_id"] = payload["right_paralog_id"].numpy()
            frame["effective_split"] = effective_split_dir
            if "availability" in payload:
                availability = payload["availability"].float().numpy()
                for index in range(availability.shape[1]):
                    frame[f"availability_{index}"] = availability[:, index]
            if "state_context" in payload:
                state_context = payload["state_context"].float().numpy()
                for index in range(state_context.shape[1]):
                    frame[f"state_{index}"] = state_context[:, index]
            out_path = output_root / f"{phase}_{split_partition}_features.parquet"
            frame.to_parquet(out_path, index=False)
            outputs[f"{phase}_{split_partition}"] = str(out_path)
    return outputs
