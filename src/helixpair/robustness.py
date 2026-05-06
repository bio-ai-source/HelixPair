from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from helixpair.inference import predict_bundle
from helixpair.io_utils import ensure_dir, write_json, write_table
from helixpair.metrics import binary_classification_metrics


def run_robustness_assays(
    config: dict,
    checkpoint_path: str | Path,
    bundle_path: str | Path,
    output_dir: str | Path,
    metadata_path: str | Path | None = None,
) -> pd.DataFrame:
    output_dir = ensure_dir(output_dir)
    assays = []

    def identity(batch):
        return batch

    def availability_noise(level: float):
        def _transform(batch):
            if "availability" in batch:
                noise = torch.randn_like(batch["availability"]) * level
                batch = dict(batch)
                batch["availability"] = torch.clamp(batch["availability"] + noise, min=0.0)
            return batch

        return _transform

    def state_corruption(batch):
        if "state_context" in batch:
            batch = dict(batch)
            permutation = torch.randperm(batch["state_context"].shape[-1], device=batch["state_context"].device)
            batch["state_context"] = batch["state_context"][:, permutation]
        return batch

    def anchor_shift(batch):
        batch = dict(batch)
        batch["left_anchor_offsets"] = torch.roll(batch["left_anchor_offsets"], shifts=1, dims=1)
        batch["right_anchor_offsets"] = torch.roll(batch["right_anchor_offsets"], shifts=-1, dims=1)
        return batch

    def interface_shuffle(batch):
        if "interface_tensor_shuffle" in batch:
            batch = dict(batch)
            batch["interface_tensor"] = batch["interface_tensor_shuffle"]
        return batch

    def interface_shuffle_stress(batch):
        batch = dict(batch)
        if "interface_tensor_core_rerandomized" in batch:
            batch["interface_tensor"] = batch["interface_tensor_core_rerandomized"]
        elif "interface_tensor_shuffle" in batch:
            batch["interface_tensor"] = batch["interface_tensor_shuffle"]
        if "left_anchor_offsets" in batch:
            batch["left_anchor_offsets"] = torch.roll(batch["left_anchor_offsets"], shifts=2, dims=1)
        if "right_anchor_offsets" in batch:
            batch["right_anchor_offsets"] = torch.roll(batch["right_anchor_offsets"], shifts=-2, dims=1)
        return batch

    transforms = {
        "baseline": identity,
        "availability_noise_0.05": availability_noise(0.05),
        "availability_noise_0.10": availability_noise(0.10),
        "availability_noise_0.20": availability_noise(0.20),
        "state_corruption": state_corruption,
        "anchor_shift": anchor_shift,
        "interface_shuffle": interface_shuffle,
        "interface_shuffle_stress": interface_shuffle_stress,
    }

    for name, transform in transforms.items():
        frame = predict_bundle(config, checkpoint_path, bundle_path, metadata_path=metadata_path, batch_transform=transform)
        metrics = binary_classification_metrics(frame["label"], frame["full_score"])
        assays.append({"assay": name, **metrics})
    result = pd.DataFrame.from_records(assays)
    write_table(result, output_dir / "robustness_metrics.parquet")
    write_json(output_dir / "robustness_metrics.json", result.to_dict(orient="records"))
    return result
