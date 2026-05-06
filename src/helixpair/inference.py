from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch

from helixpair.data import load_tensor_dict, make_loader
from helixpair.io_utils import read_table
from helixpair.training import _device_from_config, _forward_model, build_model


def load_model_for_inference(config: dict, checkpoint_path: str | Path):
    device = _device_from_config(config)
    model = build_model(config).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def predict_bundle(
    config: dict,
    checkpoint_path: str | Path,
    bundle_path: str | Path,
    metadata_path: str | Path | None = None,
    batch_transform: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
) -> pd.DataFrame:
    model, device = load_model_for_inference(config, checkpoint_path)
    loader = make_loader(bundle_path, batch_size=int(config["training"].get("batch_size", 128)), shuffle=False)
    metadata = read_table(metadata_path) if metadata_path and Path(metadata_path).exists() else None
    rows: list[dict] = []
    index = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            if batch_transform is not None:
                batch = batch_transform(batch)
            outputs = _forward_model(model, batch)
            full_score = outputs.usage_probability if outputs.usage_probability is not None else torch.sigmoid(
                -(outputs.intrinsic_monomer_energy + outputs.biochemical_residual)
            )
            availability_only = outputs.availability_only_probability
            for local_index in range(full_score.shape[0]):
                row = {
                    "bundle_index": index,
                    "label": float(batch["labels"][local_index].detach().cpu()),
                    "full_score": float(full_score[local_index].detach().cpu()),
                    "geometry_residual": float(outputs.geometry_residual[local_index].detach().cpu()),
                    "bridge_residual": float(outputs.bridge_residual[local_index].detach().cpu()),
                    "biochemical_residual": float(outputs.biochemical_residual[local_index].detach().cpu()),
                }
                if availability_only is not None:
                    row["availability_only_score"] = float(availability_only[local_index].detach().cpu())
                if outputs.state_gate is not None:
                    row["state_gate"] = float(outputs.state_gate[local_index].detach().cpu())
                    row["state_correction"] = float(outputs.state_correction[local_index].detach().cpu())
                if metadata is not None and index < len(metadata):
                    row.update(metadata.iloc[index].to_dict())
                rows.append(row)
                index += 1
    return pd.DataFrame.from_records(rows)


def bundle_shapes(bundle_path: str | Path) -> dict[str, tuple[int, ...]]:
    payload = load_tensor_dict(bundle_path)
    return {key: tuple(value.shape) for key, value in payload.items()}
