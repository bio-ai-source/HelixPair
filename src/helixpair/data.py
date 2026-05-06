from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from helixpair.io_utils import resolve_path


class TensorDictDataset(Dataset):
    def __init__(self, tensors: dict[str, torch.Tensor]):
        lengths = {value.shape[0] for value in tensors.values()}
        if len(lengths) != 1:
            raise ValueError(f"Tensor lengths do not match: {sorted(lengths)}")
        self.tensors = tensors
        self.length = next(iter(lengths))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.tensors.items()}


def load_tensor_dict(path: str | Path) -> dict[str, torch.Tensor]:
    path = resolve_path(path)
    if path.suffix == ".pt":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return {key: value if isinstance(value, torch.Tensor) else torch.as_tensor(value) for key, value in payload.items()}
    if path.suffix == ".npz":
        with np.load(path) as payload:
            return {key: torch.from_numpy(payload[key]) for key in payload.files}
    raise ValueError(f"Unsupported tensor bundle: {path}")


def make_loader(
    path: str | Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    persistent_workers = num_workers > 0
    return DataLoader(
        TensorDictDataset(load_tensor_dict(path)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def save_tensor_dict(path: str | Path, tensors: dict[str, Any]) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".pt":
        torch.save(tensors, path)
        return
    np.savez(
        path,
        **{key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value for key, value in tensors.items()},
    )
