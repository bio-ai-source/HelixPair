from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import torch

from helixpair.data import make_loader
from helixpair.io_utils import ensure_dir, write_json, write_table
from helixpair.training import _batch_to_device, _device_from_config, _forward_model, build_model


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def run_efficiency_profile(config: dict, bundle_path: str | Path, output_dir: str | Path, warmup_steps: int = 2, measure_steps: int = 5) -> dict[str, float]:
    output_dir = ensure_dir(output_dir)
    device = _device_from_config(config)
    model = build_model(config).to(device)
    model.eval()
    loader = make_loader(bundle_path, batch_size=int(config["training"].get("batch_size", 128)), shuffle=False)
    iterator = iter(loader)
    batch = _batch_to_device(next(iterator), device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(warmup_steps):
            _forward_model(model, batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        total_examples = 0
        for _ in range(measure_steps):
            outputs = _forward_model(model, batch)
            total_examples += batch["labels"].shape[0]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
    metrics = {
        "parameter_count": float(_parameter_count(model)),
        "examples_per_second": float(total_examples / max(elapsed, 1e-6)),
        "latency_per_batch_ms": float(1000.0 * elapsed / max(measure_steps, 1)),
        "peak_memory_mb": float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0,
    }
    write_json(output_dir / "efficiency.json", metrics)
    write_table(pd.DataFrame([metrics]), output_dir / "efficiency.parquet")
    return metrics
