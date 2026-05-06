from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AnchorRecord:
    seq_id: str
    tf_id: str
    start: int
    end: int
    strand: str
    score: float
    offset_margin: int


@dataclass(slots=True)
class PairRecord:
    seq_id: str
    left_tf: str
    right_tf: str
    left_start: int
    left_end: int
    right_start: int
    right_end: int
    orientation: str
    center_distance: float
    edge_gap: float
    overlap_len: float
    coarse_additive_score: float


@dataclass(slots=True)
class AssetManifestRecord:
    dataset_id: str
    phase: str
    source_type: str
    url: str
    local_dir: str
    required: bool
    priority: int
    description: str
    version: str = ""
    subset_rule: str = ""
    status: str = "registered"
    downloaded_at: str = ""
    checksum: str = ""
    notes: str = ""


@dataclass(slots=True)
class RunArtifacts:
    output_dir: Path
    config_path: Path
    manifest_path: Path
    metrics_path: Path
    predictions_path: Path
    error_analysis_path: Path
    seed_manifest_path: Path
    ledger_path: Path
    reports_dir: Path
    figures_dir: Path
    tables_dir: Path
    logs_dir: Path
    checkpoint_path: Path


@dataclass(slots=True)
class RunManifest:
    run_id: str
    phase: str
    scenario: str
    split_name: str
    seed: int
    started_at: str
    device: str
    config_path: str
    output_dir: str
    tags: list[str] = field(default_factory=list)
    inputs: dict[str, str] = field(default_factory=dict)
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DataAcceptanceReport:
    dataset: str
    status: str
    summary: dict[str, Any] = field(default_factory=dict)
