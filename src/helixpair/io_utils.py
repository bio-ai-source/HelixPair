from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from helixpair.constants import PROJECT_ROOT
from helixpair.types import RunArtifacts, RunManifest


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if os.name == "nt":
        return candidate
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", str(path))
    if match:
        drive, rest = match.groups()
        return Path("/mnt") / drive.lower() / rest.replace("\\", "/")
    return candidate


def ensure_dir(path: str | Path) -> Path:
    directory = resolve_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    with resolve_path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_json(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_table(path: str | Path):
    import pandas as pd

    path = resolve_path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def write_table(frame, path: str | Path) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    elif path.suffix in {".tsv", ".txt"}:
        frame.to_csv(path, sep="\t", index=False)
    else:
        frame.to_csv(path, index=False)


def write_text(path: str | Path, text: str) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def timestamp_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    path = resolve_path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-").lower()


def split_token(split_name: str) -> str:
    token = stable_slug(split_name or "default")
    return token or "default"


def build_run_id(phase: str, scenario: str, split_name: str, seed: int, suffix: str = "") -> str:
    split_part = split_token(split_name)
    suffix_part = f"-{stable_slug(suffix)}" if suffix else ""
    return f"{phase}-{scenario}-{split_part}-seed{seed}-{timestamp_token()}{suffix_part}"


def timestamped_run_dir(
    phase: str,
    scenario: str,
    split_name: str,
    seed: int,
    run_id: str | None = None,
) -> Path:
    run_id = run_id or build_run_id(phase, scenario, split_name, seed)
    return ensure_dir(PROJECT_ROOT / "checkpoints" / phase / scenario / split_token(split_name) / f"seed_{seed}" / run_id)


def _checkpoints_root(config: dict[str, Any]) -> Path:
    return ensure_dir(config.get("paths", {}).get("checkpoints", PROJECT_ROOT / "checkpoints"))


def _results_root(config: dict[str, Any]) -> Path:
    results_cfg = config.get("results", {})
    return ensure_dir(results_cfg.get("root", config.get("paths", {}).get("results", PROJECT_ROOT / "results")))


def _ledger_path(config: dict[str, Any]) -> Path:
    results_cfg = config.get("results", {})
    return resolve_path(results_cfg.get("ledger_path", _results_root(config) / "run_ledger.jsonl"))


def ensure_commit_hash(project_root: str | Path) -> Path:
    project_root = resolve_path(project_root)
    repro_root = ensure_dir(project_root / "repro")
    commit_hash_path = repro_root / "commit_hash.txt"
    if commit_hash_path.exists() and commit_hash_path.read_text(encoding="utf-8").strip():
        return commit_hash_path
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            commit_hash_path.write_text(completed.stdout.strip() + "\n", encoding="utf-8")
    except Exception:
        pass
    if not commit_hash_path.exists():
        commit_hash_path.write_text("unknown\n", encoding="utf-8")
    return commit_hash_path


def init_run_artifacts(config: dict[str, Any], phase: str, scenario: str, seed: int, suffix: str = "") -> RunArtifacts:
    split_name = str(config.get("runtime", {}).get("split_name", "default"))
    split_dir = split_token(split_name)
    runtime_suffix = str(config.get("runtime", {}).get("run_suffix", "")).strip()
    effective_suffix = suffix or runtime_suffix
    run_id = build_run_id(phase, scenario, split_name, seed, suffix=effective_suffix)
    ensure_commit_hash(config.get("paths", {}).get("project_root", PROJECT_ROOT))
    checkpoints_root = _checkpoints_root(config)
    output_dir = ensure_dir(checkpoints_root / phase / scenario / split_dir / f"seed_{seed}" / run_id)
    reports_dir = ensure_dir(output_dir / "reports")
    figures_dir = ensure_dir(output_dir / "figures")
    tables_dir = ensure_dir(output_dir / "tables")
    logs_dir = ensure_dir(output_dir / "logs")
    results_root = _results_root(config)
    checkpoint_path = output_dir / f"{phase}.pt"
    prediction_root = results_root / "per_example_predictions" / phase / scenario / split_dir
    if effective_suffix:
        prediction_root = prediction_root / stable_slug(effective_suffix)
    return RunArtifacts(
        output_dir=output_dir,
        config_path=output_dir / "config.yaml",
        manifest_path=output_dir / "run_manifest.json",
        metrics_path=output_dir / "metrics.json",
        predictions_path=prediction_root / f"seed_{seed}.parquet",
        error_analysis_path=output_dir / "error_analysis.md",
        seed_manifest_path=output_dir / "seed_manifest.json",
        ledger_path=_ledger_path(config),
        reports_dir=reports_dir,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        logs_dir=logs_dir,
        checkpoint_path=checkpoint_path,
    )


def build_run_manifest(
    run_artifacts: RunArtifacts,
    config: dict[str, Any],
    phase: str,
    scenario: str,
    seed: int,
    tags: list[str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> RunManifest:
    split_name = str(config.get("runtime", {}).get("split_name", "default"))
    return RunManifest(
        run_id=run_artifacts.output_dir.name,
        phase=phase,
        scenario=scenario,
        split_name=split_name,
        seed=seed,
        started_at=utc_timestamp(),
        device=str(config.get("runtime", {}).get("device", "cpu")),
        config_path=str(resolve_path(run_artifacts.config_path)),
        output_dir=str(resolve_path(run_artifacts.output_dir)),
        tags=tags or [],
        inputs={key: str(value) for key, value in config.get("data", {}).items() if isinstance(value, (str, Path))},
        overrides=overrides or {},
    )


def record_run_ledger(path: str | Path, manifest: RunManifest, metrics: dict[str, Any] | None = None) -> None:
    payload = {
        "run_id": manifest.run_id,
        "phase": manifest.phase,
        "scenario": manifest.scenario,
        "split_name": manifest.split_name,
        "seed": manifest.seed,
        "started_at": manifest.started_at,
        "device": manifest.device,
        "config_path": manifest.config_path,
        "output_dir": manifest.output_dir,
        "inputs": manifest.inputs,
        "tags": manifest.tags,
        "overrides": manifest.overrides,
    }
    if metrics is not None:
        payload["metrics"] = metrics
    append_jsonl(path, payload)
