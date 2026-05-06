from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str((ROOT / "src").resolve()))

from helixpair.config import dump_config, load_config  # noqa: E402
from helixpair.io_utils import ensure_dir, write_table, write_text  # noqa: E402
from helixpair.training import train_phase  # noqa: E402


SCENARIO = "phase45_hematopoiesis"
DEFAULT_SPLITS = ("default", "unseen_pair", "unseen_state")
DEFAULT_SEEDS = (11, 17, 23, 29, 47)
DEFAULT_PHASES = ("phase4", "phase5")
REPORT_ROOT = ROOT / "reports" / "external_phase45" / "formal_multiseed"


def _latest_checkpoint(phase: str, split_name: str, seed: int) -> Path | None:
    root = ROOT / "checkpoints" / phase / SCENARIO / split_name / f"seed_{seed}"
    if not root.exists():
        return None
    candidates = [path / f"{phase}.pt" for path in root.iterdir() if path.is_dir() and (path / f"{phase}.pt").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.as_posix()))


def _phase_config_path(phase: str) -> Path:
    group = "phase4_usage" if phase == "phase4" else "phase5_functional"
    return ROOT / "configs" / group / f"{SCENARIO}.yaml"


def _phase_bundle_root(phase: str, split_name: str) -> Path:
    return ROOT / "data_processed" / SCENARIO / phase / split_name


def _prediction_path(phase: str, split_name: str, seed: int) -> Path:
    return ROOT / "results" / "per_example_predictions" / phase / SCENARIO / split_name / f"seed_{seed}.parquet"


def _inventory_row(phase: str, split_name: str, seed: int, overwrite: bool) -> dict[str, object]:
    checkpoint = _latest_checkpoint(phase, split_name, seed)
    prediction_path = _prediction_path(phase, split_name, seed)
    bundle_root = _phase_bundle_root(phase, split_name)
    train_bundle = bundle_root / "train_bundle.pt"
    valid_bundle = bundle_root / "valid_bundle.pt"
    previous_phase_checkpoint = _latest_checkpoint("phase4", split_name, seed) if phase == "phase5" else None

    missing_artifacts: list[str] = []
    if checkpoint is None:
        missing_artifacts.append("checkpoint")
    if not prediction_path.exists():
        missing_artifacts.append("per_example_predictions")
    if not train_bundle.exists():
        missing_artifacts.append("train_bundle")
    if not valid_bundle.exists():
        missing_artifacts.append("valid_bundle")
    if phase == "phase5" and previous_phase_checkpoint is None:
        missing_artifacts.append("phase4_warm_start_checkpoint")

    status = "complete"
    blocked_reason = ""
    ready_for_training = False

    if overwrite:
        if not train_bundle.exists() or not valid_bundle.exists():
            status = "blocked_missing_bundles"
            blocked_reason = "missing_bundles"
        elif phase == "phase5" and previous_phase_checkpoint is None:
            status = "blocked_missing_phase4_warm_start"
            blocked_reason = "missing_phase4_warm_start"
        else:
            status = "ready_to_train"
            ready_for_training = True
    elif checkpoint is not None and prediction_path.exists():
        status = "complete"
    elif checkpoint is not None and not prediction_path.exists():
        status = "needs_prediction_regeneration"
        blocked_reason = "missing_predictions_only"
    elif not train_bundle.exists() or not valid_bundle.exists():
        status = "blocked_missing_bundles"
        blocked_reason = "missing_bundles"
    elif phase == "phase5" and previous_phase_checkpoint is None:
        status = "blocked_missing_phase4_warm_start"
        blocked_reason = "missing_phase4_warm_start"
    else:
        status = "ready_to_train"
        ready_for_training = True

    return {
        "phase": phase,
        "split_name": split_name,
        "seed": int(seed),
        "status": status,
        "checkpoint_exists": checkpoint is not None,
        "checkpoint": str(checkpoint) if checkpoint is not None else "",
        "prediction_exists": prediction_path.exists(),
        "prediction_path": str(prediction_path),
        "train_bundle_exists": train_bundle.exists(),
        "valid_bundle_exists": valid_bundle.exists(),
        "previous_phase_checkpoint_exists": previous_phase_checkpoint is not None,
        "previous_phase_checkpoint": str(previous_phase_checkpoint) if previous_phase_checkpoint is not None else "",
        "ready_for_training": ready_for_training,
        "blocked_reason": blocked_reason,
        "missing_artifacts": ",".join(missing_artifacts),
        "missing_artifact_count": int(len(missing_artifacts)),
    }


def collect_artifact_inventory(
    *,
    splits: list[str] | tuple[str, ...],
    seeds: list[int] | tuple[int, ...],
    phases: list[str] | tuple[str, ...],
    overwrite: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split_name in [str(value) for value in splits]:
        for seed in [int(value) for value in seeds]:
            for phase in [str(value) for value in phases]:
                rows.append(_inventory_row(phase, split_name, seed, bool(overwrite)))
    return rows


def write_inventory_reports(
    rows: list[dict[str, object]],
    *,
    report_root: str | Path = REPORT_ROOT,
) -> dict[str, object]:
    report_root = ensure_dir(report_root)
    frame = pd.DataFrame.from_records(rows).sort_values(["split_name", "phase", "seed"]).reset_index(drop=True)
    missing = frame.loc[frame["status"].astype(str) != "complete"].copy()
    blocked = frame.loc[frame["status"].astype(str).str.startswith("blocked_")].copy()
    ready = frame.loc[frame["ready_for_training"].astype(bool)].copy()

    inventory_csv = report_root / "artifact_inventory.csv"
    missing_csv = report_root / "missing_artifacts.csv"
    summary_json = report_root / "inventory_summary.json"
    summary_markdown = report_root / "inventory_summary.md"
    write_table(frame, inventory_csv)
    write_table(missing, missing_csv)

    status_counts = {str(key): int(value) for key, value in frame["status"].value_counts().sort_index().to_dict().items()}
    summary_payload = {
        "scenario": SCENARIO,
        "total_requested_rows": int(len(frame)),
        "complete_rows": int((frame["status"].astype(str) == "complete").sum()),
        "ready_to_train_rows": int(len(ready)),
        "blocked_rows": int(len(blocked)),
        "not_immediately_fillable_rows": int(len(blocked)),
        "missing_checkpoint_rows": int((~frame["checkpoint_exists"].astype(bool)).sum()),
        "missing_prediction_rows": int((~frame["prediction_exists"].astype(bool)).sum()),
        "status_counts": status_counts,
        "ready_to_train_by_phase": {
            str(key): int(value) for key, value in ready.groupby("phase").size().sort_index().to_dict().items()
        },
        "blocked_by_phase": {
            str(key): int(value) for key, value in blocked.groupby("phase").size().sort_index().to_dict().items()
        },
        "outputs": {
            "artifact_inventory_csv": str(inventory_csv),
            "missing_artifacts_csv": str(missing_csv),
            "inventory_summary_json": str(summary_json),
            "inventory_summary_markdown": str(summary_markdown),
        },
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Phase45 Formal Multiseed Inventory",
        "",
        f"- Scenario: `{SCENARIO}`",
        f"- Requested rows: `{summary_payload['total_requested_rows']}`",
        f"- Complete rows: `{summary_payload['complete_rows']}`",
        f"- Ready to train now: `{summary_payload['ready_to_train_rows']}`",
        f"- Not immediately fillable: `{summary_payload['not_immediately_fillable_rows']}`",
        f"- Status counts: `{json.dumps(status_counts, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    if not ready.empty:
        lines.extend(
            [
                "## Ready To Train Now",
                "",
                "| Phase | Split | Seed | Missing Artifacts |",
                "| --- | --- | ---: | --- |",
            ]
        )
        for row in ready.itertuples(index=False):
            lines.append(f"| {row.phase} | {row.split_name} | {row.seed} | {row.missing_artifacts or '-'} |")
        lines.append("")
    if not blocked.empty:
        lines.extend(
            [
                "## Not Immediately Fillable",
                "",
                "| Phase | Split | Seed | Status | Missing Artifacts |",
                "| --- | --- | ---: | --- | --- |",
            ]
        )
        for row in blocked.itertuples(index=False):
            lines.append(
                f"| {row.phase} | {row.split_name} | {row.seed} | {row.status} | {row.missing_artifacts or '-'} |"
            )
        lines.append("")
    write_text(summary_markdown, "\n".join(lines).rstrip() + "\n")
    return summary_payload


def _build_phase_config(phase: str, split_name: str, seed: int, device: str) -> dict[str, object]:
    cfg = load_config(_phase_config_path(phase))
    cfg["runtime"]["scenario"] = SCENARIO
    cfg["runtime"]["device"] = str(device)
    cfg["runtime"]["seed"] = int(seed)
    cfg["runtime"]["split_name"] = str(split_name)
    cfg["runtime"]["requested_split_name"] = str(split_name)
    cfg["runtime"]["amp"] = False if str(device).lower() == "cpu" else bool(cfg["runtime"].get("amp", True))
    bundle_root = _phase_bundle_root(phase, split_name)
    cfg["data"]["train_bundle"] = str(bundle_root / "train_bundle.pt")
    cfg["data"]["valid_bundle"] = str(bundle_root / "valid_bundle.pt")
    return cfg


def _result_row(phase: str, split_name: str, seed: int, status: str, checkpoint: Path | None, metrics: dict[str, object] | None = None) -> dict[str, object]:
    metrics = metrics or {}
    return {
        "phase": phase,
        "split_name": split_name,
        "seed": int(seed),
        "status": status,
        "checkpoint": str(checkpoint) if checkpoint is not None else "",
        "auprc": float(metrics.get("auprc", 0.0)),
        "auroc": float(metrics.get("auroc", 0.0)),
        "ece": float(metrics.get("ece", 0.0)),
        "best_valid_loss": float(metrics.get("best_valid_loss", 0.0)),
    }


def execute_training_plan(
    *,
    splits: list[str] | tuple[str, ...],
    seeds: list[int] | tuple[int, ...],
    phases: list[str] | tuple[str, ...],
    device: str,
    overwrite: bool = False,
    report_root: str | Path = REPORT_ROOT,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    report_root = ensure_dir(report_root)
    rows: list[dict[str, object]] = []
    recorded_complete: set[tuple[str, str, int]] = set()
    resolved_keys: set[tuple[str, str, int]] = set()

    while True:
        inventory_rows = collect_artifact_inventory(
            splits=splits,
            seeds=seeds,
            phases=phases,
            overwrite=bool(overwrite),
        )
        inventory_manifest = write_inventory_reports(inventory_rows, report_root=report_root)
        ready_rows = [row for row in inventory_rows if bool(row["ready_for_training"])]

        for inventory in inventory_rows:
            if str(inventory["status"]) != "complete" or bool(overwrite):
                continue
            key = (str(inventory["phase"]), str(inventory["split_name"]), int(inventory["seed"]))
            if key in recorded_complete or key in resolved_keys:
                continue
            recorded_complete.add(key)
            checkpoint_value = str(inventory["checkpoint"])
            checkpoint = Path(checkpoint_value) if checkpoint_value else None
            row = _result_row(key[0], key[1], key[2], "skipped_existing", checkpoint)
            row["prediction_path"] = str(inventory["prediction_path"])
            rows.append(row)

        if not ready_rows:
            final_rows = [
                inventory
                for inventory in inventory_rows
                if str(inventory["status"]) != "complete" or bool(overwrite)
            ]
            unresolved_keys = {
                (str(item["phase"]), str(item["split_name"]), int(item["seed"])) for item in final_rows
            }
            rows = [
                row
                for row in rows
                if (str(row["phase"]), str(row["split_name"]), int(row["seed"])) not in unresolved_keys
            ]
            for inventory in final_rows:
                checkpoint_value = str(inventory["checkpoint"])
                checkpoint = Path(checkpoint_value) if checkpoint_value else None
                row = _result_row(
                    str(inventory["phase"]),
                    str(inventory["split_name"]),
                    int(inventory["seed"]),
                    str(inventory["status"]),
                    checkpoint,
                )
                row["prediction_path"] = str(inventory["prediction_path"])
                row["blocked_reason"] = str(inventory["blocked_reason"])
                row["missing_artifacts"] = str(inventory["missing_artifacts"])
                rows.append(row)
            return rows, inventory_manifest

        for inventory in ready_rows:
            split_name = str(inventory["split_name"])
            seed = int(inventory["seed"])
            phase = str(inventory["phase"])
            split_root = ensure_dir(Path(report_root) / split_name)
            cfg = _build_phase_config(phase, split_name, seed, str(device))
            config_path = split_root / f"{phase}_seed_{seed}.yaml"
            dump_config(cfg, config_path)
            metrics = train_phase(cfg, phase=phase)
            checkpoint = _latest_checkpoint(phase, split_name, seed)
            row = _result_row(phase, split_name, seed, "trained", checkpoint, metrics)
            row["config_path"] = str(config_path)
            row["prediction_path"] = str(_prediction_path(phase, split_name, seed))
            rows.append(row)
            resolved_keys.add((phase, split_name, seed))
            print(json.dumps(row, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train missing phase45_hematopoiesis phase4/phase5 seeds across the canonical splits."
    )
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--phases", nargs="+", choices=["phase4", "phase5"], default=list(DEFAULT_PHASES))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Only write the artifact inventory / missing-artifact reports without starting training.",
    )
    args = parser.parse_args()

    report_root = ensure_dir(REPORT_ROOT)
    inventory_rows = collect_artifact_inventory(
        splits=args.splits,
        seeds=args.seeds,
        phases=args.phases,
        overwrite=bool(args.overwrite),
    )
    inventory_manifest = write_inventory_reports(inventory_rows, report_root=report_root)
    if args.inventory_only:
        print(json.dumps({"inventory_outputs": inventory_manifest["outputs"], "summary": inventory_manifest}, ensure_ascii=False))
        return

    rows, final_inventory_manifest = execute_training_plan(
        splits=args.splits,
        seeds=args.seeds,
        phases=args.phases,
        device=str(args.device),
        overwrite=bool(args.overwrite),
        report_root=report_root,
    )
    summary_path = report_root / "summary.json"
    summary_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "rows": len(rows),
                "inventory_outputs": final_inventory_manifest["outputs"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
