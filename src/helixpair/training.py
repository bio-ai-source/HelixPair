from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from helixpair.audits import availability_gap_test, bridge_leakage_audit, helical_recovery_score, identifiability_audit
from helixpair.acceptance import validate_phase_readiness
from helixpair.config import dump_config
from helixpair.data import make_loader
from helixpair.io_utils import build_run_manifest, init_run_artifacts, read_table, record_run_ledger, resolve_path, split_token, write_json, write_text
from helixpair.losses import (
    binary_nll_loss,
    bridge_null_regularization,
    calibration_loss,
    categorical_kl_loss,
    grouped_softmax_ranking_loss,
    monotonicity_penalty,
    probability_bce,
    spacing_emd_loss,
)
from helixpair.metrics import binary_classification_metrics, expected_calibration_error, spacing_distribution_metrics
from helixpair.model import HelixPairModel
from helixpair.splits import phase_split_name


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: dict[str, Any]) -> HelixPairModel:
    model_cfg = config["model"]
    ablation_cfg = config.get("ablations", {})
    use_shape_channels = not bool(ablation_cfg.get("disable_shape_channels", False))
    sequence_channels = 9 if use_shape_channels else 5
    interface_channels = 13 if use_shape_channels else 9
    return HelixPairModel(
        num_families=model_cfg["num_families"],
        num_subfamilies=model_cfg["num_subfamilies"],
        num_paralogs=model_cfg["num_paralogs"],
        num_tfs=model_cfg["num_tfs"],
        geometry_dim=model_cfg["geometry_dim"],
        availability_dim=model_cfg["availability_dim"],
        state_dim=model_cfg["state_dim"],
        sequence_channels=int(model_cfg.get("sequence_channels", sequence_channels))
        if use_shape_channels
        else sequence_channels,
        embedding_dim=model_cfg.get("embedding_dim", 32),
        rank=model_cfg.get("rank", 8),
        interface_channels=int(model_cfg.get("interface_channels", interface_channels))
        if use_shape_channels
        else interface_channels,
        num_gap_bins=model_cfg.get("num_gap_bins", 25),
        helical_order=model_cfg.get("helical_order", 2),
        use_hierarchy_embedding=not bool(ablation_cfg.get("disable_hierarchy_embedding", False)),
        use_anchor_refinement=not bool(ablation_cfg.get("disable_anchor_refinement", False)),
        use_geometry_head=not bool(ablation_cfg.get("disable_geometry_head", False)),
        use_bridge_head=not bool(ablation_cfg.get("disable_bridge_head", False)),
        use_state_gate=not bool(ablation_cfg.get("disable_state_gate", False)),
        use_partition=not bool(ablation_cfg.get("disable_partition_head", False)),
        use_availability=not bool(ablation_cfg.get("disable_availability", False)),
        disable_helical_basis=bool(ablation_cfg.get("disable_helical_basis", False)),
        disable_uncertainty_ensemble=bool(ablation_cfg.get("disable_uncertainty_ensemble", False)),
        monomer_hidden_dim=model_cfg.get("monomer_hidden_dim", 64),
        bridge_hidden_dim=model_cfg.get("bridge_hidden_dim", 64),
        bridge_kernel_size=model_cfg.get("bridge_kernel_size", 7),
        state_gate_dropout_p=model_cfg.get("state_gate_dropout_p", 0.1),
        enable_signed_gain_diagnostic=bool(ablation_cfg.get("enable_signed_gain_diagnostic", False)),
    )


def _device_from_config(config: dict[str, Any]) -> torch.device:
    requested = str(config["runtime"].get("device", "cuda"))
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _left_right_ids(batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    left = {
        "family_id": batch["left_family_id"],
        "subfamily_id": batch["left_subfamily_id"],
        "paralog_id": batch["left_paralog_id"],
        "tf_id": batch["left_tf_id"],
    }
    right = {
        "family_id": batch["right_family_id"],
        "subfamily_id": batch["right_subfamily_id"],
        "paralog_id": batch["right_paralog_id"],
        "tf_id": batch["right_tf_id"],
    }
    return left, right


def _make_grad_scaler(device: torch.device, enabled: bool):
    class _NoOpScaledLoss:
        def __init__(self, loss: torch.Tensor):
            self.loss = loss

        def backward(self) -> None:
            self.loss.backward()

    class _NoOpGradScaler:
        def scale(self, loss: torch.Tensor) -> _NoOpScaledLoss:
            return _NoOpScaledLoss(loss)

        def step(self, optimizer) -> None:
            optimizer.step()

        def update(self) -> None:
            return None

        def is_enabled(self) -> bool:
            return False

    try:
        return torch.amp.GradScaler(device_type=device.type, enabled=enabled)
    except TypeError:  # torch<2.3 compatibility
        if device.type == "cuda":
            return torch.cuda.amp.GradScaler(enabled=enabled)
        return _NoOpGradScaler()


def _autocast_context(device: torch.device, enabled: bool):
    try:
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    except TypeError:  # torch<2.3 compatibility
        if device.type == "cuda":
            return torch.cuda.amp.autocast(enabled=enabled)
        return torch.cuda.amp.autocast(enabled=False)


def _save_checkpoint(model: nn.Module, output_dir: Path, name: str) -> None:
    torch.save(model.state_dict(), output_dir / name)


def _previous_phase(phase: str) -> str | None:
    phase_order = ["phase1", "phase2", "phase3", "phase4", "phase5"]
    if phase not in phase_order:
        return None
    index = phase_order.index(phase)
    if index == 0:
        return None
    return phase_order[index - 1]


def _split_name_from_config(config: dict[str, Any]) -> str:
    return str(config.get("runtime", {}).get("split_name", "default"))


def _requested_split_name_from_config(config: dict[str, Any]) -> str:
    runtime = config.get("runtime", {})
    return str(runtime.get("requested_split_name") or runtime.get("split_name", "default"))


def _effective_split_name(config: dict[str, Any], phase: str) -> str:
    runtime = config.get("runtime", {})
    explicit = runtime.get("phase_split_name")
    if explicit:
        return str(explicit)
    return phase_split_name(_requested_split_name_from_config(config), phase)


def _latest_phase_checkpoint(config: dict[str, Any], phase: str, scenario: str, seed: int) -> Path | None:
    checkpoints_root = resolve_path(config["paths"]["checkpoints"])
    phase_root = checkpoints_root / phase / scenario / split_token(_effective_split_name(config, phase)) / f"seed_{seed}"
    if not phase_root.exists():
        return None
    candidates = [path for path in phase_root.glob(f"*/{phase}.pt") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.as_posix()))


def _bundle_metadata_path(bundle_path: str | Path) -> Path:
    return resolve_path(bundle_path).with_name(resolve_path(bundle_path).name.replace("_bundle.pt", "_metadata.parquet"))


def _attach_prediction_metadata(
    prediction_frame: pd.DataFrame,
    bundle_path: str | Path,
    phase: str,
    scenario: str,
    split_name: str,
    seed: int,
    checkpoint_path: Path,
    run_id: str,
) -> pd.DataFrame:
    metadata_path = _bundle_metadata_path(bundle_path)
    enriched = prediction_frame.copy()
    if metadata_path.exists():
        metadata = pd.read_parquet(metadata_path).reset_index(drop=True)
        metadata = metadata.drop(columns=[column for column in metadata.columns if column in enriched.columns], errors="ignore")
        enriched = pd.concat([metadata, enriched.reset_index(drop=True)], axis=1)
    availability_signal = enriched.get("availability_signal")
    if availability_signal is None and "availability_stratum" in enriched.columns:
        availability_signal = pd.to_numeric(enriched["availability_stratum"], errors="coerce")
    if availability_signal is not None:
        numeric_signal = pd.Series(availability_signal).astype(float)
        if numeric_signal.notna().sum() > 0:
            try:
                enriched["availability_stratum"] = pd.qcut(
                    numeric_signal.rank(method="first"),
                    q=min(5, numeric_signal.notna().sum()),
                    labels=False,
                    duplicates="drop",
                ).astype("Int64").astype(str)
            except ValueError:
                enriched["availability_stratum"] = numeric_signal.round(3).astype(str)
    enriched["phase"] = phase
    enriched["scenario"] = scenario
    enriched["split_name"] = split_name
    enriched["seed"] = seed
    enriched["checkpoint_id"] = run_id
    enriched["checkpoint_path"] = str(checkpoint_path)
    return enriched


def _write_validation_figure(prediction_frame: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    positive = prediction_frame[prediction_frame["label"].astype(float) > 0.5]
    negative = prediction_frame[prediction_frame["label"].astype(float) <= 0.5]
    axes[0].hist(negative["full_score"], bins=20, alpha=0.6, label="negative")
    axes[0].hist(positive["full_score"], bins=20, alpha=0.6, label="positive")
    axes[0].set_title("Validation Score")
    axes[0].set_xlabel("full_score")
    axes[0].legend()
    if {"edge_gap", "full_score"}.issubset(prediction_frame.columns):
        spacing = prediction_frame.groupby("edge_gap", as_index=False)["full_score"].mean()
        axes[1].plot(spacing["edge_gap"], spacing["full_score"], marker="o")
        axes[1].set_title("Gap Landscape")
        axes[1].set_xlabel("edge_gap")
        axes[1].set_ylabel("mean full_score")
    else:
        axes[1].axis("off")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def _write_error_analysis(
    prediction_frame: pd.DataFrame,
    output_path: Path,
    phase: str,
    scenario: str,
    split_name: str,
    seed: int,
    initialized_from: Path | None,
    best_valid: float,
    trainable_modules: list[str],
    metrics: dict[str, Any],
) -> None:
    enriched = prediction_frame.copy()
    enriched["error"] = (enriched["label"].astype(float) - enriched["full_score"].astype(float)).abs()
    hardest = enriched.sort_values("error", ascending=False).head(8)
    lines = [
        "# Error Analysis",
        "",
        f"- Phase: {phase}",
        f"- Scenario: {scenario}",
        f"- Split: {split_name}",
        f"- Seed: {seed}",
        f"- Initialized from: {initialized_from if initialized_from is not None else 'scratch'}",
        f"- Best validation loss: {best_valid:.6f}",
        f"- Trainable modules: {', '.join(trainable_modules)}",
        f"- AUROC: {metrics.get('auroc', 0.0):.4f}",
        f"- AUPRC: {metrics.get('auprc', 0.0):.4f}",
        f"- ECE: {metrics.get('ece', 0.0):.4f}",
        "",
        "## Hardest Examples",
        "",
    ]
    columns = [column for column in ["seq_id", "left_tf", "right_tf", "state_label", "label", "full_score", "error"] if column in hardest.columns]
    for row in hardest[columns].itertuples(index=False):
        row_dict = row._asdict()
        lines.append("- " + ", ".join(f"{key}={row_dict[key]}" for key in columns))
    write_text(output_path, "\n".join(lines))


def _load_compatible_state_dict(model: HelixPairModel, state_dict: dict[str, torch.Tensor]) -> tuple[int, int]:
    current_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in current_state and tuple(current_state[key].shape) == tuple(value.shape)
    }
    if not compatible:
        return 0, len(current_state)
    current_state.update(compatible)
    model.load_state_dict(current_state)
    return len(compatible), len(current_state)


def _initialize_model(
    model: HelixPairModel,
    config: dict[str, Any],
    phase: str,
    scenario: str,
    seed: int,
) -> Path | None:
    training_cfg = config.get("training", {})
    init_checkpoint = training_cfg.get("init_checkpoint")
    checkpoint_path: Path | None = None
    if init_checkpoint:
        checkpoint_path = resolve_path(init_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing init checkpoint: {checkpoint_path}")
    elif bool(training_cfg.get("init_from_previous_phase", True)):
        previous_phase = _previous_phase(phase)
        if previous_phase is None:
            return None
        checkpoint_path = _latest_phase_checkpoint(config, previous_phase, scenario, seed)
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"{phase} expects a {previous_phase} checkpoint under "
                f"{resolve_path(config['paths']['checkpoints']) / previous_phase / scenario / split_token(_effective_split_name(config, previous_phase)) / f'seed_{seed}'}."
            )
    if checkpoint_path is None:
        return None
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    loaded_count, total_count = _load_compatible_state_dict(model, state_dict)
    if loaded_count == 0:
        raise RuntimeError(f"No compatible parameters could be loaded from {checkpoint_path}.")
    if loaded_count == total_count:
        return checkpoint_path
    return Path(f"{checkpoint_path} [partial {loaded_count}/{total_count}]")


def _phase_parameters(model: HelixPairModel, config: dict[str, Any], phase: str) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    freeze_schedule = config.get("training", {}).get("freeze_schedule", {})
    trainable_modules = freeze_schedule.get(phase, [])
    if not trainable_modules:
        trainable_modules = ["embedding", "monomer_head", "geometry_head", "bridge_head", "chemical_potential_head", "state_gate_head", "partition_head"]
    if phase in {"phase4", "phase5"} and bool(config.get("training", {}).get("allow_joint_residual_tuning_if_identifiable", False)):
        reference_graph_value = str(config.get("evaluation", {}).get("reference_graph", "")).strip()
        reference_graph_path = resolve_path(reference_graph_value) if reference_graph_value else None
        if reference_graph_path is not None and reference_graph_path.exists():
            identifiability = identifiability_audit(read_table(reference_graph_path))
            if bool(identifiability.get("joint_finetune_allowed", False)):
                trainable_modules = list(
                    dict.fromkeys(list(trainable_modules) + list(config.get("training", {}).get("joint_residual_modules", [])))
                )
    for name, parameter in model.named_parameters():
        parameter.requires_grad = any(module_name in name for module_name in trainable_modules)
    return list(trainable_modules)


def _forward_model(model: HelixPairModel, batch: dict[str, torch.Tensor]):
    left_ids, right_ids = _left_right_ids(batch)
    return model(
        left_anchor_offsets=batch["left_anchor_offsets"],
        right_anchor_offsets=batch["right_anchor_offsets"],
        left_ids=left_ids,
        right_ids=right_ids,
        geometry_features=batch["geometry_features"],
        interface_tensor=batch["interface_tensor"],
        availability=batch.get("availability"),
        state_context=batch.get("state_context"),
        compatibility=batch.get("compatibility"),
    )


def _bridge_controls(model: HelixPairModel, outputs, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if not model.use_bridge_head:
        zeros = torch.zeros_like(outputs.bridge_residual)
        return zeros, zeros
    shuffle_residual, _ = model.bridge_head(batch["interface_tensor_shuffle"], outputs.pair_embedding)
    rerand_residual, _ = model.bridge_head(batch["interface_tensor_core_rerandomized"], outputs.pair_embedding)
    return shuffle_residual, rerand_residual


def _build_phase_loss(phase: str, model: HelixPairModel, outputs, batch: dict[str, torch.Tensor], config: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    loss_terms: dict[str, torch.Tensor] = {}
    aux_outputs: dict[str, torch.Tensor] = {}
    if phase == "phase1":
        logits = -outputs.intrinsic_monomer_energy
        loss_terms["monomer_bce"] = binary_nll_loss(logits, batch["labels"])
    elif phase == "phase2":
        interaction_logits = -(outputs.intrinsic_monomer_energy + outputs.biochemical_residual)
        loss_terms["pair_bce"] = binary_nll_loss(interaction_logits, batch["labels"])
        loss_terms["spacing_emd"] = float(config["losses"].get("lambda_spacing_emd", 0.2)) * spacing_emd_loss(
            batch["spacing_target"], outputs.gap_logits
        )
        loss_terms["spacing_kl"] = float(config["losses"].get("lambda_spacing_kl", 0.1)) * categorical_kl_loss(
            batch["spacing_target"], outputs.gap_logits
        )
        loss_terms["orientation_ce"] = float(config["losses"].get("lambda_orientation", 0.1)) * categorical_kl_loss(
            batch["orientation_target"], outputs.orientation_logits
        )
        loss_terms["composite_bce"] = float(config["losses"].get("lambda_composite_aux", 0.1)) * binary_nll_loss(
            outputs.composite_logit, batch["composite_target"]
        )
        shuffled_bridge, rerandomized_bridge = _bridge_controls(model, outputs, batch)
        aux_outputs["bridge_score_interface_shuffle"] = shuffled_bridge
        aux_outputs["bridge_score_core_rerandomized"] = rerandomized_bridge
        loss_terms["bridge_null"] = float(config["losses"].get("lambda_bridge_null", 0.1)) * bridge_null_regularization(
            outputs.bridge_residual, shuffled_bridge
        )
        gap_probs = torch.softmax(outputs.gap_logits, dim=-1)
        loss_terms["geometry_smooth"] = float(config["losses"].get("lambda_geometry_smooth", 0.1)) * (
            (gap_probs[:, 1:] - gap_probs[:, :-1]).square().mean()
        )
    elif phase == "phase3":
        if outputs.availability_only_probability is None or batch.get("availability") is None:
            raise ValueError("phase3 requires availability and state context")
        loss_terms["availability_only_bce"] = probability_bce(outputs.availability_only_probability, batch["labels"])
        loss_terms["calibration"] = float(config["losses"].get("lambda_calibration", 0.1)) * calibration_loss(
            outputs.availability_only_probability, batch["labels"]
        )
        mean_mu = 0.5 * (outputs.left_chemical_potential + outputs.right_chemical_potential)
        loss_terms["monotonicity"] = float(config["losses"].get("lambda_monotonicity", 0.1)) * monotonicity_penalty(
            mean_mu, batch["availability"]
        )
    else:
        if outputs.usage_probability is None:
            raise ValueError(f"{phase} requires availability and state context")
        loss_terms["usage_bce"] = probability_bce(outputs.usage_probability, batch["labels"])
        loss_terms["calibration"] = float(config["losses"].get("lambda_calibration", 0.1)) * calibration_loss(
            outputs.usage_probability, batch["labels"]
        )
        if phase in {"phase4", "phase5"}:
            loss_terms["state_small_correction"] = float(config["losses"].get("lambda_state_small_correction", 0.1)) * outputs.state_correction.square().mean()
        if phase == "phase4" and batch.get("state_group_id") is not None:
            loss_terms["state_ranking"] = float(config["losses"].get("lambda_state_ranking", 1.0)) * grouped_softmax_ranking_loss(
                outputs.usage_probability,
                batch["labels"],
                batch["state_group_id"],
            )
        if phase == "phase5":
            loss_terms["functional_composite"] = float(config["losses"].get("lambda_composite_aux", 0.1)) * binary_nll_loss(
                outputs.composite_logit, batch["composite_target"]
            )
    total_loss = torch.stack(list(loss_terms.values())).sum()
    scalar_terms = {key: float(value.detach().cpu()) for key, value in loss_terms.items()}
    return total_loss, scalar_terms, aux_outputs


def _scores_for_phase(phase: str, outputs) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    if phase == "phase1":
        score = torch.sigmoid(-outputs.intrinsic_monomer_energy)
        additive = torch.sigmoid(-outputs.intrinsic_monomer_energy)
        return score, None, additive
    if phase == "phase2":
        full_score = torch.sigmoid(-(outputs.intrinsic_monomer_energy + outputs.biochemical_residual))
        additive = torch.sigmoid(-outputs.intrinsic_monomer_energy)
        return full_score, None, additive
    full_score = outputs.usage_probability
    availability_only = outputs.availability_only_probability
    additive = torch.sigmoid(-(outputs.left_intrinsic_energy + outputs.right_intrinsic_energy))
    return full_score, availability_only, additive


def train_phase(config: dict[str, Any], phase: str) -> dict[str, Any]:
    scenario = str(config["runtime"]["scenario"])
    split_name = _split_name_from_config(config)
    requested_split_name = _requested_split_name_from_config(config)
    seed = int(config["runtime"]["seed"])
    validate_phase_readiness(config, phase)
    seed_everything(seed)
    device = _device_from_config(config)
    run = init_run_artifacts(config, phase=phase, scenario=scenario, seed=seed)
    Path(run.predictions_path).parent.mkdir(parents=True, exist_ok=True)

    dump_config(config, run.config_path)
    run_manifest = build_run_manifest(run, config, phase=phase, scenario=scenario, seed=seed)
    write_json(run.manifest_path, asdict(run_manifest))

    model = build_model(config).to(device)
    initialized_from = _initialize_model(model, config, phase, scenario, seed)
    trainable_modules = _phase_parameters(model, config, phase)
    num_workers = int(config["training"].get("num_workers", 0))
    pin_memory = bool(config["training"].get("pin_memory", True)) and device.type == "cuda"
    train_loader = make_loader(
        config["data"]["train_bundle"],
        batch_size=int(config["training"]["batch_size"]),
        shuffle=phase != "phase4",
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = make_loader(
        config["data"]["valid_bundle"],
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"].get("weight_decay", 1e-2)),
    )
    scaler = _make_grad_scaler(device, enabled=device.type == "cuda" and bool(config["runtime"].get("amp", True)))
    best_state = None
    best_valid = float("inf")
    stale_epochs = 0
    patience = int(config["training"].get("early_stop_patience", 8))
    gradient_accumulation = max(int(config["training"].get("gradient_accumulation", 1)), 1)
    best_rows: list[dict[str, float]] = []
    best_terms: dict[str, float] = {}
    best_gap_targets: list[np.ndarray] = []
    best_gap_predictions: list[np.ndarray] = []

    for _epoch in range(int(config["training"]["epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = _batch_to_device(batch, device)
            with _autocast_context(device, enabled=scaler.is_enabled()):
                outputs = _forward_model(model, batch)
                loss, _, _ = _build_phase_loss(phase, model, outputs, batch, config)
                loss = loss / gradient_accumulation
            scaler.scale(loss).backward()
            if batch_index % gradient_accumulation == 0 or batch_index == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"].get("gradient_clip", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        model.eval()
        valid_losses: list[float] = []
        term_history: dict[str, list[float]] = {}
        prediction_rows: list[dict[str, float]] = []
        gap_targets: list[np.ndarray] = []
        gap_predictions: list[np.ndarray] = []
        with torch.no_grad():
            for batch in valid_loader:
                batch = _batch_to_device(batch, device)
                outputs = _forward_model(model, batch)
                valid_loss, terms, aux_outputs = _build_phase_loss(phase, model, outputs, batch, config)
                valid_losses.append(float(valid_loss.detach().cpu()))
                for key, value in terms.items():
                    term_history.setdefault(key, []).append(value)
                full_score, availability_only, additive_null = _scores_for_phase(phase, outputs)
                if "bridge_score_interface_shuffle" not in aux_outputs and model.use_bridge_head:
                    shuffled_bridge, core_rerandomized = _bridge_controls(model, outputs, batch)
                else:
                    shuffled_bridge = aux_outputs.get("bridge_score_interface_shuffle")
                    core_rerandomized = aux_outputs.get("bridge_score_core_rerandomized")
                availability_strata = batch["availability"].mean(dim=-1) if "availability" in batch else torch.zeros_like(full_score)
                for index in range(full_score.shape[0]):
                    prediction_rows.append(
                        {
                            "label": float(batch["labels"][index].detach().cpu()),
                            "full_score": float(full_score[index].detach().cpu()),
                            "availability_only_score": float(availability_only[index].detach().cpu()) if availability_only is not None else float("nan"),
                            "additive_null_score": float(additive_null[index].detach().cpu()),
                            "bridge_score": float(outputs.bridge_residual[index].detach().cpu()),
                            "geometry_residual": float(outputs.geometry_residual[index].detach().cpu()),
                            "biochemical_residual": float(outputs.biochemical_residual[index].detach().cpu()),
                            "bridge_score_interface_shuffle": float(shuffled_bridge[index].detach().cpu()) if shuffled_bridge is not None else float("nan"),
                            "bridge_score_core_rerandomized": float(core_rerandomized[index].detach().cpu()) if core_rerandomized is not None else float("nan"),
                            "availability_signal": float(availability_strata[index].detach().cpu()),
                            "availability_stratum": f"{float(availability_strata[index].detach().cpu()):.2f}",
                            "composite_target": float(batch["composite_target"][index].detach().cpu()) if "composite_target" in batch else float("nan"),
                            "state_gate": float(outputs.state_gate[index].detach().cpu()) if outputs.state_gate is not None else float("nan"),
                            "state_correction": float(outputs.state_correction[index].detach().cpu()) if outputs.state_correction is not None else float("nan"),
                            "cooperative_gain": float(outputs.cooperative_gain[index].detach().cpu()) if outputs.cooperative_gain is not None else float("nan"),
                            "signed_pair_gain": float(outputs.signed_pair_gain[index].detach().cpu()) if outputs.signed_pair_gain is not None else float("nan"),
                            "monomer_free_energy": float(outputs.monomer_free_energy[index].detach().cpu()) if outputs.monomer_free_energy is not None else float("nan"),
                        }
                    )
                if "spacing_target" in batch:
                    gap_targets.extend(batch["spacing_target"].detach().cpu().numpy())
                    gap_predictions.extend(torch.softmax(outputs.gap_logits, dim=-1).detach().cpu().numpy())
        current_valid = float(np.mean(valid_losses)) if valid_losses else 0.0
        if current_valid < best_valid:
            best_valid = current_valid
            stale_epochs = 0
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_rows = prediction_rows
            best_terms = {key: float(np.mean(values)) for key, values in term_history.items()}
            best_gap_targets = gap_targets
            best_gap_predictions = gap_predictions
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    _save_checkpoint(model, run.output_dir, f"{phase}.pt")

    prediction_frame = _attach_prediction_metadata(
        pd.DataFrame.from_records(best_rows),
        bundle_path=config["data"]["valid_bundle"],
        phase=phase,
        scenario=scenario,
        split_name=split_name,
        seed=seed,
        checkpoint_path=run.checkpoint_path,
        run_id=run.output_dir.name,
    )
    prediction_frame.to_parquet(run.predictions_path, index=False)
    metrics = binary_classification_metrics(prediction_frame["label"], prediction_frame["full_score"])
    metrics["ece"] = expected_calibration_error(prediction_frame["label"], prediction_frame["full_score"])
    metrics["best_valid_loss"] = best_valid
    metrics["trainable_modules"] = trainable_modules
    metrics["split_name"] = split_name
    metrics["requested_split_name"] = requested_split_name
    metrics.update(best_terms)
    metrics["additive_null_auprc"] = binary_classification_metrics(
        prediction_frame["label"],
        prediction_frame["additive_null_score"],
    )["auprc"]
    metrics["additive_null_auroc"] = binary_classification_metrics(
        prediction_frame["label"],
        prediction_frame["additive_null_score"],
    )["auroc"]
    metrics["relative_gain_vs_additive_auprc"] = float(metrics["auprc"] - metrics["additive_null_auprc"])
    if prediction_frame["availability_only_score"].notna().any():
        availability_frame = prediction_frame[prediction_frame["availability_only_score"].notna()].copy()
        availability_metrics = binary_classification_metrics(
            availability_frame["label"],
            availability_frame["availability_only_score"],
        )
        metrics["availability_only_auprc"] = availability_metrics["auprc"]
        metrics["availability_only_auroc"] = availability_metrics["auroc"]
        metrics["availability_only_ece"] = expected_calibration_error(
            availability_frame["label"],
            availability_frame["availability_only_score"],
        )
        metrics["relative_gain_vs_availability_auprc"] = float(metrics["auprc"] - availability_metrics["auprc"])
    if best_gap_targets and best_gap_predictions:
        spacing_metrics = spacing_distribution_metrics(
            np.mean(np.asarray(best_gap_targets), axis=0),
            np.mean(np.asarray(best_gap_predictions), axis=0),
        )
        metrics.update(spacing_metrics)
        metrics["helical_recovery_score"] = helical_recovery_score(
            metrics.get("spacing_emd", 0.0),
            metrics.get("spacing_kl", 0.0),
        )
    else:
        metrics["helical_recovery_score"] = 0.0
    audit_payload = {}
    if {"bridge_score", "bridge_score_interface_shuffle", "bridge_score_core_rerandomized"}.issubset(prediction_frame.columns):
        audit_payload["bridge"] = bridge_leakage_audit(prediction_frame)
        metrics.update(audit_payload["bridge"])
    if prediction_frame["availability_only_score"].notna().any():
        audit_payload["availability_gap"] = availability_gap_test(
            prediction_frame[prediction_frame["availability_only_score"].notna()].copy()
        )
        metrics.update(audit_payload["availability_gap"])
    write_json(run.metrics_path, metrics)
    if audit_payload:
        write_json(run.reports_dir / "audits.json", audit_payload)
    write_json(
        run.seed_manifest_path,
        {
            "phase": phase,
            "scenario": scenario,
            "split_name": split_name,
            "requested_split_name": requested_split_name,
            "seed": seed,
            "run_id": run.output_dir.name,
            "checkpoint_path": str(run.checkpoint_path),
        },
    )
    _write_error_analysis(
        prediction_frame=prediction_frame,
        output_path=run.error_analysis_path,
        phase=phase,
        scenario=scenario,
        split_name=split_name,
        seed=seed,
        initialized_from=initialized_from,
        best_valid=best_valid,
        trainable_modules=trainable_modules,
        metrics=metrics,
    )
    _write_validation_figure(prediction_frame, run.figures_dir / "validation_scores.pdf")
    record_run_ledger(run.ledger_path, run_manifest, metrics=metrics)
    return metrics
