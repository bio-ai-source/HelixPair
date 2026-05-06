from __future__ import annotations

import torch
import torch.nn.functional as F


def bridge_null_regularization(bridge_score: torch.Tensor, shuffled_bridge_score: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(shuffled_bridge_score, torch.zeros_like(shuffled_bridge_score)) + 0.1 * F.l1_loss(
        bridge_score - shuffled_bridge_score,
        torch.zeros_like(bridge_score),
    )


def geometry_smoothness_regularization(geometry_landscape: torch.Tensor) -> torch.Tensor:
    if geometry_landscape.ndim == 1:
        geometry_landscape = geometry_landscape.unsqueeze(0)
    diff = geometry_landscape[:, 1:] - geometry_landscape[:, :-1]
    return diff.square().mean() if diff.numel() else geometry_landscape.new_tensor(0.0)


def spacing_emd_loss(target: torch.Tensor, prediction_logits: torch.Tensor) -> torch.Tensor:
    prediction = torch.softmax(prediction_logits, dim=-1)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return (target.cumsum(dim=-1) - prediction.cumsum(dim=-1)).abs().mean()


def categorical_kl_loss(target: torch.Tensor, prediction_logits: torch.Tensor) -> torch.Tensor:
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    prediction = torch.softmax(prediction_logits, dim=-1).clamp_min(1e-8)
    return (target * (target.clamp_min(1e-8).log() - prediction.log())).sum(dim=-1).mean()


def binary_nll_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, labels.float())


def probability_bce(probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = torch.logit(probabilities.clamp(1e-6, 1 - 1e-6))
    return F.binary_cross_entropy_with_logits(logits, labels.float())


def grouped_softmax_ranking_loss(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    group_ids: torch.Tensor,
) -> torch.Tensor:
    logits = torch.logit(probabilities.clamp(1e-6, 1 - 1e-6))
    labels = labels.float()
    group_ids = group_ids.reshape(-1)
    losses = []
    for group_id in torch.unique(group_ids):
        mask = group_ids == group_id
        if int(mask.sum()) < 2:
            continue
        group_logits = logits[mask]
        group_labels = labels[mask]
        if float(group_labels.sum()) <= 0.0 or float(group_labels.sum()) >= float(mask.sum()):
            continue
        positive_weights = group_labels / group_labels.sum().clamp_min(1e-6)
        losses.append(-(positive_weights * F.log_softmax(group_logits, dim=0)).sum())
    if not losses:
        return probabilities.new_tensor(0.0)
    return torch.stack(losses).mean()


def calibration_loss(probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.float()
    return ((probabilities - labels) ** 2).mean()


def monotonicity_penalty(predictions: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
    if predictions.ndim == 1:
        predictions = predictions.unsqueeze(-1)
    sorted_indices = torch.argsort(availability.mean(dim=-1))
    sorted_predictions = predictions[sorted_indices].squeeze(-1)
    deltas = sorted_predictions[1:] - sorted_predictions[:-1]
    return torch.relu(-deltas).mean() if deltas.numel() else predictions.new_tensor(0.0)
