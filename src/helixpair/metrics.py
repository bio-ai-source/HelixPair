from __future__ import annotations

from typing import Iterable

import numpy as np


def binary_classification_metrics(labels: Iterable[int], scores: Iterable[float]) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score

    labels = np.asarray(list(labels), dtype=np.int64)
    scores = np.asarray(list(scores), dtype=np.float32)
    if labels.size == 0:
        return {"auroc": 0.0, "auprc": 0.0, "brier": 0.0}
    metrics = {
        "auprc": float(average_precision_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
        "nll": float(log_loss(labels, np.clip(scores, 1e-6, 1 - 1e-6), labels=[0, 1])),
    }
    metrics["auroc"] = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else 0.0
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(labels, scores >= 0.5))
    return metrics


def expected_calibration_error(labels: Iterable[int], scores: Iterable[float], bins: int = 10) -> float:
    labels = np.asarray(list(labels), dtype=np.int64)
    scores = np.asarray(list(scores), dtype=np.float32)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        mask = (scores >= bin_edges[index]) & (scores < bin_edges[index + 1] if index < bins - 1 else scores <= bin_edges[index + 1])
        if not mask.any():
            continue
        confidence = float(scores[mask].mean())
        accuracy = float(labels[mask].mean())
        ece += abs(confidence - accuracy) * float(mask.mean())
    return ece


def top_k_recall(labels: Iterable[int], scores: Iterable[float], k: int = 50) -> float:
    labels = np.asarray(list(labels), dtype=np.int64)
    scores = np.asarray(list(scores), dtype=np.float32)
    if labels.sum() == 0:
        return 0.0
    indices = np.argsort(scores)[::-1][:k]
    return float(labels[indices].sum() / labels.sum())


def emd_1d(target: Iterable[float], prediction: Iterable[float]) -> float:
    target = np.asarray(list(target), dtype=np.float32)
    prediction = np.asarray(list(prediction), dtype=np.float32)
    target = target / max(target.sum(), 1e-6)
    prediction = prediction / max(prediction.sum(), 1e-6)
    return float(np.abs(np.cumsum(target) - np.cumsum(prediction)).sum())


def spacing_distribution_metrics(target: Iterable[float], prediction: Iterable[float]) -> dict[str, float]:
    target = np.asarray(list(target), dtype=np.float32)
    prediction = np.asarray(list(prediction), dtype=np.float32)
    target = target / max(target.sum(), 1e-6)
    prediction = prediction / max(prediction.sum(), 1e-6)
    return {
        "spacing_kl": float(np.sum(target * np.log((target + 1e-8) / (prediction + 1e-8)))),
        "spacing_emd": emd_1d(target, prediction),
    }


def relative_ood_gain(full_score: float, baseline_score: float) -> float:
    denominator = max(abs(baseline_score), 1e-6)
    return float((full_score - baseline_score) / denominator)
