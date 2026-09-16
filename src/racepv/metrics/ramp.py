"""Classification and event metrics for photovoltaic ramp forecasts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_CLASS_NAMES = ("stable", "ramp-up", "ramp-down")


def _labels_and_probabilities(
    y_true: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    truth = np.asarray(y_true)
    predicted = np.asarray(predictions)
    probabilities: np.ndarray | None
    if predicted.shape == truth.shape + (3,):
        probabilities = predicted.astype(float)
        if np.any(~np.isfinite(probabilities)):
            raise ValueError("class probabilities contain non-finite values")
        labels = np.argmax(probabilities, axis=-1)
    elif predicted.shape == truth.shape:
        probabilities = None
        labels = predicted
    else:
        raise ValueError("predictions must be labels or have shape y_true.shape + [3]")
    numeric_truth = np.asarray(y_true, dtype=float)
    numeric_labels = np.asarray(labels, dtype=float)
    truth = truth.astype(int, copy=False)
    labels = labels.astype(int, copy=False)
    valid = np.isfinite(numeric_truth) & np.isin(numeric_truth, [0.0, 1.0, 2.0])
    valid &= np.isfinite(numeric_labels) & np.isin(numeric_labels, [0.0, 1.0, 2.0])
    if mask is not None:
        try:
            valid &= np.broadcast_to(np.asarray(mask, dtype=bool), truth.shape)
        except ValueError as exc:
            raise ValueError("mask is not broadcastable to label shape") from exc
    return truth[valid], labels[valid], probabilities[valid] if probabilities is not None else None


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Binary average precision with deterministic, tie-aware thresholds."""

    truth = np.asarray(y_true, dtype=bool).reshape(-1)
    scores = np.asarray(y_score, dtype=float).reshape(-1)
    valid = np.isfinite(scores)
    truth, scores = truth[valid], scores[valid]
    positives = int(np.sum(truth))
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    truth, scores = truth[order], scores[order]
    tp = np.cumsum(truth)
    fp = np.cumsum(~truth)
    threshold_ends = np.r_[np.flatnonzero(np.diff(scores) != 0), scores.size - 1]
    recalls = tp[threshold_ends] / positives
    precisions = tp[threshold_ends] / (tp[threshold_ends] + fp[threshold_ends])
    recall_increments = np.diff(np.r_[0.0, recalls])
    return float(np.sum(recall_increments * precisions))


def _one_vs_rest(
    truth: np.ndarray,
    labels: np.ndarray,
    class_index: int,
) -> dict[str, float | int]:
    actual = truth == class_index
    predicted = labels == class_index
    tp = int(np.sum(actual & predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    tn = int(np.sum(~actual & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    csi = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "support": int(np.sum(actual)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "csi": float(csi),
    }


def ramp_classification_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    *,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    mask: np.ndarray | None = None,
) -> dict[str, object]:
    """Three-class precision/recall/F1/CSI/AUCPR plus macro averages."""

    if len(class_names) != 3:
        raise ValueError("class_names must contain stable, ramp-up, and ramp-down names")
    truth, labels, probabilities = _labels_and_probabilities(y_true, predictions, mask)
    if truth.size == 0:
        raise ValueError("no valid state labels are available")
    confusion = np.zeros((3, 3), dtype=int)
    np.add.at(confusion, (truth, labels), 1)
    per_class: dict[str, dict[str, float | int]] = {}
    for index, name in enumerate(class_names):
        values = _one_vs_rest(truth, labels, index)
        if probabilities is not None:
            values["auprc"] = average_precision(truth == index, probabilities[:, index])
        per_class[str(name)] = values

    macro_keys = ["precision", "recall", "f1", "csi"]
    if probabilities is not None:
        macro_keys.append("auprc")
    macro = {
        key: float(np.nanmean([float(values[key]) for values in per_class.values()]))
        for key in macro_keys
    }
    accuracy = float(np.mean(truth == labels))
    return {
        "count": int(truth.size),
        "accuracy": accuracy,
        "confusion_matrix": confusion,
        "class_names": tuple(str(name) for name in class_names),
        "per_class": per_class,
        "macro": macro,
    }


def binary_ramp_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Collapse up/down into a single ramp class and report event detection skill."""

    if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be a finite value inside (0, 1]")

    truth = np.asarray(y_true)
    predicted = np.asarray(predictions)
    if predicted.shape == truth.shape + (3,):
        score = np.asarray(predicted, dtype=float)[..., 1:].sum(axis=-1)
        label = score >= threshold
    elif predicted.shape == truth.shape:
        if np.issubdtype(predicted.dtype, np.floating) and np.all(
            (predicted[np.isfinite(predicted)] >= 0)
            & (predicted[np.isfinite(predicted)] <= 1)
        ):
            score = predicted.astype(float)
            label = score >= threshold
        else:
            score = None
            label = predicted.astype(int) != 0
    else:
        raise ValueError("predictions must be state labels, ramp scores, or [.., 3] probabilities")
    numeric_truth = np.asarray(truth, dtype=float)
    integer_truth = truth.astype(int)
    actual = integer_truth != 0
    # -100 is the repository-wide ignore label for night/missing targets.  More
    # generally, only the documented three labels are allowed into the metric.
    valid = np.isfinite(numeric_truth) & np.isin(numeric_truth, [0.0, 1.0, 2.0])
    valid &= np.isfinite(np.asarray(label, dtype=float))
    if score is None:
        valid &= np.isin(np.asarray(predicted, dtype=float), [0.0, 1.0, 2.0])
    if score is not None:
        valid &= np.isfinite(score)
    if mask is not None:
        valid &= np.broadcast_to(np.asarray(mask, dtype=bool), truth.shape)
    actual, label = actual[valid], label[valid]
    tp = int(np.sum(actual & label))
    fp = int(np.sum(~actual & label))
    fn = int(np.sum(actual & ~label))
    tn = int(np.sum(~actual & ~label))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    csi = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    result: dict[str, float | int] = {
        "count": int(actual.size),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "csi": float(csi),
    }
    if score is not None:
        result["auprc"] = average_precision(actual, score[valid])
    return result


@dataclass(frozen=True)
class RampEvent:
    """One contiguous, direction-specific ramp event."""

    direction: int
    start: int
    end: int
    amplitude: float | None = None


def extract_ramp_events(
    states: np.ndarray,
    *,
    power: np.ndarray | None = None,
) -> list[RampEvent]:
    """Extract contiguous up/down runs from a one-dimensional state sequence."""

    labels = np.asarray(states, dtype=int).reshape(-1)
    if np.any((labels < 0) | (labels > 2)):
        raise ValueError("states must use labels 0, 1, and 2")
    values = None if power is None else np.asarray(power, dtype=float).reshape(-1)
    if values is not None and values.size != labels.size:
        raise ValueError("power and states must have equal lengths")
    events: list[RampEvent] = []
    index = 0
    while index < labels.size:
        direction = int(labels[index])
        if direction == 0:
            index += 1
            continue
        end = index
        while end + 1 < labels.size and int(labels[end + 1]) == direction:
            end += 1
        amplitude = None
        if values is not None and index > 0:
            baseline_index = index - 1
            if np.isfinite(values[baseline_index]) and np.isfinite(values[end]):
                amplitude = float(values[end] - values[baseline_index])
        events.append(RampEvent(direction, index, end, amplitude))
        index = end + 1
    return events


def ramp_event_errors(
    y_true_states: np.ndarray,
    y_pred_states: np.ndarray,
    *,
    y_true_power: np.ndarray | None = None,
    y_pred_power: np.ndarray | None = None,
    step_minutes: int = 15,
    tolerance_steps: int = 4,
) -> dict[str, float | int]:
    """Greedily match same-direction events and report timing/amplitude errors."""

    if step_minutes <= 0 or tolerance_steps < 0:
        raise ValueError("step_minutes must be positive and tolerance_steps non-negative")
    truth_events = extract_ramp_events(y_true_states, power=y_true_power)
    pred_events = extract_ramp_events(y_pred_states, power=y_pred_power)
    candidates = []
    for truth_index, truth in enumerate(truth_events):
        for pred_index, prediction in enumerate(pred_events):
            distance = abs(prediction.start - truth.start)
            if truth.direction == prediction.direction and distance <= tolerance_steps:
                candidates.append((distance, truth_index, pred_index))
    candidates.sort()
    used_truth: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[RampEvent, RampEvent]] = []
    for _, truth_index, pred_index in candidates:
        if truth_index not in used_truth and pred_index not in used_pred:
            used_truth.add(truth_index)
            used_pred.add(pred_index)
            matches.append((truth_events[truth_index], pred_events[pred_index]))

    timing = np.asarray(
        [(prediction.start - truth.start) * step_minutes for truth, prediction in matches],
        dtype=float,
    )
    amplitude = np.asarray(
        [
            abs(float(prediction.amplitude) - float(truth.amplitude))
            for truth, prediction in matches
            if truth.amplitude is not None and prediction.amplitude is not None
        ],
        dtype=float,
    )
    return {
        "true_events": len(truth_events),
        "predicted_events": len(pred_events),
        "matched_events": len(matches),
        "false_alarms": len(pred_events) - len(matches),
        "missed_events": len(truth_events) - len(matches),
        "mean_signed_timing_error_minutes": float(np.mean(timing)) if timing.size else float("nan"),
        "mean_absolute_timing_error_minutes": float(np.mean(np.abs(timing)))
        if timing.size
        else float("nan"),
        "amplitude_mae": float(np.mean(amplitude)) if amplitude.size else float("nan"),
    }


multiclass_ramp_metrics = ramp_classification_metrics
