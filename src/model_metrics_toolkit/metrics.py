"""Dependency-free classification and regression metrics."""

from math import sqrt
from collections.abc import Sequence


def classification_metrics(
    actual: Sequence[int],
    predicted: Sequence[int],
) -> dict[str, float]:
    """Calculate accuracy, precision, recall, and F1 for binary labels."""
    if not actual or len(actual) != len(predicted):
        raise ValueError("Inputs must be non-empty and have equal lengths.")
    if any(value not in (0, 1) for value in (*actual, *predicted)):
        raise ValueError("Binary labels must be 0 or 1.")

    tp = sum(a == 1 and p == 1 for a, p in zip(actual, predicted))
    tn = sum(a == 0 and p == 0 for a, p in zip(actual, predicted))
    fp = sum(a == 0 and p == 1 for a, p in zip(actual, predicted))
    fn = sum(a == 1 and p == 0 for a, p in zip(actual, predicted))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "accuracy": (tp + tn) / len(actual),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def regression_metrics(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> dict[str, float]:
    """Calculate mean absolute error, mean squared error, and RMSE."""
    if not actual or len(actual) != len(predicted):
        raise ValueError("Inputs must be non-empty and have equal lengths.")

    errors = [a - p for a, p in zip(actual, predicted)]
    mae = sum(abs(error) for error in errors) / len(errors)
    mse = sum(error**2 for error in errors) / len(errors)

    return {
        "mae": mae,
        "mse": mse,
        "rmse": sqrt(mse),
    }
