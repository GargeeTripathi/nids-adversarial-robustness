"""
Detection metrics for NIDS evaluation.

Accuracy alone is close to useless here and actively misleading on some of these datasets:
CICIDS2017 is roughly 80% benign, so a model that never raises an alarm scores 80% accuracy
while detecting nothing. The metrics that matter operationally are:

* **Detection rate** (recall / true-positive rate) - the fraction of real attacks caught.
  Every missed attack is an intrusion that proceeded unnoticed.
* **False-positive rate** - the fraction of benign flows wrongly alarmed. This is what
  determines whether a NIDS is deployable at all: at 10 Gbps, even a 0.1% FPR generates
  more alerts per hour than any team can triage. Alert fatigue kills real deployments.
* **ROC-AUC** - threshold-independent ranking quality, letting comparisons avoid the
  arbitrary 0.5 cut-off.

`robustness_score` combines clean and adversarial performance into one comparable number
for the RO1 architecture ranking.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict:
    """
    Full metric suite for binary intrusion detection (0 = benign, 1 = attack).

    Args:
        y_score: predicted probability of the attack class, for threshold-free metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        # Detection rate is recall on the attack class, named for the security audience.
        "detection_rate": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "n_samples": int(len(y_true)),
    }

    if y_score is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))

    return metrics


def robustness_score(clean_metrics: dict, adversarial_metrics: dict) -> dict:
    """
    Summarise how much of a model's clean performance survives attack.

    `retention` is the primary quantity: adversarial accuracy as a fraction of clean
    accuracy. It is reported alongside the raw values rather than instead of them, because
    a model that is uniformly bad has high retention while being useless - retention is
    only meaningful once clean performance is known to be adequate.
    """
    clean_accuracy = clean_metrics["accuracy"]
    adversarial_accuracy = adversarial_metrics["accuracy"]
    clean_detection = clean_metrics["detection_rate"]
    adversarial_detection = adversarial_metrics["detection_rate"]

    return {
        "clean_accuracy": clean_accuracy,
        "adversarial_accuracy": adversarial_accuracy,
        "accuracy_retention": adversarial_accuracy / clean_accuracy if clean_accuracy else 0.0,
        "absolute_accuracy_drop": clean_accuracy - adversarial_accuracy,
        "clean_detection_rate": clean_detection,
        "adversarial_detection_rate": adversarial_detection,
        "detection_retention": (
            adversarial_detection / clean_detection if clean_detection else 0.0
        ),
        "absolute_detection_drop": clean_detection - adversarial_detection,
    }


def per_attack_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attack_categories: np.ndarray,
    minimum_support: int = 20,
) -> dict:
    """
    Detection rate per attack family.

    Aggregate numbers hide the operationally important variation: a model can post a strong
    overall detection rate while missing an entire low-frequency attack class such as U2R
    (privilege escalation), which is precisely the class whose consequences are worst.
    """
    breakdown = {}
    for category in np.unique(attack_categories):
        mask = (attack_categories == category) & (y_true == 1)
        support = int(mask.sum())
        if support < minimum_support:
            continue
        breakdown[str(category)] = {
            "support": support,
            "detection_rate": float((y_pred[mask] == 1).mean()),
        }
    return dict(sorted(breakdown.items(), key=lambda kv: -kv[1]["support"]))


def summarise_table(results: dict[str, dict], columns: list[str]) -> str:
    """Render a nested results dict as a fixed-width table for logs and the report."""
    if not results:
        return "(no results)"

    name_width = max(len(str(k)) for k in results) + 2
    header = "model".ljust(name_width) + "".join(c.rjust(14) for c in columns)
    lines = [header, "-" * len(header)]

    for name, row in results.items():
        line = str(name).ljust(name_width)
        for column in columns:
            value = row.get(column)
            line += (f"{value:.4f}" if isinstance(value, (int, float)) else str(value)).rjust(14)
        lines.append(line)

    return "\n".join(lines)
