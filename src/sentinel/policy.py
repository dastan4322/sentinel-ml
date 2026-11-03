import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


def choose_threshold(y, scores, fp_cost=5.0, fn_cost=250.0, max_alert_rate=0.02):
    """Exact sweep over distinct score groups; ties receive the same decision."""
    y, scores = np.asarray(y), np.asarray(scores)
    if y.ndim != 1 or scores.shape != y.shape or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must be one-dimensional probabilities aligned with labels")
    if not np.isfinite([fp_cost, fn_cost, max_alert_rate]).all():
        raise ValueError("Policy parameters must be finite")
    if len(y) == 0 or not np.isfinite(scores).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid validation labels/scores")
    if fp_cost < 0 or fn_cost < 0 or not 0 <= max_alert_rate <= 1:
        raise ValueError("Invalid policy costs or alert capacity")
    order = np.argsort(-scores, kind="stable")
    s, labels = scores[order], y[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s) - 1]
    count = ends + 1
    tp = np.cumsum(labels)[ends]
    feasible = count / len(y) <= max_alert_rate
    # A threshold above one implements the legitimate no-alert policy.
    thresholds = np.r_[1.0000001, s[ends][feasible]]
    costs = np.r_[
        fn_cost * y.sum(),
        fp_cost * (count[feasible] - tp[feasible]) + fn_cost * (y.sum() - tp[feasible]),
    ]
    best = int(np.argmin(costs))
    return float(thresholds[best]), float(costs[best])


def evaluate(y, scores, threshold, fp_cost=5.0, fn_cost=250.0):
    pred = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "brier_score": float(brier_score_loss(y, scores)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "alert_rate": float(pred.mean()),
        "cost": float(fp_cost * fp + fn_cost * fn),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
