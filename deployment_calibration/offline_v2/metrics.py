"""Robust, audited prediction metrics for offline_v2.

Every metric here is written to be correct at pilot scale (small N, ties, single-class
test sets). Where sklearn provides a tie-aware reference we delegate to it; the pure-numpy
paths are only fallbacks with identical semantics and are covered by the same unit tests.

Design rules (see docs/offline_v2/pilot_reanalysis_v1.md):
  * AUROC is tie-aware (constant predictor -> 0.5, not an argsort artefact).
  * Single-class test sets return NaN and emit a warning rather than a silent number.
  * No metric silently coerces NaN into a plausible-looking value.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np

try:  # sklearn is the reference implementation; fall back to numpy if unavailable.
    from sklearn.metrics import average_precision_score, roc_auc_score
    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover - env always has sklearn, but stay safe.
    _HAVE_SKLEARN = False


def _as_float(a: Sequence) -> np.ndarray:
    return np.asarray(a, dtype=float).ravel()


def _binary_ok(y: np.ndarray) -> bool:
    """True iff y has both classes present (needed for ranking metrics)."""
    uniq = np.unique(y[~np.isnan(y)])
    return uniq.size >= 2


def auroc(y_true: Sequence, y_score: Sequence) -> float:
    """Tie-aware AUROC.

    Returns NaN (with a warning) when the test set is single-class, since AUROC is
    undefined there. A constant score column yields exactly 0.5.
    """
    y = _as_float(y_true)
    s = _as_float(y_score)
    if y.size == 0:
        warnings.warn("auroc: empty input -> NaN", RuntimeWarning)
        return float("nan")
    if not _binary_ok(y):
        warnings.warn("auroc: single-class test set -> NaN", RuntimeWarning)
        return float("nan")
    if _HAVE_SKLEARN:
        return float(roc_auc_score(y, s))
    return _auroc_numpy(y, s)


def _auroc_numpy(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney U with average ranks for ties (matches sklearn on binary labels)."""
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    n = len(s)
    # assign average ranks to tie groups
    pos = 0
    while pos < n:
        j = pos
        while j + 1 < n and s_sorted[j + 1] == s_sorted[pos]:
            j += 1
        avg = (pos + j) / 2.0 + 1.0  # 1-based average rank
        ranks[order[pos:j + 1]] = avg
        pos = j + 1
    pos_mask = y == 1
    n_pos = pos_mask.sum()
    n_neg = (y == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = ranks[pos_mask].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def auprc(y_true: Sequence, y_score: Sequence) -> float:
    """Area under precision-recall curve (average precision). NaN if single-class."""
    y = _as_float(y_true)
    s = _as_float(y_score)
    if y.size == 0 or not _binary_ok(y):
        warnings.warn("auprc: empty/single-class test set -> NaN", RuntimeWarning)
        return float("nan")
    if _HAVE_SKLEARN:
        return float(average_precision_score(y, s))
    # numpy fallback: step-wise AP
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / max(y.sum(), 1e-12)
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - prev_r)
        prev_r = r
    return float(ap)


def brier(y_true: Sequence, y_prob: Sequence) -> float:
    """Mean squared error of probabilistic prediction. Defined even for single-class."""
    y = _as_float(y_true)
    p = _as_float(y_prob)
    if y.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def calibration_error(y_true: Sequence, y_prob: Sequence, n_bins: int = 10) -> float:
    """Expected Calibration Error (equal-width bins). Small-N: fewer effective bins."""
    y = _as_float(y_true)
    p = _as_float(y_prob)
    if y.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    n = len(y)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf = p[m].mean()
        acc = y[m].mean()
        ece += (m.sum() / n) * abs(acc - conf)
    return float(ece)


def mae(y_true: Sequence, y_pred: Sequence) -> float:
    y = _as_float(y_true)
    p = _as_float(y_pred)
    if y.size == 0:
        return float("nan")
    return float(np.mean(np.abs(p - y)))


def success_accuracy(y_true: Sequence, y_prob: Sequence, thresh: float = 0.5) -> float:
    y = _as_float(y_true)
    p = _as_float(y_prob)
    if y.size == 0:
        return float("nan")
    return float(np.mean((p >= thresh).astype(float) == y))


def macro_f1_failure(y_true: Sequence, y_prob: Sequence, thresh: float = 0.5) -> float:
    """Macro-F1 over {success, failure} treating FAILURE as the positive class of interest.

    Macro-averages the F1 of both classes so a rare-failure regime is not masked by
    success-class dominance.
    """
    y = _as_float(y_true)
    pred = (_as_float(y_prob) >= thresh).astype(float)
    if y.size == 0:
        return float("nan")
    f1s = []
    for cls in (0.0, 1.0):  # 0 = failure, 1 = success
        tp = np.sum((pred == cls) & (y == cls))
        fp = np.sum((pred == cls) & (y != cls))
        fn = np.sum((pred != cls) & (y == cls))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))


def balanced_accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    """Mean of per-class recall. NaN if no labels."""
    y = np.asarray(y_true)
    p = np.asarray(y_pred)
    if y.size == 0:
        return float("nan")
    classes = np.unique(y)
    recalls = []
    for c in classes:
        m = y == c
        if m.sum() == 0:
            continue
        recalls.append(np.mean(p[m] == c))
    return float(np.mean(recalls)) if recalls else float("nan")


def macro_f1_multiclass(y_true: Sequence, y_pred: Sequence) -> float:
    y = np.asarray(y_true)
    p = np.asarray(y_pred)
    if y.size == 0:
        return float("nan")
    classes = np.unique(np.concatenate([y, p]))
    f1s = []
    for c in classes:
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))


def confusion_matrix(y_true: Sequence, y_pred: Sequence, labels: Sequence | None = None):
    """Return (labels, matrix) with rows=true, cols=pred."""
    y = np.asarray(y_true)
    p = np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(np.concatenate([y, p]).tolist()))
    labels = list(labels)
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for t, pr in zip(y, p):
        if t in idx and pr in idx:
            m[idx[t], idx[pr]] += 1
    return labels, m


def prediction_metrics(y_true, y_prob, err_true, err_pred, time_true, time_pred) -> dict:
    """Bundle all prediction-side metrics for a test fold."""
    return {
        "auroc": auroc(y_true, y_prob),
        "auprc": auprc(y_true, y_prob),
        "brier": brier(y_true, y_prob),
        "calibration_error": calibration_error(y_true, y_prob),
        "success_accuracy": success_accuracy(y_true, y_prob),
        "failure_macro_f1": macro_f1_failure(y_true, y_prob),
        "task_error_mae": mae(err_true, err_pred),
        "exec_time_mae": mae(time_true, time_pred),
        "n": int(len(y_true)),
    }
