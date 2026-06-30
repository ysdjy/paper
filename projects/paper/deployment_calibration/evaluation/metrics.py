"""Prediction + candidate-selection metrics (pure numpy). Contract section 4."""

from __future__ import annotations

import numpy as np


def auroc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    pos = p[y == 1]; neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney U / (n_pos*n_neg)
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), float); ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def accuracy(y, p, thr=0.5):
    return float(np.mean((np.asarray(p) >= thr).astype(int) == np.asarray(y)))


def mae(y, yh):
    y = np.asarray(y, float); yh = np.asarray(yh, float)
    m = np.isfinite(y) & np.isfinite(yh)
    return float(np.mean(np.abs(y[m] - yh[m]))) if m.any() else float("nan")


def f1_macro(y_true, y_pred, labels):
    f1s = []
    for c in labels:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        if tp + fp == 0 or tp + fn == 0:
            continue
        prec = tp / (tp + fp); rec = tp / (tp + fn)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else float("nan")


def candidate_utility(success: bool, task_err, alpha=2.0):
    """Utility used for selection + regret: success minus alpha*spatial error (failures penalized)."""
    e = float(task_err) if (task_err is not None and np.isfinite(task_err)) else 0.2
    return (1.0 if success else 0.0) - alpha * e


def selection_metrics(groups, score_fn, alpha=2.0):
    """groups: list of list-of-episodes (same x,g, different theta candidates).
    score_fn(ep)->predicted score (higher=better). Returns regret/top1/selected success."""
    regrets, top1, sel_succ, sel_err = [], [], [], []
    for g in groups:
        if len(g) < 2:
            continue
        util = [candidate_utility(e["y"].get("success"), e["y"].get("task_outcome_error"), alpha) for e in g]
        oracle = int(np.argmax(util))
        pick = int(np.argmax([score_fn(e) for e in g]))
        regrets.append(util[oracle] - util[pick])
        top1.append(1.0 if util[pick] >= max(util) - 1e-9 else 0.0)
        sel_succ.append(1.0 if g[pick]["y"].get("success") else 0.0)
        e = g[pick]["y"].get("task_outcome_error")
        if e is not None and np.isfinite(e):
            sel_err.append(float(e))
    return {
        "n_groups": len(regrets),
        "selection_regret": float(np.mean(regrets)) if regrets else float("nan"),
        "top1_candidate_accuracy": float(np.mean(top1)) if top1 else float("nan"),
        "selected_success_rate": float(np.mean(sel_succ)) if sel_succ else float("nan"),
        "selected_task_error_mean": float(np.mean(sel_err)) if sel_err else float("nan"),
    }
