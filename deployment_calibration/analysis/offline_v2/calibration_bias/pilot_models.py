"""Simple, interpretable pilot models with block-grouped CV (EXPLORATORY / PILOT — NOT CONFIRMATORY).

Deliberately simple (logistic / ridge / small MLP) — the goal is to detect signal, not to maximise a network.
All preprocessing is fit on the training fold only; folds are grouped by block.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score, r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler


def _clf(kind, seed):
    if kind == "logreg":
        return LogisticRegression(max_iter=2000, C=1.0)
    return MLPClassifier(hidden_layer_sizes=(16,), max_iter=2000, random_state=seed, alpha=1e-2)


def _reg(kind, seed):
    if kind == "ridge":
        return Ridge(alpha=1.0)
    return MLPRegressor(hidden_layer_sizes=(16,), max_iter=2000, random_state=seed, alpha=1e-2)


def grouped_classification(X, y, groups, kind="logreg", seed=0, n_splits=None):
    """GroupKFold classification; returns per-fold balanced accuracy + macro-F1 and out-of-fold predictions."""
    X, y, groups = np.asarray(X, float), np.asarray(y, int), np.asarray(groups)
    n_groups = len(set(groups))
    n_splits = n_splits or min(5, n_groups)
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(y), -1)
    bal, f1s = [], []
    for tr, te in gkf.split(X, y, groups):
        if len(set(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        m = _clf(kind, seed).fit(sc.transform(X[tr]), y[tr])
        pred = m.predict(sc.transform(X[te]))
        oof[te] = pred
        bal.append(balanced_accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro", zero_division=0))
    return {"balanced_accuracy_mean": float(np.mean(bal)) if bal else float("nan"),
            "macro_f1_mean": float(np.mean(f1s)) if f1s else float("nan"),
            "per_fold_balanced_accuracy": [float(x) for x in bal], "oof_pred": oof.tolist()}


def grouped_regression(X, y, groups, kind="ridge", seed=0, n_splits=None):
    X, y, groups = np.asarray(X, float), np.asarray(y, float), np.asarray(groups)
    n_splits = n_splits or min(5, len(set(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    r2s, maes, preds, idx = [], [], [], []
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        m = _reg(kind, seed).fit(sc.transform(X[tr]), y[tr])
        p = m.predict(sc.transform(X[te]))
        r2s.append(r2_score(y[te], p) if len(te) > 1 else float("nan"))
        maes.append(mean_absolute_error(y[te], p))
        preds.extend(p.tolist())
        idx.extend(te.tolist())
    return {"r2_mean": float(np.nanmean(r2s)) if r2s else float("nan"),
            "mae_mean": float(np.mean(maes)) if maes else float("nan"),
            "oof_pred": [p for _, p in sorted(zip(idx, preds))]}


def grouped_success_proba(X, y, groups, kind="logreg", seed=0, n_splits=None):
    """Out-of-fold P(success) for a per-(block,offset) success model (used by the selector)."""
    X, y, groups = np.asarray(X, float), np.asarray(y, int), np.asarray(groups)
    n_splits = n_splits or min(5, len(set(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    proba = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        if len(set(y[tr])) < 2:
            proba[te] = float(np.mean(y[tr]))                 # degenerate fold -> base rate
            continue
        m = _clf(kind, seed).fit(sc.transform(X[tr]), y[tr])
        proba[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return proba
