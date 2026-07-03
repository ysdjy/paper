"""Numpy models for models_v2: B0, B1(static), B2Mean, OracleZ.

Small, closed-form/logistic baselines. No torch. Multitask: logistic head for success,
ridge heads for error and time. B0 is a pure base-rate rule.
"""

from __future__ import annotations

import numpy as np

from . import features as F
from .base import Model, Standardizer, clip_predictions


class _LogReg:
    def __init__(self, l2=1.0, iters=1500, lr=0.3):
        self.l2, self.iters, self.lr = l2, iters, lr

    def fit(self, X, y):
        self.st = Standardizer().fit(X)
        Xs = self.st.tf(X)
        n, d = Xs.shape
        self.w = np.zeros(d)
        self.b = 0.0
        y = y.astype(float)
        if y.min() == y.max():
            self.const = float(y.mean())
            return self
        self.const = None
        for _ in range(self.iters):
            z = Xs @ self.w + self.b
            p = 1 / (1 + np.exp(-z))
            gw = Xs.T @ (p - y) / n + self.l2 * self.w / n
            gb = float((p - y).mean())
            self.w -= self.lr * gw
            self.b -= self.lr * gb
        return self

    def prob(self, X):
        if getattr(self, "const", None) is not None:
            return np.full(len(X), self.const)
        return 1 / (1 + np.exp(-(self.st.tf(X) @ self.w + self.b)))


class _Ridge:
    def __init__(self, l2=1.0):
        self.l2 = l2

    def fit(self, X, y):
        self.st = Standardizer().fit(X)
        Xs = np.c_[self.st.tf(X), np.ones(len(X))]
        d = Xs.shape[1]
        A = Xs.T @ Xs + self.l2 * np.eye(d)
        A[-1, -1] -= self.l2
        self.w = np.linalg.solve(A, Xs.T @ y)
        return self

    def pred(self, X):
        return np.c_[self.st.tf(X), np.ones(len(X))] @ self.w


class _StaticNumpy(Model):
    """Shared logistic+ridge over a featurizer(e,H). Subclasses set name + featurize."""
    name = "static"

    def featurize(self, e, H):
        return F.static_features(e)

    def fit(self, train_pairs, val_pairs=None, seed: int = 0):
        X = np.array([self.featurize(e, H) for e, H in train_pairs], dtype=float)
        ys = np.array([F.targets(e)[0] for e, _ in train_pairs])
        ye = np.array([F.targets(e)[1] for e, _ in train_pairs])
        yt = np.array([F.targets(e)[2] for e, _ in train_pairs])
        self.clf = _LogReg().fit(X, ys)
        self.reg_e = _Ridge().fit(X, ye)
        self.reg_t = _Ridge().fit(X, yt)
        return self

    def predict(self, e, H):
        X = np.array([self.featurize(e, H)], dtype=float)
        return clip_predictions({
            "p_success": float(self.clf.prob(X)[0]),
            "pred_error": float(self.reg_e.pred(X)[0]),
            "pred_time": float(self.reg_t.pred(X)[0]),
        })


class B0(Model):
    """Base rate / default rule from train."""
    name = "B0_baserate"

    def fit(self, train_pairs, val_pairs=None, seed: int = 0):
        t = np.array([F.targets(e) for e, _ in train_pairs], dtype=float)
        self._p, self._e, self._t = float(t[:, 0].mean()), float(t[:, 1].mean()), float(t[:, 2].mean())
        return self

    def predict(self, e, H):
        return clip_predictions({"p_success": self._p, "pred_error": self._e, "pred_time": self._t})


class B1(_StaticNumpy):
    name = "B1_static"


class B2Mean(_StaticNumpy):
    name = "B2_mean"
    reads_history = True

    def featurize(self, e, H):
        return F.static_features(e) + F.history_mean(H)


class OracleZ(_StaticNumpy):
    name = "OracleZ"
    reads_secret = True

    def featurize(self, e, H):
        z = float((e.get("secret_deployment_state") or {}).get("damping", 0.0))
        return F.static_features(e) + [z]
