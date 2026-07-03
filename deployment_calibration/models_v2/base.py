"""Unified model interface for models_v2.

Every model maps (candidate episode e, leakage-safe history H) -> {p_success, pred_error,
pred_time}. Models are trained on a list of (episode, history) pairs from TRAIN sessions only.

Two families:
  * numpy models (B0/B1/B2Mean/OracleZ): closed-form or logistic/ridge, no torch.
  * torch models (DeepSets/GRU): trained with early stopping on a validation fold.

The base handles standardization of the static feature block and a multitask head contract.
"""

from __future__ import annotations

import numpy as np

from . import features as F


class Standardizer:
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-8
        return self

    def tf(self, X):
        return (np.asarray(X, dtype=float) - self.mu) / self.sd


class Model:
    name = "base"
    reads_history = False
    reads_secret = False

    def fit(self, train_pairs, val_pairs=None, seed: int = 0):  # train_pairs: [(e,H),...]
        raise NotImplementedError

    def predict(self, e: dict, H: list) -> dict:
        raise NotImplementedError

    def predict_batch(self, pairs) -> list:
        return [self.predict(e, H) for e, H in pairs]

    # ---- serialization (subclasses may override) ----
    def config(self) -> dict:
        return {"name": self.name, "reads_history": self.reads_history,
                "reads_secret": self.reads_secret}


def clip_predictions(pred: dict) -> dict:
    """Physical clip for regression heads (bounds fixed, see utility.ERROR_CLIP/TIME_CLIP)."""
    from ..offline_v2.utility import ERROR_CLIP, TIME_CLIP
    pred = dict(pred)
    pred["pred_error"] = float(min(max(pred["pred_error"], ERROR_CLIP[0]), ERROR_CLIP[1]))
    pred["pred_time"] = float(min(max(pred["pred_time"], TIME_CLIP[0]), TIME_CLIP[1]))
    pred["p_success"] = float(min(max(pred["p_success"], 0.0), 1.0))
    return pred
