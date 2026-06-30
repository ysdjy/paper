"""Round-1 baselines + featurizers (pure numpy; no sklearn/torch dependency).

Baselines (contract section 2):
  B0 Rule       : constant / default-theta predictor (no learning)
  B1 State-only : features = x(deploy-available) + g + theta
  B2 State+Hist : B1 + aggregated last-K action-consequence history of the same session
Vision/VLM baselines (B3-B6) are deferred (no camera/VLM in round-1).

Models are small self-contained numpy logistic + ridge regressors so the eval runs anywhere.
"""

from __future__ import annotations

import math

import numpy as np

DRAWERS = ["top_drawer", "middle_drawer"]
REGIONS = ["px_py", "px_ny", "nx_py", "nx_ny"]


def _onehot(val, vocab):
    v = [0.0] * len(vocab)
    if val in vocab:
        v[vocab.index(val)] = 1.0
    return v


def state_features(ep: dict) -> list[float]:
    """x (deploy-available) + g. handle_pose is privileged -> excluded from the fair state vector;
    a separate '+handle' ablation can add it."""
    x = ep.get("x", {})
    g = ep.get("g", {})
    jp = x.get("initial_robot_joint_position") or [0.0] * 9
    feats = []
    feats += _onehot(x.get("drawer_name"), DRAWERS)
    feats += _onehot(x.get("workspace_region"), REGIONS)
    feats.append(float(x.get("initial_drawer_position") or 0.0))
    feats.append(float(x.get("gripper_width") or 0.0))
    feats += [float(v) for v in jp[:7]]
    feats.append(float(g.get("target_open_position") or 0.0))
    return feats


def theta_features(ep: dict) -> list[float]:
    t = ep.get("theta", {})
    go = t.get("grasp_offset_local_xyz") or [0.0, 0.0, 0.0]
    return [float(t.get("max_pos_step") or 0.0), float(t.get("pull_lead") or 0.0),
            float(t.get("pre_grasp_clearance") or 0.0), float(t.get("approach_line_lead") or 0.0),
            float(go[1])]


def history_features(ep: dict, by_session: dict, K: int) -> list[float]:
    """Aggregate the last K episodes (same session, earlier order) -> deployment-state estimate."""
    sid = ep.get("session_id")
    order = ep.get("order_in_session", 0)
    hist = [e for e in by_session.get(sid, []) if e.get("order_in_session", 0) < order]
    hist = sorted(hist, key=lambda e: e.get("order_in_session", 0))[-K:]
    if not hist:
        return [0.0, 0.0, 0.0, 0.0, 0.0]  # no history -> zeros (+ count 0)
    succ = [1.0 if e["y"].get("success") else 0.0 for e in hist]
    err = [float(e["y"].get("task_outcome_error") or 0.0) for e in hist]
    tim = [float(e["y"].get("skill_elapsed_time") or 0.0) for e in hist]
    hre = [float(e["y"].get("handle_relative_error_mean") or 0.0) for e in hist]
    return [float(np.mean(succ)), float(np.mean(err)), float(np.mean(tim)), float(np.mean(hre)), float(len(hist))]


def featurize(ep: dict, mode: str, by_session: dict, K: int = 4) -> list[float]:
    if mode == "B1":
        return state_features(ep) + theta_features(ep)
    if mode == "B2":
        return state_features(ep) + theta_features(ep) + history_features(ep, by_session, K)
    raise ValueError(mode)


# --- tiny numpy models -------------------------------------------------------
class StandardScaler:
    def fit(self, X):
        self.m = X.mean(0); self.s = X.std(0); self.s[self.s < 1e-8] = 1.0; return self
    def transform(self, X):
        return (X - self.m) / self.s


class LogReg:
    """L2-regularized logistic regression via gradient descent (numpy)."""
    def __init__(self, l2=1e-2, lr=0.3, iters=2000):
        self.l2, self.lr, self.iters = l2, lr, iters
    def fit(self, X, y):
        self.sc = StandardScaler().fit(X); Xs = self.sc.transform(X)
        n, d = Xs.shape; self.w = np.zeros(d); self.b = 0.0
        y = y.astype(float)
        if y.sum() == 0 or y.sum() == n:   # degenerate (all one class)
            self.const = float(y.mean()); self.w = None; return self
        self.const = None
        for _ in range(self.iters):
            z = Xs @ self.w + self.b; p = 1 / (1 + np.exp(-z))
            gw = Xs.T @ (p - y) / n + self.l2 * self.w; gb = float((p - y).mean())
            self.w -= self.lr * gw; self.b -= self.lr * gb
        return self
    def predict_proba(self, X):
        if self.w is None:
            return np.full(X.shape[0], self.const)
        z = self.sc.transform(X) @ self.w + self.b
        return 1 / (1 + np.exp(-z))


class Ridge:
    def __init__(self, l2=1.0):
        self.l2 = l2
    def fit(self, X, y):
        self.sc = StandardScaler().fit(X); Xs = self.sc.transform(X)
        n, d = Xs.shape; A = Xs.T @ Xs + self.l2 * np.eye(d)
        self.w = np.linalg.solve(A, Xs.T @ (y - y.mean())); self.b = float(y.mean()); return self
    def predict(self, X):
        return self.sc.transform(X) @ self.w + self.b
