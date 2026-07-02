"""Baselines for drawer deployment calibration (v2). Pure numpy (eval runs without Isaac).

Predictors (predict success prob, task_outcome_error, skill_elapsed_time from an episode's decision inputs):
  B0            constant default-rule (base rates from train)
  B1            static: features(x_observable, theta, g)
  B2Mean        B1 + mean-aggregated probe history (weak history baseline)
  B2Seq         B1 + DeepSets(mean+max pool) over probe history entries (variable K)
  OracleZ       B1 + true secret damping (upper bound; audit only)
Utility (frozen): U = p_success - lam_err*pred_error - lam_time*pred_time.
Oracle-Candidate (in run_session_eval) picks the truly-best candidate for the regret lower bound.
"""

from __future__ import annotations

import numpy as np

THETA_KEYS = ["grasp_offset_local_y", "max_pos_step", "pull_lead"]


# ---------------- featurizers ----------------
def feat_static(e: dict) -> list:
    th = e["theta"]; g = e["g"]; x = e.get("x", {})
    f = [th[k] for k in THETA_KEYS]
    f += [g["target_open_position"], float(x.get("initial_mechanism_joint_pos", 0.0))]
    # mechanism identity one-hot-ish (observable): member cabinet vs sektion
    f += [1.0 if x.get("member") == "sektion_cabinet" else 0.0]
    return f


def _probe_feat(h: dict) -> list:
    th = h.get("theta", {});
    return [th.get(k, 0.0) for k in THETA_KEYS] + [
        float(h.get("success", 0.0)), float(h.get("task_outcome_error", 0.0)),
        float(h.get("skill_elapsed_time", 0.0)), float(h.get("pull_phase_duration", 0.0)),
        float(h.get("final_joint_position", 0.0))]


def feat_history_deepsets(H: list) -> list:
    if not H:
        z = [0.0] * 8
        return z + z + [0.0]                       # mean(8) + max(8) + count
    M = np.array([_probe_feat(h) for h in H], dtype=float)
    return list(M.mean(0)) + list(M.max(0)) + [float(len(H))]


def feat_history_mean(H: list) -> list:
    if not H:
        return [0.0, 0.0, 0.0, 0.0]
    M = np.array([[float(h.get("success", 0.0)), float(h.get("task_outcome_error", 0.0)),
                   float(h.get("pull_phase_duration", 0.0)), float(h.get("final_joint_position", 0.0))]
                  for h in H], dtype=float)
    return list(M.mean(0))


# ---------------- tiny numpy models ----------------
class Standardizer:
    def fit(self, X):
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-8; return self
    def tf(self, X):
        return (X - self.mu) / self.sd


class LogReg:
    def __init__(self, l2=1.0, iters=800, lr=0.3):
        self.l2, self.iters, self.lr = l2, iters, lr
    def fit(self, X, y):
        self.st = Standardizer().fit(X); Xs = self.st.tf(X)
        n, d = Xs.shape; self.w = np.zeros(d); self.b = 0.0
        y = y.astype(float)
        if y.min() == y.max():                      # degenerate: constant
            self.const = float(y.mean()); return self
        self.const = None
        for _ in range(self.iters):
            z = Xs @ self.w + self.b; p = 1 / (1 + np.exp(-z))
            gw = Xs.T @ (p - y) / n + self.l2 * self.w / n; gb = float((p - y).mean())
            self.w -= self.lr * gw; self.b -= self.lr * gb
        return self
    def prob(self, X):
        if getattr(self, "const", None) is not None:
            return np.full(len(X), self.const)
        z = self.st.tf(X) @ self.w + self.b; return 1 / (1 + np.exp(-z))


class Ridge:
    def __init__(self, l2=1.0):
        self.l2 = l2
    def fit(self, X, y):
        self.st = Standardizer().fit(X); Xs = np.c_[self.st.tf(X), np.ones(len(X))]
        d = Xs.shape[1]; A = Xs.T @ Xs + self.l2 * np.eye(d); A[-1, -1] -= self.l2  # don't penalize bias
        self.w = np.linalg.solve(A, Xs.T @ y); return self
    def pred(self, X):
        return np.c_[self.st.tf(X), np.ones(len(X))] @ self.w


# ---------------- predictor wrappers ----------------
class Predictor:
    name = "base"
    def featurize(self, e, H):
        return feat_static(e)
    def fit(self, episodes, histories):
        X = np.array([self.featurize(e, H) for e, H in zip(episodes, histories)], dtype=float)
        ys = np.array([1.0 if e["y"]["success"] else 0.0 for e in episodes])
        ye = np.array([float(e["y"]["task_outcome_error"]) for e in episodes])
        yt = np.array([float(e["y"]["skill_elapsed_time"]) for e in episodes])
        self.clf = LogReg().fit(X, ys); self.reg_e = Ridge().fit(X, ye); self.reg_t = Ridge().fit(X, yt)
        return self
    def predict(self, e, H):
        X = np.array([self.featurize(e, H)], dtype=float)
        return {"p_success": float(self.clf.prob(X)[0]),
                "pred_error": float(self.reg_e.pred(X)[0]),
                "pred_time": float(self.reg_t.pred(X)[0])}


class B0(Predictor):
    name = "B0_default"
    def fit(self, episodes, histories):
        ys = np.array([1.0 if e["y"]["success"] else 0.0 for e in episodes])
        self._p = float(ys.mean()); self._e = float(np.mean([e["y"]["task_outcome_error"] for e in episodes]))
        self._t = float(np.mean([e["y"]["skill_elapsed_time"] for e in episodes])); return self
    def predict(self, e, H):
        return {"p_success": self._p, "pred_error": self._e, "pred_time": self._t}


class B1(Predictor):
    name = "B1_static"


class B2Mean(Predictor):
    name = "B2_mean"
    def featurize(self, e, H):
        return feat_static(e) + feat_history_mean(H)


class B2Seq(Predictor):
    name = "B2_seq_deepsets"
    def featurize(self, e, H):
        return feat_static(e) + feat_history_deepsets(H)


class OracleZ(Predictor):
    name = "OracleZ"
    def featurize(self, e, H):
        z = float((e.get("secret_deployment_state") or {}).get("damping", 0.0))
        return feat_static(e) + [z]


ALL_PREDICTORS = [B0, B1, B2Mean, B2Seq, OracleZ]
