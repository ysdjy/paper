"""D3 held-out hidden-state identification (offline_v2).

D3 asks: can the probe history alone identify the hidden deployment state (damping level)?
This is a DIAGNOSTIC of state observability, NOT part of any deployable model, and it must
NEVER read candidate outcomes.

The original v2 code fit a nearest-centroid on ALL sessions and evaluated on the SAME sessions
(acc = 1.0 by construction). We replace that with honest held-out estimation:

  * leave_one_session_out: centroid from all OTHER sessions predicts the held-out session.
  * train_test: centroid from train sessions, evaluated on held-out test sessions.

Reports balanced accuracy, macro-F1, confusion matrix, and a session-bootstrap CI on accuracy.
Features come ONLY from probe episodes (per-session probe summary).
"""

from __future__ import annotations

import numpy as np

from .bootstrap import session_bootstrap
from .metrics import balanced_accuracy, confusion_matrix, macro_f1_multiclass


def _probe_feat(probe: dict) -> list:
    y = probe["y"]
    th = probe["theta"]
    return [
        float(th.get("grasp_offset_local_y", 0.0)),
        float(th.get("max_pos_step", 0.0)),
        float(th.get("pull_lead", 0.0)),
        1.0 if y["success"] else 0.0,
        float(y["task_outcome_error"]),
        float(y["skill_elapsed_time"]),
        float(y.get("phase_durations", {}).get("PULL", 0.0)),
        float(y["final_joint_position"]),
    ]


def session_features(episodes) -> dict:
    """session_id -> {"feat": mean probe feature vector, "label": hidden_state_id}."""
    probes = [e for e in episodes if e.get("episode_role") == "probe"]
    by_sess: dict = {}
    for e in probes:
        s = by_sess.setdefault(e["session_id"], {"feats": [], "label": e.get("hidden_state_id")})
        s["feats"].append(_probe_feat(e))
    return {sid: {"feat": np.mean(v["feats"], axis=0), "label": v["label"]}
            for sid, v in by_sess.items()}


def _nearest_centroid_predict(train_items, test_feat):
    labels = sorted({lbl for _, lbl in train_items})
    cents = {}
    for lbl in labels:
        F = np.array([f for f, l in train_items if l == lbl])
        cents[lbl] = F.mean(axis=0)
    # standardize distance by per-dim std across train (avoid a dominant-scale feature)
    allF = np.array([f for f, _ in train_items])
    sd = allF.std(axis=0) + 1e-8
    return min(cents, key=lambda l: np.linalg.norm((test_feat - cents[l]) / sd))


def d3_leave_one_session_out(episodes, n_boot: int = 2000, seed: int = 0) -> dict:
    feats = session_features(episodes)
    sids = sorted(feats)
    labels = [feats[s]["label"] for s in sids]
    if len(set(labels)) < 2:
        return {"available": False, "reason": "single hidden-state class"}
    preds = {}
    for held in sids:
        train_items = [(feats[s]["feat"], feats[s]["label"]) for s in sids if s != held]
        preds[held] = _nearest_centroid_predict(train_items, feats[held]["feat"])
    y_true = [feats[s]["label"] for s in sids]
    y_pred = [preds[s] for s in sids]
    lab, cm = confusion_matrix(y_true, y_pred)
    correct = {s: 1.0 if preds[s] == feats[s]["label"] else 0.0 for s in sids}

    def acc_stat(resampled):
        vals = [correct[s] for s in resampled]
        return float(np.mean(vals)) if vals else float("nan")

    boot = session_bootstrap(sids, acc_stat, n_boot=n_boot, seed=seed)
    n_classes = len(set(labels))
    return {
        "available": True,
        "protocol": "leave_one_session_out",
        "accuracy": float(np.mean(list(correct.values()))),
        "accuracy_ci": {"ci_low": boot["ci_low"], "ci_high": boot["ci_high"], "n_boot": boot["n_boot"]},
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "macro_f1": macro_f1_multiclass(y_true, y_pred),
        "chance": 1.0 / n_classes,
        "n_sessions": len(sids),
        "n_classes": n_classes,
        "confusion_labels": lab,
        "confusion_matrix": cm.tolist(),
        "per_session": {s: {"true": feats[s]["label"], "pred": preds[s]} for s in sids},
    }


def d3_train_test(episodes, train_sids, test_sids) -> dict:
    feats = session_features(episodes)
    train_items = [(feats[s]["feat"], feats[s]["label"]) for s in train_sids if s in feats]
    if len({l for _, l in train_items}) < 2:
        return {"available": False, "reason": "train has <2 hidden-state classes"}
    y_true, y_pred = [], []
    for s in test_sids:
        if s not in feats:
            continue
        y_true.append(feats[s]["label"])
        y_pred.append(_nearest_centroid_predict(train_items, feats[s]["feat"]))
    if not y_true:
        return {"available": False, "reason": "no test sessions with probes"}
    lab, cm = confusion_matrix(y_true, y_pred)
    return {
        "available": True,
        "protocol": "train_centroid_test_eval",
        "accuracy": float(np.mean([1.0 if p == t else 0.0 for p, t in zip(y_pred, y_true)])),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "macro_f1": macro_f1_multiclass(y_true, y_pred),
        "chance": 1.0 / len(set(y_true)) if y_true else float("nan"),
        "n_test_sessions": len(y_true),
        "confusion_labels": lab,
        "confusion_matrix": cm.tolist(),
    }
