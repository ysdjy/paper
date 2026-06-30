"""Offline evaluation for open_drawer round-1 (no Isaac).

Loads episodes.jsonl, splits by reset_index (no leakage, bias R10), trains B0/B1/B2 to predict
success + task_outcome_error, and evaluates prediction + candidate-selection (regret/top-1/Oracle gap).
Also reports H1 evidence (does theta/x move y) and a H2 check (does +history help).

Usage:
  python projects/deployment_calibration/evaluation/run_eval.py \
      --episodes projects/deployment_calibration/data/kill_test/episodes.jsonl \
      --out      projects/deployment_calibration/data/kill_test/_eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.predictors import LogReg, Ridge, featurize
from evaluation.metrics import accuracy, auroc, brier, mae, selection_metrics
from contracts.episode_schema import load_episodes


def valid(ep):
    return isinstance(ep.get("y"), dict) and "success" in ep.get("y", {})


def split_by_reset(eps, test_frac=0.4, seed=0):
    idxs = sorted({e["reset_index"] for e in eps})
    rng = np.random.default_rng(seed); rng.shuffle(idxs)
    n_test = max(1, int(round(len(idxs) * test_frac)))
    test_ri = set(idxs[:n_test])
    tr = [e for e in eps if e["reset_index"] not in test_ri]
    te = [e for e in eps if e["reset_index"] in test_ri]
    return tr, te, test_ri


def Xy(eps, mode, by_session, K=4):
    X = np.array([featurize(e, mode, by_session, K) for e in eps], float)
    ys = np.array([1 if e["y"].get("success") else 0 for e in eps], int)
    yerr = np.array([float(e["y"].get("task_outcome_error") or np.nan) for e in eps], float)
    return X, ys, yerr


def groups_of(eps):
    g = {}
    for e in eps:
        g.setdefault(e.get("candidate_group"), []).append(e)
    return [v for v in g.values() if len(v) >= 2]


def h1_evidence(eps):
    """Does theta move outcomes? corr(theta dim, task_error/time) on successes + outcome spread."""
    succ = [e for e in eps if e["y"].get("success")]
    out = {}
    if len(succ) >= 5:
        for name, fn in [("max_pos_step", lambda e: e["theta"]["max_pos_step"]),
                         ("pull_lead", lambda e: e["theta"]["pull_lead"]),
                         ("grasp_offset_y", lambda e: e["theta"]["grasp_offset_local_xyz"][1])]:
            xv = np.array([fn(e) for e in succ]);
            er = np.array([float(e["y"].get("task_outcome_error") or np.nan) for e in succ])
            tm = np.array([float(e["y"].get("skill_elapsed_time") or np.nan) for e in succ])
            def corr(a, b):
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() < 4 or a[m].std() < 1e-9 or b[m].std() < 1e-9: return None
                return float(np.corrcoef(a[m], b[m])[0, 1])
            out[name] = {"corr_task_error": corr(xv, er), "corr_time": corr(xv, tm)}
    succ_rate = float(np.mean([1 if e["y"].get("success") else 0 for e in eps]))
    errs = [float(e["y"]["task_outcome_error"]) for e in eps if e["y"].get("task_outcome_error") is not None]
    fmodes = {}
    for e in eps:
        fr = e["y"].get("failure_reason", "NONE"); fmodes[fr] = fmodes.get(fr, 0) + 1
    return {"success_rate": succ_rate, "task_error_min": (min(errs) if errs else None),
            "task_error_max": (max(errs) if errs else None), "failure_modes": fmodes, "theta_corr": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    eps = [e for e in load_episodes(args.episodes) if valid(e)]
    by_session = {}
    for e in eps:
        by_session.setdefault(e.get("session_id"), []).append(e)
    report = {"n_episodes": len(eps), "h1_evidence": h1_evidence(eps)}

    tr, te, test_ri = split_by_reset(eps, seed=args.seed)
    report["split"] = {"n_train": len(tr), "n_test": len(te), "test_reset_indices": sorted(test_ri)}
    te_groups = groups_of(te)

    results = {}
    # B0 rule: predict train success-rate as prob; selection picks default theta (max_pos_step closest to 0.02)
    base_p = float(np.mean([1 if e["y"].get("success") else 0 for e in tr])) if tr else 0.5
    def b0_score(e):
        return -abs(e["theta"]["max_pos_step"] - 0.02)  # rule: prefer mid step
    results["B0_rule"] = {
        "success_auroc": auroc([1 if e["y"].get("success") else 0 for e in te], [base_p] * len(te)),
        "success_brier": brier([1 if e["y"].get("success") else 0 for e in te], [base_p] * len(te)),
        "selection": selection_metrics(te_groups, b0_score),
    }
    # B1 / B2 learned
    for mode in ["B1", "B2"]:
        Xtr, ytr, etr = Xy(tr, mode, by_session, args.K)
        Xte, yte, ete = Xy(te, mode, by_session, args.K)
        clf = LogReg().fit(Xtr, ytr)
        pte = clf.predict_proba(Xte)
        # error regressor trained on successful train episodes
        succ_mask = (ytr == 1) & np.isfinite(etr)
        if succ_mask.sum() >= 4:
            reg = Ridge(l2=1.0).fit(Xtr[succ_mask], etr[succ_mask])
            err_pred_te = reg.predict(Xte)
        else:
            reg = None; err_pred_te = np.full(len(te), float(np.nanmean(etr)) if np.isfinite(etr).any() else 0.05)
        # per-episode predicted score for selection: P(success) - 2*pred_error
        pred_by_id = {id(e): float(pte[i] - 2.0 * (err_pred_te[i] if np.isfinite(err_pred_te[i]) else 0.1))
                      for i, e in enumerate(te)}
        def score(e, _m=pred_by_id):
            return _m.get(id(e), -1e9)
        results[mode] = {
            "success_auroc": auroc(yte, pte),
            "success_brier": brier(yte, pte),
            "success_acc": accuracy(yte, pte),
            "task_error_mae": mae(ete[yte == 1], err_pred_te[yte == 1]) if (yte == 1).any() else None,
            "selection": selection_metrics(te_groups, score),
        }
    report["results"] = results
    (out / "eval_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # console summary
    print(f"== open_drawer round-1 eval ==  episodes={len(eps)}  test={len(te)} groups={len(te_groups)}")
    print("H1:", json.dumps(report["h1_evidence"], default=str)[:400])
    for k, v in results.items():
        s = v["selection"]
        print(f"{k:8s} succ_auroc={_f(v.get('success_auroc'))} brier={_f(v.get('success_brier'))} "
              f"err_mae={_f(v.get('task_error_mae'))} | regret={_f(s['selection_regret'])} "
              f"top1={_f(s['top1_candidate_accuracy'])} sel_succ={_f(s['selected_success_rate'])}")
    print(f"report -> {out/'eval_report.json'}")


def _f(x):
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return "nan"


if __name__ == "__main__":
    main()
