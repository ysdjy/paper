"""Stage-2/3 evaluation (v2): leakage-safe history build + baselines + D1-D5 + regret + adaptation curve.

Splits by session, builds each candidate's history ONLY from that session's probe episodes (order <
candidate), trains B0/B1/B2-Mean/B2-Seq/Oracle-Z on train, evaluates on test. Utility frozen:
U = p_success - LAM_ERR*pred_error - LAM_TIME*pred_time.

    python projects/paper/deployment_calibration/evaluation/run_session_eval_v2.py <run_dir> [--k 0,1,2,4]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PAPER / "deployment_calibration"))
sys.path.insert(0, str(_PAPER / "deployment_calibration" / "evaluation"))

from baselines.predictors_v2 import ALL_PREDICTORS, B1, B2Seq  # noqa: E402
from session_split_v2 import load_episodes, split_sessions, split_manifest  # noqa: E402

LAM_ERR, LAM_TIME = 1.0, 0.02      # frozen utility weights (documented; not tuned on test)


def _hist_entry(e: dict) -> dict:
    y = e["y"]
    return {"theta": e["theta"], "g": e["g"], "success": y["success"],
            "failure_reason": y["failure_reason"], "task_outcome_error": y["task_outcome_error"],
            "skill_elapsed_time": y["skill_elapsed_time"],
            "pull_phase_duration": y.get("phase_durations", {}).get("PULL", 0.0),
            "final_joint_position": y["final_joint_position"], "mechanism_id": e["mechanism_id"]}


def build_history(episodes, session_id, before_order, k=None):
    """Probes of THIS session with order < before_order (decision-legal). Truncate to first k."""
    probes = sorted([e for e in episodes if e["session_id"] == session_id
                     and e.get("episode_role") == "probe" and e["order_in_session"] < before_order],
                    key=lambda e: e["order_in_session"])
    if k is not None:
        probes = probes[:k]
    return [_hist_entry(p) for p in probes]


def candidates_in(episodes, sids):
    return [e for e in episodes if e["session_id"] in sids and e.get("episode_role") == "candidate"]


def auroc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    pos = y == 1; neg = y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(p); ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def true_utility(e):
    y = e["y"]
    return (1.0 if y["success"] else 0.0) - LAM_ERR * y["task_outcome_error"] - LAM_TIME * y["skill_elapsed_time"]


def eval_predictor(P, train_eps, train_H, test_eps, test_H, episodes, k):
    P.fit(train_eps, train_H)
    ys = np.array([1.0 if e["y"]["success"] else 0.0 for e in test_eps])
    preds = [P.predict(e, H) for e, H in zip(test_eps, test_H)]
    ps = np.array([pr["p_success"] for pr in preds])
    err_mae = float(np.mean(np.abs([pr["pred_error"] for pr in preds] - np.array([e["y"]["task_outcome_error"] for e in test_eps]))))
    time_mae = float(np.mean(np.abs([pr["pred_time"] for pr in preds] - np.array([e["y"]["skill_elapsed_time"] for e in test_eps]))))
    # regret over candidate groups
    groups = {}
    for e, pr in zip(test_eps, preds):
        gid = e.get("candidate_group", e["session_id"])
        U = pr["p_success"] - LAM_ERR * pr["pred_error"] - LAM_TIME * pr["pred_time"]
        groups.setdefault(gid, []).append((U, true_utility(e), e["y"]["success"]))
    regrets, top1, sel_succ = [], [], []
    for gid, cand in groups.items():
        best_true = max(c[1] for c in cand)
        pick = max(cand, key=lambda c: c[0])
        regrets.append(best_true - pick[1])
        top1.append(1.0 if pick[1] >= best_true - 1e-9 else 0.0)
        sel_succ.append(1.0 if pick[2] else 0.0)
    return {"auroc": auroc(ys, ps), "brier": brier(ys, ps), "err_mae": round(err_mae, 4),
            "time_mae": round(time_mae, 3), "regret": round(float(np.mean(regrets)), 4),
            "top1_acc": round(float(np.mean(top1)), 3), "sel_success_rate": round(float(np.mean(sel_succ)), 3),
            "n_test": len(test_eps), "n_groups": len(groups)}


def diagnostics(episodes):
    """D1 (damping affects y), D2 (damping leaks from x), D3 (history identifies state)."""
    import itertools
    probes = [e for e in episodes if e.get("episode_role") == "probe"]
    out = {}
    # D1: for each (drawer, probe theta signature), spread of mean final across damping levels
    by = {}
    for e in probes:
        key = (e["drawer_name"], round(e["theta"]["pull_lead"], 3), round(e["theta"]["max_pos_step"], 3))
        by.setdefault(key, {}).setdefault(e["hidden_state_id"], []).append(e["y"]["final_joint_position"])
    spreads_final, spreads_pull = [], []
    for key, lv in by.items():
        means = {k: float(np.mean(v)) for k, v in lv.items()}
        if len(means) >= 2:
            spreads_final.append(max(means.values()) - min(means.values()))
    for e in probes:
        pass
    bypull = {}
    for e in probes:
        key = (e["drawer_name"], round(e["theta"]["pull_lead"], 3))
        bypull.setdefault(key, {}).setdefault(e["hidden_state_id"], []).append(e["y"].get("phase_durations", {}).get("PULL", 0.0))
    for key, lv in bypull.items():
        means = {k: float(np.mean(v)) for k, v in lv.items()}
        if len(means) >= 2:
            spreads_pull.append(max(means.values()) - min(means.values()))
    out["D1_damping_affects_final_maxspread"] = round(float(np.mean(spreads_final)), 4) if spreads_final else 0.0
    out["D1_damping_affects_pulldur_maxspread"] = round(float(np.mean(spreads_pull)), 4) if spreads_pull else 0.0
    # D2: variance of initial x (robot joints + drawer init) across damping -> should be ~0
    xs = np.array([[float(e["x"].get("initial_mechanism_joint_pos", 0.0))] + list(e["x"].get("tcp_pos", [0, 0, 0]))
                   for e in probes], dtype=float)
    out["D2_x_std_across_all"] = round(float(xs.std(0).max()), 6) if len(xs) else 0.0
    # D3: can probe history predict damping level? nearest-centroid on per-session probe feature means
    from baselines.predictors_v2 import feat_history_deepsets
    sess = {}
    for e in probes:
        sess.setdefault(e["session_id"], {"lvl": e["hidden_state_id"], "H": []})
        sess[e["session_id"]]["H"].append(_hist_entry(e))
    feats, labels = [], []
    for sid, s in sess.items():
        feats.append(feat_history_deepsets(s["H"])); labels.append(s["lvl"])
    acc = float("nan")
    if len(set(labels)) >= 2 and len(feats) >= 4:
        F = np.array(feats); L = np.array(labels)
        cent = {l: F[L == l].mean(0) for l in set(labels)}
        pred = [min(cent, key=lambda l: np.linalg.norm(f - cent[l])) for f in F]
        acc = round(float(np.mean([p == t for p, t in zip(pred, L)])), 3)
    out["D3_history_identifies_state_acc"] = acc
    out["D3_chance"] = round(1.0 / max(1, len(set(labels))), 3)
    return out


def main() -> int:
    apx = argparse.ArgumentParser()
    apx.add_argument("run_dir")
    apx.add_argument("--k", default="0,1,2,4")
    apx.add_argument("--fracs", default="0.34,0.33,0.33", help="train,val,test session fractions")
    a = apx.parse_args()
    run = Path(a.run_dir)
    episodes = load_episodes(run)
    fr = tuple(float(x) for x in a.fracs.split(","))
    split = split_sessions(episodes, fracs=fr, seed=0)
    man = split_manifest(episodes, split)

    train_c = candidates_in(episodes, split["train"] + split["val"])
    test_c = candidates_in(episodes, split["test"])
    Ks = [int(x) for x in a.k.split(",")]
    Kfull = max([e.get("history_cutoff", 3) for e in episodes] + [3])

    def hists(cands, k):
        return [build_history(episodes, e["session_id"], e["order_in_session"], k=k) for e in cands]

    results = {}
    for P_cls in ALL_PREDICTORS:
        P = P_cls()
        results[P.name] = eval_predictor(P, train_c, hists(train_c, Kfull), test_c, hists(test_c, Kfull), episodes, Kfull)

    # adaptation curve: B2Seq at K=0,1,2,4
    adapt = {}
    for k in Ks:
        P = B2Seq()
        adapt[k] = eval_predictor(P, train_c, hists(train_c, k), test_c, hists(test_c, k), episodes, k)

    diag = diagnostics(episodes)
    meta = json.loads((run / "metadata.json").read_text()) if (run / "metadata.json").exists() else {}
    out = {"run": run.name, "split": man["counts"], "disjoint_split": man["disjoint"],
           "n_train_cand": len(train_c), "n_test_cand": len(test_c),
           "damping_verified": meta.get("damping_verified"), "predictors": results,
           "adaptation_curve_B2Seq": adapt, "diagnostics": diag,
           "utility": {"LAM_ERR": LAM_ERR, "LAM_TIME": LAM_TIME}}
    (run / "evaluation").mkdir(exist_ok=True)
    json.dump(out, open(run / "evaluation" / "eval_v2.json", "w"), indent=1)
    json.dump(man, open(run / "split_manifest.json", "w"), indent=1)

    print(json.dumps({"split": out["split"], "diagnostics": diag}, indent=1))
    print("\npredictor           AUROC  Brier  errMAE  timeMAE  regret  top1  selSucc")
    for nm, r in results.items():
        print(f"{nm:18s} {r['auroc']!s:>6} {r['brier']:.3f} {r['err_mae']:>6} {r['time_mae']:>7} "
              f"{r['regret']:>7} {r['top1_acc']:>5} {r['sel_success_rate']:>7}")
    print("\nadaptation (B2Seq) K -> regret / auroc:")
    for k, r in adapt.items():
        print(f"  K={k}: regret={r['regret']} auroc={r['auroc']} selSucc={r['sel_success_rate']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
