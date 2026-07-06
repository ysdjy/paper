"""Scientific pilot analysis over the 306-episode band-edge exploratory run (EXPLORATORY / PILOT — NOT
CONFIRMATORY). Answers Q1 oracle headroom, Q2 probe→hidden-state identifiability, Q3 K0/K1 candidate selection,
plus a history ablation, a leakage check, and block-bootstrap CIs. Writes inventory + report + summary + figures
+ tables. Emits a single Go/No-Go pilot status. Does NOT launch Isaac and does NOT touch confirmatory seeds.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from deployment_calibration.analysis.offline_v2.calibration_bias import pilot_data_loader as DL
from deployment_calibration.analysis.offline_v2.calibration_bias import pilot_features as FT
from deployment_calibration.analysis.offline_v2.calibration_bias import pilot_models as MD
from deployment_calibration.analysis.offline_v2.calibration_bias import pilot_statistics as ST

DOCS = Path(__file__).resolve().parents[4] / "docs/offline_v2/calibration_bias"
FIGS = DOCS / "figures"
TABLES = DOCS / "tables"
HEADROOM_THRESH = 0.15
IDENT_ACC_THRESH, IDENT_GAIN_THRESH = 0.65, 0.10
SELECT_GAIN_THRESH = 0.10
LABEL = "EXPLORATORY / PILOT — NOT CONFIRMATORY"


def _save_fig(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.suptitle(LABEL, fontsize=8, y=0.995, color="crimson")
    fig.tight_layout()
    fig.savefig(FIGS / name, dpi=110)
    plt.close(fig)


def _write_csv(name, rows, header):
    TABLES.mkdir(parents=True, exist_ok=True)
    with open(TABLES / name, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


# --------------------------------------------------------------------------- inventory
def build_inventory(cap, episodes, data_path):
    cells, blocks, offsets, hidden = cap["cells"], cap["blocks"], cap["offsets"], cap["hidden"]
    complete = all((b, o) in cells for b in blocks for o in offsets)
    probe_ok = all(cap["probe"][b] is not None for b in blocks)
    resid = [hidden[b]["residual_bias_y"] for b in blocks]
    inv = {
        "label": LABEL, "data_path": str(data_path), "data_sha256": DL.data_sha256(data_path),
        "n_episodes": len(episodes), "n_blocks": len(blocks), "n_offsets": len(offsets), "offsets": offsets,
        "trials_per_block": {str(b): sum(1 for o in offsets if (b, o) in cells) for b in blocks},
        "grid_complete_18x17": complete, "emulated_probe_offset": DL.PROBE_OFFSET, "probe_present_all_blocks": probe_ok,
        "nominal_bias_y_set": sorted({round(hidden[b]["nominal_bias_y"], 4) for b in blocks}),
        "residual_bias_y_min": min(resid), "residual_bias_y_max": max(resid),
        "native_session_probe_plus_3candidate_design": False,
        "native_role_counts": {"candidate": len(episodes), "probe": 0},
        "fields_present": {k: True for k in ("block_id", "block_seed", "planned_episode_id", "theta.grasp_offset_local_y",
                                             "secret_deployment_state.residual_bias_y", "secret_deployment_state.nominal_bias_y",
                                             "y.success", "y.task_outcome_error", "y.phase_goal_error",
                                             "y.phase_durations", "x.robot_joint_pos")},
        "missing_for_native_probe_candidate_design": ["episode_role=probe", "explicit per-session candidate triplet"],
        "sufficiency": ("SUFFICIENT for block-level oracle-headroom + probe-EMULATED identifiability/selection; "
                        "does NOT contain a native probe+3-candidate session design, and all blocks share "
                        "nominal_bias_y=0 so hidden-state variation is limited to small residual draws."),
    }
    return inv


# --------------------------------------------------------------------------- Q1 oracle headroom
def analysis_oracle_headroom(cap):
    cells, blocks, offsets, hidden = cap["cells"], cap["blocks"], cap["offsets"], cap["hidden"]
    succ = {(b, o): DL.success(cells[(b, o)]) for b in blocks for o in offsets}
    oracle = np.array([1 if any(succ[(b, o)] for o in offsets) else 0 for b in blocks], float)
    # best-single via leave-one-block-out: pick offset with max success on the OTHER blocks, score held-out block
    bs = []
    for i, b in enumerate(blocks):
        train = [bb for bb in blocks if bb != b]
        counts = {o: sum(succ[(bb, o)] for bb in train) for o in offsets}
        best_o = max(offsets, key=lambda o: counts[o])
        bs.append(succ[(b, best_o)])
    bs = np.array(bs, float)
    per_off = {o: float(np.mean([succ[(b, o)] for b in blocks])) for o in offsets}
    headroom = ST.paired_block_bootstrap_ci(oracle, bs)
    # candidate winner vs residual (best offset by native success, ties -> closest to -residual)
    winners = []
    for b in blocks:
        good = [o for o in offsets if succ[(b, o)]]
        w = min(good, key=lambda o: abs(o + hidden[b]["residual_bias_y"])) if good else None
        winners.append((b, hidden[b]["residual_bias_y"], w, sum(succ[(b, o)] for o in offsets)))
    # figures
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["best-single\n(LOBO)", "oracle"], [bs.mean(), oracle.mean()], color=["#4C72B0", "#55A868"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("selected success (native)")
    ax.set_title(f"Q1 oracle headroom = {headroom['point']:+.3f} "
                 f"[{headroom['lo']:+.3f},{headroom['hi']:+.3f}]")
    _save_fig(fig, "fig_pilot_oracle_headroom.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(o) for o in offsets], [per_off[o] for o in offsets], color="#C44E52")
    ax.axhline(1.0, ls=":", c="k"); ax.set_ylabel("success rate over 18 blocks"); ax.set_xlabel("offset")
    ax.set_title("Q1 candidate success by offset (band edge at |offset|>=0.03)")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    _save_fig(fig, "fig_pilot_candidate_success_by_residual.png")

    _write_csv("pilot_session_table.csv",
               [[b, round(hidden[b]["residual_bias_y"], 5), round(hidden[b]["nominal_bias_y"], 4),
                 w, n, int(oracle[i]), int(bs[i])] for i, (b, _, w, n) in enumerate(winners)],
               ["block", "residual_bias_y", "nominal_bias_y", "best_offset", "n_success_of_17",
                "oracle_success", "best_single_success"])
    return {"oracle_success_mean": float(oracle.mean()), "best_single_success_mean": float(bs.mean()),
            "oracle_headroom": headroom, "per_offset_success": per_off,
            "n_blocks_any_success": int(oracle.sum()), "go_nogo_1_pass": bool(headroom["point"] >= HEADROOM_THRESH)}


# --------------------------------------------------------------------------- Q2 identifiability
def analysis_identifiability(cap, seeds=(0, 1, 2, 3, 4)):
    blocks, hidden, probe = cap["blocks"], cap["hidden"], cap["probe"]
    resid = np.array([hidden[b]["residual_bias_y"] for b in blocks], float)
    sign = (resid > 0).astype(int)
    # K0 = static block features (probe episode's STATIC part only, no probe outcome)
    X0 = np.array([[v for v in FT.static_features(probe[b]).values()] for b in blocks], float)
    k0_names = list(FT.static_features(probe[blocks[0]]).keys())
    # K1 = K0 + probe history
    def k1_row(b):
        d = FT.static_features(probe[b]); d.update(FT.probe_history_features(probe[b]))
        return d
    k1_names = list(k1_row(blocks[0]).keys())
    X1 = np.array([[v for v in k1_row(b).values()] for b in blocks], float)
    groups = np.array(blocks)
    leak = FT.leakage_check(k1_names)

    def multiseed_cls(X):
        accs = [MD.grouped_classification(X, sign, groups, "logreg", s)["balanced_accuracy_mean"] for s in seeds]
        return accs
    k0_acc, k1_acc = multiseed_cls(X0), multiseed_cls(X1)
    k0_reg = MD.grouped_regression(X1[:, :X0.shape[1]], resid, groups, "ridge")  # K0 features only
    k1_reg = MD.grouped_regression(X1, resid, groups, "ridge")
    # feature vs residual correlations (probe features)
    corrs = {}
    for j, name in enumerate(k1_names):
        col = X1[:, j]
        corrs[name] = float(np.corrcoef(col, resid)[0, 1]) if np.std(col) > 0 else float("nan")
    top = sorted(((abs(v), k, v) for k, v in corrs.items() if not np.isnan(v)), reverse=True)[:6]

    fig, ax = plt.subplots(figsize=(6, 4))
    key = "probe_phase_goal_error"
    xv = X1[:, k1_names.index(key)] if key in k1_names else X1[:, 0]
    ax.scatter(xv, resid, c=sign, cmap="coolwarm", s=40)
    ax.set_xlabel(key); ax.set_ylabel("hidden residual_bias_y (label)")
    ax.set_title(f"Q2 probe feature vs hidden residual (r={corrs.get(key, float('nan')):+.2f})")
    _save_fig(fig, "fig_pilot_probe_feature_vs_residual.png")

    k0_ci = ST.block_bootstrap_ci(np.array(k0_acc)); k1_ci = ST.block_bootstrap_ci(np.array(k1_acc))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["K0 (static)", "K1 (static+probe)"], [np.mean(k0_acc), np.mean(k1_acc)],
           yerr=[[np.mean(k0_acc) - k0_ci["lo"], np.mean(k1_acc) - k1_ci["lo"]],
                 [k0_ci["hi"] - np.mean(k0_acc), k1_ci["hi"] - np.mean(k1_acc)]],
           color=["#8172B3", "#CCB974"], capsize=5)
    ax.axhline(0.5, ls=":", c="k", label="chance"); ax.axhline(IDENT_ACC_THRESH, ls="--", c="green")
    ax.set_ylabel("residual-sign balanced accuracy"); ax.set_ylim(0, 1.05); ax.legend()
    ax.set_title("Q2 hidden-state identifiability (5 seeds)")
    _save_fig(fig, "fig_pilot_hidden_state_identifiability.png")

    k1_mean, k0_mean = float(np.mean(k1_acc)), float(np.mean(k0_acc))
    return {"k0_signacc_mean": k0_mean, "k1_signacc_mean": k1_mean, "k1_minus_k0_signacc": k1_mean - k0_mean,
            "k0_signacc_seeds": k0_acc, "k1_signacc_seeds": k1_acc,
            "k0_residual_r2": k0_reg["r2_mean"], "k1_residual_r2": k1_reg["r2_mean"],
            "k1_residual_mae": k1_reg["mae_mean"], "top_probe_feature_correlations": [(k, round(v, 3)) for _, k, v in top],
            "leakage_check": leak, "n_blocks": len(blocks), "n_pos_resid": int(sign.sum()),
            "go_nogo_2_pass": bool(k1_mean >= IDENT_ACC_THRESH and (k1_mean - k0_mean) >= IDENT_GAIN_THRESH)}


# --------------------------------------------------------------------------- Q3 selector
def analysis_selector(cap, seeds=(0, 1, 2, 3, 4)):
    cells, blocks, offsets, probe = cap["cells"], cap["blocks"], cap["offsets"], cap["probe"]
    candidate_offsets = [o for o in offsets if o != DL.PROBE_OFFSET]          # exclude the emulated probe
    succ = {(b, o): DL.success(cells[(b, o)]) for b in blocks for o in candidate_offsets}
    rows, groups, yv = [], [], []
    k0_rows, k1_rows = [], []
    for b in blocks:
        ph = FT.probe_history_features(probe[b])
        for o in candidate_offsets:
            st = FT.static_features(cells[(b, o)])
            k0_rows.append(list(st.values()))
            d = dict(st); d.update(ph); k1_rows.append(list(d.values()))
            groups.append(b); yv.append(succ[(b, o)]); rows.append((b, o))
    X0, X1, yv, groups = np.array(k0_rows, float), np.array(k1_rows, float), np.array(yv, int), np.array(groups)
    oracle = np.array([1 if any(succ[(b, o)] for o in candidate_offsets) else 0 for b in blocks], float)
    # best-single over candidate set (LOBO)
    bs = []
    for b in blocks:
        tr = [bb for bb in blocks if bb != b]
        counts = {o: sum(succ[(bb, o)] for bb in tr) for o in candidate_offsets}
        best_o = max(candidate_offsets, key=lambda o: counts[o]); bs.append(succ[(b, best_o)])
    bs = np.array(bs, float)

    def selected_success(X, seed):
        proba = MD.grouped_success_proba(X, yv, groups, "logreg", seed)
        sel = []
        for b in blocks:
            idx = [i for i, (bb, _) in enumerate(rows) if bb == b]
            best = max(idx, key=lambda i: (proba[i] if not np.isnan(proba[i]) else -1))
            sel.append(yv[best])
        return np.array(sel, float)

    k0_sel = np.mean([selected_success(X0, s) for s in seeds], axis=0)
    k1_sel = np.mean([selected_success(X1, s) for s in seeds], axis=0)
    k0_seed_means = [float(selected_success(X0, s).mean()) for s in seeds]
    k1_seed_means = [float(selected_success(X1, s).mean()) for s in seeds]
    diff = ST.paired_block_bootstrap_ci(k1_sel, k0_sel)
    diff_bs = ST.paired_block_bootstrap_ci(k1_sel, bs)
    regret_k1 = ST.block_bootstrap_ci(oracle - k1_sel)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    vals = [bs.mean(), k0_sel.mean(), k1_sel.mean(), oracle.mean()]
    ax.bar(["best-single", "K0", "K1", "oracle"], vals, color=["#4C72B0", "#8172B3", "#CCB974", "#55A868"])
    ax.set_ylim(0, 1.08); ax.set_ylabel("selected success (native)")
    ax.set_title(f"Q3 selector: K1-K0={diff['point']:+.3f} [{diff['lo']:+.3f},{diff['hi']:+.3f}]")
    _save_fig(fig, "fig_pilot_k0_k1_selected_success.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    r0 = ST.block_bootstrap_ci(oracle - k0_sel)
    ax.bar(["K0 regret", "K1 regret"], [r0["point"], regret_k1["point"]],
           yerr=[[r0["point"] - r0["lo"], regret_k1["point"] - regret_k1["lo"]],
                 [r0["hi"] - r0["point"], regret_k1["hi"] - regret_k1["point"]]], capsize=5, color="#C44E52")
    ax.set_ylabel("regret to oracle"); ax.set_title("Q3 selector regret to oracle")
    _save_fig(fig, "fig_pilot_selector_regret.png")

    _write_csv("pilot_fold_metrics.csv",
               [["K0", s, m] for s, m in zip(seeds, k0_seed_means)] + [["K1", s, m] for s, m in zip(seeds, k1_seed_means)],
               ["model", "seed", "selected_success"])
    _write_csv("pilot_bootstrap_metrics.csv",
               [["oracle_headroom_selectorset", diff_bs["point"], diff_bs["lo"], diff_bs["hi"]],
                ["k1_minus_k0", diff["point"], diff["lo"], diff["hi"]],
                ["k1_regret", regret_k1["point"], regret_k1["lo"], regret_k1["hi"]]],
               ["metric", "point", "ci_lo", "ci_hi"])
    seeds_same_dir = sum(1 for a, b_ in zip(k1_seed_means, k0_seed_means) if a - b_ >= 0)
    return {"best_single_success": float(bs.mean()), "k0_selected_success": float(k0_sel.mean()),
            "k1_selected_success": float(k1_sel.mean()), "oracle_success": float(oracle.mean()),
            "k1_minus_k0": diff, "k1_minus_best_single": diff_bs, "k1_regret_to_oracle": regret_k1,
            "k0_seed_means": k0_seed_means, "k1_seed_means": k1_seed_means, "seeds_k1_ge_k0": seeds_same_dir,
            "candidate_offsets": candidate_offsets,
            "go_nogo_3_pass": bool(diff["point"] >= SELECT_GAIN_THRESH and diff_bs["point"] >= SELECT_GAIN_THRESH
                                   and seeds_same_dir >= 4)}


# --------------------------------------------------------------------------- ablation
def analysis_ablation(cap, seeds=(0, 1, 2)):
    blocks, hidden, probe = cap["blocks"], cap["hidden"], cap["probe"]
    sign = np.array([(hidden[b]["residual_bias_y"] > 0) for b in blocks], int)
    groups = np.array(blocks)
    subsets = {
        "A_probe_success_only": ["probe_task_outcome_error"],   # proxy: near-boundary outcome magnitude
        "B_final_error_only": ["probe_phase_goal_error", "probe_handle_relative_error"],
        "C_time_only": ["probe_skill_elapsed_time"],
        "D_success_plus_error": ["probe_task_outcome_error", "probe_phase_goal_error"],
        "E_full_history": list(FT.probe_history_features(probe[blocks[0]]).keys()),
    }
    rows = []
    out = {}
    for name, keys in subsets.items():
        feat = np.array([[FT.probe_history_features(probe[b]).get(k, 0.0) for k in keys] for b in blocks], float)
        acc = float(np.mean([MD.grouped_classification(feat, sign, groups, "logreg", s)["balanced_accuracy_mean"]
                             for s in seeds]))
        out[name] = acc; rows.append([name, ";".join(keys), round(acc, 3)])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(out.keys()), list(out.values()), color="#937860")
    ax.axhline(0.5, ls=":", c="k"); ax.set_ylabel("residual-sign balanced accuracy"); ax.set_ylim(0, 1.05)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=7)
    ax.set_title("Q-ablation: which probe feedback identifies hidden state")
    _save_fig(fig, "fig_pilot_history_ablation.png")
    _write_csv("pilot_ablation_metrics.csv", rows, ["subset", "features", "sign_balanced_accuracy"])
    return out


# --------------------------------------------------------------------------- driver
def main():
    path = DL.resolve_data_path()
    episodes = DL.load_episodes(path)
    cap = DL.build_capability_map(episodes)
    inv = build_inventory(cap, episodes, path)
    q1 = analysis_oracle_headroom(cap)
    q2 = analysis_identifiability(cap)
    q3 = analysis_selector(cap)
    abl = analysis_ablation(cap)

    # verdict logic
    if not inv["grid_complete_18x17"]:
        status = "SCIENTIFIC_PILOT_MINIMAL_DATA_COLLECTION_REQUIRED"
    elif q1["go_nogo_1_pass"] and q3["go_nogo_3_pass"]:
        status = "SCIENTIFIC_PILOT_SUPPORTS_CONFIRMATORY_RUN"
    else:
        status = "SCIENTIFIC_PILOT_REQUIRES_HYPOTHESIS_OR_TASK_REVISION"

    summary = {"label": LABEL, "status": status, "data_sha256": inv["data_sha256"],
               "n_blocks": inv["n_blocks"], "n_offsets": inv["n_offsets"],
               "go_nogo_1_oracle_headroom": {"value": q1["oracle_headroom"], "pass": q1["go_nogo_1_pass"],
                                             "threshold": HEADROOM_THRESH},
               "go_nogo_2_identifiability": {"k0_signacc": q2["k0_signacc_mean"], "k1_signacc": q2["k1_signacc_mean"],
                                             "k1_minus_k0": q2["k1_minus_k0_signacc"], "pass": q2["go_nogo_2_pass"]},
               "go_nogo_3_selection_gain": {"k1_minus_k0": q3["k1_minus_k0"], "k1_minus_best_single": q3["k1_minus_best_single"],
                                            "pass": q3["go_nogo_3_pass"]},
               "q1": q1, "q2": q2, "q3": q3, "ablation": abl,
               "leakage_ok": q2["leakage_check"]["ok"]}
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "scientific_pilot_summary.json").write_text(json.dumps(summary, indent=2))
    (DOCS / "scientific_pilot_data_inventory.json").write_text(json.dumps(inv, indent=2))
    (DOCS / "scientific_pilot_feature_allowlist.json").write_text(json.dumps(
        {"label": LABEL, "k0_static": list(FT.static_features(cap["probe"][cap["blocks"][0]]).keys()),
         "k1_probe_history": list(FT.probe_history_features(cap["probe"][cap["blocks"][0]]).keys()),
         "denylist": list(FT.LEAKAGE_DENYLIST)}, indent=2))
    (DOCS / "scientific_pilot_leakage_check.json").write_text(json.dumps(q2["leakage_check"], indent=2))
    print("STATUS:", status)
    print("Q1 oracle headroom:", q1["oracle_headroom"])
    print("Q2 K0/K1 sign acc:", round(q2["k0_signacc_mean"], 3), round(q2["k1_signacc_mean"], 3),
          "resid R2 K1:", round(q2["k1_residual_r2"], 3))
    print("Q3 best/K0/K1/oracle:", round(q3["best_single_success"], 3), round(q3["k0_selected_success"], 3),
          round(q3["k1_selected_success"], 3), round(q3["oracle_success"], 3), "K1-K0:", q3["k1_minus_k0"]["point"])
    print("leakage_ok:", q2["leakage_check"]["ok"])
    return summary


if __name__ == "__main__":
    main()
