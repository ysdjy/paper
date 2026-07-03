"""End-to-end offline evaluation pipeline (offline_v2).

Consumes one run's episodes.jsonl (read-only) and emits the audited artifact set:
  validation_report.json  split_manifest.json  metrics.json  metrics_ci.json
  predictions.jsonl  candidate_rankings.jsonl  oracle_analysis.json
  adaptation_curve.csv  main_results.csv  ablation_results.csv  summary.md
  utility_config.json  provenance.json

Everything is split BY SESSION first; histories are built leakage-safe within each split;
utility weights are frozen (loaded from utility_config, never tuned on test); CIs are
session-level bootstrap; matched-bank decision-value metrics are gated by the pairing validator.
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import numpy as np

from deployment_calibration.models_v2 import DEPLOYABLE, REGISTRY
from deployment_calibration.models_v2.train import ensemble_predict, save_models, train_seeds
from deployment_calibration.offline_v2 import metrics as MET
from deployment_calibration.offline_v2.bootstrap import make_session_stat, session_bootstrap
from deployment_calibration.offline_v2.d3 import d3_leave_one_session_out, d3_train_test
from deployment_calibration.offline_v2.data import load_run
from deployment_calibration.offline_v2.history import assert_history_legal, build_history
from deployment_calibration.offline_v2.oracle import (
    oracle_candidate_per_group, optimal_candidate_switch_rate,
    pairwise_rank_reversal_rate, selection_regret, value_of_state_information)
from deployment_calibration.offline_v2.pairing import build_matched_bank, validate_bank_report
from deployment_calibration.offline_v2.splits import audit_split, split_manifest, split_sessions
from deployment_calibration.offline_v2.utility import (
    UtilityConfig, predicted_utility, true_utility)


# ----------------------------------------------------------------- helpers
def build_pairs(episodes, sids, k):
    """(candidate episode, leakage-safe history at cutoff k) for candidates in sids."""
    pairs = []
    for e in episodes:
        if e.get("episode_role") != "candidate" or e["session_id"] not in sids:
            continue
        H = build_history(episodes, e, k=k)
        assert_history_legal(e, H, all_episodes=episodes, k=k)  # defensive
        pairs.append((e, H))
    return pairs


def groups_from_pairs(pairs):
    g = {}
    for e, _ in pairs:
        g.setdefault(e.get("candidate_group") or e["session_id"], []).append(e)
    return g


def _model_chosen_indices(models, test_pairs, cfg: UtilityConfig):
    """For each test candidate_group, the model's argmax-predicted-utility candidate_index."""
    by_group = {}
    preds = {}
    for e, H in test_pairs:
        gid = e.get("candidate_group") or e["session_id"]
        pr = ensemble_predict(models, e, H)
        u = predicted_utility(pr["p_success"], pr["pred_error"], pr["pred_time"], cfg)
        by_group.setdefault(gid, []).append((u, e.get("candidate_index"), e, pr))
    chosen = {}
    for gid, lst in by_group.items():
        u, idx, e, pr = max(lst, key=lambda t: t[0])
        chosen[gid] = idx
    return chosen, by_group


def _pred_arrays(models, test_pairs):
    y, ph, ee_t, ee_p, tt_t, tt_p, rows = [], [], [], [], [], [], []
    for e, H in test_pairs:
        pr = ensemble_predict(models, e, H)
        yv = 1.0 if e["y"]["success"] else 0.0
        y.append(yv); ph.append(pr["p_success"])
        ee_t.append(float(e["y"]["task_outcome_error"])); ee_p.append(pr["pred_error"])
        tt_t.append(float(e["y"]["skill_elapsed_time"])); tt_p.append(pr["pred_time"])
        rows.append({"episode_id": e["episode_id"], "session_id": e["session_id"],
                     "candidate_group": e.get("candidate_group"), "candidate_index": e.get("candidate_index"),
                     "y_success": yv, "p_success": pr["p_success"], "p_success_std": pr.get("p_success_std"),
                     "pred_error": pr["pred_error"], "true_error": float(e["y"]["task_outcome_error"]),
                     "pred_time": pr["pred_time"], "true_time": float(e["y"]["skill_elapsed_time"])})
    return dict(y=y, ph=ph, ee_t=ee_t, ee_p=ee_p, tt_t=tt_t, tt_p=tt_p, rows=rows)


# ----------------------------------------------------------------- main
def evaluate_run(run_dir, out_dir, *, utility_cfg: UtilityConfig | None = None,
                 models=None, seeds=(0, 1, 2, 3, 4), n_boot=2000, fracs=(0.34, 0.33, 0.33),
                 split_seed=0, k_max=None, adapt_ks=(0, 1, 2, 3)) -> dict:
    run = load_run(run_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    episodes = run.episodes
    cfg = utility_cfg or UtilityConfig()
    cfg.save(out / "utility_config.json")
    models = models or DEPLOYABLE
    (out / "provenance.json").write_text(json.dumps(run.provenance(), indent=2))

    # ---- split + audit ----
    split = split_sessions(episodes, fracs=fracs, seed=split_seed)
    man = split_manifest(episodes, split, split_seed, fracs)
    (out / "split_manifest.json").write_text(json.dumps(man, indent=2))
    trainval = split["train"] + split["val"]

    # ---- validation report (bank pairing + split audit + leakage guard) ----
    bank = build_matched_bank(episodes)
    validation = {
        "provenance": run.provenance(),
        "split_audit": man["audit"],
        "matched_bank": validate_bank_report(episodes),
        "n_candidates": len(run.candidates()),
        "n_probes": len(run.probes()),
        "n_sessions": len(run.sessions()),
    }
    (out / "validation_report.json").write_text(json.dumps(validation, indent=2))

    if k_max is None:
        k_max = max([e.get("history_cutoff", 3) for e in episodes] + [3])

    # ---- train each model on trainval, predict on test, at full K ----
    train_pairs = build_pairs(episodes, set(trainval), k=k_max)
    test_pairs = build_pairs(episodes, set(split["test"]), k=k_max)
    val_pairs = build_pairs(episodes, set(split["val"]), k=k_max) or None

    metrics, metrics_ci, all_pred_rows, main_rows = {}, {}, [], []
    ranking_rows = []
    fitted = {}
    for name in list(models) + ["OracleZ"]:
        fit = train_seeds(name, train_pairs, val_pairs, seeds=seeds)
        fitted[name] = fit
        save_models(fit, out / "models", name)
        arrs = _pred_arrays(fit, test_pairs)
        pm = MET.prediction_metrics(arrs["y"], arrs["ph"], arrs["ee_t"], arrs["ee_p"],
                                    arrs["tt_t"], arrs["tt_p"])
        # selection / regret
        groups = groups_from_pairs(test_pairs)
        chosen, by_group = _model_chosen_indices(fit, test_pairs, cfg)
        reg = selection_regret(groups, chosen, cfg)
        pm.update({"mean_regret": reg["mean_regret"], "median_regret": reg["median_regret"],
                   "top1_acc": reg["top1_acc"], "selected_success_rate": reg["selected_success_rate"],
                   "selected_task_error": reg["selected_task_error"], "selected_time": reg["selected_time"],
                   "n_groups": len(groups)})
        metrics[name] = pm
        # session-bootstrap CIs for the headline metrics
        metrics_ci[name] = _bootstrap_metrics(arrs["rows"], reg["rows"], n_boot=n_boot, seed=split_seed)
        # collect prediction + ranking rows
        for r in arrs["rows"]:
            all_pred_rows.append({"model": name, **r})
        for gid, lst in by_group.items():
            ranked = sorted(lst, key=lambda t: -t[0])
            for rank, (u, idx, e, pr) in enumerate(ranked):
                ranking_rows.append({"model": name, "candidate_group": gid, "rank": rank,
                                     "candidate_index": idx, "pred_utility": u,
                                     "true_utility": true_utility(e, cfg),
                                     "chosen": 1 if idx == chosen[gid] else 0,
                                     "true_success": bool(e["y"]["success"])})
        main_rows.append({"model": name, **{k: pm.get(k) for k in
                          ("auroc", "auprc", "brier", "calibration_error", "failure_macro_f1",
                           "task_error_mae", "exec_time_mae", "mean_regret", "median_regret",
                           "top1_acc", "selected_success_rate", "n", "n_groups")}})

    # ---- oracle decision-value analysis ----
    oc = oracle_candidate_per_group(groups_from_pairs(test_pairs), cfg)
    oracle_analysis = {
        "oracle_candidate_test_groups": oc,
        "oracle_candidate_mean_true_utility": float(np.mean([v["best_true_utility"] for v in oc.values()])) if oc else float("nan"),
        "matched_bank_available": bank["matched"],
        "matched_bank_reason": bank["reason"],
        "value_of_state_information": value_of_state_information(bank, cfg),
        "optimal_candidate_switch_rate": optimal_candidate_switch_rate(bank, cfg),
        "pairwise_rank_reversal_rate": pairwise_rank_reversal_rate(bank, cfg),
    }
    (out / "oracle_analysis.json").write_text(json.dumps(oracle_analysis, indent=2, default=str))

    # ---- adaptation curve (K sweep) for history models ----
    adapt = _adaptation_curve(episodes, trainval, split["test"], val_sids=split["val"],
                              cfg=cfg, seeds=seeds, ks=adapt_ks)
    _write_adaptation_csv(out / "adaptation_curve.csv", adapt)

    # ---- D3 held-out ----
    d3 = {"leave_one_session_out": d3_leave_one_session_out(episodes, n_boot=n_boot, seed=split_seed),
          "train_test": d3_train_test(episodes, trainval, split["test"])}

    # ---- ablation: utility sensitivity (recompute selection under variants) ----
    ablation_rows = _utility_ablation(fitted, test_pairs, cfg)

    # ---- write remaining artifacts ----
    (out / "metrics.json").write_text(json.dumps({"metrics": metrics, "d3": d3}, indent=2, default=str))
    (out / "metrics_ci.json").write_text(json.dumps(metrics_ci, indent=2, default=str))
    with open(out / "predictions.jsonl", "w") as f:
        for r in all_pred_rows:
            f.write(json.dumps(r) + "\n")
    with open(out / "candidate_rankings.jsonl", "w") as f:
        for r in ranking_rows:
            f.write(json.dumps(r, default=str) + "\n")
    _write_csv(out / "main_results.csv", main_rows)
    _write_csv(out / "ablation_results.csv", ablation_rows)
    summary = _summary_md(run, man, metrics, metrics_ci, oracle_analysis, d3, adapt, validation)
    (out / "summary.md").write_text(summary)

    return {"out_dir": str(out), "metrics": metrics, "metrics_ci": metrics_ci,
            "oracle": oracle_analysis, "d3": d3, "validation": validation, "summary_md": summary}


# ----------------------------------------------------------------- sub-steps
def _bootstrap_metrics(pred_rows, reg_rows, n_boot, seed):
    """Session-bootstrap CIs for AUROC (recomputed per resample) and mean regret."""
    out = {}
    # regret: nested mean over sessions
    stat_reg, sids = make_session_stat(reg_rows, "regret", session_key="session_id")
    if sids:
        out["mean_regret"] = session_bootstrap(sids, stat_reg, n_boot=n_boot, seed=seed)
    # AUROC: resample sessions, recompute AUROC over their pooled candidates
    by_sess = {}
    for r in pred_rows:
        by_sess.setdefault(r["session_id"], []).append((r["y_success"], r["p_success"]))
    psids = list(by_sess)

    def auroc_stat(resampled):
        ys, ps = [], []
        for sid in resampled:
            for yv, pv in by_sess.get(sid, []):
                ys.append(yv); ps.append(pv)
        if len(set(ys)) < 2:
            return float("nan")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return MET.auroc(ys, ps)

    if psids:
        out["auroc"] = session_bootstrap(psids, auroc_stat, n_boot=n_boot, seed=seed)
    # selected success rate
    stat_ss, _ = make_session_stat(reg_rows, "selected_success", session_key="session_id")
    if sids:
        out["selected_success_rate"] = session_bootstrap(sids, stat_ss, n_boot=n_boot, seed=seed)
    return out


def _adaptation_curve(episodes, trainval, test_sids, val_sids, cfg, seeds, ks):
    """For each K, retrain history models with histories truncated to K; report test metrics."""
    rows = {}
    for name in ("B1_static", "B2_mean", "B2_deepsets", "B2_gru"):
        rows[name] = {}
        for k in ks:
            tr = build_pairs(episodes, set(trainval), k=k)
            te = build_pairs(episodes, set(test_sids), k=k)
            va = build_pairs(episodes, set(val_sids), k=k) or None
            fit = train_seeds(name, tr, va, seeds=seeds)
            arrs = _pred_arrays(fit, te)
            groups = groups_from_pairs(te)
            chosen, _ = _model_chosen_indices(fit, te, cfg)
            reg = selection_regret(groups, chosen, cfg)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                au = MET.auroc(arrs["y"], arrs["ph"])
            rows[name][k] = {"auroc": au, "brier": MET.brier(arrs["y"], arrs["ph"]),
                             "mean_regret": reg["mean_regret"], "top1_acc": reg["top1_acc"],
                             "selected_success_rate": reg["selected_success_rate"]}
    return rows


def _utility_ablation(fitted, test_pairs, base_cfg):
    """Recompute selection metrics under success_only / success_error / full utility variants."""
    from deployment_calibration.offline_v2.utility import utility_variants
    rows = []
    groups = groups_from_pairs(test_pairs)
    for variant, vcfg in utility_variants().items():
        for name, fit in fitted.items():
            chosen, _ = _model_chosen_indices(fit, test_pairs, vcfg)
            reg = selection_regret(groups, chosen, vcfg)
            rows.append({"utility_variant": variant, "model": name,
                         "mean_regret": reg["mean_regret"], "top1_acc": reg["top1_acc"],
                         "selected_success_rate": reg["selected_success_rate"]})
    return rows


# ----------------------------------------------------------------- io
def _write_csv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_adaptation_csv(path, adapt):
    rows = []
    for name, byk in adapt.items():
        for k, m in byk.items():
            rows.append({"model": name, "K": k, **m})
    _write_csv(path, rows)


def _fmt_ci(ci):
    if not ci or "ci_low" not in ci:
        return "n/a"
    return f"{ci.get('point', float('nan')):.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"


def _summary_md(run, man, metrics, metrics_ci, oracle, d3, adapt, validation):
    L = []
    L.append(f"# Offline evaluation summary — {run.run_dir.name}\n")
    p = run.provenance()
    L.append(f"- source: `{p['source_run_path']}`")
    L.append(f"- source git commit: `{p['source_git_commit']}`  sha256(episodes): `{p['episodes_sha256'][:16]}…`")
    L.append(f"- episodes: {p['n_episodes']}  sessions: {validation['n_sessions']}  "
             f"probes: {validation['n_probes']}  candidates: {validation['n_candidates']}")
    a = man["audit"]
    L.append(f"- split (by session): {man['counts']}  audit ok: **{a['ok']}**  "
             f"pairwise-disjoint: {a['pairwise_disjoint']['all_empty']}  "
             f"groups-within-split: {a['candidate_groups']['all_within_one_split']}\n")
    mb = validation["matched_bank"]
    L.append(f"- matched candidate bank: **{mb['is_matched_bank']}** "
             f"(matched groups: {mb['n_matched_groups']})")
    if not mb["is_matched_bank"]:
        L.append(f"  - reason: {mb['reason']}")
    L.append("\n## Prediction metrics (test)\n")
    L.append("| model | AUROC (95% CI) | Brier | failMacroF1 | errMAE | timeMAE | meanRegret (CI) | top1 | selSucc |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for name, m in metrics.items():
        ci = metrics_ci.get(name, {})
        L.append(f"| {name} | {_fmt_ci(ci.get('auroc'))} | {m['brier']:.3f} | "
                 f"{m['failure_macro_f1']:.3f} | {m['task_error_mae']:.3f} | {m['exec_time_mae']:.2f} | "
                 f"{_fmt_ci(ci.get('mean_regret'))} | {m['top1_acc']:.2f} | {m['selected_success_rate']:.2f} |")
    L.append("\n## Decision-value / oracle analysis\n")
    vsi = oracle["value_of_state_information"]
    if vsi.get("available"):
        L.append(f"- VSI = {vsi['vsi']:.4f}  (state-aware {vsi['state_aware_mean_true_utility']:.4f} "
                 f"− state-agnostic {vsi['state_agnostic_mean_true_utility']:.4f})")
        L.append(f"- optimal-candidate switch rate: {oracle['optimal_candidate_switch_rate'].get('switch_rate')}")
        L.append(f"- pairwise rank-reversal rate: {oracle['pairwise_rank_reversal_rate'].get('reversal_rate')}")
    else:
        L.append(f"- VSI / switch / reversal: **not computable** — {oracle['matched_bank_reason']}")
    L.append("\n## D3 hidden-state identification (held out)\n")
    d = d3["leave_one_session_out"]
    if d.get("available"):
        L.append(f"- LOSO accuracy: {d['accuracy']:.3f} "
                 f"[{d['accuracy_ci']['ci_low']:.3f}, {d['accuracy_ci']['ci_high']:.3f}]  "
                 f"chance {d['chance']:.3f}  balAcc {d['balanced_accuracy']:.3f}  macroF1 {d['macro_f1']:.3f}")
        L.append(f"- confusion labels {d['confusion_labels']}: {d['confusion_matrix']}")
    L.append("\n_Small-N pilot: treat all numbers as EXPLORATORY; CIs are wide by construction._\n")
    return "\n".join(L)
