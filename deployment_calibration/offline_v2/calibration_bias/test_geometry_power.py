"""Test-geometry learned-selector power (offline, synthetic; NO Isaac, no confirmatory data).

Phase goal: fix the *test geometry*, not the model or the 0.15 threshold. The prior phase
(learned_selector_power) found B2/K1 genuinely learns to use the probe (K1~0.98 vs K0~0.82,
history increment ~0.17) but the pre-registered power event

    CI_lower( B2_K1_success - train/val-only best_single_success ) >= 0.15

could not be met at any block count -- NOT because the model can't learn, but because the old test
nominal {-0.03,+0.03} put best-single(offset 0) still ~0.82 success, so the structural gain was only
~0.18, too close to the 0.15 bar and too high-variance per block (even the ORACLE failed the event).

This phase freezes the PRIMARY test nominal at {-0.035, +0.035}:
  * best-single offset 0 now sits at |eff| = |0.035 + residual| in [0.025, 0.045], straddling the empirical
    hard edge |eff|=0.03425 -> best-single test success ~0.44 (was ~0.82);
  * the state-aware offset +-0.04 gives |eff| = |-/+0.005 + residual| <= 0.015 -> ~1.0 success;
  * so the structural gain jumps to ~0.55 (was ~0.18), well clear of 0.15;
  * the fixed probe -0.04 remains a deterministic 1-bit discriminator (success at +0.035, fail at -0.035).

Everything else is FROZEN from the prior phase (bank, train/val nominals, residual, probe offset, real
models_v2.DeepSets and its hyperparameters, model seeds, taus). We only move the test nominal outward, a
choice made BEFORE any confirmatory data exists (the actual hidden state stays inside the train support
[-0.05,+0.05]; -0.035 -> [-0.045,-0.025], +0.035 -> [+0.025,+0.045]).

This module reuses the FROZEN atomic pieces of learned_selector_power (outcome model, episode/probe
builders, real-DeepSets train/select, best-single rule, block bootstrap, power event) and only
parametrizes the test nominal set. No model hyperparameter is touched.
"""

from __future__ import annotations

import numpy as np

from deployment_calibration.offline_v2.calibration_bias import learned_selector_power as L

# ---------------- frozen (inherited unchanged from the prior phase) ----------------
BANK = L.BANK                       # (-0.04, 0.0, 0.04)
TRAIN_NOMINALS = L.TRAIN_NOMINALS   # (-0.04,-0.02,0,0.02,0.04)
VAL_NOMINALS = L.VAL_NOMINALS       # (-0.01, 0.01)
PROBE_OFFSET = L.PROBE_OFFSET       # -0.04
TAU_POINT, TAU_LOWER, TAU_UPPER, TAU_ISO = L.TAU_POINT, L.TAU_LOWER, L.TAU_UPPER, L.TAU_ISO
DEEPSETS_HP = L.DEEPSETS_HP
MODEL_SEEDS = L.MODEL_SEEDS

# ---------------- NEW frozen primary test geometry (this phase) ----------------
TEST_NOMINALS_PRIMARY = (-0.035, 0.035)
# Pre-registered sensitivity anchors only -- NOT allowed to replace the primary (protocol section 10).
TEST_NOMINALS_SENS_TIGHT = (-0.034, 0.034)
TEST_NOMINALS_SENS_WIDE = (-0.036, 0.036)


def make_dataset(train_blocks, val_blocks, test_blocks, tau, rng, test_nominals):
    """Parametrized clone of learned_selector_power.make_dataset with configurable test nominals.

    Independent residual per block per split; block ids disjoint across splits within one replicate.
    Reuses the frozen episode/probe builders so labels are identical to the prior phase modulo geometry.
    """
    res = L.draw_residuals(train_blocks + val_blocks + test_blocks, rng)
    tr_r = res[:train_blocks]
    va_r = res[train_blocks:train_blocks + val_blocks]
    te_r = res[train_blocks + val_blocks:]

    def sessions(residuals, nominals, bid0):
        out = []
        for bi, r in enumerate(residuals):
            for nb in nominals:
                cands = []
                for o in BANK:
                    e = L._episode(nb, r, o, tau)
                    H = [L._probe_entry(nb, r, tau)]     # K=1 history (fixed probe -0.04)
                    cands.append((e, H))
                out.append({"block_id": bid0 + bi, "nominal": nb, "residual": r, "actual": nb + r,
                            "candidates": cands})
        return out

    tr = sessions(tr_r, TRAIN_NOMINALS, 0)
    va = sessions(va_r, VAL_NOMINALS, 1000)
    te = sessions(te_r, test_nominals, 2000)
    return tr, va, te


def run_replicate(train_blocks, val_blocks, test_blocks, tau, rep_seed, test_nominals,
                  model_seeds=MODEL_SEEDS):
    """One Monte-Carlo replicate. Mirrors learned_selector_power.run_replicate but with the new test
    geometry. Best-single uses train+val ONLY; test never participates in selection."""
    L._CUR_TAU[0] = tau
    rng = np.random.default_rng(rep_seed)
    tr, va, te = make_dataset(train_blocks, val_blocks, test_blocks, tau, rng, test_nominals)
    bs_off = L.best_single(tr, va, tau)
    bs_succ_sess = [L.succ(s["nominal"], s["residual"], bs_off, tau) for s in te]
    oracle_succ_sess = [L.succ(s["nominal"], s["residual"], L.oracle_select(s), tau) for s in te]

    k1_by_seed, k0_by_seed = [], []
    conv_ok = True
    for ms in model_seeds:
        try:
            s1, _ = L.train_select(tr, va, te, 1, ms)
            s0, _ = L.train_select(tr, va, te, 0, ms)
        except Exception:
            conv_ok = False
            s1 = bs_succ_sess[:]; s0 = bs_succ_sess[:]
        k1_by_seed.append(s1); k0_by_seed.append(s0)
    k1 = np.array(k1_by_seed, float)
    k0 = np.array(k0_by_seed, float)
    k1_sess = k1.mean(0); k0_sess = k0.mean(0)
    bs = np.array(bs_succ_sess, float); orc = np.array(oracle_succ_sess, float)

    blocks = sorted(set(s["block_id"] for s in te))
    def by_block(vals):
        return np.array([np.mean([vals[i] for i, s in enumerate(te) if s["block_id"] == b]) for b in blocks])

    gain_block = by_block(k1_sess) - by_block(bs)
    hist_block = by_block(k1_sess) - by_block(k0_sess)
    struct_gain_block = by_block(orc) - by_block(bs)
    seed_gains = [float(np.mean(by_block(k1[si]) - by_block(bs))) for si in range(len(model_seeds))]

    # selected-offset distribution over the seed-averaged argmax (diagnostic; from seed 0's chosen offsets)
    return {"best_single_offset": bs_off, "gain_block": gain_block.tolist(),
            "struct_gain_block": struct_gain_block.tolist(),
            "mean_gain": float(np.mean(gain_block)),
            "mean_struct_gain": float(np.mean(struct_gain_block)),
            "k1_test_success": float(np.mean(k1_sess)), "k0_test_success": float(np.mean(k0_sess)),
            "best_single_test_success": float(np.mean(bs)), "oracle_test_success": float(np.mean(orc)),
            "history_increment": float(np.mean(hist_block)),
            "seed_gains": seed_gains, "converged": conv_ok, "n_test_blocks": len(blocks)}


def struct_power_event(rep, ci_low, gain_min=0.15, oracle_min=0.85):
    """Structural/oracle power event: same CI bar but on the ORACLE gain (best possible state knowledge).
    Reported SEPARATELY from learned power; never substituted for it."""
    nondeg = np.var(rep["struct_gain_block"]) > 1e-12 or (0 < rep["best_single_test_success"] < 1)
    return bool(ci_low >= gain_min and rep["mean_struct_gain"] > 0
                and rep["oracle_test_success"] >= oracle_min and nondeg)


def run_power(train_blocks, val_blocks, test_blocks, tau, n_reps, test_nominals,
              rep_seed0=0, n_boot=2000):
    """Estimate learned AND structural power for one design/tau/geometry over n_reps replicates."""
    learned_events, struct_events = [], []
    gains, sgains, k1s, k0s, bss, orcs, incrs, convs = [], [], [], [], [], [], [], []
    bs_offsets = []
    seed_ok_count = 0
    boot_rng = np.random.default_rng(9999)
    for i in range(n_reps):
        rep = run_replicate(train_blocks, val_blocks, test_blocks, tau, rep_seed0 + i, test_nominals)
        lo, _ = L.block_bootstrap_ci(rep["gain_block"], boot_rng, n_boot=n_boot)
        slo, _ = L.block_bootstrap_ci(rep["struct_gain_block"], boot_rng, n_boot=n_boot)
        ev = L.power_event(rep, lo)
        learned_events.append(bool(ev["event"] and rep["converged"]))
        struct_events.append(struct_power_event(rep, slo))
        gains.append(rep["mean_gain"]); sgains.append(rep["mean_struct_gain"])
        k1s.append(rep["k1_test_success"]); k0s.append(rep["k0_test_success"])
        bss.append(rep["best_single_test_success"]); orcs.append(rep["oracle_test_success"])
        incrs.append(rep["history_increment"]); convs.append(rep["converged"])
        bs_offsets.append(rep["best_single_offset"])
        seed_ok_count += int(ev["seed_stability_ok"])
    p = float(np.mean(learned_events)); n = len(learned_events)
    sp = float(np.mean(struct_events))
    z = 1.96
    half = z * np.sqrt(p * (1 - p) / n) if n else 0.0
    shalf = z * np.sqrt(sp * (1 - sp) / n) if n else 0.0
    return {"train_blocks": train_blocks, "val_blocks": val_blocks, "test_blocks": test_blocks,
            "tau": tau, "test_nominals": list(test_nominals), "n_reps": n,
            "learned_power": p, "learned_power_ci95": [max(0.0, p - half), min(1.0, p + half)],
            "struct_power": sp, "struct_power_ci95": [max(0.0, sp - shalf), min(1.0, sp + shalf)],
            "mean_learned_gain": float(np.mean(gains)), "mean_struct_gain": float(np.mean(sgains)),
            "mean_k1_success": float(np.mean(k1s)), "mean_k0_success": float(np.mean(k0s)),
            "mean_best_single_success": float(np.mean(bss)), "mean_oracle_success": float(np.mean(orcs)),
            "mean_history_increment": float(np.mean(incrs)), "frac_converged": float(np.mean(convs)),
            "frac_seed_stable": seed_ok_count / n if n else 0.0,
            "best_single_offset_counts": {str(o): int(sum(1 for x in bs_offsets if abs(x - o) < 1e-9))
                                          for o in BANK}}


# =====================================================================================
# Geometry audit (section 4) -- pure analytic/Monte-Carlo checks, no model training.
# =====================================================================================
def geometry_audit(test_nominals=TEST_NOMINALS_PRIMARY,
                   taus=(TAU_LOWER, TAU_POINT, TAU_UPPER, TAU_ISO), n_res=20000, seed=0):
    rng = np.random.default_rng(seed)
    res = np.array(L.draw_residuals(n_res, rng))
    checks = {}
    all_ok = True

    # --- state-aware: oracle offset succeeds across the WHOLE residual support at every tau ---
    sa = {}
    sa_ok = True
    for tn in test_nominals:
        off = min(BANK, key=lambda o: abs(tn + o))     # deepest compensator (extremes -> +-0.04)
        per_tau = {}
        for tau in taus:
            # worst-case over support endpoints AND sampled residuals
            frac = float(np.mean([L.succ(tn, r, off, tau) for r in res]))
            ends = [L.succ(tn, L.RESIDUAL_LO, off, tau), L.succ(tn, L.RESIDUAL_HI, off, tau)]
            per_tau[f"{tau:.5f}"] = {"offset": off, "succ_frac": frac,
                                     "worst_endpoint_ok": bool(min(ends) == 1), "abs_eff_max": float(
                                         max(abs(tn + L.RESIDUAL_LO + off), abs(tn + L.RESIDUAL_HI + off)))}
            if frac < 1.0 or min(ends) == 0:
                sa_ok = False
        sa[str(tn)] = per_tau
    checks["state_aware"] = {"per_nominal": sa, "all_support_success": sa_ok,
                             "requirement": "abs_eff<=0.015 -> success at ALL tau/all residuals"}
    all_ok &= sa_ok

    # --- best-single: train/val-only selection is offset 0, and it straddles the edge at test ---
    bs_ok = True
    bs = {}
    for tau in taus:
        # build a fresh, large train+val to make the selection statistically exact
        rng2 = np.random.default_rng(1000 + int(tau * 1e6))
        big_r = L.draw_residuals(4000, rng2)
        tr = [{"nominal": nb, "residual": r} for r in big_r[:3000] for nb in TRAIN_NOMINALS]
        va = [{"nominal": nb, "residual": r} for r in big_r[3000:] for nb in VAL_NOMINALS]
        off = L.best_single(tr, va, tau)
        # non-degenerate crossing at test: per nominal, success frac strictly inside (0,1)
        cross = {}
        nondeg = True
        for tn in test_nominals:
            frac = float(np.mean([L.succ(tn, r, off, tau) for r in res]))
            cross[str(tn)] = frac
            if not (0.02 < frac < 0.98):
                nondeg = False
        bs[f"{tau:.5f}"] = {"best_single_offset": off, "test_success_frac": cross, "non_degenerate": nondeg}
        if abs(off) > 1e-9 or not nondeg:
            bs_ok = False
    checks["best_single"] = {"per_tau": bs, "offset_is_zero_and_crosses_edge": bs_ok}
    all_ok &= bs_ok

    # --- no robust common action: no single offset succeeds on BOTH test nominals in >=90% of blocks ---
    nca_ok = True
    nca = {}
    for tau in taus:
        worst_common = 0.0
        detail = {}
        for o in BANK:
            # fraction of residual blocks where THIS offset succeeds on both test nominals simultaneously
            both = np.mean([min(L.succ(test_nominals[0], r, o, tau),
                                L.succ(test_nominals[1], r, o, tau)) for r in res])
            detail[str(o)] = float(both)
            worst_common = max(worst_common, float(both))
        nca[f"{tau:.5f}"] = {"per_offset_both_succeed_frac": detail, "max_common": worst_common}
        if worst_common >= 0.90:
            nca_ok = False
    checks["no_robust_common_action"] = {"per_tau": nca, "no_common_action": nca_ok,
                                         "requirement": "no fixed offset succeeds on both nominals in >=90% blocks"}
    all_ok &= nca_ok

    # --- probe determinism: fixed probe -0.04 is a deterministic 1-bit discriminator over full support ---
    probe_ok = True
    pr = {}
    for tau in taus:
        per = {}
        for tn in test_nominals:
            frac = float(np.mean([L.succ(tn, r, PROBE_OFFSET, tau) for r in res]))
            ends = [L.succ(tn, L.RESIDUAL_LO, PROBE_OFFSET, tau), L.succ(tn, L.RESIDUAL_HI, PROBE_OFFSET, tau)]
            per[str(tn)] = {"succ_frac": frac, "endpoints": ends}
        # deterministic AND separating: one nominal all-success, the other all-fail, at every residual
        vals = list(per.values())
        deterministic = all(v["succ_frac"] in (0.0, 1.0) and v["endpoints"][0] == v["endpoints"][1]
                            for v in vals)
        separating = len({v["succ_frac"] for v in vals}) == 2
        pr[f"{tau:.5f}"] = {"per_nominal": per, "deterministic": deterministic, "separating": separating}
        if not (deterministic and separating):
            probe_ok = False
    checks["probe"] = {"per_tau": pr, "deterministic_1bit": probe_ok,
                       "requirement": "probe -0.04 deterministically succeeds on one nominal, fails on other"}
    all_ok &= probe_ok

    checks["test_nominals"] = list(test_nominals)
    checks["train_actual_support"] = [min(TRAIN_NOMINALS) + L.RESIDUAL_LO, max(TRAIN_NOMINALS) + L.RESIDUAL_HI]
    checks["test_actual_support"] = {str(tn): [tn + L.RESIDUAL_LO, tn + L.RESIDUAL_HI] for tn in test_nominals}
    checks["test_support_within_train"] = bool(
        all(tn + L.RESIDUAL_LO >= min(TRAIN_NOMINALS) + L.RESIDUAL_LO - 1e-12
            and tn + L.RESIDUAL_HI <= max(TRAIN_NOMINALS) + L.RESIDUAL_HI + 1e-12 for tn in test_nominals))
    all_ok &= checks["test_support_within_train"]
    checks["GEOMETRY_AUDIT_PASSED"] = bool(all_ok)
    return checks
