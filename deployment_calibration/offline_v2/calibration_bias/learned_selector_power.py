"""Learned-selector power simulation for Design A (offline, synthetic; NO Isaac, no confirmatory data).

Answers: under finite train/val blocks and finite model seeds, can the REAL capacity-matched DeepSets
K=1 (one fixed probe at offset -0.04) learn to beat the train/val-only best-single on the unseen test
nominal biases, with enough statistical power? Uses the empirical HARD edge as the label generator and the
frozen v3 residual. Only the fixed probe history distinguishes sessions (x/g are identical across
sessions), so K=0 structurally reduces to a fixed offset -> best-single.

The B2/B1 models are the repository's real `models_v2.DeepSets` (K=1 vs K=0), NOT a hand classifier.
"""

from __future__ import annotations

import numpy as np

from deployment_calibration.models_v2 import DeepSets

# ---------------- frozen design (Design A) ----------------
BANK = (-0.04, 0.0, 0.04)
TRAIN_NOMINALS = (-0.04, -0.02, 0.0, 0.02, 0.04)
VAL_NOMINALS = (-0.01, 0.01)
TEST_NOMINALS = (-0.03, 0.03)
PROBE_OFFSET = -0.04
RESIDUAL_SIGMA, RESIDUAL_LO, RESIDUAL_HI = 0.005, -0.01, 0.01
TAU_POINT, TAU_LOWER, TAU_UPPER, TAU_ISO = 0.03425, 0.0342, 0.0343, 0.0325

# ---------------- FROZEN training budget (fixed before the Monte Carlo; not tuned on test) ----------------
DEEPSETS_HP = dict(d_hid=32, d_emb=16, lr=1e-2, max_epochs=200, patience=25, l2=1e-4)
MODEL_SEEDS = (1103, 2207, 3301, 4409, 5519)

# ---------------- outcome model (empirical hard edge) ----------------
def succ(nominal, residual, offset, tau):
    return 1 if abs(nominal + residual + offset) <= tau + 1e-12 else 0


def draw_residuals(n, rng):
    out = []
    while len(out) < n:
        x = rng.normal(0.0, RESIDUAL_SIGMA)
        if RESIDUAL_LO <= x <= RESIDUAL_HI:
            out.append(float(x))
    return out


def _episode(nominal, residual, offset, tau):
    s = succ(nominal, residual, offset, tau)
    return {"x": {"initial_mechanism_joint_pos": 0.0, "gripper_width": 0.08, "member": "cabinet"},
            "g": {"target_open_position": 0.2, "target_tolerance": 0.02},
            "theta": {"grasp_offset_local_y": float(offset), "max_pos_step": 0.02, "pull_lead": 0.08},
            "y": {"success": bool(s), "task_outcome_error": 0.01 if s else 0.17,
                  "skill_elapsed_time": 9.4 if s else 23.8, "final_joint_position": 0.2 if s else 0.0,
                  "phase_durations": {"PULL": 0.6 if s else 0.0}, "handle_relative_error": 0.007 if s else 0.02,
                  "failure_reason": "NONE" if s else "HANDLE_DETACHED"},
            "_offset": float(offset)}


def _probe_entry(nominal, residual, tau):
    s = succ(nominal, residual, PROBE_OFFSET, tau)
    return {"theta": {"grasp_offset_local_y": PROBE_OFFSET, "max_pos_step": 0.02, "pull_lead": 0.08},
            "success": bool(s), "task_outcome_error": 0.01 if s else 0.17,
            "skill_elapsed_time": 9.4 if s else 23.8, "pull_phase_duration": 0.6 if s else 0.0,
            "final_joint_position": 0.2 if s else 0.0}


def make_dataset(train_blocks, val_blocks, test_blocks, tau, rng):
    """Independent residual per block per split; blocks disjoint across splits within one replicate."""
    res = draw_residuals(train_blocks + val_blocks + test_blocks, rng)
    tr_r = res[:train_blocks]; va_r = res[train_blocks:train_blocks + val_blocks]
    te_r = res[train_blocks + val_blocks:]

    def sessions(residuals, nominals, bid0):
        out = []
        for bi, r in enumerate(residuals):
            for nb in nominals:
                cands = []
                for o in BANK:
                    e = _episode(nb, r, o, tau)
                    H = [_probe_entry(nb, r, tau)]              # K=1 history (the fixed probe)
                    cands.append((e, H))
                out.append({"block_id": bid0 + bi, "nominal": nb, "residual": r, "actual": nb + r,
                            "candidates": cands})
        return out

    tr = sessions(tr_r, TRAIN_NOMINALS, 0)
    va = sessions(va_r, VAL_NOMINALS, 1000)
    te = sessions(te_r, TEST_NOMINALS, 2000)
    return tr, va, te


def _pairs(sessions, k):
    """Flatten sessions to (episode, history) pairs; k=0 -> empty history, k=1 -> the probe."""
    out = []
    for s in sessions:
        for e, H in s["candidates"]:
            out.append((e, H if k == 1 else []))
    return out


def best_single(train_sessions, val_sessions, tau):
    """LEGACY rule (kept ONLY as the fix1 invariance baseline; NOT the preregistered rule).

    Uses a tau-|eff| continuous margin as the step-2 tie-break, where eff = actual_bias + offset reads the
    SECRET hidden state -- illegal for a state-agnostic best-single (BLOCKER_SECRET_TIEBREAK, Claude C). The
    tie-break is DEAD CODE on every frozen config (step-1 is always unique -> offset 0), which is exactly why
    fix1 needs no power recertification. Retained so best_single_tiebreak_invariance can compare old vs new.
    """
    sess = list(train_sessions) + list(val_sessions)
    rows = []
    for o in BANK:
        succs = [succ(s["nominal"], s["residual"], o, tau) for s in sess]
        margins = [tau - abs(s["nominal"] + s["residual"] + o) for s in sess]
        rows.append((o, float(np.mean(succs)), float(np.mean(margins))))
    max_s = max(r[1] for r in rows)
    tied = [r for r in rows if abs(r[1] - max_s) < 1e-9]
    max_m = max(r[2] for r in tied)
    tied2 = [r for r in tied if abs(r[2] - max_m) < 1e-9]
    return min(tied2, key=lambda r: abs(r[0]))[0]


def best_single_legal(train_sessions, val_sessions, tau):
    """PREREGISTERED (fix1) state-agnostic best-single. Train+val ONLY, no secret.

    Rule: (1) max observed mean binary success; (2) tie -> min |offset|; (3) tie -> fixed candidate-bank
    numeric order {-0.04, 0.00, +0.04} (earliest). No nominal/residual/actual bias, no eff_signed/abs_eff,
    no tau-|eff| margin, no oracle/secret, no test outcomes. `succ(...)` here is the OBSERVED binary
    outcome of an executed train/val candidate trial (a legal observable), not the hidden state.
    """
    sess = list(train_sessions) + list(val_sessions)
    rows = [(o, float(np.mean([succ(s["nominal"], s["residual"], o, tau) for s in sess]))) for o in BANK]
    max_s = max(r[1] for r in rows)
    tied = [r for r in rows if abs(r[1] - max_s) < 1e-9]
    min_abs = min(abs(r[0]) for r in tied)
    tied2 = [r for r in tied if abs(abs(r[0]) - min_abs) < 1e-9]
    return min(tied2, key=lambda r: BANK.index(r[0]))[0]


def oracle_select(session):
    return min(BANK, key=lambda o: abs(session["actual"] + o))


def train_select(train_sessions, val_sessions, test_sessions, k, seed):
    """Train the real DeepSets (k=0 or k=1) and select argmax p_success per test session.
    Returns (per_test_session_selected_success, diag)."""
    m = DeepSets(**DEEPSETS_HP)
    m.fit(_pairs(train_sessions, k), val_pairs=_pairs(val_sessions, k) or None, seed=seed)
    sel = []
    chosen_offsets = []
    for s in test_sessions:
        best_o, best_p = None, -1.0
        for e, H in s["candidates"]:
            p = m.predict(e, H if k == 1 else [])["p_success"]
            if p > best_p:
                best_p, best_o = p, e["_offset"]
        sel.append(succ(s["nominal"], s["residual"], best_o, tau_of(s)))
        chosen_offsets.append(best_o)
    return sel, {"chosen_offsets": chosen_offsets}


def tau_of(session):     # tau is fixed per run; helper reads it from a module-level set by run_replicate
    return _CUR_TAU[0]


_CUR_TAU = [TAU_POINT]


def run_replicate(train_blocks, val_blocks, test_blocks, tau, rep_seed, model_seeds=MODEL_SEEDS):
    _CUR_TAU[0] = tau
    rng = np.random.default_rng(rep_seed)
    tr, va, te = make_dataset(train_blocks, val_blocks, test_blocks, tau, rng)
    bs_off = best_single(tr, va, tau)
    # per test session true outcomes
    bs_succ_sess = [succ(s["nominal"], s["residual"], bs_off, tau) for s in te]
    oracle_succ_sess = [succ(s["nominal"], s["residual"], oracle_select(s), tau) for s in te]
    # learned: 5 seeds each for K=1 and K=0
    k1_by_seed, k0_by_seed = [], []
    conv_ok = True
    for ms in model_seeds:
        try:
            s1, _ = train_select(tr, va, te, 1, ms)
            s0, _ = train_select(tr, va, te, 0, ms)
        except Exception:
            conv_ok = False
            s1 = bs_succ_sess[:]; s0 = bs_succ_sess[:]
        k1_by_seed.append(s1); k0_by_seed.append(s0)
    k1 = np.array(k1_by_seed, float)      # [seed, test_session]
    k0 = np.array(k0_by_seed, float)
    # seed-averaged per test session, then group by test BLOCK (avg over the block's test nominals)
    k1_sess = k1.mean(0); k0_sess = k0.mean(0)
    bs = np.array(bs_succ_sess, float); orc = np.array(oracle_succ_sess, float)
    blocks = sorted(set(s["block_id"] for s in te))
    def by_block(vals):
        return np.array([np.mean([vals[i] for i, s in enumerate(te) if s["block_id"] == b]) for b in blocks])
    gain_block = by_block(k1_sess) - by_block(bs)     # primary per-block gain (seed-avg B2 - best-single)
    hist_block = by_block(k1_sess) - by_block(k0_sess)
    # per-seed point gain (for seed-stability gate)
    seed_gains = [float(np.mean(by_block(k1[si]) - by_block(bs))) for si in range(len(model_seeds))]
    return {"best_single_offset": bs_off, "gain_block": gain_block.tolist(),
            "mean_gain": float(np.mean(gain_block)),
            "k1_test_success": float(np.mean(k1_sess)), "k0_test_success": float(np.mean(k0_sess)),
            "best_single_test_success": float(np.mean(bs)), "oracle_test_success": float(np.mean(orc)),
            "history_increment": float(np.mean(hist_block)),
            "seed_gains": seed_gains, "converged": conv_ok, "n_test_blocks": len(blocks)}


def block_bootstrap_ci(gain_block, rng, n_boot=2000, ci=0.95):
    g = np.asarray(gain_block, float); n = len(g)
    reps = np.array([g[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    return float(np.quantile(reps, lo)), float(np.quantile(reps, hi))


def power_event(rep, ci_low, gain_min=0.15, oracle_min=0.85, seed_stab_min=4):
    """Primary power event: CI_lower(Delta) >= gain_min AND mean gain > 0 AND oracle >= 0.85 AND
    best-single non-degenerate (labels vary) AND seed stability (>=4/5 seed point gains >= gain_min and
    none < 0)."""
    seeds_ok = sum(1 for g in rep["seed_gains"] if g >= gain_min) >= seed_stab_min \
        and all(g >= 0 for g in rep["seed_gains"])
    nondeg = np.var(rep["gain_block"]) > 1e-12 or (0 < rep["best_single_test_success"] < 1)
    return {"ci_low": ci_low, "event": bool(ci_low >= gain_min and rep["mean_gain"] > 0
                                            and rep["oracle_test_success"] >= oracle_min
                                            and nondeg and seeds_ok),
            "seed_stability_ok": bool(seeds_ok), "nondegenerate": bool(nondeg)}


def run_power(train_blocks, val_blocks, test_blocks, tau, n_reps, rep_seed0=0, n_boot=2000):
    events, gains, k1s, k0s, bss, orcs, incrs, convs = [], [], [], [], [], [], [], []
    seed_ok_count = 0
    boot_rng = np.random.default_rng(9999)
    for i in range(n_reps):
        rep = run_replicate(train_blocks, val_blocks, test_blocks, tau, rep_seed0 + i)
        lo, _ = block_bootstrap_ci(rep["gain_block"], boot_rng, n_boot=n_boot)
        ev = power_event(rep, lo)
        # conservative: a non-converged replicate counts as a power failure
        events.append(bool(ev["event"] and rep["converged"]))
        gains.append(rep["mean_gain"]); k1s.append(rep["k1_test_success"]); k0s.append(rep["k0_test_success"])
        bss.append(rep["best_single_test_success"]); orcs.append(rep["oracle_test_success"])
        incrs.append(rep["history_increment"]); convs.append(rep["converged"])
        seed_ok_count += int(ev["seed_stability_ok"])
    p = float(np.mean(events)); n = len(events)
    z = 1.96
    half = z * np.sqrt(p * (1 - p) / n) if n else 0.0
    return {"train_blocks": train_blocks, "val_blocks": val_blocks, "test_blocks": test_blocks, "tau": tau,
            "n_reps": n, "power": p, "power_ci95": [max(0.0, p - half), min(1.0, p + half)],
            "mean_learned_gain": float(np.mean(gains)), "mean_k1_success": float(np.mean(k1s)),
            "mean_k0_success": float(np.mean(k0s)), "mean_best_single_success": float(np.mean(bss)),
            "mean_oracle_success": float(np.mean(orcs)), "mean_history_increment": float(np.mean(incrs)),
            "frac_converged": float(np.mean(convs)), "frac_seed_stable": seed_ok_count / n if n else 0.0}
