"""Driver: learned-selector power Monte Carlo for Design A -> all artifacts. Offline; no Isaac.

Runs the primary 9/6/9 at the 3 primary taus, a frozen block-count escalation grid, structural (oracle)
power alongside learned power, threshold sensitivity, seed stability, best-single stability, and the
one-shot + amortized Net-VOI. Writes results incrementally so partial runs survive.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np

from . import learned_selector_power as L

# frozen config set (block-count escalation grid). Selection rule (frozen): smallest total-block design
# meeting power at ALL 3 primary taus; ties -> larger test, then larger train.
PRIMARY = (9, 6, 9)
GRID = [(9, 6, 9), (12, 9, 12), (18, 9, 18)]
PRIMARY_TAUS = (L.TAU_LOWER, L.TAU_POINT, L.TAU_UPPER)
GAIN_MIN = 0.15
POWER_MIN, POWER_CI_LOWER_MIN = 0.85, 0.80
# amortization horizons (T must come from deployment semantics, NOT chosen from results)
HORIZONS = (1, 2, 5, 10)
PROBE_TIME_S = 16.58        # one full-task diagnostic-trial (from redesign VOI)
LAMBDA_TIME = 0.02


def structural_power(tb, vb, te, tau, n_reps, boot=2000):
    """Oracle vs best-single (no training): expected gain + block-bootstrap power event."""
    boot_rng = np.random.default_rng(7)
    events, gains = [], []
    for i in range(n_reps):
        rng = np.random.default_rng(10_000 + i)
        tr, va, tes = L.make_dataset(tb, vb, te, tau, rng)
        bs = L.best_single(tr, va, tau)
        blocks = sorted(set(s["block_id"] for s in tes))
        def by_block(fn):
            return np.array([np.mean([fn(s) for s in tes if s["block_id"] == b]) for b in blocks])
        g = by_block(lambda s: L.succ(s["nominal"], s["residual"], L.oracle_select(s), tau)) \
            - by_block(lambda s: L.succ(s["nominal"], s["residual"], bs, tau))
        lo, _ = L.block_bootstrap_ci(g, boot_rng, n_boot=boot)
        events.append(lo >= GAIN_MIN and np.mean(g) > 0)
        gains.append(float(np.mean(g)))
    return {"power": float(np.mean(events)), "mean_gain": float(np.mean(gains))}


def amortized_net_voi(success_gain):
    per_task = success_gain                          # per-task success-utility gain
    cost = LAMBDA_TIME * PROBE_TIME_S                # one-time full-task probe cost
    return {str(T): round(T * per_task - cost, 4) for T in HORIZONS}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--grid-reps", type=int, default=200, dest="grid_reps")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    prov = {"python": platform.python_version(), "numpy": np.__version__, "torch": _torch_ver(),
            "reps_primary": a.reps, "reps_grid": a.grid_reps, "n_boot": a.boot,
            "deepsets_hp": L.DEEPSETS_HP, "model_seeds": list(L.MODEL_SEEDS),
            "gain_min": GAIN_MIN, "power_min": POWER_MIN, "power_ci_lower_min": POWER_CI_LOWER_MIN}

    results = []       # learned power per (design, tau)
    struct = []        # structural power per (design, tau)
    t0 = time.time()

    def record():
        (out / "learned_selector_power_results_v1.json").write_text(json.dumps(
            {"provenance": prov, "learned": results, "structural": struct,
             "elapsed_min": round((time.time() - t0) / 60, 1)}, indent=2))

    # primary design at all 3 taus + grid at point & upper (worst) taus
    todo = [(PRIMARY, tau, a.reps) for tau in PRIMARY_TAUS]
    for d in GRID:
        if d == PRIMARY:
            continue
        for tau in (L.TAU_POINT, L.TAU_UPPER):
            todo.append((d, tau, a.grid_reps))

    for (tb, vb, te), tau, reps in todo:
        r = L.run_power(tb, vb, te, tau, n_reps=reps, rep_seed0=0, n_boot=a.boot)
        r["passes_power"] = bool(r["power"] >= POWER_MIN and r["power_ci95"][0] >= POWER_CI_LOWER_MIN
                                 and r["mean_learned_gain"] >= GAIN_MIN)
        results.append(r)
        s = structural_power(tb, vb, te, tau, min(reps, 400), boot=a.boot)
        struct.append({"design": [tb, vb, te], "tau": tau, **s})
        print(f"[power] {tb}/{vb}/{te} tau={tau}: learned_power={r['power']:.3f} "
              f"gain={r['mean_learned_gain']:.3f} k1={r['mean_k1_success']:.3f} k0={r['mean_k0_success']:.3f} "
              f"struct_power={s['power']:.3f} struct_gain={s['mean_gain']:.3f} ({(time.time()-t0)/60:.1f}min)",
              flush=True)
        record()

    # pick the minimum design meeting power at ALL 3 primary taus (learned)
    by_design = {}
    for r in results:
        by_design.setdefault(tuple(r[k] for k in ("train_blocks", "val_blocks", "test_blocks")), {})[r["tau"]] = r
    selected = None
    for d in GRID:
        taus = by_design.get(d, {})
        if all(t in taus and taus[t]["passes_power"] for t in PRIMARY_TAUS):
            selected = list(d); break
    # net VOI from the selected (or primary) design's point-tau learned gain
    ref = by_design.get(tuple(selected) if selected else PRIMARY, {}).get(L.TAU_POINT)
    learned_gain = ref["mean_learned_gain"] if ref else 0.0
    net_voi = {"one_shot_net_voi": round(learned_gain - LAMBDA_TIME * PROBE_TIME_S, 4),
               "gross_voi_success": round(learned_gain, 4), "probe_time_cost": round(LAMBDA_TIME * PROBE_TIME_S, 4),
               "amortized_net_voi_by_horizon": amortized_net_voi(learned_gain)}

    # verdict (honest): escalation only counts if a larger design MEETS the bar AND the trend supports it.
    k1_beats_k0 = all(r["mean_history_increment"] > 0.02 for r in results)
    max_learned_power = max((r["power"] for r in results), default=0.0)
    max_struct_power = max((s["power"] for s in struct), default=0.0)
    if selected is not None:
        verdict = "POWER_SUFFICIENT_FOR_PREREG_V4" if tuple(selected) == PRIMARY else "INCREASE_BLOCK_COUNTS"
    elif max_learned_power >= 0.5 or max_struct_power >= 0.85:
        # escalation is plausibly viable (some design is approaching the bar) -> needs bigger blocks
        verdict = "INCREASE_BLOCK_COUNTS"
    elif k1_beats_k0:
        # the mechanism is learnable (K1 >> K0) but the pre-registered CI-lower>=0.15 event is
        # unreachable at any reasonable block count -- even the ORACLE fails (struct power low).
        # The threshold/claim must change, not the block count.
        verdict = "REDESIGN_MODEL_OR_CLAIM"
    else:
        verdict = "STOP_CURRENT_CONFIRMATORY_CONCEPT"
    vobj = {"verdict": verdict, "selected_design": selected,
            "primary_9_6_9_passes": bool(by_design.get(PRIMARY, {}).get(L.TAU_POINT, {}).get("passes_power", False)),
            "net_voi": net_voi, "preregistration_v4_generated": False,
            "runtime_or_confirmatory_data_generated": False,
            "note": "learned-selector power (not oracle); one-shot Net VOI honestly reported; "
                    "if verdict INCREASE_BLOCK_COUNTS the selected_design is the minimum meeting power at all 3 taus."}
    (out / "learned_selector_power_verdict_v1.json").write_text(json.dumps(vobj, indent=2))
    record()
    print(json.dumps({"verdict": verdict, "selected": selected, "one_shot_net_voi": net_voi["one_shot_net_voi"],
                      "amortized": net_voi["amortized_net_voi_by_horizon"]}, indent=2))

    # results CSV
    with open(out / "learned_selector_power_results_v1.csv", "w", newline="") as f:
        keys = ["train_blocks", "val_blocks", "test_blocks", "tau", "n_reps", "power", "power_ci95",
                "mean_learned_gain", "mean_k1_success", "mean_k0_success", "mean_best_single_success",
                "mean_oracle_success", "mean_history_increment", "frac_seed_stable", "frac_converged",
                "passes_power"]
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in results:
            w.writerow({k: (json.dumps(r[k]) if isinstance(r[k], list) else r[k]) for k in keys})
    return 0


def _torch_ver():
    try:
        import torch
        return torch.__version__
    except Exception:
        return "n/a"


if __name__ == "__main__":
    import sys
    sys.exit(main())
