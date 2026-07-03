"""Band-edge characterization experiment — frozen config, geometry, and analysis plan (offline).

This is an EXPLORATORY mechanism-characterization experiment (NOT confirmatory data). It resolves the
MUST-MODIFY items from Claude C's v3 audit: measure the true success band at the previously-unmeasured
odd |eff| points (esp. 0.03), measure edge softness, and verify the v3 block residual produces genuine
success-LABEL variation near the edge. It uses ONLY nominal_bias_y = 0.00 so no future confirmatory
test nominal condition (±0.03) is observed here.

Everything in this module is frozen BEFORE A returns data: the design, the analysis method, and the exit
criteria. The analysis functions below are the exact, pre-registered estimators B will run read-only on
A's data. `logistic center / edge scale / bin width / collision-threshold method` are fixed here and must
NOT be chosen after seeing data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .residual_nuisance import DEFAULT_RESIDUAL, ResidualNuisanceConfig

# ---- frozen design constants ----
NOMINAL_BIAS_Y = 0.00
N_BLOCKS = 18
OFFSET_GRID = (-0.050, -0.040, -0.035, -0.030, -0.025, -0.020, -0.015, -0.010, 0.000,
               0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.050)
N_OFFSETS = len(OFFSET_GRID)                     # 17
N_EPISODES = N_BLOCKS * N_OFFSETS                # 306
RUN_PROBES = False

# |eff| points the grid must cover at nominal 0 (residual 0): success / edge / fail, +/- symmetric
REQUIRED_ABS_EFF_POINTS = (0.0, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.050)

# ---- frozen analysis-plan constants (chosen pre-data) ----
BIN_WIDTH = 0.005                # m, |eff| bin for block-level success rate
LOGISTIC_INIT_CENTER = 0.03      # init only; the fit estimates the true center
LOGISTIC_INIT_SCALE = 0.005      # init only
N_BOOT = 2000
CI = 0.95
EFF03 = 0.03                     # the decisive previously-unmeasured point
# operating |eff| region whose residual-driven label variance we must confirm
OPERATING_ABS_EFF = (0.025, 0.030, 0.035)
# exit-criteria thresholds (frozen)
ASYMMETRY_MAX_CENTER_DIFF = 0.005    # m; |center_pos - center_neg| above this = severe asymmetry
EDGE_SCALE_MIN_SOFT = 1e-4           # scale below this => effectively a hard threshold
COLLISION_RESULT_SHIFT_MAX = 0.005   # m; edge-center shift when excluding collision eps above this = collision-dominated


@dataclass(frozen=True)
class BandEdgeConfig:
    nominal_bias_y: float = NOMINAL_BIAS_Y
    n_blocks: int = N_BLOCKS
    offsets: tuple = OFFSET_GRID
    run_probes: bool = RUN_PROBES
    residual: ResidualNuisanceConfig = field(default_factory=lambda: DEFAULT_RESIDUAL)
    bin_width: float = BIN_WIDTH
    n_boot: int = N_BOOT
    ci: float = CI

    @property
    def n_episodes(self) -> int:
        return self.n_blocks * len(self.offsets)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["offsets"] = list(self.offsets)
        d["n_offsets"] = len(self.offsets)
        d["n_episodes"] = self.n_episodes
        return d


DEFAULT_BAND_EDGE = BandEdgeConfig()


# ---------------------------------------------------------------- geometry
def effective_error_signed(actual_bias: float, offset: float) -> float:
    return float(actual_bias + offset)


def effective_error(actual_bias: float, offset: float) -> float:
    return abs(effective_error_signed(actual_bias, offset))


def abs_eff_coverage_nominal(cfg: BandEdgeConfig = DEFAULT_BAND_EDGE) -> set:
    """|eff| points measured at residual = 0 (nominal 0): just |offset|."""
    return {round(abs(o), 4) for o in cfg.offsets}


def abs_eff_coverage_with_residual(cfg: BandEdgeConfig = DEFAULT_BAND_EDGE) -> tuple:
    """Min/max |eff| reachable across the residual support at nominal 0 (continuous edge coverage)."""
    lo, hi = cfg.residual.lo, cfg.residual.hi
    reach = []
    for o in cfg.offsets:
        vals = [abs(o + lo), abs(o + hi), abs(o)]
        reach.append((round(min(vals), 4), round(max(vals), 4)))
    covered = set()
    for mn, mx in reach:
        covered.add((mn, mx))
    return sorted(covered)


def validate_config(cfg: BandEdgeConfig = DEFAULT_BAND_EDGE) -> dict:
    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    add("nominal_bias_is_zero", cfg.nominal_bias_y == 0.0, f"nominal={cfg.nominal_bias_y}")
    add("episode_count_306", cfg.n_episodes == 306, f"{cfg.n_blocks}x{len(cfg.offsets)}={cfg.n_episodes}")
    add("n_blocks_18", cfg.n_blocks == 18)
    add("n_offsets_17", len(cfg.offsets) == 17)
    add("no_probes", cfg.run_probes is False)
    cov = abs_eff_coverage_nominal(cfg)
    missing = [p for p in REQUIRED_ABS_EFF_POINTS if round(p, 4) not in cov]
    add("covers_required_abs_eff_points", not missing, f"missing={missing}")
    add("covers_success_region", 0.0 in cov and 0.010 in cov)
    add("covers_known_edge_0.02", 0.020 in cov)
    add("covers_unmeasured_0.025_0.03_0.035", all(p in cov for p in (0.025, 0.030, 0.035)))
    add("covers_fail_region_0.04_0.05", 0.040 in cov and 0.050 in cov)
    # +/- symmetry of the offset grid
    pos = sorted(o for o in cfg.offsets if o > 0)
    neg = sorted((-o for o in cfg.offsets if o < 0))
    add("offset_grid_symmetric", pos == neg, f"pos={pos} negabs={neg}")
    # residual matches the v3 frozen residual
    add("residual_matches_v3", cfg.residual.sigma == 0.005 and cfg.residual.lo == -0.01
        and cfg.residual.hi == 0.01)
    ok = all(c["ok"] for c in checks)
    return {"ok": ok, "checks": checks, "abs_eff_coverage_nominal": sorted(cov),
            "n_episodes": cfg.n_episodes}


# ---------------------------------------------------------------- frozen analysis estimators
def _logistic(eff, center, scale):
    return 1.0 / (1.0 + np.exp((np.asarray(eff, float) - center) / scale))


def block_success_rate_by_eff_bin(records, bin_width: float = BIN_WIDTH) -> dict:
    """Block-level success rate per |eff| bin. `records`: dicts with block_id, eff (|actual+offset|),
    success (0/1). Returns {bin_lo: {mean, n_blocks, block_rates}} using block means (nested-correct)."""
    by_bin_block = {}
    for r in records:
        b = np.floor(r["eff"] / bin_width) * bin_width
        by_bin_block.setdefault(round(float(b), 4), {}).setdefault(r["block_id"], []).append(int(r["success"]))
    out = {}
    for b, blocks in sorted(by_bin_block.items()):
        block_rates = [float(np.mean(v)) for v in blocks.values()]
        out[b] = {"mean": float(np.mean(block_rates)), "n_blocks": len(block_rates),
                  "block_rates": block_rates}
    return out


def fit_logistic_edge(records) -> dict:
    """Fit p_succ = sigmoid((center - |eff|)/scale). Returns center, scale (scale>0 = softness)."""
    from scipy.optimize import curve_fit
    eff = np.array([r["eff"] for r in records], float)
    y = np.array([int(r["success"]) for r in records], float)
    if len(set(y)) < 2:
        return {"converged": False, "reason": "single-class success", "center": None, "scale": None}
    try:
        popt, _ = curve_fit(lambda e, c, s: 1.0 / (1.0 + np.exp((e - c) / s)), eff, y,
                            p0=[LOGISTIC_INIT_CENTER, LOGISTIC_INIT_SCALE],
                            bounds=([0.0, 1e-5], [0.10, 0.05]), maxfev=20000)
        return {"converged": True, "center": float(popt[0]), "scale": float(abs(popt[1]))}
    except Exception as ex:
        return {"converged": False, "reason": str(ex), "center": None, "scale": None}


def fit_isotonic_edge(records) -> dict:
    """Monotone (decreasing in |eff|) success curve via isotonic regression."""
    from sklearn.isotonic import IsotonicRegression
    eff = np.array([r["eff"] for r in records], float)
    y = np.array([int(r["success"]) for r in records], float)
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(eff, y)
    grid = np.round(np.arange(0.0, 0.061, 0.005), 4)
    p = iso.predict(grid)
    # isotonic "edge" = |eff| where predicted success crosses 0.5
    cross = None
    for i in range(len(grid) - 1):
        if p[i] >= 0.5 >= p[i + 1]:
            cross = float(grid[i] + 0.005 * (p[i] - 0.5) / max(p[i] - p[i + 1], 1e-9))
            break
    return {"grid": grid.tolist(), "success_pred": [float(x) for x in p], "edge_center_cross0.5": cross}


def block_bootstrap_edge(records, n_boot: int = N_BOOT, ci: float = CI, seed: int = 0) -> dict:
    """Block bootstrap of the logistic (center, scale). Resamples BLOCKS (not episodes)."""
    rng = np.random.default_rng(seed)
    by_block = {}
    for r in records:
        by_block.setdefault(r["block_id"], []).append(r)
    blocks = list(by_block)
    point = fit_logistic_edge(records)
    centers, scales = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(blocks), size=len(blocks))
        sample = [rr for i in idx for rr in by_block[blocks[i]]]
        fit = fit_logistic_edge(sample)
        if fit["converged"]:
            centers.append(fit["center"]); scales.append(fit["scale"])
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    def qci(a):
        a = np.asarray(a, float)
        return [float(np.quantile(a, lo)), float(np.quantile(a, hi))] if len(a) else [float("nan")] * 2
    return {"point_center": point.get("center"), "point_scale": point.get("scale"),
            "center_ci": qci(centers), "scale_ci": qci(scales),
            "n_valid_boot": len(centers), "n_boot": n_boot,
            "n_blocks": len(blocks)}


def symmetry_test(records) -> dict:
    """Fit the edge separately on the positive and negative signed-eff sides; compare centers."""
    pos = [r for r in records if r.get("eff_signed", r["eff"]) > 0]
    neg = [r for r in records if r.get("eff_signed", r["eff"]) < 0]
    fp, fn = fit_logistic_edge(pos), fit_logistic_edge(neg)
    diff = (abs(fp["center"] - fn["center"]) if fp.get("center") is not None
            and fn.get("center") is not None else None)
    return {"pos_fit": fp, "neg_fit": fn, "center_abs_diff": diff,
            "severe_asymmetry": (diff is not None and diff > ASYMMETRY_MAX_CENTER_DIFF)}


def eff03_variation(records, eff_point: float = EFF03, tol: float = 0.003) -> dict:
    """Is there genuine success-probability variation near |eff| = 0.03 (not all one label)?"""
    near = [r for r in records if abs(r["eff"] - eff_point) <= tol]
    if not near:
        return {"available": False, "reason": "no episodes near |eff|=0.03"}
    ys = [int(r["success"]) for r in near]
    return {"available": True, "n": len(near), "success_rate": float(np.mean(ys)),
            "has_both_labels": len(set(ys)) > 1}


def residual_label_variation(records, operating_abs_eff=OPERATING_ABS_EFF, tol: float = 0.003) -> dict:
    """Does the block residual produce success-LABEL variation across blocks in the operating region?

    For each operating |eff|, check whether the per-block success rate varies (some block succeeds,
    another fails) -> genuine block-level label variation (the v3 residual doing real work).
    """
    out = {}
    any_var = False
    for e in operating_abs_eff:
        near = [r for r in records if abs(r["eff"] - e) <= tol]
        by_block = {}
        for r in near:
            by_block.setdefault(r["block_id"], []).append(int(r["success"]))
        rates = [float(np.mean(v)) for v in by_block.values()]
        varies = len(rates) > 0 and (min(rates) < 1.0 and max(rates) > 0.0)
        any_var = any_var or varies
        out[str(e)] = {"n_blocks": len(rates), "block_success_rates": rates,
                       "label_varies_across_blocks": varies}
    return {"per_operating_eff": out, "any_block_level_variation": any_var}


# ---------------------------------------------------------------- exit criteria (frozen)
def evaluate_exit_criteria(*, eff025_035_both_labels: bool, residual_block_variation: bool,
                           edge_center, edge_scale, edge_estimable: bool, severe_asymmetry: bool,
                           collision_sensor_available: bool, instrumentation_complete: bool,
                           result_collision_dominated: bool) -> dict:
    """Frozen PASS_TO_V4 / MODIFY_RESIDUAL_OR_DESIGN decision from analysis outputs."""
    pass_conds = {
        "eff_0.025_0.035_has_both_success_and_failure": eff025_035_both_labels,
        "residual_produces_block_level_label_variation": residual_block_variation,
        "edge_center_and_scale_estimable": edge_estimable and edge_center is not None
        and edge_scale is not None,
        "no_severe_unexplained_asymmetry": not severe_asymmetry,
        "collision_sensor_available": collision_sensor_available,
        "instrumentation_complete": instrumentation_complete,
        "edge_result_not_collision_dominated": not result_collision_dominated,
    }
    modify_conds = {
        "success_still_hard_threshold": (edge_scale is not None and edge_scale < EDGE_SCALE_MIN_SOFT),
        "residual_no_edge_variation": not residual_block_variation,
        "eff_0.03_all_same_label": not eff025_035_both_labels,
        "collision_dominates_edge": result_collision_dominated,
        "instrumentation_unavailable": not instrumentation_complete,
    }
    verdict = "PASS_TO_V4" if all(pass_conds.values()) else "MODIFY_RESIDUAL_OR_DESIGN"
    return {"verdict": verdict, "pass_conditions": pass_conds, "modify_triggers": modify_conds}
