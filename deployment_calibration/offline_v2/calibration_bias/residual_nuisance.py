"""Block-level residual calibration nuisance (offline schema, v3).

The v2 power model assumed a per-block grasp perturbation that moved the effective offset. The ACTUAL
runtime injects a calibration bias and adds only initial-joint / target jitter, which the capability map
showed did NOT flip success labels. v3 replaces that assumption with a nuisance the runtime CAN
implement and that genuinely moves the success outcome: a per-block residual on the calibration bias.

    actual_bias_y = nominal_bias_y + residual_bias_y

Frozen distribution (physical rationale below):
  residual_bias_y ~ TruncatedNormal(mean=0, sigma=0.005 m, range [-0.01, +0.01] m)
  * one residual per nuisance BLOCK, reused across all bias levels within that block's split;
  * blocks independent; train/val/test residual draws + seeds fully isolated;
  * nominal_bias_y and actual_bias_y live ONLY in secret/audit fields;
  * a deployable model may read NEITHER residual_bias_y NOR actual_bias_y;
  * real handle geometry is unchanged; NO artificial noise is added to the candidate action.

Physical rationale: a handle-pose calibration is never perfect from session to session. A zero-mean
residual of a few millimetres (sigma 5 mm, capped at 10 mm) is a realistic recalibration error on top of
the commanded (nominal) bias. Because it lives on the SAME axis as the hidden state, it directly perturbs
|actual_bias + offset| — the quantity the success band is defined on — so it produces genuine
success-label variance near the band edge, unlike joint/target jitter. It is capped at 10 mm so that even
the extreme nominal biases (+/-0.04) stay inside the 7-offset compensable range (actual in [-0.05,+0.05]).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ResidualNuisanceConfig:
    sigma: float = 0.005          # m
    lo: float = -0.01             # m (truncation)
    hi: float = 0.01              # m
    mean: float = 0.0
    axis: str = "handle_local_y"
    scheme: str = "one zero-mean residual per nuisance block, reused across the split's bias levels; " \
                  "blocks independent; residual/actual_bias are secret/audit-only"

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_RESIDUAL = ResidualNuisanceConfig()

# fields that carry the residual / actual bias — audit only, NEVER model-legal
RESIDUAL_SECRET_KEYS = ("residual_bias_y", "actual_bias_y", "nominal_bias_y")


def _truncnorm_draw(rng, mean, sigma, lo, hi):
    """Rejection-sample a truncated normal (support tiny -> rejection is fine and deterministic)."""
    for _ in range(1000):
        x = rng.normal(mean, sigma)
        if lo <= x <= hi:
            return float(x)
    return float(min(max(mean, lo), hi))


def draw_block_residuals(n_blocks: int, seed: int, cfg: ResidualNuisanceConfig = DEFAULT_RESIDUAL):
    """Deterministic per-block residuals (numpy Generator seeded)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    return [_truncnorm_draw(rng, cfg.mean, cfg.sigma, cfg.lo, cfg.hi) for _ in range(n_blocks)]


def actual_bias(nominal_bias: float, residual: float) -> float:
    return float(nominal_bias) + float(residual)


def residual_std_effective(cfg: ResidualNuisanceConfig = DEFAULT_RESIDUAL) -> float:
    """SD of the truncated normal (for reporting / power calibration)."""
    a, b = (cfg.lo - cfg.mean) / cfg.sigma, (cfg.hi - cfg.mean) / cfg.sigma
    phi = lambda z: math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    Phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
    Z = Phi(b) - Phi(a)
    var = cfg.sigma ** 2 * (1 + (a * phi(a) - b * phi(b)) / Z - ((phi(a) - phi(b)) / Z) ** 2)
    return float(math.sqrt(max(var, 0.0)))


def assert_residual_not_in_x(episode: dict, x_key: str = "x") -> None:
    """Guard: residual / actual / nominal bias must never appear inside x (model input)."""
    for k in episode.get(x_key, {}):
        low = str(k).lower()
        if any(s in low for s in ("residual", "actual_bias", "nominal_bias")):
            raise ValueError(f"episode {episode.get('episode_id')} x carries residual/actual key {k!r}")


def actual_bias_in_compensable_range(nominal_biases, offsets, cfg: ResidualNuisanceConfig = DEFAULT_RESIDUAL,
                                     band: float = 0.02) -> dict:
    """For every nominal bias and every residual in support, is some offset within the success band?"""
    worst = []
    for b in nominal_biases:
        for r in (cfg.lo, 0.0, cfg.hi):
            a = actual_bias(b, r)
            best = min(abs(a + o) for o in offsets)
            worst.append({"nominal": b, "residual": r, "actual": round(a, 4),
                          "min_abs_eff": round(best, 4), "compensable": best <= band + 1e-9})
    ok = all(w["compensable"] for w in worst)
    return {"ok": ok, "band": band, "cases": worst,
            "max_actual_bias": max(abs(actual_bias(b, cfg.hi)) for b in nominal_biases) if nominal_biases else 0.0}
