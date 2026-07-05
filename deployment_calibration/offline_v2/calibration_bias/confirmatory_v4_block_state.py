"""Frozen confirmatory-v4 block-state sampler (offline; the UNIQUE production source of truth).

Closes GENERATOR_IMPLEMENTATION_BLOCKED_MISSING_FROZEN_BLOCK_STATE_SAMPLER: the manifest requires each block
to carry `residual_value` and `nuisance_values` (both covered by the full phase hash), but until now only the
subseeds were frozen -- there was no unique `subseed -> value` function, so two generators could produce
different valid manifests. This module fixes the deterministic mapping. It is the ONLY production sampler; no
second copy may exist elsewhere.

Scientific probability law is UNCHANGED from residual_nuisance.DEFAULT_RESIDUAL:
    residual_bias_y ~ TruncatedNormal(mean=0.0, sigma=0.005 m, support=[-0.01,+0.01] m)
This amendment only pins the deterministic RNG + truncation + the (already-absent) additional block nuisance
to `none_v1`. No Isaac, no generator, no manifest instance, no confirmatory data.
"""

from __future__ import annotations

from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_identity as ID
from deployment_calibration.offline_v2.calibration_bias.residual_nuisance import DEFAULT_RESIDUAL

RESIDUAL_SAMPLER_VERSION = "pcg64_rejection_v1"
NUISANCE_POLICY_VERSION = "none_v1"
MAX_ATTEMPTS = 1000

# frozen distribution parameters -- taken from DEFAULT_RESIDUAL so they cannot drift
RESIDUAL_MEAN = float(DEFAULT_RESIDUAL.mean)     # 0.0
RESIDUAL_SIGMA = float(DEFAULT_RESIDUAL.sigma)   # 0.005
RESIDUAL_LO = float(DEFAULT_RESIDUAL.lo)         # -0.01
RESIDUAL_HI = float(DEFAULT_RESIDUAL.hi)         # +0.01

_SUBSEED_MOD = 2 ** 63 - 1                        # subseeds are in [0, 2**63-2]


class BlockStateSamplerError(ValueError):
    verdict = "EXPERIMENT_INVALID_MANIFEST_INTEGRITY"


class BlockStateSamplerExhausted(RuntimeError):
    verdict = "EXPERIMENT_INVALID_MANIFEST_INTEGRITY"


def validate_subseed(subseed) -> int:
    """A subseed must be a non-bool int in [0, 2**63-1) (the range of PRE.subseed)."""
    if isinstance(subseed, bool) or not isinstance(subseed, int):
        raise BlockStateSamplerError(f"subseed must be a non-bool int, got {subseed!r}")
    if subseed < 0 or subseed >= _SUBSEED_MOD:
        raise BlockStateSamplerError(f"subseed {subseed} out of range [0, {_SUBSEED_MOD - 1}]")
    return subseed


def residual_value_from_subseed(subseed: int) -> float:
    """Deterministic single residual draw from a frozen PCG64(subseed) via rejection sampling.

    A fresh Generator(PCG64(subseed)) is created per block -> NO global RNG, NO stream splitting, NO
    default_rng future-default, NO Python hash(); iteration-order independent. Rejection (no clip / no
    fallback-to-mean); on 1000 rejections the manifest construction fails (BlockStateSamplerExhausted)."""
    import numpy as np
    validate_subseed(subseed)
    rng = np.random.Generator(np.random.PCG64(subseed))
    for _attempt in range(MAX_ATTEMPTS):
        x = float(rng.normal(loc=RESIDUAL_MEAN, scale=RESIDUAL_SIGMA))
        if RESIDUAL_LO <= x <= RESIDUAL_HI:
            return x
    raise BlockStateSamplerExhausted(
        f"residual sampler exhausted {MAX_ATTEMPTS} attempts for subseed {subseed} "
        f"(no draw in [{RESIDUAL_LO},{RESIDUAL_HI}]) -> EXPERIMENT_INVALID_MANIFEST_INTEGRITY")


def residual_value(split: str, block_index: int) -> float:
    return residual_value_from_subseed(ID.block_residual_subseed(split, block_index))


def nuisance_values_from_subseed(subseed: int) -> dict:
    """Frozen nuisance policy none_v1: v4 has NO additional block-level nuisance (v3 removed the artificial
    grasp perturbation; residual_bias_y is the only block-level hidden variation). The nuisance_subseed
    remains a domain-separated, hash-covered reserved field, but its resolved value is exactly {}."""
    validate_subseed(subseed)
    return {}


def nuisance_values(split: str, block_index: int) -> dict:
    return nuisance_values_from_subseed(ID.block_nuisance_subseed(split, block_index))


def resolved_block_state(split: str, block_index: int) -> dict:
    """The unique resolved block state (fresh dict each call)."""
    rs = ID.block_residual_subseed(split, block_index)
    ns = ID.block_nuisance_subseed(split, block_index)
    return {
        "residual_subseed": rs,
        "nuisance_subseed": ns,
        "residual_value": residual_value_from_subseed(rs),
        "nuisance_values": nuisance_values_from_subseed(ns),
    }


SPEC = {
    "module": "deployment_calibration.offline_v2.calibration_bias.confirmatory_v4_block_state",
    "residual": {"version": RESIDUAL_SAMPLER_VERSION, "bit_generator": "numpy.random.PCG64",
                 "seed_input": "block_residual_subseed", "mean": RESIDUAL_MEAN, "sigma": RESIDUAL_SIGMA,
                 "support": [RESIDUAL_LO, RESIDUAL_HI], "support_inclusive": True,
                 "max_attempts": MAX_ATTEMPTS, "fallback": None,
                 "on_exhaustion": "EXPERIMENT_INVALID_MANIFEST_INTEGRITY"},
    "nuisance": {"version": NUISANCE_POLICY_VERSION, "seed_input": "block_nuisance_subseed",
                 "exact_values": {}, "additional_block_nuisance_enabled": False},
    "validator_recomputes_exact_values": True,
}
