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


class BlockStateCompatibilityError(RuntimeError):
    """Raised when the frozen known-answer gate fails (RNG / Generator.normal / float drift)."""
    verdict = "EXPERIMENT_INVALID_MANIFEST_INTEGRITY"


def validate_subseed(subseed) -> int:
    """A subseed must be a non-bool int in [0, 2**63-1) (the range of PRE.subseed)."""
    if isinstance(subseed, bool) or not isinstance(subseed, int):
        raise BlockStateSamplerError(f"subseed must be a non-bool int, got {subseed!r}")
    if subseed < 0 or subseed >= _SUBSEED_MOD:
        raise BlockStateSamplerError(f"subseed {subseed} out of range [0, {_SUBSEED_MOD - 1}]")
    return subseed


# ============================ FROZEN KNOWN-ANSWER VECTORS (source-of-truth literals) ============================
KNOWN_ANSWER_VERSION = "pcg64_normal_floathex_kat_v1"
# (split, block_index, residual_subseed, residual_float_hex, nuisance_subseed) -- FROZEN literals; expected
# values are NOT regenerated from the current sampler. nuisance_values is exactly {} for all (none_v1).
KNOWN_ANSWER_VECTORS = (
    ("train", 0, 3900147458896997762, "-0x1.7881c155e5258p-13", 6562159524509754206),
    ("train", 8, 7943510405471906039, "-0x1.0ee242e52d4dfp-11", 1000704721152147675),
    ("validation", 0, 4622689590762937920, "-0x1.a18fd3343f967p-8", 4916055577791758597),
    ("validation", 5, 1974286722429745049, "-0x1.666e311f5b1ecp-10", 263647283272395115),
    ("test", 0, 3357889320340643563, "-0x1.111697894f781p-8", 7725492040517457760),
    ("test", 8, 1871260554784712498, "0x1.ca81c6291fbbap-9", 6818819167133771356),
)


def _residual_value_from_subseed_unchecked(subseed: int) -> float:
    """PRIVATE unchecked core: a single residual draw from a frozen PCG64(subseed) via rejection sampling.

    A fresh Generator(PCG64(subseed)) per block -> NO global RNG, NO stream splitting, NO default_rng
    future-default, NO Python hash(); iteration-order independent. Rejection (no clip / no fallback-to-mean);
    on 1000 rejections raises BlockStateSamplerExhausted. Called ONLY by validate_known_answer_vectors and
    the gated public residual API -- a generator MUST NOT call this directly (it bypasses the KAT gate)."""
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


def validate_known_answer_vectors() -> dict:
    """Bit-exact compatibility gate for PCG64 + Generator.normal + Python float. Recomputes each FROZEN
    vector from the unchecked core and compares float.hex(); any mismatch -> BlockStateCompatibilityError.
    Recording numpy version alone is NOT the gate; this KAT is. Returns a small read-only report."""
    import numpy as np
    for split, bi, rs_lit, hex_lit, ns_lit in KNOWN_ANSWER_VECTORS:
        rs = ID.block_residual_subseed(split, bi)
        ns = ID.block_nuisance_subseed(split, bi)
        if rs != rs_lit or ns != ns_lit:
            raise BlockStateCompatibilityError(
                f"KAT subseed mismatch {split} {bi}: residual {rs} vs {rs_lit}, nuisance {ns} vs {ns_lit} "
                f"| numpy={np.__version__} sampler={RESIDUAL_SAMPLER_VERSION} kat={KNOWN_ANSWER_VERSION}")
        actual = _residual_value_from_subseed_unchecked(rs).hex()
        if actual != hex_lit:
            raise BlockStateCompatibilityError(
                f"KAT hex mismatch for vector ({split},{bi}): expected {hex_lit} actual {actual} "
                f"| numpy={np.__version__} sampler={RESIDUAL_SAMPLER_VERSION} kat={KNOWN_ANSWER_VERSION}")
        if nuisance_values_from_subseed(ns) != {}:
            raise BlockStateCompatibilityError(
                f"KAT nuisance policy drift for ({split},{bi}): expected {{}} "
                f"| kat={KNOWN_ANSWER_VERSION}")
    if RESIDUAL_SAMPLER_VERSION != "pcg64_rejection_v1" or KNOWN_ANSWER_VERSION != "pcg64_normal_floathex_kat_v1":
        raise BlockStateCompatibilityError("KAT version constants drifted")
    return {"passed": True, "known_answer_version": KNOWN_ANSWER_VERSION,
            "sampler_version": RESIDUAL_SAMPLER_VERSION, "numpy_version": np.__version__,
            "vectors_checked": len(KNOWN_ANSWER_VECTORS)}


def require_known_answer_compatibility() -> dict:
    """MANDATORY preflight gate. Runs the KAT every call (no cache -> a failure can never be masked by a
    stale success, and a monkeypatched version/constant is always re-checked). Cheap (6 short PCG64 draws)."""
    return validate_known_answer_vectors()


def residual_value_from_subseed(subseed: int) -> float:
    """GATED public residual draw: validate subseed -> mandatory KAT gate -> unchecked core."""
    validate_subseed(subseed)
    require_known_answer_compatibility()
    return _residual_value_from_subseed_unchecked(subseed)


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
    """The unique resolved block state (fresh dict each call). Uses the GATED public residual API (the
    mandatory KAT runs via residual_value_from_subseed); never calls the unchecked core directly."""
    rs = ID.block_residual_subseed(split, block_index)
    ns = ID.block_nuisance_subseed(split, block_index)
    return {
        "residual_subseed": rs,
        "nuisance_subseed": ns,
        "residual_value": residual_value_from_subseed(rs),      # gated
        "nuisance_values": nuisance_values_from_subseed(ns),
    }


# public API only; the unchecked core (_residual_value_from_subseed_unchecked) is NOT exported and a
# generator must never call it (it bypasses the mandatory KAT gate).
__all__ = ["RESIDUAL_SAMPLER_VERSION", "NUISANCE_POLICY_VERSION", "KNOWN_ANSWER_VERSION", "MAX_ATTEMPTS",
           "KNOWN_ANSWER_VECTORS", "BlockStateSamplerError", "BlockStateSamplerExhausted",
           "BlockStateCompatibilityError", "validate_subseed", "validate_known_answer_vectors",
           "require_known_answer_compatibility", "residual_value_from_subseed", "residual_value",
           "nuisance_values_from_subseed", "nuisance_values", "resolved_block_state", "SPEC"]

SPEC = {
    "module": "deployment_calibration.offline_v2.calibration_bias.confirmatory_v4_block_state",
    "residual": {"version": RESIDUAL_SAMPLER_VERSION, "bit_generator": "numpy.random.PCG64",
                 "seed_input": "block_residual_subseed", "mean": RESIDUAL_MEAN, "sigma": RESIDUAL_SIGMA,
                 "support": [RESIDUAL_LO, RESIDUAL_HI], "support_inclusive": True,
                 "max_attempts": MAX_ATTEMPTS, "fallback": None,
                 "on_exhaustion": "EXPERIMENT_INVALID_MANIFEST_INTEGRITY"},
    "nuisance": {"version": NUISANCE_POLICY_VERSION, "seed_input": "block_nuisance_subseed",
                 "exact_values": {}, "additional_block_nuisance_enabled": False},
    "compatibility_gate": {"version": KNOWN_ANSWER_VERSION,
                           "function": "confirmatory_v4_block_state.require_known_answer_compatibility",
                           "vectors": len(KNOWN_ANSWER_VECTORS), "comparison": "Python float.hex() exact",
                           "numpy_version_recorded": True, "numpy_version_alone_is_not_the_gate": True,
                           "on_mismatch": "EXPERIMENT_INVALID_MANIFEST_INTEGRITY"},
    "unchecked_core_private": "_residual_value_from_subseed_unchecked (not exported; generators must not call it)",
    "validator_recomputes_exact_values": True,
}
