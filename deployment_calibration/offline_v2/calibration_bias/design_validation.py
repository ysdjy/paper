"""Pre-data confirmatory-design validation for the calibration-bias stage (offline).

Uses ONLY the exploration-derived success geometry (success iff |bias + offset| <= band, band = 0.02
m from the capability map) to check a proposed split BEFORE any confirmatory data exists. No
confirmatory outcomes are read.

The decisive check (the v1 flaw): if a single fixed offset satisfies the success band for EVERY
held-out test bias, then a state-agnostic best-single offset attains full success on test and the H3
success-only gain is 0 by construction. The split is only valid if the per-test-bias success-offset
sets have an EMPTY common intersection (no offset covers all test biases), and every test bias is
bracketed by train levels (interpolation).
"""

from __future__ import annotations

COMPENSATION_BAND = 0.02   # half-width, from the capability map (success iff |bias+offset| <= 0.02)


def predicted_success_offsets(bias: float, offsets, band: float = COMPENSATION_BAND) -> set:
    """Offsets predicted to succeed at `bias` under the compensation band (design geometry)."""
    return {round(float(o), 6) for o in offsets if abs(bias + o) <= band + 1e-9}


def _bracketed(bias: float, train_levels) -> bool:
    return any(t < bias - 1e-12 for t in train_levels) and any(t > bias + 1e-12 for t in train_levels)


def validate_confirmatory_split(train, val, test, offsets, band: float = COMPENSATION_BAND) -> dict:
    """Return a report; ok=False if the split cannot support a success-only H3 gain.

    Checks:
      * every test bias is bracketed by train levels on both sides (interpolation);
      * the per-test-bias success-offset sets have an EMPTY intersection (no single offset covers all
        test biases) -> a fixed best-single offset cannot attain full test success;
      * train/val/test bias levels are pairwise disjoint.
    """
    train = [round(float(b), 6) for b in train]
    val = [round(float(b), 6) for b in val]
    test = [round(float(b), 6) for b in test]

    tr, va, te = set(train), set(val), set(test)
    disjoint = not (tr & va or tr & te or va & te)

    bracket = {str(b): _bracketed(b, train) for b in test}
    all_bracketed = all(bracket.values()) and len(test) > 0

    succ_sets = {str(b): sorted(predicted_success_offsets(b, offsets, band)) for b in test}
    inter = set.intersection(*[predicted_success_offsets(b, offsets, band) for b in test]) if test else set()
    empty_intersection = len(inter) == 0

    # best a single fixed offset can do on test = max over offsets of (#test biases it satisfies)/(#test)
    best_single_cover = 0.0
    for o in offsets:
        cover = sum(1 for b in test if abs(b + o) <= band + 1e-9) / max(1, len(test))
        best_single_cover = max(best_single_cover, cover)
    # a state-aware selector can compensate each test bias iff each has a non-empty success set
    state_aware_cover = 1.0 if all(len(predicted_success_offsets(b, offsets, band)) > 0 for b in test) else 0.0
    max_success_only_gain = state_aware_cover - best_single_cover

    ok = disjoint and all_bracketed and empty_intersection
    return {
        "ok": bool(ok),
        "band": band,
        "splits": {"train": train, "val": val, "test": test},
        "bias_levels_disjoint": disjoint,
        "test_bracketed_by_train": bracket,
        "all_test_bracketed": all_bracketed,
        "test_success_offset_sets": succ_sets,
        "common_success_offset_across_test": sorted(inter),
        "empty_intersection": empty_intersection,
        "best_single_max_test_success": best_single_cover,
        "state_aware_max_test_success": state_aware_cover,
        "max_achievable_success_only_gain": max_success_only_gain,
        "reason": None if ok else _reason(disjoint, all_bracketed, empty_intersection, sorted(inter)),
    }


def _reason(disjoint, bracketed, empty_inter, inter):
    if not disjoint:
        return "train/val/test bias levels overlap"
    if not bracketed:
        return "a test bias is not bracketed by train levels (would be extrapolation, not interpolation)"
    if not empty_inter:
        return (f"offset(s) {inter} satisfy the success band for EVERY test bias -> a fixed best-single "
                f"offset attains full test success -> success-only H3 gain is 0 by construction")
    return "unknown"
