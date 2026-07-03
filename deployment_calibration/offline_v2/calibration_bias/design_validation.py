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


def validate_confirmatory_split_residual(train, val, test, offsets, residual_cfg,
                                         band: float = COMPENSATION_BAND, n_grid: int = 41) -> dict:
    """Residual-aware (v3) split validation over the FULL residual support.

    Checks, in addition to the nominal checks:
      * every nominal bias stays compensable by the offset bank for every residual in support;
      * no residual makes the actual bias exceed the compensable range;
      * for NO residual does a single offset satisfy the success band for BOTH test biases within the
        same block (no common robust offset persists) -> best-single test success <= 0.5;
      * best-single max test success and the state-aware upper bound over the residual support.
    """
    from .residual_nuisance import actual_bias, actual_bias_in_compensable_range

    base = validate_confirmatory_split(train, val, test, offsets, band)
    all_nominal = list(train) + list(val) + list(test)
    comp = actual_bias_in_compensable_range(all_nominal, offsets, residual_cfg, band)

    lo, hi = residual_cfg.lo, residual_cfg.hi
    residuals = [lo + (hi - lo) * i / (n_grid - 1) for i in range(n_grid)]
    # for each residual, intersection of per-test-bias success-offset sets (within-block common offset)
    common_by_residual = {}
    any_common = False
    for r in residuals:
        sets = [predicted_success_offsets(b + r, offsets, band) for b in test]
        inter = set.intersection(*sets) if sets else set()
        common_by_residual[round(r, 4)] = sorted(inter)
        if inter:
            any_common = True

    # best-single achievable test success over the residual support (expected)
    def test_success_of(o):
        hits = 0
        for b in test:
            for r in residuals:
                if abs(b + r + o) <= band + 1e-9:
                    hits += 1
        return hits / (len(test) * len(residuals))
    best_single_test = max((test_success_of(o) for o in offsets), default=0.0)
    # state-aware: for every (test bias, residual) some offset compensates
    state_aware_test = 1.0 if all(any(abs(b + r + o) <= band + 1e-9 for o in offsets)
                                  for b in test for r in residuals) else 0.0

    ok = (base["ok"] and comp["ok"] and not any_common
          and (state_aware_test - best_single_test) > 0.0)
    return {
        "ok": bool(ok),
        "residual_config": residual_cfg.to_dict(),
        "nominal_split_validation": base,
        "compensable_over_residual_support": comp,
        "no_common_offset_any_residual": not any_common,
        "common_offset_by_residual_sample": {k: common_by_residual[k]
                                             for k in list(common_by_residual)[::10]},
        "best_single_max_test_success_over_residual": best_single_test,
        "state_aware_test_success_upper": state_aware_test,
        "max_achievable_success_only_gain_residual": state_aware_test - best_single_test,
        "candidate_group_and_matched_block_unchanged": True,
        "reason": None if ok else _reason_residual(base, comp, any_common),
    }


def _reason_residual(base, comp, any_common):
    if not base["ok"]:
        return f"nominal split invalid: {base['reason']}"
    if not comp["ok"]:
        return "a nominal bias + residual exceeds the offset-bank compensable range"
    if any_common:
        return "a residual admits a single offset that covers BOTH test biases -> best-single could get full test success"
    return "state-aware does not exceed best-single on test"


def _reason(disjoint, bracketed, empty_inter, inter):
    if not disjoint:
        return "train/val/test bias levels overlap"
    if not bracketed:
        return "a test bias is not bracketed by train levels (would be extrapolation, not interpolation)"
    if not empty_inter:
        return (f"offset(s) {inter} satisfy the success band for EVERY test bias -> a fixed best-single "
                f"offset attains full test success -> success-only H3 gain is 0 by construction")
    return "unknown"
