# Confirmatory hard-edge empirical outcome model v1 (offline)

Claude B. The primary outcome model for the redesign and the future confirmatory analysis, derived from the
306-episode run. Because the 306 data are **completely label-separated** in |eff|, we do **not** use an
unsupported soft logistic — the primary model is a **hard threshold** with an empirically-identified
interval, plus sensitivity.

## Primary model (frozen)
    success = 1[ |actual_bias + grasp_offset| ≤ τ ]

- Empirically identified interval (from the 306 run): **max successful |eff| = 0.0342 m**,
  **min failed |eff| = 0.0343 m** → `0.0342 < τ < 0.0343`.
- **Point value:** τ = **0.03425** (interval midpoint).
- **Isotonic 0.5-crossing:** 0.0325 (used as a sensitivity anchor, not the point value).
- The logistic fit's scale is **unidentifiable** (complete separation → scale → 0); we keep the logistic
  only as a *descriptive* boundary fit and never assign it a fabricated identifiable scale (e.g. 0.03).

## Sensitivity set (frozen — the primary conclusion must be robust across ALL of these)
    τ ∈ { 0.0342 (lower), 0.03425 (point), 0.0343 (upper), 0.0325 (isotonic) }

## Required robustness reporting
For any design decision or the future confirmatory analysis, report and require robustness over:
1. **point threshold** τ = 0.03425;
2. **lower-bound** τ = 0.0342 (harder: shrinks the success region);
3. **upper-bound** τ = 0.0343;
4. **isotonic** τ = 0.0325 (widest transition anchor);
5. **block-bootstrap edge-center uncertainty** (306 logistic center CI [0.03412, 0.03452]);
6. **isotonic** monotone sensitivity.

A design's non-degeneracy and gain must hold across the **whole** interval, not only at the point value
(Design A: gain ∈ [0.222, 0.333], non-degenerate at every τ — see the comparison).

## What must NOT be done
- Do **not** substitute a soft logistic with an assumed scale (e.g. `edge_scale = 0.03`) — it is not
  supported by the data (perfect separation).
- Do **not** tune τ, the residual, the offsets, or the bins to obtain a desired result; the threshold
  interval is fixed by the 306 measurement.
- The logistic remains available only as a **descriptive** overlay.

## Use in the confirmatory design
The hard model + sensitivity is the outcome model behind the candidate-bank geometry validation, the
non-degeneracy check, and (next phase) the learned-selector power simulation's success labels. The learned
selector will additionally carry finite-N / model-seed / probe-misread noise; the empirical hard edge is the
ground-truth label generator for those simulations.
