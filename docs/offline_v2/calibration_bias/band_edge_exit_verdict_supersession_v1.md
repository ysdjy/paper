# Band-edge exit verdict — supersession note v1 (offline; minor issue 7.1)

Claude B (fix1). Clarifies the evidence chain around the stale exit verdict without modifying any prior
artifact. Machine form: `band_edge_exit_verdict_supersession_v1.json`.

## What the old verdict said
`band_edge_exit_verdict_v1.json` returned **`MODIFY_RESIDUAL_OR_DESIGN`** with:
`boundary_mixed_labels_0.025_0.035 = true`, `residual_induces_block_label_variation = false`,
edge_center ≈ 0.034256, isotonic crossing 0.0325, boundary_success_rate 0.9306. That audit was run for the
**original test geometry with test nominals at ±0.03** and the then-current design.

## Why it is superseded (for the confirmatory operating point)
The confirmatory design **moved the test nominals to ±0.035** (test-geometry power study). At ±0.035:
- the state-agnostic best-single (offset 0) straddles the empirical edge, `|eff| ∈ [0.025, 0.045]` →
  block-level label variation is now present at test (the very property the old verdict found missing at ±0.03);
- the state-aware ±0.04 stays deep (`|eff| ≤ 0.015`), giving a large, low-variance decision-value gain;
- this was certified `POWER_SUFFICIENT_FOR_PREREG_V4` (`test_geometry_power_verdict_v1.json`), using the same
  independent 306 band-edge data (`episodes.jsonl` sha256 `131750e1…9524c20e`) and the same edge center
  0.03425 / interval [0.0342, 0.0343].

So the `MODIFY_RESIDUAL_OR_DESIGN` conclusion — correct for ±0.03 — **does not apply** to the ±0.035
operating point; the design was in fact modified (test geometry), which is exactly what that verdict asked for.

## What this note does NOT do
- It does **not** modify `band_edge_exit_verdict_v1.json` (kept as the historical record for ±0.03).
- It does **not** recompute or alter the 306 data or any prior analysis.
- The old verdict remains valid **for the geometry it audited**.

This is a documentation-only clarification of the evidence chain, per fix1 §7.1.
