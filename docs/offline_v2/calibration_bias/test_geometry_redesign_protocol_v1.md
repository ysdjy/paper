# Test-geometry redesign & power protocol v1 (offline; NO Isaac, no confirmatory data)

Claude B. Pre-registers the frozen protocol for the test-geometry power study, executed on branch
`experiment/offline-calibration-test-geometry-power-v1` from
`46d03afb41f514e87e69180c7543c1b01389ce2c`. Machine form: `test_geometry_power_config_v1.json`.

## 1. Why this phase
The prior learned-selector power phase found the mechanism is real (B2/K1 uses the probe: K1≈0.98 vs
K0≈0.82, history increment ≈0.17) but the pre-registered power event

```
CI_lower( B2_K1_success − train/val-only best_single_success ) ≥ 0.15
```

could not be powered at any block count — because the old test nominal `{-0.03,+0.03}` left best-single
(offset 0) at ≈0.82 success, so the structural gain was only ≈0.18 (too close to 0.15, too high-variance
per block; even the **oracle** failed the event). The diagnosis was **test geometry**, not model or
threshold. This phase fixes only the geometry.

## 2. Frozen (unchanged from prior phase)
```
candidate bank            = {-0.04, 0.00, +0.04}
train nominal biases      = {-0.04,-0.02,0.00,+0.02,+0.04}
validation nominal biases = {-0.01,+0.01}
residual ~ TruncatedNormal(0, σ=0.005, support [-0.01,+0.01])
K=1 fixed probe offset    = -0.04
primary τ  = 0.03425 ; sensitivity τ = {0.0342, 0.03425, 0.0343} ; isotonic τ = 0.0325 (anchor only)
model      = real models_v2.DeepSets, d_hid=32 d_emb=16 lr=1e-2 max_epochs=200 patience=25 l2=1e-4
model seeds= {1103,2207,3301,4409,5519}
```
No model hyperparameter is tuned.

## 3. NEW primary test geometry (frozen BEFORE any confirmatory data)
```
primary test nominal = {-0.035, +0.035}
```
Rationale: empirical edge ≈ `|eff|=0.0343`; best-single offset 0 at ±0.035 gives `|eff| ∈ [0.025,0.045]`,
straddling the edge (test success ≈0.44); state-aware ±0.04 gives `|eff| ≤ 0.015` (≈1.0); structural gain
jumps to ≈0.55. `±0.035` is unseen **nominal** deployment conditions; the **actual hidden state**
`[-0.045,-0.025] ∪ [+0.025,+0.045]` stays inside the train actual support `[-0.05,+0.05]`. Allowed framing:
"generalization to unseen nominal deployment conditions, actual hidden state still inside training support."
Forbidden framing: "unseen hidden state."

## 4. Geometry audit (gate, pre-power) → `test_geometry_geometry_audit_v1.json`
At `tn ∈ {-0.035,+0.035}`, residual `∈[-0.01,+0.01]`, offset `∈{-0.04,0,+0.04}`, τ`∈{0.0342,0.03425,0.0343,0.0325}`:
- **state-aware**: oracle offset ±0.04 succeeds on the whole residual support at every τ (`|eff|≤0.015`).
- **best-single**: train/val-only selection stays offset 0 and crosses the edge at test (0<frac<1, non-degenerate).
- **no robust common action**: no fixed offset succeeds on both nominals in ≥90% of residual blocks.
- **probe**: fixed probe −0.04 is a deterministic 1-bit discriminator (success at +0.035, fail at −0.035)
  over the full support and all τ.
- **support**: test actual support ⊂ train actual support.
If any check fails → stop and report `GEOMETRY_REDESIGN_FAILED`. (Result: **PASSED**, all τ incl. isotonic.)

## 5. Translation-equivalence audit (read-only) → `test_geometry_translation_equivalence_audit_v1.md`
Using 135 map / 306 data / runtime code only, verify nominal bias & offset affect the result ONLY via
`eff = actual_bias + offset`. Output `ADDITIVE_TRANSLATION_EQUIVALENCE_SUPPORTED` or
`NEW_RUNTIME_VALIDATION_REQUIRED`. (Result: **SUPPORTED**; success perfectly separated by `abs_eff`;
bias & offset share the `handle_local_y` additive axis; controller params don't co-vary; only deep-fail
non-selected actions extrapolate monotonically.)

## 6. Power study
Real DeepSets K=0/K=1, block grid `{9/6/9, 12/9/12, 18/9/18}`, ≥500 Monte-Carlo replicates per primary
design/τ, 5 model seeds, test-block bootstrap `n_boot=2000`. Non-converged training counts as a power
**failure** (conservative). Sample-size rule: **smallest total-block design that clears the bar at ALL
three primary τ; ties → larger test, then larger train.**

## 7. Primary comparator & bar (UNCHANGED — threshold NOT lowered)
```
Primary contrast : B2_K1_selected_success − train/val-only best_single_success
Power event      : CI_lower(Δ) ≥ 0.15
  AND mean Δ > 0 ; oracle success ≥ 0.85 ; best-single non-degenerate ;
  ≥4/5 seed point gains ≥ 0.15 ; no seed point gain < 0
Power passes     : learned power ≥ 0.85 AND its MC 95% CI lower ≥ 0.80, at ALL of τ∈{0.0342,0.03425,0.0343}
```
τ=0.0325 (isotonic) is sensitivity only.

## 8. Separate reporting
structural/oracle gain & power; learned B2 gain & power; K1 vs K0 history increment; seed stability;
best-single offset stability; selected-offset distribution; no-robust-common-action; leakage. Oracle power
is **never** substituted for learned power.

## 9. Probe cost (secondary)
Reuse the frozen full-task probe cost/time weight: `Gross VOI_1task`, `Net VOI_1task = gross − λ·t`
(λ=0.02, t=16.58 s), `NetVOI(T)` for T∈{1,2,5,10}. Even if one-shot Net VOI turns positive it stays
secondary; the primary claim remains "history-conditioned diagnostic information improves subsequent
action-selection success."

## 10. Anti-post-hoc-selection
Primary is `±0.035` only. Pre-allowed sensitivity: `±0.034`, `±0.036` (9/6/9, all τ). Sensitivity cannot
replace the primary; if ±0.035 failed and ±0.036 passed we would NOT swap — a new freeze would be required.

## 11. Verdict (one of)
`POWER_SUFFICIENT_FOR_PREREG_V4` / `INCREASE_BLOCK_COUNTS` / `CLAIM_REDESIGN_REQUIRED` /
`NEW_RUNTIME_VALIDATION_REQUIRED` / `STOP_CURRENT_CONFIRMATORY_CONCEPT`. → `test_geometry_power_verdict_v1.json`.

## 12. Hard constraints
No Isaac; no runtime/confirmatory episodes; no confirmatory generator; no preregistration v4. Writes only
under the three allowed calibration_bias dirs. 306 raw data, runtime, capability map, and all prior
preregistration/redesign artifacts are untouched.
