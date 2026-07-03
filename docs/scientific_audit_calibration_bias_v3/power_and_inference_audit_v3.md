# Power & inference audit (Audit C)

**Auditor:** Claude C. Independent recomputation from the 135-ep capability map and the v3 power/design
artifacts. This is the decisive audit area.

## Summary of verdicts
1. The power analysis measures **structural / oracle** decision value, **not** a learned B2 selector. 
2. The v3 power sim and the v3 design-validation use **mutually inconsistent success-band models**, and
   the confirmatory operating points sit at `|bias+offset|` values the exploration **never measured**.
3. Under the design's own (hard) band, the residual produces **~0 success-label variance at the operating
   points** → the block-bootstrap CI is at risk of being **degenerate**, reproducing the v2 problem.
None is fatal to the *structural science* (which is genuine), but all three block a clean GO.

## 1. Structural/oracle power ≠ learned-selector power
`power.py` (and `run_confirmatory_design_v3.predicted_k_curve`) model the gain as
**state-aware oracle** = best offset per test bias (deterministic, deep-in-band success) vs a fixed
best-single offset. The K-curve assumes the selector **perfectly classifies** biases by probe-outcome
pattern and picks the class-optimal offset. **Not modeled:** B2 prediction error, finite train-N (9 train
blocks), model-seed variance, and K=1 mis-classification.

But the frozen **GO conditions require the *learned* selector**: §4(3) "success-only selected-success gain
≥ 0.15 AND frozen VSI ≥ 0.05," §4(4) "history selector beats best-single toward the state-aware oracle,"
§4(5) "stable across model seeds." **Power for those learned-selector quantities is not established.**

Mitigating (but not sufficient): the K=1 discrimination is a clean **1-bit** rule — probe `m040`
succeeds at +0.03 (`|+0.03−0.04|=0.01`) and fails at −0.03 (`|−0.03−0.04|=0.07`), so the ideal selector
is trivial to *state*. That makes the structural proxy plausible, but **plausible ≠ demonstrated**: 9
train blocks with the "probe-fail → which of {0,−0.02,−0.04}" ambiguity (probe-fail maps to 3 train
nominals with 3 different optimal offsets) can still mislead a learned head. **Required:** a
learned-selector power simulation (probe→classification→offset with finite-N + seed noise), OR the power
claim must be **explicitly relabeled "structural/oracle decision-value power"** and the paper must not
present it as B2-model-test power.

## 2. Band-model inconsistency + unmeasured operating points (first-hand)
- **Design-validation & K-curve** use the **hard band**: `COMPENSATION_BAND = 0.02`, success ⇔
  `|bias+offset| ≤ 0.02` (confirmed constant `design_validation.COMPENSATION_BAND`).
- **Power sim** uses a **soft edge centred at 0.03**: `p_succ = sigmoid((0.03 − |eff|)/s)`,
  `band_center=0.03` (confirmed in `power_baseline_residual.model`). At the best-single point |eff|=0.03
  this gives 0.5; under the hard band it gives 0 (fail).
- **The exploration measured neither at |eff|=0.03.** Recomputed: the capability grid has bias∈{even×0.02}
  and offset∈{even×0.02}, so every measured `|eff| ∈ {0, 0.02, 0.04, 0.06, 0.08, 0.10}`. The confirmatory
  **val/test biases {±0.01, ±0.03} are odd×0.01**, so *all* their operating points land at
  `|eff| ∈ {0.01, 0.03, 0.05, …}` — **0 of which the exploration sampled** (verified: every test operating
  |eff| returns "measured? False").

The entire decision-value signal (best-single behaviour and the band edge) lives at **|eff|=0.03, an
unmeasured point**, and the two frozen artifacts disagree about what happens there.

## 3. Under the hard band, the residual creates ~0 variance at the operating points
Recomputed (hard band, residual ∈ [−0.01,+0.01]):
- State-aware operating offsets for the test biases give `|eff|=0.01` → **deterministic success**, no flip.
- Best-single offset 0.0 gives `|eff|=0.03` → **deterministic fail** except at the exact residual cap
  ±0.01 (measure-tiny boundary). Gain ≈ **1.0**, near-deterministic.
- Probe `m040` outcomes at the test biases are deep (|eff|=0.01 / 0.07) → **deterministic**, no flip.

So *if the true band is the measured hard `≤0.02`*, the confirmatory statistic is **near-deterministic**:
state-aware ≈ 1.0, best-single ≈ 0.0, on every block. The **block-bootstrap over 9 test blocks then has
≈ 0 variance → a degenerate CI**, and the independence gate's "genuine residual-driven variance"
(§4(6)) is **not met** — the residual would be present but inert at the operating cells, exactly the v2
failure the residual was introduced to fix. The v3 power numbers (0.89–0.97) get their non-degenerate
variance **only** from the soft-edge assumption (band 0.03), which contradicts the hard band and is
unmeasured.

**Note on effect direction:** the *point* gain is robust — whether best-single is 0 (hard) or 0.5 (soft),
gain ≥ 0.5 ≥ the 0.15 threshold — so H3's structural existence is safe. The risk is entirely in the
**variance / CI / independence** machinery, not the sign of the effect.

## 4. Are 9 test blocks enough?
- For the **structural** gain (~0.5–1.0 vs threshold 0.15): yes under the soft model; **vacuously yes but
  CI-degenerate** under the hard model.
- For the **learned** selector's 95% CI: **not established** (no learned-selector sim).
- **675 episodes / 75 sessions** arithmetic verified (9·5+6·2+9·2 = 75 sessions × 9 eps = 675). The scale
  is reasonable **only if** the residual actually produces analyzable variance at the operating points —
  which §2–§3 show is unverified.

## 5. Required modifications (minimal) and permitted statements
**Must fix (preregistration-level, pre-implementation):**
1. **Measure the band edge at odd |eff|** (especially 0.03, also 0.01/0.05) in a small pre-confirmatory
   exploration (e.g. add bias or offset at 0.01 resolution near the edge). Locate the true edge and its
   softness; **confirm the residual induces genuine success-label variance at the confirmatory operating
   cells** (or adjust residual magnitude / operating offsets so it does).
2. **Reconcile the success-band model** between `power.py` (0.03 soft) and `design_validation` (0.02 hard);
   recalibrate `edge_scale` from measured data, not assumption.
3. **Add a learned-selector power simulation** (probe→class→offset with finite-N + model-seed noise), OR
   relabel the current power as **structural/oracle** and restrict the powered claim accordingly.

**Do NOT:** choose the residual σ, edge_scale, block count, or λ after seeing any confirmatory outcome.

**Permitted statement:** "the design is powered to detect the *structural* (oracle) decision-value gain."
**Forbidden until fixed:** "the confirmatory run is powered to detect the learned history-selector's
gain," or any power claim that assumes the soft 0.03 edge without measuring it.
