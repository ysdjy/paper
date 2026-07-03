# Learned-selector power plan v1 (FROZEN method; NOT run yet)

Claude B, branch `experiment/offline-calibration-bias-v4`. Freezes the METHOD for the learned-selector
power simulation that Claude C's audit requires (MUST-MODIFY #3): the v3 power measured only
**structural/oracle** decision value, but the frozen GO conditions require the **learned B2 selector**.
This plan is **not executed now** — it runs only after the band-edge experiment returns an empirical
success model. Freezing it now prevents choosing any knob after seeing data.

## Why (Audit C §1)
`power.py`/`predicted_k_curve` model the gain as **state-aware oracle vs fixed best-single**, assuming the
selector perfectly classifies bias by probe-outcome and picks the class-optimal offset. Not modeled: B2
prediction error, finite train-N (9 train blocks), model-seed variance, and K=1 mis-classification (the
probe-fail class maps to 3 train nominals with 3 different optimal offsets). So the current number is
**structural/oracle power**, not learned-selector power, and must be reported as such until this sim runs.

## Empirical model input (from the band-edge experiment)
The learned-selector sim uses the **empirically fitted** success model from
`band_edge.fit_logistic_edge` / `fit_isotonic_edge` — the measured edge **center** and **scale** (and
the ± symmetry result) — NOT the assumed soft-0.03 edge. If the band experiment returns
MODIFY_RESIDUAL_OR_DESIGN, this sim does not run until the residual/design is fixed.

## Frozen simulation method
For each candidate confirmatory design (default: the frozen train 9 / val 6 / test 9 blocks):
1. **Generate full synthetic datasets** at the future train/val/test **nominal biases** with the v3
   block residual (`actual = nominal + residual`) and the **empirical** success model; blocks disjoint
   across splits; residual fixed per block.
2. **Generate real diagnostic-trial (probe) outcomes and candidate outcomes** from the same success model
   — including **probe-outcome mis-reads** (a probe is itself a Bernoulli full-task trial near the edge,
   so its outcome can mis-classify the bias).
3. **Run the actual DeepSets K=0 / K=1 training + selection pipeline** (`models_v2`), with:
   - **finite train blocks** (the frozen train count),
   - **≥ 5 model seeds**,
   - **best-single offset chosen on train/val only**, test used for final evaluation only.
4. **CI** = block bootstrap over the **test** nuisance blocks of the learned selector's
   `actual B2 selected-success gain over best-single`.
5. **Power** = `P( 95% CI lower bound of [B2 selected-success gain over best-single] ≥ 0.15 )`.

## Reporting (both, separately)
- **structural / oracle power** (the current v3 number, correctly labeled), and
- **learned-selector power** (this sim). The paper may only claim "powered to detect the learned selector's
  gain" from the second number.

## Decision rule (frozen)
If **learned-selector power < 0.8**: **either** increase the block count **or** narrow the claim (e.g.
scope to the structural existence of decision value / a demonstrated positive control). It is **forbidden**
to keep citing structural power in place of learned-selector power.

## Guardrails (frozen)
- No knob (residual σ, edge scale, block count, λ, best-single offset, model hyperparameters) may be chosen
  using any **confirmatory** outcome. The empirical success model comes only from the **exploratory**
  band-edge data.
- Probe = **repeated-task deployment diagnostic full-task trial**; its time/failure/recovery + reset cost
  enters Net VOI (not just time). No "online / in-task / low-cost / non-destructive" framing.
- Do not treat argmax offset switches as utility; do not treat replicate blocks as independent unless the
  re-run independence audit passes with genuine (band-edge-confirmed) variance; do not extrapolate a
  single-scalar-sim-bias, 3-mm-band, no-perception/backlash/payload result to a real robot.

## Tooling status
The simulation reuses `models_v2` (DeepSets), `offline_v2/calibration_bias/{power,residual_nuisance}.py`,
and `band_edge.py` (empirical model). The runnable implementation is deferred to the post-band-data phase;
this document is the frozen specification it must follow.
