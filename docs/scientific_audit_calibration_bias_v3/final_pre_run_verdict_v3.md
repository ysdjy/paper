# Final pre-run verdict — calibration-bias preregistered v3

**Auditor:** Claude C (independent, pre-run). No Isaac launch, no confirmatory data, no A/B code or
preregistration modified. Verdict drawn from the seven area audits in this directory, all anchored to
first-hand recomputation.

## VERDICT: **MODIFY_PREREGISTRATION**

Not GO (the frozen statistical inference rests on unmeasured / internally-inconsistent assumptions that
could make the confirmatory analysis degenerate). Not STOP (the science is sound: no leakage, a large and
*real* structural decision-value upper bound, genuine history information — none of the STOP conditions
holds).

### Why not STOP
- **History carries genuine incremental information:** probe `m040` cleanly discriminates the two test
  biases (1-bit), and optimal offset ≈ −bias moves with the state.
- **No hidden-state leakage:** bias/actual/nominal/residual are secret-only; `x` whitelist + two guards +
  first-hand data check are clean; matched offset bank and leakage-safe history verified.
- **The oracle upper bound exists and is large:** recomputed frozen VSI **+0.52**, success-only VSI
  **+0.40** on train; **no robust generalist** (worst-case gap 1.42 ≫ 0.02). This is a genuine
  improvement over the v2 damping stage (where VSI≈0). There *is* usable decision-value headroom.

### Why not GO
The frozen inference machinery is not yet on solid ground:
- **Power covers only structural/oracle selection, not the learned B2 selector** the GO conditions require.
- **The success-band model is internally inconsistent** (hard 0.02 in design-validation/K-curve vs soft
  0.03 in the power sim) and **every confirmatory val/test operating point sits at an unmeasured `|eff|`**
  (0.01/0.03/0.05 — the exploration only measured even multiples of 0.02).
- **Under the design's own hard band, the residual produces ~0 success-label variance at the operating
  points** → the 9-block bootstrap CI risks being **degenerate**, reproducing the v2 variance failure the
  residual was meant to fix.
- **The collision instrumentation is off and under-specified**, so the confirmatory data could not exclude
  a collision cause for the band-edge (APPROACH-timeout) failures that coincide with the operating region.

None of these is a scientific dead end; each is a fixable pre-data gap. Hence MODIFY, not STOP.

---

## MUST MODIFY (preregistration-level; fix before any runtime implementation)

1. **Measure the success band at odd `|eff|` (esp. 0.03).** Run a small pre-confirmatory exploration
   sampling `|bias+offset| ∈ {0.01, 0.03, 0.05}` (bias or offset at 0.01 resolution near the edge).
   Deliverables: the true edge location + softness, and a demonstration that the block residual induces
   **genuine success-label variance at the confirmatory operating cells**. If it does not, adjust the
   residual magnitude and/or the operating offsets so it does. *(Resolves Audit C §2–3; unblocks the
   independence/variance risk in Audit F §7.)*
2. **Reconcile the success-band model** between `power.py` (0.03 soft) and `design_validation`
   (0.02 hard); recalibrate `edge_scale` from the measured edge, not an assumption. *(Audit C §2.)*
3. **Add a learned-selector power simulation** (probe → classification → offset selection under finite
   train-N and model-seed noise), **or** relabel the current analysis as *structural/oracle* power and
   scope the powered claim to it (do not present it as B2-model-test power). *(Audit C §1.)*
4. **Concretize and enable the collision/contact criterion** — a `ContactSensor` (frozen force threshold,
   per-episode max-force + contact-phase log) or a reproducible signed-distance no-collision check.
   Replace the vague "availability or verification" wording. *(Audit E #7.)*
5. **Correct the hidden-state wording** throughout the preregistration/paper: the experiment tests
   **generalization to unseen *nominal* deployment conditions with in-support actual state**, NOT "unseen
   hidden state / unseen physical bias." Test actual-bias support ⊂ train actual-bias support (verified).
   *(Audit A.)*

## RECOMMENDED ENHANCEMENTS (strengthen, not strictly required)

- **Net VOI net of probe risk:** charge probe failure/recovery + a reset cost, not only probe time; state
  the reset assumption. Frame probes explicitly as **diagnostic full-task trials for repeated-task,
  same-deployment calibration** — forbid "online/in-task/low-cost/non-destructive" wording. *(Audit D.)*
- **Primary inference on continuous outcomes** (task_error, handle-error, time) alongside success-only, as
  Claude B's own independence adjudication recommended — mid-band success is deterministic, so continuous
  outcomes carry more signal per block. *(Audit F §7, independence_adjudication_v1.)*
- **Gate `OBSERVABLE_NUISANCE_KEYS`:** drop them for the confirmatory run, or prove each is independent of
  `actual_bias` with a leakage test before exposing to a model. *(Audit B §5.)*
- **Scope the decision-value claim:** state that the test set was *constructed* as two symmetric extremes
  requiring opposite offsets (a pre-registered positive control), so decision value is demonstrated to
  *exist*, not shown to be generically large. *(Audit A §5.)*

## Minimal modification scope

The scientific core (bias split, 7-offset matched bank, probe order, utility, AND gate, VSI/oracle/VOI
machinery, leakage guards, block-bootstrap design) is **sound and can be retained**. The required changes
are confined to: **(i) one small band-edge exploration + a re-derived, self-consistent band/power model;
(ii) a learned-selector power sim or an honest relabel; (iii) a concrete collision criterion; (iv) claim
wording (hidden-state + probe semantics).** Block counts (9/6/9), 675 episodes, and the split structure do
not need to change *if* item (1) shows genuine operating-point variance; if it does not, the residual /
operating offsets (not the block counts) are what must move.

## Guardrails for the next phase (not authorized to act on here)
- **No runtime implementation is authorized by this audit.** If, after the MUST-MODIFY items, the design
  is re-frozen, the sequence is: A implements generator + residual + the six instrumentation fields →
  **schema / leakage / instrumentation / safety smoke tests** → **only then** does a full 675-episode run
  require **explicit user approval**.
- **Do not** pick residual σ, edge_scale, block counts, λ, or the best-single offset using any confirmatory
  outcome. **Do not** treat argmax offset switches as utility, treat replicate blocks as independent unless
  the re-run independence audit passes with genuine variance, or extrapolate the sim result to a real robot
  (single scalar sim bias, 3-mm-scale band, no real perception/backlash/payload distribution).
