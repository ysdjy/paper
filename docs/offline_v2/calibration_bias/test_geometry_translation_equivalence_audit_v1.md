# Additive translation-equivalence audit v1 (read-only; NO new Isaac data)

Claude B. This phase moves the **primary test nominal** outward to `{-0.035, +0.035}` (unseen deployment
conditions). Before running any power, we must justify — using ONLY existing evidence (135 capability map,
306 band-edge data, runtime code) — that changing the nominal bias and the applied offset affects the
outcome **only** through the additive effective error

```
eff = actual_bias + offset
```

and not through any other pathway (target pose, controller gain, damping, timeout, state-machine
transitions, collision, or the success definition). No new runtime episodes were generated.

## Verdict: `ADDITIVE_TRANSLATION_EQUIVALENCE_SUPPORTED`

Three independent lines of evidence, plus one honestly-flagged caveat.

### 1. Code: bias and offset act additively on the SAME axis
- `deployment_calibration/adapters/articulated_drawer_v2.py:140`
  `p["grasp_offset_local_xyz"] = [0.0, float(theta["grasp_offset_local_y"]), 0.0]`
  → the applied offset enters as a **pure additive translation** on the drawer-link local **+Y** grasp axis.
- `deployment_calibration/adapters/handle_calibration_bias_v1.py:32`
  `BIAS_AXIS = "handle_local_y"  # drawer LINK local +Y (same frame as grasp_offset_local_y)`
  → the hidden calibration bias is injected on the **same** local-Y axis.
- Because both the bias and the offset are additive displacements on one shared axis, the physics can only
  ever see their **sum** `eff = actual_bias + offset`. There is no separate channel by which the nominal
  bias value or the offset value could reach the controller/scene independently of `eff`.
- `deployment_calibration/data_generation/band_edge_validators_v1.py:75`: within a block "target jitter
  identical; ONLY grasp_offset_local_y / candidate id differ" — confirming offset is the *only* thing that
  varies across the bank; everything else is held fixed.

### 2. 306 data: success is a pure function of `abs_eff`; nothing else co-varies
From `deployment_calibration/data/band_edge_full_306_v1/episodes.jsonl` (sha256
`131750e1…9524c20e`, **unchanged**), n=306:
- **Perfect label separation by `abs_eff`**: max `abs_eff` among **successes** = `0.03423`; min `abs_eff`
  among **failures** = `0.03428`. Success is determined by `|eff|` alone — no other recorded field
  (damping, target, timeout, contact, transitions) is needed to explain a single label. This is the
  empirical hard edge the whole study rests on.
- **Controller / scene parameters do NOT co-vary with offset or bias**:
  - `g.target_tolerance` — single value `0.02` across all 306.
  - `damping_eff` — single value `3.0` across all 306 (damping is an independent nuisance dimension, not a
    function of bias/offset).
  - `g.target_open_position` — varies only in `[0.195, 0.205]` as the **independent target-jitter nuisance**
    (18 distinct values, one per nuisance block), uncorrelated with the offset grid.
- So the mapping "nominal bias / offset → success" factors through `eff` with no residual dependence on any
  other pipeline variable.

### 3. Coverage: the decision-relevant `eff` region is fully observed (interpolation)
- Observed `eff_signed` range in the 306: **[-0.0567, +0.0592]**; observed `abs_eff` range
  **[0.0005, 0.0592]**. The hard edge (τ≈0.0343) is bracketed by real successes (up to 0.03423) and real
  failures (from 0.03428).
- The `±0.035` geometry's **decision-relevant** actions all sit inside the observed range:
  - state-aware offset ±0.04 → `|eff| ≤ 0.015` (deep success region, richly observed);
  - best-single offset 0 → `|eff| ∈ [0.025, 0.045]` (straddles the **observed** edge);
  - probe −0.04 at +0.035 → `eff = −0.005` (observed success region).
- The offset grid itself (`o-0.050 … o+0.050`, incl. `o±0.035`) was **already run** in the 306, so the
  additive offset channel is directly exercised.

### Honest caveat (the one place we extrapolate)
The `±0.035` geometry's **non-selected / deliberately-wrong** candidate actions and the probe-**fail**
branch reach `|eff|` up to **0.085**, beyond the observed max `0.0592`:
- test +0.035 with offset +0.04 → `eff ∈ [0.065, 0.085]`;
- probe −0.04 at −0.035 → `eff = −0.075`.
These lie in the **deep-fail** regime. Under the pre-established **hard-edge** outcome model
(`success = 1[|eff| ≤ τ]`), anything already failing at `|eff|=0.0592` also fails at `0.085` — a **monotone**
extrapolation of the fail region, not of the decision boundary. No action the selector would *choose* lives
in the extrapolated region. This is the single modelling assumption we lean on rather than direct
observation, and it is the weakest link if the true edge were non-monotone far from τ (there is no evidence
it is).

## Consequence
Additive translation-equivalence is **supported** for the decision-relevant region, so moving the test
nominal to `±0.035` is a legitimate translation of the effective-error distribution into unseen **nominal**
deployment conditions whose **actual hidden state** `[-0.045,-0.025] ∪ [+0.025,+0.045]` remains inside the
train actual support `[-0.05,+0.05]`. We therefore proceed with the offline power study and do **not** raise
`NEW_RUNTIME_VALIDATION_REQUIRED`. Should a reviewer reject the monotone deep-fail extrapolation, the
fallback is a small confirmatory runtime batch at the ±0.035 nominals — explicitly **out of scope** for this
offline phase.
```
verdict = ADDITIVE_TRANSLATION_EQUIVALENCE_SUPPORTED
```
