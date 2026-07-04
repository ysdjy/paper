# Band-edge characterization — analysis v1 (read-only, exploratory)

Claude B, branch `experiment/offline-calibration-band-edge-analysis-v1`. Read-only analysis of the
authorized 306-episode **exploratory** band-edge run under the FROZEN analysis plan (0.005 m bins,
logistic + isotonic, block-bootstrap n=2000, block statistical unit, frozen exit criteria). **Nothing
was chosen from the data.** Not confirmatory data; never mixed into any future confirmatory train/test.
No Isaac, no new runtime data, no second 306, no learned-selector power, no preregistration v4.

## Provenance / reproduction
- input: `deployment_calibration/data/band_edge_full_306_v1/` (source commit `e447f00`)
- episodes.jsonl sha256: `131750e158032e53e5c8daab6e94530c70de1224f05b8fc79a7497009524c20e` (matches authorization)
- config sha256 `2f20429b…`, science-manifest sha256 `e2c8216e…` (both match)
- analysis: python 3.11.15, numpy 1.26.0; bin width 0.005, n_boot 2000, CI 0.95; bootstrap **seed 320295137**
  (protocol did not freeze a seed → derived deterministically from the episodes hash and recorded).
- reproduce: `python -m deployment_calibration.offline_v2.calibration_bias.run_band_edge_analysis <data_dir> --out <docs>`

## Exit verdict: **MODIFY_RESIDUAL_OR_DESIGN**
The run measured the true success boundary cleanly, but two frozen MODIFY triggers fire:
1. **the success band is a HARD threshold with an unidentifiable scale** (complete label separation);
2. **the v3 block residual is inert at the future confirmatory operating region** (every operating cell is
   deterministic), so the confirmatory block-bootstrap CI would be degenerate — the exact failure mode the
   band-edge study was designed to detect.
Everything else is clean (boundary mixed labels present, no severe asymmetry, ContactSensor available,
instrumentation complete, edge not collision-driven, data integrity PASS). This is not a STOP (no leakage,
contract, or frozen-plan breach) and not a data failure.

## 1. Data integrity — PASS (see `band_edge_data_integrity_audit_v1.md`)
All 21 structural checks + the three identities recomputed independently pass (COMPLETE, self-check ok, 306
unique, 18×17, frozen grid, nominal 0, 0 probes, 0 smoke, all planned/no unplanned, config/science/episodes
hashes match, code_commit e447f00, schema+leakage valid, matched-block valid, instrumentation complete,
ContactSensor available). `actual = nominal + residual`, `eff_signed = actual + offset`, `abs_eff =
|eff_signed|` hold to 1e-6/1e-9. 211 success / 95 failure.

## 2. The success boundary — HARD threshold at |eff| ≈ 0.0343
Binned success (`band_edge_binned_success_rates_v1.csv`, block-bootstrap 95% CI; plot
`band_edge_success_vs_abseff_v1.png`):

| bin [lo,hi) | n | blocks | success rate (95% CI) |
|---|---|---|---|
| 0.000–0.025 (5 bins) | 144 | 18 | **1.00** (all) |
| **0.025–0.030** | 36 | 18 | **1.00** |
| **0.030–0.035** | 36 | 18 | **0.86 [0.75, …]**  (31 succ / 5 fail — MIXED) |
| 0.035–0.040 | 28 | 18 | 0.00 |
| ≥ 0.040 | 62 | — | 0.00 |

**Complete separation:** max success |eff| = **0.0342**, min failure |eff| = **0.0343** → **no overlap**.
- **Logistic** (`band_edge_logistic_fit_v1.json`): `edge_center = 0.03426` [95% CI **0.03412, 0.03452**];
  `edge_scale → 1e-5` (hit the lower bound) — the scale is **NOT identifiable** because the labels are
  perfectly separated (the MLE scale → 0, a hard step). This is reported honestly, not smoothed.
- **Isotonic** (`band_edge_isotonic_fit_v1.json`): 0.5-crossing **0.0325** [CI 0.0325, 0.0325], 0
  unobtainable crossings — a monotone step consistent with the hard edge; the small logistic/isotonic
  center offset (0.0343 vs 0.0325) is the difference between the mid-transition-bin isotonic estimate and
  the separating-point logistic center.

Both models agree the boundary is sharp; neither is discarded. The edge (~0.034) is **larger** than the
capability-map assumption (0.02–0.03) — a genuine measurement the confirmatory design must use.

## 3. Boundary mixed labels (0.025 ≤ |eff| ≤ 0.035) — `BOUNDARY_MIXED_LABELS = true`
72 episodes / 18 blocks, 93% success; the frozen bin **[0.030, 0.035)** (which *contains* 0.03, not
exactly equals it) has **31 success / 5 failure (86%)** — both labels present. See
`band_edge_residual_label_variation_v1.md`.

## 4. Directional symmetry — `NO_SEVERE_ASYMMETRY`
neg-direction center 0.03386, pos-direction center 0.03452, |Δ| = **0.00067 < 0.005** threshold; matched-bin
rate differences small (`band_edge_directional_asymmetry_v1.md`, plot `band_edge_directional_v1.png`).

## 5. Residual-induced label variation — mechanism present, but INERT at the operating region
- **Observed (mechanism present):** offsets ±0.030 / ±0.035 / ±0.040 show **mixed** success/failure **across
  the 18 blocks** (e.g. +0.030: 14 succ / 4 fail; −0.035: 7 / 11) — the residual genuinely flips labels for
  offsets whose |eff| sits near the 0.034 edge (plot `band_edge_per_offset_blocks_v1.png`).
- **Future v3 operating region (decisive):** mapping the v3 confirmatory test nominal biases (±0.03) + the
  FROZEN residual support [−0.01, +0.01] to |eff|, **every operating cell is deterministic (probability 0 or
  1)**: the state-aware compensating offset gives |eff| ∈ [0, 0.02] (deep in-band → always success); the
  v3 best-single offset gives |eff| deep in-band or past the edge (always success / always fail). **None
  straddles the 0.034 edge.** So at the operating cells the residual produces **~0 block-level label
  variation** → a **degenerate** confirmatory block-bootstrap CI.
- Verdict: `RESIDUAL_INDUCES_BLOCK_LABEL_VARIATION` (at the operating region) = **false**. The residual is
  not merely nonzero (it flips edge-adjacent offsets) but it does **not** create variance where H3 needs it.

## 6. Collision & mechanism — `EDGE_NOT_COLLISION_DRIVEN = true`
All 306 episodes: `contact_sensor_available = true`, **max unintended contact force = 0.0 N**, 0 episodes
with any contact. Collision plays **no** role; full vs collision-excluded results are identical. Frozen
threshold rule → `THRESHOLD_NOT_IDENTIFIABLE_FROM_FROZEN_RULE` (an all-zero distribution has no noise floor
to place a threshold; any ε>0 separates). Failure mechanism vs |eff| is compensation-consistent:
HANDLE_DETACHED/PULL near the edge (|eff| 0.034–0.059, mean 0.039), POSITION_TIMEOUT/APPROACH far (0.037–0.057,
mean 0.046). See `band_edge_collision_and_instrumentation_v1.md`.

## 7. Instrumentation / confounds — complete, edge not a joint/IK artifact
`INSTRUMENTATION_COMPLETE = true`; **0** episodes with real joint-limit involvement (meaningful negative
signed margin AND limit-clamp), **0** IK failures →
`PRIMARY_EDGE_MECHANISM_NOT_EXPLAINED_BY_JOINT_OR_IK_CONFOUND = true`. The ~−1e-5 rad signed-margin
excursions are the documented soft-limit numerical artifact (schema v1.1 clarification), not joint-limit
failures.

## 8. Continuous outcomes (exploratory secondary) — `band_edge_continuous_outcomes_v1.md`
success vs failure means: task_error 0.011 vs 0.174; time 9.4 s vs 23.8 s; true-handle-error-at-close 0.0072
vs 0.023 m — failures have larger close-time handle error and longer time, consistent with off-centre grasp.

## What MODIFY means here (recommendation only — NOT implemented this phase)
The measured hard edge (~0.034) + the 0.02-spaced offset grid + the small residual (σ=0.005, |·|≤0.01) make
the state-aware compensated |eff| stay ≤ 0.02 for all residuals — never straddling the edge — so the residual
is inert at the operating cells and the confirmatory CI would be degenerate. A future
MODIFY_RESIDUAL_OR_DESIGN cycle (separate authorization) should consider: a larger residual σ/support so the
compensated |eff| can cross the measured edge; and/or placing the confirmatory decision region near |eff|
≈ 0.034; and/or a finer offset grid so compensation is imperfect enough to straddle. **B does not implement
any of this here** — no residual/design change, no learned-selector power, no preregistration v4, no
confirmatory data.

_All figures are labelled "EXPLORATORY characterization, NOT confirmatory."_
