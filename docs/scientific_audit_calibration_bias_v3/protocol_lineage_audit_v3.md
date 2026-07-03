# Protocol lineage audit — calibration-bias preregistered v3

**Auditor:** Claude C (independent, pre-run). No Isaac launch, no confirmatory data generated, no A/B
runtime/offline/preregistration file modified. All facts re-derived first-hand from the integration
snapshot.

## Provenance (verified)
- Audit branch/worktree derived from `paper/experiment/integration-calibration-bias-preregistered-v3`
  @ **`2102563d07425a45de1fde15c62bcdac1ba712fd`** (the "origin/…" in the task is on the `paper` remote).
- Integration merges (verified in `git log`): runtime `experiment/runtime-calibration-bias-v1` (`24919cd`)
  + offline `experiment/offline-calibration-bias-v1` (`8b3f638`), both `--no-ff`. The unrelated
  monorepo branch `experiment/scientific-audit-v2` was **NOT** merged (confirmed absent from history).
- Capability-map data `calibration_bias_capability_map_v2_20260703_162715/episodes.jsonl` sha256
  **`de21417390a80b5b71a2c593d810fdd35553d008124db9ea7739d69a1365a030`** — **matches the manifest**.
- **`confirmatory_data_exists = false`** — verified: only the 135-ep exploratory capability map and an
  injection smoke exist under `data/`; no confirmatory / selection run present.
- Torch-free framework tests: **27 passed** (design_validation, power, residual, splits, validator).
  (The one collection error is a missing `torch` in the base interpreter, not a code fault.)

## v1 → v2 → v3 lineage (all pre-data)
| version | change | pre-data? |
|---|---|---|
| v1 | initial preregistration + freeze proposal (bias split, 7-offset bank, probes, AND gate, 6 instrumentation fields) | yes |
| v2 | pre-data split fix; power sizing assuming a per-block **grasp perturbation δ** moving the effective offset | yes |
| **v3** | replaces v2's δ (which the runtime does **not** inject — capability map showed 0 label flips) with a per-block **residual calibration bias** `actual = nominal + residual`; re-runs design validation + power; split/bank/probes/utility/gate/instrumentation unchanged | yes |

**Assessment.** The lineage is clean and honestly documented; v3 is genuinely pre-confirmatory-data and
is not tuning to a confirmatory result (there is none). The v2→v3 motivation is correct: v2's δ was not
runtime-faithful.

## Caveat that propagates into the power/inference audit
v3 claims the residual is "runtime-aligned" and "genuinely moves the outcome." This is **only partially
substantiated**: the residual is implementable (it rides the same handle-Y axis as the bias — see the
residual/leakage audit), but whether it produces *genuine success-label variance at the confirmatory
operating points* is **unverified**, because those operating points sit at `|bias+offset|` values the
exploration never measured, and the v3 artifacts use two mutually inconsistent success-band models (hard
`≤0.02` in design-validation/K-curve vs a soft edge centred at `0.03` in the power sim). See
`power_and_inference_audit_v3.md`. The lineage is sound; the *evidentiary basis of the v3 power fix is
not yet closed*.
