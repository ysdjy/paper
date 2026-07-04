# Probe cost & claim scope v1 (offline)

Claude B. Separates the two claims the learned-selector power bears on, and reports the probe economics
honestly. The probe is a **repeated-task deployment diagnostic full-task trial** (never "low-cost",
"non-destructive", or "one-shot in-task"). Numbers finalize from `learned_selector_power_results_v1.json`.

## Claim S — success decision value (the mechanism)
> A history-conditioned selector that runs one probe learns to pick the compensating action and improves
> subsequent grasp **success** over the state-agnostic best-single.

This is supported at the **mechanism** level: the real DeepSets K=1 reaches ~0.98 test success vs the
capacity-matched K=0 / best-single ~0.81 — a **history increment (K1 − K0) ≈ 0.17–0.18**, robustly positive
across designs. Whether it clears the strict pre-registered **gain-over-best-single ≥ 0.15 with a
non-degenerate block-bootstrap CI and 4/5 seed stability** is the learned-power question
(`learned_selector_power_verdict_v1.json`).

## Claim V — net deployment value (probe cost)
> Benefit after paying the probe's cost.

Probe cost (one full-task diagnostic trial, time proxy from the 306 run):
`cost = λ_time × probe_time = 0.02 × 16.58 s ≈ 0.332`.

- **One-shot Net VOI** = gross success VOI − cost = (learned gain ≈ 0.14–0.18) − 0.332 ≈ **−0.15 to −0.19
  (NEGATIVE)**. A single-task probe does **not** pay for itself under the frozen utility.
- **Amortized** `NetVOI_horizon(T) = T × per_task_success_gain − one_time_probe_cost`. For a per-task gain
  g (from the run) and cost 0.332:

  | T | NetVOI (g≈0.16) |
  |---|---|
  | 1 | ≈ −0.17 |
  | 2 | ≈ −0.01 |
  | 5 | ≈ +0.47 |
  | 10 | ≈ +1.28 |

  (Exact values with the run's g are in `learned_selector_power_verdict_v1.json`.) Break-even is around
  T ≈ 2 tasks — but **T must come from deployment semantics, not chosen from these results.** We do **not**
  set T = 2 just because that is where it turns positive.

## Recommendation to preregistration v4 (proposal only — not decided here)
The old unconditional **"Net VOI(K1) > 0"** GO gate must NOT be reused unchanged (it is false one-shot).
Two honest options:

- **Option 1 — Success-primary (recommended default).** Primary GO = the **success decision value** (and,
  given the learned-power findings, likely the **history increment K1 vs K0** rather than the strict
  gain-over-best-single ≥ 0.15 bar). Net VOI is secondary and its one-shot negativity is stated openly. The
  paper claims decision value on **success**, not single-task net utility.
- **Option 2 — Repeated-deployment value.** Only if the real application guarantees the **same** hidden
  calibration state persists across ≥ T subsequent tasks, freeze an externally-justified horizon T from the
  deployment workflow (e.g. a fixed number of picks between recalibrations) and amortize the probe cost.
  T is set by task semantics, never by the power results.

**Recommended: Option 1** — the demonstrated, robust quantity is the success/history mechanism; single-task
net utility is honestly negative and should not be claimed. This is a **proposal**; preregistration v4 is
NOT issued here.
