# Probe cost & VOI v1 — test geometry ±0.035 (offline)

Claude B. Recomputes gross/net VOI with the **frozen** full-task probe cost and time weight from the prior
phase, under the new ±0.035 geometry. The probe is a **repeated-task deployment diagnostic full-task trial**
— NOT low-cost, non-destructive, or online. Machine form: `test_geometry_power_verdict_v1.json` → `net_voi`.

## Frozen cost model (unchanged)
```
probe = one full-task deployment trial at the fixed probe offset -0.04
probe_time = 16.58 s
lambda_time = 0.02  (success-equivalent per second)
probe_cost = lambda_time * probe_time = 0.3316   (success-equivalent units)
```

## VOI at the minimal passing design (9/6/9, τ=0.03425)
```
Gross VOI_1task = learned gain (B2_K1 success - best-single success) = 0.520
Net VOI_1task   = 0.520 - 0.3316 = +0.188
break-even T    = probe_cost / gross_gain = 0.64 tasks
```

### Amortized Net VOI(T) = T · gross_gain − probe_cost
| horizon T | Net VOI |
|---|---|
| 1 | **+0.188** |
| 2 | +0.708 |
| 5 | +2.268 |
| 10 | +4.867 |

## Reading (Net VOI now positive one-shot — but kept secondary)
- Under the old ±0.03 geometry the gross gain was only ~0.167, so one-shot Net VOI was **−0.16** (a single
  probe did not pay for itself). At ±0.035 the gross gain jumps to ~0.52, so the **one-shot Net VOI is now
  positive (+0.188)** and break-even is below a single task (T ≈ 0.64).
- **This is reported as secondary and does not change the primary claim.** The probe cost/time weight were
  frozen *before* these results; we did not tune them to make VOI positive. The gain rose purely because the
  test geometry moved best-single into its failure region — the same reason power rose. The larger deployment
  horizons (T = 2/5/10) make the amortized value large, but T must come from **deployment semantics**
  (how many repeated-task trials share one hidden calibration state), not from these numbers.
- **Primary claim remains**: *history-conditioned diagnostic information improves subsequent action-selection
  success* — evidenced by the learned success increment K1−K0 ≈ 0.52 and learned power ≥ 0.984. The positive
  Net VOI is a welcome corollary, not the headline, and is honestly attributable to the same geometry shift
  that a skeptic could argue makes the probe "worth it" only because the deployment condition is now harder
  for the state-agnostic baseline.
