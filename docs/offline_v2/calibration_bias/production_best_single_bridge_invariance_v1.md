# Production best-single ↔ power bridge invariance v1 (offline; BLOCKER 1 evidence)

Claude B (fix2). Machine-auditable proof that the new **observed-outcome-only production** best-single selector
(`confirmatory_v4_selection.select_best_single`) selects **exactly the same offset** as the
`SIMULATION_ONLY_REFERENCE` (`learned_selector_power.best_single_legal`) on every frozen power-cert config —
so replacing the simulation helper with the production interface requires **no power recertification**.
Machine form: `production_best_single_bridge_invariance_v1.json`.

## What was compared
For every frozen power-cert config we regenerate the synthetic train/validation split, then:
1. convert each synthetic candidate into an `ObservedCandidateOutcome` (`split, session_id,
   planned_episode_id, offset, y.success`), filling `y.success` with the synthetic candidate outcome;
2. run the **production** selector on those observed records (`validate_completeness=False`, since the
   larger grid designs have ≠171 records — the fixed-171 confirmatory gate is tested separately);
3. run the **simulation** selector `best_single_legal` (which reconstructs the label from the secret
   `nominal/residual/tau`);
4. compare the selected offsets.

The production selector **never reads** nominal / residual / actual bias / eff / tau / oracle / a success
model / test records — only the observed `y.success`.

## Frozen replay grid
```
test geometry = ±0.035
block designs = 9/6/9, 12/9/12, 18/9/18
primary tau   = 0.0342, 0.03425, 0.0343
seeds/config  = 500      → 9 configs × 500 = 4500 comparisons
```

## Result
| quantity | value |
|---|---|
| total configs checked | **4500** |
| production vs simulation mismatch count | **0** |
| agree on all | **True** |
| `power_recertification_required` | **False** |

Every (design × τ) cell has `mismatch = 0` (`…json → per_config`).

## Reading
- The production selector, using **only observed candidate `y.success`**, reproduces the simulation
  reference's selection in 4500/4500 configs. The simulation reference's secret reconstruction was therefore
  **non-decisive** for the selection — exactly what Claude C found for the tie-break in fix1, now shown for
  the whole selection path against a genuinely secret-free production implementation.
- Because best-single is unchanged, the primary contrast `B2_K1 − best-single`, the effect, and the power
  result are unchanged → `power_recertification_required = false`.

## Gate
```
production_selector_offset == simulation_reference_offset  for all 4500 configs   ✓
→ fix accepted with NO power recertification
```
Any mismatch would have produced `POWER_RECERTIFICATION_REQUIRED`; none occurred.
