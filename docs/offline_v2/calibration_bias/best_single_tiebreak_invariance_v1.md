# Best-single tie-break selection invariance v1 (offline; BLOCKER A evidence)

Claude B (fix1). Machine-auditable proof that removing the illegal secret tie-break from best-single
(`tau − |eff|`, where `eff = actual_bias + offset` reads the hidden state) changes **no** selection on any
frozen power-certification config — so the fix requires **no power recertification**. Machine form:
`best_single_tiebreak_invariance_v1.json`.

## What was compared
For every frozen power-cert config, we regenerate ONLY the train/validation split (no model training, no
full power run) and compare the offset selected by:
- **old rule** (illegal): max mean success → max mean `τ−|eff|` (**secret**) → min |offset|;
- **new legal rule** (fix1): max mean success → min |offset| → fixed bank order `{-0.04, 0, +0.04}`.

Implementations: `learned_selector_power.best_single` (legacy baseline, retained only for this proof) vs
`learned_selector_power.best_single_legal` (the preregistered rule).

## Frozen replay grid
```
test geometry = ±0.035
block designs = 9/6/9, 12/9/12, 18/9/18
primary tau   = 0.0342, 0.03425, 0.0343
replicate seeds = 500 per (design, tau)   → 9 configs × 500 = 4500 replicates
```

## Result
| quantity | value |
|---|---|
| total replicates checked | **4500** |
| step-1 tie count | **0** |
| old/new selected-offset mismatch count | **0** |
| selected-offset distribution | **offset 0 : 4500 / 4500** (−0.04:0, +0.04:0) |
| all replicates select offset 0 | **True** |
| `power_recertification_required` | **False** |

Per-config: every one of the 9 (design × τ) cells has `step1_ties = 0` and `mismatch = 0`
(`best_single_tiebreak_invariance_v1.json` → `configs`).

## Reading
- **Step 1 is always strictly unique.** In 4500/4500 replicates a single offset (always 0) maximizes the
  observed train/val mean success by a margin, so the step-2 tie-break **never fires**. The secret
  `τ−|eff|` term was therefore **dead code** — it never influenced a single selection.
- **Old and new rules are identical on the entire frozen grid** (0 mismatches). Replacing the illegal
  tie-break with the legal `min |offset| → fixed bank order` leaves best-single = 0 everywhere.
- Because best-single is unchanged, the primary contrast `B2_K1 − best-single`, the effect size, and the
  power result are all unchanged. This matches Claude C's independent finding and the pre-registered
  `power_recertification_required = false`.

## Gate
```
step1_tie_count = 0            ✓
selection_mismatch_count = 0   ✓
→ fix accepted with NO power recertification
```
Had either count been non-zero, this phase would have stopped and reported `POWER_RECERTIFICATION_REQUIRED`
rather than claiming the fix complete. Neither occurred.
