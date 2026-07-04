# Learned-selector power protocol v1 (offline, FROZEN before the Monte Carlo)

Claude B, branch `experiment/offline-calibration-learned-selector-power-v1`. Offline synthetic power
analysis for the selected **Design A** candidate bank. **No Isaac, no runtime episode, no confirmatory
data, no preregistration v4.** This freezes the method BEFORE the Monte Carlo; nothing below is chosen from
the power results. Machine form: `learned_selector_power_config_v1.json`.

## Question (learned, not oracle)
> Under finite train/val blocks and finite model seeds, with the fixed K=1 probe, can the REAL
> capacity-matched DeepSets B2 learn to select the compensating action and beat the train/val-only
> best-single with enough power? — reported **separately** from structural/oracle power.

## Systems
- **B0** base-rate.
- **B1 / K=0** static capacity-matched DeepSets, **empty history**. (Because x/g are identical across
  sessions and only the offset varies, K=0 structurally reduces to a fixed offset = best-single.)
- **B2 / K=1** the SAME DeepSets backbone + the fixed first probe (offset −0.04) outcome as history.
  Uses the repository's real `models_v2.DeepSets` history encoder — **no** hand sign-classifier, oracle
  label, or lookup table.
- **Oracle** state-aware (knows actual_bias) — structural upper bound only; never trains/selects for B2.
- **Best-single** chosen on **train+val only** (max mean success; tie → max mean margin; tie → min |offset|);
  test never participates.

## Leakage constraints (enforced + tested)
Legal model inputs: `x`, `g`, `theta`(offset), `H_K1`(probe outcome). The model may **not** read nominal /
residual / actual bias, `eff_signed`, `abs_eff`, block/residual/seed ids, future candidate outcomes, test
split identity, the oracle action, or any hidden-state class label. The only K=1-vs-K=0 difference is the
frozen probe history. Probe-history fields are limited to the probe's offset + outcome + task_error + time
+ pull-duration + final position (the frozen schema); no field added after seeing power.

## Synthetic outcome model (empirical HARD edge — no fabricated soft scale)
`actual = nominal + residual`; `success = 1[|actual + offset| ≤ τ]`. Primary τ = 0.03425; sensitivity
τ ∈ {0.0342, 0.03425, 0.0343} (all three required); τ = 0.0325 (isotonic) reported as sensitivity only.
Residual ~ TruncNormal(0, 0.005, [−0.01, +0.01]), one per block, shared across the block's nominals /
probe / candidates; blocks independent; disjoint across train/val/test within a replicate. No artificial
label/probe/soft-edge noise is added.

## Candidate bank & probe (frozen)
Bank **{−0.04, 0, +0.04}** (Design A). Fixed K=1 first probe offset **−0.04** (a repeated-task deployment
diagnostic **full-task trial**, not a cost-free internal query). K=0 exposes no probe history.

## Training budget (FROZEN; not tuned on test)
`models_v2.DeepSets(d_hid=32, d_emb=16, lr=1e-2, max_epochs=200, patience=25, l2=1e-4)`, Adam, early
stopping on the validation multitask loss, deterministic per seed. **5 model seeds** {1103, 2207, 3301,
4409, 5519}, all reported (never seed-cherry-picked). Selection = argmax predicted `p_success` per session.

## Split & block-count grid
Primary {train 9, val 6, test 9}. Escalation grid (frozen): {9/6/9, 12/6/12, 12/9/12, 15/9/15, 18/9/18}.
Selection rule (frozen): the **smallest total-block** design meeting the power bar at **all 3 primary τ**;
ties → larger test, then larger train. Blocks/seeds/residuals independent; a block never crosses a split
within a replicate; test used only for final evaluation.

## Estimators & power event
- per test block: avg over the block's test nominals of the **seed-averaged** B2 selected success.
- `Delta_success = mean_test_blocks( seed-avg B2 selected success − best-single success )`.
- primary 95% CI: **block bootstrap over test nuisance blocks** (n_boot 2000).
- **power event** (all must hold): `CI_lower(Delta) ≥ 0.15` AND mean Delta > 0 AND oracle success ≥ 0.85
  AND best-single non-degenerate AND **seed stability** (≥ 4/5 seed point gains ≥ 0.15 and none < 0).
- power = fraction of Monte-Carlo replicates whose event holds. **≥ 200** replicates (300 primary / 200
  grid); a non-converged replicate counts **conservatively as a power failure**.
- acceptance: estimated learned power **≥ 0.85** AND its 95% Monte-Carlo binomial CI lower ≥ 0.80.

## Structural vs learned (reported separately)
For every design/τ: structural/oracle gain + power (oracle vs best-single, no training) AND learned B2 gain
+ power AND K=0 power AND the K1−K0 history increment. If structural power is high but learned power is low,
the verdict is model/sample-size insufficiency — **oracle power is never substituted for learned power.**

## Probe cost / claim scope (Claim S vs Claim V)
- **Claim S (success decision value):** history-conditioned selection improves task success — the primary
  learned-power target.
- **Claim V (net deployment value):** benefit after probe cost. One-shot `Net VOI = gross success VOI −
  λ_time·probe_time` is reported honestly (expected negative for a one-shot full-task probe), plus
  `NetVOI_horizon(T) = T·per-task gain − one-time probe cost` for T ∈ {1,2,5,10}. **T must come from
  deployment semantics, not the power results.** The old unconditional "Net VOI(K1) > 0" GO gate is not
  reused; v4 must choose Option 1 (success-primary) or Option 2 (repeated-deployment) — see
  `probe_cost_and_claim_scope_v1.md`.

## Exit verdicts (this phase; no v4 issued)
`POWER_SUFFICIENT_FOR_PREREG_V4` (9/6/9 passes all bars) / `INCREASE_BLOCK_COUNTS` (a larger grid design
passes) / `REDESIGN_MODEL_OR_CLAIM` (no reasonable grid clears it, or K1 ≯ K0) / `STOP_CURRENT_CONFIRMATORY_CONCEPT`.
