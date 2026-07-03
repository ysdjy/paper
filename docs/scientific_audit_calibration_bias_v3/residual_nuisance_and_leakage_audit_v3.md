# Residual-nuisance & leakage audit (Audit B)

**Auditor:** Claude C. Verifies the residual implementation vs the docs and audits every leakage path.
Note: the **confirmatory generator does not exist yet** — this audits the residual/leakage *spec*
(`residual_nuisance.py`, `schema.py`, `validator.py`, `splits.py`) and the exploratory generator/data.

## 1. Residual implementation vs documentation — **consistent**
`residual_nuisance.py` matches the freeze proposal exactly:
- `residual ~ TruncatedNormal(mean=0, σ=0.005, [−0.01, +0.01])`; effective SD computed analytically
  = 0.004398 (matches `confirmatory_design_validation_v3.json` `residual_effective_sd`). ✓
- **One residual per block**, drawn by a seeded `np.random.default_rng(seed)`; deterministic. ✓
- `actual_bias = nominal + residual`; rides the **same handle-Y axis** as the bias, so it enters
  `|actual_bias + offset|` (the success quantity) — verified against `handle_calibration_bias_v1.py`
  (`commanded = true + bias + offset`). The residual is thus **physically implementable** via the same
  `biased_spec(spec, actual_bias)` path the exploratory injection already uses. ✓

## 2. Secret/audit isolation of state fields — **enforced**
- `RESIDUAL_SECRET_KEYS = (residual_bias_y, actual_bias_y, nominal_bias_y)` — all audit-only.
- `schema.FORBIDDEN_X_SUBSTRINGS` includes `bias, secret, hidden, calib, ground_truth, effective_error,
  true_offset` → every residual/actual/nominal key is barred from `x`.
- Two independent guards: `assert_residual_not_in_x` and `assert_no_bias_in_x`.
- **First-hand check on the 135-ep data:** `x` keys = {drawer_name, gripper_width,
  initial_mechanism_joint_pos, mechanism_id, member, robot_joint_pos, tcp_pos, tcp_quat} — **no bias /
  secret / nuisance key present.** `hidden_state_id`, `secret_deployment_state`, `block_id`, `block_seed`
  are top-level provenance, **not** inside `x` and not in the model-legal input set.

## 3. Block-reuse across bias — **correct (and correctly certified healthy)**
The same residual/block is reused across the split's bias levels (matched context). `validator
.nuisance_bias_independence` **accepts** paired reuse-across-bias while **rejecting** a 1:1 seed↔bias
permutation. This is the intended matched-block design (a block gives a common nuisance context under
every bias), not a bias-encoding leak. Verified on exploratory data: one block → identical `x`/`g` across
all 5 biases; poison tests for the 1:1 map pass.

## 4. Split isolation (block, seed, residual) — **enforced by `audit_split`**
`splits.audit_split` requires: bias levels pairwise-disjoint + partition; **sessions disjoint**; **nuisance
seeds disjoint**; **nuisance blocks disjoint across train/val/test** (blocks may still pair *within* a
split). Since each block carries a distinct seed→distinct residual draw, train/val/test residuals are
disjoint by construction. This is the correct isolation for the confirmatory design.

## 5. Indirect leakage paths — audited
| path | verdict | note |
|---|---|---|
| bias/actual/nominal → `x` | **blocked** | whitelist + two guards + first-hand data check |
| raw nuisance **seed/block id** → model feature | **blocked** | `validator` bars `nuisance_seed`/`nuisance_block_id` from `x`; they are provenance |
| robot init state → bias | **no leak** | `nuisance_robot_joint_delta` perturbs initial joints but is drawn from a **bias-independent** master RNG; it can appear in `x.robot_joint_pos` yet carries no bias signal |
| target jitter → bias | **no leak** | `target_jitter` bias-independent |
| execution order → bias | **no leak** | sessions executed in randomized order; `execution_position` is provenance, and order is independent of bias |
| candidate group varies only θ | **holds** | matched offset bank: identical offset-id set + identical θ per offset across bias; single target (verified `matched_offset_bank`) |
| matched block varies only hidden state | **holds** | within a matched block, `x`/`g` identical across bias; only `actual_bias` (state) differs |
| **`OBSERVABLE_NUISANCE_KEYS` → model** | **LATENT RISK** | schema declares `observed_init_offset / observed_friction_proxy / observed_sensor_noise_level` **model-legal**. They are **absent** from the exploratory data (no such fields), so no current leak — but if the confirmatory generator emits them, any correlation with `actual_bias` would let a model recover the hidden state *without probes*, bypassing the probe-history design. **Must be gated.** |

## 6. Verdict
**No leakage in the spec or the exploratory data.** The residual is faithfully implemented, secret-only,
correctly reused across bias, and split-isolated. Two forward-looking requirements for the (not-yet-built)
confirmatory generator:
1. **Re-run `validator.validate` + `assert_no_bias_in_x` + `assert_residual_not_in_x` on the confirmatory
   episodes** and gate the run on PASS (the schema is a spec; only the real generator's output can be
   certified).
2. **Either drop `OBSERVABLE_NUISANCE_KEYS` for the confirmatory run, or prove each observable feature is
   statistically independent of `actual_bias`** before exposing it to a model (add a leakage test).
