# Preregistration v4 — independent adversarial pre-run audit (Claude C)

**Auditor:** Claude C, independent, **read-only, adversarial, pre-run**. No Isaac launch, no generator,
no manifest instance, no confirmatory data, no run authorization. Only three files added (this report,
its JSON, and `test_preregistration_v4_audit_c.py`).

**Audited commit:** `9a02eb5ae320564bc8710380b919101d6537c02a`
(branch `experiment/offline-calibration-preregistration-v4`).
**Power certification:** `2bf7217907f24ae54a08db71bbdcf624b110ccf0` (`test-geometry power +-0.035`).

## VERDICT: **MODIFY_PREREGISTRATION_V4**

Four **fixable** blockers; none is a scientific / power / leakage dead end. The core science (genuine
non-degenerate structural + **learned** decision value at ±0.035), the leakage guards, the six
instrumentation fields, the model/loss/feature code, and the primary statistics are all **sound and
verified first-hand**. **Power re-certification is NOT required** for any of the four fixes.

`authorizes_generator_implementation = false` (GO is withheld pending the four fixes).

---

## What v4 got right (verified first-hand, not from B's summary)

- **Responds correctly to the v3 audit.** v3's degenerate-CI risk (residual inert at operating points) is
  **resolved by moving test to ±0.035**: I recomputed from the 306-ep band-edge data (sha
  `131750e158…`, matches) that the success edge is a sharp step at ≈0.033 (1.0 for |eff|≤0.030 → 0.37 at
  0.035 → 0.0 at ≥0.040), that the **state-aware region |eff|∈[0,0.015] is deterministic success (1.0)**,
  and that the **best-single offset-0 at ±0.035 (|eff|∈[0.025,0.045]) is MIXED (0.53)** and mixes across
  the 18 blocks — so the block-bootstrap CI is **non-degenerate**. The residual now genuinely flips labels
  at the operating offsets.
- **Instrumentation blocker (v3) resolved.** All six failure-mechanism fields are present in 306/306
  (`minimum_joint_limit_margin`, `ik_failure_count`, **both** `joint_limit_clamp_count` and
  `joint_step_clamp_count` separated, `failure_phase`, `true_handle_error_at_close`,
  `gripper_width_at_close`), and the ContactSensor is available on 306/306 with **max unintended force
  0.0 N** (positive control ~240 N).
- **Power is the real LEARNED model, not oracle.** The certifying study (`test_geometry_power.py`) trains
  the repository `models_v2.DeepSets` (K1 vs K0, 5 seeds) and reports **learned power 0.984–0.986
  separately** from structural 1.000. It uses the same bank/probe/residual/blocks/HP/seeds/best-single as
  v4. The earlier ±0.03 study honestly returned `REDESIGN_MODEL_OR_CLAIM` (even the oracle failed there);
  the fix is entirely in the test geometry, not the threshold or the model.
- **Feature allowlist == real code.** `features.PROBE_KEYS` (8, in order) == history allowlist;
  `static_features` (8) == static allowlist; no secret field is reachable. Verified by importing the code.
- **Model/loss == real code.** `_loss = bce + mse_e + mse_t` — **BCE + MSE + MSE, weight 1 each** (the
  “2×MSE vs MSE+MSE” question resolves to MSE+MSE; **no mismatch**). Full-batch Adam(weight_decay=l2),
  z-normalized regression targets, patience early-stop with best-state restore, per-seed determinism.
- **Design & claim scope.** Bank {−0.04,0,+0.04}, nominals train/val/test, blocks 9/6/9, 75 sessions,
  **300** trials (recomputed), probe≠candidate (distinct trials, outcome not reused), task-level reset,
  block-constant hidden state, split-independent blocks, **test actual support ⊂ train actual support**,
  no robust single action, claim scope bounded to *unseen nominal conditions, in-support actual state*.
- B's suite: **147 passed**. My independent suite: **12 passed, 4 xfailed** (the four blockers).

---

## Blockers (must fix before GO)

### A. `BLOCKER_SECRET_TIEBREAK` — best-single reads the secret hidden state
`best_single` rule step (2) = "tie → max mean frozen continuous margin **τ − |eff|**", and
`eff = actual_bias + offset`; `actual_bias` is in the `secret_denylist`. An oracle/secret quantity must
not participate in the state-agnostic best-single baseline (the very thing B2 must beat). Present in the
frozen rule (`confirmatory_v4_config.json` `best_single.rule[1]`, prereg §7, analysis-plan §2) **and** in
the power code (`learned_selector_power.py:99-111`, `margins = tau − abs(nominal+residual+offset)`).

- **Independent finding:** best-single = offset **0.0 in 200/200** replicates; step-1 (max mean success)
  ties in **0/200** → the secret margin is **never consulted (dead code)**.
- **Minimal fix:** replace step (2) with an **observable** train/val tie-break (min mean
  `task_outcome_error`, or min mean `|final_joint_position − target|`, or min mean `skill_elapsed_time`),
  **or delete it** (step1 → min|offset| → numeric order). Remove the secret margin from the power code too.
- **Power recert:** **not needed** — step-1 is always unique, so the fix provably cannot change any
  selection; prove from the saved best-single logs / the 200/200 check.

### B. `BLOCKER_SEEDS_NOT_FROZEN` — randomization seeds are not pinned
Only `model_seeds {1103,2207,3301,4409,5519}` are frozen. `master_seed` (and the residual / block-order /
session-order / nominal-order / candidate-order / **bootstrap** seeds derived from it) have **no frozen
value and no derivation rule**; `manifest_spec §3` has A *"Fix master_seed"* at construction — a run-time
choice. Listing field **names** in `manifest_fields` is not a freeze.

- **Minimal fix:** freeze an exact integer `master_seed` and `bootstrap_seed` in `confirmatory_v4_config
  .json` (sub-seeds already derive deterministically), **or** a rule such as
  `master_seed = int(sha256(config_hash)[:8], 16)`. The bootstrap seed must be frozen before unseal.

### C. `BLOCKER_SEAL_SCHEME_NOT_UNIQUE` — the seal step is an either/or
Seal STEP 2 = "test is **either (a)** not yet generated, **or (b)** generated into a sealed store"
(`seal_unseal_protocol.md`, `preregistration_v4.json seal_unseal[1]`, prereg §10). A preregistration must
freeze exactly one executable scheme so no run-time discretion exists over when/how test outcomes come
into being.

- **Minimal fix:** pick **one** — recommended **Scheme 2** (freeze+hash the test *manifest* now; generate
  test *trials* only after the model + analysis hashes are frozen, from that frozen manifest, with no test
  seed/manifest re-selection). Define who may read secret/test, prove the feature cache holds no test
  outcome, and state that any accidental early unseal makes the experiment **INVALID**.

### D. `BLOCKER_MODEL_FAILURE_SEMANTICS` — model-training failure conflated with trial-invalid
A model-training exception is labelled a "**TECHNICAL-INVALID event (§17)**", mixing an analysis-stage
model-fit failure into the **300-runtime-trial** invalid budget (max 15). There is **no rule that all 5
model seeds must converge and freeze a hash**, and the seed-stability gate "≥4/5 seeds ≥0.15" could
silently **PASS with a missing (failed) seed** — dropping a seed.

- **Minimal fix:** separate model-training-failure handling from the 300-trial technical-invalid count.
  Freeze: all 5 seeds MUST converge with a frozen `state_dict` hash; a training exception → deterministic
  **same-seed** retry (fixed max retries), **no seed replacement**; if any seed still fails, the PRIMARY
  analysis is **INVALID** (not a 4/5 pass). State the seed-gate denominator is fixed at 5 (a missing seed
  is a failed seed, never an excluded one).

---

## Minor issues (not blockers)

1. **Stale band-edge exit verdict.** `band_edge_exit_verdict_v1.json` still says
   `MODIFY_RESIDUAL_OR_DESIGN` / `residual_induces_block_label_variation=false` — that was the **±0.03**
   analysis. ±0.035 resolves it (verified above from the 306 data). Add an explicit ±0.035 operating-point
   residual-variation confirmation and supersede the stale verdict so the evidence chain is unambiguous
   (data already exist in the 306 §8.1 table; documentation only).
2. **τ sensitivity band narrow.** Power τ ∈ {0.0342,0.03425,0.0343} covers only the logistic neighbourhood,
   not the isotonic crossing 0.0325. The untested lower τ makes the effect **larger** (best-single fails
   harder) → conservative, not threatening. Report power at τ=0.0325 for completeness.
3. **`probe_vector` silent zero-fill** (`h.get(field, 0.0)`) — a missing required probe field becomes 0.0.
   Have the generator assert the 8 probe-allowlist fields present (schema-completeness is already a
   technical-invalid condition); add a guard test.
4. **DeepSets `__init__` defaults** (max_epochs=300, patience=30) differ from the frozen 200/25; both power
   and confirmatory pass explicit values so there is no practical mismatch — align the defaults (cosmetic).

---

## Answers to the audit questions
| area | verdict |
|---|---|
| overall design | **PASS** (recomputed: 300 trials, test⊂train actual support, no robust action) |
| power ↔ v4 consistency | **PASS** — certifying study trains the real DeepSets at ±0.035; **learned** power 0.986 reported separately from structural 1.000; params match |
| best-single leakage | **BLOCKER_SECRET_TIEBREAK** (dead code; fix needs no recert) |
| seed completeness | **BLOCKER_SEEDS_NOT_FROZEN** (only model seeds frozen) |
| seal/unseal | **BLOCKER_SEAL_SCHEME_NOT_UNIQUE** (either/or) |
| model / training consistency | **PASS** (loss = BCE+MSE+MSE, matches code) |
| technical-invalid / retry | **PASS** on trials; **BLOCKER_MODEL_FAILURE_SEMANTICS** on model-fit |
| collision | **PASS (minor)** — strict >0 N supportable; secondary/intention-to-analyze |
| feature leakage | **PASS** — allowlists == real code; no secret reachable |
| instrumentation | **PASS** — six fields present 306/306; v3 blocker resolved |
| primary statistics | **PASS** — block unit, 2-nominal→5-seed, 2000 block bootstrap, CI≥0.15, seed gate |
| manifest implementability | **MOSTLY PASS** — one gap = unfrozen master/bootstrap seed (Blocker B) |
| claim scope | **PASS** — bounded; over-claims disclaimed |

## Authorization
- `authorizes_generator_implementation`: **false**
- `authorizes_confirmatory_data_generation`: **false**
- `authorizes_confirmatory_run`: **false**
- `power_recertification_required`: **false**

After B fixes the four blockers (all local, non-design), a **re-audit by Claude C** is required before GO.
Because none of the fixes touches the estimator, model, geometry, or effect, the re-audit is expected to
be short and to reuse the existing ±0.035 power certification.
