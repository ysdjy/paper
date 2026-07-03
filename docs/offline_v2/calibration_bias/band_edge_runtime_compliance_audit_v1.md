# Band-edge runtime compliance audit v1 (pre-run, read-only)

Claude B, branch `experiment/offline-calibration-band-edge-pre-run-audit-v1`. Independent, first-hand
pre-run implementation-compliance audit of Claude A's band-edge runtime
(`experiment/runtime-calibration-band-edge-v1` @ `2ac0c9f`). **No Isaac launched, no new runtime data
generated, no 306 run, no A files modified.** Decides whether to authorize the full **306-episode
exploratory band-edge characterization**. Every claim below is recomputed from the code / manifest /
smoke artifacts — not taken from A's summary.

## VERDICT: **MODIFY_RUNTIME**

The **scientific core is fully compliant and high quality** — the frozen plan, residual, six
instrumentation groups, collision sensor, leakage/matched-block guards, and both smokes all pass
independent checks. The **only** blocker is the **306-run execution harness reliability**: the full-run
path in `generate_calibration_bias_band_edge_v1._run_isaac` lacks **resume**, **fail-fast on invalid
records**, **dirty-worktree refusal**, and a **run-failing self-check**. These are small, localized,
science-untouched fixes that must land and be re-verified before a ~3-hour formal run is authorized.

Not AUTHORIZE (the task's authorization requires "resume / manifest / uniqueness 机制可靠"; resume is
absent). Not MODIFY_PROTOCOL (the frozen protocol IS implementable and correctly implemented; nothing in
the design must change — joint-margin and collision are handled additively/concretely).

---

## §3 Plan & config consistency — PASS (independently rebuilt)
Rebuilt the full plan from the frozen `PlanConfig(master_seed=8801150)` and compared to the committed
`band_edge_manifest_v1/manifest.json`:

| # | check | result |
|---|---|---|
| 1 | config SHA256 == frozen | **PASS** `2f20429b5273cb7d269ae0e78f6361e98d2d7cd138b55324549b36a98a2dac86` |
| 2 | 18 blocks | **PASS** |
| 3 | 17 offsets per block | **PASS** (each block exactly 17) |
| 4 | 306 unique episodes | **PASS** (`len(set(block,offset)) == 306`) |
| 5 | 0 probes | **PASS** (`run_probes=False`, `assert_no_probes` ok) |
| 6 | nominal bias == 0 | **PASS** (all blocks) |
| 7 | offsets == frozen `OFFSET_GRID` value-for-value | **PASS** |
| 8 | no extra hidden offset/bias correction | **PASS** (`actual = nominal + residual` only; no second term) |
| 9 | block + within-block offset order frozen in manifest | **PASS** (randomized orders reproduced from master seed) |
| 10 | resume does not resample residual/nuisance | **PASS (plan)** — `load_or_build` loads existing manifest; a fresh build is deterministic from `master_seed` (residuals + block order reproduce byte-for-byte) |
| 11 | duplicate episode rejected | **PASS** (`reject_duplicate_episodes` on id + (block,offset)) |
| 12 | formal vs smoke separate dir + role marker | **PASS** (`smoke_only` flag, `run_id` dir, `exclude_smoke`) |

The committed 306-manifest's residuals, randomized block order, and planned episodes match my rebuild
exactly. Operating region is covered: near `|eff|` 0.025 / 0.030 / 0.035 the plan has 86 / 85 / 76
episodes (the residual continuously fills the edge between the offset multiples).

## §4 Residual consistency — PASS
Runtime **reuses B's frozen implementation verbatim** — no second algorithm:
- `residual_nuisance.draw_block_residuals(18, residual_seed, ResidualNuisanceConfig(0.005,[-0.01,0.01]))`;
  I recomputed `draw_block_residuals(...)` and it equals the manifest's `residual_bias_y` per block.
- one residual per block, **shared by all 17 offsets** (matched-block check); initial joint perturbation,
  target jitter, initial x, mechanism, target all shared within block; blocks independent.
- `actual_bias_y = nominal(0) + residual_bias_y` (verified per block); nominal/residual/actual only in
  `secret_deployment_state` (never in x/g/theta/H — see §9). **PASS.**

## §5 Joint-limit margin semantic verdict — **ACCEPT_ADDITIVE_SIGNED_DIAGNOSTIC**
### 5.1 No information lost — PASS
All three fields co-exist on every record (verified on the 6 smoke records):
`minimum_joint_limit_margin_rad` (frozen, clamped ≥0), `minimum_joint_limit_margin_rad_signed` (raw
signed, e.g. −1.4e−05), `joint_limit_margin_negative_flag` (True when signed<0). The **signed value is
captured before clipping** in `JointMarginTracker` (`min_rad` tracks the raw min; `result()` clamps only
the frozen field and reports the raw + flag separately). No raw information is discarded.
### 5.2/5.3 Verdict
The soft-limit numerical excursion (~1e−5 rad) is a real physics/soft-limit artifact, transparently
handled: the frozen field satisfies its declared `[0, joint_range]` range and B's `validate_record`
`>= 0` check, while the raw excursion is preserved. This is **ACCEPT_ADDITIVE_SIGNED_DIAGNOSTIC**. B
publishes an **additive schema v1.1 clarification** (documenting the three field semantics) in
`band_edge_runtime_field_mapping_v1.md` — additive, pre-run, no runtime change required.

## §6 Six instrumentation groups — PASS (implemented in runtime, not only synthetic)
Verified in `band_edge_instrumentation_core_v1.py` + `_runtime_v1.py` + `band_edge_runtime_v1.py`, and
present on every one of the 6 smoke records:
- **5.1 joint margin** — per-step over 7 arm joints, episode-min, joint index, phase, normalized, signed. PASS.
- **5.2 IK** — `ik_solve/failure/max_consecutive/phase_counts` from `InstrumentedIKAdapter.solve().success`
  ONLY; a skill **timeout is NOT counted as an IK failure** (only `solve()` returning `success=False`). PASS.
- **5.3 clamps** — `joint_step_clamp_count` (step saturates `max_joint_step`) and `joint_limit_clamp_count`
  (`q_des` on a soft limit) are **separate**; `detect_clamps` can flag **both on one solve**. PASS.
- **5.4 failure phase** — `failure_phase` + `failure_transition` from `skill.runtime.history` last state
  before `FAILED`. PASS.
- **5.5 CLOSE→PULL snapshot** — captured at the single `CLOSE_GRIPPER→PULL` control step; uses the **TRUE
  unbiased handle** (`_true_handle_world` = link pose ⊕ `spec.handle_local_pos`, NOT the biased perceived
  handle); 3D + local-Y error, TCP pose, true-handle pose; **null + explicit `close_snapshot_reason`** when
  the transition is not reached (verified: `+0.05` fails pre-CLOSE → null+reason). PASS.
- **5.6 gripper at close** — width + command at the same step. PASS.
- **6 collision** — `CollisionMonitor` on `ContactSensor.net_forces_w`, **non-finger** links (arm+hand),
  finger↔handle grasp excluded; max force / frame count / first phase / by-phase / available / backend.
  Records net-force magnitude (sufficient to freeze a force threshold post-run); no per-pair detail (not
  required for a force threshold in this single-contactable-body scene). PASS.

## §7 Contact-sensor smoke — PASS (independently read)
`band_edge_contact_smoke_v1_20260704_000958/contact_smoke_result.json`: `sensor_available=true`; clean
baseline **0.000 N**; positive control (hand driven into cabinet) **240.084 N** (16 contact frames);
`force_gap 240.08 N`; `discriminates=true`. The positive control runs in a **smoke-only** path (phase
`"SMOKE"`), does not modify formal skill parameters, writes to its own smoke dir, carries
`smoke_only=true`, and is excluded from formal analysis. It validates the **detector only** and does
**not** freeze the formal collision threshold. PASS.

## §8 Clear-zone smoke — PASS
`band_edge_smoke_v1_20260704_001540`: 2 blocks × {−0.05, 0, +0.05} = **6 episodes**; offsets confirmed
`{-0.05, 0.0, 0.05}` only — the core edge points ±0.025/0.030/0.035 are **not** used and **no edge was
fit**; `smoke_only=true`; residual/offset/block count unchanged from the frozen plan (same master seed);
`matched_block_ok` (single actual_bias/block, initial-x spread 0, only offset varies); `exclude_smoke`
drops all 6. Physical sanity: offset 0 → success (close snapshot captured, TRUE-handle error ~0.005 m);
±0.05 → fail. Not used as a boundary conclusion. PASS.

## §9 Leakage & data contract — PASS (independently run on the 6 smoke records)
`validate_record` (B, frozen) and `validate_record_full` (schema+leakage) pass on all 6; **no** secret /
residual / actual / nominal key in x/g/theta/H; `OBSERVABLE_NUISANCE_KEYS` gate **disabled** (none may
enter a model channel); band-edge data has **no probes**; candidate outcome not in history (no history
here); execution order / seeds are in the manifest, **not** in model channels; the record validator is
**called on every episode** in the generator. PASS on data-contract correctness.
**BUT** the enforcement is not fail-fast on the formal path — see §10.

## §10 Run reliability — **FAIL (blockers for the 306 run)**
| protection | status |
|---|---|
| manifest-first | **PASS** (`build_plan` before any episode) |
| per-episode flush | **PASS** (`ep_f.flush()` per line) |
| unique episode id | **PASS** (`{run_id}_b{block}_{offset_id}`) |
| deterministic / no resample | **PASS** |
| metadata: commit / dirty / config hash | **PASS** (in manifest + per record) |
| **crash resume / skip-completed** | **FAIL** — `_run_isaac` opens `episodes.jsonl` with `"w"` (truncate) and loops **all** planned episodes; no existing-episode load, no skip, no resume-verify vs manifest. A crash in a ~2.75 h run loses all progress. |
| **fail-fast on invalid record** | **FAIL** — an invalid record is **printed and written** (`_validation_ok=False`), the loop **continues**; the frozen protocol requires "invalid record → immediate fail-fast, not written as formal data." |
| **dirty-worktree refusal for full** | **FAIL (soft)** — dirty is recorded but the full run is not refused/blocked when dirty; a formal run should refuse (or hard-mark) a dirty worktree. |
| **self-check fails the run** | **FAIL** — matched-block / duplicate / no-probes are computed at the end and written to summary, but **errors do not abort or fail** the run. |

Per the task ("如缺少关键恢复或完整性机制，裁决 MODIFY_RUNTIME"), the missing resume + fail-fast +
run-failing self-check are the decisive gap.

## §11 Freeze integrity for the 306 run — PASS (with §10 caveat)
Nothing that must stay frozen is changed by the runtime: 18 blocks, 17 offsets, residual σ/support, master
seed, nuisance pairing, skill params, damping (3.0), target (0.20), collision backend, instrumentation
fields, success definition — all consistent with the frozen config. The formal collision threshold is
**not** frozen (correct): the 306 run records **raw** collision quantities; B selects the threshold
post-run by the pre-frozen rule and reports full + collision-sensitivity. No mid-run threshold/stop
adjustment is possible in the harness (no such hook). PASS.

## Minor (non-blocking) notes
- A's two new tests (`test_band_edge_plan_v1`, `test_band_edge_instrumentation_core_v1`) are structured as
  `if __name__ == "__main__"` scripts, so `pytest deployment_calibration/tests/` collects **0** of them
  (they PASS when run as scripts). Recommend converting to `def test_*` so they run in the CI sweep and are
  covered by the "self-check" gate. (B's frozen `tests/offline_v2/calibration_bias/` regression: 59 passed.)

## Required runtime changes before AUTHORIZE (minimal, science-untouched)
1. **Resume**: on the full path, if `episodes.jsonl` exists, load completed `(block, offset)` keys, verify
   they match the manifest, open in **append** mode, and **skip** completed episodes (never re-execute,
   never resample).
2. **Fail-fast**: on the full path, a record failing `validate_record_full` must **abort the run**
   (non-zero exit) and must **not** be written as formal data.
3. **Dirty-worktree**: the full run must **refuse** (or require an explicit override flag) when the
   worktree is dirty; generate the formal run from a clean commit.
4. **Hard self-check**: after all episodes, a matched-block / duplicate / no-probes / schema failure must
   make the run **exit non-zero** and mark the run invalid.
5. (Minor) Make the two runtime tests pytest-discoverable.

After these land and a short re-verify (a fresh manifest + a 6-ep smoke that exercises the resume + a
forced-invalid-record fail-fast), B can re-audit and, if clean, issue `AUTHORIZE_306_EXPLORATORY_RUN`.

## Scope of any future authorization
Only the **306-episode exploratory band-edge characterization**. Never authorizes: confirmatory
generator, confirmatory data, learned-selector power, preregistration v4, or the 675-episode confirmatory
run.
