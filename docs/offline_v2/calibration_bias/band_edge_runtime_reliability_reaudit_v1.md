# Band-edge runtime reliability re-audit v1 (pre-run, read-only)

Claude B, branch `experiment/offline-calibration-band-edge-authorization-v1`. Independent, first-hand
re-audit of Claude A's reliability fixes (`experiment/runtime-calibration-band-edge-reliability-v1` @
`e447f00`) against my prior `MODIFY_RUNTIME` verdict (`630d81c`). **No Isaac launched, no new runtime data
generated, no 306 run, no A files modified.** Every result below is recomputed / re-run from the code and
smoke artifacts — not taken from A's summary.

## VERDICT: **AUTHORIZE_306_EXPLORATORY_RUN**

All four prior blockers (resume, fail-fast, dirty-source refusal, run-failing hard self-check) and the
pytest-collection minor are **fixed and independently verified**. The fixes touch only the run harness; the
frozen science is unchanged (plan diff = a single pure-identity field). No leakage, no data-contract or
frozen-plan violation → not `MODIFY_RUNTIME`, not `STOP`.

Authorizes exactly **one** full 306-episode **exploratory** band-edge characterization run from the clean
source commit `e447f00` using the frozen manifest/config; resume within the same run dir + same manifest is
allowed. See `band_edge_306_run_authorization_v1.json`.

---

## §3 Persistence — **PASS**
`_run_isaac` builds each record in memory → `validate_record_full` → (on pass) `atomic_write_record`
(`write_json_atomic`: `records/<id>.json.tmp` → `flush` → `os.fsync` → `os.replace` atomic rename → dir
`fsync`). Verified:
- an invalid record is **never** committed (validation precedes persist; §5);
- `*.tmp` is never a committed record (`committed_records` reads only `*.json`);
- `episodes.jsonl` is a **derived view** rebuilt in manifest planned order from committed records
  (`rebuild_episodes_jsonl`);
- duplicate committed `(block,offset)` / id is rejected (resume `load_and_verify_completed` + `hard_self_check`);
- a **parseable-but-inconsistent** committed record → `run_status=INVALID`, non-zero exit (verified via
  `load_and_verify_completed` raising on any manifest mismatch).
- **Minor (transparent, non-blocking):** an *unparseable* committed `*.json` is treated as absent and the
  episode is re-executed deterministically (same residual from the frozen manifest, then `os.replace`
  overwrites the corrupt file) rather than marking the run `INVALID`. Because atomic rename makes a
  half-written committed file impossible and the re-execution is deterministic + no-resample, this is safe;
  it self-heals a corrupted file instead of aborting. Noted, not a blocker.

## §4 Resume / skip-completed — **PASS** (smoke independently re-verified)
Manifest-first identity freezes `run_id`, `config_sha256`, `science_manifest_sha256`, `source_commit`,
`master_seed`, planned `(block,offset)` + `planned_episode_id`s, output dir. On resume the existing run
manifest's science/config hash must equal the frozen plan's, else `INVALID`+non-zero. Each committed record
is re-validated (`validate_existing_record`) for: JSON parse, `smoke_only` == run mode, `planned_episode_id`
∈ manifest, `(block,offset)` ∈ manifest, uniqueness, `config_sha256` + `science_manifest_sha256` match,
`secret` nominal/residual/actual == manifest block, `nuisance_robot_joint_delta` + `nuisance_target_jitter`
== manifest, and schema+leakage. Completed episodes are **skipped, never re-executed, never resampled**;
only `remaining` run; a `resume_log.jsonl` records time/commit/hash/completed/remaining/status.

Independently re-ran `load_and_verify_completed` + `hard_self_check` on the resume-smoke dir → both OK, 6
records valid, all `smoke_only`. The `resume_smoke_result.json` matches:
`first_pass_completed=3` (IN_PROGRESS), `resumed_skipped=3`, `resumed_executed=3`, `total_unique=6`,
`duplicate=0`, `science_manifest_sha256` **identical across passes** (`24fced22…`, which I recomputed),
`nuisance_equality_verified`, final `COMPLETE`, `self_check_ok`.

## §5 Invalid-record fail-fast — **PASS** (forced-invalid smoke independently confirmed)
Code path: build → `validate_record_full` → on FAIL: **not committed**, `errors/fail_<id>.json` saved,
`run_status=FAILED`, loop **stops**, process exits **non-zero** (`return 4`). `--inject_invalid_record_at`
and `--stop_after_new_episodes` and `--allow_dirty` are **refused on the full run** (`return 2`).
Independently confirmed on `band_edge_invalid_smoke`: injected `b01_o-0.050`
(`minimum_joint_limit_margin_rad=-1.0`) → `invalid_record_written=false` (only `b01_o+0.050` committed; the
injected id is **absent** from `records/`), `prior_valid_retained`, `subsequent_episodes_executed=false`,
`run_status=FAILED`, error artifact present, non-zero exit; the recorded validation error is B's own schema
`>=0` check catching `-1.0`.

## §6 Dirty-source policy — **PASS**
`assert_clean_source_tree_for_full_run` runs `git status --porcelain`; the **full** run refuses unless the
tree is clean, with only the run's own output dir and `deployment_calibration/data/` allowed to differ
(renames handled). Independently exercised `_classify_porcelain`: a frozen-config edit → **offending**
(refused); a source-code edit → **offending** (refused); only-output change → **clean**. No `--allow-dirty`
bypass exists for full (it is a forbidden test-only flag on the full path). Smoke may run dirty (records
`smoke_only=true` + `dirty_worktree`); `run_metadata.source_commit` = actual HEAD.

## §7 Hard self-check — **PASS**
`hard_self_check` verifies (full): count==306, 18 blocks, 17 offsets each, offsets == frozen grid, no
duplicate id/(block,offset), no probe, run-consistent smoke flag, all planned IDs present + none unplanned,
per-record schema+leakage, matched-block, config + `science_manifest_sha256` consistency, instrumentation
required fields + `contact_sensor_available` present. **The self-check controls the exit status**: any
failure → `run_status=INVALID` + `return 5`; all-pass → `COMPLETE` + `return 0`. Fault cases
(hash mismatch, duplicate, unplanned, matched-block, invalid schema, etc.) are covered by
`test_band_edge_reliability_v1.test_hard_self_check_clean_and_faults` (passing).

## §8 Pytest collection — **PASS**
`pytest --collect-only` collects **6 `def test_*`** across the three runtime files
(`test_band_edge_plan_and_validators`, `test_band_edge_instrumentation_cores`, and 4 reliability tests:
hard-self-check, dirty-source classification, atomic-persist+resume-verify, science-manifest-SHA stability)
— none rely on `if __name__ == "__main__"`. Full suite: **99 tests collected**, matching A's "99 passed";
I re-ran the 6 runtime tests (**6 passed**) and B's frozen `offline_v2/calibration_bias/` (**59 passed**) —
no regression, no unexplained skip/xfail.

## §9 Science freeze unchanged — **PASS**
`git diff` of `band_edge_plan_v1.py` (pre-reliability → reliability) is a **single line**: adding
`planned_episode_id = f"b{block:02d}_{offset_id}"` (run-independent, deterministic — a pure identity, no
science). Independently rebuilt the full plan: 306 = 18×17, no probes, nominal 0, offsets == frozen grid,
**residuals == `draw_block_residuals(18, residual_seed, ResidualNuisanceConfig(0.005,[-0.01,0.01]))`**,
planned `(block,offset)` order deterministic from the unchanged `master_seed=8801150`. New reliability code
(persistence/resume/self-check) and test hooks do not touch the science path. Full-plan
`science_manifest_sha256` = `e2c8216ed2b2bae19d03815ed922eaf49a6870afce72d7d3fb544e98cd7c86ba` (stable: the
hash excludes volatile git/run provenance).

## §10 Pre-authorization checks — **PASS**
The full run is enforced to: fixed HEAD (`e447f00`) + **clean** worktree (dirty refusal), frozen manifest
(`build_plan` deterministic), `smoke_only=false` (mode `full`), **test-only params forbidden**, 0 probes
(`run_probes=False` + self-check), a new output dir, no mixing of a different manifest (resume verifies the
science/config hash else `INVALID`), metadata with `source_commit` + `config_sha256` +
`science_manifest_sha256`, `contact_sensor_available` required per record + in self-check, and **no in-run
adjustment** hook for residual/offset/threshold/block-count/stop-rule/success-definition. The **collision
threshold is intentionally NOT frozen**: the 306 run records only **raw** contact quantities; B selects the
threshold post-run by the pre-frozen rule (never from confirmatory data) and reports full +
collision-sensitivity — A does no in-run classification.

---

## Authorized scope (and only this)
**One** full 306-episode **exploratory** band-edge characterization run, from the frozen manifest/config and
the clean source commit `e447f00`; **resume allowed** within the same run dir + same manifest.

## Still forbidden
Changing the science plan; a second 306 run with a different seed; confirmatory generator or data;
learned-selector power; preregistration v4; the 675-episode confirmatory run; any mid-run parameter/threshold
/stop-rule tuning. A GO on any of these requires a separate, explicit authorization.
