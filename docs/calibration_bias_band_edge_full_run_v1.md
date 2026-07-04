# Band-edge full 306 exploratory run v1

Provenance for the ONE authorized full 306-episode **exploratory** band-edge characterization run.
**Exploratory mechanism-characterization data — NOT confirmatory data.** No science analysis was performed
by Claude A; the frozen analysis is Claude B's read-only job.

## Authorization
- Authorization verdict: `AUTHORIZE_306_EXPLORATORY_RUN`
- Authorization audit commit: `0fbe68a295702df131af6c4187d5315d2a7f5447`
- Authorized runtime commit: `e447f00728b9b3dd26cbc85ecc3f0c2de17822f5`
- Authorized branch: `experiment/runtime-calibration-band-edge-reliability-v1`
- Authorized `config_sha256`: `2f20429b5273cb7d269ae0e78f6361e98d2d7cd138b55324549b36a98a2dac86`
- Authorized `science_manifest_sha256`: `e2c8216ed2b2bae19d03815ed922eaf49a6870afce72d7d3fb544e98cd7c86ba`
- Authorized: 306 episodes / 18 blocks / 17 offsets / master_seed 8801150 / run_count 1 / resume allowed.

## Source & branch
- Generation source commit: **`e447f00728b9b3dd26cbc85ecc3f0c2de17822f5`** (HEAD at run; tree clean; the
  authorization audit commit `0fbe68a…` was **not** merged).
- Branch / worktree: `experiment/runtime-calibration-band-edge-reliability-v1` /
  `/home1/banghai/Documents/IsaacLab/projects/paper_runtime_calibration_band_edge_reliability_v1`.

## Preflight (all PASS — `preflight.json`)
HEAD==`e447f00…`, tree clean, `config_sha256`==`2f20429b…`, `science_manifest_sha256`==`e2c8216e…`,
planned 306, blocks 18, offsets/block 17, no probes, nominal_bias 0, master_seed 8801150, smoke_only false,
no test-only flags.

## Exact command
```
./isaaclab.sh -p .../generate_calibration_bias_band_edge_v1.py --headless --mode full \
    --i_have_explicit_user_approval --config <launch_config.yaml> --run_id band_edge_full_306_v1
```
`--config` is an **out-of-repo** launch yaml (`launch_config.yaml`, copied into the run dir) providing only
the frozen defaults (`master_seed 8801150`, no subset keys) so the source tree stayed clean; the resulting
plan is byte-identical to the authorized plan (both hashes match; frozen `config_sha256` unchanged). No
test-only flags (`--stop_after_new_episodes` / `--inject_invalid_record_at` / `--allow_dirty`) were used;
the full path forbids them.

## Run
- run_id: `band_edge_full_306_v1`; dir:
  `deployment_calibration/data/band_edge_full_306_v1/`
- Start `2026-07-04 16:54:05` → end `2026-07-04 19:13:46`; **wall clock 139.6 min**.
- **Interruptions: 0; resumes: 0** (single uninterrupted run — `resume_log.jsonl`).
- Final `run_status = COMPLETE`; process exit 0; `self_check.json ok=true` (records 306 / planned 306 /
  is_full=true, errors=[]).

## §7 success verification (independently re-derived — all OK, `integrity_summary.json`)
`run_status=COMPLETE`, `self_check.ok`, episode count 306, unique 306, blocks 18, each block 17 offsets,
offset grid == frozen, no probes, no smoke records, no duplicate ids, no duplicate (block,offset), all
planned ids present, no unplanned ids, schema valid (all), leakage clean, matched-block valid,
`config_sha256` correct, `science_manifest_sha256` correct, instrumentation complete (0 missing fields),
contact fields present.

## §8 raw integrity counts (NO science analysis)
- 306 episodes: **211 success / 95 failure**.
- failure_reason: `NONE` 211, `POSITION_TIMEOUT` 62, `HANDLE_DETACHED` 33.
- failure_phase (of failures): `APPROACH` 62, `PULL` 33.
- ContactSensor available on all 306; **max unintended contact force 0.00 N** (min 0.00 N) — no unintended
  arm/hand–cabinet contact recorded in any episode (all failures are grasp misses, not collisions). Raw
  quantity only; no threshold chosen, no collision-confound labels applied.
- instrumentation missing fields: 0.

No `abs_eff` plotting, edge estimation, logistic/isotonic fit, collision-threshold selection, residual
edge-variance test, or PASS_TO_V4 judgement was performed. Those are Claude B's frozen read-only analysis.

## Data paths & hashes (`artifact_hashes.json`)
- `episodes.jsonl` sha256 `131750e158032e53e5c8daab6e94530c70de1224f05b8fc79a7497009524c20e`
- `manifest.json` `d6c2df38…`, `run_metadata.json` `b320b6ae…`, `run_status.json` `073275ed…`,
  `self_check.json` `680f20b3…`, `authorization_snapshot.json` `b83379dd…`,
  `integrity_summary.json` `f51ae069…`.
- frozen config on disk sha256 `2f20429b…` == authorized; `science_manifest_sha256` `e2c8216e…` == authorized.

## State flags
```
full_306_run_executed = true
confirmatory_data_exists = false
confirmatory_generator_implemented = false
science_analysis_by_A = none
second_306_run = false
```
Data is exploratory, one authorized run. Handoff to Claude B for the frozen read-only analysis.
