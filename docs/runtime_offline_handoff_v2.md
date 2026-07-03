# Runtime → offline handoff — v2

How Claude B reads the data Claude A produces. Repo root here = `projects/paper/` (standalone git repo,
remote `ysdjy/paper`). Paths below are relative to that root unless absolute.

## Data location
- Each run: `deployment_calibration/data/<run_id>/`
  - `episodes.jsonl` — one JSON object per episode (the primary handoff).
  - `trajectories/<episode_id>.npz` — per-step trajectory (optional for offline eval).
  - `metadata.json` — run config, damping levels, git commit, `damping_verified`, sessions list.
  - (formal runs also carry: `split_manifest.json`, `run_command.txt`, `git_status.txt`, etc.)
- **Current pilot data (absolute)**:
  `/home1/banghai/Documents/IsaacLab/projects/paper/deployment_calibration/data/damping_pilot_v2_20260703_000056/`
- Capability map: `deployment_calibration/data/capability_map_v2_20260702_231522/`
- Stage-0 validation: `deployment_calibration/data/stage0_drawer_validation_v2_20260702_210317/`

## Episode contract (`deployment_calibration/contracts/episode_schema_v2.py`, `CONTRACT_VERSION="open_drawer_v2"`)
One execution = `(x, g, theta, H, z_secret) -> y`. Each `episodes.jsonl` record has:
- `episode_id`, `session_id`, `mechanism_id`, `drawer_name`, `episode_role`, `order_in_session`.
- `x` — observable initial state (model-legal). `g` — task target. `theta` — effective params used.
- `y` — outcomes. `secret_deployment_state` / `hidden_state_id` — ORACLE/AUDIT ONLY.
- `candidate_group`, `candidate_index`, `probe_index`, `history_cutoff`.
- provenance: `contract_version`, `git_commit`, `dirty_worktree`, `full_reset_verified`.

## Key semantics
- **`session_id`**: one deployment session = a contiguous period where the hidden state (drawer joint
  DAMPING) is FIXED. All episodes with the same `session_id` share the same hidden damping.
- **`episode_role`**: `"probe"` or `"candidate"`.
- **Order**: within a session, episodes are ordered by `order_in_session`. **All probes precede all
  candidates.** Probe episodes are the decision-legal history.
- **`probe_index`**: which fixed probe (0..K-1) this is (see `data_generation/probe_library_v2.py`).
- **`candidate_group`**: candidates sharing the SAME (mechanism, hidden state, initial condition, target,
  probe history); they differ ONLY in `theta`. Selection/regret is computed per group.
- **`candidate_index`**: index within the group.
- **`history_cutoff`**: number of probe results legally visible before this candidate's decision (= K).

## Building history (leakage-safe — do exactly this)
For a candidate episode `e`, its legal history = probe episodes with
`session_id == e.session_id` AND `episode_role == "probe"` AND `order_in_session < e.order_in_session`,
sorted by `order_in_session`, truncated to first K. Reference implementation:
`evaluation/run_session_eval_v2.py:build_history`. FORBIDDEN in history: any candidate result (self,
same-group, future), any other session, any train label into test history. Split BY SESSION first,
then build history within each split.

## Fields a NORMAL (deployable) model may read
- From `x`: `mechanism_id`, `drawer_name`, `member`, `robot_joint_pos`, `tcp_pos`, `tcp_quat`,
  `gripper_width`, `initial_mechanism_joint_pos` (see `episode_schema_v2.X_OBSERVABLE_KEYS`).
- `g` (target), `theta` (candidate params), and `H` (probe outcomes: g_i, theta_i, success,
  failure_reason, task_outcome_error, skill_elapsed_time, pull_phase_duration, handle_relative_error).

## Fields ONLY an Oracle / audit may read
- `secret_deployment_state` (e.g. `{"damping": 40.0}`), `hidden_state_id`.
- These MUST NOT enter ordinary model input `x` (the schema's `to_dict` raises if a damping/secret/hidden
  key appears in `x`). Oracle-Z / Oracle-Candidate use them for upper bounds only.

## Offline evaluation entry points (current; B may extend under offline_v2/)
- Split by session: `evaluation/session_split_v2.py`.
- Baselines: `baselines/predictors_v2.py` (B0/B1/B2-Mean/B2-Seq/Oracle-Z, numpy).
- Full eval (D1–D5 + regret + adaptation): `evaluation/run_session_eval_v2.py`.
  Run: `python projects/paper/deployment_calibration/evaluation/run_session_eval_v2.py <run_dir> --fracs 0.34,0.33,0.33 --k 0,1,2,4`
- Leakage test: `tests/test_history_no_leakage_v2.py`.

## Known pilot caveats for B to address offline (no new sim needed)
- Small test (3 groups) → AUROC/regret need bootstrap CIs and small-N-safe estimators.
- D5 (regret) inconclusive at pilot scale; B should quantify with CIs and held-out D3, and define the
  regret estimator the formal run will use.
