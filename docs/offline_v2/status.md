# offline_v2 status — Claude B

Branch `experiment/offline-eval-v2`, worktree `projects/paper_offline_v2`. Offline/CPU only;
never launches Isaac; only writes under `deployment_calibration/{offline_v2,models_v2,
evaluation/offline_v2,tests/offline_v2}/` and `docs/offline_v2/`.

## Current commit
Run `git log --oneline -10` on this branch. Latest work: **Phase 2** — analysis of Claude A's
matched-candidate selection-interaction pilot (verdict MODIFY).

## Phase 2 (selection_interaction_pilot_v1) — DONE
Source (read-only): `.../data/selection_interaction_pilot_v1_20260703_110727/`; source commit
`1921641acec5a7e153f45f909c69613ed6036a12`; candidate_bank_sha256 `d1d80d9646aae536…`;
dirty_worktree false; damping_verified true; full_reset_all_verified true.

- **Pairing/leakage**: ALL PASS — 162 eps, 9 matched groups (target×replicate × 3 damping),
  27 selection groups, candset/θ/target consistent, no candidate-outcome in history, no secret in x.
- **Replicate independence**: technical repeats — success/failure identical in 45/45 cells, only 3
  unique hidden states. CIs are target-block bootstrap, EXPLORATORY. (`replicate_independence_audit_v1.md`)
- **Decision value (frozen U)**: VSI = **+0.00059** (independently matches A's +0.0006); switch rate
  0.667; rank reversal 0.170; robust-generalist(c1_steady) gap to state-aware oracle = VSI; best-single
  archetype = c1_steady. selSucc = 1.00 for every policy.
- **Prediction**: capacity-matched history gain B1 AUROC 0.87 → B2-mean 1.00 (Brier 0.152→0.022).
- **Selection**: B2 regret 0.000 vs B1 0.026 but robust generalist 0.0007 ≈ oracle; stable over 3 split
  seeds (B2 0.000/0.0002/0.0002).
- **VOI**: Gross ≈ VSI (0.0006); **Net VOI negative** (K=1 −0.19) once probe time charged.
- **Sensitivity (supplementary)**: VSI≈0 under success/success+error; grows to only 0.067 at λ_time 0.5;
  low VSI is candidate-bank structure (robust generalist exists), not light speed weight. Frozen U kept.
- **Verdict: MODIFY**; recommend Route B (handle-pose / calibration-bias hidden state that directly
  moves the optimal grasp candidate) + nuisance variation, over Route A. (`final_go_modify_stop_v1.md`,
  `selection_interaction_analysis_v1.md`)
- New code: `offline_v2/replicate_audit.py`, pairing (target,replicate) keying + `full_pairing_validation`,
  oracle policy baselines (`robust_generalist`/`best_single_archetype`/`gross_net_voi`),
  `evaluation/offline_v2/run_selection_interaction.py`; tests `test_matched_bank_v2.py`.
- Artifacts: `deployment_calibration/evaluation/offline_v2/selection_interaction_pilot_v1_20260703_110727/`.
- Reproduce: `python -m deployment_calibration.evaluation.offline_v2.run_selection_interaction <A_run_dir>
  --out <OUT> --seeds 5 --n-boot 2000 --split-seeds 0,1,2`.

## Phase 1 (72-ep damping pilot) — DONE (below)

## Completed
- **Robust metrics** (`offline_v2/metrics.py`): tie-aware AUROC (sklearn), AUPRC, Brier, ECE,
  success-acc, failure macro-F1, balanced-acc, macro-F1, confusion. NaN+warn on single-class.
- **Split audit** (`offline_v2/splits.py`): session split + strict pairwise-disjoint, partition,
  and candidate-group-within-one-split checks; manifest.
- **Leakage-safe history** (`offline_v2/history.py`): same-session, order<candidate, ≤K,
  whitelisted fields; `assert_history_legal` rejects future/cross-session/candidate-field/
  privileged/cutoff violations (content-checked).
- **Oracle decision-value** (`offline_v2/oracle.py`): Oracle-Candidate regret; state-agnostic /
  state-aware oracles; VSI; optimal-candidate switch rate; pairwise rank-reversal — all gated by
  the matched-bank pairing validator (`offline_v2/pairing.py`), returning `available:false`+reason
  on non-matched data.
- **Bootstrap** (`offline_v2/bootstrap.py`): session-level nested resampling (candidate groups
  nested in sessions), 2000 default / test-mode down to 50.
- **Utility** (`offline_v2/utility.py`): frozen U = p − λ_err·err − λ_time·time; clip bounds fixed;
  `utility_config.json`; success-only / success+error / full sensitivity variants.
- **Held-out D3** (`offline_v2/d3.py`): leave-one-session-out + train/test centroid; bal-acc,
  macro-F1, confusion, bootstrap CI.
- **Models** (`models_v2/`): B0, B1, B2Mean (numpy); real torch **DeepSets** (permutation-invariant)
  and **GRU** (order-sensitive) with multitask heads, early stopping, ≥5 seeds, save/load; OracleZ.
- **Pipeline + CLI** (`evaluation/offline_v2/pipeline.py`, `offline_v2/cli.py`): `validate`,
  `evaluate[-pilot]`, `train`, `run-all` → emits validation_report / split_manifest / metrics /
  metrics_ci / predictions / candidate_rankings / oracle_analysis / adaptation_curve / main_results
  / ablation_results / summary.md / utility_config / provenance.
- **Pilot reanalysis** → `docs/offline_v2/pilot_reanalysis_v1.md` (D1–D5 re-graded).

## Test command / result
`conda activate env_isaaclab && python -m pytest deployment_calibration/tests/offline_v2/ -q`
→ **29 passed** (metrics ties/NaN, split audit, history poison-rejection, oracle+pairing on a
synthetic matched bank, DeepSets permutation-invariance + history-beats-static).

## Data source
`deployment_calibration/data/damping_pilot_v2_20260703_000056/` (read-only), source commit
`ec18e4cd…`, sha256(episodes) `a732b2dc…`. Canonical eval artifacts:
`deployment_calibration/evaluation/offline_v2/damping_pilot_v2_20260703_000056/`.

## Known limitations
- Pilot is **not a matched candidate bank** (0/45 shared θ) → VSI / switch / reversal uncomputable.
- 9 sessions / 3 test groups → all point estimates EXPLORATORY; CIs huge; D3 LOSO CI degenerate.
- Torch models overfit at N=30 train (DeepSets AUROC 0.95 at K=0, before any history) — report the
  capacity-matched history effect (`B2_mean`, DeepSets K-sweep), not raw torch level.

## Next steps
1. Await Claude A's `selection_interaction_pilot_v1_<ts>` (matched bank). Read by absolute path,
   record source_run_path / source_git_commit / candidate_bank_sha256.
2. On arrival: pairing validator → switch/reversal/VSI → B0/B1/B2/Oracle → session bootstrap →
   adaptation K=0..3 → GO/MODIFY/STOP against the §9 criteria.
3. If predictive gain but VSI≈0 / no rank switch: conclude "prediction not selection"; recommend A
   widen candidate space / add grasp-offset-interacting hidden var (no runtime edits by B).
