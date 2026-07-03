# offline_v2 status — Claude B

Branch `experiment/offline-eval-v2`, worktree `projects/paper_offline_v2`. Offline/CPU only;
never launches Isaac; only writes under `deployment_calibration/{offline_v2,models_v2,
evaluation/offline_v2,tests/offline_v2}/` and `docs/offline_v2/`.

## Current commit
Run `git log --oneline -8` on this branch. Latest work: pilot reanalysis + history models + full
offline pipeline/CLI.

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
