# calibration-bias offline status — Claude B

Branch `experiment/offline-calibration-bias-v1`, worktree
`projects/paper_calibration_bias_offline_v1` (branched from `origin/experiment/integration-damping-v1`).
Offline/CPU only; never launches Isaac; only writes under
`deployment_calibration/{offline_v2,evaluation/offline_v2,tests/offline_v2}/calibration_bias/` and
`docs/offline_v2/calibration_bias/`. Reuses the frozen offline_v2 framework + models_v2.

## Stage: protocol + tooling + synthetic validation — DONE (no real data yet)

### Delivered code
- `offline_v2/calibration_bias/schema.py` — spec-only layer (does NOT touch frozen contract):
  hidden `bias_y`, controllable `grasp_offset_local_y`, model-legal vs hidden fields, FieldMap.
- `offline_v2/calibration_bias/validator.py` — leakage/pairing gate + poison-tested checks
  (no bias/seed in x; probes carry no candidate fields; leakage-safe history; nuisance⊥bias;
  matched offset bank consistent).
- `offline_v2/calibration_bias/splits.py` — bias-level split with held-out INTERIOR levels,
  double isolation (level+session+seed), interpolation bracketing audit, manifest.
- `offline_v2/calibration_bias/independence.py` — nuisance provenance + replicate variance +
  near-duplicate detection + effective-N + **blocker rule**.
- `offline_v2/calibration_bias/oracle.py` — matched offset bank (bias↔damping alias to reuse
  offline_v2.oracle): oracle-candidate / state-agnostic / state-aware / VSI (frozen + success-only) /
  switch / reversal / best-single offset / robust-offset existence / best-offset-per-bias.
- `offline_v2/calibration_bias/voi.py` — probe-time + Gross/Net VOI (frozen + success-only).
- `offline_v2/calibration_bias/synthetic.py` — controllable compensation world (`synthetic_only`).
- `evaluation/offline_v2/calibration_bias/pipeline.py` — validate → independence → decision-value →
  bias-level split → held-out selection + adaptation → VOI → **frozen exploration gate** → artifacts.

### Tests — `pytest deployment_calibration/tests/offline_v2/calibration_bias/` → **20 passed**
(+ 48 total offline_v2, framework unaffected). Poison: bias/seed/block-id/effective-error in x,
candidate masquerading as probe, block↔bias 1:1. Independence: exploration paired block-reuse across
bias is LEGAL. Split: partition/pairwise-disjoint(levels+sessions+seeds+blocks)/interpolation/<5-levels;
confirmatory block-crossing-split flagged. Pipeline: positive world (gate PASS, VSI>0.05 AND success
gain>0.15, no robust offset, DeepSets K0→K2 history gain), negative world (robust offset → gate fails),
deterministic-replicate → blocker; **gate criterion-3 AND logic** (time-only VSI without success gain
FAILS; success gain without VSI FAILS).

### Frozen exploration gate (GO-to-preregistration; thresholds fixed in the draft)
1. ≥3 bias levels have a different best offset. 2. no robust generalist (worst-case gap ≤0.02 → none).
3. **success-only gain ≥0.15 AND frozen VSI ≥0.05 (both; AND not OR)** — 3a necessary for physical
decision value, 3b necessary for combined utility. 4. independence blocker False. 5. validation ok.
Net VOI is NOT a capability-map gate but MUST be positive in the final confirmatory GO.

### Held-out bias split scheme
≥5–7 continuous levels; extremes→train; ≥1–2 unseen **interior** levels→test (bracketed by train =
interpolation); intermediate→val; disjoint by level+session+**nuisance seed + nuisance block**;
deterministic seed/block never crosses split. Manifest + pairwise-disjoint unit tests.

### Nuisance-block rules (exploration vs confirmatory)
- Exploration Capability Map: same block MAY be reused across ALL bias levels (paired matched context);
  probe+all candidates in a session share one x/g nuisance; raw seed/block id never in `x`; independence
  treats block-reuse-across-bias as HEALTHY and rejects only a 1:1 block↔bias encoding.
- Confirmatory: train/val/test use non-overlapping block ids + seeds; a block must NOT cross a split;
  within a split a block may still pair across that split's bias levels. Enforced by split audit.

### Independence blocker rule
same-bias sessions near-deterministic AND seeds absent/not-varied-within-level ⇒ blocker=True ⇒
CIs may not narrow by session count, GO disallowed. Formal CIs bootstrap over seeds/level blocks.

### Synthetic pipeline result (SYNTHETIC ONLY, `synthetic_demo_v1/`)
frozen VSI 0.590, success-only VSI 0.571, switch 1.0, reversal 0.417, no robust offset, 5 distinct
best offsets, gate PASSED. Held-out interior biases: DeepSets/GRU selSucc≈1.0 (regret≈0) vs all
no-history/best-single selSucc 0.0 (regret 1.07). DeepSets AUROC K0 0.39→K2 0.97. Net VOI +0.43 (K1)
→ +0.03 (K3). Negative control → robust offset + VSI≈0; deterministic control → blocker.

### KEY protocol insight (baked into the draft)
Compensation success is a non-monotonic BAND in offset, so **linear B1 vs B2-mean understates the
history effect** (both near chance). The honest H1 capacity control is **DeepSets K=0 vs K>0**; a
feature-enriched linear baseline is reported as a secondary capacity-matched comparison.

## Waiting on Claude A capability map — ingestion plan
1. Read A's exploratory capability map **read-only** by absolute path; record source_run_path /
   source_git_commit / candidate_bank_sha256 / dirty_worktree / reset flags.
2. Run validator → independence → decision-value → exploration gate.
3. Issue **one** freeze proposal (bias levels, offset bank, probe set, split).
4. Commit `preregistration_v1` (final) BEFORE any confirmatory data.
5. Do NOT request the confirmatory run; wait for user confirmation.

Docs: `preregistration_draft_v1.md`, this file.
