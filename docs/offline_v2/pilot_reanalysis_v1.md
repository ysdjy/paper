# 72-episode pilot — independent offline reanalysis (v1)

Claude B, branch `experiment/offline-eval-v2`. This is an **independent re-computation** of the
`damping_pilot_v2_20260703_000056` pilot using the audited `offline_v2` framework. It does not
reuse the numbers in `docs/paper_damping_pilot_v1.md` or `data/.../evaluation/eval_v2.json`; it
recomputes everything from `episodes.jsonl` with leakage-safe, tie-aware, held-out estimators
and session-level bootstrap CIs.

## Data provenance
- source: `deployment_calibration/data/damping_pilot_v2_20260703_000056/`
- source git commit: `ec18e4cdd347a64cc5e5cb0f2a7950de7f0c87fa` (branch `main`, dirty worktree)
- sha256(episodes.jsonl): `a732b2dc66bb45b7…`
- 72 episodes = 9 sessions × (3 probes + 1 group × 5 candidates); middle_drawer only.
- hidden state = drawer-joint DAMPING, fixed per session at L1=3 / L2=20 / L3=40 (3 sessions each).
- `damping_verified: true`, `damping_set_maxerr: 0.0`.

Regenerate: `python -m deployment_calibration.offline_v2.cli evaluate <run_dir> --seeds 5 --n-boot 2000`.
Artifacts under `deployment_calibration/evaluation/offline_v2/damping_pilot_v2_20260703_000056/`.

---

## 1. Three methodological bugs in the original eval (fixed here)

**(a) AUROC was not tie-aware.** The original `auroc()` ranked scores with `np.argsort`, which
breaks ties arbitrarily. For a constant predictor (B0) every score is tied, so it returned
**0.179** (see `eval_v2.json`) instead of the correct **0.500**. `offline_v2.metrics.auroc`
delegates to `sklearn.roc_auc_score` (tie-aware) and returns NaN + a warning on single-class
folds. Unit tests pin constant→0.5, perfect→1, reversed→0, single-class→NaN.

**(b) The "disjoint split" check was a non-check.** The original tested
`len(train ∩ val ∩ test) == 0` — a three-way intersection that is empty even when train and val
fully overlap. `offline_v2.splits.audit_split` enforces **pairwise** disjointness, a **partition**
check (no session dropped or duplicated), and **candidate-group-within-one-split**. The pilot
split passes all of these — the conclusion held, but the original test could not have caught a
violation.

**(c) D3 was evaluated in-sample.** The original fit a nearest-centroid on **all 9** sessions and
scored the **same** 9 → acc = 1.0 by construction. `offline_v2.d3` uses **leave-one-session-out**
(centroid from the other 8) and a train/test-centroid variant, with a session bootstrap CI.

A fourth issue is a **mislabel, not a bug**: the model called `B2_seq_deepsets` in
`baselines/predictors_v2.py` is a hand-crafted mean+max feature vector, not a DeepSets network.
`models_v2` ships a real permutation-invariant DeepSets and an order-sensitive GRU (both torch,
early-stopped, ≥5 seeds); the hand-feature version is retained as `B2_mean`.

---

## 2. Headline results (test fold, split seed 0, session-bootstrap 95% CI)

| model | AUROC (95% CI) | Brier | fail-macroF1 | errMAE | mean regret (95% CI) | top1 | selSucc |
|---|---|---|---|---|---|---|---|
| B0 baserate | 0.500 [0.500, 0.500] | 0.259 | 0.348 | 0.083 | 0.480 [0.00, 1.42] | 0.33 | 0.67 |
| B1 static | 0.536 [0.00, 0.75] | 0.276 | 0.550 | 0.084 | 0.580 [0.00, 1.41] | 0.33 | 0.67 |
| **B2 mean** | **0.696 [0.00, 0.78]** | 0.210 | 0.785 | 0.073 | 0.580 [0.00, 1.41] | 0.33 | 0.67 |
| B2 deepsets | 0.893 [0.25, 1.00] | 0.076 | 0.932 | 0.030 | 0.000 [0.00, 0.00] | 1.00 | 1.00 |
| B2 gru | 0.893 [0.25, 1.00] | 0.069 | 0.932 | 0.027 | 0.000 [0.00, 0.00] | 1.00 | 1.00 |
| OracleZ | 0.714 [0.00, 0.78] | 0.217 | 0.722 | 0.072 | 0.580 [0.00, 1.41] | 0.33 | 0.67 |

The CIs are enormous: **9 sessions, 3 in test, 3 candidate groups.** Every point estimate here is
**exploratory**. Two structural cautions:

- **The torch models' AUROC gain is capacity, not history.** In the adaptation curve, DeepSets/GRU
  at **K=0 (no probe history at all)** already reach AUROC **0.95 / 0.98** and regret 0 — before
  seeing a single probe. Their level is set by a flexible nonlinear trunk overfitting 30 training
  candidates, not by history. The **capacity-matched** history signal is the linear `B2_mean`
  (same head as `B1`), which moves **K=0: 0.536 → K=1: 0.714 → K=3: 0.696**. That is the honest
  history effect; the DeepSets number confounds capacity with history and even exceeds OracleZ,
  which is only possible as small-N overfitting.

- **Regret is unstable across splits (below), so the seed-0 "regret 0 / top1 1.0" is luck.**

### Split-robustness (3 stratified split seeds; each puts 1 session/damping in each fold)

| metric | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| B1 AUROC | 0.54 | 0.52 | 0.70 |
| B2-mean AUROC | 0.70 | 0.56 | 0.82 |
| **DeepSets AUROC** | 0.89 | 0.67 | 0.96 |
| DeepSets mean regret | **0.00** | **1.01** | **0.005** |
| DeepSets top1 | 1.00 | 0.00 | 0.67 |
| B0 mean regret | 0.48 | 0.95 | 0.49 |
| OracleZ mean regret | 0.58 | 1.40 | 1.04 |
| D3 LOSO acc | 1.00 | 1.00 | 1.00 |

Prediction ordering (B2 ≥ B1) is **directionally robust**. Selection (regret / top1) is **not**:
DeepSets regret swings from best (0.00) to **worse than B0** (1.01 vs 0.95 at seed 1). Decisively,
**OracleZ — which knows the true damping — also fails to reduce regret** (1.40 at seed 1), i.e. the
regret signal at this scale is dominated by 3-group noise, and even perfect state knowledge shows
no selection advantage here.

---

## 3. Held-out D3 (hidden-state identification)

Leave-one-session-out nearest-centroid on per-session probe-feature means:

- **accuracy 1.000, balanced-accuracy 1.000, macro-F1 1.000** (chance 0.333), confusion matrix
  diagonal `[[3,0,0],[0,3,0],[0,0,3]]`. Robust across all three split seeds.
- The bootstrap CI is degenerate `[1.000, 1.000]` because **every** held-out session is classified
  correctly (only 9 sessions). Read this as "cleanly separable in this pilot," **not** as a tight
  interval — 9 sessions cannot produce a meaningful CI.

This **corrects the methodology** (the original was in-sample) while the **conclusion stands**:
damping is strongly identifiable from probe outcomes (driven by the ~1.14 s pull-duration spread
across levels). D3 does not read any candidate outcome.

---

## 4. Why switch / rank-reversal / VSI are NOT reported

The pairing validator (`offline_v2.pairing.build_matched_bank`) reports **`is_matched_bank: false`**:
**0 of 45** candidate theta-signatures are shared across ≥2 damping levels. Each session drew its
own 5 candidate θ's independently, so there is **no matched candidate bank**. Consequently:

- **Optimal-candidate switch rate**, **pairwise rank-reversal rate**, and **VSI** are **not
  computable** and are returned as `available: false` with a reason — never fabricated.
- Without them we **cannot** claim that knowing damping changes *which candidate is best*. This is
  the crux: D4 shows history improves *outcome prediction*; nothing in this pilot can show it
  improves *plan selection*.

Per the offline plan (§9): if the formal data likewise shows predictive gain but VSI ≈ 0 / no rank
switch, the honest conclusion is **"history improves outcome prediction, not plan selection"**, and
the recommendation to Claude A is to build a **matched candidate bank** (same θ set across damping)
and, if VSI is still ≈ 0, to widen the candidate space or introduce a hidden variable that
interacts more directly with grasp offset. B does not modify runtime.

---

## 5. Re-graded D1–D5 (independent; supersedes the v1 grades)

| # | claim | v1 grade | **B re-grade** | basis |
|---|---|---|---|---|
| D1 | damping changes outcomes | YES | **Supported (descriptive)** | per-level candidate success 10/5/3 of 15, monotonic; final-pos spread 0.118 m, pull-dur spread 1.14 s. Descriptive over full data, not held-out. |
| D2 | damping hidden from x | YES | **Supported** | `x_std across levels = 0` (identical resets); contract guard blocks damping/secret/hidden in x. B1>0.5 comes from θ→success, not x-leakage. |
| D3 | probe history identifies hidden state | YES (in-sample) | **Supported, held-out** | LOSO acc 1.00 (chance 0.33), robust over 3 seeds. CI degenerate at 9 sessions. |
| D4 | history improves candidate prediction | YES | **Supported but exploratory; capacity-controlled only** | capacity-matched `B2_mean` K=0→K≥1 AUROC 0.54→0.71, Brier 0.28→0.21, errMAE 0.084→0.073, in all 3 seeds. Torch models' higher AUROC is capacity+overfit (K=0 already 0.95), not history. |
| D5 | prediction gain reduces selection regret | INCONCLUSIVE | **Not supported at pilot scale; not yet testable** | regret unstable across splits (DeepSets 0.00/1.01/0.005), sometimes worse than B0; OracleZ also fails to reduce regret; not a matched bank → VSI/switch/reversal uncomputable. |

**Net:** the pilot supports *hidden-state exists (D1), is hidden (D2), is recoverable from probes
(D3), and probe history incrementally improves outcome prediction (D4, capacity-controlled)*. It
does **not** support the selection/decision-value claim (D5) and **structurally cannot** measure it
(no matched candidate bank, 3 test groups). This is a stronger, more precise statement than the v1
"inconclusive."

---

## 6. Verdict on the pilot: **MODIFY** (unchanged label, sharper reason)

Not a STOP: D1–D4 hold under corrected methodology, no leakage, split is a valid partition. Not a
GO: the decision-value core of the paper (does knowing the hidden state change *which plan you
pick*?) is unmeasured and, on this data, structurally unmeasurable.

**Required before a formal claim (informs Claude A; B does not implement runtime):**
1. **Matched candidate bank** — the same candidate θ-set replayed under every damping level, so
   switch-rate / rank-reversal / VSI become computable and the pairing validator passes.
2. **Many more candidate groups** — ≥ ~24 test groups (multiple groups/session × targets) so
   regret and top-1 have signal beyond quantization noise.
3. Freeze λ on validation only; keep the frozen `utility_config.json`; report ≥5 seeds with CIs.
4. Expect the torch-vs-linear gap to shrink once N grows; report the **capacity-matched** history
   effect (linear B2_mean and DeepSets K=0→K sweep) alongside the raw model, not the raw model alone.

Everything in this document is reproducible from `episodes.jsonl` via the `offline_v2` CLI; no Isaac
Sim, no runtime edits, no synthetic data promoted to results.
