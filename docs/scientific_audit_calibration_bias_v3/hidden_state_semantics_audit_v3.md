# Hidden-state semantics audit (Audit A)

**Question:** what hidden state actually determines execution outcome, and can the paper claim it tests
"unseen hidden-state interpolation"? All support values re-derived first-hand from
`confirmatory_design_validation_v3.json` and the residual model.

## 1. The outcome-determining hidden state is `actual_bias_y`, not `nominal`

From the frozen algebra (`handle_calibration_bias_v1.py`, verified):
`commanded_grasp = true_handle + actual_bias + offset`, and success ⇔ `|actual_bias + offset| ≤ 0.02`
(the empirical band — recomputed below). Therefore **the hidden state that decides the outcome is
`actual_bias_y = nominal_bias_y + residual_bias_y`.** `nominal` is a *commanded label*; `residual` is a
per-block calibration error. Neither alone drives the outcome — their **sum** does.

**Recomputed success band (135 eps):** success = 1.00 at `|bias+offset|` ∈ {0.00, 0.02}; 0.00 at ≥ 0.04.
Success ⇔ `|actual_bias + offset| ≤ 0.02`, sharp and monotone.

## 2. Is `residual` "just nuisance" or part of the hidden state? — **Part of the hidden state.**

The residual sits on the **same axis** as the bias and enters the exact quantity the success band is
defined on (`|actual_bias + offset|`). It is therefore a genuine **component of the outcome-determining
hidden state**, not noise external to it. It is "nuisance" only in the *operational* sense that (a) no
model reads it and (b) it is small (SD ≈ 0.0044 m) relative to the nominal separation that drives the
*decision*. **The paper must not treat `actual` as the hidden state while simultaneously calling
`residual` outside-the-state noise** — that is internally inconsistent. Correct framing: the hidden state
is `actual_bias`, decomposed into a decision-dominant commanded component (`nominal`) and a small
stochastic component (`residual`).

## 3. Does train actual-support cover test actual-support? — **YES, fully.**

Recomputed from the residual support `[−0.01, +0.01]`:

| split | nominal levels | actual-bias support (nominal ± residual) |
|---|---|---|
| train | {−0.04,−0.02,0,+0.02,+0.04} | **[−0.05, +0.05] (contiguous)** |
| val | {−0.01,+0.01} | [−0.02, +0.02] |
| **test** | {−0.03,+0.03} | **[−0.04,−0.02] ∪ [+0.02,+0.04]** |

Both test intervals are **subsets of the train actual support**. The *same actual physical state* recurs
in both splits — e.g. actual = −0.03 arises in train (nominal −0.04, residual +0.01) **and** in test
(nominal −0.03, residual 0). At the level of the outcome-determining hidden state, **train and test share
support; the test physical state is not held out.**

## 4. Adjudication — what the experiment actually tests

- **NOT** "unseen actual physical state" (test actual ⊂ train actual support).
- **NOT** truly novel hidden physics.
- **IT IS** generalization to **unseen *nominal* deployment conditions** — the commanded-bias labels
  {−0.03,+0.03} are interior values not present in train — while the underlying **actual calibration
  state remains within the training-covered range**. That is *interpolation in the commanded-condition
  label space with in-support actual state.*

### Required paper wording (do NOT use "unseen hidden bias")
> **Allowed:** "generalization to unseen *nominal* deployment conditions (interior interpolation of the
> commanded calibration bias), with the underlying actual calibration state lying within the training
> support."
>
> **Forbidden:** "unseen hidden state," "unseen physical/actual bias," "extrapolation to novel hidden
> physics," or any phrasing implying the test physical state was not present in training.

## 5. Secondary semantics flag — the decision value is *constructed*, not generic

The "guaranteed measurable" decision value (best-single test success ≤ 0.5; state-aware 1.0; gain 0.5)
follows **by construction** from choosing the two test biases as the **outermost interior levels
{−0.03,+0.03}** — 0.06 apart, requiring opposite-sign offsets, with empty common-offset set. This is a
legitimate **pre-registered positive-control** demonstrating that *when deployment conditions genuinely
require different actions*, state-aware selection helps — and, unlike the v2 damping stage, here the
structural decision value is **real** (no robust generalist; recomputed frozen VSI +0.52, success-only
VSI +0.40 on train). But the claim must be scoped: it demonstrates decision value **exists in a
constructed two-extreme scenario**, not that calibration generically yields large decision value. Report
the decision value alongside the design fact that the test set was chosen to require opposite offsets.
