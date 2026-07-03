# Final Recommendation (v1, PRELIMINARY) — independent audit

**Auditor:** Claude C (independent). **Status:** preliminary. Claude B's Phase-2 selection-interaction
analysis on A's matched data is **not yet pushed** (offline `status.md`: "await Claude A's matched bank").
I therefore ran B's **unmodified** estimators on A's matched data as a Phase-2 preview; this verdict will
be finalized when B pushes Phase 2. Nothing was modified; no result-dependent tuning was done.

---

## Independent verdict: **MODIFY**

Applying the pre-agreed rubric:

| criterion | required for | finding |
|---|---|---|
| prediction **and** selection both robust; state-aware decision value ≫ robust no-history policy | **GO** | **fails** — selection null: VSI = +0.00067 (full utility), 0.0 otherwise; state-agnostic robust generalist `c1_steady` (27/27) ≈ oracle. |
| history no incremental info, **or** hidden-state leakage, **or** oracle has no usable upper bound | **STOP** | **fails all three** — history *does* carry incremental predictive info (D3/D4, capacity-matched); **no leakage** (verified); the oracle upper bound **is** computable — it is simply ≈0. |
| state identifiable **and** prediction improved, but under frozen utility VSI very low / selection unstable | **MODIFY** | **matches exactly** — D3/D4 hold; VSI≈0 and selection is unstable/zero under the frozen utility. |

This **converges with Claude B's Phase-1 MODIFY**, reached independently. The pilot is not a STOP (the
science is sound, no leakage, D1–D4 hold) and not a GO (the decision-value core — "does knowing the state
change which plan you pick?" — is null under a defensible utility).

### Why not STOP, precisely
The STOP clause "Oracle has no usable upper bound" is *nearly* triggered: the decision-value ceiling
(VSI) exists but sits at the floor (~0), so there is **no headroom for any selector**. That kills the
*selection* claim but not the *prediction* claim or the framework. The honest move is to **retire the
selection claim and publish the prediction+null-selection finding**, not to abandon the work.

### Why not GO
GO requires state-aware decision value ≫ a robust no-history policy. Here they are **equal to ~1e-3**.
Switch-rate (67 %) and rank-reversal (18 %) are landscape statistics, not decision value; using them for
GO would be the argmax-switch-as-utility error the audit charter forbids.

---

## Guardrails for Phase 2 / the paper (binding)

1. **Do not manufacture selection value.** A's §5 (remove the robust generalist; raise λ_time) would
   flip VSI by construction *after* observing the null. Permitted **only** if pre-registered,
   independently motivated, and the resulting claim is explicitly scoped to "banks lacking a robust
   generalist." Otherwise it is fitting the experiment to the desired conclusion. **B's §6 (keep the
   frozen utility; freeze λ on validation only) is the correct stance and should govern.**
2. **λ stays frozen a-priori.** Never select λ after seeing test. (Relabel `selected_on="validation"` →
   "a-priori frozen": there is effectively no validation set — 3 val sessions of the same 3 states — so
   the "validation-selected" provenance is inaccurate, though harmless since λ was not tuned on test.)
3. **Report VSI, not switch/reversal, as the decision-value metric.** State both, but headline VSI.
   Never present a switch-rate as evidence that selection helps.
4. **Treat replicates as blocks, not independent sessions.** Bootstrap over states/replicate-blocks;
   effective independent n over the hidden state is ~3, not 9. Do not report "n=9 sessions" of state
   evidence.
5. **Scope every claim to simulation.** 3 hand-picked sim damping values, no domain randomization, no
   sensor noise, deterministic-ish physics, no held-out state → **no real-robot deployment claim**.
   D3/D4 generalization to *unseen* damping is untested and must not be implied.
6. **Charge probe cost.** With VSI≈0, probing to enable selection is net-negative; report utility net of
   probe execution time, or state explicitly that probes are treated as free and why.
7. **Keep the honest headline** (claims doc): calibration buys prediction, not selection, when a robust
   generalist exists. That is the defensible contribution.

---

## What would move this to GO (legitimately)

- A **pre-registered** bank/utility (fixed before data) in which **no single candidate is near-optimal
  across states** — motivated by the task, not by the null — **and** VSI materially > 0 with a
  session/replicate-block bootstrap CI excluding 0, **and** a **capacity-matched** history model (not the
  over-capacity torch trunk) realizing a real fraction of VSI, **and** ≥1 **held-out hidden state**
  (requires >3 damping values, ideally a continuum) so identification is a generalization result, not a
  3-cluster lookup. Absent these, MODIFY stands.

---

## Audit trail (reproducibility)

- Sources: `paper/experiment/runtime-selection-v2` (`af4a270…`), `paper/experiment/offline-eval-v2`
  (`98c418c…`); read via detached worktrees, no branch mutated.
- First-hand checks: bank-sha/θ-purity/coverage/leakage-surface on A's `episodes.jsonl`; replicate-noise
  vs signal; B's `pytest` (13/13 leakage+oracle) re-run; **B's unmodified `pairing`+`oracle`+`utility`
  run on A's matched data** → `is_matched_bank: true`, 3 groups, VSI {success_only 0.0, success_error
  0.0, full +0.00067}, switch 2/3, reversal 16/90.
- Not done (per charter): no Isaac launch, no model/data/utility edits, no λ or result-driven tuning.
- **Finalize when:** Claude B pushes Phase 2. Expected to reproduce VSI≈0 (it uses the same estimators);
  if B's pushed Phase 2 instead reports a positive VSI, first check whether the bank or utility changed
  (Guardrail 1) before accepting it.
