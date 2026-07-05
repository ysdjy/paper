# FINAL-001 / FINAL-002 final regression (Claude C)

**Auditor:** Claude C, independent, **read-only**, **regression-only** (FINAL-001 and FINAL-002 only; no
re-audit of FINAL-003/004/005; no new issue IDs). No Isaac, no generator, no formal manifest/checkpoint/
runtime/confirmatory data.

**Audited completion commit:** `48f28ac3dc9836680c010ff58e0cccd1c669b01e`; previous regression `ad2a9ca`;
frozen-issue source `01ac19d`; audited snapshot `b983b7e`; power-cert `2bf7217`.

## VERDICT: **FROZEN_ISSUE_BATCH_FIX_INCOMPLETE**

Failing IDs (originals only): **FINAL-001, FINAL-002**. FINAL-003/004/005 remain **UNCHANGED_RESOLVED**.
No `CRITICAL_NEW_EVIDENCE`. **Power re-certification not required.**

> **Verdict correction (transparency).** A first pass concluded GO after checking only the *new*
> `deep_manifest_validator.checks` block. Continued verification — a config-wide scan of **all** validator
> description blocks — found the **originally-flagged** `manifest_integrity.deep_validation.checks` block
> still carrying the stale descriptions. The verdict is corrected to INCOMPLETE. This is the same §2.5/§4.5
> active-description criterion, still unmet in a second config location — **not** new evidence.

| id | status |
|---|---|
| FINAL-001 | **INCOMPLETE** |
| FINAL-002 | **INCOMPLETE** |
| FINAL-003 | UNCHANGED_RESOLVED |
| FINAL-004 | UNCHANGED_RESOLVED |
| FINAL-005 | UNCHANGED_RESOLVED |

## Root cause (both issues, one cause)
The audited snapshot `b983b7e` had **one** validator description: `manifest_integrity.deep_validation.checks`
(13 checks) — the block the prior regression flagged for FINAL-001 §2.5 ("resolved_candidate_order is a bank
permutation") and FINAL-002 §4.5 ("config/commit hex formats; …; device==cpu"). The completion patch **added
a second, parallel block** `deep_manifest_validator.checks` (16 checks) with the **corrected** wording, but
**did not remove or update** the originally-flagged `manifest_integrity.deep_validation.checks`. The active
config now holds **two contradictory validator descriptions** — one correct, one stale.

## FINAL-001 — INCOMPLETE
- **Resolved (verified):** the §2.1–2.4 CODE is fully correct — canonical storage-order functions
  (15/57/228, 9/18/72); the validator rejects shuffled `blocks`/`sessions`/`trials` lists with distinct
  messages (incl. a shuffled trials list whose `execution_order_index` stays a complete 0..227 set); two
  reference constructions give identical canonical JSON + full hash (manifest bytes uniquely determined);
  storage order ≠ execution order (separated). The **new** `deep_manifest_validator.checks` block and the
  `storage_order_vs_execution_order` field are correct.
- **Still INCOMPLETE (§2.5):** `manifest_integrity.deep_validation.checks[6]` still reads *"…
  resolved_candidate_order **is a bank permutation**"* and `checks[9]` still reads *"execution_order_index …
  complete 0..N-1"* — the old semantics the completion was required to remove; that block has no
  canonical-storage-order description.
- **Fix:** remove the superseded `manifest_integrity.deep_validation.checks` block (consolidate to the one
  corrected `deep_manifest_validator.checks`), or update its stale lines to the seed-recomputed /
  canonical-storage wording. There must be exactly one, non-contradictory active validator description.

## FINAL-002 — INCOMPLETE
- **Resolved (verified):** the exact-value CODE checks are fully valid (config-hash flip / wrong
  `schema_version` / determinism `True`-for-`1` / GPU / extra key all → INVALID). The **new**
  `deep_manifest_validator.checks` block describes them correctly.
- **Still INCOMPLETE (§4.5):** `manifest_integrity.deep_validation.checks` still contains *"frozen_seeds ==
  preregistered 15; **config/commit hex formats**; manifest_algorithm_version; **device==cpu**"* — the exact
  stale summary §4.5 required removed.
- **Fix:** in the same consolidation, replace that line with the exact-check description already in
  `deep_manifest_validator.checks`.

## Invariants (unchanged) & power
Bank `{-0.04,0,+0.04}`, test `±0.035`, blocks 9/6/9, `POWER_SUFFICIENT_FOR_PREREG_V4`, 306 `131750e1…` —
unchanged. FINAL-003/004/005 source files unchanged. Only a documentation-string consolidation is needed →
**`power_recertification_required = false`**.

## Tests
- **B completion** `test_final_001_002_completion.py` → **12 passed** (does not scan the second
  `manifest_integrity.deep_validation.checks` block — hence the gap).
- **C final regression** `test_final_001_002_regression_c.py` → **9 passed, 2 xfailed** (FINAL-001 §2.5 and
  FINAL-002 §4.5 stale second block).

## Authorization
| flag | value |
|---|---|
| authorizes_generator_implementation | **false** |
| authorizes_generator_smoke | **false** |
| authorizes_formal_manifest_generation | **false** |
| authorizes_confirmatory_data_generation | **false** |
| authorizes_confirmatory_run | **false** |
| power_recertification_required | **false** |

**Next step:** Claude B removes/consolidates the stale `manifest_integrity.deep_validation.checks` block so
exactly one non-contradictory active validator description remains; Claude C then re-regresses **only**
FINAL-001 and FINAL-002. On green (the CODE is already correct), all five are RESOLVED →
`GO_TO_GENERATOR_IMPLEMENTATION`. The ±0.035 power certification carries over unchanged.
