# Test report (Fix4 snapshot freeze)

Environment: conda `env_isaaclab`, Python 3.11.15, torch 2.7.0+cu128, numpy 1.26.0 (see
`environment_snapshot.md`). Command base:
`pytest deployment_calibration/tests/offline_v2/calibration_bias/`.

## 1. Active B-owned suite (authoritative) — **PASS**
Run excluding the four C historical audit files:
```
pytest deployment_calibration/tests/offline_v2/calibration_bias/ \
  --ignore=.../test_preregistration_v4_audit_c.py \
  --ignore=.../test_preregistration_v4_fix1_reaudit_c.py \
  --ignore=.../test_preregistration_v4_fix2_final_reaudit_c.py \
  --ignore=.../test_preregistration_v4_fix3_final_reaudit_c.py
```
Result: **244 passed** (≈90 s). No active failures → snapshot is **not** blocked by
`FINAL_SNAPSHOT_BLOCKED_ACTIVE_TEST_FAILURE`.

Active B-owned files: `test_preregistration_v4.py`, `…_fix1.py`, `…_fix2.py`, `…_fix3.py`, `…_fix4.py`,
plus the earlier offline analysis tests (band-edge, confirmatory-redesign, learned-selector-power,
test-geometry-power).

## 2. C historical audit files (audit-only; classified separately)
These encode Claude C's expectations **against the commit each audited**. After the subsequent fixes, their
`strict-xfail` markers flip to `XPASS(strict)` (reported as "failed") — the **intended** machine-checkable
signal that the audited blockers were resolved — and a few PASS assertions are superseded by later API
tightening. They are **not** active regressions.

| C audit file | result | reading |
|---|---|---|
| `test_preregistration_v4_audit_c.py` | 12 passed, 4 failed | 4 xfail→XPASS = the 4 round-1 blockers (A/B/C/D) fixed |
| `test_preregistration_v4_fix1_reaudit_c.py` | 11 passed, 1 xfailed, 1 failed | 1 xfail→XPASS (identity format) fixed; 1 stays xfail |
| `test_preregistration_v4_fix2_final_reaudit_c.py` | 9 passed, 4 failed | 3 xfail→XPASS (guard/identity-domain/hash) + 1 superseded PASS (rebind to `select_best_single_confirmatory`) |
| `test_preregistration_v4_fix3_final_reaudit_c.py` | 6 passed, 1 xfailed, 4 failed | 3 xfail→XPASS (offset/full-validator/stale-spec) + 1 superseded PASS (bogus manifest now rejected); combined-hash stays xfail (omitting required kwargs → `TypeError`) |

These outcomes are **expected and documented** (see `known_issues_at_freeze.md` §D). C's audit files were
**not** modified.

## 3. Determination
- Active B-owned tests: **all pass (244)**.
- No active failure. Snapshot proceeds.
