# Confirmatory v4 generator + smoke — Claude B implementation-consistency audit (read-only)

Independent, read-only audit of Claude A's KAT-gated generator + `smoke_only` isolation.

- audited commit: `b6e6700c0cffe41a9c861809566ae97e008e0c1b`
- source GO: `c8ac0af29a0884c8efb520151729d515613d70bb` (ahead_by 1; only `.gitignore` + 5 A files changed;
  frozen active modules / KAT sampler / identity / validator / selection / power / models_v2 / 306 untouched)
- **VERDICT: `GENERATOR_SMOKE_BATCH_FIX_REQUIRED`** — 5 BLOCKING + 2 NONBLOCKING. `issue_list_frozen = true`.
- Machine form: `confirmatory_v4_generator_audit_b.json`. Findings are demonstrated by
  `test_confirmatory_v4_generator_audit_b.py` (12 passed).
- **No formal manifest / confirmatory data / run is authorized. Claude A must not proceed to formal freeze.**

## What is correct (verified)
- **KAT order & no bypass:** `build_phase_manifest_in_memory` runs auth → `BS.require_known_answer_
  compatibility()` → commit validation → construction → `validate_fully_resolved_phase_manifest` → full hash.
  A KAT hex drift blocks the builder and the hash (no phase/hash/summary). The generator never calls the
  private unchecked core and never wraps `MI.reference_phase_manifest` (independent construction).
- **Phase builder:** counts 15/57/228 and 9/18/72; blocks/sessions/trials lists in canonical storage order
  (`ID.canonical_phase_{block,session,trial}_identities`); `execution_order_index` from
  `resolve_phase_execution_plan`; block state from `BS.resolved_block_state`; deep validation + hash pass;
  byte-equal to the reference fixture given the same provenance inputs.
- **Authorization gate & formal writer:** `smoke_only=False` and any formal/data/run flag are refused; the
  formal writer is LOCKED (raises before any filesystem effect even with a forged
  `GeneratorAuthorization(smoke_only=False, formal_manifest_generation_authorized=True)`); there is no second
  file-writing entry point in the module.
- **Baseline failure classification:** full `calibration_bias` suite at HEAD = 415 passed / 25 failed / 2
  xfailed; the 25 failures are **identical** to the pristine `c8ac0af` tree (intersection 25, **0 new**, 0
  baseline-only), all in Claude C historical audit/reaudit files (strict-xfail → XPASS flips), and **none
  imports/executes the generator**. A's smoke-report claim is accurate.

## BLOCKING issues (fix all before any formal gate)

### GEN-B-001 — `GeneratedPhase` scientific payload is mutable → validated-hash binding breakable
`GeneratedPhase(frozen=True)` only freezes attribute rebinding; `unsealed_manifest` holds the live `manifest`
dict. A caller can set `generated.unsealed_manifest["blocks"][0]["residual_value"] = 0.00777` or
`["trials"][0]["execution_order_index"] = 999` **after** validation/hash while `full_manifest_sha256` stays
fixed → the object no longer matches its own hash, with no re-validation. (§7) The binding must be a deep
immutable snapshot / deep-copy-on-access / controlled serialization.

### GEN-B-002 — `TrialExecutionEnvelope.public_trial_spec` is mutable → secret injectable after the leakage scan
`build_trial_execution_envelope` runs `assert_no_secret_in_public(public)` then stores the same mutable dict.
A caller can then `envelope.public_trial_spec["residual_bias"] = 0.009` — a denylist key injected **after** the
scan passed, so a downstream consumer of the envelope sees leaked secret state. (§11/§7) The public payload
must be immutable after the scan (or re-scanned at the boundary).

### GEN-B-003 — hardcoded `target_open_position: 0.20` is an unfrozen scientific task parameter
`build_trial_execution_envelope` writes `public_trial_spec["task"]["target_open_position"] = 0.20` as a bare
literal. The active preregistration only allowlists the **field name** `g.target_open_position`
(`STATIC_ALLOWLIST`); it freezes **no value** (`PRE` has no `TARGET_OPEN_POSITION`; `PRIMARY_SUCCESS` freezes
only `target_tolerance`). So the generator invents a scientific task value with no active source of truth, in
the reusable (future-production) envelope builder, and it is not marked as a synthetic smoke fixture. (§10) It
must import a frozen active value, or be explicitly a named synthetic-smoke fixture that cannot enter a
production envelope.

### GEN-B-004 — smoke build accepts real-looking commits and yields an undistinguished full hash
`build_phase_manifest_in_memory` validates `commits` only as 40-hex, so under `smoke_only=True` a caller may
pass real `CommitContext("1"*40, "2"*40, "3"*40)` and obtain a complete `full_manifest_sha256`. The hashed
manifest carries **no** smoke marker (only the wrapper `GeneratedPhase.smoke_only` flag), so a serialized
smoke hash is byte-identical in form to a future formal freeze — the distinction relies on caller discipline,
which §8 explicitly calls insufficient. (§8) Smoke auth must force placeholder commits, or the hashed
payload/anchor must carry an unbypassable smoke marker.

### GEN-B-005 — environment provenance is incomplete and silently lossy (no completeness contract)
`environment_versions` is a required top-level field covered by the full hash, but the deep validator has **no
content contract**: `EnvironmentVersionContext({})` builds and validates (empty `{}` accepted). The smoke CLI
`_env_versions()` records only `numpy` + `torch` with `except Exception: pass` (silently swallowing a torch
import/version failure) and omits python / OS-platform / Isaac status / GPU-driver. A future formal manifest
could thus be frozen with empty/partial/misleading provenance, and a missing field is undetectable. (§9/§13.2)
The provenance needs a required-key contract and must record failures explicitly (`torch unavailable`), not
drop them.

## NONBLOCKING issues

### GEN-B-006 — `SessionRunController.freeze_selection` offset domain is lax (does not affect ordering/hash)
`freeze_selection` accepts `False` (`float(False)=0.0` → in bank), the numeric string `"0.0"`
(`float("0.0")=0.0` → in bank), and an empty `evidence_hash`, and stores `float(selected_offset)` rather than
the exact frozen bank float; `session_complete` is reachable without any candidate-execution evidence. The
BLOCKING guarantees hold (selection can only freeze after `PROBE_COMPLETE`; candidate outcomes are rejected
from the selector), and the authoritative production selection (`confirmatory_v4_selection`) is already strict,
so this state-interface laxness is nonblocking. (§12) Recommend reusing the strict
`_canonical_candidate_offset` domain + a non-empty evidence contract.

### GEN-B-007 — smoke summary write is non-atomic (does not affect gate auditability)
`run_smoke` writes with `with open(out,"w"): json.dump(...)` (no temp + fsync + atomic replace), silently
overwrites an existing summary, has no write-complete marker/self-hash, and the returned dict carries
`_summary_path` that the on-disk file lacks. A truncated summary is invalid JSON (not mistakable as success),
so this is nonblocking. (§13.1) Recommend temp-write + `os.replace` + a completion marker.

## Sub-gate summary
KAT gate PASS · phase builder PASS · hash-binding immutability **FAIL (GEN-B-001)** · authorization gate PASS
· commit-context smoke semantics **FAIL (GEN-B-004)** · environment provenance **FAIL (GEN-B-005)** ·
public/secret isolation **FAIL (GEN-B-002)** · session-state interface PASS (nonblocking GEN-B-006) · smoke
artifact isolation PASS (nonblocking GEN-B-007) · formal-writer zero side-effect PASS · baseline failure
classification PASS.

`power_recertification_required = false` (no scientific input/design change; all findings are
implementation/provenance/immutability of the generator layer). Post-report new issues: CRITICAL_NEW_EVIDENCE
only.
