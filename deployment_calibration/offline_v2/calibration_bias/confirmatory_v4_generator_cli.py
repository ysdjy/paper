"""Smoke-only CLI for the confirmatory v4 generator.

    python -m deployment_calibration.offline_v2.calibration_bias.confirmatory_v4_generator_cli \
        --smoke-only --phase train_validation

KAT preflight -> full 10-key smoke provenance (GEN-B-005) -> in-memory build -> deterministic rebuild ->
compare canonical JSON + full hash -> ATOMIC SMOKE_ONLY summary (GEN-B-007: temp+fsync+os.replace,
read-back, idempotent, collision-guarded). NO full manifest, NO Isaac, NO model. Refuses (zero side
effects) without --smoke-only or with --formal / --output.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile

from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_identity as ID
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_block_state as BS
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_generator as GEN

BANNER = "*** SMOKE_ONLY — NOT A FORMAL CONFIRMATORY ARTIFACT ***"
SUMMARY_SCHEMA_VERSION = "confirmatory_v4_generator_smoke_summary_v1"


class SmokeArtifactCollisionError(RuntimeError):
    verdict = "GENERATOR_SMOKE_ARTIFACT_COLLISION"


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def _smoke_dir() -> str:
    return os.path.join(_repo_root(), "artifacts", "smoke_only", "confirmatory_v4_generator")


def _env_versions() -> "GEN.EnvironmentVersionContext":
    """GEN-B-005: the EXACT 10-key smoke provenance (explicit; no silent drop; no Isaac import/launch)."""
    import numpy as np
    try:
        import torch
        torch_version = str(torch.__version__)
    except Exception as exc:                                   # explicit, never silent
        torch_version = f"UNAVAILABLE:{type(exc).__name__}"
    values = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch_version,
        "os_system": platform.system() or "UNKNOWN",
        "os_release": platform.release() or "UNKNOWN",
        "machine": platform.machine() or "UNKNOWN",
        "isaac_status": "NOT_IMPORTED_NOT_LAUNCHED",
        "gpu_status": "NOT_USED_CPU_SMOKE",
        "execution_mode": "SMOKE_ONLY",
    }
    env = GEN.EnvironmentVersionContext(values)
    GEN.validate_smoke_environment_provenance(env)             # enforce the frozen 10-key contract
    return env


def _atomic_write_smoke_summary(path: str, payload: dict) -> None:
    """GEN-B-007: same-dir temp -> flush -> fsync -> os.replace -> read-back verify; idempotent on identical
    content; SmokeArtifactCollisionError on a differing existing file (never silent overwrite); temp cleaned
    on any failure. The parent dir is created ONLY here (after all KAT/build/rebuild succeeded)."""
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    if os.path.exists(path):
        existing = open(path, "r").read()
        try:
            same = json.loads(existing) == payload
        except Exception:
            same = False
        if same:
            return                                             # idempotent success
        raise SmokeArtifactCollisionError(f"refusing to overwrite differing smoke summary at {path}")
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".SMOKE_ONLY_tmp_", dir=d)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(body)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        tmp = None
        try:
            dfd = os.open(d, os.O_DIRECTORY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except Exception:
            pass
        readback = open(path, "r").read()
        if json.loads(readback) != payload:
            raise RuntimeError("smoke summary read-back mismatch")
    finally:
        if tmp is not None and os.path.exists(tmp):
            os.remove(tmp)


def build_smoke_summary(phase: str) -> dict:
    """KAT preflight -> full provenance -> build twice -> compare -> return the summary dict (== disk)."""
    kat = BS.require_known_answer_compatibility()
    auth = GEN.GeneratorAuthorization(smoke_only=True)
    commits = GEN.smoke_placeholder_commits()
    env = _env_versions()
    g1 = GEN.build_phase_manifest_in_memory(phase, auth=auth, commits=commits, environment_versions=env)
    g2 = GEN.build_phase_manifest_in_memory(phase, auth=auth, commits=commits, environment_versions=env)
    rebuild_match = (g1.canonical_manifest_json == g2.canonical_manifest_json
                     and g1.full_manifest_sha256 == g2.full_manifest_sha256)
    if not rebuild_match:
        raise RuntimeError("deterministic rebuild mismatch -> refusing (no anchor emitted)")
    m = g1.unsealed_manifest
    return {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "smoke_only": True, "formal": False, "not_for_confirmatory_use": True, "not_a_formal_freeze": True,
        "phase": phase, "counts": dict(m["counts"]),
        "kat_passed": bool(kat.get("passed")), "kat_version": kat.get("known_answer_version"),
        "kat_sampler_version": kat.get("sampler_version"), "kat_numpy_version": kat.get("numpy_version"),
        "kat_vectors_checked": kat.get("vectors_checked"),
        "deep_validation_passed": True, "deterministic_rebuild_match": True,
        "storage_execution_orders_distinct": _orders_distinct(m),
        "commit_context_is_smoke_placeholder": GEN.commit_context_is_smoke_placeholder(commits),
        "environment_versions": dict(env.values),
        "smoke_only_in_memory_hash": g1.full_manifest_sha256, "not_a_formal_manifest_anchor": True,
        "formal_manifest_written": False, "confirmatory_records_written": False,
        "model_checkpoint_written": False, "isaac_launched": False,
        "summary_path": os.path.join(_smoke_dir(), f"SMOKE_ONLY_{phase}_summary.json"),
        "write_complete": True,
    }


def run_smoke(phase: str) -> dict:
    """Build the summary, then ATOMICALLY write it. Returns the summary dict == the on-disk JSON."""
    summary = build_smoke_summary(phase)
    _atomic_write_smoke_summary(summary["summary_path"], summary)
    return summary


def _orders_distinct(m: dict) -> bool:
    storage = [t["canonical_trial_identity"] for t in m["trials"]]
    by_exec = [t["canonical_trial_identity"] for t in sorted(m["trials"], key=lambda t: t["execution_order_index"])]
    return storage != by_exec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="confirmatory v4 generator smoke (smoke-only)")
    ap.add_argument("--smoke-only", action="store_true", help="REQUIRED; there is no formal mode")
    ap.add_argument("--phase", choices=["train_validation", "test"], required=True)
    ap.add_argument("--formal", action="store_true", help="(refused)")
    ap.add_argument("--output", default=None, help="(refused)")
    args = ap.parse_args(argv)

    if args.formal or args.output is not None:
        print(BANNER); print("[REFUSED] formal / --output not authorized (smoke-only). Zero side effects.",
                             file=sys.stderr)
        return 2
    if not args.smoke_only:
        print(BANNER); print("[REFUSED] --smoke-only is required; no default formal mode. Zero side effects.",
                             file=sys.stderr)
        return 2

    print(BANNER, flush=True)
    try:
        s = run_smoke(args.phase)
    except Exception as ex:
        print(f"[SMOKE FAILED] {type(ex).__name__}: {ex}", file=sys.stderr)
        print(BANNER)
        return 1
    print(f"[smoke] phase={s['phase']} counts={s['counts']} kat={s['kat_version']}"
          f"({s['kat_vectors_checked']}) deep_valid={s['deep_validation_passed']} "
          f"rebuild_match={s['deterministic_rebuild_match']} orders_distinct={s['storage_execution_orders_distinct']} "
          f"write_complete={s['write_complete']} -> {s['summary_path']}", flush=True)
    print(f"[smoke] env_keys={sorted(s['environment_versions'])} "
          f"formal_manifest_written={s['formal_manifest_written']} isaac_launched={s['isaac_launched']}", flush=True)
    print(BANNER, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
