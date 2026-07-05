"""Smoke-only CLI for the confirmatory v4 generator.

    python -m deployment_calibration.offline_v2.calibration_bias.confirmatory_v4_generator_cli \
        --smoke-only --phase train_validation

KAT preflight -> in-memory build -> deterministic rebuild -> compare canonical JSON + full hash -> write a
SMOKE_ONLY summary (NO full manifest, NO Isaac, NO model). Refuses (zero side effects) without --smoke-only
or with --formal / --output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_identity as ID
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_block_state as BS
from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_generator as GEN

BANNER = "*** SMOKE_ONLY — NOT A FORMAL CONFIRMATORY ARTIFACT ***"


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def _smoke_dir() -> str:
    return os.path.join(_repo_root(), "artifacts", "smoke_only", "confirmatory_v4_generator")


def _env_versions() -> dict:
    import numpy as np
    out = {"numpy": np.__version__}
    try:
        import torch
        out["torch"] = torch.__version__
    except Exception:
        pass
    return out


def run_smoke(phase: str) -> dict:
    """KAT preflight, build twice, compare, write SMOKE_ONLY summary; return the summary dict."""
    kat = BS.require_known_answer_compatibility()              # explicit preflight (build re-runs it too)
    auth = GEN.GeneratorAuthorization(smoke_only=True)
    commits = GEN.smoke_placeholder_commits()
    env = GEN.EnvironmentVersionContext(_env_versions())

    g1 = GEN.build_phase_manifest_in_memory(phase, auth=auth, commits=commits, environment_versions=env)
    g2 = GEN.build_phase_manifest_in_memory(phase, auth=auth, commits=commits, environment_versions=env)
    json1 = ID.canonical_json(g1.unsealed_manifest)
    json2 = ID.canonical_json(g2.unsealed_manifest)
    rebuild_match = (json1 == json2) and (g1.full_manifest_sha256 == g2.full_manifest_sha256)

    summary = {
        "smoke_only": True, "formal": False, "not_for_confirmatory_use": True, "phase": phase,
        "counts": dict(g1.unsealed_manifest["counts"]),
        "kat_passed": bool(kat.get("passed")), "kat_version": kat.get("known_answer_version"),
        "kat_sampler_version": kat.get("sampler_version"), "kat_numpy_version": kat.get("numpy_version"),
        "kat_vectors_checked": kat.get("vectors_checked"),
        "deep_validation_passed": True,       # build_phase_manifest_in_memory raises if it does not
        "deterministic_rebuild_match": bool(rebuild_match),
        "storage_execution_orders_distinct": _orders_distinct(g1.unsealed_manifest),
        "commit_context_is_smoke_placeholder": GEN.commit_context_is_smoke_placeholder(commits),
        "not_a_formal_freeze": True,
        "formal_manifest_written": False, "confirmatory_records_written": False, "model_checkpoint_written": False,
        "isaac_launched": False,
        # in-memory hash is explicitly NOT a formal anchor
        "smoke_only_in_memory_hash": g1.full_manifest_sha256, "not_a_formal_manifest_anchor": True,
    }
    if not rebuild_match:
        raise RuntimeError("deterministic rebuild mismatch -> refusing (no anchor emitted)")

    os.makedirs(_smoke_dir(), exist_ok=True)
    out = os.path.join(_smoke_dir(), f"SMOKE_ONLY_{phase}_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=1)
    summary["_summary_path"] = out
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
          f"rebuild_match={s['deterministic_rebuild_match']} "
          f"orders_distinct={s['storage_execution_orders_distinct']} -> {s['_summary_path']}", flush=True)
    print(f"[smoke] formal_manifest_written={s['formal_manifest_written']} "
          f"confirmatory_records_written={s['confirmatory_records_written']} isaac_launched={s['isaac_launched']}",
          flush=True)
    print(BANNER, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
