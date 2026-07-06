"""Preflight-only CLI for the confirmatory v4 formal manifest writer (offline; no Isaac, no GPU).

This CLI CANNOT generate a formal manifest. It only verifies that the current Claude C gate authorizes writer
*implementation* while keeping actual generation *locked*, and that the generation loader refuses the current
gate file. There is no `--force`, no environment-variable escape, and no code path that writes a formal
artifact. Output status on success: FORMAL_WRITER_IMPLEMENTED_GENERATION_LOCKED.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from deployment_calibration.offline_v2.calibration_bias import confirmatory_v4_formal_manifest_writer as FW

BANNER = "*** FORMAL WRITER PREFLIGHT — IMPLEMENTATION ONLY, GENERATION LOCKED ***"
C_GATE_RELPATH = "docs/offline_v2/calibration_bias/confirmatory_v4_generator_smoke_final_audit_c.json"


def _repo_root() -> Path:
    # module lives at <root>/deployment_calibration/offline_v2/calibration_bias/<file>
    # parents: [0]=calibration_bias [1]=offline_v2 [2]=deployment_calibration [3]=<repo root>
    return Path(__file__).resolve().parents[3]


def preflight(c_gate_path: Path) -> dict:
    with open(c_gate_path, "r", encoding="utf-8") as fh:
        gate = json.load(fh)
    writer_impl = gate.get("authorizes_formal_manifest_writer_implementation")
    gen = gate.get("authorizes_formal_manifest_generation")
    verdict = gate.get("verdict")
    if writer_impl is not True:
        raise SystemExit("FORMAL_WRITER_IMPLEMENTATION_BLOCKED_MISSING_C_GATE: "
                         "authorizes_formal_manifest_writer_implementation != true")
    if gen is not False:
        raise SystemExit("FORMAL_WRITER_PREFLIGHT_ABORT: current gate unexpectedly authorizes generation")

    # prove the generation loader REFUSES the current gate file (writer stays locked)
    loader_rejected = False
    reason = ""
    try:
        FW.load_and_verify_formal_manifest_authorization(
            str(c_gate_path), authorization_commit="0" * 40, expected_writer_commit="0" * 40)
    except (FW.FormalManifestGenerationNotAuthorized, FW.FormalAuthorizationIntegrityError) as exc:
        loader_rejected = True
        reason = f"{type(exc).__name__}: {exc}"
    if not loader_rejected:
        raise SystemExit("FORMAL_WRITER_PREFLIGHT_ABORT: generation loader did NOT reject the current C gate")

    return {"verdict": verdict, "authorizes_formal_manifest_writer_implementation": True,
            "authorizes_formal_manifest_generation": False,
            "generation_loader_rejected_current_gate": True,
            "generation_loader_rejection_reason": reason,
            "status": "FORMAL_WRITER_IMPLEMENTED_GENERATION_LOCKED"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Confirmatory v4 formal writer preflight (implementation only).")
    parser.add_argument("--preflight-only", action="store_true",
                        help="verify writer implementation is authorized and generation is locked (the only mode)")
    parser.add_argument("--c-gate", default=None, help="path to the Claude C generator-smoke final gate JSON")
    args = parser.parse_args(argv)

    print(BANNER)
    if not args.preflight_only:
        print("refusing: this CLI only supports --preflight-only; formal manifest generation is NOT authorized.")
        print(BANNER)
        return 2
    gate_path = Path(args.c_gate) if args.c_gate else _repo_root() / C_GATE_RELPATH
    result = preflight(gate_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(result["status"])
    print(BANNER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
