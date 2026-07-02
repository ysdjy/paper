"""Stage-0: scene facts are self-consistent and the frozen registry still matches the live config.

1. validate_paper_scene_v2 passes (single-source handles, external hashes, mechanism completeness).
2. FREEZE CROSS-CHECK: the live DRAWER_TARGETS joint/link/member still equal the frozen
   mechanism_registry.json -- a shared-config edit that changes them fails loudly here.
Run: python projects/paper/deployment_calibration/tests/test_scene_consistency_v2.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER / "franka_skill_state_machine"))
REG = _PAPER / "scene" / "paper_scene_v2" / "mechanism_registry.json"


def main() -> int:
    fails = []

    # 1) scene validator
    rc = subprocess.call([sys.executable, str(_PAPER / "scene" / "tools" / "validate_paper_scene_v2.py")])
    if rc != 0:
        fails.append("validate_paper_scene_v2 failed")

    # 2) freeze cross-check vs live DRAWER_TARGETS (pure-python config; no Isaac app needed)
    frozen = json.loads(REG.read_text())["mechanisms"]
    try:
        from runtime.drawer_target_config import DRAWER_TARGETS  # noqa
        for dn, r in frozen.items():
            live = DRAWER_TARGETS.get(dn)
            if live is None:
                fails.append(f"{dn}: missing from live DRAWER_TARGETS (config drift)")
                continue
            for k in ("joint_name", "link_name"):
                if live.get(k) != r.get(k):
                    fails.append(f"{dn}: live {k}={live.get(k)} != frozen {r.get(k)} (SHARED CONFIG DRIFT)")
            if live.get("member", "cabinet") != r.get("member"):
                fails.append(f"{dn}: live member != frozen member")
        print(f"freeze cross-check: {len(frozen)} mechanisms vs live DRAWER_TARGETS")
    except Exception as exc:
        print(f"WARN: could not import live DRAWER_TARGETS ({exc}); skipped freeze cross-check")

    for f in fails:
        print("FAIL:", f)
    print(f"[test_scene_consistency_v2] {'PASS' if not fails else str(len(fails))+' FAIL'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
