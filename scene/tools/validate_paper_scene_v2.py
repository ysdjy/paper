"""Validate scene/paper_scene_v2/ self-consistency + single-source + external assets.

Pure-python (no Isaac). Exit 0 if all checks pass, non-zero otherwise. Run:
    python projects/paper/scene/tools/validate_paper_scene_v2.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
OUT = _PAPER / "scene" / "paper_scene_v2"
ACTIVE = _PAPER / "franka_v1_skill_lab" / "scene" / "saved_scenes" / "v1_active"
_ROOT = _PAPER.parents[1]


def _load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    fails, warns = [], []

    def check(cond, msg):
        (None if cond else fails.append(msg))

    for f in ("scene_manifest.json", "object_states.json", "mechanism_registry.json",
              "grasp_poses.json", "camera_registry.json", "external_assets.json", "version.json"):
        check((OUT / f).exists(), f"missing {f}")
    if fails:
        for m in fails:
            print("FAIL:", m)
        return 1

    manifest = _load(OUT / "scene_manifest.json")
    objs = _load(OUT / "object_states.json")["objects"]
    mech = _load(OUT / "mechanism_registry.json")["mechanisms"]
    gp2 = _load(OUT / "grasp_poses.json")["poses"]
    ext = _load(OUT / "external_assets.json")["assets"]
    ver = _load(OUT / "version.json")

    # 1) version has a commit
    check(bool(ver.get("source_commit")), "version.json missing source_commit")

    # 2) enabled members present in object_states
    for name in manifest.get("enabled_members", []):
        check(name in objs, f"enabled member '{name}' not in object_states")

    # 3) each mechanism resolves + member present + handle pose present
    for dn, r in mech.items():
        check(r.get("present_in_scene") is True, f"mechanism {dn} not present in scene")
        check(r.get("member") in objs, f"mechanism {dn} member '{r.get('member')}' not in object_states")
        check(bool(r.get("handle_local_pos")) and bool(r.get("handle_local_quat")),
              f"mechanism {dn} missing handle local pose")
        check(r.get("joint_damping") is not None, f"mechanism {dn} missing joint_damping (hidden axis baseline)")

    # 4) SINGLE SOURCE: paper_scene_v2/grasp_poses handles == active grasp_poses.json handles
    active_gp = _load(ACTIVE / "grasp_poses.json").get("poses", {})
    for hname, e in gp2.items():
        a = active_gp.get(hname)
        check(a is not None, f"handle {hname} not in active grasp_poses.json (single-source drift)")
        if a is not None:
            check(a.get("pos") == e.get("pos") and a.get("quat") == e.get("quat"),
                  f"handle {hname} pose DIVERGED from active source (pos/quat mismatch)")

    # 5) mechanism handle_local matches paper grasp_poses single source
    for dn, r in mech.items():
        hn = f"handle_{dn}"
        if hn in gp2:
            check(gp2[hn].get("pos") == r.get("handle_local_pos"),
                  f"mechanism {dn} handle_local_pos != grasp_poses[{hn}]")

    # 6) external assets exist + hashes match
    for rel, e in ext.items():
        if not e.get("exists"):
            warns.append(f"external asset absent: {rel}")
            continue
        ap = Path(e["abs_path"])
        if not ap.exists():
            fails.append(f"external asset path gone: {ap}")
            continue
        if isinstance(e.get("sha256"), str) and len(e["sha256"]) == 64:
            check(_sha(ap) == e["sha256"], f"external asset hash changed: {rel}")

    for m in warns:
        print("WARN:", m)
    for m in fails:
        print("FAIL:", m)
    if fails:
        print(f"\n[validate_paper_scene_v2] {len(fails)} FAILURES, {len(warns)} warnings")
        return 1
    print(f"[validate_paper_scene_v2] PASS ({len(mech)} mechanisms, {len(manifest.get('enabled_members', []))} "
          f"enabled members, {len(gp2)} single-source handles, {len(warns)} warnings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
