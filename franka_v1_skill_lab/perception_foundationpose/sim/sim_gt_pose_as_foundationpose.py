# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Stage-1 perception: sim ground-truth pose dressed up as FoundationPose output.

STATUS: ready (sim read needs Isaac; --demo mode runs anywhere).

Reads the world-frame pose of the V1 tracked objects (cube_1/2/3, knife,
cabinet, microwave) from a live IsaacLab scene and emits the unified
``PoseResult`` (list of ``PoseEntry``) so that downstream consumers — the pi0.5
observation adapter and the skill-runtime debug pose source — can be developed
and validated against the SAME interface the real FoundationPose runner will
later satisfy. confidence is always 1.0 (it's ground truth).

Two ways in:
  * ``poses_from_provider(provider)`` — pass a SceneStateProvider (or anything
    exposing ``get_state().objects[name].pose`` as [x,y,z, qw,qx,qy,qz]); used
    by Isaac entry scripts.
  * ``--demo`` CLI — print a synthetic PoseResult with no Isaac, used by the
    smoke test.

NOTE on quaternion order: IsaacLab object poses are [x,y,z, qw,qx,qy,qz]
(w-first). The unified PoseEntry uses [x,y,z,w] (w-last), matching the
FoundationPose / ROS convention. ``_wxyz_to_xyzw`` does the conversion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECTS_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECTS_DIR))

from franka_v1_skill_lab.perception_foundationpose.sim.pose_entry import (  # noqa: E402
    PoseEntry,
    PoseResult,
)

# Default objects to track (subset of the V1 contract that has a graspable/known body).
DEFAULT_TRACKED = ("cube_1", "cube_2", "cube_3", "knife", "cabinet", "microwave")


def _wxyz_to_xyzw(q: list[float]) -> list[float]:
    """[w,x,y,z] -> [x,y,z,w]."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return [x, y, z, w]


def poses_from_provider(
    provider,
    tracked: tuple[str, ...] = DEFAULT_TRACKED,
    mesh_paths: dict[str, str] | None = None,
) -> PoseResult:
    """Build a PoseResult from a SceneStateProvider-like object.

    ``provider.get_state().objects[name].pose`` is expected to be a tensor or
    list ``[x, y, z, qw, qx, qy, qz]`` (IsaacLab w-first convention).
    """
    mesh_paths = mesh_paths or {}
    state = provider.get_state()
    result = PoseResult(source="sim_gt", frame="world")
    for name in tracked:
        obj = state.objects.get(name)
        if obj is None:
            continue
        pose = obj.pose
        try:
            pose = pose.detach().cpu().tolist()
        except AttributeError:
            pose = list(pose)
        position = [float(pose[0]), float(pose[1]), float(pose[2])]
        quat_xyzw = _wxyz_to_xyzw([float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])])
        result.objects.append(
            PoseEntry(
                name=name,
                position=position,
                quat=quat_xyzw,
                confidence=1.0,
                pose_in_camera=None,
                mesh_path=mesh_paths.get(name),
                mask_path=None,
            )
        )
    return result


def _demo_result() -> PoseResult:
    """A synthetic PoseResult with no Isaac dependency (smoke test fixture)."""
    fake = {
        "cube_1": ([0.45, 0.05, 0.02], [0, 0, 0, 1]),
        "cube_2": ([0.50, 0.10, 0.02], [0, 0, 0, 1]),
        "cube_3": ([0.55, 0.00, 0.02], [0, 0, 0, 1]),
        "knife": ([0.40, -0.10, 0.02], [0, 0, 0.3827, 0.9239]),
        "cabinet": ([0.80, 0.00, 0.20], [0, 0, 0, 1]),
        "microwave": ([0.30, 0.40, 0.15], [0, 0, 1, 0]),
    }
    result = PoseResult(source="mock", frame="world")
    for name, (pos, quat) in fake.items():
        result.objects.append(PoseEntry(name=name, position=pos, quat=quat, confidence=1.0))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Sim GT pose -> FoundationPose-like output.")
    ap.add_argument("--demo", action="store_true", help="Emit a synthetic PoseResult (no Isaac).")
    ap.add_argument("--out", default=None, help="Write the PoseResult JSON to this path.")
    args = ap.parse_args()

    if not args.demo:
        print(
            "Live-scene mode requires an IsaacLab env; call poses_from_provider(provider) from an "
            "Isaac entry script, or run with --demo for the offline contract check.",
            file=sys.stderr,
        )
        return 2

    result = _demo_result()
    payload = json.dumps(result.to_dict(), indent=2)
    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
        print(f"[sim_gt] wrote {args.out}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
