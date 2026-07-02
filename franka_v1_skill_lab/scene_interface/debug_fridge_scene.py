#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Headless diagnostic for the latest (fridge) scene — characterize every skill's geometry in ONE run.

Builds the SAME deployed scene the skill-test UI uses (replace microwave -> fridge, refine handle
collisions, load scene_v1_latest), then prints the ground truth the skills need:

  * fridge "microwave" articulation: joint names / body names / joint limit,
    a scan of the door joint over several angles -> which link moves (the door), the world hinge
    position, the world hinge axis (from the link-quat delta), the sweep sign, and the door link's
    world AABB at closed (to place the handle grasp offset);
  * cabinet drawer handles + door handle world poses (read_handles_in_base) so we can confirm the
    cabinet rescale (z 0.62 -> 0.40) did not move the calibrated drawer handles off the bars;
  * cube_1/2/3 + knife world positions and the Franka base, to sanity-check grasp reachability.

Run:
    ./isaaclab.sh -p projects/franka_v1_skill_lab/scene_interface/debug_fridge_scene.py --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECTS_DIR = Path(__file__).resolve().parents[2]   # projects/
sys.path.insert(0, str(_PROJECTS_DIR))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no_fridge", action="store_true", help="keep microwave (debug the old asset).")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    rc = _run(args, app_launcher)
    simulation_app.close()
    return rc


def _run(args, app_launcher) -> int:
    import math

    import torch

    from franka_v1_skill_lab.scene import V1_BASE_TASK_ID
    from franka_v1_skill_lab.scene_interface import ResetMode, SceneConfig, SceneMode, SceneSession

    cfg = SceneConfig(
        mode=SceneMode.TEST, task_id=V1_BASE_TASK_ID, device=args.device, headless=True,
        enable_cameras=False, enable_fp=False, apply_saved_camera_offsets=False,
        free_microwave_door=False,                       # keep the door controllable for the scan
        load_latest_scene=True,
        add_microwave_stand=False,
        replace_microwave_with_fridge=not args.no_fridge,
        lock_knife=True,
        spawn_init_markers=False,
        refine_handle_collisions=True,                   # match the deployed handle collision
        control_hz=50.0, reset_mode=ResetMode.STATIC, seed=args.seed,
    )
    session = SceneSession.launch(cfg, _app_launcher=app_launcher)
    env, provider = session.env, session.provider
    session.reset()

    import isaaclab.utils.math as math_utils

    def P(*a):
        print(*a, flush=True)

    P("\n================= FRIDGE SCENE DIAGNOSTIC =================")

    scene = env.unwrapped.scene
    art = scene["microwave"]   # member name kept as "microwave"; now holds the fridge
    jnames = list(art.data.joint_names)
    bnames = list(art.data.body_names)
    P(f"[door-asset] member='microwave'  joint_names={jnames}  body_names={bnames}")
    try:
        lower = art.data.joint_pos_limits[0, :, 0].tolist()
        upper = art.data.joint_pos_limits[0, :, 1].tolist()
        P(f"[door-asset] joint limits: " + ", ".join(
            f"{n}=[{lo:.3f},{hi:.3f}]" for n, lo, hi in zip(jnames, lower, upper)))
    except Exception as exc:
        P(f"[door-asset] joint limit read failed: {exc}")
    P(f"[door-asset] default_joint_pos={art.data.default_joint_pos[0].tolist()}")

    # ---- door joint scan: write each angle, settle, read every body pose ----
    if len(jnames) == 0:
        P("[door-asset] no DOF joints -> not an articulated door; aborting scan.")
    else:
        jid = 0   # single revolute DOF expected; if multiple, scan the first
        if len(jnames) > 1:
            P(f"[door-asset] NOTE multiple DOFs; scanning joint[0]='{jnames[0]}'")
        lo = float(art.data.joint_pos_limits[0, jid, 0])
        hi = float(art.data.joint_pos_limits[0, jid, 1])
        angles = [a for a in (0.0, 30.0, 60.0, 90.0, 120.0) if math.radians(a) <= hi + 1e-3]
        if not angles:
            angles = [0.0, math.degrees(hi) * 0.5, math.degrees(hi)]
        P(f"[door-scan] scanning '{jnames[jid]}' over {angles} deg (limit [{lo:.3f},{hi:.3f}] rad)")

        body_pose_by_angle = {}
        for deg in angles:
            ang = math.radians(deg)
            q = art.data.joint_pos.clone()
            q[:, jid] = ang
            art.write_joint_state_to_sim(q, torch.zeros_like(q))
            for _ in range(8):
                st = provider.get_state()
                env.step(provider.make_hold_joint_action(st, 1.0))
            actual = float(art.data.joint_pos[0, jid])
            poses = {}
            for i, n in enumerate(bnames):
                p = art.data.body_pos_w[0, i]
                qd = art.data.body_quat_w[0, i]
                poses[n] = (p.clone(), qd.clone())
            body_pose_by_angle[deg] = (actual, poses)
            line = " | ".join(f"{n}:({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})" for n, (p, _) in poses.items())
            P(f"[door-scan] cmd={deg:5.1f}deg actual={math.degrees(actual):6.2f}deg  {line}")

        # which link moved the most between closed and max-open -> the door
        a0 = angles[0]
        a1 = angles[-1]
        moved = {}
        for n in bnames:
            p0 = body_pose_by_angle[a0][1][n][0]
            p1 = body_pose_by_angle[a1][1][n][0]
            moved[n] = float(torch.linalg.norm(p1 - p0))
        door_link = max(moved, key=moved.get)
        P(f"[door-scan] per-link displacement {a0}->{a1}deg: " +
          ", ".join(f"{n}={d:.3f}m" for n, d in moved.items()))
        P(f"[door-scan] => DOOR link (moves most) = '{door_link}'")

        # world hinge axis from the door-link quaternion delta (q1 * q0^-1 -> axis,angle)
        q0 = body_pose_by_angle[a0][1][door_link][1]
        q1 = body_pose_by_angle[a1][1][door_link][1]
        q_rel = math_utils.quat_mul(q1.unsqueeze(0), math_utils.quat_inv(q0.unsqueeze(0)))[0]
        w = float(max(-1.0, min(1.0, q_rel[0])))
        rel_ang = 2.0 * math.acos(w)
        axis = q_rel[1:4]
        na = float(torch.linalg.norm(axis))
        axis = (axis / na) if na > 1e-6 else axis
        P(f"[door-scan] door-link rotation {a0}->{a1}deg = {math.degrees(rel_ang):.1f}deg "
          f"world axis=({axis[0]:.3f},{axis[1]:.3f},{axis[2]:.3f})  (≈(0,0,±1) => vertical hinge)")

        # hinge position: door-link body origin at closed (if body frame sits on the joint) +
        # also the static (non-door) link pos for reference
        p_door0 = body_pose_by_angle[a0][1][door_link][0]
        P(f"[door-scan] door-link body origin @closed = ({p_door0[0]:.3f},{p_door0[1]:.3f},{p_door0[2]:.3f})  "
          f"(skill uses link body origin as hinge)")
        for n in bnames:
            if n != door_link:
                pn = body_pose_by_angle[a0][1][n][0]
                P(f"[door-scan]   other body '{n}' @closed = ({pn[0]:.3f},{pn[1]:.3f},{pn[2]:.3f})")

        # door-link world AABB at closed -> locate the graspable free edge for the handle offset
        try:
            from pxr import Usd, UsdGeom

            # restore closed for the bbox read
            q = art.data.joint_pos.clone(); q[:, jid] = math.radians(a0)
            art.write_joint_state_to_sim(q, torch.zeros_like(q))
            for _ in range(8):
                st = provider.get_state(); env.step(provider.make_hold_joint_action(st, 1.0))
            stage = env.unwrapped.sim.stage
            link_path = f"/World/envs/env_0/Microwave/{door_link}"
            prim = stage.GetPrimAtPath(link_path)
            if prim.IsValid():
                bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]).ComputeWorldBound(prim)
                rng = bbox.ComputeAlignedRange()
                mn, mx = rng.GetMin(), rng.GetMax()
                P(f"[door-scan] door-link '{door_link}' world AABB @closed: "
                  f"min=({mn[0]:.3f},{mn[1]:.3f},{mn[2]:.3f}) max=({mx[0]:.3f},{mx[1]:.3f},{mx[2]:.3f})")
                lp = p_door0; lq = body_pose_by_angle[a0][1][door_link][1]
                P(f"[door-scan] door-link body origin=({lp[0]:.4f},{lp[1]:.4f},{lp[2]:.4f}) "
                  f"quat_wxyz=({lq[0]:.4f},{lq[1]:.4f},{lq[2]:.4f},{lq[3]:.4f})")
                # door front-face normal candidate: R(link_quat) @ local -Z (microwave convention)
                for lbl, lv in (("R@-Z", [0, 0, -1]), ("R@+X", [1, 0, 0]), ("R@+Y", [0, 1, 0])):
                    v = math_utils.quat_apply(lq.unsqueeze(0),
                                              torch.tensor([[float(x) for x in lv]], device=lp.device))[0]
                    P(f"[door-scan]   link {lbl} -> world=({v[0]:.3f},{v[1]:.3f},{v[2]:.3f})")
                # back out link-frame offset for several free-edge handle world candidates.
                # hinge x≈min_x; free edge = max_x side; robot-facing face = max_y; grasp mid-height.
                mid_z = (mn[2] + mx[2]) * 0.5
                cands = {
                    "freeedge_face_mid": (mx[0] - 0.03, mx[1], mid_z),
                    "freeedge_face_lower": (mx[0] - 0.03, mx[1], mn[2] + 0.30),
                    "freeedge_protrude": (mx[0] - 0.03, mx[1] + 0.02, mid_z),
                }
                for name, hw in cands.items():
                    handle_world = torch.tensor([float(hw[0]), float(hw[1]), float(hw[2])], device=lp.device)
                    off, _ = math_utils.subtract_frame_transforms(
                        lp.unsqueeze(0), lq.unsqueeze(0), handle_world.unsqueeze(0))
                    P(f"[door-scan] handle '{name}' world=({hw[0]:.3f},{hw[1]:.3f},{hw[2]:.3f}) "
                      f"-> link offset=({off[0,0]:.4f},{off[0,1]:.4f},{off[0,2]:.4f})")
            else:
                P(f"[door-scan] door-link prim not found at {link_path} (instanced?) -> skip AABB")
        except Exception as exc:
            P(f"[door-scan] AABB read failed: {exc}")

    # ---- all handles in world (drawers / door) + cube/knife reachability ----
    P("\n----- handles (world) + grasp targets -----")
    handles = session.handle_poses_in_base()
    robot = scene["robot"]
    base = robot.data.root_pos_w[0]
    P(f"[robot] base_pos_w=({base[0]:.3f},{base[1]:.3f},{base[2]:.3f})")
    for name, h in handles.items():
        pw = h.get("position_world")
        if pw is None:
            continue
        d = ((pw[0] - float(base[0])) ** 2 + (pw[1] - float(base[1])) ** 2) ** 0.5
        P(f"[handle] {name:<16s} world=({pw[0]:.3f},{pw[1]:.3f},{pw[2]:.3f}) "
          f"calib={h.get('calibrated')} func={h.get('functional')} planar_dist_from_base={d:.3f}")
    st = provider.get_state()
    for nm in ("cube_1", "cube_2", "cube_3", "knife"):
        obj = st.objects.get(nm)
        if obj is None:
            continue
        p = obj.pose.pos_w
        d = ((float(p[0]) - float(base[0])) ** 2 + (float(p[1]) - float(base[1])) ** 2) ** 0.5
        P(f"[object] {nm:<8s} world=({float(p[0]):.3f},{float(p[1]):.3f},{float(p[2]):.3f}) "
          f"planar_dist_from_base={d:.3f}")

    P("\n================= END DIAGNOSTIC =================\n")
    session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
