#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Precise handle localization: enumerate the child meshes of each appliance door/drawer link and
print their INDIVIDUAL world AABBs, so we can tell the door slab / drawer box apart from the graspable
handle bar -- and back out the true handle offset in the link body frame.

Runs with fabric OFF (SceneConfig.use_fabric=False) so USD/BBox reads return live physics poses, not
stale authored ones (the previous whole-link AABB was taken with fabric on).

Run:
    ./isaaclab.sh -p projects/franka_v1_skill_lab/scene_interface/debug_handle_meshes.py --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECTS_DIR))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()
    app_launcher = AppLauncher(args)
    rc = _run(args, app_launcher)
    app_launcher.app.close()
    return rc


def _run(args, app_launcher) -> int:
    import torch

    from franka_v1_skill_lab.scene import V1_BASE_TASK_ID
    from franka_v1_skill_lab.scene_interface import ResetMode, SceneConfig, SceneMode, SceneSession

    cfg = SceneConfig(
        mode=SceneMode.TEST, task_id=V1_BASE_TASK_ID, device=args.device, headless=True,
        use_fabric=False,                                # clean physics/USD poses
        enable_cameras=False, enable_fp=False, apply_saved_camera_offsets=False,
        free_microwave_door=False, load_latest_scene=True, add_microwave_stand=False,
        replace_microwave_with_fridge=True, lock_knife=True, spawn_init_markers=False,
        refine_handle_collisions=True, control_hz=50.0, reset_mode=ResetMode.STATIC, seed=args.seed,
    )
    session = SceneSession.launch(cfg, _app_launcher=app_launcher)
    env, provider = session.env, session.provider
    session.reset()

    import isaaclab.utils.math as math_utils
    from pxr import Usd, UsdGeom

    def P(*a):
        print(*a, flush=True)

    stage = env.unwrapped.sim.stage
    bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                          [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=True)

    def enumerate_meshes(link_path: str, label: str):
        prim = stage.GetPrimAtPath(link_path)
        P(f"\n----- {label}: meshes under {link_path} -----")
        if not prim.IsValid():
            P(f"  INVALID prim path"); return []
        rows = []
        for p in Usd.PrimRange(prim):
            if not (p.IsA(UsdGeom.Mesh) or p.IsA(UsdGeom.Gprim)):
                continue
            try:
                rng = bb.ComputeWorldBound(p).ComputeAlignedRange()
                mn, mx = rng.GetMin(), rng.GetMax()
                if rng.IsEmpty():
                    continue
                size = (mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2])
                ctr = ((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2)
                rows.append((str(p.GetPath()), mn, mx, size, ctr))
            except Exception as exc:
                P(f"  bbox failed {p.GetPath()}: {exc}")
        # sort by volume so the big slab/box is last, small handle parts first
        rows.sort(key=lambda r: r[3][0] * r[3][1] * r[3][2])
        for path, mn, mx, size, ctr in rows:
            name = path.split("/")[-1]
            P(f"  {name:<28s} ctr=({ctr[0]:.3f},{ctr[1]:.3f},{ctr[2]:.3f}) "
              f"size=({size[0]:.3f},{size[1]:.3f},{size[2]:.3f}) "
              f"min=({mn[0]:.3f},{mn[1]:.3f},{mn[2]:.3f}) max=({mx[0]:.3f},{mx[1]:.3f},{mx[2]:.3f})")
        return rows

    def link_pose(asset_name, link_name):
        a = provider.scene[asset_name]
        names = list(a.data.body_names)
        idx = next((i for i, n in enumerate(names) if n == link_name), None)
        if idx is None:
            idx = next((i for i, n in enumerate(names) if link_name in n), None)
        return a.data.body_pos_w[0, idx].clone(), a.data.body_quat_w[0, idx].clone()

    def offset_in_link(asset_name, link_name, world_pt):
        lp, lq = link_pose(asset_name, link_name)
        off, _ = math_utils.subtract_frame_transforms(
            lp.unsqueeze(0), lq.unsqueeze(0),
            torch.tensor([[float(world_pt[0]), float(world_pt[1]), float(world_pt[2])]], device=lp.device))
        return lp, lq, off[0]

    P("\n================= HANDLE MESH DIAGNOSTIC (fabric off) =================")
    robot = provider.scene["robot"]
    base = robot.data.root_pos_w[0]
    P(f"[robot] base=({base[0]:.3f},{base[1]:.3f},{base[2]:.3f})")

    # ---- FRIDGE door = link_1 ----
    lp, lq = link_pose("microwave", "link_1")
    P(f"\n[FRIDGE] link_1 body origin(live)=({lp[0]:.4f},{lp[1]:.4f},{lp[2]:.4f}) "
      f"quat=({lq[0]:.4f},{lq[1]:.4f},{lq[2]:.4f},{lq[3]:.4f})")
    rows = enumerate_meshes("/World/envs/env_0/Microwave/link_1", "FRIDGE door link_1")
    # heuristic handle = smallest-footprint mesh that is NOT the big slab; report its offset
    if rows:
        for path, mn, mx, size, ctr in rows[:4]:
            _, _, off = offset_in_link("microwave", "link_1", ctr)
            d = ((ctr[0] - float(base[0])) ** 2 + (ctr[1] - float(base[1])) ** 2) ** 0.5
            P(f"  -> '{path.split('/')[-1]}' link-offset=({off[0]:.4f},{off[1]:.4f},{off[2]:.4f}) "
              f"planar_dist_from_base={d:.3f}")

    # ---- CABINET top drawer = link_0 (joint_0) ----
    for ln in ("link_0", "link_2"):
        lp, lq = link_pose("cabinet", ln)
        P(f"\n[CABINET] {ln} body origin(live)=({lp[0]:.4f},{lp[1]:.4f},{lp[2]:.4f}) "
          f"quat=({lq[0]:.4f},{lq[1]:.4f},{lq[2]:.4f},{lq[3]:.4f})")
        rows = enumerate_meshes(f"/World/envs/env_0/Cabinet/{ln}", f"CABINET {ln}")
        if rows:
            for path, mn, mx, size, ctr in rows[:3]:
                _, _, off = offset_in_link("cabinet", ln, ctr)
                P(f"  -> '{path.split('/')[-1]}' link-offset=({off[0]:.4f},{off[1]:.4f},{off[2]:.4f})")

    # ---- REAL handle bar = the cabinet handle-proxy prims (children of the drawer links, sit on the
    #      actual bar). Compare to the config-derived position the skills target. ----
    P("\n----- REAL handle-proxy world pos vs config-derived (what the skills target) -----")
    handles = session.handle_poses_in_base()
    proxy_by_drawer = {
        "top_drawer": ("link_0", "TopHandleProxy"),
        "middle_drawer": ("link_2", "MiddleHandleProxy"),
        "bottom_drawer": ("link_1", "BottomHandleProxy"),
    }
    xf = UsdGeom.XformCache(Usd.TimeCode.Default())
    for name, (ln, proxy) in proxy_by_drawer.items():
        pp = f"/World/envs/env_0/Cabinet/{ln}/{proxy}"
        prim = stage.GetPrimAtPath(pp)
        real = None
        if prim.IsValid():
            t = xf.GetLocalToWorldTransform(prim).ExtractTranslation()
            real = (float(t[0]), float(t[1]), float(t[2]))
        cfg = handles.get(name, {}).get("position_world")
        if real and cfg:
            dx = ((real[0] - cfg[0]) ** 2 + (real[1] - cfg[1]) ** 2 + (real[2] - cfg[2]) ** 2) ** 0.5
            P(f"  {name:<14s} REAL=({real[0]:.3f},{real[1]:.3f},{real[2]:.3f}) "
              f"config=({cfg[0]:.3f},{cfg[1]:.3f},{cfg[2]:.3f})  OFF BY {dx*100:.1f}cm")
        else:
            P(f"  {name:<14s} real={'(proxy missing)' if real is None else real} config={cfg}")
    P("  (microwave_door config -> %s)" % str(handles.get("microwave_door", {}).get("position_world")))

    P("\n================= END =================\n")
    session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
