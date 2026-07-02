"""Export the paper drawer-experiment scene (v2) from a LIVE runtime capture.

Builds the SAME scene the GUI test platform builds (via SceneSession, headless), lets it
settle, then dumps the authoritative runtime ground-truth into scene/paper_scene_v2/:

    scene_manifest.json     # authoritative: enabled subset + all fields the paper reads
    object_states.json      # every scene member's runtime root pose / scale / joints / actuators
    mechanism_registry.json # per drawer: member/joint/link/handle/actuator (member-aware)
    grasp_poses.json        # drawer handle link-local poses copied from the active single source
    camera_registry.json    # front camera registered (disabled by default for drawer profile)
    external_assets.json     # external USD paths + size + mtime + sha256
    version.json            # scene_version, source_commit, dirty flag, timestamp
    README.md

Does NOT assume the old captured_scene_v1.json is current -- it re-captures live.
Nothing is overwritten outside scene/paper_scene_v2/. Run headless:

    ./isaaclab.sh -p projects/paper/scene/tools/export_paper_scene_v2.py --headless
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]          # projects/paper
_ROOT = _PAPER.parents[1]                              # IsaacLab root
sys.path.insert(0, str(_PAPER))
sys.path.insert(0, str(_PAPER / "franka_skill_state_machine"))

from franka_v1_skill_lab.scene import V1_BASE_TASK_ID          # noqa: E402
from franka_v1_skill_lab.scene_interface import ResetMode, SceneConfig, SceneMode  # noqa: E402

OUT = _PAPER / "scene" / "paper_scene_v2"
ACTIVE = _PAPER / "franka_v1_skill_lab" / "scene" / "saved_scenes" / "v1_active"
SCENE_VERSION = "paper_scene_v2"

# Which members the paper DRAWER experiment enables (others kept in scene but marked disabled).
ENABLED = {"robot", "cabinet", "sektion_cabinet"}
DISABLED_BUT_KEPT = {"coffee_machine"}   # revolute extension, off by default

from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--settle_steps", type=int, default=40)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _git(*a) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(_PAPER), *a], text=True).strip()
    except Exception:
        return ""


def _fl(t):
    return [round(float(v), 6) for v in t.reshape(-1).tolist()]


def main() -> int:
    import torch  # noqa: F401
    from franka_v1_skill_lab.scene_interface import SceneSession

    # minimal drawer scene: no cameras, load latest saved scene, drop microwave+dishwasher.
    cfg = SceneConfig(
        mode=SceneMode.TEST, task_id=V1_BASE_TASK_ID, device=args.device, headless=True,
        enable_cameras=False, enable_fp=False, free_microwave_door=False,
        load_latest_scene=True, add_microwave_stand=False, replace_microwave_with_fridge=False,
        lock_knife=True, enable_collision_monitor=False, spawn_init_markers=False,
        refine_handle_collisions=True, apply_saved_camera_offsets=False,
        reset_mode=ResetMode.STATIC, seed=1, disable_auto_reset=True,
        exclude_members=("microwave", "dishwasher"), hidden_members=(),
    )
    session = SceneSession.launch(cfg, _app_launcher=app_launcher)
    env, env_cfg = session.env, session.env_cfg
    scene = env.unwrapped.scene
    env.reset(seed=1)
    for _ in range(args.settle_steps):
        try:
            provider = session.provider
            env.step(provider.make_hold_joint_action(provider.get_state(), 1.0))
        except Exception:
            env.step(env.action_space.sample() * 0.0)

    OUT.mkdir(parents=True, exist_ok=True)

    # ---- object_states: every articulation/rigid member ----
    def scale_of(name):
        m = getattr(env_cfg.scene, name, None)
        sp = getattr(m, "spawn", None)
        return list(getattr(sp, "scale", (1.0, 1.0, 1.0))) if sp is not None else [1.0, 1.0, 1.0]

    objs = {}
    for name in scene.keys():
        try:
            a = scene[name]
            data = getattr(a, "data", None)
            if data is None or not hasattr(data, "root_pos_w"):
                continue
            e = {"root_pos_w": _fl(data.root_pos_w[0]), "root_quat_w": _fl(data.root_quat_w[0]),
                 "scale": scale_of(name)}
            jn = list(getattr(data, "joint_names", []) or [])
            if jn:
                e["joint_names"] = jn
                e["joint_pos"] = _fl(data.joint_pos[0])
                e["default_joint_pos"] = _fl(data.default_joint_pos[0])
                for k in ("joint_stiffness", "joint_damping"):
                    if hasattr(data, k):
                        e[k] = _fl(getattr(data, k)[0])
            bn = list(getattr(data, "body_names", []) or [])
            if bn:
                e["body_names"] = bn
            objs[name] = e
        except Exception as exc:
            objs[name] = {"error": str(exc)}
    json.dump({"scene_version": SCENE_VERSION, "note": "runtime ground-truth capture (post reset+settle)",
               "objects": objs}, open(OUT / "object_states.json", "w"), indent=1)

    # ---- mechanism_registry: per drawer (member-aware) ----
    import isaaclab.utils.math as mu  # noqa
    from runtime.drawer_target_config import DRAWER_TARGETS
    active_gp = json.loads((ACTIVE / "grasp_poses.json").read_text()).get("poses", {})
    mech = {}
    for dn, dc in DRAWER_TARGETS.items():
        member = dc.get("member", "cabinet")
        rec = {"drawer_name": dn, "member": member, "joint_name": dc.get("joint_name"),
               "link_name": dc.get("link_name"), "handle_frame": dc.get("handle_frame"),
               "config_handle_offset": list(dc.get("handle_offset", (0, 0, 0)))}
        gp = active_gp.get(f"handle_{dn}")
        if gp:
            rec["handle_pose_source"] = "grasp_poses.json (single source)"
            rec["handle_local_pos"] = gp.get("pos")
            rec["handle_local_quat"] = gp.get("quat")
            rec["handle_link"] = gp.get("link")
        else:
            rec["handle_pose_source"] = "MISSING in grasp_poses.json"
        try:
            a = scene[member]
            jn = list(a.data.joint_names)
            bn = list(a.data.body_names)
            if dc.get("joint_name") in jn:
                ji = jn.index(dc["joint_name"])
                rec["joint_index"] = ji
                rec["init_joint_pos"] = round(float(a.data.joint_pos[0, ji]), 6)
                for k in ("joint_stiffness", "joint_damping"):
                    if hasattr(a.data, k):
                        rec[k] = round(float(getattr(a.data, k)[0, ji]), 6)
            li = next((i for i, n in enumerate(bn) if n == dc.get("link_name")), None)
            if li is not None:
                rec["link_index"] = li
                rec["link_world_pos"] = _fl(a.data.body_pos_w[0, li])
                rec["link_world_quat"] = _fl(a.data.body_quat_w[0, li])
                if gp and gp.get("pos") and gp.get("quat"):
                    lp = torch.tensor([gp["pos"]], dtype=torch.float32, device=a.data.body_pos_w.device)
                    lq = torch.tensor([gp["quat"]], dtype=torch.float32, device=a.data.body_pos_w.device)
                    wp, wq = mu.combine_frame_transforms(
                        a.data.body_pos_w[0, li].unsqueeze(0), a.data.body_quat_w[0, li].unsqueeze(0), lp, lq)
                    rec["handle_world_pos"] = _fl(wp[0])
                    rec["handle_world_quat"] = _fl(wq[0])
            rec["present_in_scene"] = True
        except Exception as exc:
            rec["present_in_scene"] = False
            rec["resolve_error"] = str(exc)
        mech[dn] = rec
    json.dump({"scene_version": SCENE_VERSION, "mechanisms": mech}, open(OUT / "mechanism_registry.json", "w"), indent=1)

    # ---- grasp_poses (drawer handles only) copied from single source ----
    handles = {k: v for k, v in active_gp.items() if k.startswith("handle_") and v.get("member") in ("cabinet", "sektion_cabinet")}
    json.dump({"scene_version": SCENE_VERSION, "frame": "reference_local",
               "note": "SINGLE SOURCE for paper drawer handle poses. Synced from active grasp_poses.json; "
                       "do not let the two silently overwrite each other.", "quat_order": "wxyz",
               "source": str(ACTIVE / "grasp_poses.json"), "poses": handles},
              open(OUT / "grasp_poses.json", "w"), indent=1)

    # ---- camera_registry (front camera registered, disabled for drawer profile) ----
    json.dump({"scene_version": SCENE_VERSION,
               "cameras": {"vla_front_static": {"role": "front RGB", "enabled": False,
                           "note": "registered for later VLM stage; disabled in drawer experiment profile."}}},
              open(OUT / "camera_registry.json", "w"), indent=1)

    # ---- external_assets (reuse archive hashing scheme) ----
    EXT = ["simv2/USD/Cabinet_44853/configuration/cabinet_base.usd",
           "simv2/USD/Cabinet_44853/configuration/cabinet_physics.usd",
           "Connection/assets/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd",
           "SapienAssetPipeline/usd_assets/CoffeeMachine_103046/coffeemachine.usd",
           "SapienAssetPipeline/usd_assets/IsaacProps/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd"]
    ext = {}
    for rel in EXT:
        ap_ = _ROOT / rel
        if not ap_.exists():
            ext[rel] = {"exists": False}
            continue
        st = ap_.stat()
        ext[rel] = {"exists": True, "abs_path": str(ap_), "bytes": st.st_size,
                    "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
                    "sha256": _sha(ap_) if st.st_size < 300 * (1 << 20) else "SKIPPED"}
    json.dump({"scene_version": SCENE_VERSION, "isaaclab_root": str(_ROOT), "assets": ext},
              open(OUT / "external_assets.json", "w"), indent=1)

    # ---- version ----
    status = _git("status", "--short")
    json.dump({"scene_version": SCENE_VERSION, "source_commit": _git("rev-parse", "HEAD"),
               "branch": _git("branch", "--show-current"), "dirty_worktree": bool(status.strip()),
               "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"), "task_id": V1_BASE_TASK_ID,
               "capture_tool": "scene/tools/export_paper_scene_v2.py"},
              open(OUT / "version.json", "w"), indent=1)

    # ---- authoritative scene_manifest ----
    manifest_objs = {}
    for name, e in objs.items():
        if "error" in e:
            continue
        enabled = name in ENABLED
        manifest_objs[name] = {**e, "member": name, "enabled": enabled,
                               "role": ("privileged: actuator damping is the hidden deployment axis"
                                        if name in ("cabinet", "sektion_cabinet") else "observable"),
                               "note": ("kept but disabled for drawer profile" if name in DISABLED_BUT_KEPT
                                        else ("props not participating" if name.startswith("Prop_") else ""))}
        if name.startswith("Prop_"):
            manifest_objs[name]["enabled"] = False
    json.dump({"scene_version": SCENE_VERSION, "source_commit": _git("rev-parse", "HEAD"),
               "task_id": V1_BASE_TASK_ID, "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "enabled_members": sorted(ENABLED), "disabled_but_kept": sorted(DISABLED_BUT_KEPT),
               "handle_pose_source": "scene/paper_scene_v2/grasp_poses.json",
               "privileged_note": "actuator joint_damping of cabinet/sektion_cabinet drawers is z_secret; "
                                  "it must NEVER enter ordinary model input x.",
               "mechanisms": mech, "objects": manifest_objs},
              open(OUT / "scene_manifest.json", "w"), indent=1)

    print(f"[export_paper_scene_v2] wrote {len(list(OUT.glob('*.json')))} json files to {OUT}", flush=True)
    print(f"[export_paper_scene_v2] enabled={sorted(ENABLED)} mechanisms={list(mech.keys())}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    rc = main()
    simulation_app.close()
    sys.exit(rc)
