# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""V1 joint-action skill UI entry — STATUS: wrapper.

This is the single V1 entry point for the joint-action Franka skill state
machine. It is a thin wrapper around the proven legacy entry

    projects/franka_skill_state_machine/entries/skill_test_ui_joint.py

and adds two things on top WITHOUT editing the legacy file:

  1. ``--scene_registry`` — resolve the active V1 scene (USD + manifest) and log
     it as provenance.
  2. ``--enable_wrist_d435`` / ``--show_camera_debug`` / ``--camera_name`` — attach
     the shared-scene wrist D435 (same camera the layout editor / teleop / pi0.5
     use) to the env cfg and, optionally, print its mount + pose. This is done by
     wrapping ``gymnasium.make`` before the legacy entry runs, so the legacy UI,
     markers, backends and joint-action contract are reused verbatim and the
     original working command keeps working unchanged.

Usage (V1, with camera):
    ./isaaclab.sh -p projects/franka_v1_skill_lab/skill_runtime/entries/skill_test_ui_joint_v1.py \
        --num_envs 1 --show_affordance_debug --show_camera_debug \
        --grasp_backend joint_ik --place_backend joint_ik --drawer_backend ik_pull \
        --scene_registry projects/franka_v1_skill_lab/scene/saved_scenes/v1_active/scene_v1_registry.json \
        --seed 1

All V1-level flags are optional; omit them and the legacy base task cfg
(Isaac-Stack-Cube-Franka-JointPolicy-v0) is used unchanged. ``--show_camera_debug``
and ``--enable_wrist_d435`` auto-enable Isaac camera rendering (--enable_cameras).

The state machine does NOT use FoundationPose to control the arm — it only loads
the same instrumented V1 scene and reports whether the wrist D435 is available.

KNOWN LIMITATION (TODO): the registry currently restores scene *provenance*
(which USD/manifest is active) and validates the object contract, but does NOT
yet re-apply saved object poses from the V1 manifest into the live env. The base
task cfg already matches the V0 fixed scene, so the running scene is correct
today. See docs/migration_report.md.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

# --- locate this project + the legacy entry ---------------------------------
_THIS = Path(__file__).resolve()
# …/franka_v1_skill_lab/skill_runtime/entries/skill_test_ui_joint_v1.py
_PROJECT_ROOT = _THIS.parents[2]           # franka_v1_skill_lab
_PROJECTS_DIR = _PROJECT_ROOT.parent       # projects/
_LEGACY_ENTRY = (
    _PROJECTS_DIR / "franka_skill_state_machine" / "entries" / "skill_test_ui_joint.py"
).resolve()

# Make `from franka_v1_skill_lab... import ...` importable.
sys.path.insert(0, str(_PROJECTS_DIR))

# FoundationPose capture config (set in main(), read in the gym.make patch).
_CAPTURE: dict = {"dir": None, "target": "cube_1", "every": 10}


def _install_fp_capture(env) -> None:
    """Wrap env.step to save the FP-camera RGB-D + GT target pose each Nth step.

    Reuses the proven grasp skill: the legacy loop calls ``env.step(action)``; we
    shadow it so each captured step writes rgb/<i>.png, depth/<i>.npy, cam_K.txt
    and appends camera_pose_world + GT cube pose to frames.json. The output dir
    feeds projects/foundationpose_deploy/scripts/track_and_evaluate.py.
    """
    import json
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from franka_v1_skill_lab.sensors.d435.d435_observation_adapter import WristCameraAdapter

    out = Path(_CAPTURE["dir"]).resolve()
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    target = _CAPTURE["target"]
    every = _CAPTURE["every"]
    adapter = WristCameraAdapter(env, "foundationpose_d435_rgbd")
    frames: list = []
    counters = {"step": 0, "saved": 0}
    orig_step = env.step
    wrote_K = {"done": False}

    def _capture_step(action):
        result = orig_step(action)
        i = counters["step"]
        counters["step"] += 1
        if i % every != 0:
            return result
        try:
            fr = adapter.capture(require_depth=True)
            idx = counters["saved"]
            counters["saved"] += 1
            Image.fromarray(np.asarray(fr["rgb"])[..., :3].astype("uint8")).save(out / "rgb" / f"{idx:06d}.png")
            np.save(out / "depth" / f"{idx:06d}.npy", np.asarray(fr["depth"], dtype="float32"))
            if not wrote_K["done"]:
                K = fr["intrinsics"]
                np.savetxt(out / "cam_K.txt",
                           np.array([[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1]]), fmt="%.8f")
                wrote_K["done"] = True
            obj = env.unwrapped.scene[target]
            pos = obj.data.root_pos_w[0].detach().cpu().numpy().tolist()
            quat = obj.data.root_quat_w[0].detach().cpu().numpy().tolist()  # wxyz
            frames.append({
                "frame": idx, "step": i,
                "camera_pose_world": fr["camera_pose_world"],   # Isaac 'world' convention
                "gt_pose_world": {"position": [float(v) for v in pos], "quat_wxyz": [float(v) for v in quat]},
            })
            (out / "frames.json").write_text(json.dumps({"target": target, "frames": frames}, indent=2))
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[skill_v1] capture step {i} failed: {exc}", flush=True)
        return result

    env.step = _capture_step
    print(f"[skill_v1] FoundationPose capture ON -> {out} (target={target}, every {every} steps)", flush=True)


def _pop_flag_with_value(argv: list[str], flag: str) -> str | None:
    """Remove ``flag <value>`` (or ``flag=value``) from argv; return the value."""
    out_value: str | None = None
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == flag:
            if i + 1 < len(argv):
                out_value = argv[i + 1]
                del argv[i : i + 2]
            else:
                del argv[i]
            continue
        if tok.startswith(flag + "="):
            out_value = tok.split("=", 1)[1]
            del argv[i]
            continue
        i += 1
    return out_value


def _pop_flag(argv: list[str], flag: str) -> bool:
    """Remove a boolean ``flag`` from argv; return True if it was present."""
    present = False
    while flag in argv:
        argv.remove(flag)
        present = True
    return present


def _resolve_and_log_registry(registry_path: str | None) -> None:
    """Resolve the V1 active scene and print provenance. Never fatal."""
    try:
        from franka_v1_skill_lab.scene import resolve_active_scene  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[skill_v1] could not import scene contract ({exc}); using base task cfg.", flush=True)
        return
    info = resolve_active_scene(registry_path)
    print("[skill_v1] ---- V1 scene registry ----", flush=True)
    print(f"[skill_v1]   task_id      = {info['task_id']}", flush=True)
    print(f"[skill_v1]   control_mode = {info['control_mode']}", flush=True)
    print(f"[skill_v1]   active_usd   = {info['usd']} (exists={info['usd_exists']})", flush=True)
    print(f"[skill_v1]   manifest     = {info['manifest']} (exists={info['manifest_exists']})", flush=True)
    sensors = info.get("sensors") or {}
    if sensors:
        print(f"[skill_v1]   sensors      = {sorted(sensors.keys())}", flush=True)
    if not info["usd_exists"]:
        print(
            "[skill_v1]   NOTE: no saved V1 USD yet -> running on the base task cfg "
            "(Isaac-Stack-Cube-Franka-JointPolicy-v0). Save one in the layout editor first.",
            flush=True,
        )
    print("[skill_v1]   NOTE: per-object pose restore from manifest is not yet applied (TODO).", flush=True)
    print("[skill_v1] ---------------------------", flush=True)


def _install_d435_gym_patch(camera_name: str, render_camera: bool, want_depth: bool,
                            show_debug: bool, show_body: bool) -> None:
    """Wrap gymnasium.make so the legacy entry builds the env WITH both wrist cameras.

    gymnasium is a light import (no Isaac), and the legacy entry does
    ``import gymnasium as gym; env = gym.make(TASK_ID, cfg=env_cfg)``. Patching
    the module attribute here (before runpy executes the legacy file) means the
    legacy call resolves to our wrapper, which attaches the two wrist cameras to
    ``cfg`` and optionally prints the camera debug block after the env is built.
    """
    import gymnasium as gym

    _orig_make = gym.make

    def _patched_make(*args, **kwargs):
        cfg = kwargs.get("cfg")
        if cfg is not None:
            try:
                from franka_v1_skill_lab.sensors.d435.d435_scene_cfg import attach_wrist_cameras

                attach_wrist_cameras(
                    cfg,
                    enable_cameras=render_camera,
                    show_body=show_body,
                    enable_fp_depth=want_depth,
                    enable_vla_depth=False,
                )
                mode = "rendering RGB-D + RGB" if render_camera else "config only"
                body = "body VISIBLE" if show_body else "body hidden"
                print(
                    f"[skill_v1] wrist cameras attached (foundationpose_d435_rgbd + "
                    f"vla_libero_eye_in_hand) [{mode}, {body}].",
                    flush=True,
                )
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[skill_v1] WARNING: could not attach wrist cameras: {exc}", flush=True)
        env = _orig_make(*args, **kwargs)
        if show_debug:
            try:
                from franka_v1_skill_lab.sensors.d435.d435_visual_debug import print_camera_debug

                only = None if camera_name in ("all", "both", "") else camera_name
                print_camera_debug(env, camera_name=only, tag="skill_v1")
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[skill_v1] WARNING: camera debug failed: {exc}", flush=True)
        if _CAPTURE.get("dir"):
            try:
                _install_fp_capture(env)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[skill_v1] WARNING: could not install FP capture: {exc}", flush=True)
        return env

    gym.make = _patched_make


def main() -> None:
    argv = sys.argv[1:]

    # --- V1-level flags (must be stripped before handing argv to legacy) -----
    registry_path = _pop_flag_with_value(argv, "--scene_registry")
    camera_name = _pop_flag_with_value(argv, "--camera_name") or _pop_flag_with_value(argv, "--camera") or "all"
    show_camera_debug = _pop_flag(argv, "--show_camera_debug")
    enable_wrist_cameras = _pop_flag(argv, "--enable_wrist_cameras") or _pop_flag(argv, "--enable_wrist_d435")
    show_camera_body = _pop_flag(argv, "--show_camera_body")
    _pop_flag(argv, "--hide_camera_body")  # default; accepted for parity
    no_depth = _pop_flag(argv, "--no_depth")
    # FoundationPose capture: save the FP camera frame + GT target pose each step.
    capture_dir = _pop_flag_with_value(argv, "--capture_fp_sequence")
    capture_target = _pop_flag_with_value(argv, "--capture_target") or "cube_1"
    capture_every = int(_pop_flag_with_value(argv, "--capture_every") or "10")
    if capture_dir:
        _CAPTURE["dir"] = capture_dir
        _CAPTURE["target"] = capture_target
        _CAPTURE["every"] = max(1, capture_every)
        enable_wrist_cameras = True  # capture needs the rendered FP camera

    _resolve_and_log_registry(registry_path)

    want_camera = show_camera_debug or enable_wrist_cameras
    if want_camera:
        # The rendered RGB-D camera needs Isaac camera rendering enabled. The
        # legacy entry constructs ``AppLauncher(headless=args_cli.headless)`` and
        # does NOT forward --enable_cameras, so the CLI flag alone would be
        # ignored and Isaac would reject the spawned camera. AppLauncher also
        # honours the ENABLE_CAMERAS env var (app_launcher.py), so set that — it
        # is the path the legacy call actually respects.
        import os

        os.environ["ENABLE_CAMERAS"] = "1"
        if "--enable_cameras" not in argv:
            argv.append("--enable_cameras")
        print("[skill_v1] enabled camera rendering (ENABLE_CAMERAS=1) for the wrist cameras.", flush=True)
        _install_d435_gym_patch(
            camera_name=camera_name,
            render_camera=True,
            want_depth=not no_depth,
            show_debug=show_camera_debug,
            show_body=show_camera_body,
        )

    if not _LEGACY_ENTRY.is_file():
        raise FileNotFoundError(f"Legacy joint skill entry not found: {_LEGACY_ENTRY}")

    # Hand the (V1-stripped) argv to the legacy entry and run it as __main__.
    sys.argv = [str(_LEGACY_ENTRY)] + argv
    runpy.run_path(str(_LEGACY_ENTRY), run_name="__main__")


if __name__ == "__main__":
    main()
