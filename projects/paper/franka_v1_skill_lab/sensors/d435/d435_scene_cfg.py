# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Attach the TWO wrist cameras to a V1 env cfg.

This is the ONE place the cameras are added to the shared V1 scene. Every Isaac
entry point (layout editor, skill runtime, teleop collection, debug dumps) calls
:func:`attach_wrist_cameras` on its parsed ``env_cfg`` so they all load the same
instrumented scene.

Attached to ``env_cfg.scene`` (directly under ``panda_hand/``):

1. ``foundationpose_d435_rgbd`` — RGB + depth, 640x480 (FoundationPose).
2. ``vla_libero_eye_in_hand``   — RGB (opt. depth), 128x128/224x224 (VLA/pi0.5).
3. ``wrist_d435_body``          — the visible D435 housing cuboid, **HIDDEN by
   default** (collision-free; pass ``show_body=True`` / ``--show_camera_body``).

Cameras are only added when ``enable_cameras=True`` (the renderer needs
``--enable_cameras``). The (invisible) body can be added regardless so the prim
exists in the saved USD.

Isaac imports are lazy so this module imports without the Isaac app running.
"""

from __future__ import annotations

from . import d435_config as cfg


def make_camera_cfg(profile, width: int | None = None, height: int | None = None,
                    enable_depth: bool | None = None, enable_mask: bool = False):
    """Build a :class:`CameraCfg` from a :class:`CameraProfile`."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    w = int(width or profile.width)
    h = int(height or profile.height)
    return CameraCfg(
        prim_path=profile.prim_path,
        update_period=0.0,
        height=h,
        width=w,
        data_types=profile.data_types(want_depth=enable_depth, want_mask=enable_mask),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=profile.focal_length,
            focus_distance=400.0,
            horizontal_aperture=profile.horizontal_aperture,
            clipping_range=profile.clipping_range,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=tuple(profile.offset_pos),
            rot=tuple(profile.offset_rot),
            convention=profile.convention,
        ),
    )


def make_body_cfg(visible: bool = False):
    """Build the D435 housing cuboid (visual only). Hidden unless ``visible``."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    return AssetBaseCfg(
        prim_path=cfg.D435_BODY_PRIM,
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=cfg.D435_BODY_LOCAL_POS,
            rot=cfg.D435_BODY_LOCAL_ROT,
        ),
        # No collision_props / no rigid_props -> pure visual, zero physics effect.
        spawn=sim_utils.CuboidCfg(
            size=cfg.D435_BODY_SIZE,
            visible=bool(visible),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=cfg.D435_BODY_COLOR, roughness=0.5
            ),
        ),
    )


def _normalize_vla_resolution(vla_resolution):
    if vla_resolution is None:
        return cfg.VLA_PROFILE.width, cfg.VLA_PROFILE.height
    w, h = int(vla_resolution[0]), int(vla_resolution[1])
    if (w, h) not in cfg.VLA_SUPPORTED_RESOLUTIONS:
        print(
            f"[wrist_cameras] WARNING: vla_resolution {(w, h)} is not in "
            f"{cfg.VLA_SUPPORTED_RESOLUTIONS}; using it anyway.",
            flush=True,
        )
    return w, h


def attach_wrist_cameras(
    env_cfg,
    enable_cameras: bool = True,
    show_body: bool = False,
    enable_fp_depth: bool = True,
    enable_vla_depth: bool = False,
    vla_resolution=None,
    enable_front: bool = True,
    enable_front_depth: bool = False,
    front_resolution=None,
    enable_left: bool = True,
    enable_left_depth: bool = True,
    left_resolution=None,
    enable_right: bool = True,
    enable_right_depth: bool = True,
    right_resolution=None,
    enable_top: bool = True,
    enable_top_depth: bool = True,
    top_resolution=None,
    rerenders_on_reset: int = 3,
):
    """Attach ALL V1 cameras (+ hidden body) to ``env_cfg`` in place.

    Attaches the two wrist cameras AND the fixed third-person front camera
    (``vla_front_static``), so the shared scene carries the full set used by
    pi0.5/VLA (``image`` = front, ``wrist_image`` = wrist) and FoundationPose.

    Args:
        env_cfg: parsed manager-based env cfg (mutated in place).
        enable_cameras: add the rendered :class:`CameraCfg` objects (needs
            ``--enable_cameras``). The body prim is always added.
        show_body: make the D435 housing cuboid visible (default hidden).
        enable_fp_depth: FoundationPose camera renders depth (default True).
        enable_vla_depth: VLA wrist camera also renders depth (default False).
        vla_resolution: (w, h) override for the VLA wrist camera (128/224).
        enable_front: also attach the fixed front camera (default True).
        enable_front_depth: render depth on the front camera too (needed to use the
            front view for FoundationPose, which requires RGB-D). Default False
            (the front cam is normally the RGB-only LeRobot `image`).
        front_resolution: (w, h) override for the front camera (default 256x256).

    Returns:
        The same ``env_cfg``.
    """
    # Visible (or hidden) housing body — always present so it lands in the USD.
    env_cfg.scene.wrist_d435_body = make_body_cfg(visible=show_body)

    if enable_cameras:
        env_cfg.scene.foundationpose_d435_rgbd = make_camera_cfg(
            cfg.FP_PROFILE, enable_depth=enable_fp_depth
        )
        vw, vh = _normalize_vla_resolution(vla_resolution)
        env_cfg.scene.vla_libero_eye_in_hand = make_camera_cfg(
            cfg.VLA_PROFILE, width=vw, height=vh, enable_depth=enable_vla_depth
        )
        if enable_front:
            fw, fh = (front_resolution if front_resolution
                      else (cfg.FRONT_PROFILE.width, cfg.FRONT_PROFILE.height))
            env_cfg.scene.vla_front_static = make_camera_cfg(
                cfg.FRONT_PROFILE, width=int(fw), height=int(fh), enable_depth=enable_front_depth
            )
        if enable_left:
            lw, lh = (left_resolution if left_resolution
                      else (cfg.LEFT_PROFILE.width, cfg.LEFT_PROFILE.height))
            env_cfg.scene.vla_left_static = make_camera_cfg(
                cfg.LEFT_PROFILE, width=int(lw), height=int(lh), enable_depth=enable_left_depth
            )
        if enable_right:
            rw, rh = (right_resolution if right_resolution
                      else (cfg.RIGHT_PROFILE.width, cfg.RIGHT_PROFILE.height))
            env_cfg.scene.vla_right_static = make_camera_cfg(
                cfg.RIGHT_PROFILE, width=int(rw), height=int(rh), enable_depth=enable_right_depth
            )
        if enable_top:
            tw, th = (top_resolution if top_resolution
                      else (cfg.TOP_PROFILE.width, cfg.TOP_PROFILE.height))
            env_cfg.scene.vla_top_static = make_camera_cfg(
                cfg.TOP_PROFILE, width=int(tw), height=int(th), enable_depth=enable_top_depth
            )
        if hasattr(env_cfg, "num_rerenders_on_reset"):
            env_cfg.num_rerenders_on_reset = max(
                rerenders_on_reset, int(getattr(env_cfg, "num_rerenders_on_reset", 0))
            )
    return env_cfg


def cameras_attached(env_cfg) -> bool:
    """True if the wrist cameras are present on ``env_cfg.scene``."""
    return hasattr(env_cfg.scene, cfg.FP_CAM_NAME) and hasattr(env_cfg.scene, cfg.VLA_CAM_NAME)


# ---------------------------------------------------------------------------
# backwards-compat: the old single-camera API
# ---------------------------------------------------------------------------
def make_d435_camera_cfg(enable_depth: bool = True, enable_mask: bool = False):
    return make_camera_cfg(cfg.FP_PROFILE, enable_depth=enable_depth, enable_mask=enable_mask)


def make_d435_body_cfg():
    return make_body_cfg(visible=True)


def attach_d435(env_cfg, enable_camera: bool = True, enable_depth: bool = True,
                enable_mask: bool = False, enable_visual_body: bool = True,
                rerenders_on_reset: int = 3):
    """Back-compat shim: attach both wrist cameras.

    Maps the old single-D435 call onto :func:`attach_wrist_cameras`.
    ``enable_visual_body`` here controls whether the body is *visible* (the old
    behaviour spawned a visible body); the new default hides it.
    """
    return attach_wrist_cameras(
        env_cfg,
        enable_cameras=enable_camera,
        show_body=enable_visual_body,
        enable_fp_depth=enable_depth,
        rerenders_on_reset=rerenders_on_reset,
    )


def d435_attached(env_cfg) -> bool:
    return hasattr(env_cfg.scene, cfg.FP_CAM_NAME)
