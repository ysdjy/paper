# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""UI#2 —— 状态机技能测试 UI（从 franka_skill_state_machine/entries/skill_test_ui_joint.py 抽出）。

为什么有这个文件：那个 entry 的**模块顶层**就建 AppLauncher + parse_args（import 即触发），
无法被别的 entry import 复用，且不能改它（状态机模块那边维护）。所以把它里面**可复用的
UI 窗口 + glue helper** 原样抽到这里（可 import、无模块顶层副作用），只 import 干净的后端子模块。
控制算法仍在状态机模块（SkillExecutor / 各 skill），这里只是 UI + 把 command 转 action 的胶水。

注意：本模块顶层 import torch/isaac/legacy 子模块，**不是 plain-venv 安全**，只能在 AppLauncher
之后、ensure_legacy_on_path() 之后被 import（由 test_mode_ui entry 负责）。不要从 __init__ 导出。
"""

from __future__ import annotations

import math
import time

import torch

import isaaclab.utils.math as math_utils
from learned_drawer.official_drawer_policy import OfficialDrawerPolicyWrapper  # noqa: F401
from runtime.base_skill import get_speed_scale, pose_tensor, set_speed_scale
from runtime.debug_visualizer import DebugVisualizer, OBJECT_AXIS_LENGTH, OBJECT_AXIS_WIDTH
from runtime.drawer_obs_adapter import DrawerObsAdapter, SelectedDrawerObsAdapter  # noqa: F401
from runtime.drawer_target_config import functional_drawers  # noqa: F401
from runtime.microwave_door_config import DOOR_TARGETS, HANDLE_OFFSET_LOCAL
from runtime.scene_state_provider import PoseState, SceneStateProvider
from runtime.simple_scene_layout import SimpleLayoutResult  # noqa: F401
from runtime.skill_request import SkillRequest
from runtime.skill_types import ExecutionStatus, SkillType  # noqa: F401
from runtime.target_registry import TargetRegistry
from runtime.ui_controller import UIController
from skills.microwave_door_skill import DoorIKConfig, _grasp_quat_vertical_edge
from state_machine.skill_executor import SkillExecutor

SKILL_LABELS = [
    ("Grasp", SkillType.GRASP),
    ("Place", SkillType.PLACE),
    ("Open Drawer", SkillType.OPEN_DRAWER),
    ("Close Drawer", SkillType.CLOSE_DRAWER),
    ("Open Door", SkillType.OPEN_DOOR),
    ("Close Door", SkillType.CLOSE_DOOR),
]

# 显示名 Fridge（资产其实是冰箱）；内部目标值保持 "microwave"，不破坏门技能/配置引用。
DOOR_TARGETS_UI = [("Fridge", "microwave")]

DRAWER_TARGETS = [
    ("Bottom Drawer (bottom_drawer)", "bottom_drawer"),
    ("Middle Drawer (middle_drawer)", "middle_drawer"),
    ("Top Drawer (top_drawer)", "top_drawer"),
]

DEFAULT_PLACE_POINTS = {
    "point_a": [0.42, 0.10, 0.00],
    "point_b": [0.55, 0.10, 0.00],
    "point_c": [0.68, 0.10, 0.00],
    "point_d": [0.55, 0.20, 0.00],
}
PLACE_POINT_VISUAL_Z_OFFSET = 0.005
PLACE_POINT_AXIS_LENGTH = 0.035
PLACE_POINT_AXIS_WIDTH = 0.002
PLACE_MOVE_STEPS = [("1 cm", 0.010), ("5 cm", 0.050), ("10 cm", 0.100)]

SPEED_PRESETS = [
    ("0.5x (slow)", 0.5),
    ("1x (default tune)", 1.0),
    ("2x", 2.0),
    ("3x", 3.0),
    ("5x (fast)", 5.0),
]


def closest_speed_index(value: float) -> int:
    return min(range(len(SPEED_PRESETS)), key=lambda i: abs(SPEED_PRESETS[i][1] - value))


def enable_collision_debug_visualization():
    import carb

    settings = carb.settings.get_settings()
    settings.set_int("/persistent/physics/visualizationDisplayColliders", 2)
    settings.set_bool("/persistent/physics/visualizationDisplayColliderNormals", False)


class SkillTestWindow:
    def __init__(self, controller: UIController, executor: SkillExecutor, registry: TargetRegistry):
        import omni.ui as ui

        self.ui = ui
        self.controller = controller
        self.executor = executor
        self.registry = registry
        self.target_keys = [key for key, _ in registry.display_targets()]
        self.drawer_target_keys = [key for _, key in DRAWER_TARGETS]
        self.selected_drawer = self.drawer_target_keys[0]
        self.door_target_keys = [key for _, key in DOOR_TARGETS_UI]
        self.selected_door = self.door_target_keys[0]
        self.place_point_keys = list(DEFAULT_PLACE_POINTS.keys())
        self.place_points = {key: list(value) for key, value in DEFAULT_PLACE_POINTS.items()}
        self.selected_place_point = "point_a"
        self.place_move_step = PLACE_MOVE_STEPS[0][1]
        self.status_labels = {}
        self.window = ui.Window("Franka Skill Test (Joint)", width=460, height=720)
        with self.window.frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label("Skill")
                self.skill_model = ui.ComboBox(0, *[label for label, _ in SKILL_LABELS]).model
                self.skill_model.add_item_changed_fn(self._on_skill_changed)
                ui.Label("Grasp target")
                self.target_model = ui.ComboBox(1, *[label for _, label in registry.display_targets()]).model
                self.target_model.add_item_changed_fn(self._on_target_changed)
                ui.Label("Drawer target")
                self.drawer_model = ui.ComboBox(0, *[label for label, _ in DRAWER_TARGETS]).model
                self.drawer_model.add_item_changed_fn(self._on_drawer_changed)
                ui.Label("Door target (Open/Close Door)")
                self.door_model = ui.ComboBox(0, *[label for label, _ in DOOR_TARGETS_UI]).model
                self.door_model.add_item_changed_fn(self._on_door_changed)
                with ui.HStack(spacing=6):
                    ui.Button("Start", clicked_fn=self._start)
                    ui.Button("Stop", clicked_fn=self.controller.request_stop)
                    ui.Button("Resume", clicked_fn=self.controller.request_resume)
                    ui.Button("Reset", clicked_fn=self.controller.request_reset)
                ui.Label("Skill speed (all skills)")
                self.speed_model = ui.ComboBox(
                    closest_speed_index(get_speed_scale()), *[label for label, _ in SPEED_PRESETS]
                ).model
                self.speed_model.add_item_changed_fn(self._on_speed_changed)
                ui.Label("Place point")
                self.place_point_model = ui.ComboBox(0, *self.place_point_keys).model
                self.place_point_model.add_item_changed_fn(self._on_place_point_changed)
                ui.Label("Place move step")
                self.place_step_model = ui.ComboBox(0, *[label for label, _ in PLACE_MOVE_STEPS]).model
                self.place_step_model.add_item_changed_fn(self._on_place_step_changed)
                with ui.VStack(spacing=4, height=0):
                    with ui.HStack(spacing=6):
                        ui.Label("")
                        ui.Button("Up (+Z)", clicked_fn=lambda: self._move_place_point(0.0, 0.0, 1.0))
                        ui.Label("")
                    with ui.HStack(spacing=6):
                        ui.Label("")
                        ui.Button("Forward (+X)", clicked_fn=lambda: self._move_place_point(1.0, 0.0, 0.0))
                        ui.Label("")
                    with ui.HStack(spacing=6):
                        ui.Button("Left (+Y)", clicked_fn=lambda: self._move_place_point(0.0, 1.0, 0.0))
                        ui.Button("Reset Point", clicked_fn=self._reset_place_point)
                        ui.Button("Right (-Y)", clicked_fn=lambda: self._move_place_point(0.0, -1.0, 0.0))
                    with ui.HStack(spacing=6):
                        ui.Label("")
                        ui.Button("Backward (-X)", clicked_fn=lambda: self._move_place_point(-1.0, 0.0, 0.0))
                        ui.Label("")
                    with ui.HStack(spacing=6):
                        ui.Label("")
                        ui.Button("Down (-Z)", clicked_fn=lambda: self._move_place_point(0.0, 0.0, -1.0))
                        ui.Label("")
                for key in (
                    "selected_skill", "selected_target", "selected_drawer", "selected_place_point",
                    "place_point_x", "place_point_y", "place_point_z", "place_move_step", "speed_scale",
                    "held_object", "active_skill", "backend", "control_mode", "paused",
                    "latched_gripper_command", "runtime_status", "state", "elapsed", "position_error",
                    "orientation_error_deg", "gripper_width", "layout_seed", "reset_index", "layout_valid",
                    "cabinet_root_pose", "cube_1_pose", "cube_2_pose", "cube_3_pose", "knife_pose",
                    "target_pose", "drawer_control_mode", "drawer_joint_name", "drawer_joint_position",
                    "drawer_joint_target", "drawer_runtime_status", "handle_pose", "last_failure", "last_result",
                ):
                    self.status_labels[key] = ui.Label(f"{key}:")

    def _on_skill_changed(self, model, item):
        self.controller.selected_skill = SKILL_LABELS[model.get_item_value_model().as_int][1]

    def _on_target_changed(self, model, item):
        self.controller.selected_target = self.target_keys[model.get_item_value_model().as_int]

    def _on_drawer_changed(self, model, item):
        self.selected_drawer = self.drawer_target_keys[model.get_item_value_model().as_int]

    def _on_door_changed(self, model, item):
        self.selected_door = self.door_target_keys[model.get_item_value_model().as_int]

    def _on_speed_changed(self, model, item):
        applied = set_speed_scale(SPEED_PRESETS[model.get_item_value_model().as_int][1])
        print(f"[UI] skill speed scale set to {applied:.2f}x", flush=True)

    def _on_place_point_changed(self, model, item):
        self.selected_place_point = self.place_point_keys[model.get_item_value_model().as_int]

    def _on_place_step_changed(self, model, item):
        self.place_move_step = PLACE_MOVE_STEPS[model.get_item_value_model().as_int][1]

    def _move_place_point(self, x_dir: float, y_dir: float, z_dir: float):
        point = self.place_points[self.selected_place_point]
        point[0] += x_dir * self.place_move_step
        point[1] += y_dir * self.place_move_step
        point[2] += z_dir * self.place_move_step

    def _reset_place_point(self):
        self.place_points[self.selected_place_point] = list(DEFAULT_PLACE_POINTS[self.selected_place_point])

    def _held_object_name(self) -> str:
        held = self.executor.held_object
        return held.object_name if held is not None else "None"

    def _start(self):
        if self.controller.selected_skill == SkillType.PLACE:
            selected_xyz = list(self.place_points[self.selected_place_point])
            held_name = self.executor.held_object.object_name if self.executor.held_object is not None else None
            self.controller.queue_request(
                make_request(SkillType.PLACE, held_name,
                             place_point_name=self.selected_place_point, place_point_xyz=selected_xyz)
            )
            return
        if self.controller.selected_skill in (SkillType.OPEN_DRAWER, SkillType.CLOSE_DRAWER):
            backend = self.executor.backend.drawer_backend
            target = self.selected_drawer
            if backend == "none":
                print("[UI] drawer_backend='none': Open/Close Drawer disabled. Relaunch with "
                      "--drawer_backend scripted_joint or custom_selected_policy.", flush=True)
                return
            if backend == "scripted_joint":
                print("[BASELINE WARNING] scripted_joint directly commands drawer_joint_target; arm holds still.",
                      flush=True)
            if backend in ("custom_selected_policy", "ik_pull") and target == "bottom_drawer":
                print("[UI][WARNING] bottom_drawer is locked/non-functional in the asset; proceeding anyway.",
                      flush=True)
            self.controller.queue_request(make_request(self.controller.selected_skill, target))
            return
        if self.controller.selected_skill in (SkillType.OPEN_DOOR, SkillType.CLOSE_DOOR):
            self.controller.queue_request(make_request(self.controller.selected_skill, self.selected_door))
            return
        self.controller.queue_request(make_request(self.controller.selected_skill, self.controller.selected_target))

    def update(self, state, executor: SkillExecutor, layout_result):
        result = executor.last_result
        active = executor.active_skill
        plan = getattr(getattr(active, "runtime", None), "filtered_plan", None)
        target_pose_state = getattr(plan, "target_pose", None) or getattr(plan, "handle_pose", None)
        target_pose = target_pose_state.as_pose_tensor() if target_pose_state is not None else None
        handle_pose_state = getattr(plan, "handle_pose", None)
        drawer_runtime = getattr(active, "runtime", None)
        drawer_joint_name = getattr(drawer_runtime, "drawer_joint_name", "joint_0")
        drawer_joint_position = None
        cabinet = state.objects.get("cabinet")
        if cabinet is not None:
            drawer_joint_position = cabinet.joint_pos.get(drawer_joint_name)
        elapsed = 0.0
        if active is not None:
            elapsed = max(0.0, state.sim_time - getattr(active.runtime, "start_time", state.sim_time))
        orientation_error = getattr(getattr(active, "runtime", None), "final_error_ori", None)
        drawer_joint_target = getattr(drawer_runtime, "drawer_joint_target", None)
        point = self.place_points[self.selected_place_point]
        latched = executor.latched_command
        values = {
            "selected_skill": self.controller.selected_skill.value,
            "selected_target": self.controller.selected_target,
            "selected_drawer": self.selected_drawer,
            "selected_place_point": self.selected_place_point,
            "place_point_x": f"{point[0]:.3f}", "place_point_y": f"{point[1]:.3f}",
            "place_point_z": f"{point[2]:.3f}", "place_move_step": f"{self.place_move_step:.3f}",
            "speed_scale": f"{get_speed_scale():.2f}x", "held_object": self._held_object_name(),
            "active_skill": None if active is None else active.__class__.__name__,
            "backend": None if active is None else getattr(active, "backend", None),
            "control_mode": "joint", "paused": executor.paused,
            "latched_gripper_command": None if latched is None else f"{latched.gripper_command:.1f}",
            "runtime_status": executor.runtime_status, "state": executor.current_state_name,
            "elapsed": f"{elapsed:.2f}",
            "position_error": str(getattr(getattr(active, "runtime", None), "final_error_pos", None)),
            "orientation_error_deg": None if orientation_error is None else f"{math.degrees(orientation_error):.2f}",
            "gripper_width": f"{state.robot.gripper_width:.5f}",
            "layout_seed": None if layout_result is None else layout_result.seed,
            "reset_index": None if layout_result is None else layout_result.reset_index,
            "layout_valid": None if layout_result is None else True,
            "cabinet_root_pose": None if layout_result is None else layout_result.object_poses.get("cabinet"),
            "cube_1_pose": None if layout_result is None else layout_result.object_poses.get("cube_1"),
            "cube_2_pose": None if layout_result is None else layout_result.object_poses.get("cube_2"),
            "cube_3_pose": None if layout_result is None else layout_result.object_poses.get("cube_3"),
            "knife_pose": None if layout_result is None else layout_result.object_poses.get("knife"),
            "target_pose": _short_pose(target_pose),
            "drawer_control_mode": getattr(drawer_runtime, "drawer_control_mode", None),
            "drawer_joint_name": drawer_joint_name,
            "drawer_joint_position": None if drawer_joint_position is None else f"{drawer_joint_position:.5f}",
            "drawer_joint_target": None if drawer_joint_target is None else f"{drawer_joint_target:.5f}",
            "drawer_runtime_status": getattr(drawer_runtime, "state", None),
            "handle_pose": _short_pose(handle_pose_state.as_pose_tensor() if handle_pose_state is not None else None),
            "last_failure": None if result is None else result.failure_reason,
            "last_result": None if result is None else result.final_status.value,
        }
        for key, label in self.status_labels.items():
            label.text = f"{key}: {values[key]}"


def _short_pose(pose) -> str:
    if pose is None:
        return "None"
    values = pose.detach().cpu().tolist()
    return "[" + ", ".join(f"{v:.3f}" for v in values[:3]) + "]"


def make_request(skill_type, target, place_point_name: str = "point_a", place_point_xyz=None) -> SkillRequest:
    if skill_type == SkillType.PLACE:
        selected_xyz = list(DEFAULT_PLACE_POINTS[place_point_name] if place_point_xyz is None else place_point_xyz)
        return SkillRequest(
            request_id=f"{skill_type.value}_{target or 'none'}_{time.time_ns()}",
            skill_type=skill_type, source_object=target, destination_type="point",
            destination_object=place_point_name,
            parameters={"target_frame": "env_local",
                        "target_surface_xyz": [selected_xyz[0], selected_xyz[1], selected_xyz[2]]},
        )
    if skill_type in (SkillType.OPEN_DRAWER, SkillType.CLOSE_DRAWER):
        return SkillRequest(
            request_id=f"{skill_type.value}_{target}_{time.time_ns()}", skill_type=skill_type,
            source_object=None, destination_type="drawer", destination_object=target or "bottom_drawer",
            parameters={"drawer_link": "link_1"},
        )
    if skill_type in (SkillType.OPEN_DOOR, SkillType.CLOSE_DOOR):
        return SkillRequest(
            request_id=f"{skill_type.value}_{target}_{time.time_ns()}", skill_type=skill_type,
            source_object=None, destination_type="door", destination_object=target or "microwave",
        )
    return SkillRequest(
        request_id=f"{skill_type.value}_{target}_{time.time_ns()}", skill_type=skill_type,
        source_object=target if skill_type == SkillType.GRASP else None,
    )


def command_to_action(provider: SceneStateProvider, command, state):
    if command.control_mode == "joint":
        if command.raw_joint_action is not None:
            return provider.make_joint_action_from_raw(command.raw_joint_action)
        if command.joint_target is not None:
            return provider.make_joint_action_from_q_des(command.joint_target, command.gripper_command)
        return provider.make_hold_joint_action(state, None)
    return provider.make_action(command.tcp_pose_w, command.gripper_command)


def settle_layout(env, provider: SceneStateProvider) -> None:
    provider.reset_cabinet_joint("joint_0", 0.0)
    state = provider.get_state()
    hold_action = provider.make_hold_joint_action(state, 1.0)
    for _ in range(5):
        env.step(hold_action)


def apply_drawer_joint_command(provider: SceneStateProvider, command, baseline_warned: dict) -> None:
    if command.drawer_joint_target is None:
        return
    if not baseline_warned["done"]:
        print("[BASELINE] scripted_joint is directly commanding drawer joint, not learned physical pulling.",
              flush=True)
        baseline_warned["done"] = True
    provider.set_cabinet_joint_target(command.drawer_joint_name or "joint_0", command.drawer_joint_target)


def draw_handle_markers(visualizer: DebugVisualizer, handle_adapters: dict):
    for target, adapter in handle_adapters.items():
        try:
            handle = adapter.selected_handle_pos_w()[0]
        except Exception:
            continue
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=handle.device)
        visualizer.update_pose(f"handle_{target}", torch.cat((handle, quat)), axis_length=0.05, axis_width=0.0025)


_MW_GRASP_CFG = DoorIKConfig()


def draw_microwave_grasp(visualizer: DebugVisualizer, provider: SceneStateProvider):
    try:
        mw = provider.scene["microwave"]
        dcfg = DOOR_TARGETS["microwave"]
        bnames = list(mw.data.body_names)
        if dcfg["link_name"] not in bnames:
            return
        eid = provider.env_id
        door_idx = bnames.index(dcfg["link_name"])
        dev = mw.data.body_pos_w.device
        lp = mw.data.body_pos_w[eid, door_idx]
        lq = mw.data.body_quat_w[eid, door_idx]
        offset = torch.tensor(HANDLE_OFFSET_LOCAL, device=dev, dtype=torch.float32)
        handle, _ = math_utils.combine_frame_transforms(lp.unsqueeze(0), lq.unsqueeze(0), offset.unsqueeze(0))
        handle = handle[0]
        down = torch.tensor([0.0, 0.0, -1.0], device=dev)
        if _MW_GRASP_CFG.approach_mode == "radial":
            horiz = -(handle - lp).clone()
        else:
            n = math_utils.quat_apply(lq.unsqueeze(0), torch.tensor([[0.0, 0.0, -1.0]], device=dev))[0].clone()
            horiz = n
        horiz[2] = 0.0
        horiz = horiz / torch.linalg.norm(horiz)
        p = math.radians(_MW_GRASP_CFG.approach_pitch_deg)
        approach = horiz * math.cos(p) + down * math.sin(p)
        approach = approach / torch.linalg.norm(approach)
        quat = _grasp_quat_vertical_edge(approach, dev)
        pre = handle - approach * _MW_GRASP_CFG.pre_grasp_clearance
        visualizer.update_pose("microwave_grasp", pose_tensor(PoseState(handle, quat)), use_coordinate_arrows=True)
        visualizer.update_pose("microwave_pre_grasp", pose_tensor(PoseState(pre, quat)),
                               axis_length=0.05, axis_width=0.0025)
    except Exception:
        return


def update_debug_visuals(visualizer: DebugVisualizer, state, executor: SkillExecutor, place_points=None):
    visualizer.update_pose("current_tcp", pose_tensor(state.robot.tcp_pose), use_coordinate_arrows=True)
    for object_name in ("cube_1", "cube_2", "cube_3", "knife"):
        obj = state.objects.get(object_name)
        if obj is not None:
            visualizer.update_pose(f"object_{object_name}", pose_tensor(obj.pose),
                                   axis_length=OBJECT_AXIS_LENGTH, axis_width=OBJECT_AXIS_WIDTH)
    if place_points:
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=state.env_origin_w.device)
        visual_offset = torch.tensor([0.0, 0.0, PLACE_POINT_VISUAL_Z_OFFSET],
                                     dtype=torch.float32, device=state.env_origin_w.device)
        for point_name, point_xyz in place_points.items():
            point_local = torch.tensor(point_xyz, dtype=torch.float32, device=state.env_origin_w.device)
            marker_pos_w = state.env_origin_w + point_local + visual_offset
            visualizer.update_pose(f"place_{point_name}", torch.cat((marker_pos_w, quat), dim=-1),
                                   axis_length=PLACE_POINT_AXIS_LENGTH, axis_width=PLACE_POINT_AXIS_WIDTH)
    active = executor.active_skill
    runtime = getattr(active, "runtime", None)
    if runtime is None:
        return
    plan = getattr(runtime, "filtered_plan", None)
    if plan is not None and hasattr(plan, "handle_pose"):
        visualizer.update_pose("drawer_handle", pose_tensor(plan.handle_pose))
        visualizer.update_pose("drawer_pre_target", pose_tensor(plan.pre_handle_pose))
        action_pose = (plan.push_target_pose if getattr(active, "request", None)
                       and active.request.skill_type == SkillType.CLOSE_DRAWER else plan.pull_pose)
        visualizer.update_pose("drawer_action_target", pose_tensor(action_pose))
    if getattr(runtime, "last_command_pose", None) is not None:
        visualizer.update_pose("current_stage_target", pose_tensor(runtime.last_command_pose),
                               use_coordinate_arrows=True)
