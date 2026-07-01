# BrainInterface 接口文档（API Reference）

> 模块：`franka_v1_skill_lab/scene_interface/brain_interface.py`
> 面向：「大脑」（高层编排器 / LLM Agent）。把整个 V1 场景模块封装成 **感知 / 控制 / 认知** 三个类，
> 聚合在 `BrainInterface` 下。底层复用 `SceneSession`（进程内、无 ZMQ）+ `SkillExecutor`（状态机）。

## 全局约定（所有接口一致）

| 项 | 约定 |
|---|---|
| 关节 | 7 维**绝对关节弧度** `q_des`，与 `robot.data.joint_pos[:7]` 同序 |
| 夹爪（对外） | `0=open / 1=close`（pi0.5/LeRobot 约定），也接受 `"open"/"close"`；内部自动换算成 env 的 `+1/-1` |
| 四元数 | 一律 **wxyz** |
| 坐标系 | 位姿默认 `frame="world"`，可传 `frame="base"` 转到机器人基座系 |
| 图像 | RGB=`HxWx3 uint8`；Depth=`HxW float32`（米） |
| 目标引用 | 物体/抽屉/门一律用**字符串名**（如 `"cube_1"`、`"middle_drawer"`、`"microwave"`） |

---

## 顶层：`BrainInterface`

### 构造 / 生命周期

| 方法 | 签名 | 说明 |
|---|---|---|
| `launch` | `BrainInterface.launch(config: SceneConfig) -> BrainInterface` | 自建并启动场景（自管理 AppLauncher）。库/脚本用法。 |
| `attach` | `BrainInterface.attach(session: SceneSession) -> BrainInterface` | 复用已有 session（不接管其生命周期）。 |
| `__init__` | `BrainInterface(session, *, owns_session=False)` | 也直接当 test_mode 的 `--controller`（见下）。 |
| `close` | `close() -> None` | 仅当 `owns_session=True` 才真正关 session/AppLauncher。 |
| 上下文管理 | `with BrainInterface.launch(cfg) as brain: ...` | `__enter__/__exit__` 自动 close。 |

### 子接口（属性）

| 属性 | 类型 | 职责 |
|---|---|---|
| `.perception` | `ScenePerception` | 读仿真器数据 |
| `.control` | `SkillControl` | 给状态机下指令 |
| `.cognition` | `TaskCognition` | 能力发现 + 前置校验 + 执行反馈 + 回合 + 可视化 |
| `.session` | `SceneSession` | 底层场景句柄（一般不用直接碰） |

### 运行（自管理循环）

| 方法 | 签名 | 说明 |
|---|---|---|
| `tick` | `tick() -> dict` | 推进一个仿真步（算动作→`env.step`，无动作则 hold）。返回 `skill_status()` 快照。 |
| `run_skill` | `run_skill(skill, target=None, target_pose=None, params=None, max_steps=3000, settle=20) -> dict\|None` | **阻塞**跑完一个技能：下指令→循环 tick 到结束→settle。返回 `last_result()`。 |
| `reset` | `reset(reset_mode=None, seed=None) -> Observation` | 重置场景（随机布局）+ 清空状态机。 |
| `observe` | `observe() -> Observation` | 一帧完整观测。 |

### controller hook（被 test_mode 每帧调用）

| 方法 | 签名 | 说明 |
|---|---|---|
| `build_window` | `build_window() -> None` | 可选 UI 钩子；大脑无 UI，留空。 |
| `on_reset` | `on_reset() -> None` | 布局 reset 后丢弃当前技能。 |
| `step` | `step(session) -> action\|None` | 每帧返回一帧 env 动作或 `None`（交回 UI/hold）。 |

---

## 1) `ScenePerception`（感知类）—— 只读

### 全量观测
| 方法 | 返回 | 说明 |
|---|---|---|
| `observe()` | `Observation` | pi05 块 + raw 块 + 两相机 + foundationpose 块 + 把手。**重**（抓相机）。 |
| `get_state()` | `SceneState` | 底层 tensor 态（控制/认知内部用）。 |

### 相机（`camera ∈ {"front","wrist"}`）
| 方法 | 返回 |
|---|---|
| `get_rgb(camera="front")` | `HxWx3 uint8` 或 `None` |
| `get_depth(camera="wrist")` | `HxW float32`（米，需 `enable_fp=True`）或 `None` |
| `get_rgbd(camera="wrist")` | `(rgb, depth)` 同视角 |
| `get_camera_intrinsics(camera="front")` | `{fx,fy,cx,cy,width,height}` |
| `get_camera_pose(camera="front")` | `{position[3], quat_wxyz[4]}`（世界系光心） |

### 物体 / 机器人
| 方法 | 返回 |
|---|---|
| `get_object_pose(name, frame="world")` | `{position[3], quat_wxyz[4], frame}` 或 `None`（cube_1/2/3、knife） |
| `list_object_poses(frame="world")` | `{name: {position, quat_wxyz, frame}}` |
| `get_ee_pose(frame="world")` | `{position[3], quat_wxyz[4], frame}`（TCP） |
| `get_gripper_width()` | `float`（米） |
| `get_arm_joints()` | `[7]` 手臂关节弧度 |
| `get_robot_joints()` | `{pos[9], vel[9], arm[7], gripper_width}` |

### 家电关节 / 把手
| 方法 | 返回 |
|---|---|
| `get_articulation_joint(asset, joint)` | `{asset, joint, pos, vel}` 或 `None`。asset ∈ `cabinet/microwave/coffee_machine`；如抽屉行程、门角度、咖啡把手角。 |
| `get_handle_poses(frame="base")` | `{name: {asset, link, calibrated, functional, position[3], quat_wxyz[4]}}`。name ∈ `microwave_door/top_drawer/middle_drawer/bottom_drawer/coffee_lever`。`frame="world"` 取世界系。 |

### FoundationPose
| 方法 | 返回 / 说明 |
|---|---|
| `get_foundationpose_input(which="wrist", dump_dir=None)` | FP 进程输入 bundle（同视角 RGBD + 内外参 + 物体清单）或 `None`。 |
| `set_pose_estimator(fn)` | 注入真实 FP：`fn(name, fp_input) -> {position, quat_wxyz}` 或 `4x4`。 |
| `get_estimated_pose(name, which="wrist")` | 估计位姿。注入了 estimator 用 FP（`pose_source="foundationpose"`）；否则退化为仿真 GT（`pose_source="sim_gt"`）。 |

---

## 2) `SkillControl`（控制类）—— 大脑 → 状态机

> 指令只是**入队**；真正的 `executor.start` 在下一次 `tick()/step()` 用当帧 `SceneState` 执行。

### 下指令
| 方法 | 签名 | 说明 |
|---|---|---|
| `command_skill` | `command_skill(skill, target=None, target_pose=None, params=None) -> str` | **一步到位**下一条技能指令，返回 `request_id`。 |
| `select_skill` | `select_skill(skill) -> None` | 分步：选技能。 |
| `set_target_object` | `set_target_object(name) -> None` | 分步：选目标物体/抽屉/门。 |
| `set_target_pose` | `set_target_pose(xyz) -> None` | 分步：选目标位置（**仅 place**，env-local `[x,y,z]`）。 |
| `set_params` | `set_params(**params) -> None` | 分步：附加参数（如 `drawer_link`）。 |
| `start` | `start() -> str` | 用已选 skill/target/pose/params 入队，返回 `request_id`。 |

`skill` 取值：`"grasp"|"place"|"open_drawer"|"close_drawer"|"open_door"|"close_door"` 或 `SkillType`。

### 运行时控制
| 方法 | 签名 | 说明 |
|---|---|---|
| `pause` | `pause() -> None` | 暂停（冻结当前姿态，之后 tick 发 hold）。 |
| `resume` | `resume() -> bool` | 继续被暂停的技能。 |
| `stop` | `stop() -> None` | 停止（软停 = pause，保留 `held_object`）。 |
| `abort` | `abort() -> None` | **硬中止**：丢弃技能 + 清空状态机，回 idle。 |
| `is_busy` | `is_busy() -> bool` | 有技能在跑或待启动。 |

### 低层透传
| 方法 | 签名 | 说明 |
|---|---|---|
| `command_joints` | `command_joints(q_des7, gripper=1) -> None` | 入队一次性手动关节动作（仅无技能在跑时生效）。 |
| `compute_action` | `compute_action(state) -> action\|None` | 每帧算 env 动作（由 `tick/step` 调用，一般不直接用）。 |

---

## 3) `TaskCognition`（认知类）—— 闭环所需的「第三层」

### 能力发现
| 方法 | 返回 |
|---|---|
| `list_skills()` | `[{skill, target_kind, needs_target, params, desc, success}, ...]` |
| `list_objects()` | `{graspable:[...], drawer:[...], door:[...], place_point:[...]}` |
| `list_targets(skill)` | 该技能合法目标域 `[...]` |
| `describe_skill(skill)` | `{skill, target_kind, needs_target, params, desc, success}` |
| `can_execute(skill, target=None)` | `{ok: bool, reason: str}`（前置校验：目标合法？grasp 时是否空手？place 时是否持物？） |

### 执行反馈（闭环必需）
| 方法 | 返回 |
|---|---|
| `skill_status()` | `{active, phase, runtime_status, held_object, busy}` |
| `is_holding()` | 当前夹着的物体名 或 `None` |
| `last_result()` | `{request_id, skill, target, success, status, failure_reason, elapsed, position_error, orientation_error, gripper_width}` 或 `None` |

### 回合管理
| 方法 | 签名 | 说明 |
|---|---|---|
| `reset` | `reset(reset_mode=None, seed=None) -> Observation` | 重置布局（`reset_mode ∈ {"static","region","legacy"}`）+ 清空状态机。 |
| `settle` | `settle(steps=8)` | 空跑稳定物理/渲染。 |
| `save_scene` | `save_scene(note=None) -> dict` | 当前布局存为 `scene_v1_latest`（**机器人姿态不存**）。返回保存信息/`{"error":...}`。 |

### 可视化 / 调试
| 方法 | 说明 |
|---|---|
| `show_targets(poses, names=None, use_arrows=True, axis_length=0.08)` | 画目标位姿箭头（headless 静默）。 |
| `clear_targets()` | 清除箭头。 |
| `set_collision_visible(on)` | 碰撞体可视化开关。 |
| `set_markers_visible(on)` | 黄色 InitCorner 区域块开关。 |

---

## 模块级常量

```python
GRASP_TARGETS = ("cube_1", "cube_2", "cube_3", "knife")
DRAWER_TARGETS = ("middle_drawer", "top_drawer", "bottom_drawer")
DOOR_TARGETS = ("microwave",)        # 成员名（即便换成冰箱仍叫 microwave）
PLACE_POINTS = ("point_a",)
```

## 返回的数据结构（来自 `observation.py`）

`Observation`：`pi05`（`image/wrist_image/state8`）、`raw`（`joint_pos/arm_joint_pos/joint_vel/tcp_pose[x,y,z,qx,qy,qz,qw]/gripper_width/base_pose_w`）、
`cameras`（`{"front":CameraView,"wrist":CameraView}`）、`foundationpose`（`FoundationPoseBlock`，可 `.fp_input(which)`）、`handles`、`sim_time`。

> 注意 `raw.tcp_pose` 是 `xyzw` 顺序的便捷量；而 `ScenePerception.get_ee_pose()` 统一返回 **wxyz**。
