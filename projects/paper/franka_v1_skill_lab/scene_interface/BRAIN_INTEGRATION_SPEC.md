# 大脑 ↔ 场景接口 对接规范（Integration Spec）

> 读者：**开发「大脑」（高层编排器 / VLA / LLM Agent）的工程师**。
> 你不需要读底层源码。本文说清：**能读哪些数据、怎么读、数据格式**；**能下发哪些指令、格式**；
> 以及**大脑与接口的交互规则**。
>
> 接口入口：`franka_v1_skill_lab/scene_interface/brain_interface.py` 的 `BrainInterface`。
> 三个子接口：`brain.perception`（读）、`brain.control`（下指令）、`brain.cognition`（自省/反馈/回合）。

---

## 0. 全局数据约定（先读这张表，后面不再重复）

| 概念 | 约定 | 备注 |
|---|---|---|
| 关节 `q_des` | `list[float]` 长度 7，**绝对关节弧度** | 与机器人手臂 7 轴同序 |
| 夹爪指令（下发） | `0 = 张开 / 1 = 闭合`（整数或浮点），也接受 `"open"/"close"` | 内部自动换算，**你只用 0/1** |
| 夹爪开度（读取） | `float`，单位**米**（两指间距） | `~0.08` 全开，`<0.02` 基本闭合 |
| 四元数 | `list[float]` 长度 4，顺序 **`[w, x, y, z]`** | 唯一例外见 §1.1 `raw.tcp_pose` |
| 位置 | `list[float]` 长度 3，单位**米** `[x, y, z]` | |
| 坐标系 | 默认 `frame="world"`（世界系）；可传 `frame="base"`（机器人基座系） | 大脑做相对推理建议用 `base` |
| 图像 RGB | `numpy.ndarray`，`shape=(H, W, 3)`，`dtype=uint8` | 不是 JSON，是内存数组 |
| 图像 Depth | `numpy.ndarray`，`shape=(H, W)`，`dtype=float32`，单位**米** | `0` 或 `NaN` 表示无效像素 |
| 目标物体引用 | **字符串名**，如 `"cube_1"`、`"knife"`、`"middle_drawer"`、`"microwave"` | 不用 index、不用 prim 路径 |
| 角度/时间 | 弧度 rad / 秒 s | |

除图像（numpy 数组）外，所有读取接口都返回 **JSON 可序列化**的 `dict / list / float / str`。

---

## 1. 怎么拿到接口句柄（连接方式）

### 方式 A：自管理（大脑自己启动场景）

```python
from franka_v1_skill_lab.scene_interface.brain_interface import BrainInterface
from franka_v1_skill_lab.scene_interface.config import SceneConfig, SceneMode

brain = BrainInterface.launch(SceneConfig(
    mode=SceneMode.TEST,        # 或 SceneMode.USE
    enable_cameras=True,        # 要读相机必须 True，且需在 GPU 机器上
    enable_fp=True,             # 要读深度/FoundationPose 必须 True
))
brain.reset()                   # 开一局
# ... 用 brain.perception / brain.control / brain.cognition ...
brain.close()
```

**限制**：一个进程只能有一个自管理场景（仿真器是进程级单例）。

### 方式 B：挂到已运行的场景（推荐用于带 UI/图像查看器联调）

场景由 test_mode 启动，`BrainInterface` 作为 controller 被每帧驱动：

```bash
./isaaclab.sh -p projects/franka_v1_skill_lab/scene_interface/test_mode_ui.py \
    --controller franka_v1_skill_lab.scene_interface.brain_interface:BrainInterface
```

两种方式拿到的 `brain` 对象、三个子接口、所有数据格式**完全一致**。区别只在「谁推进仿真」（见 §4 交互规则）。

---

## 2. 能读哪些数据、怎么读（`brain.perception`）

> 推荐用下面的**细粒度 getter**（返回轻量 dict/数组）。需要一次性拿全部时用 `observe()`（§1.1）。

### 2.1 相机（两路：`"front"` 第三人称静态 / `"wrist"` 腕部随手）

| 调用 | 返回格式 | 示例 |
|---|---|---|
| `get_rgb(camera="front")` | `ndarray(H,W,3) uint8` 或 `None` | — |
| `get_depth(camera="wrist")` | `ndarray(H,W) float32` 米 或 `None` | — |
| `get_rgbd(camera="wrist")` | `(rgb, depth)` 元组 | — |
| `get_camera_intrinsics(camera="front")` | `dict` | `{"fx":615.0,"fy":615.0,"cx":320.0,"cy":240.0,"width":640,"height":480}` |
| `get_camera_pose(camera="front")` | `dict`（世界系光心位姿） | `{"position":[0.9,0.0,0.6],"quat_wxyz":[0.0,0.7,0.0,0.7]}` |

```python
rgb = brain.perception.get_rgb("front")          # (480,640,3) uint8
rgb_w, depth_w = brain.perception.get_rgbd("wrist")
K = brain.perception.get_camera_intrinsics("wrist")
```

> `None` 表示该相机没挂（启动时 `enable_cameras=False` 或非 GPU 环境）。

### 2.2 物体位姿（`cube_1` / `cube_2` / `cube_3` / `knife`）

```python
brain.perception.get_object_pose("cube_1", frame="base")
```
返回（未知物体返回 `None`）：
```json
{"position": [0.45, 0.10, 0.06], "quat_wxyz": [1.0, 0.0, 0.0, 0.0], "frame": "base"}
```
一次拿全部：`list_object_poses(frame="world") -> {"cube_1": {...}, "knife": {...}, ...}`

### 2.3 机器人末端 / 夹爪 / 关节

| 调用 | 返回 |
|---|---|
| `get_ee_pose(frame="world")` | `{"position":[3], "quat_wxyz":[4], "frame":"world"}` |
| `get_gripper_width()` | `float`（米） |
| `get_arm_joints()` | `list[float]` 长度 7 |
| `get_robot_joints()` | `{"pos":[9], "vel":[9], "arm":[7], "gripper_width": float}` |

### 2.4 家电关节（抽屉行程 / 门角度 / 咖啡把手）

```python
brain.perception.get_articulation_joint("cabinet", "joint_1")   # 某抽屉
brain.perception.get_articulation_joint("microwave", "joint_0") # 门铰链(冰箱门为 joint_1)
```
返回（读不到返回 `None`）：
```json
{"asset": "cabinet", "joint": "joint_1", "pos": 0.123, "vel": 0.0}
```
> `pos` 对移动关节（抽屉）是**米**，对旋转关节（门）是**弧度**。`asset ∈ {"cabinet","microwave","coffee_machine"}`。

### 2.5 把手 / 拉手 / 门把手位姿（开关门/抽屉技能的抓取目标）

```python
brain.perception.get_handle_poses(frame="base")
```
返回：
```json
{
  "microwave_door": {"asset":"microwave","link":"link_0","calibrated":true,"functional":true,
                     "position":[0.42,-0.18,0.31],"quat_wxyz":[0.5,0.5,-0.5,0.5]},
  "middle_drawer":  {"asset":"cabinet","link":"link_1","calibrated":true,"functional":true,
                     "position":[...],"quat_wxyz":[...]},
  "top_drawer": {...}, "bottom_drawer": {...}, "coffee_lever": {...}
}
```
只返回场景中实际存在的把手。`frame="world"` 取世界系。

### 2.6 FoundationPose（感知位姿估计）

| 调用 | 说明 |
|---|---|
| `get_foundationpose_input(which="wrist", dump_dir=None) -> dict\|None` | 把同视角 RGBD + 内外参 + 物体清单打包成 FP 进程的输入 bundle |
| `set_pose_estimator(fn)` | 注入真实 FP 回调：`fn(name, fp_input) -> {"position":[3],"quat_wxyz":[4]}` 或 `4x4` 矩阵 |
| `get_estimated_pose(name, which="wrist") -> dict\|None` | 估计位姿 |

`get_estimated_pose` 返回：
```json
{"position":[...], "quat_wxyz":[...], "pose_source":"foundationpose"}
```
**未注入** estimator 时退化为仿真真值，并标注来源：
```json
{"position":[...], "quat_wxyz":[...], "frame":"world", "pose_source":"sim_gt"}
```

### 1.1 一次性全量观测 `observe()`（可选）

`brain.perception.observe()` 返回一个 `Observation` 对象（Python dataclass，按属性访问）：

| 字段 | 类型 / 格式 |
|---|---|
| `obs.pi05.image` | front RGB，`ndarray` 已缩放到 `vla_rgb_size`（默认 256x256x3 uint8）或 `None` |
| `obs.pi05.wrist_image` | wrist RGB，同上 |
| `obs.pi05.state8` | `list[float]` 长度 8 = `arm_joint_pos[7] + [gripper_width]` |
| `obs.raw.joint_pos` / `joint_vel` | `list[float]` 长度 9 |
| `obs.raw.arm_joint_pos` | `list[float]` 长度 7 |
| `obs.raw.tcp_pose` | `list[float]` 长度 7，顺序 **`[x,y,z, qx,qy,qz, qw]`**（注意是 xyzw，便捷量） |
| `obs.raw.gripper_width` | `float` 米 |
| `obs.raw.base_pose_w` | `list[float]` 长度 7，`[x,y,z, qw,qx,qy,qz]` |
| `obs.cameras["front"]` / `["wrist"]` | `CameraView`：`.rgb`、`.depth`、`.intrinsics`、`.camera_pose_world` |
| `obs.foundationpose` | `FoundationPoseBlock` 或 `None`；`.fp_input(which="wrist")` 取 FP 输入 |
| `obs.handles` | 同 §2.5（base 系） |
| `obs.sim_time` | `float` 秒 |

> `observe()` 会触发抓取两路相机，开销大；只读单项时优先用 §2 的 getter。

---

## 3. 能下发哪些指令、格式（`brain.control`）

接口提供 6 个技能（动词）。**目标用字符串名**，取值域见 §5 `list_*`。

| skill（字符串） | 必填目标 | 目标取值 | 额外参数 |
|---|---|---|---|
| `"grasp"` | 目标物体 | `cube_1/cube_2/cube_3/knife` | — |
| `"place"` | 否（用 target_pose） | — | `target_pose=[x,y,z]`（env-local，默认 `[0.45,0,0.06]`） |
| `"open_drawer"` | 抽屉名 | `middle_drawer/top_drawer/bottom_drawer` | `params={"drawer_link":"link_1"}`（默认） |
| `"close_drawer"` | 抽屉名 | 同上 | 同上 |
| `"open_door"` | 门名 | `microwave` | — |
| `"close_door"` | 门名 | `microwave` | — |

### 3.1 下指令（一步到位，最常用）

```python
request_id = brain.control.command_skill("grasp", target="cube_1")
request_id = brain.control.command_skill("place", target_pose=[0.45, 0.10, 0.06])
request_id = brain.control.command_skill("open_drawer", target="middle_drawer")
```
- 返回 `request_id`（字符串），用于在 `last_result()` 里对账。
- **指令是「入队」**：调用后不会立刻动，下一次推进（`tick()`/`run_skill()`/controller 的 `step`）才真正启动。见 §4。

### 3.2 下指令（分步选择，等价）

```python
brain.control.select_skill("open_drawer")
brain.control.set_target_object("middle_drawer")
brain.control.set_params(drawer_link="link_1")
request_id = brain.control.start()
```

### 3.3 运行时控制

| 调用 | 作用 |
|---|---|
| `pause()` | 暂停当前技能（冻结在当前姿态，之后推进只 hold，不前进） |
| `resume() -> bool` | 继续被暂停的技能（返回是否成功） |
| `stop()` | 软停（= pause，**保留**已抓物体上下文） |
| `abort()` | 硬中止：丢弃当前技能 + 清空状态机，回 idle（**不保留**抓取上下文） |
| `is_busy() -> bool` | 是否有技能在跑或待启动 |

### 3.4 低层透传（可选，绕过技能直接发关节）

```python
brain.control.command_joints([j1,j2,j3,j4,j5,j6,j7], gripper=0)   # 一次性手动关节
```
> 仅在**没有技能在跑**时生效；用于大脑自己做 IK/轨迹时逐帧伺服（每帧调一次）。

---

## 4. 大脑 ↔ 接口 交互规则（务必遵守）

1. **仿真靠「推进」驱动，指令是入队**。
   - 方式 A（自管理）：你必须主动推进。两种：
     - 阻塞式 `brain.run_skill(skill, target=..., ...) -> result_dict`：下指令并**跑到技能结束**，返回结果。
     - 手动式：`command_skill(...)` 后循环 `brain.tick()`，直到 `is_busy()` 变 False。`tick()` 每调一次推进**一个仿真步**并返回状态快照。
   - 方式 B（controller）：场景每帧自动调 `step()`，你**只管下指令 + 轮询状态**，不要也不能自己 `tick()`。

2. **同一时刻只有一个技能**。在一个技能进行中再 `command_skill(...)` 会**抢占/替换**当前技能（旧技能被暂停）。要干净结束请先 `abort()`。

3. **下指令前先校验**：`brain.cognition.can_execute(skill, target)` 返回 `{"ok":bool,"reason":str}`。
   规则：`grasp` 需要手是空的；`place` 需要手里有物体；门/抽屉/抓取目标必须在合法取值域内。

4. **闭环靠反馈**（见 §5）：
   - 进行中轮询 `skill_status()`（`active/phase/runtime_status/held_object/busy`）。
   - 结束后读 `last_result()`（`success/status/failure_reason/...`），据此决定重试或换策略。

5. **一个回合**：`brain.reset(reset_mode=..., seed=...)` 重置布局并清空状态机，返回首帧 `Observation`。
   `reset_mode ∈ {"static","region","legacy"}`（默认 `region`：物体在初始化区域内随机）。

6. **线程模型**：接口**非线程安全**。若大脑在另一线程下指令，请保证下指令与推进不并发写状态
   （典型做法：大脑线程只调 `control.command_skill/pause/...` 和 `cognition/perception` 读，推进在主线程）。

7. **错误处理**：
   - 非法 skill 字符串 → `command_skill` 抛 `ValueError`（先用 `list_skills()` 取合法值）。
   - 非法/缺失 target → 建议先 `can_execute` 拦截；若直接下发，接口会用默认目标兜底（行为可能非预期）。
   - 相机/FP 未启用 → 对应读取返回 `None`，不抛错。

### 典型闭环（伪代码）

```python
brain.reset(reset_mode="region")

# 1) 感知
cube = brain.perception.get_object_pose("cube_1", frame="base")

# 2) 校验
chk = brain.cognition.can_execute("grasp", "cube_1")
if not chk["ok"]:
    handle(chk["reason"]); ...

# 3) 下指令并等结果（阻塞式）
res = brain.run_skill("grasp", target="cube_1")
if res["success"]:
    res2 = brain.run_skill("place", target_pose=[0.45, 0.10, 0.06])

# —— 或者非阻塞，边跑边监控/可打断 ——
brain.control.command_skill("open_door", target="microwave")
while brain.control.is_busy():
    st = brain.tick()                         # 推进一步
    if need_replan(brain.perception.get_depth("wrist")):
        brain.control.pause(); ...; brain.control.resume()
print(brain.cognition.last_result())
```

---

## 5. 自省 / 反馈接口（`brain.cognition`）

### 5.1 能力发现（让大脑动态读出动作空间，不要写死）

| 调用 | 返回 |
|---|---|
| `list_skills()` | `[{"skill","target_kind","needs_target","params","desc","success"}, ...]` |
| `list_objects()` | `{"graspable":[...],"drawer":[...],"door":[...],"place_point":[...]}` |
| `list_targets(skill)` | 该技能合法目标 `list[str]` |
| `describe_skill(skill)` | 单个技能描述（同 `list_skills` 的元素） |

`list_skills()` 元素示例：
```json
{"skill":"open_drawer","target_kind":"drawer","needs_target":true,
 "params":{"drawer_link":"link_1 (default)"},
 "desc":"Grasp drawer handle and pull the cabinet drawer open (pure-physical).",
 "success":"drawer prismatic joint past open threshold"}
```

### 5.2 前置校验

```python
brain.cognition.can_execute("place", target=None)
# -> {"ok": false, "reason": "place needs a held object; nothing is grasped"}
```

### 5.3 执行反馈

`skill_status()`：
```json
{"active": true, "phase": "MOVE_TO_PRE_GRASP", "runtime_status": "running",
 "held_object": null, "busy": true}
```
`is_holding() -> "cube_1"` 或 `None`。

`last_result()`（还没跑过返回 `None`）：
```json
{"request_id":"grasp_cube_1_173...","skill":"grasp","target":"cube_1",
 "success":true,"status":"succeeded","failure_reason":null,
 "elapsed":3.42,"position_error":0.004,"orientation_error":0.01,"gripper_width":0.021}
```
`status ∈ {"succeeded","failed","stopped","not_implemented"}`；失败时 `failure_reason` 形如
`"IK_UNREACHABLE"`、`"GRASP_VERIFICATION_FAILED"`、`"POSITION_TIMEOUT"` 等。

### 5.4 回合 / 可视化

| 调用 | 作用 |
|---|---|
| `reset(reset_mode=None, seed=None) -> Observation` | 重置布局 + 清状态机 |
| `settle(steps=8)` | 空跑稳定物理/渲染 |
| `save_scene(note=None) -> dict` | 当前布局存为最新场景（机器人姿态不存） |
| `show_targets(poses, names=None)` / `clear_targets()` | 目标位姿箭头（仅 TEST/GUI 可见） |
| `set_collision_visible(on)` / `set_markers_visible(on)` | 碰撞体 / 初始化区域块显隐 |

---

## 6. 速查：方法 → 返回类型

| 接口 | 方法 | 返回 |
|---|---|---|
| perception | `get_rgb/get_depth` | `ndarray` 或 `None` |
| perception | `get_object_pose/get_ee_pose/get_camera_pose/get_estimated_pose` | `dict{position,quat_wxyz,...}` 或 `None` |
| perception | `get_gripper_width` | `float` |
| perception | `get_arm_joints` | `list[7]` |
| perception | `get_robot_joints` | `dict` |
| perception | `get_articulation_joint` | `dict` 或 `None` |
| perception | `get_handle_poses` | `dict{name->pose}` |
| control | `command_skill/start` | `str`（request_id） |
| control | `resume/is_busy` | `bool` |
| control | `select_skill/set_*/pause/stop/abort/command_joints` | `None` |
| cognition | `list_skills/list_objects/list_targets/describe_skill` | `list/dict` |
| cognition | `can_execute/skill_status/last_result` | `dict`（`last_result` 可能 `None`） |
| cognition | `is_holding` | `str` 或 `None` |
| BrainInterface | `tick` | `dict`（skill_status 快照） |
| BrainInterface | `run_skill` | `dict`（last_result）或 `None` |
| BrainInterface | `reset/observe` | `Observation` |

---

## 8. 大脑如何对接 / 部署（输出规范 + 即插即用骨架）

**部署模型**：你的大脑（任意框架：VLA / LLM / 策略网络）打包成一个 Python 对象，**进程内**
和本接口交互——读观测、出命令。没有 ZMQ/HTTP，零拷贝、低延迟。你只需做一件事：
**把你大脑的输出整理成下面的「命令 dict」格式**，其余由接口负责。

### 8.1 大脑输入：一帧观测 dict

调 `brain.observation_dict(frame="base", include_images=True)` 得到（图像是 numpy 数组，进程内直传）：
```python
{
  "sim_time": 12.34,
  "ee_pose":     {"position":[...], "quat_wxyz":[...], "frame":"base"},
  "gripper_width": 0.021,
  "arm_joints":  [j1,...,j7],
  "objects":     {"cube_1":{"position":[...],"quat_wxyz":[...],"frame":"base"}, "knife":{...}, ...},
  "handles":     {"microwave_door":{...}, "middle_drawer":{...}, ...},
  "skill_status":{"active":false,"phase":"IDLE","runtime_status":"idle","held_object":null,"busy":false},
  "is_holding":  null,
  "rgb_front":   ndarray(H,W,3) uint8,     # include_images=True 时
  "rgb_wrist":   ndarray(H,W,3) uint8,
  "depth_wrist": ndarray(H,W) float32
}
```

### 8.2 大脑输出：一条命令 dict（**你要对齐的就是这个 schema**）

```python
# 下技能（最常用）
{"op":"command", "skill":"grasp", "target":"cube_1"}
{"op":"command", "skill":"place", "target_pose":[0.45,0.10,0.06]}
{"op":"command", "skill":"open_drawer", "target":"middle_drawer", "params":{"drawer_link":"link_1"}}

# 运行时控制
{"op":"pause"} | {"op":"resume"} | {"op":"stop"} | {"op":"abort"}

# 回合 / 低层 / 空操作 / 结束
{"op":"reset", "reset_mode":"region"}
{"op":"joints", "q_des":[7 floats], "gripper":0}     # 绕过技能，直接发关节
{"op":"noop"}                                         # 本帧不下新命令，继续推进
{"op":"halt"}                                         # 结束 run_policy 循环
```
字段取值域同 §3（skill/target/target_pose/params）。`op` 缺省为 `"command"`。

把命令交给接口：`ack = brain.apply_command(cmd)`，返回
`{"ok":true, "op":"command", "request_id":"..."}` 或 `{"ok":false, "op":..., "reason":"..."}`
（接口会先用 `can_execute` 校验，非法命令不会执行，直接回 `ok:false` + 原因）。

### 8.3 即插即用：实现一个 `act(obs)->cmd` 就能跑

你的大脑只要实现一个方法：

```python
class MyBrain:
    def act(self, obs: dict) -> dict | None:
        # 你的模型在这里：读 obs，产出一条命令 dict
        if obs["is_holding"] is None and not obs["skill_status"]["busy"]:
            return {"op": "command", "skill": "grasp", "target": "cube_1"}
        return {"op": "noop"}        # 没有新命令就让仿真继续推进
```

部署到我的电脑后，一行驱动：

```python
from franka_v1_skill_lab.scene_interface.brain_interface import BrainInterface
from franka_v1_skill_lab.scene_interface.config import SceneConfig, SceneMode

brain = BrainInterface.launch(SceneConfig(mode=SceneMode.TEST, enable_cameras=True, enable_fp=True))
brain.reset(reset_mode="region")
result = brain.run_policy(MyBrain(), obs_frame="base", include_images=True)
print(result)         # {"steps": N, "last_result": {...}}
brain.close()
```

`run_policy` 每帧自动：`build obs -> your_brain.act(obs) -> apply_command -> tick`，直到你的大脑
返回 `{"op":"halt"}`。**你只写 `act`，不碰仿真循环。**

> 不想用 `run_policy` 也行：自己循环 `obs = brain.observation_dict(); cmd = my_brain.act(obs);
> brain.apply_command(cmd); brain.tick()`——完全等价，方便你插自定义日志/打断逻辑。

### 8.4 对接 checklist（交付给大脑开发者）

1. 你的模型输出 → 映射成 §8.2 的命令 dict（skill 名用 `list_skills()` 的合法值；target 用 `list_targets(skill)`）。
2. 实现 `act(obs: dict) -> cmd_dict | None`。
3. 不确定能不能下某命令时，先看 obs 里的 `skill_status / is_holding`，或调 `brain.cognition.can_execute(skill, target)`。
4. 用 `brain.cognition.last_result()` 拿成败做闭环；失败看 `failure_reason`。
5. 交付物：一个含 `act` 的 Python 类 + 它的依赖（放进我机器的 venv）。其余我这边 `run_policy` 跑起来即可。

---

## 7. 启动配置要点（`SceneConfig` 与读取能力的关系）

| 你想读的数据 | 启动时必须 |
|---|---|
| 相机 RGB | `enable_cameras=True` + GPU |
| 深度 / RGBD / FoundationPose 输入 | `enable_cameras=True` + `enable_fp=True` |
| 物体/关节/把手/末端位姿 | 无特殊要求（始终可读） |
| 目标箭头 / 碰撞体可视化 | `mode=SceneMode.TEST` 且非 headless（要看见） |

其余字段（任务、设备、布局、冰箱替换等）见 `config.py` 的 `SceneConfig`，对数据格式无影响。

---

**配套文档**：`BRAIN_INTERFACE_API.md`（逐方法签名速查）、`BRAIN_INTERFACE_GUIDE.md`（设计动机与三层划分）。
本文是给大脑开发者的对接契约，**以本文的数据格式为准**。
