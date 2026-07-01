# BrainInterface 说明文档（设计 + 用法）

> 配套：[BRAIN_INTERFACE_API.md](BRAIN_INTERFACE_API.md)（逐方法签名）。
> 代码：`scene_interface/brain_interface.py`。

## 1. 这是什么 / 为什么

你已经有了一整套场景模块：`SceneSession`（进程内启动场景、读观测、发关节指令）、状态机
`SkillExecutor`（抓取/放置/开关门/开关抽屉技能）、两路相机、FoundationPose 桥。但这些散在
不同文件里、各有各的调用方式。**`BrainInterface` 把它们收口成一个面向「大脑」的统一门面**，
让高层编排器（或 LLM Agent）不用关心底层接线，只面对三个职责清晰的类：

```
大脑 (LLM / 编排器)
   │  perceive            command              introspect / feedback
   ▼                        ▼                        ▼
ScenePerception ───►  SkillControl  ───►  SkillExecutor(状态机)  ──► env
   ▲                        │                        │
   └──────── TaskCognition（能力发现 / 前置校验 / 执行反馈 / 回合 / 可视化）
```

- **ScenePerception（感知类）**：大脑的「眼睛」。读相机、物体位姿、关节、夹爪、把手、FoundationPose。
- **SkillControl（控制类）**：大脑的「嘴」。和状态机对齐——选技能、选目标物体、选目标位置、夹爪、
  暂停/继续/停止/中止。**这就是「大脑给状态机下达指令」**。
- **TaskCognition（认知类）= 你原本没考虑到的那层**：大脑的「自我意识 + 反馈」。详见第 4 节。

## 2. 三句话上手

```python
from franka_v1_skill_lab.scene_interface.brain_interface import BrainInterface
from franka_v1_skill_lab.scene_interface.config import SceneConfig, SceneMode

brain = BrainInterface.launch(SceneConfig(mode=SceneMode.TEST, enable_cameras=True,
                                          replace_microwave_with_fridge=True, lock_knife=True))
brain.reset()                                   # 随机布局 + 清状态机
res = brain.run_skill("open_door", target="microwave")   # 阻塞跑完一个技能
print(res)                                      # {success, status, failure_reason, ...}
brain.close()
```

> 注意：库用法下必须在 **GPU 机器**、且 `SceneConfig.enable_cameras=True` 时 `launch` 才会挂相机。
> 一个进程只能有一个自管理 session（SimulationApp 是进程级单例）。

## 3. 两种运行方式

### 方式 A：自管理循环（headless 大脑 / 脚本 / 评测）

```python
brain = BrainInterface.launch(cfg)
brain.reset()

# 闭环：感知 -> 决策 -> 下指令 -> 等反馈
rgb   = brain.perception.get_rgb("front")
cube  = brain.perception.get_object_pose("cube_1", frame="base")
check = brain.cognition.can_execute("grasp", "cube_1")     # {'ok':True,...}
if check["ok"]:
    brain.run_skill("grasp", target="cube_1")
    brain.run_skill("place", target_pose=[0.45, 0.10, 0.06])
print(brain.cognition.last_result())
brain.close()
```

需要**边跑边感知 / 可随时打断**时，别用阻塞的 `run_skill`，自己 tick：

```python
brain.control.command_skill("open_drawer", target="middle_drawer")
while brain.control.is_busy():
    status = brain.tick()                 # 推进一步，返回 {active,phase,runtime_status,...}
    if some_condition(brain.perception.get_depth("wrist")):
        brain.control.pause()             # 暂停
        ...                               # 大脑重新规划
        brain.control.resume()            # 继续
res = brain.cognition.last_result()
```

### 方式 B：作为 test_mode 的 `--controller`（复用现有两行启动 + UI + 图像查看器）

```bash
# 终端 1：场景 + 状态机（BrainInterface 直接当 controller）
./isaaclab.sh -p projects/franka_v1_skill_lab/scene_interface/test_mode_ui.py \
    --controller franka_v1_skill_lab.scene_interface.brain_interface:BrainInterface

# 终端 2：外部实时图像查看器（front/wrist + FoundationPose 处理后图像）
python projects/franka_v1_skill_lab/scene_interface/image_viewer.py   # http://localhost:8088
```

test_mode 会 `BrainInterface(session)`，每帧调 `step(session)->action` 驱动机器人。此时
`build_window/on_reset/step` 这三个 hook 已内建实现，无需你写适配器。

> 想让大脑在跑 UI 的同时下指令：把这个 `BrainInterface` 实例暴露给你的大脑线程/进程，
> 调 `brain.control.command_skill(...)` 即可（控制状态是共享的，`step` 每帧消费）。

## 4. 第三个类：我补的「认知层」TaskCognition —— 为什么需要它

感知类只回答「世界现在长什么样」，控制类只负责「去做某事」。但一个**自主大脑闭环**还缺四样
东西，它们既不是传感器读数也不是技能指令，所以单独成一类：

1. **能力发现（动作空间自省）**：LLM 大脑不该把技能名/可选目标写死。`list_skills()` /
   `list_objects()` / `list_targets(skill)` / `describe_skill(skill)` 让大脑**动态读出**
   「我有哪些动词、对哪些名词、每个要什么参数、成功判据是什么」。

2. **前置校验**：`can_execute(skill, target)` 在真正下指令前回答「这条命令现在合不合法」——
   目标存在吗？grasp 时手是空的吗？place 时手里有东西吗？避免无效指令浪费一整段执行。

3. **执行反馈（闭环必需）**：`skill_status()`（跑到哪一步）+ `last_result()`（成没成、失败原因、
   位姿误差）。没有反馈，大脑就是开环瞎指挥；有了它才能「失败→换策略」。

4. **回合 + 可视化管理**：`reset()`（随机布局开新回合）、`save_scene()`（存最新布局）、
   `settle()`、以及 `set_collision_visible/show_targets` 等调试开关。这些是运行一个 episode
   绕不开、但不属于感知/控制的运维动作。

> 一句话：**感知=看，控制=做，认知=知道自己能做什么、做得怎么样、以及管好这一局**。

## 5. 关键约定（务必记住）

- **关节**：7 维绝对弧度 `q_des`。**夹爪**：对外 `0=open/1=close`（也收 `"open"/"close"`）。
- **四元数**：`get_*` 一律 **wxyz**（唯一例外：`Observation.raw.tcp_pose` 是 `xyzw` 便捷量）。
- **坐标系**：默认世界系；传 `frame="base"` 得到机器人基座系（大脑做相对推理更稳）。
- **目标用字符串名**：`cube_1`/`knife`/`middle_drawer`/`microwave`。门成员名永远是 `microwave`
  （即便资产被替换成冰箱）——门关节由门技能自己解析。
- **指令是入队**：`command_skill` 不立即执行，下一次 `tick()/step()` 才用当帧状态启动；这样和
  状态机「每帧拿最新 SceneState」对齐，也保证 pause/abort 时序正确。

## 6. 现状 / 边界（哪些是真实的，哪些要你接）

| 能力 | 状态 |
|---|---|
| 感知：相机 RGB/Depth/RGBD、物体/EE/夹爪/关节/把手位姿 | ✅ 直接可用（需挂相机） |
| 控制：grasp/place/open-close door/drawer + pause/resume/stop/abort | ✅ 复用现有状态机（与 `SkillTestController` 同一套 backend） |
| 认知：能力发现 / 前置校验 / 执行反馈 / reset / save / 可视化 | ✅ 可用 |
| `get_estimated_pose` 用**真实** FoundationPose | ⚠️ 需 `set_pose_estimator(fn)` 注入 FP 进程回调；不注入则返回仿真 GT（`pose_source="sim_gt"`） |
| place 的目标点 | 目前用 `target_surface_xyz`（env-local）；非 place 技能的位姿由场景几何自动求解（忽略 `target_pose`） |
| `abort` | 硬中止 = 丢弃技能 + 清空状态机（不保留 held 上下文）；`stop` 是软停（保留 held） |

## 7. 文件位置

- 接口实现：`scene_interface/brain_interface.py`
- 接口文档：`scene_interface/BRAIN_INTERFACE_API.md`
- 本说明：`scene_interface/BRAIN_INTERFACE_GUIDE.md`
- 底层：`scene_interface/session.py`（SceneSession）、`runtime/scene_state_provider.py`（SceneStateProvider）、
  `state_machine/skill_executor.py`（SkillExecutor）、`state_machine/skill_test_controller.py`（控制接线参考）。
