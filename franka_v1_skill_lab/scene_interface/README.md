# scene_interface — V1 场景统一接口

场景管理层。**其他模块必须从这里启动底层 Isaac 场景、读观测、发机器人 joint+夹爪指令。**
进程内 facade（无 ZMQ/ROS）——像正常启动 isaacsim、读状态、发控制一样。

## 0. 统一测试入口（所有测试都在这里跑）

**`test_mode_ui.py` 是统一测试场景**，状态机调试 / FoundationPose / VLA 模型测试都在它里面进行。两行启动：
```bash
# 终端1：测试场景（GUI）
./isaaclab.sh -p projects/franka_v1_skill_lab/scene_interface/test_mode_ui.py
# 终端2：实时图像查看器（网页版，浏览器开 http://localhost:8088）
python projects/franka_v1_skill_lab/scene_interface/image_viewer.py --connect tcp://localhost:5557 --port 8088
```
test_mode 提供：Franka 控制(可达性) / 关节驱动(抽屉·门·咖啡机把手) / 资产实时摆放(含黄框) / 保存场景 /
两相机 RGBD(末端+前面，同视角) / 碰撞·箭头·黄框可视化开关 / **外部控制器接入**。

### 它提供给各模块测试的全部信息（`session.observe()` 每帧）
- **VLA / pi0.5**：`obs.pi05.image`(前视 RGB) / `obs.pi05.wrist_image`(腕部 RGB) / `obs.pi05.state8`(`[arm7, gripper]`)。
- **FoundationPose**：`obs.foundationpose.fp_input("wrist"|"front")`(RGBD+内参+相机世界位姿) / `obs.foundationpose.object_gt_poses` / `obs.cameras["front"|"wrist"]`(rgb+depth 同视角)。
- **状态机**：`obs.raw`(joint_pos/arm_joint_pos/joint_vel/tcp_pose/gripper_width/base_pose_w) / `obs.handles`(门把手·抽屉把手·咖啡机把手在**机器人基坐标系**下的 pose) / `session.provider`(底层 IK/cabinet 等)。
- 实时画面：相机帧经 ZMQ 发到 image_viewer（网页），FoundationPose 处理后的图也往同一总线发即可同屏显示。

### 可视化开关（调试者自选，Visualization 面板 / API）
- `session.set_collision_visible(bool)` — 碰撞体可视化。
- `session.show_target_poses(poses, names=)` / `session.clear_target_poses()` — 目标位姿箭头（任意模块可画自己的目标）。
- `session.set_markers_visible(bool)` — 黄色 InitCorner 区域标记方块。
- UI 里的 **Visualization** 面板有三个复选框现场切换。

### 模块怎么接入测试（两种）
**(A) 把控制接入 test_mode 的 UI（我来测）**：实现一个控制器类（持有 session、每帧产 env action），用 `--controller` 加载：
```python
# 你的模块里，例如 franka_skill_state_machine 提供：
class SkillTestController:
    def __init__(self, session):
        self.s = session            # SceneSession：.env/.provider/.observe()/.step()/可视化开关
    def build_window(self):         # 可选：建自己的 omni.ui 面板(技能切换/目标选择…)
        ...
    def on_reset(self):             # 可选：Reset 时回调
        ...
    def step(self, session):
        # 每帧调；返回一个 env action 张量(用 session.provider.make_joint_action_from_q_des 造)，或 None(交还默认)
        st = session.provider.get_state()
        cmd = my_skill_executor.step(st, dt)
        return session.provider.make_joint_action_from_q_des(cmd.joint_target, cmd.gripper_command)
```
```bash
./isaaclab.sh -p projects/franka_v1_skill_lab/scene_interface/test_mode_ui.py \
    --controller your_pkg.your_module:SkillTestController
```
控制器返回 action 时优先驱动机器人；返回 None 时交还 UI#1/hold。它能用 session 的可视化开关画自己的目标箭头。

**(B) 自己写 entry 用 SceneSession 库**（见 §1 库用法）：完全自主，launch→observe→step。

## 1. 两种用法

### 库用法（推荐，给 pi0.5 / FoundationPose / 技能等模块）
```python
from franka_v1_skill_lab.scene_interface import SceneSession, SceneConfig, ResetMode

cfg = SceneConfig(headless=True, enable_cameras=True, enable_fp=True, reset_mode=ResetMode.REGION)
with SceneSession.launch(cfg) as session:          # AppLauncher 在 launch 内部起
    obs = session.reset(reset_index=0)
    for _ in range(400):
        q_des, grip = my_policy(obs.pi05.image, obs.pi05.wrist_image, obs.pi05.state8)
        obs = session.step(joint_target=q_des, gripper=grip)   # 绝对 7 关节 + 0=open/1=close
```
仍需在 Isaac python 下运行：`./isaaclab.sh -p your_script.py`。

### entry 用法（脚本自建 AppLauncher）
```bash
./isaaclab.sh -p projects/franka_v1_skill_lab/scene_interface/entry.py \
    --headless --enable_cameras --reset_mode region --enable_fp --steps 60 --num_resets 2
```
见 `entry.py`（把已建好的 AppLauncher 传给 `SceneSession.launch(cfg, _app_launcher=...)`）。

## 2. 动作语义（重要）

**绝对关节，不是相对。** `step(joint_target, gripper)`：
- `joint_target` = 7 维**绝对关节弧度**（q_des），与 pi0.5 在 LeRobot 上的 `action` 一致。
- `gripper` = LeRobot/pi0.5 约定 **0=open / 1=close**（也接受 "open"/"close"）。
- 接口内部用 `make_joint_action_from_q_des` 把绝对 q_des 换算成 env 的 raw action
  （`raw=(q_des-offset)/scale`），并把 gripper 0/1 转成 env 的 ±1。调用方**不用关心**这层换算。

`state8`（pi05 块）= `[arm_joint_pos(7), gripper_width(1)]`，与训练数据布局一致。

## 3. 相机模型（两相机，RGB+RGBD，同视角）

两个相机 **front + wrist**，每个是 ONE 物理相机，同时渲染 RGB 和 depth，所以 **VLA 用的 RGB
和 FoundationPose 用的 RGBD 视角完全一致**：
- `obs.cameras["front"|"wrist"]` = `CameraView`（rgb 640×480 + depth + 内参 + 世界位姿）——单一数据源。
- `obs.pi05.image` / `wrist_image` = front/wrist 的 RGB 缩放到 `vla_rgb_size`（默认 256×256）。
- `obs.foundationpose.fp_input("wrist"|"front")` = 同相机的 RGBD 输入包（复用现有 builder）。

落地方式：纯接口层——启动时给 front+wrist 两个 VLA 相机都开 depth、都设 640×480，FP 从同一相机读；
旧的独立 `foundationpose_d435_rgbd` 相机不启用。**不改 sensors 包**。

> ⚠️ 注意（视角后期可调）：旧前视训练相机是 256×256 方形，现在统一成 640×480(4:3) 再缩到 256×256，
> 长宽比变了 → pi0.5 前视输入分布与旧训练数据有差异，必要时需重新对齐/微调。相机分辨率/视角
> 由 `SceneConfig.camera_resolution` / `vla_rgb_size` 调整。

## 4. reset 模式（默认 REGION）

- `REGION`（默认）：cube_1/2/3 + knife 在四黄块 **InitCorner 区域**内随机（动态读 manifest，符合初始化区域契约）。
- `LEGACY`：旧 `SimpleSceneLayoutManager` 硬编码区域（向后兼容旧 eval）。
- `STATIC`：USD 原样，不动物体（调试 / FoundationPose 标定）。

## 5. 观测结构

`obs = session.observe()`（reset/step 也返回）：
- `obs.raw` — `joint_pos(9)` / `arm_joint_pos(7)` / `joint_vel` / `tcp_pose[7]` / `gripper_width`(米，非0/1) / `base_pose_w[7]`(机器人基座世界位姿)
- `obs.pi05` — `image` / `wrist_image` / `state8`
- `obs.cameras` — `{"front","wrist"}` 的 RGBD `CameraView`
- `obs.foundationpose`（enable_fp 时）— 两相机 RGBD + `object_gt_poses`（世界系）+ `fp_input(which)`
- `obs.handles` — 见 §6
- `obs.to_dict()` — 跨进程序列化（只放数值与图像 shape，图像请自行 base64）

## 5b. 加载最新保存场景（默认开）

`SceneConfig.load_latest_scene=True`（默认）：建 env 前用最新 `scene_v1_latest` 的 manifest 覆盖
家电(cabinet/microwave/coffee)、cube、knife 的 `init_state.pos/rot` + `spawn.scale`，使**运行场景 ==
你在布局编辑器里存的最新场景**（base task cfg 里写死的旧位姿被覆盖）。`scene_registry_path` 可指定
非默认 registry；`load_latest_scene=False` 用 base cfg。
> 局限：manager 环境只 spawn base cfg 里已有的成员。你在编辑器里**新增**的 prim（如 `MicrowaveStand`、
> InitCorner 标定块、额外相机）不在 base cfg 里，**不会被 spawn**；只有位姿/缩放被同步。要让新增物体
> 物理存在，需要在 `stack_joint_pos_env_cfg.py` 加对应 scene 成员（另议）。

## 6. 把手 / 拉手 pose（机器人基坐标系，给状态机模块）

`obs.handles`（或轻量 `session.handle_poses_in_base()`，不抓相机）返回：
```python
{
  "microwave_door":  {"asset","link","calibrated","functional",
                      "position":[x,y,z], "quat_wxyz":[w,x,y,z],          # 机器人基座系
                      "position_world":[...], "quat_wxyz_world":[...]},   # 世界系
  "top_drawer": {...}, "middle_drawer": {...}, "bottom_drawer": {...},
  "coffee_lever": {...},   # calibrated=False（偏移未标定，先给 link 原点）
}
```
- 偏移量来自项目已标定的单一可信源（`microwave_door_config` / `custom_drawer_config`），算法与
  `SelectedDrawerObsAdapter` / `microwave_door_skill` 一致（offset 直接套用，不乘 scale）。
- `position`/`quat_wxyz` 已变换到**机器人基坐标系**（`subtract_frame_transforms`）。也给了世界系版本。
- `functional=False`（bottom_drawer）= 当前资产卡住打不开；`calibrated=False`（coffee_lever）= 偏移待标定。

> 物理可开性现状（资产/物理层，**不在本接口**）：碰撞体已存在（mesh→convex_hull）。卡点：
> bottom_drawer 闭合穿透、microwave 门凸包重叠(~10°)、coffee_lever 被 actuator 高刚度锁死。
> 前两个需凸分解重做碰撞体 + GPU 迭代；coffee_lever 需 env actuator 改自由摆动。见交付说明。

## 6. 约束 / 坑

- **一个进程一个自管理 session**：SimulationApp 是进程级单例；多任务用 `num_envs>1` 或多进程。
- **相机需 `--enable_cameras` + GPU**：库用法下 launch 自动设；entry 用法需显式加 flag。
- **控制频率**：`control_hz → decimation=round(100/control_hz)`；`step()`/`reset()` 自动推进 sim_time。
- 场景本身只读：要改场景去 `layout_editor` 点 Save（见 `scene/AI_SCENE_ACCESS.md`）。
