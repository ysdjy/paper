# paper — Franka 技能状态机实验平台 + 部署条件化计划评估

本仓库是**论文实验专用**的独立工作区，包含：一个基于 Isaac Lab 的 **Franka 关节动作技能状态机**（抓取 / 放置 / 开关抽屉 / 开关门 / 咖啡机），配套的**测试场景与可视化 UI**，以及论文方法层 **Deployment-Conditioned Plan Evaluation（部署条件化计划评估）**。

> 与"感知/记忆"等其它项目**代码完全隔离**、独立运行。仓库根即 `projects/paper` 的内容。

---

## 目录结构

```
paper/
├── franka_v1_skill_lab/        # 测试平台/工坊（场景、UI、传感器、感知、遥操作、pi0.5）
│   ├── scene_interface/        #   test_mode_ui.py + skill_test_controller + 各 UI 面板
│   ├── layout_editor/          #   场景布局编辑器（保存/复现 scene_v1_latest）
│   ├── scene/                  #   任务 id、保存的场景 (saved_scenes/v1_active/*)
│   ├── sensors/                #   D435 / ZED 相机
│   ├── perception_foundationpose/, perception_qwen/, scene_describer/   # 感知后端
│   ├── teleop_collection/      #   GELLO 遥操作数据采集
│   └── pi05_training/          #   pi0.5 训练/转换
├── franka_skill_state_machine/ # 技能后端（执行状态机，纯物理 IK，无作弊关节目标）
│   ├── skills/                 #   grasp / place / open_drawer / close_drawer / microwave_door / move_to_pose
│   ├── runtime/                #   IK 适配、抽屉/门配置、观测适配、日志、参数解析
│   ├── state_machine/          #   skill_executor + skill_test_controller（技能测试主控 + coffee）
│   └── learned_drawer/         #   官方抽屉 RL 策略（备选，默认用 IK）
├── deployment_calibration/     # 论文方法层：契约/采样/适配/基线/评估 + 数据
├── scene/                      # 冻结的场景任务副本（stackpkg + paper_tasks + captured_scene）
└── docs/                       # 平台审计 / 实验契约 / kill test / 设计文档
```

---

## 环境准备

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
cd <IsaacLab>/projects/paper
```

底层基座任务：`Isaac-Stack-Cube-Franka-JointPolicy-v0`（8 维关节动作：7 臂 + 1 夹爪 + 1 抽屉标志；20Hz；DLS 微分 IK）。技能全程**纯物理抓取拉动**，不写关节目标作弊。

---

## 快速开始

### 1) 可视化测试（GUI + 抓取面板 + 碰撞体可视化）
```bash
./isaaclab.sh -p franka_v1_skill_lab/scene_interface/test_mode_ui.py \
  --controller state_machine.skill_test_controller:SkillTestController \
  --grasp --viz --scene --no_cameras --no_stream
```
- 技能面板：选目标（桌面物体/把手/抽屉/门/咖啡机）→ 执行对应技能
- Grasp Pose 面板：调各把手/物体的抓取位姿（**技能实时取用**，单一来源）
- Viz 面板：Show Colliders（绿色碰撞体）/ Show grasp blocks / 目标箭头

### 2) 无 GUI 自动回归（出成功/失败）
```bash
SKILL_TEST_AUTORUN="open_drawer:top_drawer,close_drawer:top_drawer,open_drawer:sektion_top_drawer,close_drawer:sektion_top_drawer" \
./isaaclab.sh -p franka_v1_skill_lab/scene_interface/test_mode_ui.py \
  --controller state_machine.skill_test_controller:SkillTestController \
  --headless --no_cameras
```

### 3) 咖啡机技能（夹住把手方块，绕竖直世界 Z 轴旋转 joint_5）
```bash
SKILL_TEST_COFFEE=1 SKILL_TEST_COFFEE_TARGET=0.55 \
./isaaclab.sh -p franka_v1_skill_lab/scene_interface/test_mode_ui.py \
  --controller state_machine.skill_test_controller:SkillTestController \
  --headless --no_cameras
```

### 4) 看相机画面（网页版）
去掉 `--no_cameras`/`--no_stream` 启动仿真；另一终端：
```bash
python franka_v1_skill_lab/scene_interface/image_viewer.py --connect tcp://localhost:5557 --port 8088
# 浏览器打开 http://localhost:8088  （front/wrist RGB+Depth）
```

---

## UI 面板（`--ui all` 或单独 `--franka/--asset/--grasp/...`；默认全关）

| 标志 | 面板 | 功能 |
|---|---|---|
| `--franka` | 机器人控制 | 关节/任务空间读写 + 执行 |
| `--asset` | 资产位姿编辑 | 实时移动/旋转/缩放场景物体 |
| `--grasp` | 抓取位姿编辑 | 物体+把手抓取 pose（抽屉/咖啡技能取此值） |
| `--waypoints` | 技能途径点 | 每个技能的过渡绕行途径点 |
| `--joint` | 家电关节驱动 | 抽屉/门/咖啡机关节目标 |
| `--camera` | 相机视角 | 调相机位姿（需相机） |
| `--scene` | 场景 | 重置 + 保存最新场景 |
| `--viz` | 可视化 | 碰撞体 / 目标箭头 / grasp blocks |

其它：`--keep_fridge`（保留冰箱，门技能 target=fridge 需要）、`--keep_dishwasher`、`--no_props`、`--describe`（VLM 场景描述）。

---

## 技能清单与状态

| 技能 | 目标 | 状态 |
|---|---|---|
| **抓取 grasp** | 桌面物体 / 碗 | ✅ IK 抓取（rest_offset 防穿模） |
| **放置 place** | KLT 筐 / 指定点 | ✅ |
| **开抽屉 open_drawer** | 桌面柜 `top/middle_drawer`、地面 Sektion `sektion_top/bottom_drawer` | ✅ 先 ARC_TO_FACE 转身对正 → 抓把手 → 沿抓取轴直线拉 |
| **关抽屉 close_drawer** | 同上 | ✅ 先关节空间 ARC_TO_FACE 转身（不自撞）→ 抓 → 推回 |
| **咖啡机 coffee** | `handle_coffee_lever` | ✅ 夹把手方块，绕**竖直世界 Z 轴**(joint_5) 闭环旋转到目标角 |
| **开关门 door** | 微波炉 / 冰箱(`--keep_fridge`) | ✅ 自动定位把手 + 铰链摆动 |

**关键机制**
- **单一来源把手 pose**：抽屉抓取姿态**只**取 Grasp Pose 面板 / `grasp_poses.json` 的值（`override_grasp_local` 实时注入 obs_adapter）；已删除代理/网格/计算等其它 pose 来源，杜绝分叉。
- **ARC_TO_FACE**：够不到/斜后方的柜子先**关节空间旋转第 1 轴**把整体姿态转向柜子，再规划抓取，避免仰身/自撞。
- **咖啡机世界 Z 轴**：`joint_5/link_5` 实测绕世界 Z 水平摆动；直接用世界 Z 轴 + 关节锚点做闭环旋转（不再靠脆弱的探测标定）。
- **防穿模**：桌面刚体 prop spawn 时 `rest_offset=0.002` + 提高解穿插速度，夹取不再穿入。

---

## deployment_calibration —— 论文方法层

预测多维技能执行结果 `y`，输入 `(x, g, θ, H)`：部署状态 `x`、计划 `g`、扰动 `θ`、近期历史 `H`。

```
contracts/       episode schema（契约版本、θ 范围）
data_generation/ 采样器 + generate_open_drawer.py（每 episode 全复位，独立性已修）
adapters/        Isaac 环境适配（reset_full + run episode）
baselines/       B0 / B1 / B2(+history)
evaluation/      AUROC / regret 评估
data/            run 输出（episodes.jsonl + trajectories/*.npz）
```

Round-1 结论见 `docs/paper_round1_kill_test_v1.md`（裁决 MODIFY：管线可信，干净 sim 无可校准漂移 → 需注入 per-session 隐藏 damping/friction 漂移）。

---

## 备注

- 场景资产（Cabinet_44853 / CoffeeMachine / SAPIEN props / Franka USD）从共享 Isaac Lab 仓库树解析——本仓库只含**代码 + 场景清单 + 抓取标定**，不含大资产二进制。
- `scene/saved_scenes/v1_active/scene_v1_latest.usd` 为布局编辑器的场景快照（可由 "Save V1" 重生）。
- 已排除 `__pycache__` / 传感器 debug 输出 / logs（见 `.gitignore`）。
- `projects/paper` 为独立 git 仓库；推送：`cd projects/paper && git add -A && git commit -m "..." && git push`。
