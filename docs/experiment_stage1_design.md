# 第一阶段实验设计 (experiment_stage1_design)

本文件定义放置 / 打开抽屉两个技能的数据生成实验设计：变量分层、控制事实、参数分类与范围、采样设计、
成功判据与脱离判据的依据、统一数据 schema。实现见 `experiment_stage1_implementation.md`，
结果见 `experiment_stage1_sensitivity_report.md`。

## 1. 变量严格分层 (x, g, θ, y)

每条 episode 的 JSON 把四组变量分开存储（`runtime/episode_logger.py`），**任务目标绝不混入参数字典**：

| 组 | 字段名 | 放置 | 抽屉 |
|----|--------|------|------|
| 初始状态/环境 x | `initial_state` | initial_robot_joint_position, initial_tcp_pose, initial_object_pose, object_to_tcp_transform, object_type | initial_robot_joint_position, initial_tcp_pose, initial_drawer_position, drawer_name, handle_pose |
| 任务目标 g | `task_target` | target_surface_xyz, task_target_object_pose | target_open_position, drawer_name |
| 执行参数 θ | `execution_parameters`(+`requested_parameters`) | 见 §3 | 见 §3 |
| 结果 y | `outcomes` | 见 §6 | 见 §6 |

`(x, g, θ) → y`。`requested_parameters` 是用户请求原值；`execution_parameters` 是带 provenance 的
生效值（`ParamTrace`：requested/default/effective/min/max/clamped/rejected/source/applied_at）。

## 2. 真实控制事实（审计确认，硬编码进 controller 字段）

```
physics_dt        = 0.01 s        (stack_env_cfg.py)
decimation        = 5
control_dt        = 0.05 s        (= physics_dt × decimation)
control_frequency = 20 Hz
arm action        = JointPositionAction(scale=1.0, use_default_offset=True), 7+1 维
IK                = DLS, command_type=pose, absolute, max_joint_step=0.20 rad/step
```

- `max_position_step` 等称为 **command step size / nominal Cartesian progression / TCP target update step**，
  **不**等于真实末端速度。真实速度由相邻轨迹帧测得：
  `measured_tcp_linear_speed = ||p_t − p_{t−1}|| / control_dt`，角速度用四元数测地角 / control_dt
  （`runtime/trajectory_logger.py`）。
- 全局 speed scale 固定 1.0（pilot runner 强制），保证可复现。

## 3. 执行参数分类与第一轮范围

按物理含义分四类（§11 of task）。前三类是科研执行参数 θ，判定类只记录+固定。

### 放置 θ（围绕真实默认值，步长 0.5–1.5×、时长 0.5–2×）

| 参数 | 类别 | 默认 | 第一轮档位(低/中/高) |
|------|------|------|------|
| `release_clearance` | 几何 | 0.0 | 0.0 / 0.02 / 0.04 |
| `descend_max_position_step` | 运动 | 0.006 | 0.003 / 0.006 / 0.009 |
| `move_max_position_step` | 运动 | 0.012 | 0.006 / 0.012 / 0.018 |
| `open_duration` | 时序 | 0.45 | 0.25 / 0.45 / 0.90 |
| `settle_after_release_duration` | 时序 | 0.5 | 0.2 / 0.5 / 1.0 |
| pre_place_height / retreat_height / *_orientation_step / retreat_max_position_step | 几何/运动 | 见 cfg | 本轮不重点扫，可调 |

`release_clearance` 只改变**执行**（物体释放高度 = surface + support_offset + base_clearance(0.002) +
release_clearance），评估目标始终是任务支撑面位姿（不含 release_clearance）。

### 抽屉 θ + 任务目标

| 参数 | 类别 | 默认 | 第一轮档位 |
|------|------|------|------|
| `max_pos_step` | 运动 | 0.020 | 0.010 / 0.020 / 0.030 |
| `pull_lead` | 运动 | 0.08 | 0.04 / 0.08 / 0.12 |
| `grasp_offset_local_xyz` | 几何(link局部) | (0,0,0) | (0,0,0)/(0,0.03,0)/(0,−0.03,0) |
| `close_duration` | 时序 | 1.0 | 0.5 / 1.0 / 1.5 |
| `target_open_position` (任务目标g) | 几何 | 0.20 | 0.10 / 0.20 / 0.30 |

`grasp_offset_local_xyz` 在抽屉 **link 局部系**叠加到把手偏移后再经 link 变换到世界
（`open_drawer_skill._grasp_pose`），用于人为制造精抓/边缘/滑脱/抓空，绝非世界硬偏移。
`target_open_position` 限制在关节合法范围 [0, 0.8]，成功判据随之变（§6）。

### 判定参数（记录+固定，非第一批科研变量）
放置：position_threshold(0.020)、orientation_threshold(10°)、stable_cycles(5)、state_timeout(12)。
抽屉：reach_pos/ori_threshold、reach_stable_cycles、target_tolerance(0.02)。
> position_threshold 由 0.012 调到 0.020：单步 DLS IK 在工作空间边缘稳态残差 ~1.5cm，0.012 会卡死
> MOVE_TO_PRE_PLACE 触发 POSITION_TIMEOUT；2cm 仅为“可推进”阈值，最终精度由 VERIFY 基于物体判定。

## 4. 采样设计

- 第一轮**只做单参数敏感性**（`--sampler one_param`），不做多参数联合网格。
- 每参数 3 档（低/中/高），**每档重复 ≥5 次**。
- **配对设计**：重复 r 在所有档位用相同 `reset_index`（相同初始布局/物体位姿），使档位间比较成对、
  消除环境噪声混淆（`run_stage1_pilot._build_specs`）。
- 环境随机化（`SimpleSceneLayoutManager`，按 reset_index 确定性）：放置随机化 cube 初始 XY+yaw；
  抽屉布局随机但目标抽屉关节按 `--initial_drawer_open` 复位（top→joint_0 / middle→joint_2，
  二者都正确复位，不只复位 joint_0）。本轮不做视觉/质量/摩擦随机化。

## 5. 放置成功判据（基于物体，不是状态机到末态）

`success` = 同时满足（`place_skill.PlaceSkill._verify`）：
```
object_position_error ≤ object_position_tolerance (0.030 m, 依 cube 边长 ~0.04 m 设定)
AND object_orientation_error ≤ object_orientation_tolerance (20°)
AND object_final_linear_speed ≤ 0.02 m/s
AND object_final_angular_speed ≤ 0.5 rad/s
AND 连续 settle_required_cycles(3) 帧速度达标 (settling_time 记录到)
AND not object_dropped (物体 z 低于应停高度 0.05m 以上)
AND not object_out_of_bounds (xy 偏离目标 > 0.15m)
AND not 超时
```
`legacy_reached` 仅为调试标志（旧状态机是否走到开夹爪末态），**禁止**用作 success。
默认参数下阈值经标定使正常放置仍判真实成功（回归验证），未为兼容无限放宽。
物体误差相对**任务目标**计算，TCP 误差仅辅助。

## 6. 结果字段 (outcomes)

放置：elapsed_time, object_position_error, object_orientation_error, settling_time,
maximum/mean_tcp_tracking_error, object_final_linear/angular_speed, object_dropped,
object_out_of_bounds, success, failure_reason, legacy_reached, contact_*(=null)。

抽屉：elapsed_time, initial/target/final_drawer_position, drawer_position_error,
drawer_overshoot(final−target，自由滑动会过冲), maximum/mean_tcp_tracking_error,
maximum/mean_handle_relative_error, handle_detached, target_reached_time, timeout,
success, failure_reason, contact_*(=null)。

## 7. 把手脱离判据（无接触传感器）

`open_drawer_skill._track_pull`：仅当在 **PULL** 状态下，**同时**满足一段时间才判脱离，避免单帧误触：
```
state == PULL
AND TCP 与实时把手世界位距离 > handle_detach_error (默认 0.06 m)
AND 抽屉关节连续 handle_detach_no_progress_window(1.5 s) 无进展(进展阈 0.003 m)
AND 尚未达到目标
```
依据：握持时夹爪包住把手，TCP≈把手（误差几 cm）；滑脱时把手停住而 TCP 继续向 lead 目标外移→距离增大，
配合“抽屉不再前进”双重条件区分“滑脱(HANDLE_DETACHED)”与“拉不动(DRAWER_OPEN_TIMEOUT，握持但到达限位)”。
阈值将由默认成功轨迹的 handle_relative_error 分布校准（见 sensitivity report，用回归数据统计 p95）。

## 8. 统一数据 schema 与目录

```
experiments/stage1/<run_id>/
  episodes.jsonl            # 每行一 episode（四组变量 + controller + success/failure + trajectory_file）
  trajectories/<id>.npz     # 统一 step schema（见下），不适用字段填 NaN，skill_state 为字符串
  metadata.json parameter_ranges.json environment_config.json git_commit.txt run_command.txt
```
Step schema（`trajectory_logger.UNIFIED_SCHEMA`）：sim_time, elapsed_time, skill_state,
joint_position(9), joint_velocity(9), tcp_position(3), tcp_orientation(4), target_tcp_position(3),
target_tcp_orientation(4), tcp_position_error, tcp_orientation_error, measured_tcp_linear_speed,
measured_tcp_angular_speed, gripper_command；放置加 object_position/orientation/lin/ang_velocity、
object_position_error；抽屉加 drawer_joint_position/velocity, handle_position,
target_handle_grasp_position, handle_relative_position_error, drawer_progress；
contact_available=false, contact_force=NaN, collision_available=false, collision_flag=NaN。

大数据加入 .gitignore，仓库只留 schema 示例 + ≤10 条示例 episode + 小轨迹 + 图 + 报告。
