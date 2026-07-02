# 第一阶段实施说明 (experiment_stage1_implementation)

记录第一阶段代码实现：文件清单、参数接通方式、状态机改造、日志、运行命令、最小测试结果，以及
后续 Stage 1.5 接触传感计划。设计依据见 `experiment_stage1_design.md`。

## 1. 新增 / 修改文件

| 文件 | 类型 | 内容 |
|------|------|------|
| `runtime/experiment_params.py` | 新增 | `ParamResolver` + `get_{float,int,bool,vec3}_parameter`，provenance(`ParamTrace`)，优先级 request>config>const，越界裁剪/危险带拒绝(`InvalidExperimentParameter`) |
| `runtime/skill_result.py` | 改 | 向后兼容扩展：outcomes / legacy_reached / task_target / requested_parameters / effective_parameters |
| `runtime/skill_types.py` | 改 | 新增 FailureReason：INVALID_EXPERIMENT_PARAMETER / OBJECT_OUT_OF_BOUNDS / PLACE_NOT_SETTLED / PLACE_VERIFICATION_FAILED / HANDLE_DETACHED / DRAWER_TARGET_NOT_REACHED / SETUP_GRASP_FAILED |
| `runtime/place_skill.py` | 改 | 新状态机 + 参数化 + release_clearance + 物体级成功 + 遥测/统计/outcomes |
| `skills/place_skill.py` | 改 | joint 包装直通 resolver / last_telemetry / outcomes |
| `skills/open_drawer_skill.py` | 改 | 参数化 + target_open_position + grasp_offset_local_xyz + 脱离判定 + 遥测/统计/outcomes |
| `runtime/trajectory_logger.py` | 新增 | Step NPZ 统一 schema + measured speed 工具 |
| `runtime/episode_logger.py` | 新增 | Episode JSONL + 运行元数据(metadata/parameter_ranges/environment_config/git_commit/run_command) |
| `entries/run_stage1_pilot.py` | 新增 | 批量 pilot 入口（num_envs=1） |
| `analysis/analyze_stage1_pilot.py` | 新增 | 敏感性分析 + 图表 + 报告骨架 |

未改动原入口 `skill_test_ui_joint.py` / `skill_sequence_joint.py`（默认行为不变；新参数都有默认值）。

## 2. 参数接通方式（证明参数真实进入执行）

- 每个技能 `start()` 用 `ParamResolver(request.parameters)` 解析（见 design §3 范围），把生效值写进
  `self._eff`，执行中**直接使用 `self._eff[...]`**（如 `step_pose(..., self._eff["max_pos_step"], ...)`、
  下降步长、各时长）。`grasp_offset_local_xyz` 进入 `_grasp_pose()` 的 link 局部组合；
  `target_open_position` 进入 PULL 的成功判据。
- 每条 episode 落 `execution_parameters`（ParamTrace 列表）：可核对 effective_value、source、clamped、
  applied_at，从数据层面证明“参数→执行”链路。非法参数 → 该 episode 被拒，`failure_reason=
  INVALID_EXPERIMENT_PARAMETER`，不静默忽略。

## 3. 放置状态机改造

`MOVE_TO_PRE_PLACE → DESCEND_TO_RELEASE → OPEN_GRIPPER → WAIT_FOR_SETTLE → RETREAT → VERIFY_PLACE`。
- 保留：抓取时 `object_to_tcp`、由物体目标位姿反算 TCP 目标、DLS IK 包装、默认接近/下降行为。
- `release_clearance` 只改释放高度；评估目标 = 任务支撑面位姿（不含 release_clearance）。
- `WAIT_FOR_SETTLE` 实时读物体线/角速度，记 `settling_time`（首次连续达标的时刻），固定 dwell =
  `settle_after_release_duration`（真实时序参数，影响 elapsed_time）。
- `RETREAT` 开夹后沿 +Z 抬 `retreat_height`，避免带动物体。
- `VERIFY_PLACE` 基于物体真值判成功（design §5）。`legacy_reached` 仅调试。

## 4. 抽屉状态机改造

保留 `TURN_TO_FACE→MOVE_TO_PRE_GRASP→APPROACH→CLOSE_GRIPPER→PULL→SETTLE→RELEASE`、实时把手读取、
开门方向、物理拉动、不写关节目标、top/middle 支持。新增：参数化、`target_open_position`（成功
= actual ≥ target − tol，记 `drawer_overshoot`）、`grasp_offset_local_xyz`（link 局部）、把手脱离判定
（design §7）、遥测/统计/outcomes。

## 5. 运行命令

```bash
conda activate env_isaaclab
# 默认回归
./isaaclab.sh -p projects/franka_skill_state_machine/entries/run_stage1_pilot.py \
  --skill place --episodes 10 --seed 42 --sampler fixed --run_id regression_place --headless
./isaaclab.sh -p .../run_stage1_pilot.py --skill open_drawer --drawer top_drawer \
  --episodes 10 --seed 42 --sampler fixed --run_id regression_drawer_top --headless
./isaaclab.sh -p .../run_stage1_pilot.py --skill open_drawer --drawer middle_drawer \
  --episodes 10 --seed 42 --sampler fixed --run_id regression_drawer_middle --headless
# 单参数敏感性（示例）
./isaaclab.sh -p .../run_stage1_pilot.py --skill place --sampler one_param \
  --param descend_max_position_step --values 0.003,0.006,0.009 --repeats 5 --seed 42 --headless
./isaaclab.sh -p .../run_stage1_pilot.py --skill open_drawer --drawer top_drawer --sampler one_param \
  --param grasp_offset_local_xyz --values "0,0,0;0,0.03,0;0,-0.03,0" --repeats 5 --seed 42 --headless
# 分析
python projects/franka_skill_state_machine/analysis/analyze_stage1_pilot.py \
  --root projects/franka_skill_state_machine/experiments/stage1 \
  --out  projects/franka_skill_state_machine/experiments/stage1/_analysis
```

## 6. 最小测试结果（提交时）

- `experiment_params` 自测 12 例全过（默认/请求/裁剪/拒绝/类型/NaN/vec3/未知键）。
- logger 自测：统一 schema、NaN→null、四组分离、可恢复、git_commit 记录。
- 仿真冒烟：grasp+place（seed42）走全 6 状态成功，cube 落点误差 ~0.9cm；open_drawer:top 拉到 0.182 成功。
- 默认回归 place 10/10 成功（object_position_error 0.002–0.007 m）。抽屉见 sensitivity_report。

## 7. Stage 1.5 Contact Sensing Plan（本轮不执行）

本轮 `contact_available=false`、接触/碰撞字段全 `null`/`NaN`，先验证“无接触数据时
时间—精度—轨迹—失败链路是否成立”，并避免同时改控制环境与技能逻辑。后续最小化加接触感知的改动：

1. **环境配置**（`source/.../franka/stack_joint_policy_env_cfg.py` 或其父 `stack_joint_pos_env_cfg.py`）：
   - 给 Franka 双指与目标体打开 `activate_contact_sensors=True`（当前多处为 False）。
   - 在 `SceneCfg` 增 `ContactSensorCfg`：
     - 手指：`prim_path=".../panda_leftfinger"`、`".../panda_rightfinger"`；
     - 放置物体：`prim_path=".../cube_.*"`；抽屉把手 link：`".../Cabinet/link_0"`（top）/`link_2`（middle）。
     - `history_length`>0 以取峰值力；`track_air_time=False`。
2. **状态读取**（`runtime/scene_state_provider.py`）：新增 `read_contact_forces()` 从
   `scene["contact_*"].data.net_forces_w`（或 `force_matrix_w`）取每体接触力范数与最大值。
3. **技能 outcomes / 轨迹**：把 `contact_available=true`、`maximum_contact_force`、逐帧 `contact_force`、
   `collision_flag` 填入（schema 字段已预留，无需改 schema）。
4. **影响评估**：接触传感器会增加每步开销且可能需调 `contact_offset`；先在 1 env 验证，再决定是否纳入
   pilot。**本轮不动这些**。

## 8. 已知问题 / 限制

- 单环境顺序执行（两入口都 num_envs=1）。
- 中抽屉(middle_drawer)可达性弱于 top（把手更靠里，接近 IK 边缘），默认成功率低于 top（见报告）；
  bottom 锁死不纳入。
- `position_threshold` 由 0.012 提到 0.020（design §3 注）。
- 自由滑动抽屉可能过冲，已用 `drawer_overshoot` 如实记录。
