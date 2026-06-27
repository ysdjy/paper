# 第一阶段代码审计 — Franka 技能执行结果数据生成系统

> 范围：放置（place）与打开抽屉（open_drawer）两个核心技能；抓取作为放置前置辅助；门/咖啡机仅保留接口。
> 本文件只做事实审计 + 文件级实施方案，不含任何代码改动。所有结论基于真实代码，标注 `文件:行号`。

---

## 0. 已阅读的文件（含依赖）

主链路（任务第七节指定）：
- `runtime/place_skill.py`
- `skills/place_skill.py`
- `skills/open_drawer_skill.py`
- `skills/close_drawer_skill.py`
- `runtime/skill_request.py`
- `runtime/skill_result.py`
- `runtime/base_skill.py`
- `runtime/scene_state_provider.py`
- `runtime/ik_joint_adapter.py`
- `state_machine/skill_executor.py`
- `entries/skill_sequence_joint.py`
- `entries/skill_test_ui_joint.py`

为确认控制频率/动作契约/资产事实额外阅读的依赖：
- `runtime/drawer_obs_adapter.py`（`SelectedDrawerObsAdapter` 把手读取）
- `runtime/drawer_target_config.py` → `source/.../franka/custom_drawer_config.py`（抽屉中央配置）
- `runtime/simple_scene_layout.py`（场景随机化能力）
- `source/.../franka/stack_joint_policy_env_cfg.py` 与父类 `stack_joint_pos_env_cfg.py` / `stack_env_cfg.py`（环境/动作/dt/接触传感器）

---

## 1. 底层环境与控制契约（已核实）

| 项 | 值 | 来源 |
|----|----|------|
| 部署任务 | `Isaac-Stack-Cube-Franka-JointPolicy-v0` | `franka/__init__.py:27` |
| Env cfg | `FrankaCubeStackJointPolicyEnvCfg`(← `FrankaCubeStackEnvCfg` ← `StackEnvCfg`) | `stack_joint_policy_env_cfg.py:37` |
| 物理步长 `sim.dt` | **0.01 s（100 Hz）** | `stack_env_cfg.py:192` |
| `decimation` | **5** | `stack_env_cfg.py:189` |
| **控制步长 control_dt** | **0.05 s（20 Hz）** = dt×decimation | `skill_sequence_joint.py:268` |
| 手臂动作 | `JointPositionActionCfg(scale=1.0, use_default_offset=True)` | `stack_joint_policy_env_cfg.py:63-67` |
| 动作维度 | 8 = 7 臂关节 + 1 夹爪 | `scene_state_provider.py:186-203` |
| 夹爪约定 | `+1` 张开 / `-1` 闭合（二值） | `scene_state_provider.py:193` |
| IK | DLS DifferentialIKController，单步/帧，`max_joint_step=0.20 rad` 钳制 | `ik_joint_adapter.py:50,156` |
| 多环境 | **不支持**：两个入口都强制 `num_envs==1` | `skill_sequence_joint.py:198`, `skill_test_ui_joint.py:525` |

> 注意：任务示例 JSON 写的是 `control_dt=0.02`，**真实值是 0.05 s（20 Hz）**，必须以真实值入库。
> 注意：技能 `step(state, dt)` 的 `dt` 形参在技能内部基本未用；所有计时用 `state.sim_time` 的差值，`sim_time` 由入口 `provider.set_sim_time()` 每步 +0.05 推进（`skill_sequence_joint.py:319`）。

### 1.1 接触 / 碰撞信息是否可用

- 场景内**所有**资产 `activate_contact_sensors=False`，且 scene 中**没有任何 `ContactSensor`**。
  来源：`stack_joint_pos_env_cfg.py:201,273,326,364,429,...`（多处 `activate_contact_sensors=False`），全文件无 `ContactSensorCfg`。
- 结论：**接触力 / 碰撞计数当前不可获取**。本阶段 `contact_available=false`、`collision_available=false`、相关字段写 `null`（禁止填 0 伪装）。
- 可获取的“物理真值”：刚体物体的 `root_lin_vel_w` / `root_ang_vel_w`（`scene_state_provider.py:348-349`，cube 走 `_rigid_object_state`），抽屉关节 `joint_pos/joint_vel`（articulation），把手世界位姿（link∘offset）。**这些足以支撑放置的稳定/掉落判定与抽屉的开度/跟踪判定**，无需接触力。

---

## 2. 放置技能 — 当前真实流程

文件：`runtime/place_skill.py`（状态机本体）+ `skills/place_skill.py`（IK 包装，把 TCP 目标解成 q_des）。

### 2.1 状态列表与切换（`place_skill.py:116-145`）

```
IDLE
 └─start→ MOVE_TO_PRE_PLACE
            └─[TCP到位: pos≤0.012m 且 ori≤8°, 连续5帧]→ DESCEND_TO_PLACE   (place_skill.py:124,264-279)
DESCEND_TO_PLACE
            └─[同上阈值, 连续5帧]→ OPEN_GRIPPER                            (place_skill.py:133)
OPEN_GRIPPER
            └─[夹爪开 open_duration=0.45s 后]→ SUCCEEDED                   (place_skill.py:138-139)
任意 reach 状态超时(state_timeout=10s)→ FAILED(POSITION_TIMEOUT)            (place_skill.py:273-278)
```

只有 **3 个活动状态**：`MOVE_TO_PRE_PLACE → DESCEND_TO_PLACE → OPEN_GRIPPER`。无 settle / retreat / verify。

### 2.2 每个状态目标 Pose 来源（`place_skill.py:173-237`）

- `target_surface_xyz` ← `request.parameters["target_surface_xyz"]`（env-local，必填，缺失/非 3 维即失败，`place_skill.py:239-252`）。
- 物体目标位 = `env_origin + [sx, sy, sz + support_offset_z + PLACE_CLEARANCE]`；
  `support_offset_z` 来自硬编码 `OBJECT_SUPPORT_OFFSET_Z`（cube=0.0203, knife=0.095，`place_skill.py:21-26`），`PLACE_CLEARANCE=0.002`（`:27`）。
- 物体目标姿态 = **当前被抓物体的姿态**（`held.pose.quat_w`，`:205`，即维持原朝向）。
- 预放置位 = 物体目标位 + `[0,0,PRE_PLACE_HEIGHT]`，`PRE_PLACE_HEIGHT=0.100`（**模块级硬编码**，`:28,206`）。
- TCP 目标 = 用抓取时记录的 `object_to_tcp`（`held_object.object_to_tcp_pos/quat`）把“物体目标位姿”换算成 TCP 位姿（`:207-224`）。
- 每帧命令 = `step_pose(上一命令位姿, 目标, max_pos_step, max_ori_step)`，再被 `_SPEED_SCALE` 全局缩放（`base_skill.py:96-116`）。

### 2.3 当前成功判据（**核心问题**）

`place_skill.py:134-139`：进入 `OPEN_GRIPPER` 后，夹爪给 `+1` 持续 `open_duration=0.45s`，**直接 `_succeed`**。
- **完全不检查物体**：不看物体最终位置/姿态误差、不看线/角速度、不看是否掉落、不看是否越界、不看是否稳定。
- 即“成功 = 状态机走到末状态”，正是任务第十七节 No-Go 中明令禁止的标签来源。
- `position_error`/`orientation_error` 记录的是 **TCP vs 期望 TCP** 的误差（`:141-143`），不是物体误差。

### 2.4 放置可调参数现状

`PlaceSkillConfig`（dataclass，`place_skill.py:43-53`）含：`position_threshold=0.012`、`orientation_threshold=8°`、`stable_cycles=5`、`max_position_step=0.012`、`max_orientation_step=5°`、`descend_position_step=0.006`、`descend_orientation_step=4°`、`open_duration=0.45`、`state_timeout=10.0`。

**但是**：
- `SkillExecutor` 构造 `PlaceSkill`/`PlaceJointSkill` 时**不传 config**（`skill_executor.py:142-144`）→ 永远用默认 `PlaceSkillConfig()`。
- `request.parameters` 当前**只读 `target_surface_xyz`**（和被忽略的 `target_frame`）。
- ⇒ **放置实际可经 `request.parameters` 覆盖的执行参数 = 仅 `target_surface_xyz`（几何目标点）**。其余全部为默认常量，无法实验。

---

## 3. 打开抽屉技能 — 当前真实流程

文件：`skills/open_drawer_skill.py`（`OpenDrawerIKSkill`，backend=`ik_pull`，真实物理拉动）。

### 3.1 状态列表与切换（`open_drawer_skill.py:216-302`）

```
IDLE
 └─start→ TURN_TO_FACE        (默认 use_turn_to_face=True; 否则 MOVE_TO_PRE_GRASP 或 MOVE_TO_HOME)
TURN_TO_FACE   [joint1 转到面向把手, |Δq1|≤0.12 连续3帧, 或 face_timeout=5s] → MOVE_TO_PRE_GRASP  (:228-243)
MOVE_TO_PRE_GRASP  目标=handle + pre_grasp_clearance(0.12)*open_dir
               [pos≤0.03m 且 ori≤18°, 连续6帧; 或 reach_timeout=16s 软进入] → APPROACH        (:244-248,380-400)
APPROACH       冻结直线(standoff→handle)+朝向, carrot 直线滑入到 handle
               [到 handle 端点, 同 reach 阈值] → CLOSE_GRIPPER                                  (:249-263)
CLOSE_GRIPPER  夹爪 -1 持续 close_duration=1.0s → PULL                                          (:264-268)
PULL           目标=handle + pull_lead(0.08)*open_dir, 夹爪 -1
               [抽屉关节位 ≥ success_threshold(0.20)] → SETTLE                                  (:269-286)
               [pull_timeout=16s 未达] → FAILED(DRAWER_OPEN_TIMEOUT)
SETTLE         保持夹紧 settle_duration=0.5s（让自由滑动惯性衰减）→ RELEASE                      (:287-293)
RELEASE        夹爪 +1 持续 release_duration=0.4s → SUCCEEDED                                   (:294-299)
```

### 3.2 每个状态目标 Pose 来源（`open_drawer_skill.py:127-180`）

- 把手世界位：**每帧实时**读 `SelectedDrawerObsAdapter.selected_handle_pos_w()` = 抽屉 link 体位姿 ∘ `handle_offset`（link 局部）。`handle_offset` 来源优先级（`drawer_obs_adapter.py:136-254`）：①场景编辑保存的 `grasp_poses.json` 的 `handle_<drawer>`；②HandleProxy prim；③mesh 自动定位；④中央配置 `custom_drawer_config.py:HANDLE_LOCAL_OFFSET`。
- 开门方向 `open_dir`：由柜体 root 四元数推出（`drawer_ik_common.open_direction_world`），世界 −X。
- 抓取朝向：`grasp_quat_from_open_dir(open_dir)`；或 `request.parameters["override_grasp_local"]`（link 局部位姿，`:166-177`）。
- 抓取/拉动目标 = `handle + lead*open_dir`，`lead` 在不同状态分别为 `pre_grasp_clearance / 0 / pull_lead`。
- 每帧命令 = `step_pose(当前TCP, 目标, max_pos_step=0.02, max_ori_step=6°)` → DLS IK。

### 3.3 当前成功判据

`open_drawer_skill.py:278`：`current_joint_pos ≥ success_threshold`，其中 `success_threshold` 取自 `DRAWER_TARGETS[target]["success_threshold"]=0.20`（`custom_drawer_config.py:42`），**不从 request 取**，固定 0.20。
- **优点**：基于真实机构状态（抽屉关节位），不是“状态机走完”。符合科研要求。
- **缺点**：①目标开度写死 0.20，无法每次请求指定；②无“把手脱离/滑脱”判据；③拉动过程的跟踪误差、把手相对误差不进入结果。

### 3.4 抽屉可调参数现状

`OpenDrawerIKConfig`（dataclass，`open_drawer_skill.py:32-69`）含大量参数：`pre_grasp_clearance=0.12`、`pull_lead=0.08`、`max_pos_step=0.02`、`max_ori_step=6°`、`reach_pos_threshold=0.03`、`reach_ori_threshold=18°`、`reach_stable_cycles=6`、`approach_line_lead=0.03`、`close_duration=1.0`、`reach_timeout=16`、`pull_timeout=16`、`settle_duration=0.5`、`release_duration=0.4`、`use_turn_to_face`、`face_*`、`home_*`、`soft_reach_*` 等。

**但是**：
- `SkillExecutor._make_drawer_skill` 用 `config=self.backend.drawer_open_ik_config`（`skill_executor.py:206-207`），而 `JointBackendConfig.drawer_open_ik_config` **默认 None**（`skill_executor.py:53`）→ 永远用默认 `OpenDrawerIKConfig()`。两个入口也都不设置它。
- `request.parameters` 当前**只读 `override_grasp_local`**（抓取位姿覆盖）。
- ⇒ **抽屉实际可经 `request.parameters` 覆盖的执行参数 = 仅 `override_grasp_local`（抓取位姿）**；目标开度、拉动量、步长、夹取时长全为默认常量。

### 3.5 抽屉资产事实（`custom_drawer_config.py`，debug 脚本已确认）

| target | joint | link | 可动 | 开向 |
|--------|-------|------|------|------|
| top_drawer | joint_0 | link_0 | ✅ | 世界 −X |
| middle_drawer | joint_2 | link_2 | ✅ | 世界 −X |
| bottom_drawer | joint_1 | link_1 | ❌ LOCKED（闭合处穿模） | — |

- prismatic，局部轴 Z，**limits [0, 0.8]**，closed=0，open_direction=+1。
- 抽屉作动器：`ik_pull` 后端把 `cabinet.actuators["drawers"]` 设为 `stiffness=0, damping=2`（自由滑动，不与夹爪对抗），其它后端 `stiffness=10, damping=1`（`skill_sequence_joint.py:213-218`）。
- ⚠️ headless 入口 `_settle` 只 `reset_cabinet_joint("joint_0", 0.0)`（`skill_sequence_joint.py:190`）——**只复位 top 抽屉关节**；中抽屉(joint_2)初始开度需另行设置。

### 3.6 关闭抽屉（次要，仅记录）

`skills/close_drawer_skill.py` 是开抽屉镜像：`...→PUSH`，目标 `handle - push_lead*open_dir`，成功判据 `joint ≤ close_success_threshold=0.01`（`close_drawer_skill.py:255`）。参数同样只有 `override_grasp_local` 可经 request 覆盖。本阶段不做大规模采集。

---

## 4. 参数现状汇总表

| 技能 | 现可经 `request.parameters` 覆盖 | 在 Config dataclass（但当前未接通 request） | 代码硬编码常量 |
|------|------|------|------|
| 放置 | `target_surface_xyz` | `PlaceSkillConfig` 全部 9 项 | `OBJECT_SUPPORT_OFFSET_Z`、`PLACE_CLEARANCE`、`PRE_PLACE_HEIGHT` |
| 开抽屉 | `override_grasp_local` | `OpenDrawerIKConfig` 全部 ~20 项 | `success_threshold`(取自中央配置)、`HOME_Q_VERTICAL_RAISED` |

> 核心结论：**两个技能的“执行参数”几乎都没有从 `request.parameters` 接通到执行计算**。Config 里虽有字段，但构造时不传 config、request 里不读 → 改 request 不改变任何运动。这正是本阶段第一项必须建设的能力（统一参数解析 + 优先级 request > config > 常量）。

---

## 5. 结果日志现状与缺失

### 5.1 当前写了什么
- `SkillResult`（`skill_result.py:20-34`）字段：`success, final_status, failure_reason, elapsed_time, final_tcp_pose, final_object_pose, position_error, orientation_error, gripper_width, state_history`。
- `SkillExecutor` 每个技能结束把 `SkillResult` 追加到 `logs/skill_tests/grasp_results.jsonl`（`skill_executor.py:290-293`）。
- `skill_sequence_joint.py` 另写**序列级**记录到 `logs/skill_tests/joint_sequence_results.jsonl`（`:332-353`：含 elapsed、final_status、final_robot_joint_pos、final_tcp_pose、drawer_joint_pos）。
- `state_history`：仅**状态切换记录**（transition），不是逐帧轨迹（`place_skill.py:293-315`、`open_drawer_skill.py:414-429`）。

### 5.2 相对任务第十二节 schema 的缺失
- **放置缺**：object_position_error、object_orientation_error、object_lin/ang_speed、object_dropped、object_out_of_bounds、settling_time、max/mean_tcp_tracking_error、contact_* 全缺。
- **抽屉缺**：initial/target/final_drawer_position、drawer_position_error、max/mean_tcp_tracking_error、max/mean_handle_relative_error、handle_detached、target_reached_time、逐帧 drawer_joint_pos/vel & handle_pos & tracking_error 全缺。
- **通用缺**：Episode 级 JSONL（带 controller / initial_state / target / requested_parameters / effective_parameters / outcomes / trajectory_file）、Step 级 NPZ/Parquet 轨迹、`metadata.json / parameter_ranges.json / environment_config.json / git_commit.txt / run_command.txt`。
- 无 `effective_parameters`（生效值/裁剪/拒绝）追踪。

---

## 6. 运行入口与命令（已核实）

- 交互 UI：`entries/skill_test_ui_joint.py`（num_envs=1，点按钮跑技能、可编辑放置点/抓取位姿）。
- headless 序列：`entries/skill_sequence_joint.py`（num_envs=1，`--sequence grasp:cube_1,place:point_a,...`）。
  ```bash
  conda activate env_isaaclab
  ./isaaclab.sh -p projects/franka_skill_state_machine/entries/skill_sequence_joint.py \
    --num_envs 1 --sequence grasp:cube_1,place:point_a \
    --grasp_backend joint_ik --place_backend joint_ik --drawer_backend ik_pull \
    --seed 1 --max_steps 3000 --headless
  ```
- **无批量 pilot 入口**；UI 不适合批量。两个入口都**仅单环境**。
- 放置依赖前置抓取：`PlaceSkill` 需要 `held_object` 上下文（抓取成功时由 `SkillExecutor._save_held_object_context` 记录 `object_to_tcp`，`skill_executor.py:376-389`）。⇒ **放置 pilot 每个 episode 必须先 grasp 再 place**。

---

## 7. 文件级实施方案（待你确认后再写代码）

> 严格按任务第六节顺序：审计→参数接口→放置→抽屉→日志→pilot→分析。下面是“准备改哪些文件、各改什么”。

### 7.1 新增：统一参数解析（不破坏默认）
- **新增** `runtime/experiment_params.py`：`get_float_parameter / get_int_parameter / get_vec3_parameter`，做类型/有限数/范围检查+默认值+裁剪记录；返回 `(value, ParamTrace)`，`ParamTrace` 记 requested/effective/default/clamped/rejected/作用位置。汇总成 `effective_parameters`。

### 7.2 放置改造
- **改** `runtime/place_skill.py`：
  - 扩 `PlaceSkillConfig`，并在 `start()` 用参数解析把第九节参数从 `request.parameters` 接通（request>config>常量）；把 `PRE_PLACE_HEIGHT`、`target_offset_xyz` 等从常量挪进可调。
  - 扩状态：`...OPEN_GRIPPER → WAIT_FOR_SETTLE → RETREAT → VERIFY_PLACE → SUCCEEDED/FAILED`。
  - 重写成功判据为**基于物体**（位置/姿态误差 + 线/角速度 + 稳定周期 + 未掉落 + 未越界 + 未超时），阈值按场景尺寸（cube ~4cm）推导，不照抄提示数字。
  - 采集逐帧轨迹 + 终态多维 outcomes（见第十二节），接触相关写 `null`。
- **改** `skills/place_skill.py`：把 config/参数透传给内层 `PlaceSkill`（构造处接 request.parameters）。
- **改** `state_machine/skill_executor.py`：构造 place 时允许传 config / 让技能自行从 request 解析（保持默认行为）。

### 7.3 抽屉改造
- **改** `skills/open_drawer_skill.py`：
  - 在 `start()` 接通第十节参数（`pull_lead/max_pos_step/pre_grasp_clearance/close_duration/...`）。
  - 新增 `grasp_offset_local_xyz`（**在把手/link 局部系**），真正进入 `_grasp_pose()`（叠加到 link-local offset，再 ∘ link 位姿）。
  - 新增 `target_open_position`（限制在 [0,0.8] 合法范围）；成功判据改为 `actual ≥ requested_target - tol`（默认 0.20 保持现状）。`target_open_position` 与 `target_open_delta` 二选一（明确优先级：position 优先）。
  - 新增把手脱离判据（候选：TCP–目标把手距离超阈 / 抽屉持续不动 / 夹爪闭合但把手相对误差突增；接触不可用故不用接触判据）——判据与依据写入报告。
  - 采集逐帧（drawer pos/vel、handle pos、handle 相对误差、tcp tracking error、state）+ 终态多维 outcomes。
- **改** `state_machine/skill_executor.py`：把 `drawer_open_ik_config` / request 参数接通（默认 None → 默认 config，保持现状）。
- **改** `runtime/scene_state_provider.py`（小改）：放置/抽屉判定需要的物体速度、抽屉 joint_vel 已可读；如需统一暴露则补 helper（不改控制路径）。

### 7.4 结果与日志
- **改** `runtime/skill_result.py`：扩 `SkillResult`，新增第十二节 outcomes 字段（向后兼容，新增字段默认 None）。
- **新增** `runtime/episode_logger.py`：写 Episode 级 JSONL（controller/initial_state/target/requested/effective/outcomes/trajectory_file）。
- **新增** `runtime/trajectory_logger.py`：写 Step 级 NPZ（统一 schema，不适用字段留空/NaN）。
- **新增** 数据集目录元数据写出：`metadata.json / parameter_ranges.json / environment_config.json / git_commit.txt / run_command.txt`。

### 7.5 批量 pilot 入口
- **新增** `entries/run_stage1_pilot.py`：`--skill {place,open_drawer} --episodes --seed --sampler {sobol,lhs,stratified} --output_dir --headless`；固定种子、参数采样、每 episode 重置布局/抽屉初态、超时保护、异常不中断、自动保存、可恢复（已完成跳过）。**不混入 UI**。

### 7.6 分析
- **新增** `analysis/analyze_stage1_pilot.py`：产出第十六节图表与统计（参数↔时间/精度/稳定/掉落/滑脱、时间–精度散点、相关性、有效范围、无效参数清单、trade-off）。

### 7.7 文档
- 本文件 + 后续 `docs/experiment_stage1_design.md`、`experiment_stage1_implementation.md`、`experiment_stage1_pilot_report.md`。

---

## 8. Pilot 参数建议范围（第一轮，围绕真实默认值，0.5–1.5× 步长 / 0.5–2× 时长）

> 放置控制步长真实默认：move=0.012、descend=0.006（m/帧 @20Hz）；抽屉：max_pos_step=0.020、pull_lead=0.08。按第十一节命名为 **command step size / nominal Cartesian progression**，非真实末端速度。

### 放置
| 参数 | 类别 | 默认 | 建议范围 | 备注 |
|------|------|------|---------|------|
| `pre_place_height` | 几何 | 0.100 | 0.05–0.15 | 释放高度 |
| `descend_max_position_step` | 运动 | 0.006 | 0.003–0.009 | 下降命令步长 |
| `move_max_position_step` | 运动 | 0.012 | 0.006–0.018 | 接近命令步长 |
| `open_duration` | 时序 | 0.45 | 0.25–0.90 | 开夹时长 |
| `settle_after_release_duration` | 时序 | (新) | 0.2–1.0 | 释放后等待 |
| `target_offset_xyz` | 几何 | (0,0,0) | 各轴 ±0.00–0.02 | 目标点扰动（cube ~4cm） |
| `position_threshold` | 判定 | 0.012 | 0.008–0.02 | 不作为能力参数主来源 |
| `stable_cycles` | 判定 | 5 | 3–8 | 同上 |

### 抽屉（top + middle）
| 参数 | 类别 | 默认 | 建议范围 | 备注 |
|------|------|------|---------|------|
| `pull_lead` | 运动 | 0.08 | 0.04–0.12 | 拉动提前量 |
| `max_pos_step` | 运动 | 0.020 | 0.010–0.030 | 命令步长 |
| `grasp_offset_local_xyz` | 几何 | (0,0,0) | 各轴 ±0.00–0.03 | 局部系；制造精抓/边缘/滑脱/抓空 |
| `target_open_position` | 几何 | 0.20 | 0.10–0.35（分层，∈[0,0.8]） | 成功判据随之变 |
| `pre_grasp_clearance` | 几何 | 0.12 | 0.06–0.18 | 预抓退让 |
| `close_duration`(夹取) | 时序 | 1.0 | 0.5–1.5 | 夹紧时长 |

环境随机化（`SimpleSceneLayoutManager` 已支持 cube/knife 区域内随机 XY+yaw，按 reset_index 确定性）：
- 放置：物体初始位（已支持）、目标放置点、机械臂初态、物体类型（cube_1/2/3）；质量/摩擦随机化**当前未接线**（需确认是否加）。
- 抽屉：抽屉初始开度（需新增写入，注意中抽屉是 joint_2）、目标开度、抓取偏移、拉动参数、top/middle。
- 第一轮**不**引入视觉随机化。采样优先 Sobol/LHS/分层，**不**用全组合网格。

---

## 9. 风险点

1. **参数未接线是根因**：当前改 request 不改变任何运动（第 4 节）。若不先建参数解析+接通，整个实验无基础。
2. **放置成功判据需重定义**：现为“夹爪开 0.45s 即成功”（第 2.3 节），与第六节硬约束冲突。改为物体级判据后，**部分当前“成功”可能被重判为失败**——与第 6.5 节“保留默认可运行路径”存在张力。建议：默认阈值按场景标定使默认行为仍判成功，并保留 `legacy_reached` 标志区分“状态机到末态”与“真实成功”。**需你确认**。
3. **放置依赖前置抓取**：grasp 的抓取位姿差异会污染 place 指标。对策：每 episode 在 place 起点快照 `initial_state`（含 TCP/物体/object_to_tcp），把抓取结果作为初始状态的一部分，而非混入 place 参数效应。
4. **接触/碰撞不可用**（第 1.1 节）：`contact_available=false`。若论文需要接触/碰撞维度，需新增 `ContactSensor`（手指 + 抽屉/把手），这是 **cfg 增量**（非“重写 Isaac Lab 环境”），但仍属环境改动——**需你确认是否纳入本阶段**。
5. **单环境吞吐**：两个入口仅 num_envs=1，pilot 为顺序单环境；50–100 episode×（grasp+place 各数十秒）可接受但偏慢。是否要我评估多环境（较大改动）**需确认**。
6. **抽屉中抽屉初态复位缺口**：`_settle` 只复位 joint_0；变中抽屉初始开度需 runner 显式写 joint_2。
7. **自由滑动 + 目标开度**：`ik_pull` 把抽屉设为自由滑动，物理拉到指定 `target_open_position` 后可能因惯性滑过；SETTLE 已部分处理，需验证可停在目标附近（否则 final≠target 会成系统偏差，可作为一个真实“执行结果”维度，但要如实标注）。
8. **全局速度缩放**：`_SPEED_SCALE` 会乘到每步增量；pilot 必须固定 `--speed 1.0` 保证可复现。
9. **IK 单步 + max_joint_step=0.20 钳制**：命令步长是真实运动参数，但 IK 自身钳制是上限；需在报告中区分“命令步长”与“实测末端速度”（实测可由轨迹算）。

---

## 10. 等待确认的问题（实施前）

1. **接触维度**：本阶段接受 `contact_available=false`（字段写 null），还是授权我新增手指/把手 `ContactSensor`（cfg 增量）？
2. **放置成功重定义**：是否同意默认改为**物体级**成功判据（按场景尺寸标定容差，cube ~4cm），同时保留 `legacy_reached` 兼容旧行为？
3. **抽屉目标开度**：确认成功判据改为“达到 `target_open_position`（默认 0.20 保持现状）”，第一轮范围 0.10–0.35（top/middle 可达性内）？
4. **吞吐**：单环境顺序跑 50–100 episode 可接受，还是要我先评估多环境改造（更大改动、可能触及 env/adapter）？
5. **物体集与质量/摩擦随机化**：放置物体用 cube_1/2/3（是否含 knife）？是否纳入小范围质量/摩擦随机化（需确认安全且真实生效）？
6. **数据落盘路径**：建议 `docs/`（报告）+ `projects/franka_skill_state_machine/experiments/stage1/<run>/`（episode JSONL + trajectories/ NPZ + 元数据），数据目录加入 .gitignore（只提交 10 条示例）。确认路径？

---

（本审计为第一次提交，仅审计 + 文件级方案。待上述问题确认后再进入实施。）
</content>
</invoke>
