# 正式实验协议 v1 (paper_experiment_contract_v1)  — FROZEN

> 技能：`open_drawer`（top + middle）。HEAD `f2e77e7e` 起的新数据契约。协议冻结后**不得在看到结果后
> 修改主要指标**。所有正式数据须满足 R1–R4、R8、R10 的修复/标注。

## 1. 一次候选执行 (x, g, θ, H) → y

### x — 执行前状态（episode 起点，机器人已完整 reset）
deploy-可得（输入合法）：
- `initial_robot_joint_position`(9)、`initial_tcp_pose`(7)、`gripper_width`
- `drawer_name`(top/middle)、`workspace_region`（柜/把手所在象限，离散）
- `initial_drawer_position`（真机由关节编码器可得 → 合法输入）
- `handle_pose_estimate`（**sim GT；真机需 FoundationPose/标定** → 标记 `privileged_in_sim=true`，
  round-1 可用作输入但单列，便于做“去特权”消融）

### g — 任务目标
- `target_open_position`（∈[0.05,0.8]，第一轮分层 0.10/0.18/0.26/0.34）
- `drawer_name`
- 成功容差 `target_tolerance=0.02`（冻结）

### θ — 候选执行参数（真入执行，已实证有效；剔除无效）
- `max_pos_step` ∈[0.008,0.035]
- `pull_lead` ∈[0.03,0.14]
- `grasp_offset_local_xyz`（link 局部）各轴 ∈[−0.06,0.06]
- `pre_grasp_clearance` ∈[0.06,0.18]
- `approach_line_lead` ∈[0.01,0.06]
- **剔除**：`close_duration`（无效，bias R 已证）；判定/超时参数固定记录，不作 θ。

### H — 近期动作—后果历史
- `H_t = {(x_i, g_i, θ_i, y_i)}_{i=t-K..t-1}`，**同技能、同 deployment session**，K∈{1,4,8}（消融）。
- 历史按时间顺序，episode 间机器人已 reset（H 表达的是“系统/部署状态”，非物理遗留）。

### y — 真实执行结果（标签）
- `success`（bool，机构级：final_drawer_position ≥ target − tol 且无脱离）
- `failure_reason` ∈ {NONE, REACH_TIMEOUT, PULL_TIMEOUT, HANDLE_DETACHED, INVALID_PARAM}
- `final_drawer_position`、`task_outcome_error = |final − target|`、`drawer_overshoot = final − target`
- `skill_elapsed_time`（sim）、`phase_durations`{turn,reach,approach,grip,pull,settle}、`wall_clock_time`
- `command_tracking_error`{max,mean}（TCP vs 当帧命令）、`phase_goal_error`{max,mean}（TCP vs 阶段目标）、
  `handle_relative_error`{max,mean}
- 接触维度 round-1 = null（Stage 1.5）

> 误差三分（R3 修复）：phase_goal_error / command_tracking_error / task_outcome_error 必须分开存储与命名。
> 时间分层（R4）：simulation_time / skill_elapsed_time / phase_durations / wall_clock_time。

## 2. 信息公平性（baseline 可见集合）
| baseline | 可见输入 |
|----------|----------|
| B0 Rule | 仅 g（用默认/规则 θ） |
| B1 State-only | x(deploy-可得) + g + θ |
| B2 State+History | x + g + θ + H |
| B3 Generic Vision | + 通用视觉特征（round-1 暂缺相机 → 推后） |
| B4 Frozen VLM feat | + 冻结 VLM 特征（推后） |
| B5 VLM explicit | + VLM 难度/可行性判断（推后） |
| B6 Full | x+g+θ+H+VLM（推后） |
| Oracle | 用真实 y 选最优候选（仅算 regret 上界，不作输入） |

- **sim GT**：精确 `final_drawer_position`/关节真值仅作 **label**；`handle_pose` 作输入时标 privileged，
  并提供“去 handle 特权”消融（用粗略象限替代），以评估 sim2real 乐观度。
- 真机映射：initial_drawer_position←编码器；handle_pose←FoundationPose；其余同。

## 3. 数据划分（防泄漏 R10）
- 按 **deployment session / reset_index 分组** 划分 train/val/test（同初态不得跨集）。
- 候选选择评估：每个 (x,g) 下生成 N 个 θ 候选，模型排序 vs Oracle 排序 → regret / top-1 / pairwise acc。

## 4. 冻结的主要指标（看到结果前固定）
- 预测：success AUROC、Brier、failure-type F1、task_outcome_error MAE、skill_elapsed_time MAE。
- **决策（最重要）**：candidate selection regret、top-1 candidate accuracy、选中候选实际成功率、
  选中候选实际 task_outcome_error、与 Oracle 差距。
- H1 判据：θ/x 对 y 产生稳定（跨重复可复现）差异。
- H2 判据：B2(+H) 在预测与 regret 上**显著优于** B1。
