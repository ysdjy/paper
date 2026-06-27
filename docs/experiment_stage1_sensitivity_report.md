# 第一阶段敏感性报告 (experiment_stage1_sensitivity_report)

全部为 **Isaac Lab 真实仿真结果**（`env_isaaclab`，Quadro RTX 8000，control 20 Hz）。数据/图表在
`projects/franka_skill_state_machine/experiments/stage1/`（`_analysis/summary.md`、`*.png`）。
分析脚本：`analysis/analyze_stage1_pilot.py`。固定 seed=42、speed_scale=1.0、配对 reset_index。

> 有效性判据：某指标在档位间的“均值跨度 / 档内噪声(std)”≥1.5 视为 effective；同时报告档位值与指标的
> Pearson 相关。成功率均基于**真实物体/机构状态**（非状态机到末态）。

## 0. 默认回归（行为未被破坏）

| run | n | success | 说明 |
|-----|---|---------|------|
| place (默认参数) | 10 | **10/10** | object_position_error 0.002–0.007 m，settle ~0.15 s，elapsed 3–5 s |
| open_drawer top (默认) | 10 | **10/10** | final ≈0.183 m（稳定），err ~0.017 |
| open_drawer middle (默认) | 10 | **0/10** | `MOVE_TO_PRE_GRASP` 够不到（pos_err 0.13–0.32 m）→ 见 §4 限制 |

## 1. 单参数敏感性结果

### 放置（每档 n=3 有效；5 重复中约 40% 因 grasp setup_failure 被正确剔除，见 §5）

| 参数 | 类别 | 档位→elapsed(s) | 档位→object_err(m) | 相关 | 判定 |
|------|------|----------------|--------------------|------|------|
| `move_max_position_step` | 运动 | 5.47→4.28→4.00 | 0.0026→0.0023→0.0024 | time r=−0.94, err r=−0.75 | **有效** |
| `release_clearance` | 几何 | 4.48→4.12→4.12 | 0.0026→0.0030→**0.0037** | err r=**+0.98**, time r=−0.87 | **有效** |
| `open_duration` | 时序 | 4.22→7.30→7.60 | err r=+0.73 | err 有效 | **有效** |
| `settle_after_release_duration` | 时序 | 4.12→7.30→7.63 | err r=+0.77 | err 有效 | **有效** |
| `descend_max_position_step` | 运动 | 5.15→7.22→7.17(噪声) | 0.0027→0.0024→0.0026 | 无单调 | **无可测效应** |

### 抽屉 top（每档 n=5，全 100% 成功）

| 参数 | 类别 | 档位→elapsed(s) | 档位→drawer_err(m) | 相关 | 判定 |
|------|------|----------------|--------------------|------|------|
| `max_pos_step` | 运动 | **16.60→9.69→8.68** | **0.0219→0.0172→0.0089** | time r=−0.92, err r=**−0.99** | **有效** |
| `pull_lead` | 运动 | 11.67→10.05→9.94 | 0.0182→0.0181→0.0172 | time r=−0.89, err r=−0.91 | **有效** |
| `grasp_offset_local_xyz` | 几何(link局部) | 8.77 / 11.67 / 10.82 | −Y:0.0140 / 0:0.0173 / **+Y:0.0687** | err 跨度 ×5 | **有效** |
| `target_open_position`(任务目标) | 几何 | 10.81/10.11/10.37 | err r=−0.97, track r=−0.89 | err/track 有效 | **有效** |
| `close_duration` | 时序 | 11.17/10.05/10.46 | 0.0177/0.0180/0.0176 | 无单调 | **无可测效应** |

图：`_analysis/<skill>_<param>_per_level.png`、`_analysis/<skill>_<param>_time_vs_accuracy.png`。

## 2. 七个问题的回答

**Q1 哪些执行参数真实影响执行时间？**
抽屉 `max_pos_step`（16.6→8.7 s，r=−0.92）、`pull_lead`（r=−0.89）；放置 `move_max_position_step`
（5.5→4.0 s，r=−0.94）、`open_duration`（r=+0.79）、`settle_after_release_duration`（r=+0.84）、
`release_clearance`（r=−0.87）。命令步长越大、用时越短；时序参数线性加到总时长。

**Q2 哪些执行参数真实影响终态精度？**
抽屉 `max_pos_step`（drawer_err 0.022→0.009，r=−0.99）、`grasp_offset_local_xyz`（+Y 偏移使 err
0.014→**0.069**）、`pull_lead`（r=−0.91）、`target_open_position`（r=−0.97）；放置 `release_clearance`
（object_err 随抬高单调 ↑，r=**+0.98**）、`move_max_position_step`（r=−0.75）、`open_duration`/`settle`（r≈+0.75）。

**Q3 哪些执行参数真实影响轨迹误差？**
抽屉 `grasp_offset_local_xyz`、`target_open_position` 对 `maximum_tcp_tracking_error` 有效
（边缘抓取/目标改变接近几何）；`max_pos_step`/`pull_lead` 趋势一致但跨度略低于阈值。
放置的 max_tcp_tracking_error 由**接近段**主导，对本批 θ 不敏感（档内 std 极小、跨度小）→ 放置应以
**物体误差**而非 TCP 跟踪误差作为主精度指标。

**Q4 哪些参数能产生可重复失败？**
- 放置 `OBJECT_OUT_OF_BOUNDS`：在 **reset_index=102**（配对设计里每档同一 r=2）跨所有档位**可重复**出现
  （object 终态偏目标 0.17–0.20 m）；且在最高 `move_max_position_step` 档，**同一初始条件**失败模式变为
  `PLACE_VERIFICATION_FAILED`（err 0.073）→ **相同 x、不同 θ → 不同失败模式**。
- 抽屉 `middle_drawer`：可重复 `POSITION_TIMEOUT`（reach）。
- 抓取 setup_failure：在难布局上可重复（已与 place 失败分离）。
- `grasp_offset_local_xyz` +Y：可重复地把 err 推高近 5×（边缘抓取退化），虽未到失败阈但是确定的质量退化。

**Q5 是否存在成功率相近但执行质量明显不同的参数组？**
**是，且非常清晰。** 抽屉 `max_pos_step` 三档**都 100% 成功**，但 elapsed 16.6 vs 8.7 s、drawer_err
0.022 vs 0.009 m；`grasp_offset_local_xyz` 三档**都 100% 成功**，但 err 0.014 vs 0.069 m。
→ 单一成功标签无法区分这些方案，**必须**用多维结果描述——本阶段的核心论据成立。

**Q6 是否具备进入 50–100 次联合 Pilot 的条件？**
具备（见 §3 Go/No-Go）。建议联合 Pilot 输入参数空间剔除无效参数（`close_duration`、
`descend_max_position_step`），并提高放置重复数以补偿 grasp setup_failure。

**Q7 Go / No-Go？** → **GO**（见 §3）。

## 3. Go / No-Go 判据逐条

| # | 判据 | 结论 |
|---|------|------|
| 1 | place 与 drawer 都能自动批量运行 | ✅ `run_stage1_pilot.py` |
| 2 | 参数真实改变执行轨迹 | ✅ grasp_offset 改 err ×5、max_pos_step 改时间 ×2 |
| 3 | 执行时间存在稳定变化 | ✅ 多参数 r≈0.9 |
| 4 | ≥1 精度/跟踪指标稳定变化 | ✅ drawer_err r=−0.99 等 |
| 5 | ≥1 失败模式可重复 | ✅ place OOB@reset102；middle reach |
| 6 | 成功率相近但时间/误差不同 | ✅ drawer 全 100% 但时间/精度差异大 |
| 7 | 结果不是仅由超时/标签制造 | ✅ success 基于物体/关节真值；有效指标是连续时间/误差 |
| 8 | 数据 schema 完整 | ✅ 四组分离 + 统一 step NPZ |
| 9 | 固定 seed 可复现 | ✅ 配对 reset_index 确定性；档内方差极小 |
| 10 | 默认行为未破坏 | ✅ place 10/10、top 10/10 |

No-Go 触发项：**无**（差异非噪声；失败非全 IK 崩溃；成功非状态机末态）。→ **GO**。

## 4. 已知限制

- **中抽屉不可达**：默认 middle 0/10（`MOVE_TO_PRE_GRASP` reach 超界）。联合 Pilot 暂用 top；
  middle 需调几何/seed 或降低 pre_grasp_clearance（属后续）。bottom 锁死不纳入。
- **放置有效 n 偏小（=3）**：5 重复中 ~40% 在难布局上 grasp setup_failure 被剔除。趋势相关系数已高
  （0.75–0.98），但联合 Pilot 建议每点 ≥8 重复或改善抓取以增样本。
- **放置 TCP 跟踪误差不区分本批 θ**：放置主精度用物体误差。
- 接触维度本轮为空（Stage 1.5 计划见 implementation 文档）。

## 5. 无效 / 应剔除参数（第一版模型输入）

- `close_duration`（抽屉）：握持时长在 0.5–1.5 s 内对时间/精度无可测影响。
- `descend_max_position_step`（放置）：在 0.003–0.009 内对终态精度无单调影响（下降段短）。

## 6. 进入联合 Pilot 的建议参数空间

放置 θ：`move_max_position_step`、`release_clearance`、`open_duration`、`settle_after_release_duration`
（+ 任务目标 target_surface_xyz、初态）。
抽屉 θ：`max_pos_step`、`pull_lead`、`grasp_offset_local_xyz`（+ 任务目标 target_open_position、初态）。
采样：Sobol/LHS 联合，每技能 50–100 次；放置提高重复或改善抓取。
