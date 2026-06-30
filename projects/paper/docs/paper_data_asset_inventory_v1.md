# 论文数据资产盘点 v1 (paper_data_asset_inventory_v1)

> 审计 HEAD `f2e77e7e`. 结论基于真实文件读取。**核心裁定：现有全部 Stage-1 数据仅为平台 pilot / 变量
> 筛选用，不可直接作为正式论文训练集或主要结果**（原因见每行 + bias register R1）。

## 1. Stage-1 仿真数据（`projects/franka_skill_state_machine/experiments/stage1/`）

| run_id | skill | commit | n(plan) | n(file) | succ | setup_flag | seed | 正式可用? | 原因 |
|--------|-------|--------|---------|---------|------|-----------|------|----------|------|
| regression_place | place | 4693462 | 10 | 10 | 10 | 4 | 42 | ❌ | episode 不独立(R1)；setup_flag 语义混淆(R2) |
| regression_drawer_top | open_drawer | 4693462 | 10 | 10 | 10 | 0 | 42 | ❌ | episode 不独立(R1) |
| regression_drawer_middle | open_drawer | 4693462 | 10 | 10 | 0 | 0 | 42 | ❌(且全失败) | 中抽屉不可达(reach) |
| place_sens_* (5 runs) | place | a7f169b | 15 each | 15 | 12 | 6 | 42 | ❌ | R1 + 有效 n 仅 3/档（重试成功被误剔，R2） |
| drawer_sens_* (5 runs) | open_drawer | a7f169b | 15 each | 15 | 15 | 0 | 42 | ❌ | R1（起始位姿随顺序漂移，与档位混杂） |

公共字段：每 run 均有 `episodes.jsonl + trajectories/*.npz + metadata.json + parameter_ranges.json +
environment_config.json + git_commit.txt + run_command.txt`；轨迹统一 schema（30 键）完整可读。
- 初态随机化：cube/knife 区域内随机（reset_index 确定性）；**但机器人起始未受控**(R1)。
- 环境 reset：仅物体+柜关节 write_joint_state，机器人未 reset。
- 标签来源：success 来自**真实物体/机构状态**（非状态机末态）✅；但误差字段命名混淆(R3)、时间未分层(R4)。
- 是否用 sim GT：是（物体/把手/关节 pose 全为 sim GT）。

## 2. 其它数据资产
| 资产 | 位置 | 状态 | 论文用途 |
|------|------|------|----------|
| HDF5 遥操作 demo (34 个) | franka_starvla_collect/data, franka_real_teleop/data, franka_v1_skill_lab/data | 存在 | π0.5/VLA 用，**round-1 不用** |
| pi05_eval rollouts (gif/json) | franka_v1_skill_lab/data/pi05_eval* | 历史评估，多为 fail | 非本论文链路 |
| libero demos (mp4) | franka_v1_skill_lab/data/libero_demos | 参考视频 | 不用 |
| logs/skill_tests/*.jsonl | logs/ | 旧 skill 测试日志 | 仅调试 |

## 3. 裁定
- **不复用任何现有 run 作为正式数据**。修复 R1–R4 后，用新的 `deployment_calibration` 数据契约重新生成。
- 现有 Stage-1 敏感性**结论**（哪些参数有效）仍可作为**变量筛选先验**（效应量很大，即使被 R1 部分混杂也大概率成立），用于设计正式参数空间——但**数值结果不进正式论文**。
- 保留现有 data（不删），仅标注为 pilot。
