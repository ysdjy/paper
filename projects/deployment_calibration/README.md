# deployment_calibration — Deployment-Conditioned Plan Evaluation and Calibration

论文方法层（round-1）。**旧 `franka_skill_state_machine` 仅作物理执行后端**；本目录负责论文数据契约、
干净的独立-episode 数据生成、baseline、指标与候选选择评估。冻结契约见
`docs/paper_experiment_contract_v1.md`，平台审计见 `docs/paper_platform_*_v1.md`。

## 关键修复（相对旧 pilot）
- **R1 episode 独立性**：`adapters/isaac_open_drawer.reset_full` 每集把 Franka 写回 default joint
  state（+零速）、重置物体/柜关节、settle → 同 reset_index 起始位姿**完全一致**（已验证）。
- **R3 误差三分**：`phase_goal_error` / `command_tracking_error` / `task_outcome_error` 分开存储命名。
- **R4 时间分层**：`skill_elapsed_time`(sim) / `phase_durations` / `wall_clock_time`。
- **R2 setup**：open_drawer 无前置技能，规避 grasp setup_failure 混杂（首技能选 open_drawer 的原因之一）。
- **R10 防泄漏**：评估按 `reset_index` 分组划分 train/test。
- **R8 特权标注**：`handle_pose` 标 `privileged_in_sim`，公平状态向量默认不含它。

## 结构
```
contracts/episode_schema.py        # Episode(x,g,theta,y) + theta 范围 + 失败归一化
data_generation/sampler.py         # (reset_index, drawer, g, theta) 候选采样 + candidate_group
adapters/isaac_open_drawer.py      # full reset + 跑一集 open_drawer + 干净标签
data_generation/generate_open_drawer.py  # Isaac 入口（每集 full reset）
baselines/predictors.py            # B0 rule / B1 state-only / B2 state+history（numpy）
evaluation/metrics.py              # AUROC/Brier/F1/MAE + selection regret/top-1
evaluation/run_eval.py             # 离线：按 reset_index 划分→训练→预测+选择评估+H1/H2
```

## 运行
```bash
conda activate env_isaaclab
# 数据生成（每集完整 reset）
./isaaclab.sh -p projects/deployment_calibration/data_generation/generate_open_drawer.py \
  --n_conditions 12 --candidates 3 --seed 7 --drawers top_drawer \
  --run_id kill_test --output_dir projects/deployment_calibration/data --headless
# 离线评估（无需 Isaac）
python projects/deployment_calibration/evaluation/run_eval.py \
  --episodes projects/deployment_calibration/data/kill_test/episodes.jsonl \
  --out      projects/deployment_calibration/data/kill_test/_eval
```

## Round-1 范围
H1（执行偏差有结构）+ H2（近期历史是否有价值）；技能 open_drawer。
VLM/vision baseline（B3–B6）、place、相机、FoundationPose、π0.5、真机推后。
