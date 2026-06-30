# paper/ — 论文项目（与平台/感知-记忆项目代码隔离，共用测试场景）

本文件夹是**论文专用**工作区。设计目标：论文项目与"感知模块/记忆模块"等其它项目**互不感染**，但
**共用同一个会持续更新的测试场景**。

## 隔离策略
- **代码隔离**：论文需要的执行后端已**冻结复制**到 `paper/skill_backend/`（来自
  `franka_skill_state_machine` 的 runtime/skills/state_machine/learned_drawer）。论文代码只 import 这份
  自有副本，**不** import 共享的 `franka_skill_state_machine`。因此平台或感知项目改动那边的代码，
  **不会影响论文实验**；反之亦然。
- **场景共享（有意）**：测试场景 = 注册的 gym 任务 `Isaac-Stack-Cube-Franka-JointPolicy-v0` + 资产
  (`simv2/`, `SapienAssetPipeline/`) + env cfg (`source/isaaclab_tasks/.../franka/`)。这部分**共享**，
  会持续更新。论文通过 task id 使用它；`skill_backend/runtime/drawer_target_config` 仍引用 source 下的
  `custom_drawer_config`（场景定义）。**这是唯一有意的共享耦合点**。
- **数据隔离**：论文数据写在 `paper/deployment_calibration/data/`（大文件 .gitignore）。

> 若将来希望连场景也固定（防止共享场景更新破坏论文），可把 env cfg + 资产也快照进 paper/；当前按你的
> 要求保持场景共享。

## 结构
```
paper/
├── skill_backend/            # 冻结的执行后端（论文自有副本；勿与平台同步）
│   ├── runtime/ skills/ state_machine/ learned_drawer/
└── deployment_calibration/   # 论文方法层（Deployment-Conditioned Plan Evaluation）
    ├── contracts/ adapters/ data_generation/ baselines/ evaluation/ configs/ data/
```

## 当前状态（round-1, open_drawer）
- 审计/契约/偏差登记：见`projects/paper/docs/paper_*_v1.md`（HEAD f2e77e7 审计）。
- 干净独立-episode 管线（修复 episode 独立性 R1）+ B0/B1/B2 + regret 评估：就绪并验证。
- Kill test 裁决 **MODIFY**（见 `projects/paper/docs/paper_round1_kill_test_v1.md`）：H1 成立、管线可信，但干净 sim 平稳→
  历史 H 无可校准漂移。

## 选定的部署漂移轴（round-1）
**机构物理：每 session 随机化抽屉关节 damping/friction（不可观测，不进 state）。**
- 物理真实、安全、可辩护（真实机构阻力随磨损/温度变化）；直接影响 open_drawer 拉动结果；
  真机对应"编码器读关节、读不到摩擦" → 契合"用近期历史校准部署状态"。
- 第二轴（感知偏置：把手 pose 漂移）留作 round-2 / H3-sim2real。

## 下一步（待执行）
1. 在 `adapters/` + `generate_open_drawer.py` 加 **per-session 隐藏 damping/friction 注入**
   （每 session 重建 env 以可靠设置 actuator；漂移值记为 secret label，不入 x）。
2. 生成数百集多-session 数据 → 重跑 `run_eval` 比 **B2(+history) vs B1** 的 AUROC/regret →
   得出 round-1 H1/H2 正式结论。
3. H3（VLM/vision）在数据链稳定后接入。

## 运行
```bash
conda activate env_isaaclab
./isaaclab.sh -p projects/paper/deployment_calibration/data_generation/generate_open_drawer.py \
  --n_conditions 12 --candidates 3 --seed 7 --run_id kill_test \
  --output_dir projects/paper/deployment_calibration/data --headless
python projects/paper/deployment_calibration/evaluation/run_eval.py \
  --episodes projects/paper/deployment_calibration/data/kill_test/episodes.jsonl \
  --out      projects/paper/deployment_calibration/data/kill_test/_eval
```
