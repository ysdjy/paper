# paper/ — 论文项目（与平台/感知-记忆项目代码隔离，共用测试场景）

本文件夹是**论文专用**工作区。设计目标：论文项目与"感知模块/记忆模块"等其它项目**互不感染**，但
**共用同一个会持续更新的测试场景**。

## 隔离策略（两项目完全独立运行）
- **执行后端隔离**：`paper/skill_backend/` = `franka_skill_state_machine` 的 runtime/skills/
  state_machine/learned_drawer **冻结副本**。论文只 import 这份副本。
- **场景隔离（已私有化）**：`paper/scene/stackpkg/` = 上游 `manipulation/stack` + franka env cfg 的
  **冻结私有副本**；`paper/scene/paper_tasks.py` 注册**论文自有任务 `Isaac-Paper-OpenDrawer-Franka-v0`**。
  论文**不再** import 共享的 `isaaclab_tasks` franka 配置（其包自动导入会把另一项目对共享 cfg 的编辑
  牵连进来）。数据生成入口直接实例化 paper cfg，不调用 `parse_env_cfg`/`import isaaclab_tasks`。
  `skill_backend` 的 `drawer_target_config`/`microwave_door_config`/`target_registry` 均改指
  `stackpkg` 私有副本。
- **资产隔离**：`paper/assets/` 存 Cabinet_44853 / Knife / CoffeeMachine / Microwave / Dishwasher /
  panda_instanceable，paper cfg 的 `_repo_path` 优先解析这里。**Sektion 橱柜（310M，另一项目频繁改动）
  默认禁用**（open_drawer 不需要；`PAPER_ENABLE_SEKTION=1` 可开）。cube 为程序化立方体无需资产。
- **数据隔离**：`paper/deployment_calibration/data/`（大文件 .gitignore）。
- **仅共享框架**：`isaaclab`/`isaaclab_assets`（Isaac Lab 框架本身）+ Isaac Sim。这是运行时基础，不是
  会被编辑的"项目"。

> 已验证：`Isaac-Paper-OpenDrawer-Franka-v0` 独立加载并跑 open_drawer 产出数据；把手位姿在 paper 内
> 回退到 config 偏移（不再依赖另一项目的 grasp_poses.json）——隔离生效。
> USD 内部为二进制绝对引用，若删除共享 `simv2/` 可能触发子引用告警（非致命）；如需彻底冻结资产可后续
> 重导出 USD。

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
