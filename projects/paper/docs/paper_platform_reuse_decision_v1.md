# 平台复用决策 v1 (paper_platform_reuse_decision_v1)

> 判据：是否直接支撑论文假设 / 是否提供可信数据 / 是否引入混杂 / 是否可复现 / 改造成本。
> 审计 HEAD `f2e77e7e`. KEEP=原样可用 / ADAPT=核心可靠需改 / REPLACE=需最小替代 / DROP=本轮不需。

## 1. 模块决策
| 模块 | 决策 | 说明 |
|------|------|------|
| 控制链 / JointPolicy env / DLS IK | **KEEP** | 真实可信，作为物理执行后端 |
| SceneStateProvider | **KEEP**（+标注特权字段） | 读 sim GT；contract 标注哪些是 deploy-不可得 |
| open_drawer skill | **KEEP** | 首技能执行后端 |
| place / grasp skill | **KEEP**（推后） | round-1 不用；保留 |
| parameter resolver | **KEEP** | 直接复用做 θ 解析+provenance |
| episode/trajectory logger | **ADAPT** | 字段语义修正（R3/R4），schema 复用 |
| pilot runner (run_stage1_pilot) | **ADAPT** | 修 R1(机器人 reset) + R2(setup 拆分)；或在新目录写干净 episode driver 复用其技能循环 |
| Scene registry / layout editor | **DROP**(round-1) | 单场景固定即可，不需编辑器 |
| 相机 / FoundationPose / 感知 | **DROP**(round-1) | H3/vision 推后；round-1 用结构化状态 |
| Qwen / scene_describer / π0.5 / policy server | **DROP**(round-1) | 不在 H1/H2 路径 |
| VLM candidate planner / hidden feature | **REPLACE**(后续新建) | 当前不存在；H3 阶段在 deployment_calibration 内最小实现 |
| 论文方法层（contract/adapter/data/models/baselines/eval） | **REPLACE/NEW** | 新建 `projects/deployment_calibration/`，不堆进旧状态机目录 |

## 2. 第一技能评分（phase 8）
评分 1–5（越高越适合论文 round-1），基于真实代码/数据，不因代码多而偏向。

| 维度 | grasp | place | open_drawer |
|------|:-:|:-:|:-:|
| 初态可控（可完全 reset） | 3 | 3 | 4（修 R1 后高；无前置） |
| 参数有效（真改执行） | 2 | 4 | **5**（max_pos_step/pull_lead/grasp_offset 实证强效） |
| 结果非平凡（成败+质量差异） | 2 | 4 | 4（需扩 θ/含 middle 诱发失败） |
| 空间结果（终态误差） | 2 | **5**（物体6D） | 4（关节 1D + overshoot） |
| 时间结果（代价） | 3 | 4 | 4 |
| 失败类型可区分 | 3 | 4（OOB/verify/dropped） | 4（timeout/detach/reach） |
| 数据效率（单位时间 episode） | 3 | 2（含抓取~30s, setup 40%） | **5**（~15-20s, 无前置） |
| VLM 相关性（候选可比） | 3 | 4 | 4（选哪个抽屉/开多大/抓取偏移） |
| 真机迁移 | 3 | 3 | 4（抽屉接触+编码器易测） |
| 混杂风险（依赖前置技能） | 4 | **1**（依赖 grasp） | **5**（无前置） |
| **合计** | 28 | 34 | **43** |

**决策：第一项正式实验技能 = `open_drawer`（top + middle）。**
理由：无前置技能（最低混杂，避开 R5）、参数实证强效、时空结果清晰、数据效率最高。
弱点（top 默认 100% 成功 → 失败多样性低）通过：扩大 θ（grasp_offset 至 ±0.06、pull/step 极值）、
纳入 middle 抽屉、修 R1 后引入受控起始状态变化 来诱发足量失败与质量谱。place 作为第二技能推后。
