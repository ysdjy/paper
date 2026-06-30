# 论文导向平台审计 v1 (paper_platform_audit_v1)

> 目的：明确当前平台**哪些模块能可信、可复现地支撑论文实验**（不评价建设成果）。结论基于本地真实代码与
> 运行结果，不依据 README/旧文档。

## 0. 冻结的仓库状态
- branch: `push-v2-skill-runtime`
- HEAD: `f2e77e7e5be04a8a3b9541f67dc87619e16cb912`
- log -5: stage1 commit 6/5/4/3/2（见 `git log`）
- 未提交：16 个已跟踪文件 modified（teleop/door 配置与若干 skill/runtime 文件）+ 大量 untracked 新项目。
  **不删除、不 reset、不 clean、不覆盖。** 本审计所有运行均基于上述 HEAD。

## 1. 控制链（真实，已核实）
```
SkillRequest → SkillExecutor → Skill State Machine → (TCP目标 step_pose → DLS IK → q_des) → 8维 joint action
→ env.step → SceneStateProvider.get_state → 状态转移 → SkillResult
```
| 项 | 值 | 来源 |
|----|----|------|
| task id | `Isaac-Stack-Cube-Franka-JointPolicy-v0` | franka/__init__.py:27 |
| physics_dt / decimation / control_dt | 0.01 s / 5 / **0.05 s (20 Hz)** | stack_env_cfg.py:189,192 |
| action | 8 = 7 臂(JointPositionAction scale=1, use_default_offset=True)+1 夹爪 | stack_joint_policy_env_cfg.py:63 |
| gripper | +1 开 / −1 合 | scene_state_provider.py:193 |
| TCP | panda_hand + local +Z 0.1034 m | ik_joint_adapter.py:28 |
| 四元数 | (w,x,y,z) | scene_state_provider 注释 |
| IK | DLS, command_type=pose, absolute, max_joint_step=0.20 rad/step | ik_joint_adapter.py |
| 状态机周期 | 每 control step 调一次 skill.step | run_stage1_pilot._run_skill_loop |
| 多环境 | **否**，两入口强制 num_envs=1 | run_stage1_pilot:198 等 |
| episode reset | ⚠️ **不重置机器人**（见 §4 / bias register R1） | run_stage1_pilot |

## 2. 技能核对（聚焦论文相关）
完整逐技能流程见 `experiment_stage1_code_audit.md`（同 HEAD）。本轮关注论文可用性：

| 技能 | 物理交互 | 可调参数真入执行 | 成功判据 | 适合论文数据 |
|------|----------|------------------|----------|--------------|
| grasp | 真实 | 部分（cube_grasp_z_offset 等） | 抓取验证+提升 | 仅作 place 前置；setup_failure ~40% |
| place | 真实 | ✅（已参数化，object-verified） | **物体级**（pos/ori/速度/掉落/越界/稳定） | 可用，但**被 grasp 前置混杂** |
| open_drawer | 真实物理拉动，不写关节目标 | ✅（max_pos_step/pull_lead/grasp_offset_local 等，已实证有效） | **机构级**（关节位 ≥ target−tol） | **最适合**（无前置、参数有效、时空结果清晰） |
| close_drawer | 真实 | 仅 override_grasp_local | 关节 ≤ 阈 | 次要 |
| open_door/close_door | 真实（旋转门） | 少 | 角度/超时 | 本轮不用 |

## 3. 场景 / 感知核对
- 技能部署环境（JointPolicy）**无相机**：front/wrist RGB、RGB-D 在该环境**不存在**，需另加（v1_skill_lab/sensors 有相机适配器，但未接入技能环境）。
- 物体/把手 pose 来源：`SceneStateProvider`（sim GT，root/body pose）；抽屉把手 = link body ∘ link-local 偏移（来自保存的 grasp_poses.json / proxy / mesh / 配置）。**均为 sim 特权信息**。
- FoundationPose：`projects/foundationpose_deploy/` 有真实 clone + 权重（独立 conda env），但**未接入技能数据链**；本轮视为接口存在、未集成。
- 保存场景/registry：`franka_v1_skill_lab/scene` + saved_scenes；本轮未独立验证 restore 一致性（不在论文 round-1 关键路径）。

## 4. episode 独立性（关键，已实证）
现有 `run_stage1_pilot.py` 每集只 `layout_manager.reset_layout`(物体+柜关节)+`_reset_drawer`+`_settle`+
`executor.reset()`，**从不重置 Franka**；`env.reset()` 仅启动时调一次。
实证（regression_drawer_top 的 `initial_robot_joint_position`）：ep0 从 home，ep1=ep0 终态，ep2–9 ≈ 上一集终态。
⇒ **episode 不独立**，起始位姿随执行顺序漂移。详见 bias register R1。**这是正式数据的头号阻塞，必须修。**

## 5. VLM / VLA 核对（仓库全搜）
| 能力 | 状态 |
|------|------|
| Qwen2.5-VL 感知 | `perception_qwen/`（server+client，本轮未跑权重）→ 存在、未验证 |
| scene_describer (GPT/VLM 场景描述) | 存在（中转 API）→ 非候选计划生成 |
| **VLM 候选计划生成 / skill blueprint** | **缺失**（无 candidate planner；source 里的 *blueprint* cfg 是无关 Isaac 遥操作模板） |
| VLM 隐藏特征提取 | **缺失** |
| π0.5 训练 / policy server | `pi05_isaacsim_baseline` 有 HTTP server（历史验证 mock+real）；本轮未跑 |
| 闭环评估 | `franka_v1_skill_lab/data/pi05_eval`（历史 rollout，多为 fail）；非本论文链路 |

**结论**：论文 H3 所需的 VLM 候选计划/特征**当前不存在**，需后续构建；round-1（H1/H2：状态+历史）**不依赖** VLM，可先做。

## 6. 审计总结论
- 控制链、open_drawer/place 技能、参数解析、logger：**真实可信**（已多次 GPU 运行+语义核验）。
- **致命阻塞**：episode 不独立（R1）；标签语义混淆（setup_failure/误差命名/时间，R2–R4）。修复前现有统计不可作正式结果。
- VLM 候选计划/相机/FoundationPose 集成：缺失或未集成，round-1 不需要。
- 推荐第一技能：**open_drawer**（评分见 reuse_decision）。
