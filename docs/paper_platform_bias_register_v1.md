# 平台偏差与风险登记 v1 (paper_platform_bias_register_v1)

> 审计 HEAD `f2e77e7e`. 严重度：CRITICAL=阻塞正式数据 / HIGH=显著偏倚结论 / MED / LOW。
> "证据"均来自真实代码或本地运行。

| # | 风险 | 证据 | 严重度 | 影响的论文结论 | 必修? | 建议处理 | 成本 | 状态 |
|---|------|------|--------|----------------|-------|----------|------|------|
| R1 | **episode 不独立（机器人不重置）** | regression_drawer_top: ep1 初始关节=ep0 终态；runner 仅启动 env.reset 一次，每集不 reset 机器人 | **CRITICAL** | H1/H2 全部统计被起始位姿×执行顺序混杂；selection regret 不可信 | ✅ | runner 每集把 Franka 写回 default joint pos(+zero vel) 并 settle；同 reset_index 验证初态一致 | 低 | **修复中(task8)** |
| R2 | setup_failure 语义混淆 | runner: 首次抓取失败即 setup_failure=True，重试成功仍 True；分析里被剔除 | HIGH | place 有效样本被错误剔除、成功率被低估/混淆 | ✅ | 拆 setup_retry_occurred / setup_attempt_count / setup_final_success / terminal_failure_stage | 低 | 待修(task8) |
| R3 | 空间误差命名误导 | place/drawer 的 maximum_tcp_tracking_error = TCP vs **阶段目标**(phase goal) | HIGH | "跟踪误差"被误读为控制跟踪；混淆 phase/command/task 三种误差 | ✅ | 重命名 phase_goal_error_max；新增 command_tracking_error(TCP vs 当帧命令) 与 task_outcome_error(物体/关节 vs 任务目标) | 低 | 待修(task8) |
| R4 | 时间语义未分层 | outcomes 仅 elapsed_time(sim) | MED | 代价指标不清；无法分阶段/墙钟 | ✅ | 记 simulation_time/skill_elapsed_time/phase_durations/wall_clock_time | 低 | 待修(task8) |
| R5 | 抓取前置筛选(place) | place 需先 grasp，setup_failure ~40% 难布局被丢 | HIGH | place 数据对“易抓布局”有选择偏倚；(x,g,θ)→y 被 grasp 混杂 | ✅(对 place) | 首技能改用**无前置**的 open_drawer；place 推后 | 低 | 已决策(reuse) |
| R6 | 状态机阈值造成标签偏倚 | 成功/前进判据用固定阈值(target_tolerance 等) | MED | 成功标签依赖阈值选择 | 部分 | 阈值写入 contract 冻结、记录连续误差另判 | 低 | contract 固定 |
| R7 | 超时造成失败偏倚 | reach/pull timeout 触发 POSITION_TIMEOUT/DRAWER_OPEN_TIMEOUT | MED | 失败类型可能由超时人为制造 | 部分 | 区分真实失败 vs 超时；超时单列 failure_reason | 低 | contract 固定 |
| R8 | sim GT 作为输入 | 物体/把手/关节 pose 全 sim GT | HIGH | state-only 用了真机不可得的特权信息 → 不公平、sim2real 乐观 | ✅(标注) | contract 明确：sim GT 仅作 label；输入只用部署可得代理；特权字段打标 | 中 | contract 固定 |
| R9 | 感知 vs 结构化状态不公平 | 无相机；state 用 GT，vision baseline 未就绪 | MED | H3/vision 对比暂不可做 | 否(round1) | round-1 只做 state/history；vision/VLM 推后 | — | 记录 |
| R10 | 训练/测试泄漏 | 配对 reset_index 跨档复用 | MED | 同初态出现在多档 → 若随机划分会泄漏 | ✅ | 按 reset_index/初态分组划分 train/test | 低 | data_gen 固定 |
| R11 | 固定 cabinet / 单场景 | 柜子位姿固定，单场景 | MED | 泛化结论受限 | 否(round1) | round-1 限定单场景结论；多场景推后 | — | 记录 |
| R12 | 只保留成功数据? | 否——失败 episode 也写入(OBJECT_OUT_OF_BOUNDS 等已记录) | LOW（未发生） | — | 保持记录全部 | — | OK |
| R13 | 参数范围过窄 | 第一轮 0.5–1.5×（避免极端） | MED | 失败率可能偏低、效应受限 | 部分 | 正式 round 适度扩范围以诱发足量失败 | 低 | data_gen 调整 |
| R14 | 结果全 0/1（无连续性）? | 否——elapsed/误差/overshoot 连续，效应显著 | LOW（未发生） | — | 继续保留连续 outcome | — | OK |
| R15 | 文档与代码版本不一致 | 旧文档 control_dt=0.02 vs 真实 0.05 | LOW | 易误用 | 部分 | 以 audit_v1 为准，旧示例标注 | — | 已纠 |
| R16 | sim/真机信息不对称 | sim 有 GT，真机需 FoundationPose/编码器 | HIGH(未来) | sim2real 迁移 | 否(round1) | contract 预留真机字段映射 | 中 | 记录 |

**优先修复（进入正式数据前）：R1, R2, R3, R4, R8（标注），R10（划分）。** R5 已由“首技能选 open_drawer”规避。
