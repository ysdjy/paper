# Round-1 Kill Test 裁决 (paper_round1_kill_test_v1)

> 技能 open_drawer。数据 `projects/deployment_calibration/data/kill_test`（36 独立 episode，每集完整
> reset，宽 θ）。评估 `.../kill_test/_eval/eval_report.json`。HEAD 见 git_commit.txt。

## 1. 结果
- **管线端到端通过**：full reset（episode 独立，已验证初态一致）→ 数据生成 → 离线评估 → regret，全链路跑通。
- **H1（执行偏差有结构）= 成立**：success_rate 0.53；成功集 task_outcome_error 0.003–0.106、失败含
  task_error 至 0.34；失败模式可重复（HANDLE_DETACHED ×15、REACH_TIMEOUT ×2）；`max_pos_step` 对
  skill_elapsed_time corr=−0.88。→ 同一 IK 技能下，x/g/θ 稳定改变成败、终态误差与代价。
- **预测/选择尚未改善（在 kill-test 规模）**：B1 succ_AUROC≈0.45、B2≈0.46（≈随机）；selection regret
  B0/B1/B2≈0.86、top1=0.2、selected_success=0.4。B2(+history) 几乎不优于 B1。

## 2. 诊断（为何预测/选择未改善）
1. **样本太小**：36 集（train 24 / test 15 / 仅 5 个候选组）无法学习多维 θ→y 映射；指标处于噪声区。
2. **平稳性（关键）**：round-1 干净 sim = 完整 reset + 固定单场景 + 无感知噪声 → **不存在部署漂移**，
   近期历史 H **没有可校准的隐藏部署状态**，因此 H2 在当前设定下**结构上接近零**。这不是平台缺陷，而是
   实验设定缺少“deployment-conditioned”的核心前提。
3. 候选组小（每组 3 个 θ），regret/top1 分辨率低。

## 3. 裁决：**MODIFY**（非 Stop，非完全 Go）
依据 task §22：管线可信、H1 成立、可信 reset 达成 → 不 Stop；但 H2 增益未显现且当前设定下结构性偏弱 →
未达完全 Go。需按下述修改后再扩大正式数据。

### 必须的修改（进入正式 round-1 数据前）
1. **引入受控非平稳（deployment drift）**：按 *session* 注入**不可观测**的部署参数，使近期历史成为有信息量
   的状态估计来源。候选（安全且真实生效）：
   - 抽屉关节 `damping`/`friction`（每 session 随机，state 不可见）；
   - 把手 pose 估计偏置（模拟感知漂移，仅影响抓取，不进 state）；
   - 控制噪声/动作缩放（每 session 固定的小扰动）。
   H2 检验 = 同一 x/g/θ 在不同隐藏 session 参数下 y 不同，且 B2(+H) 显著优于 B1。
2. **扩大数据**：每技能 数百–千 级 episode；候选组 ≥6 个 θ/组，多 session（每 session 多 episode）。
3. **诱发足量失败谱**：θ 范围已够（53% 成功）；保持。
4. （可选）**place 作为第二技能** + 中抽屉，增加结构与失败多样性。

### Modify 不触发 Stop 的理由
- 失败非全 IK 数值崩溃（是物理脱离/可达）；成功基于真实机构状态；结果含连续时间/精度差异；
  reset 可信；无需重写机器人系统。

## 4. 下一步（已就绪，待执行）
- 在 `adapters/isaac_open_drawer` 增 `session` 级隐藏参数注入（env_cfg actuator damping/friction +
  handle bias + action noise），`generate_open_drawer` 增 `--sessions/--episodes_per_session`。
- 生成 ~300–600 episode（多 session）→ 重跑 `run_eval`，看 B2 vs B1 的 AUROC/regret 是否显著改善 →
  得出 round-1 H1/H2 正式结论；H3(VLM) 在数据链稳定后接入。

> 一句话：平台已可信、H1 已证、管线已通；**要让“近期历史”有价值，必须先让部署条件真的会变**。这是下一步的
> 唯一关键改动，其余基础设施均已就位。
