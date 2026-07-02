# 接口成熟度矩阵 v1 (v1_interface_maturity_matrix_v1)

> 等级：E0 仅文档 / E1 代码存在 / E2 静态导入或编译过 / E3 mock 测试过 / E4 GPU 单次跑过 /
> E5 GPU 重复跑过 / E6 输出数据经语义核验 / E7 可用于正式论文。审计 HEAD `f2e77e7e`，
> 等级仅反映**本审计已亲自验证**的程度（保守）。

| 模块 | 等级 | 依据 / 备注 |
|------|------|------|
| SkillExecutor | E6 | 多次 GPU 跑、结果语义核验；reset 不含机器人(待 ADAPT) |
| grasp skill | E5 | GPU 重复跑；成功率受布局影响，未做正式语义核验 |
| place skill | E6 | GPU 重复、object-verified、误差核验；命名/时间待修 |
| open_drawer skill | E6 | GPU 重复、机构级核验、敏感性显著 |
| close_drawer skill | E4 | 历史跑过；本轮未重验 |
| open_door / close_door | E2 | 代码在；本轮未跑 |
| parameter resolver (experiment_params) | E7* | 12 例单测全过；纯逻辑，*=对其职责可用于论文 |
| episode logger | E6 | 自测+真实 run 产出核验 |
| trajectory logger | E6 | 统一 schema、NaN、读回核验 |
| pilot runner (run_stage1_pilot) | E5→需 ADAPT | 重复跑过，但 R1 reset 缺陷 |
| Scene contract / SceneStateProvider | E5 | 多次跑；sim GT 边界已厘清 |
| Scene registry / manifest restore | E1 | 代码在(franka_v1_skill_lab/scene)；本轮未验证 restore 一致性 |
| Layout editor | E1 | 代码在；未验证 |
| GELLO read | E1 | gello_franka_teleop 代码在；本轮未跑硬件 |
| teleop drive | E1 | 代码在；未跑 |
| HDF5 recorder | E1 | 代码在；34 个历史 hdf5 存在 |
| front RGB / wrist RGB / RGB-D | E1 | 相机适配器在 v1_skill_lab/sensors；**技能环境未接入**，未渲染验证 |
| FoundationPose | E1 | 真实 clone+权重(独立 env)；**未接入技能链**，本轮未跑 |
| π0.5 converter | E1 | 代码在；未验证 |
| π0.5 train | E1 | baseline 工程在；本轮未跑 |
| policy server | E1 | 历史 mock+real；本轮未跑 |
| closed-loop evaluation | E1 | 历史 pi05_eval 产物(多 fail)；非本论文链路 |
| Qwen3-VL / Qwen2.5-VL | E1 | perception_qwen server/client 在；权重未下/未跑 |
| VLM candidate planner | **E0** | **不存在**（论文 H3 所需，待建） |
| VLM hidden-feature extractor | **E0** | 不存在 |

**round-1 关键路径模块**（需 ≥E6 才进正式数据）：open_drawer、parameter resolver、loggers、
SceneStateProvider、pilot runner(修后)。其余 round-1 不依赖。
