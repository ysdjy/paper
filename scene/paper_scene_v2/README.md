# paper_scene_v2 — 论文 drawer 实验的唯一场景事实来源

由 `scene/tools/export_paper_scene_v2.py` 从**实时运行场景**（headless 构建 = GUI 平台同一场景）
捕获导出，`scene/tools/validate_paper_scene_v2.py` 校验。**论文数据生成器只读本目录**，不再从多个 JSON
或 README 隐式推断场景。

## 文件
| 文件 | 内容 |
|---|---|
| `scene_manifest.json` | **权威清单**：enabled 子集 + 每个对象的 pose/scale/joints/actuator + 机构注册 + privileged 说明 |
| `object_states.json` | 每个 scene 成员的运行时 root pose / scale / 关节 / actuator stiffness+damping |
| `mechanism_registry.json` | 每个抽屉：member/joint/link/handle（local+world）/actuator（member-aware） |
| `grasp_poses.json` | 抽屉把手 link 局部 pose（**单一来源**，从 active grasp_poses.json 同步） |
| `camera_registry.json` | front 相机（登记，drawer profile 默认 disabled） |
| `external_assets.json` | 外部 USD 路径 + size + mtime + sha256 |
| `version.json` | scene_version / source_commit / dirty / 时间戳 |

## enabled / disabled（drawer 实验 profile）
- **enabled**：`robot`、`cabinet`（桌面柜）、`sektion_cabinet`（地面柜）
- **disabled 但保留**：`coffee_machine`（revolute 泛化用，默认关）
- **props**：保留在场景，drawer profile 不参与（enabled=false）
- 完整 GUI 平台仍加载全部；论文实验用此精简 profile。

## 机构（5 个，member-aware）
| drawer | member | joint | link | baseline damping |
|---|---|---|---|---|
| top_drawer | cabinet | joint_0 | link_0 | 3.0 |
| middle_drawer | cabinet | joint_2 | link_2 | 3.0 |
| bottom_drawer | cabinet | joint_1 | link_1 | 3.0 |
| sektion_top_drawer | sektion_cabinet | drawer_top_joint | drawer_handle_top | 3.0 |
| sektion_bottom_drawer | sektion_cabinet | drawer_bottom_joint | drawer_handle_bottom | 3.0 |

> `joint_damping=3.0` 是运行时基线；**它是隐藏部署轴 z_secret**，pilot 会按 session 覆盖，且**绝不进普通模型输入 x**。

## 把手 pose 单一来源
`scene/paper_scene_v2/grasp_poses.json` 是论文把手 pose 的唯一来源，内容与
`franka_v1_skill_lab/.../v1_active/grasp_poses.json`（UI 侧）逐字一致（validate 强制校验）。
UI 旧文件保留；如需同步用显式脚本，**禁止两文件静默互相覆盖**。

## 三个旧场景来源的差异（审计）
| 来源 | 对象数 | 命名 | 角色 |
|---|---|---|---|
| `scene/captured_scene_v1.json` | 48 | 小写 | 旧运行时 dump（**未假设为最新**，本 v2 重新实时捕获） |
| `.../v1_active/scene_v1_latest.json` | 40 | CamelCase | 布局编辑器最新 manifest（含 schema/sensors/task） |
| `.../v1_active/scene_v1_registry.json` | 7 | 小写 | 注册表（active_manifest/usd 指针） |

v2 不删除任何旧来源；它们保留。论文只信任本目录（实时捕获 + validate 通过）。

## 重新导出 / 校验
```bash
./isaaclab.sh -p projects/paper/scene/tools/export_paper_scene_v2.py --headless
python projects/paper/scene/tools/validate_paper_scene_v2.py
```
