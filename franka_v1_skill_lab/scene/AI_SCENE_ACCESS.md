# 场景访问契约 —— 给其他模块 / AI 的规则与权限

> 适用对象：`projects/franka_v1_skill_lab/` 下除布局编辑器以外的**所有模块**，
> 以及任何在本仓库里跑脚本的 AI（skill_runtime / teleop_collection / pi05_training /
> perception_foundationpose 等）。开工前请先读完本文件。

本工作区有**唯一一个最新场景**，所有模块都基于它运行。本文件规定：你**怎么只读还原**这个
场景、你有什么**权限**、以及 cube/小刀的**初始化区域契约**。

---

## 0. 一句话规则

**最新场景是只读的单一可信源；你只能读取、不能修改。**
唯一能修改它的途径，是真人用户在布局编辑器里点击 **Save V1**。

---

## 1. 权限：只读，不可写（HARD RULE）

最新场景由两类受保护文件组成，位于
`scene/saved_scenes/v1_active/`：

| 文件 | 含义 |
|------|------|
| `scene_v1_latest.usd`     | 最新场景快照（USD） |
| `scene_v1_latest.json`    | 场景 manifest（物体 + 变换 + 传感器） |
| `scene_v1_registry.json`  | 指针，所有消费者都读它来定位上面两个文件 |
| `scene_v1_previous.usd/.json` | 上一版本备份（仅供用户回滚，**勿写**） |
| `scene_v1_latest.reproduce.md` | 自动生成的复现说明（**勿写**） |

**你不可以做：**
- ❌ 覆盖 / 删除 / 编辑上述任何文件；
- ❌ 调用 `save_registry()` / `update_active_scene()` 去改 registry；
- ❌ 用 `stage.Export()` 或 `write_text()` 写 `scene_v1_latest.*`；
- ❌ 自行 `authorize_scene_write()` 绕过保护去写场景。

**强制机制**：`scene/scene_registry.py` 有一道进程级写入闸门。`save_registry()` /
`update_active_scene()` 在未授权时会抛 `PermissionError`：

```
最新场景 (scene_v1_latest.usd / scene_v1_registry.json) 为受保护资源，
只能通过 layout_editor/layout_v1_ui.py 的 Save V1 按钮修改。…
```

只有布局编辑器的 `save_v1()`（即用户点 Save V1）会在写入前后授权 / 撤销。其它任何调用都会失败。

---

## 2. 如何只读还原 / 加载最新场景

### 方式一：查询 registry 拿到路径（推荐，纯 python，无需 GPU）

```python
from franka_v1_skill_lab.scene import resolve_active_scene
info = resolve_active_scene()
# info = {
#   'task_id': 'Isaac-Stack-Cube-Franka-JointPolicy-v0',
#   'control_mode': 'joint',
#   'usd':      '.../scene_v1_latest.usd',   'usd_exists': True,
#   'manifest': '.../scene_v1_latest.json',  'manifest_exists': True,
#   'objects': [...], 'sensors': {...},
# }
usd_path = info['usd']        # 拿去加载
```

> 始终用 `resolve_active_scene()` 动态定位，**不要把路径写死**——用户随时会保存新场景，
> 而固定名 `scene_v1_latest.*` 永远指向最新。

### 方式二：在 Isaac 里直接加载该 USD

```python
from isaaclab.sim.utils.stage import open_stage
open_stage(info['usd'])
```

### 方式三：只看物体清单（不开 Isaac）

```bash
python projects/franka_v1_skill_lab/scene/tools/print_scene_objects.py          # 读 manifest
```

### 复现说明

`scene/saved_scenes/v1_active/scene_v1_latest.reproduce.md` 由每次保存自动生成，含直接加载
命令、逐物体 pos/rot/scale/asset 清单、外部 asset 依赖、回滚步骤。USD 丢失时按它程序化重建。

---

## 3. 初始化区域契约：四个黄色 InitCorner 方块（HARD RULE）

场景里有四个黄色方块 `InitCorner_0`、`InitCorner_1`、`InitCorner_2`、`InitCorner_3`。
它们**不是道具**，而是**标定块**：四个点构成一个四边形，**就是** `cube_1 / cube_2 / cube_3`
和 `knife` 每次场景初始化时允许出现的区域。

> **规则：以后每一次场景初始化 / reset，三个 cube 和小刀都必须随机初始化在这四个点
> 构成的区域之内**（并带最小间距，避免互相穿插 / 与机械臂冲突）。

区域由用户在编辑器里拖动黄块决定，会随场景更新而改变，所以**必须每次从 manifest 动态读取，
绝不能写死坐标**。已提供纯 python helper：

```python
from franka_v1_skill_lab.scene import (
    load_init_region, sample_in_region, point_in_region, INIT_REGION_OBJECTS,
)

region = load_init_region()                  # -> [(x,y) x4]，CCW 凸多边形，来自当前 manifest
# 为 cube_1/2/3 + knife 采样合法初始 (x,y)：
xy = sample_in_region(region, count=len(INIT_REGION_OBJECTS), seed=reset_index)
# 校验任意点是否合法：
ok = point_in_region(x, y, region)
```

- `load_init_region()` 从 `scene_v1_latest.json` 读 `InitCorner_0..3` 的 `world_transform.translation`
  的 (x,y)，按质心极角排成 CCW 凸多边形。四个角缺任意一个都会抛错（场景未标定）。
- `sample_in_region()` 在区域内做拒绝采样，两两间距 > 0.15 m（与现有采样默认一致）。
- 区域顶点不足 / manifest 缺失会**显式抛错**，不要静默兜底到旧的硬编码区域。

### ⚠️ 当前实现状态（务必知晓）

- 运行时采样器 `runtime/simple_scene_layout.py` 的 `SimpleSceneLayoutManager` **目前仍用
  硬编码的 `PLACEMENT_REGION`，尚未接入 `load_init_region()`**。
- 按本契约，它**应当**改为从 `load_init_region()` 读区域。改它会改变 reset 行为（影响
  teleop 采集 / 训练分布），属于行为变更——**由用户决定何时切换**，不要擅自改。
- 在切换之前，新写的任何 reset / 随机化逻辑都应直接用本文第 3 节的 helper，以符合契约。

---

## 4. 速查

| 我想… | 用 |
|-------|----|
| 拿到最新场景 USD / manifest 路径 | `resolve_active_scene()` |
| 读取 registry（只读） | `load_registry()` |
| 拿 cube/knife 的合法初始化区域 | `load_init_region()` |
| 在区域内采样初始位置 | `sample_in_region(region, count, seed)` |
| 看物体清单 | `tools/print_scene_objects.py` |
| 修改场景 | ❌ 不行 —— 让用户在 `layout_v1_ui.py` 里 Save V1 |

有疑问先读 `scene/README.md` 与本文件；**任何写最新场景的需求都交给用户经 UI 完成。**

---

## 5. 测试接入（统一测试入口 = test_mode）

所有功能测试都在 **`scene_interface/test_mode_ui.py`** 里进行（不要自己另起一套场景）。你的模块（状态机 / FoundationPose / VLA）有两种接入方式：

- **(A) 把控制接入 test_mode 的 UI**：写一个 `Controller(session)` 类（可选 `build_window()`/`on_reset()`，必有 `step(session)->env_action|None`），用 `--controller your_pkg.module:Class` 加载。返回 action 就驱动机器人，返回 None 交还默认。
- **(B) 自己写 entry 用 `SceneSession` 库**：`launch → observe → step`，完全自主。

`session.observe()` 提供各模块所需全部信息：`pi05`(image/wrist_image/state8) 给 VLA；`foundationpose`(RGBD+内参+位姿+GT) 给 FP；`raw`+`handles`(把手在机器人基座系 pose) 给状态机；`cameras`(front/wrist 同视角 RGBD)。
可视化开关由调试者自选：`session.set_collision_visible(b)` / `session.show_target_poses(...)`+`clear_target_poses()` / `session.set_markers_visible(b)`（test_mode 的 Visualization 面板有复选框）。
实时图像：相机帧经 ZMQ 发到独立网页查看器(`image_viewer.py`)；FoundationPose 处理后的图往同一 ZMQ addr 发新 topic 即同屏显示。
详见 `scene_interface/README.md` §0。
