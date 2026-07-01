# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""门把手 / 抽屉把手 / 拉手的 pose 读取（输出**机器人基坐标系**下的位姿）。

给状态机模块用：它要读这些把手在机器人基座系下的 pose 去做抓取/开门/开抽屉。

把手偏移量是项目已标定的单一可信源，从 env cfg 的 config 模块按需 import（不复制数值，
避免漂移）：
  * 微波炉门 —— microwave_door_config.DOOR_TARGETS（link_0 + 世界尺度局部偏移，直接套用）
  * 柜子三抽屉 —— custom_drawer_config.DRAWER_TARGETS（link_0/2/1 + 世界尺度局部偏移，直接套用）
  * 咖啡机拉手 —— joint_4/link_4，**偏移尚未标定**（calibrated=False，先给 link 原点）

变换：handle_world = combine_frame_transforms(link_pose_w, offset)；
      handle_base  = subtract_frame_transforms(robot_base_w, handle_world)（世界 -> 基座系）。
与 SelectedDrawerObsAdapter / microwave_door_skill 的算法一致（offset 直接套用，不乘 scale）。

torch / isaac 全部延迟 import（只在 SceneSession.launch 之后调用）。
"""

from __future__ import annotations

# 咖啡机拉手：实测 joint_5/link_5 才是绕世界 Z【水平摆动】的把手关节(joint_4/link_4 是绕 +Y 竖直摆动的
# 旋钮，不是用户要操作的那个)。所以把手 pose 必须挂在 link_5 上(在它的局部系里设 pose，跟着 link_5 走)。
COFFEE_LEVER_LINK = "link_5"


def build_handle_specs() -> list[dict]:
    """组装把手规格表（延迟 import config，post-launch 调用一次即可缓存）。

    每条：{name, asset, link, offset(局部，世界尺度), calibrated, functional}。
    """
    specs: list[dict] = []

    # 微波炉门
    try:
        from isaaclab_tasks.manager_based.manipulation.stack.config.franka.microwave_door_config import (
            DOOR_TARGETS,
        )

        for key, d in DOOR_TARGETS.items():
            name = "microwave_door" if key == "microwave" else f"microwave_{key}"
            specs.append({
                "name": name, "asset": d["asset_name"], "link": d["link_name"],
                "offset": tuple(float(v) for v in d["handle_offset"]),
                "calibrated": True, "functional": True,
            })
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[handles] WARNING: failed to load microwave handle config: {exc}", flush=True)

    # 柜子三抽屉
    try:
        from isaaclab_tasks.manager_based.manipulation.stack.config.franka.custom_drawer_config import (
            DRAWER_TARGETS,
        )

        for key, d in DRAWER_TARGETS.items():
            specs.append({
                "name": key, "asset": "cabinet", "link": d["link_name"],
                "offset": tuple(float(v) for v in d["handle_offset"]),
                "calibrated": True, "functional": bool(d.get("functional", True)),
            })
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[handles] WARNING: failed to load drawer handle config: {exc}", flush=True)

    # 右下角地面【白色 Sektion 橱柜】(env cfg 里作为 articulation 'sektion_cabinet' 接入)的两个抽屉把手。
    # link 用把手刚体 drawer_handle_top/bottom(随抽屉滑动)。默认抓取朝向由面板按"TCP +Z 指向抽屉里"算
    # (grasp_into=True)——旧柜的 open 轴不适用于 sektion，故不用它推朝向。用户在面板里再微调+保存。
    for key, link in (("sektion_top_drawer", "drawer_handle_top"),
                      ("sektion_bottom_drawer", "drawer_handle_bottom")):
        specs.append({
            "name": key, "asset": "sektion_cabinet", "link": link,
            "offset": (0.0, 0.0, 0.0), "calibrated": False, "functional": True,
            "grasp_into": True,
        })

    # 咖啡机拉手（偏移未标定 -> 先给 link 原点，calibrated=False）
    specs.append({
        "name": "coffee_lever", "asset": "coffee_machine", "link": COFFEE_LEVER_LINK,
        "offset": (0.0, 0.0, 0.0), "calibrated": False, "functional": True,
    })
    return specs


def _link_index(asset, link_name: str):
    names = list(getattr(asset.data, "body_names", []))
    idx = next((i for i, n in enumerate(names) if n == link_name), None)
    if idx is None:
        idx = next((i for i, n in enumerate(names) if link_name in n), None)
    return idx


def read_handles_in_base(scene, env_id: int = 0, specs: list[dict] | None = None) -> dict:
    """读所有把手在机器人基坐标系下的 pose。资产/连杆不存在则跳过。

    返回 {name: {asset, link, calibrated, functional, position[3], quat_wxyz[4],
                 position_world[3], quat_wxyz_world[4]}}。
    """
    import torch

    import isaaclab.utils.math as math_utils

    specs = specs if specs is not None else build_handle_specs()
    if "robot" not in scene.keys():
        return {}
    robot = scene["robot"]
    root_pos = robot.data.root_pos_w[env_id]
    root_quat = robot.data.root_quat_w[env_id]
    device = root_pos.device

    out: dict[str, dict] = {}
    for s in specs:
        asset_name = s["asset"]
        try:
            if asset_name not in scene.keys():
                continue
            asset = scene[asset_name]
        except Exception:
            continue
        idx = _link_index(asset, s["link"])
        if idx is None:
            continue
        link_pos = asset.data.body_pos_w[env_id, idx]
        link_quat = asset.data.body_quat_w[env_id, idx]
        offset = torch.tensor(s["offset"], dtype=torch.float32, device=device)
        hpos_w, hquat_w = math_utils.combine_frame_transforms(
            link_pos.unsqueeze(0), link_quat.unsqueeze(0), offset.unsqueeze(0)
        )
        pos_b, quat_b = math_utils.subtract_frame_transforms(
            root_pos.unsqueeze(0), root_quat.unsqueeze(0), hpos_w, hquat_w
        )
        out[s["name"]] = {
            "asset": asset_name,
            "link": s["link"],
            "calibrated": bool(s.get("calibrated", True)),
            "functional": bool(s.get("functional", True)),
            "position": [float(v) for v in pos_b[0].detach().cpu().tolist()],
            "quat_wxyz": [float(v) for v in quat_b[0].detach().cpu().tolist()],
            "position_world": [float(v) for v in hpos_w[0].detach().cpu().tolist()],
            "quat_wxyz_world": [float(v) for v in hquat_w[0].detach().cpu().tolist()],
        }
    return out
