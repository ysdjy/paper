# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""RemoteQwenPerceiver -- 让 xiaoyu 记忆模块复用已起的 :5601 Qwen server,而不在本进程再加载一份 3B。

记忆模块的 VisionPerceptionAdapter 只要求 perceiver 有
    recognize_batch(image_paths, candidate_labels=None, save_memory=True) -> List[RecognitionResult]
本类用这个签名包住 HTTP /recognize:读图 -> b64 -> POST -> 把返回 dict 还原成 perception_vlm 的
RecognitionResult。这样 Qwen 只在 qwen3vl 进程(:5601)加载一次,记忆进程零额外显存。

用法(配合 PerceptionMemoryPipeline.create_from_existing):
    perceiver = RemoteQwenPerceiver(addr="http://127.0.0.1:5601")
    memory = EmbodiedManipulationMemorySystem(config=MemorySystemConfig(store_dir=...))
    pipe = PerceptionMemoryPipeline.create_from_existing(perceiver, memory)
"""

from __future__ import annotations

import base64
import json
import urllib.request
from pathlib import Path
from typing import List, Optional, Sequence

# RecognitionResult 是纯 dataclass(perception_vlm 顶层 transformers/qwen 被 try 包住,缺也能 import)。
from perception_vlm import RecognitionResult


class RemoteQwenPerceiver:
    def __init__(self, addr: str = "http://127.0.0.1:5601", timeout: float = 300.0):
        self.addr = addr.rstrip("/")
        self.timeout = timeout

    def recognize_batch(
        self,
        image_paths: Sequence[str],
        candidate_labels: Optional[Sequence[str]] = None,
        save_memory: bool = True,
    ) -> List[RecognitionResult]:
        images = []
        for p in image_paths:
            raw = Path(p).read_bytes()
            images.append({"name": Path(p).stem, "b64": base64.b64encode(raw).decode("ascii")})
        payload = {
            "images": images,
            "candidate_labels": list(candidate_labels) if candidate_labels else [],
            "save_memory": bool(save_memory),
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.addr + "/recognize", data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            resp = json.loads(r.read().decode("utf-8"))
        if not resp.get("ok"):
            raise RuntimeError(f"qwen server error: {resp.get('error')}")

        out: List[RecognitionResult] = []
        for path, rr in zip(image_paths, resp.get("results", [])):
            out.append(RecognitionResult(
                image_path=str(path),
                primary_label=str(rr.get("primary_label", "unknown")),
                confidence=float(rr.get("confidence", 0.0)),
                objects=rr.get("objects", []) or [],
                attributes=rr.get("attributes", {}) or {},
                scene=str(rr.get("scene", "")),
                reasoning=str(rr.get("reasoning", "")),
                uncertainty=str(rr.get("uncertainty", "")),
                raw_response="",
            ))
        return out
