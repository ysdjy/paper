# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""ZMQ 图像发布器：把相机帧(RGB / colorized-depth)发到一个 PUB 总线，供独立查看器(image_viewer.py)显示。

约定：multipart [topic(bytes), jpg_bytes]。topic = 流名(front_rgb / front_depth / wrist_rgb /
wrist_depth / fp_* ...)；payload = JPEG(BGR)。以后 FoundationPose 模块用同一 publisher、同一 addr
发处理后的图(新 topic)，查看器自动多显示一格。

顶层只 stdlib；zmq/cv2/numpy 延迟 import（任意有 zmq+cv2 的 env 可用，不破坏包的 isaac-free 导入）。
"""

from __future__ import annotations

import time


def colorize_depth(depth, min_m: float = 0.2, max_m: float = 3.0):
    """深度(米) -> JET 上色 BGR(uint8)，便于查看。"""
    import cv2
    import numpy as np

    d = np.asarray(depth).astype("float32")
    if d.ndim == 3:
        d = d[..., 0]
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.clip((d - min_m) / max(1e-6, (max_m - min_m)), 0.0, 1.0)
    return cv2.applyColorMap((norm * 255).astype("uint8"), cv2.COLORMAP_JET)


class ImagePublisher:
    """ZMQ PUB 图像发布器（节流到 ~stream_hz）。"""

    def __init__(self, addr: str = "tcp://*:5557", jpeg_quality: int = 80, stream_hz: float = 15.0):
        import zmq

        self._zmq = zmq
        self._ctx = zmq.Context.instance()
        self.sock = self._ctx.socket(zmq.PUB)
        self.sock.bind(addr)
        self.jpeg_quality = int(jpeg_quality)
        self._min_dt = 1.0 / max(1.0, stream_hz)
        self._last = 0.0
        print(f"[image_publisher] PUB bound to {addr} (~{stream_hz:.0f} Hz JPEG)", flush=True)

    def due(self) -> bool:
        """是否到了发送时机（节流）。"""
        now = time.monotonic()
        if now - self._last >= self._min_dt:
            self._last = now
            return True
        return False

    def send_bgr(self, topic: str, bgr) -> None:
        import cv2
        import numpy as np

        if bgr is None:
            return
        ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(bgr),
                               [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if ok:
            self.sock.send_multipart([topic.encode("utf-8"), buf.tobytes()])

    def send_rgb(self, topic: str, rgb) -> None:
        import numpy as np

        if rgb is None:
            return
        arr = np.asarray(rgb)[..., :3]
        self.send_bgr(topic, arr[..., ::-1])   # RGB -> BGR

    def send_depth(self, topic: str, depth, min_m: float = 0.2, max_m: float = 3.0) -> None:
        if depth is None:
            return
        self.send_bgr(topic, colorize_depth(depth, min_m, max_m))

    def close(self) -> None:
        try:
            self.sock.close(0)
        except Exception:
            pass
