#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""独立实时图像查看器（网页版，跨环境/可远程）。

订阅 ImagePublisher 的 ZMQ 总线，把各路命名画面(front_rgb/front_depth/wrist_rgb/wrist_depth/fp_*)
合成网格，通过 HTTP MJPEG 推给浏览器。**用浏览器看**，不依赖本地 GUI 库（env_isaaclab 的 cv2 是
headless，imshow 用不了，但 imdecode/imencode 可用）。以后 FoundationPose 往同一 addr 发处理图，
网页自动多一格。

Run（任意有 zmq+cv2 的 env，如 env_isaaclab）:
    python projects/franka_v1_skill_lab/scene_interface/image_viewer.py --connect tcp://localhost:5557 --port 8088
然后浏览器打开 http://localhost:8088  （远程则 http://<机器IP>:8088）
"""

from __future__ import annotations

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_frames: dict = {}        # topic -> JPEG bytes（publisher 已编码好）
_lock = threading.Lock()


def _sub_loop(connect: str):
    import zmq

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(connect)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    print(f"[image_viewer] SUB connected to {connect}", flush=True)
    while True:
        try:
            topic, buf = sock.recv_multipart()
            with _lock:
                _frames[topic.decode("utf-8", "ignore")] = bytes(buf)
        except Exception as exc:  # pragma: no cover
            print(f"[image_viewer] sub error: {exc}", flush=True)
            time.sleep(0.2)


def _compose_grid(tile: int) -> bytes | None:
    import cv2
    import numpy as np

    with _lock:
        items = sorted(_frames.items())
    if not items:
        return None
    imgs = []
    for name, jpg in items:
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        s = tile / max(1, max(h, w))
        r = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
        canvas = np.zeros((tile, tile, 3), dtype="uint8")
        canvas[:r.shape[0], :r.shape[1]] = r
        cv2.putText(canvas, name, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        imgs.append(canvas)
    if not imgs:
        return None
    cols = 2 if len(imgs) > 1 else 1
    rows = (len(imgs) + cols - 1) // cols
    while len(imgs) < rows * cols:
        imgs.append(np.zeros((tile, tile, 3), dtype="uint8"))
    grid = np.vstack([np.hstack(imgs[r * cols:(r + 1) * cols]) for r in range(rows)])
    ok, buf = cv2.imencode(".jpg", grid, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes() if ok else None


_PAGE = (b"<html><head><title>scene image viewer</title></head>"
         b"<body style='margin:0;background:#111;text-align:center'>"
         b"<img src='/mjpeg' style='max-width:100%;height:auto'></body></html>")


def _make_handler(tile: int):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/mjpeg":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        jpg = _compose_grid(tile)
                        if jpg is not None:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                             b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n")
                            self.wfile.write(jpg)
                            self.wfile.write(b"\r\n")
                        time.sleep(1.0 / 15.0)
                except (BrokenPipeError, ConnectionResetError):
                    return
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(_PAGE)

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description="Web (HTTP MJPEG) live image viewer for the ZMQ image bus.")
    ap.add_argument("--connect", default="tcp://localhost:5557", help="ZMQ PUB addr to subscribe.")
    ap.add_argument("--port", type=int, default=8088, help="HTTP port for the browser viewer.")
    ap.add_argument("--tile", type=int, default=420)
    args = ap.parse_args()

    threading.Thread(target=_sub_loop, args=(args.connect,), daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), _make_handler(args.tile))
    print(f"[image_viewer] open http://localhost:{args.port}  (subscribing {args.connect})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
