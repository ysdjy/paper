#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""V1 joint pi0.5 policy server (runs in .venv_openpi). STATUS: ready.

A minimal HTTP server that loads the fine-tuned pi0.5 checkpoint and serves
8-D JOINT actions (7 joint targets + 1 gripper) — unlike the legacy EE-delta
server. It keeps the full action dim (patches the LIBERO output slice from 7 to
8 so the gripper survives) and exposes the model's native observation
(`observation/image` + `observation/wrist_image` + `observation/state` joint
layout + prompt).

The IsaacLab side (env_isaaclab) talks to it over HTTP with stdlib only.

Endpoints:
  GET  /health -> {"status":"ok","backend":"pi05_joint","action_dim":8,...}
  POST /infer  <- {"state":[8], "image":<b64 png>, "wrist_image":<b64 png>, "prompt":str}
               -> {"joint_targets":[7], "gripper":float(0..1), "chunk":[[8]*horizon], "latency_ms":..}

Run (OpenPI venv):
    pi05_isaacsim_baseline/.venv_openpi/bin/python \
        projects/franka_v1_skill_lab/pi05_training/v1_joint_policy_server.py \
        --ckpt <.../3000> --port 8010
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

ACTION_DIM = 8
DEFAULT_PROMPT = "stack the blue cube on top of the red cube"

_POLICY = None
_STATE = {"requests": 0, "t0": 0.0}


def _decode_image(b64: str) -> np.ndarray:
    from PIL import Image

    raw = base64.b64decode(b64)
    return np.asarray(Image.open(io.BytesIO(raw)))[..., :3].astype(np.uint8)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence per-request stderr noise
        pass

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", ""):
            self._send(200, {
                "status": "ok", "backend": "pi05_joint", "action_dim": ACTION_DIM,
                "requests": _STATE["requests"], "uptime_s": round(time.time() - _STATE["t0"], 1),
            })
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/infer":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n).decode("utf-8"))
            obs = {
                "observation/state": np.asarray(body["state"], dtype=np.float32),
                "observation/image": _decode_image(body["image"]),
                "observation/wrist_image": _decode_image(body["wrist_image"]),
                "prompt": body.get("prompt", DEFAULT_PROMPT),
            }
            t = time.time()
            chunk = np.asarray(_POLICY.infer(obs)["actions"])[:, :ACTION_DIM]  # (horizon, 8)
            latency = (time.time() - t) * 1000.0
            _STATE["requests"] += 1
            first = chunk[0]
            self._send(200, {
                "joint_targets": first[:7].tolist(),
                "gripper": float(first[7]),
                "chunk": chunk.tolist(),
                "latency_ms": round(latency, 1),
                "backend": "pi05_joint",
            })
        except Exception as exc:  # pragma: no cover - surface to client
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pi05_franka_v1_stack")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    args = ap.parse_args()

    global _POLICY
    import openpi.policies.libero_policy as _libero
    from openpi.training import config as _config
    from openpi.policies import policy_config

    # keep all 8 dims (7 joint + gripper); stock LiberoOutputs slices to 7.
    _libero.LiberoOutputs.__call__ = lambda self, data: {"actions": np.asarray(data["actions"])[:, :ACTION_DIM]}

    print(f"[v1_server] loading {args.config} @ {args.ckpt} ...", flush=True)
    cfg = _config.get_config(args.config)
    _POLICY = policy_config.create_trained_policy(cfg, args.ckpt)
    _STATE["t0"] = time.time()
    print("[v1_server] policy loaded.", flush=True)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[v1_server] serving pi05_joint on http://{args.host}:{args.port} (action_dim={ACTION_DIM})", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
