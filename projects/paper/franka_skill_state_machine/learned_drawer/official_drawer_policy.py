"""Loader / inference wrapper for Isaac Lab's official Franka open-drawer PPO policy.

The wrapper loads a TorchScript ``policy.pt`` exported from the trained checkpoint (see
``export_official_drawer_policy.py``). It performs inference only — it never trains, never downloads
large files at runtime, and raises a clear error if no checkpoint path is provided / found.
"""

from __future__ import annotations

import os

import torch


class OfficialDrawerPolicyWrapper:
    def __init__(self, policy_path: str | os.PathLike | None, device: str | torch.device = "cuda:0"):
        if not policy_path:
            raise ValueError(
                "drawer_backend='official_joint_policy' requires --drawer_policy_path. "
                "Export it first with:\n"
                "  ./isaaclab.sh -p scripts/environments/state_machine/export_official_drawer_policy.py "
                "--task Isaac-Open-Drawer-Franka-Play-v0 --use_pretrained_checkpoint "
                "--output_path logs/policies/official_open_drawer_policy.pt --headless"
            )
        policy_path = os.fspath(policy_path)
        if not os.path.isfile(policy_path):
            raise FileNotFoundError(
                f"drawer policy not found at '{policy_path}'. Export it first with "
                "export_official_drawer_policy.py (see --help)."
            )
        self.device = torch.device(device)
        self.policy_path = policy_path
        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()
        self.last_obs_shape: tuple[int, ...] | None = None
        self.last_action_shape: tuple[int, ...] | None = None
        print(f"[OfficialDrawerPolicy] loaded torchscript policy from {policy_path} on {self.device}", flush=True)

    @torch.inference_mode()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(self.device, dtype=torch.float32)
        self.last_obs_shape = tuple(obs.shape)
        action = self.policy(obs)
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, device=self.device)
        self.last_action_shape = tuple(action.shape)
        return action

    def reset(self, dones=None):
        if hasattr(self.policy, "reset"):
            try:
                self.policy.reset(dones)
            except Exception:
                pass
