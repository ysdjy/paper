# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Franka V1 Skill Lab — unified workspace.

A single scene contract (Franka Skill Scene V1, joint action) shared by:
  * skill_runtime           — joint-action state-machine skill tester
  * teleop_collection       — GELLO teleop + demo recording
  * pi05_training           — pi0.5 fine-tune pipeline (HTTP policy server)
  * perception_foundationpose — sim GT / FoundationPose 6D object pose

See README.md and PROJECT_MAP.md.
"""
