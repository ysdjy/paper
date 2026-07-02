# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""V1 sensor modules: the two simulated wrist cameras (FoundationPose RGB-D + VLA RGB)."""

from .d435 import attach_d435  # noqa: F401  (back-compat)
from .d435 import attach_wrist_cameras  # noqa: F401
from .d435 import d435_config  # noqa: F401
