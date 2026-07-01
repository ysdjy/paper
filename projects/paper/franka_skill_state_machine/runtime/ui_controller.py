"""Thread-safe command buffer for omni.ui callbacks."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from runtime.skill_request import SkillRequest
from runtime.skill_types import SkillType


@dataclass
class PendingCommand:
    command: str
    request: SkillRequest | None = None


class UIController:
    def __init__(self):
        self._lock = threading.Lock()
        self._pending: PendingCommand | None = None
        self.selected_skill = SkillType.GRASP
        self.selected_target = "cube_2"

    def request_start(self, skill_type: SkillType | None = None, target: str | None = None):
        skill = skill_type or self.selected_skill
        source = target or self.selected_target
        request = SkillRequest(
            request_id=f"{skill.value}_{source}_{time.time_ns()}",
            skill_type=skill,
            source_object=source if skill == SkillType.GRASP else None,
            destination_object="cabinet" if skill in (SkillType.OPEN_DRAWER, SkillType.CLOSE_DRAWER) else None,
        )
        with self._lock:
            self._pending = PendingCommand("start", request)

    def queue_request(self, request: SkillRequest):
        with self._lock:
            self._pending = PendingCommand("start", request)

    def request_stop(self):
        with self._lock:
            self._pending = PendingCommand("stop")

    def request_resume(self):
        with self._lock:
            self._pending = PendingCommand("resume")

    def request_reset(self):
        with self._lock:
            self._pending = PendingCommand("reset")

    def pop(self) -> PendingCommand | None:
        with self._lock:
            command = self._pending
            self._pending = None
            return command
