# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Teleop collection state machine + keyboard / scripted event sources. STATUS: ready.

The collection loop is event-driven. A *state machine* (``CollectionStateMachine``)
owns the episode lifecycle; *event sources* feed it discrete events:

  * ``ScriptedEventSource`` — emits events on a fixed step schedule. Hardware-free
    and deterministic, so the mock smoke test and headless collection use it.
  * ``CarbKeyboardSource`` (in :mod:`teleop_collection.entries`) — subscribes to
    omni keyboard events in the GUI and pushes them here. Lazy-imported so this
    module stays plain-python.

Episode lifecycle (states):

    IDLE -> PREPARE_EPISODE -> RECORDING <-> PAUSED
                                  |  \\-> SUCCESS_PENDING -> SAVED -> PREPARE_EPISODE
                                  |  \\-> FAILED_PENDING  -> SAVED/DISCARDED -> PREPARE_EPISODE
                                  \\---> DISCARDED -> PREPARE_EPISODE
    any -> RESETTING -> PREPARE_EPISODE
    any -> EXIT

The state machine NEVER touches Isaac; the driver maps states to behaviour
(record frame? hold? reset scene? flush HDF5? quit?).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class CollectionState(str, enum.Enum):
    IDLE = "IDLE"
    PREPARE_EPISODE = "PREPARE_EPISODE"
    RECORDING = "RECORDING"
    PAUSED = "PAUSED"
    SUCCESS_PENDING = "SUCCESS_PENDING"
    FAILED_PENDING = "FAILED_PENDING"
    DISCARDED = "DISCARDED"
    SAVED = "SAVED"
    RESETTING = "RESETTING"
    EXIT = "EXIT"


class CollectionEvent(str, enum.Enum):
    START = "START"            # begin recording the prepared episode
    TOGGLE_PAUSE = "TOGGLE_PAUSE"
    MARK_SUCCESS = "MARK_SUCCESS"
    MARK_FAILED = "MARK_FAILED"
    DISCARD = "DISCARD"
    RESET = "RESET"            # discard current + reset scene
    NEXT = "NEXT"              # advance to next prepared episode
    ESTOP = "ESTOP"            # freeze joint target (latch), stays in current state
    HOME = "HOME"             # request home / hold posture
    EXIT = "EXIT"
    SELECT_SKILL = "SELECT_SKILL"   # payload: skill index 1..6


# Keyboard key -> event (documented in README + 指令.txt). Keys are carb input
# names (single upper-case char / "SPACE"). 1..6 select the skill/task family.
COLLECTION_KEYMAP: dict[str, CollectionEvent] = {
    "SPACE": CollectionEvent.TOGGLE_PAUSE,
    "S": CollectionEvent.MARK_SUCCESS,
    "F": CollectionEvent.MARK_FAILED,
    "D": CollectionEvent.DISCARD,
    "R": CollectionEvent.RESET,
    "N": CollectionEvent.NEXT,
    "E": CollectionEvent.ESTOP,
    "H": CollectionEvent.HOME,
    "Q": CollectionEvent.EXIT,
    "ENTER": CollectionEvent.START,
}
# Number keys select skill family (1..6 -> skill index).
SKILL_KEYS = {str(i): i for i in range(1, 7)}


@dataclass
class CollectionStateMachine:
    """Episode-lifecycle state machine. Pure logic; no Isaac, no I/O.

    The driver calls :meth:`handle` for each event, reads :attr:`state` every
    step, and acts on the returned :class:`Transition` side-effect flags.
    """

    state: CollectionState = CollectionState.IDLE
    estopped: bool = False
    home_requested: bool = False
    selected_skill_index: int = 0          # 0 = unset; 1..6 maps to a skill family
    episodes_saved: int = 0
    episodes_discarded: int = 0
    _log: list[str] = field(default_factory=list)

    # --- queries -------------------------------------------------------------
    def is_recording(self) -> bool:
        return self.state == CollectionState.RECORDING and not self.estopped

    def is_following(self) -> bool:
        """Should the robot track the teleop device this step?

        True while RECORDING (drive + record) and while PREPARE_EPISODE (drive so
        the human can align the arm to a start pose before pressing START — gives
        immediate "the robot follows my device" feedback). PAUSED / e-stop / any
        pending/reset state holds the current posture instead.
        """
        return self.state in (CollectionState.RECORDING, CollectionState.PREPARE_EPISODE) and not self.estopped

    def is_terminal(self) -> bool:
        return self.state == CollectionState.EXIT

    # --- lifecycle helpers ---------------------------------------------------
    def prepare_episode(self) -> None:
        """Driver calls this once a fresh episode buffer is ready."""
        self.estopped = False
        self.home_requested = False
        self.state = CollectionState.PREPARE_EPISODE

    def handle(self, event: CollectionEvent, payload=None) -> "Transition":
        """Apply one event. Returns a Transition describing side effects."""
        tr = Transition(prev=self.state)

        if event == CollectionEvent.EXIT:
            self.state = CollectionState.EXIT
            tr.should_quit = True
            return self._done(tr)

        if event == CollectionEvent.ESTOP:
            # Latch the joint target wherever we are; do not change lifecycle.
            self.estopped = not self.estopped
            tr.estop_toggled = True
            return self._done(tr)

        if event == CollectionEvent.HOME:
            self.home_requested = True
            tr.home_requested = True
            return self._done(tr)

        if event == CollectionEvent.SELECT_SKILL:
            if payload in range(1, 7):
                self.selected_skill_index = int(payload)
                tr.skill_changed = True
            return self._done(tr)

        if event == CollectionEvent.START:
            if self.state in (CollectionState.IDLE, CollectionState.PREPARE_EPISODE, CollectionState.SAVED):
                self.state = CollectionState.RECORDING
                self.estopped = False
                tr.begin_recording = True
            return self._done(tr)

        if event == CollectionEvent.TOGGLE_PAUSE:
            if self.state == CollectionState.RECORDING:
                self.state = CollectionState.PAUSED
            elif self.state == CollectionState.PAUSED:
                self.state = CollectionState.RECORDING
            return self._done(tr)

        if event == CollectionEvent.MARK_SUCCESS:
            if self.state in (CollectionState.RECORDING, CollectionState.PAUSED):
                self.state = CollectionState.SUCCESS_PENDING
                tr.flush_success = True
                self.episodes_saved += 1
            return self._done(tr)

        if event == CollectionEvent.MARK_FAILED:
            if self.state in (CollectionState.RECORDING, CollectionState.PAUSED):
                self.state = CollectionState.FAILED_PENDING
                tr.flush_failed = True   # driver decides save-failed vs discard
            return self._done(tr)

        if event == CollectionEvent.DISCARD:
            if self.state in (CollectionState.RECORDING, CollectionState.PAUSED):
                self.state = CollectionState.DISCARDED
                tr.discard = True
                self.episodes_discarded += 1
            return self._done(tr)

        if event == CollectionEvent.RESET:
            tr.discard = self.state in (CollectionState.RECORDING, CollectionState.PAUSED)
            if tr.discard:
                self.episodes_discarded += 1
            self.state = CollectionState.RESETTING
            tr.reset_scene = True
            return self._done(tr)

        if event == CollectionEvent.NEXT:
            # Acknowledge a pending save/discard/reset and prepare the next episode.
            if self.state in (
                CollectionState.SUCCESS_PENDING,
                CollectionState.FAILED_PENDING,
                CollectionState.DISCARDED,
                CollectionState.SAVED,
                CollectionState.RESETTING,
                CollectionState.IDLE,
            ):
                self.state = CollectionState.PREPARE_EPISODE
                tr.prepare_next = True
            return self._done(tr)

        return self._done(tr)

    def _done(self, tr: "Transition") -> "Transition":
        tr.new = self.state
        self._log.append(f"{tr.prev.value} --> {tr.new.value}")
        return tr


@dataclass
class Transition:
    """Side-effect flags returned by ``CollectionStateMachine.handle``."""

    prev: CollectionState
    new: CollectionState = CollectionState.IDLE
    begin_recording: bool = False
    flush_success: bool = False
    flush_failed: bool = False
    discard: bool = False
    reset_scene: bool = False
    prepare_next: bool = False
    estop_toggled: bool = False
    home_requested: bool = False
    skill_changed: bool = False
    should_quit: bool = False


@dataclass
class ScriptedEventSource:
    """Deterministic, hardware-free event source for headless / mock collection.

    Emits a fixed sequence so a full collect run (record N steps -> success ->
    next) executes with no keyboard. ``episode_len`` is the recorded-step budget
    per episode; ``num_episodes`` episodes are auto-collected, then EXIT.

    The driver calls :meth:`poll(recorded_steps)` every sim step; it returns a
    list of events to feed the state machine this step (usually empty).
    """

    episode_len: int = 300
    num_episodes: int = 1
    mark_failed_episodes: tuple[int, ...] = ()   # 0-based episode indices to MARK_FAILED instead of success
    _episode: int = 0
    _started: bool = False
    _finishing: bool = False

    def poll(self, recorded_steps: int) -> list[tuple[CollectionEvent, object]]:
        events: list[tuple[CollectionEvent, object]] = []
        if self._episode >= self.num_episodes:
            return [(CollectionEvent.EXIT, None)]
        if not self._started:
            self._started = True
            events.append((CollectionEvent.START, None))
            return events
        if recorded_steps >= self.episode_len and not self._finishing:
            self._finishing = True
            if self._episode in self.mark_failed_episodes:
                events.append((CollectionEvent.MARK_FAILED, None))
            else:
                events.append((CollectionEvent.MARK_SUCCESS, None))
            return events
        if self._finishing:
            # acknowledge -> prepare next, then re-arm for the following episode
            self._episode += 1
            self._started = False
            self._finishing = False
            events.append((CollectionEvent.NEXT, None))
            return events
        return events
