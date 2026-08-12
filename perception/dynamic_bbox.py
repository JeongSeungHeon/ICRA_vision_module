"""Thread-safe bootstrap-to-dynamic FastSAM bbox state."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Sequence


BOOTSTRAP_FIXED = "BOOTSTRAP_FIXED"
WAITING_DYNAMIC = "WAITING_DYNAMIC"
DYNAMIC_TRACKING = "DYNAMIC_TRACKING"


@dataclass(frozen=True)
class DynamicBBoxRecord:
    bbox_xyxy: tuple[float, float, float, float]
    task_id: int
    frame_seq: int
    capture_time_s: float
    received_monotonic_s: float
    hand_side: str
    hand_score: float
    object_score: float
    relation_score: float
    contact_state: str


@dataclass(frozen=True)
class DynamicBBoxDebug:
    mode: str
    bbox_xyxy: tuple[float, float, float, float] | None
    age_s: float | None
    frame_seq: int | None
    hand_side: str
    hand_score: float | None
    object_score: float | None
    relation_score: float | None
    reason: str


class DynamicFastSAMBBoxState:
    """Own accepted HOI-detector results and bounded per-camera dropout holds."""

    def __init__(
        self,
        *,
        hold_timeout_s: float = 0.5,
        result_age_timeout_s: float = 1.0,
    ) -> None:
        self.hold_timeout_s = max(float(hold_timeout_s), 0.0)
        self.result_age_timeout_s = max(float(result_age_timeout_s), 0.0)
        self._lock = threading.Lock()
        self._active = False
        self._task_id = 0
        self._records: dict[int, DynamicBBoxRecord] = {}
        self._last_frame_seq = {0: -1, 1: -1}
        self._last_reason = {0: "bootstrap_fixed", 1: "bootstrap_fixed"}

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    @property
    def task_id(self) -> int:
        with self._lock:
            return self._task_id

    def activate(self, task_id: int) -> bool:
        with self._lock:
            if self._active and self._task_id == int(task_id):
                return False
            self._active = True
            self._task_id = int(task_id)
            self._records.clear()
            self._last_frame_seq = {0: -1, 1: -1}
            self._last_reason = {0: "waiting_first_bbox", 1: "waiting_first_bbox"}
            return True

    def reset(self, task_id: int) -> None:
        with self._lock:
            self._active = False
            self._task_id = int(task_id)
            self._records.clear()
            self._last_frame_seq = {0: -1, 1: -1}
            self._last_reason = {0: "bootstrap_fixed", 1: "bootstrap_fixed"}

    def accept(
        self,
        *,
        camera_id: int,
        task_id: int,
        frame_seq: int,
        valid: bool,
        bbox_xyxy: Sequence[float],
        capture_time_s: float,
        now_ros_s: float,
        received_monotonic_s: float | None = None,
        hand_side: str = "",
        hand_score: float = 0.0,
        object_score: float = 0.0,
        relation_score: float = 0.0,
        contact_state: str = "",
        reason: str = "",
    ) -> bool:
        camera_id = int(camera_id)
        if camera_id not in (0, 1):
            return False
        received = (
            time.monotonic()
            if received_monotonic_s is None
            else float(received_monotonic_s)
        )
        with self._lock:
            if not self._active or int(task_id) != self._task_id:
                return False
            if int(frame_seq) <= self._last_frame_seq[camera_id]:
                return False
            self._last_frame_seq[camera_id] = int(frame_seq)
            if float(now_ros_s) - float(capture_time_s) > self.result_age_timeout_s:
                self._last_reason[camera_id] = "result_too_old"
                return False
            if not valid:
                self._last_reason[camera_id] = str(reason or "detector_invalid")
                return True
            values = tuple(float(value) for value in bbox_xyxy)
            if len(values) != 4 or values[2] <= values[0] or values[3] <= values[1]:
                self._last_reason[camera_id] = "invalid_bbox"
                return False
            self._records[camera_id] = DynamicBBoxRecord(
                bbox_xyxy=values,
                task_id=int(task_id),
                frame_seq=int(frame_seq),
                capture_time_s=float(capture_time_s),
                received_monotonic_s=received,
                hand_side=str(hand_side),
                hand_score=float(hand_score),
                object_score=float(object_score),
                relation_score=float(relation_score),
                contact_state=str(contact_state),
            )
            self._last_reason[camera_id] = "ok"
            return True

    def bbox_for_camera(
        self,
        camera_id: int,
        *,
        now_monotonic_s: float | None = None,
    ) -> tuple[float, float, float, float] | None:
        now = time.monotonic() if now_monotonic_s is None else float(now_monotonic_s)
        with self._lock:
            if not self._active:
                return None
            record = self._records.get(int(camera_id))
            if record is None:
                return None
            if now - record.received_monotonic_s > self.hold_timeout_s:
                self._last_reason[int(camera_id)] = "bbox_hold_expired"
                return None
            return record.bbox_xyxy

    def mode(self, *, now_monotonic_s: float | None = None) -> str:
        now = time.monotonic() if now_monotonic_s is None else float(now_monotonic_s)
        with self._lock:
            if not self._active:
                return BOOTSTRAP_FIXED
            fresh = any(
                now - record.received_monotonic_s <= self.hold_timeout_s
                for record in self._records.values()
            )
            return DYNAMIC_TRACKING if fresh else WAITING_DYNAMIC

    def debug(
        self,
        camera_id: int,
        *,
        now_monotonic_s: float | None = None,
    ) -> DynamicBBoxDebug:
        now = time.monotonic() if now_monotonic_s is None else float(now_monotonic_s)
        camera_id = int(camera_id)
        with self._lock:
            mode = BOOTSTRAP_FIXED
            if self._active:
                mode = (
                    DYNAMIC_TRACKING
                    if any(
                        now - record.received_monotonic_s <= self.hold_timeout_s
                        for record in self._records.values()
                    )
                    else WAITING_DYNAMIC
                )
            record = self._records.get(camera_id)
            age = None if record is None else max(now - record.received_monotonic_s, 0.0)
            fresh = record is not None and age <= self.hold_timeout_s
            return DynamicBBoxDebug(
                mode=mode,
                bbox_xyxy=record.bbox_xyxy if fresh else None,
                age_s=age,
                frame_seq=None if record is None else record.frame_seq,
                hand_side="" if record is None else record.hand_side,
                hand_score=None if record is None else record.hand_score,
                object_score=None if record is None else record.object_score,
                relation_score=None if record is None else record.relation_score,
                reason=self._last_reason[camera_id],
            )


__all__ = [
    "BOOTSTRAP_FIXED",
    "DYNAMIC_TRACKING",
    "WAITING_DYNAMIC",
    "DynamicBBoxDebug",
    "DynamicBBoxRecord",
    "DynamicFastSAMBBoxState",
]
