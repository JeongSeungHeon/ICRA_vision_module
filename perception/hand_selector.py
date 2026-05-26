"""Object-distance hand selection with conservative handover locking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from system.shared_state import HandCandidateState, HandState, ObjectState, SelectedHandState, Vec3

DEFAULT_CONFIG_PATH = Path("configs/handover.yaml")
KNOWN_HANDEDNESS = {"left", "right"}


@dataclass
class HandSelectorDebug:
    """최근 selection frame에서 왜 특정 hand가 선택/유지/거절됐는지 남기는 디버그 스냅샷."""

    # 현재 frame에서 후보로 인정된 hand들의 camera/handedness 목록이다.
    candidate_cameras: tuple[int, ...]
    candidate_handedness: tuple[str, ...]
    # 카메라별 후보가 선택 pool에 못 들어온 이유다. 빈 문자열이면 후보가 하나 이상 있었다는 뜻이다.
    cam0_reject_reason: str
    cam1_reject_reason: str
    # 이번 process_states()가 최종 반환한 selected hand와 내부 상태 machine의 hand다.
    chosen_camera: int | None
    current_camera: int | None
    # switch 후보가 hysteresis를 채우는 중이면 pending_*에 기록된다.
    pending_camera: int | None
    pending_frames: int
    hysteresis_frames: int
    # lock_*은 task 중 active hand를 보수적으로 유지하는 내부 선택 상태를 관찰하기 위한 값이다.
    locked_camera: int | None
    lock_stable_frames: int
    lock_after_stable_frames: int
    # selection_reason은 downstream 로그에서 가장 먼저 봐야 하는 분기 결과다.
    selection_reason: str
    chosen_candidate_id: str | None = None
    current_candidate_id: str | None = None
    pending_candidate_id: str | None = None
    object_center_base: Vec3 | None = None
    chosen_distance_m: float | None = None
    current_distance_m: float | None = None
    duplicate_drop_reasons: tuple[str, ...] = ()


class HandSelector:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        legacy_cfg = self.config.get("perception", {}).get("hand", {}).get("handedness_selection", {})
        selection_cfg = self.config.get("perception", {}).get("hand_selection", {})

        # 새 active hand로 진입하려면 palm center가 object center에서 이 거리 안에 있어야 한다.
        self.active_hand_distance_threshold_m = float(
            selection_cfg.get("active_hand_distance_threshold_m", 0.15)
        )
        # 현재 hand보다 challenger가 이 거리 이상 더 가까울 때만 switch 후보로 인정한다.
        self.switch_margin_m = float(selection_cfg.get("switch_margin_m", 0.05))
        # challenger가 연속으로 조건을 만족해야 하는 frame 수다. 손/카메라가 튀는 것을 막는다.
        self.switch_confirm_frames = max(
            int(selection_cfg.get("switch_confirm_frames", legacy_cfg.get("hysteresis_frames", 5))),
            1,
        )
        # 현재 hand가 detection에서 빠져도 이 시간 안이면 마지막 hand를 계속 반환한다.
        self.lost_timeout_s = max(float(selection_cfg.get("lost_timeout_s", 0.5)), 0.0)
        self.lock_active_hand_during_task = bool(selection_cfg.get("lock_active_hand_during_task", True))
        # true이면 active hand가 실제로 교체되는 순간 교체 기준과 거리 정보를 콘솔에 남긴다.
        self.log_active_hand_switches = bool(selection_cfg.get("log_active_hand_switches", False))
        self.candidate_identity_mode = str(selection_cfg.get("candidate_identity_mode", "handedness")).strip().lower()
        if self.candidate_identity_mode not in {"handedness", "index"}:
            self.candidate_identity_mode = "handedness"

        # _current_*는 fusion/grasp에 실제로 넘길 active hand다.
        self._current_candidate_id: str | None = None
        self._current_candidate: HandCandidateState | None = None
        # _pending_*는 switch_confirm_frames를 채우는 중인 challenger hand다.
        self._pending_candidate_id: str | None = None
        self._pending_frames = 0
        # _locked_*는 디버깅용 lock 상태다. 현재 구현은 선택 시 즉시 current를 lock 관찰값으로 기록한다.
        self._locked_candidate_id: str | None = None
        self._lock_stable_frames = 0
        # last_debug를 overlay/profile에서 읽으면 이번 frame의 선택 이유를 추적할 수 있다.
        self.last_debug: HandSelectorDebug | None = None

    @classmethod
    def from_config(cls, config_path: str | Path = DEFAULT_CONFIG_PATH) -> "HandSelector":
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        return cls(config=config)

    def reset(self) -> None:
        self._current_candidate_id = None
        self._current_candidate = None
        self._pending_candidate_id = None
        self._pending_frames = 0
        self._locked_candidate_id = None
        self._lock_stable_frames = 0
        self.last_debug = None

    def process_states(
        self,
        hand_cam0: HandState,
        hand_cam1: HandState,
        *,
        object_center_base: Vec3 | None = None,
    ) -> SelectedHandState:
        """cam0/cam1 hand 후보 중 object에 가장 가까운 active hand 하나를 고른다."""

        # 1) 카메라별 HandState 안의 hand_candidates를 flatten하고 invalid 후보는 제외한다.
        candidates, reject_reasons, duplicate_drop_reasons = self._collect_candidates(
            hand_cam0,
            hand_cam1,
            object_center_base=object_center_base,
        )
        # 2) handover에 참여하는 손은 object에 가까워야 하므로 object center와의 3D 거리를 계산한다.
        distances = self._compute_distances(candidates, object_center_base)
        # 3) dropout hold 판단에는 wall-clock 대신 입력 state들의 최신 timestamp를 사용한다.
        now_timestamp = self._resolve_now_timestamp(hand_cam0, hand_cam1, candidates)
        selection_reason = "no_valid_candidate"

        if object_center_base is None:
            # object가 없으면 새 hand를 고를 기준도 없다. 현재 hand가 살아 있으면 유지하고, 아니면 timeout만큼 hold한다.
            selected, selection_reason = self._hold_or_invalidate(
                hand_cam0,
                hand_cam1,
                candidates,
                now_timestamp,
                "no_object_center",
            )
            self._set_debug(
                candidates,
                reject_reasons,
                selected,
                object_center_base,
                distances,
                selection_reason,
                duplicate_drop_reasons,
            )
            return selected

        # object 주변 threshold 안으로 들어온 hand만 신규 선택/switch 대상으로 삼는다.
        threshold_eligible_ids = [
            candidate_id
            for candidate_id, distance in distances.items()
            if distance <= self.active_hand_distance_threshold_m
        ]
        eligible_ids = self._prefer_known_handedness_candidates(candidates, threshold_eligible_ids)
        best_candidate_id = min(eligible_ids, key=lambda candidate_id: distances[candidate_id], default=None)

        if self._current_candidate_id is None:
            # 아직 active hand가 없으면 threshold 안의 가장 가까운 후보를 바로 선택한다.
            if best_candidate_id is None:
                selected = self._build_invalid_state(hand_cam0, hand_cam1)
                selection_reason = "no_candidate_within_threshold"
            else:
                self._select_candidate(candidates[best_candidate_id])
                selected = self._build_selected_state(self._current_candidate)
                selection_reason = "select_within_object_threshold"
            self._set_debug(
                candidates,
                reject_reasons,
                selected,
                object_center_base,
                distances,
                selection_reason,
                duplicate_drop_reasons,
            )
            return selected

        current_candidate = candidates.get(self._current_candidate_id)
        if current_candidate is None:
            # 기존 active hand가 이번 frame에서 사라졌다. lost_timeout_s 안이면 마지막 값을 유지한다.
            selected, selection_reason = self._handle_current_lost(
                hand_cam0,
                hand_cam1,
                candidates,
                now_timestamp,
                best_candidate_id,
                object_center_base=object_center_base,
                distances=distances,
            )
            self._set_debug(
                candidates,
                reject_reasons,
                selected,
                object_center_base,
                distances,
                selection_reason,
                duplicate_drop_reasons,
            )
            return selected

        # 현재 active hand가 여전히 보이면 최신 후보 값으로 갱신한 뒤 switch 필요 여부를 본다.
        self._current_candidate = current_candidate
        current_distance = distances.get(self._current_candidate_id)
        challenger_id = self._find_switch_challenger(best_candidate_id, current_distance, distances)
        if challenger_id is not None:
            # challenger가 충분히 더 가깝더라도 즉시 바꾸지 않고 연속 frame 수를 쌓는다.
            if self._pending_candidate_id == challenger_id:
                self._pending_frames += 1
            else:
                self._pending_candidate_id = challenger_id
                self._pending_frames = 1
            selection_reason = "pending_switch_closer_to_object"
            if self._pending_frames >= self.switch_confirm_frames:
                # hysteresis를 채우면 active hand를 challenger로 교체한다.
                previous_candidate = self._current_candidate
                next_candidate = candidates[challenger_id]
                self._log_active_hand_switch(
                    reason="switch_after_confirmed_closer_to_object",
                    previous_candidate=previous_candidate,
                    next_candidate=next_candidate,
                    object_center_base=object_center_base,
                    distances=distances,
                    pending_frames=self._pending_frames,
                    best_candidate_id=best_candidate_id,
                )
                self._select_candidate(next_candidate)
                selection_reason = "switch_after_confirmed_closer_to_object"
        else:
            # 충분히 더 가까운 challenger가 없으면 current hand를 계속 유지한다.
            self._pending_candidate_id = None
            self._pending_frames = 0
            selection_reason = "keep_locked_active_hand"

        selected = self._build_selected_state(self._current_candidate)
        self._set_debug(
            candidates,
            reject_reasons,
            selected,
            object_center_base,
            distances,
            selection_reason,
            duplicate_drop_reasons,
        )
        return selected

    def _handle_current_lost(
        self,
        hand_cam0: HandState,
        hand_cam1: HandState,
        candidates: dict[str, HandCandidateState],
        now_timestamp: float,
        best_candidate_id: str | None,
        *,
        object_center_base: Vec3 | None,
        distances: dict[str, float],
    ) -> tuple[SelectedHandState, str]:
        """현재 active hand가 후보 pool에서 사라졌을 때 hold, replacement, invalid 중 하나를 결정한다."""

        if self._can_hold_current(now_timestamp):
            self._pending_candidate_id = None
            self._pending_frames = 0
            return self._build_selected_state(self._current_candidate), "hold_active_hand_lost_timeout"

        previous_candidate = self._current_candidate
        self._clear_current()
        if best_candidate_id is not None and best_candidate_id in candidates:
            next_candidate = candidates[best_candidate_id]
            self._log_active_hand_switch(
                reason="select_after_lost_timeout",
                previous_candidate=previous_candidate,
                next_candidate=next_candidate,
                object_center_base=object_center_base,
                distances=distances,
                pending_frames=0,
                best_candidate_id=best_candidate_id,
                lost_timeout_s=self.lost_timeout_s,
            )
            self._select_candidate(next_candidate)
            return self._build_selected_state(self._current_candidate), "select_after_lost_timeout"
        return self._build_invalid_state(hand_cam0, hand_cam1), "active_hand_lost_timeout_no_replacement"

    def _hold_or_invalidate(
        self,
        hand_cam0: HandState,
        hand_cam1: HandState,
        candidates: dict[str, HandCandidateState],
        now_timestamp: float,
        reason_prefix: str,
    ) -> tuple[SelectedHandState, str]:
        """object center처럼 선택 기준이 없을 때 기존 active hand를 유지할 수 있는지 판단한다."""

        if self._current_candidate_id in candidates:
            self._current_candidate = candidates[self._current_candidate_id]
            return self._build_selected_state(self._current_candidate), f"{reason_prefix}_keep_current"
        if self._can_hold_current(now_timestamp):
            return self._build_selected_state(self._current_candidate), f"{reason_prefix}_hold_current"
        self._clear_current()
        return self._build_invalid_state(hand_cam0, hand_cam1), reason_prefix

    def _find_switch_challenger(
        self,
        best_candidate_id: str | None,
        current_distance: float | None,
        distances: dict[str, float],
    ) -> str | None:
        """가장 가까운 후보가 current보다 switch_margin_m 이상 가까울 때만 challenger로 반환한다."""

        if best_candidate_id is None or best_candidate_id == self._current_candidate_id:
            return None
        if current_distance is None:
            return best_candidate_id
        challenger_distance = distances.get(best_candidate_id)
        if challenger_distance is None:
            return None
        if challenger_distance <= current_distance - self.switch_margin_m:
            return best_candidate_id
        return None

    def _select_candidate(self, candidate: HandCandidateState) -> None:
        """candidate를 active hand로 확정하고 pending switch 상태를 초기화한다."""

        self._current_candidate_id = self._candidate_key(candidate)
        self._current_candidate = candidate
        self._pending_candidate_id = None
        self._pending_frames = 0
        if self.lock_active_hand_during_task:
            self._locked_candidate_id = self._current_candidate_id
            self._lock_stable_frames += 1

    def _clear_current(self) -> None:
        """active/pending/lock 상태를 모두 비워 다음 frame에서 새 hand를 다시 선택하게 한다."""

        self._current_candidate_id = None
        self._current_candidate = None
        self._pending_candidate_id = None
        self._pending_frames = 0
        self._locked_candidate_id = None
        self._lock_stable_frames = 0

    def _can_hold_current(self, now_timestamp: float) -> bool:
        """마지막 active hand timestamp가 lost_timeout_s 안이면 dropout hold를 허용한다."""

        if self._current_candidate is None:
            return False
        age_s = max(float(now_timestamp) - float(self._current_candidate.timestamp), 0.0)
        return age_s <= self.lost_timeout_s

    def _set_debug(
        self,
        candidates: dict[str, HandCandidateState],
        reject_reasons: dict[int, str],
        selected: SelectedHandState,
        object_center_base: Vec3 | None,
        distances: dict[str, float],
        selection_reason: str,
        duplicate_drop_reasons: tuple[str, ...] = (),
    ) -> None:
        """selection 분기 결과와 후보 거리를 last_debug에 저장한다."""

        candidate_items = sorted(candidates.items(), key=lambda item: (int(item[1].camera_id), int(item[1].candidate_index)))
        chosen_id = selected.selected_candidate_id
        self.last_debug = HandSelectorDebug(
            candidate_cameras=tuple(int(candidate.camera_id) for _, candidate in candidate_items),
            candidate_handedness=tuple(str(candidate.handedness) for _, candidate in candidate_items),
            cam0_reject_reason=str(reject_reasons.get(0, "")),
            cam1_reject_reason=str(reject_reasons.get(1, "")),
            chosen_camera=selected.selected_camera,
            current_camera=None if self._current_candidate is None else int(self._current_candidate.camera_id),
            pending_camera=None if self._pending_candidate_id is None else int(candidates[self._pending_candidate_id].camera_id) if self._pending_candidate_id in candidates else None,
            pending_frames=self._pending_frames,
            hysteresis_frames=self.switch_confirm_frames,
            locked_camera=None if self._current_candidate is None else int(self._current_candidate.camera_id),
            lock_stable_frames=self._lock_stable_frames,
            lock_after_stable_frames=1,
            selection_reason=selection_reason,
            chosen_candidate_id=chosen_id,
            current_candidate_id=self._current_candidate_id,
            pending_candidate_id=self._pending_candidate_id,
            object_center_base=object_center_base,
            chosen_distance_m=None if chosen_id is None else distances.get(chosen_id),
            current_distance_m=None if self._current_candidate_id is None else distances.get(self._current_candidate_id),
            duplicate_drop_reasons=tuple(duplicate_drop_reasons),
        )

    def _log_active_hand_switch(
        self,
        *,
        reason: str,
        previous_candidate: HandCandidateState | None,
        next_candidate: HandCandidateState,
        object_center_base: Vec3 | None,
        distances: dict[str, float],
        pending_frames: int,
        best_candidate_id: str | None,
        lost_timeout_s: float | None = None,
    ) -> None:
        """active hand가 실제로 바뀐 순간의 판단 기준을 사람이 읽기 좋은 로그로 남긴다."""

        if not self.log_active_hand_switches:
            return

        previous_id = None if previous_candidate is None else self._candidate_key(previous_candidate)
        next_id = self._candidate_key(next_candidate)
        previous_distance = None if previous_id is None else distances.get(previous_id)
        next_distance = distances.get(next_id)
        advantage_m = (
            None
            if previous_distance is None or next_distance is None
            else float(previous_distance) - float(next_distance)
        )
        previous_age_s = (
            None
            if previous_candidate is None
            else max(float(next_candidate.timestamp) - float(previous_candidate.timestamp), 0.0)
        )

        parts = [
            "[HandSelector] ACTIVE_HAND_SWITCH",
            f"reason={reason}",
            f"prev={self._format_candidate_for_log(previous_candidate, previous_distance)}",
            f"next={self._format_candidate_for_log(next_candidate, next_distance)}",
            f"object={self._format_vec_for_log(object_center_base)}",
            f"best_candidate_id={best_candidate_id}",
            f"active_threshold={self._format_m_for_log(self.active_hand_distance_threshold_m)}",
            f"switch_margin={self._format_m_for_log(self.switch_margin_m)}",
            f"distance_advantage={self._format_m_for_log(advantage_m)}",
            f"pending_frames={int(pending_frames)}/{int(self.switch_confirm_frames)}",
        ]
        if previous_age_s is not None:
            parts.append(f"prev_age={previous_age_s:.3f}s")
        if lost_timeout_s is not None:
            parts.append(f"lost_timeout={float(lost_timeout_s):.3f}s")
        print(" ".join(parts))

    @staticmethod
    def _prefer_known_handedness_candidates(
        candidates: dict[str, HandCandidateState],
        candidate_ids: list[str],
    ) -> list[str]:
        known_ids = [
            candidate_id
            for candidate_id in candidate_ids
            if HandSelector._is_known_handedness_candidate(candidates.get(candidate_id))
        ]
        return known_ids if known_ids else list(candidate_ids)

    @staticmethod
    def _is_known_handedness_candidate(candidate: HandCandidateState | None) -> bool:
        if candidate is None:
            return False
        return str(candidate.handedness).strip().lower() in KNOWN_HANDEDNESS

    @staticmethod
    def _collect_candidates(
        hand_cam0: HandState,
        hand_cam1: HandState,
        *,
        object_center_base: Vec3 | None = None,
    ) -> tuple[dict[str, HandCandidateState], dict[int, str], tuple[str, ...]]:
        """두 카메라의 후보를 candidate_id keyed dict로 모으고 invalid 후보의 거절 이유를 기록한다."""

        candidates: dict[str, HandCandidateState] = {}
        reject_reasons: dict[int, str] = {}
        duplicate_drop_reasons: list[str] = []
        for state in (hand_cam0, hand_cam1):
            state_candidates = list(getattr(state, "hand_candidates", []) or [])
            # 구버전/테스트용 HandState처럼 scalar palm_center만 있는 경우도 하나의 후보로 변환한다.
            if not state_candidates and HandSelector._is_valid_scalar_hand_state(state):
                state_candidates = [HandSelector._candidate_from_scalar_state(state)]

            valid_for_camera = 0
            for candidate in state_candidates:
                reason = HandSelector._candidate_reject_reason(candidate)
                if reason == "":
                    candidate_key = HandSelector._candidate_key(candidate)
                    previous_candidate = candidates.get(candidate_key)
                    if previous_candidate is None:
                        candidates[candidate_key] = candidate
                    else:
                        kept_candidate, dropped_candidate = HandSelector._choose_duplicate_candidate(
                            previous_candidate,
                            candidate,
                            object_center_base=object_center_base,
                        )
                        candidates[candidate_key] = kept_candidate
                        duplicate_drop_reasons.append(
                            "duplicate_handedness_candidate:"
                            f"kept={HandSelector._candidate_key(kept_candidate)}"
                            f"/idx{int(kept_candidate.candidate_index)}"
                            f",dropped={HandSelector._candidate_key(dropped_candidate)}"
                            f"/idx{int(dropped_candidate.candidate_index)}"
                        )
                    valid_for_camera += 1
            if valid_for_camera == 0:
                reject_reasons[int(state.camera_id)] = HandSelector._camera_reject_reason(state, state_candidates)
        return candidates, reject_reasons, tuple(duplicate_drop_reasons)

    @staticmethod
    def _choose_duplicate_candidate(
        first: HandCandidateState,
        second: HandCandidateState,
        *,
        object_center_base: Vec3 | None,
    ) -> tuple[HandCandidateState, HandCandidateState]:
        first_confidence = float(first.confidence)
        second_confidence = float(second.confidence)
        if second_confidence > first_confidence:
            return second, first
        if first_confidence > second_confidence:
            return first, second

        first_distance = HandSelector._distance_to_object(first, object_center_base)
        second_distance = HandSelector._distance_to_object(second, object_center_base)
        if second_distance is not None and (first_distance is None or second_distance < first_distance):
            return second, first
        return first, second

    @staticmethod
    def _camera_reject_reason(state: HandState, state_candidates: list[HandCandidateState]) -> str:
        if state_candidates:
            return "no_valid_candidate"
        if not bool(state.hand_detected):
            return "hand_not_detected"
        if state.palm_center_base is None:
            return "missing_palm_center"
        if state.palm_normal_base is None:
            return "missing_palm_normal"
        if not bool(state.valid):
            return "state_invalid"
        return "no_valid_candidate"

    @staticmethod
    def _candidate_reject_reason(candidate: HandCandidateState | None) -> str:
        if candidate is None:
            return "missing_candidate"
        if not bool(candidate.hand_detected):
            return "hand_not_detected"
        if candidate.palm_center_base is None:
            return "missing_palm_center"
        if candidate.palm_normal_base is None:
            return "missing_palm_normal"
        if not bool(candidate.valid):
            return "state_invalid"
        return ""

    @staticmethod
    def _is_valid_scalar_hand_state(state: HandState | None) -> bool:
        if state is None:
            return False
        return bool(
            state.valid
            and state.hand_detected
            and state.palm_center_base is not None
            and state.palm_normal_base is not None
        )

    @staticmethod
    def _candidate_from_scalar_state(state: HandState) -> HandCandidateState:
        handedness = str(state.handedness)
        normalized_handedness = handedness.strip().lower()
        candidate_id = (
            f"cam{int(state.camera_id)}:{normalized_handedness}"
            if normalized_handedness in KNOWN_HANDEDNESS
            else f"cam{int(state.camera_id)}:legacy"
        )
        return HandCandidateState(
            camera_id=int(state.camera_id),
            frame_id=int(state.frame_id),
            candidate_index=0,
            candidate_id=candidate_id,
            hand_detected=bool(state.hand_detected),
            handedness=handedness,
            confidence=float(state.confidence),
            palm_center_base=state.palm_center_base,
            palm_normal_base=state.palm_normal_base,
            wrist_base=state.wrist_base,
            hand_velocity_base=state.hand_velocity_base,
            timestamp=float(state.timestamp),
            valid=bool(state.valid),
        )

    @staticmethod
    def _candidate_key(candidate: HandCandidateState) -> str:
        if candidate.candidate_id:
            return str(candidate.candidate_id)
        handedness = str(candidate.handedness).strip().lower()
        if handedness in KNOWN_HANDEDNESS:
            return f"cam{int(candidate.camera_id)}:{handedness}"
        return f"cam{int(candidate.camera_id)}:hand{int(candidate.candidate_index)}"

    @staticmethod
    def _distance_to_object(candidate: HandCandidateState, object_center_base: Vec3 | None) -> float | None:
        if object_center_base is None or candidate.palm_center_base is None:
            return None
        object_center = np.asarray(object_center_base, dtype=np.float32).reshape(3)
        palm_center = np.asarray(candidate.palm_center_base, dtype=np.float32).reshape(3)
        if not np.isfinite(palm_center).all() or not np.isfinite(object_center).all():
            return None
        return float(np.linalg.norm(palm_center - object_center))

    @staticmethod
    def _format_candidate_for_log(candidate: HandCandidateState | None, distance_m: float | None) -> str:
        if candidate is None:
            return "none"
        candidate_id = HandSelector._candidate_key(candidate)
        return (
            f"{candidate_id}"
            f"(cam={int(candidate.camera_id)},idx={int(candidate.candidate_index)},"
            f"hand={candidate.handedness},conf={float(candidate.confidence):.3f},"
            f"dist={HandSelector._format_m_for_log(distance_m)},"
            f"center={HandSelector._format_vec_for_log(candidate.palm_center_base)},"
            f"ts={float(candidate.timestamp):.3f})"
        )

    @staticmethod
    def _format_m_for_log(value_m: float | None) -> str:
        if value_m is None:
            return "none"
        return f"{float(value_m) * 1000.0:.1f}mm"

    @staticmethod
    def _format_vec_for_log(values: Vec3 | None) -> str:
        if values is None:
            return "none"
        vector = np.asarray(values, dtype=np.float32).reshape(3)
        if not np.isfinite(vector).all():
            return "invalid"
        return f"({vector[0] * 1000.0:.1f},{vector[1] * 1000.0:.1f},{vector[2] * 1000.0:.1f})mm"

    @staticmethod
    def _compute_distances(
        candidates: dict[str, HandCandidateState],
        object_center_base: Vec3 | None,
    ) -> dict[str, float]:
        """후보 palm center와 object center 사이의 base-frame 3D Euclidean distance를 계산한다."""

        if object_center_base is None:
            return {}
        distances: dict[str, float] = {}
        for candidate_id, candidate in candidates.items():
            distance = HandSelector._distance_to_object(candidate, object_center_base)
            if distance is not None:
                distances[candidate_id] = distance
        return distances

    @staticmethod
    def _resolve_now_timestamp(
        hand_cam0: HandState,
        hand_cam1: HandState,
        candidates: dict[str, HandCandidateState],
    ) -> float:
        timestamps = [float(hand_cam0.timestamp), float(hand_cam1.timestamp)]
        timestamps.extend(float(candidate.timestamp) for candidate in candidates.values())
        return max(timestamps)

    @staticmethod
    def _build_invalid_state(hand_cam0: HandState, hand_cam1: HandState) -> SelectedHandState:
        """downstream이 명확히 hand 없음으로 해석할 수 있는 invalid SelectedHandState를 만든다."""

        return SelectedHandState(
            frame_id=max(int(hand_cam0.frame_id), int(hand_cam1.frame_id)),
            selected_camera=None,
            selected_candidate_index=None,
            selected_candidate_id=None,
            handedness=None,
            confidence=0.0,
            palm_center_base=None,
            palm_normal_base=None,
            wrist_base=None,
            hand_velocity_base=None,
            timestamp=max(float(hand_cam0.timestamp), float(hand_cam1.timestamp)),
            valid=False,
        )

    @staticmethod
    def _build_selected_state(chosen_candidate: HandCandidateState) -> SelectedHandState:
        """선택된 HandCandidateState를 fusion/grasp에서 쓰는 SelectedHandState 형태로 복사한다."""

        return SelectedHandState(
            frame_id=int(chosen_candidate.frame_id),
            selected_camera=int(chosen_candidate.camera_id),
            selected_candidate_index=int(chosen_candidate.candidate_index),
            selected_candidate_id=HandSelector._candidate_key(chosen_candidate),
            handedness=str(chosen_candidate.handedness),
            confidence=float(chosen_candidate.confidence),
            palm_center_base=chosen_candidate.palm_center_base,
            palm_normal_base=chosen_candidate.palm_normal_base,
            wrist_base=chosen_candidate.wrist_base,
            hand_velocity_base=chosen_candidate.hand_velocity_base,
            timestamp=float(chosen_candidate.timestamp),
            valid=True,
        )


def object_center_for_hand_selection(object_cam0: ObjectState, object_cam1: ObjectState) -> Vec3 | None:
    """카메라별 object centroid를 point_count 가중 평균해 hand selection 기준점을 만든다."""

    usable: list[tuple[np.ndarray, float]] = []
    for state in (object_cam0, object_cam1):
        centroid = getattr(state, "centroid_base", None)
        if not bool(getattr(state, "valid", False)) or centroid is None:
            continue
        center = np.asarray(centroid, dtype=np.float32).reshape(3)
        if not np.isfinite(center).all():
            continue
        weight = max(float(getattr(state, "point_count", 0) or 0), 1.0)
        usable.append((center, weight))
    if not usable:
        return None
    total_weight = sum(weight for _, weight in usable)
    weighted_center = sum(center * weight for center, weight in usable) / total_weight
    return tuple(float(value) for value in weighted_center)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "HandSelectorDebug",
    "HandSelector",
    "object_center_for_hand_selection",
]
