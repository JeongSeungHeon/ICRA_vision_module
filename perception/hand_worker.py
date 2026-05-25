"""Reusable per-camera hand perception worker."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2 as cv
import mediapipe as mp
import numpy as np
import yaml

from calibration.extrinsics import TransformChain, load_transform_chain
from handpose3d.hand_pose_6d import estimate_palm_pose
from system.shared_state import HANDEDNESS_UNKNOWN, HandCandidateState, HandState
from utils.depth_lifter import lift_hand_pose_3d
from utils.realsense_stream import FrameBundle

DEFAULT_CONFIG_PATH = Path("configs/handover.yaml")
MP_HANDS = mp.solutions.hands


@dataclass
class HandWorkerDebug:
    camera_id: int
    infer_handedness: str | None
    keypoints_2d: np.ndarray
    points_3d_camera: np.ndarray
    points_3d_base: np.ndarray
    valid_mask: np.ndarray
    quality_scores: np.ndarray
    palm_pose_base: dict[str, Any]
    candidate_count: int = 0


class HandWorker:
    def __init__(
        self,
        camera_id: int,
        transform_chain: TransformChain,
        config: dict[str, Any],
    ) -> None:
        self.camera_id = int(camera_id)
        self.transform_chain = transform_chain
        self.config = config

        hand_cfg = config.get("perception", {}).get("hand", {})
        detector_cfg = hand_cfg.get("detector", {})
        depth_cfg = hand_cfg.get("depth_lifting", {})

        self.min_pose_quality = float(hand_cfg.get("min_pose_quality", 0.25))
        self.patch_radius = int(depth_cfg.get("patch_radius", 2))
        self.min_depth_m = float(depth_cfg.get("min_depth_m", 0.10))
        self.max_depth_m = float(depth_cfg.get("max_depth_m", 1.20))
        self.min_valid_keypoints = int(depth_cfg.get("min_valid_keypoints", 4))
        self.smoothing_window_frames = int(hand_cfg.get("smoothing_window_frames", 3)) # 수정 5 -> 3
        self.velocity_alpha = float(hand_cfg.get("velocity_alpha", 0.5))
        self._detector_min_detection_confidence = float(detector_cfg.get("min_detection_confidence", 0.5))
        self._detector_max_num_hands = int(detector_cfg.get("max_num_hands", 2))
        self._detector_min_tracking_confidence = float(detector_cfg.get("min_tracking_confidence", 0.5))

        self._hands = self._create_hands()
        self._candidate_filters: dict[str, dict[str, Any]] = {}
        self.last_debug: HandWorkerDebug | None = None

    @classmethod
    def from_config(cls, camera_id: int, config_path: str | Path = DEFAULT_CONFIG_PATH) -> "HandWorker":
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        transform_chain = load_transform_chain(config_path)
        return cls(camera_id=camera_id, transform_chain=transform_chain, config=config)

    def close(self) -> None:
        self._hands.close()

    def reset(self) -> None:
        self._candidate_filters.clear()
        self.last_debug = None
        self._hands.close()
        self._hands = self._create_hands()

    def process_frame(self, frame_bundle: FrameBundle, frame_id: int = -1) -> HandState:
        detections = self._detect_hand_keypoints(frame_bundle.color_image)
        timestamp_s = float(frame_bundle.timestamp_ms) / 1000.0
        candidates: list[HandCandidateState] = []
        debug_payloads: list[tuple[np.ndarray, str | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = []
        active_filter_keys: set[str] = set()

        for candidate_index, (keypoints_2d, handedness) in enumerate(detections):
            normalized_handedness = self._normalize_handedness_label(handedness)
            filter_key = self._candidate_filter_key(candidate_index, normalized_handedness)
            active_filter_keys.add(filter_key)
            filter_state = self._candidate_filters.setdefault(
                filter_key,
                {
                    "centers": deque(maxlen=max(self.smoothing_window_frames, 1)),
                    "normals": deque(maxlen=max(self.smoothing_window_frames, 1)),
                    "previous_rotation": None,
                    "previous_smoothed_center": None,
                    "previous_timestamp": None,
                    "smoothed_velocity": None,
                },
            )
            points_3d_camera, valid_mask, _, quality_scores = lift_hand_pose_3d(
                keypoints_2d,
                frame_bundle.depth_image_m,
                frame_bundle.intrinsics,
                patch_radius=self.patch_radius,
                min_depth_m=self.min_depth_m,
                max_depth_m=self.max_depth_m,
            )
            points_3d_base = np.full((21, 3), np.nan, dtype=np.float32)
            if np.any(valid_mask):
                points_3d_base[valid_mask] = self.transform_chain.transform_points_camera_to_base(
                    self.camera_id,
                    points_3d_camera[valid_mask],
                )

            pose_handedness = normalized_handedness if normalized_handedness is not None else HANDEDNESS_UNKNOWN
            palm_pose_base = estimate_palm_pose(
                points_3d_base,
                valid_mask,
                quality_scores=quality_scores,
                handedness=pose_handedness,
                previous_rotation=filter_state.get("previous_rotation"),
            )
            if palm_pose_base.get("valid", False):
                filter_state["previous_rotation"] = np.asarray(palm_pose_base["rotation_matrix"], dtype=np.float32)

            confidence = float(palm_pose_base.get("quality", 0.0))
            valid_keypoint_count = int(np.count_nonzero(valid_mask))
            smoothed_center = None
            smoothed_normal = None
            wrist_base = None
            velocity_base = None
            hand_detected = False
            if (
                palm_pose_base.get("valid", False)
                and valid_keypoint_count >= self.min_valid_keypoints
                and confidence >= self.min_pose_quality
            ):
                raw_center = np.asarray(palm_pose_base["position"], dtype=np.float32)
                raw_normal = np.asarray(palm_pose_base["rotation_matrix"], dtype=np.float32)[:, 2]
                filter_state["centers"].append(raw_center)
                filter_state["normals"].append(raw_normal)
                smoothed_center = self._smooth_vector_history(filter_state["centers"])
                smoothed_normal = self._normalize(self._smooth_vector_history(filter_state["normals"]))
                wrist_base = self._extract_wrist(points_3d_base, valid_mask)
                velocity_base = self._update_candidate_velocity(filter_state, smoothed_center, timestamp_s)
                hand_detected = smoothed_center is not None and smoothed_normal is not None
            else:
                filter_state["centers"].clear()
                filter_state["normals"].clear()
                filter_state["previous_smoothed_center"] = None
                filter_state["previous_timestamp"] = None
                filter_state["smoothed_velocity"] = None

            if hand_detected:
                candidate_id = f"cam{self.camera_id}:hand{candidate_index}"
                candidates.append(
                    HandCandidateState(
                        camera_id=self.camera_id,
                        frame_id=int(frame_id),
                        candidate_index=int(candidate_index),
                        candidate_id=candidate_id,
                        hand_detected=True,
                        handedness=normalized_handedness,
                        confidence=confidence,
                        palm_center_base=tuple(float(v) for v in smoothed_center),
                        palm_normal_base=tuple(float(v) for v in smoothed_normal),
                        wrist_base=None if wrist_base is None else tuple(float(v) for v in wrist_base),
                        hand_velocity_base=None if velocity_base is None else tuple(float(v) for v in velocity_base),
                        timestamp=timestamp_s,
                        valid=True,
                    )
                )
            debug_payloads.append((keypoints_2d, handedness, points_3d_camera, points_3d_base, valid_mask, quality_scores, palm_pose_base))

        for filter_key in list(self._candidate_filters.keys()):
            if filter_key not in active_filter_keys:
                del self._candidate_filters[filter_key]

        primary = max(candidates, key=lambda candidate: float(candidate.confidence), default=None)
        state = HandState(
            camera_id=self.camera_id,
            frame_id=int(frame_id),
            hand_detected=primary is not None,
            handedness=HANDEDNESS_UNKNOWN if primary is None else primary.handedness,
            confidence=0.0 if primary is None else float(primary.confidence),
            palm_center_base=None if primary is None else primary.palm_center_base,
            palm_normal_base=None if primary is None else primary.palm_normal_base,
            wrist_base=None if primary is None else primary.wrist_base,
            hand_velocity_base=None if primary is None else primary.hand_velocity_base,
            hand_candidates=candidates,
            timestamp=timestamp_s,
            valid=primary is not None,
        )
        debug = self._build_debug_payload(debug_payloads)
        self.last_debug = HandWorkerDebug(
            camera_id=self.camera_id,
            infer_handedness=debug["handedness"],
            keypoints_2d=debug["keypoints_2d"],
            points_3d_camera=debug["points_3d_camera"],
            points_3d_base=debug["points_3d_base"],
            valid_mask=debug["valid_mask"],
            quality_scores=debug["quality_scores"],
            palm_pose_base=debug["palm_pose_base"],
            candidate_count=len(candidates),
        )
        return state

    def process_latest(self, sensor_hub: Any) -> HandState:
        frame_bundle = sensor_hub.get_latest_cam0() if self.camera_id == 0 else sensor_hub.get_latest_cam1()
        sensor_state = sensor_hub.get_latest_sensor_state(self.camera_id)
        return self.process_frame(frame_bundle, frame_id=sensor_state.frame_id)

    def _detect_hand_keypoints(self, bgr_frame: np.ndarray) -> list[tuple[np.ndarray, str | None]]:
        rgb_frame = cv.cvtColor(bgr_frame, cv.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self._hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        image_height, image_width = bgr_frame.shape[:2]
        if not results.multi_hand_landmarks:
            return []

        detections: list[tuple[np.ndarray, str | None]] = []
        handedness_results = list(results.multi_handedness or [])
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            frame_keypoints = np.full((21, 2), -1, dtype=np.int32)
            handedness = None
            if hand_index < len(handedness_results) and handedness_results[hand_index].classification:
                handedness = handedness_results[hand_index].classification[0].label
            for point_index in range(21):
                landmark = hand_landmarks.landmark[point_index]
                pixel_x = int(round(image_width * landmark.x))
                pixel_y = int(round(image_height * landmark.y))
                frame_keypoints[point_index, 0] = int(np.clip(pixel_x, 0, image_width - 1))
                frame_keypoints[point_index, 1] = int(np.clip(pixel_y, 0, image_height - 1))
            detections.append((frame_keypoints, handedness))
        return detections

    def _create_hands(self) -> Any:
        return MP_HANDS.Hands(
            min_detection_confidence=self._detector_min_detection_confidence,
            max_num_hands=self._detector_max_num_hands,
            min_tracking_confidence=self._detector_min_tracking_confidence,
        )

    def _update_candidate_velocity(
        self,
        filter_state: dict[str, Any],
        smoothed_center: np.ndarray | None,
        timestamp_s: float,
    ) -> np.ndarray | None:
        if smoothed_center is None:
            return None
        previous_center = filter_state.get("previous_smoothed_center")
        previous_timestamp = filter_state.get("previous_timestamp")
        if previous_center is None or previous_timestamp is None:
            filter_state["previous_smoothed_center"] = smoothed_center.astype(np.float32)
            filter_state["previous_timestamp"] = float(timestamp_s)
            filter_state["smoothed_velocity"] = np.zeros(3, dtype=np.float32)
            return filter_state["smoothed_velocity"].copy()

        dt = max(float(timestamp_s) - float(previous_timestamp), 1e-6)
        raw_velocity = (smoothed_center - previous_center) / dt
        smoothed_velocity = filter_state.get("smoothed_velocity")
        if smoothed_velocity is None:
            filter_state["smoothed_velocity"] = raw_velocity.astype(np.float32)
        else:
            filter_state["smoothed_velocity"] = (
                self.velocity_alpha * raw_velocity + (1.0 - self.velocity_alpha) * smoothed_velocity
            ).astype(np.float32)

        filter_state["previous_smoothed_center"] = smoothed_center.astype(np.float32)
        filter_state["previous_timestamp"] = float(timestamp_s)
        return filter_state["smoothed_velocity"].copy()

    @staticmethod
    def _candidate_filter_key(candidate_index: int, handedness: str | None) -> str:
        return f"{int(candidate_index)}:{handedness or HANDEDNESS_UNKNOWN}"

    @staticmethod
    def _build_debug_payload(payloads: list[tuple[np.ndarray, str | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]]) -> dict[str, Any]:
        if payloads:
            keypoints_2d, handedness, points_3d_camera, points_3d_base, valid_mask, quality_scores, palm_pose_base = payloads[0]
            return {
                "handedness": handedness,
                "keypoints_2d": np.asarray(keypoints_2d, dtype=np.int32),
                "points_3d_camera": np.asarray(points_3d_camera, dtype=np.float32),
                "points_3d_base": np.asarray(points_3d_base, dtype=np.float32),
                "valid_mask": np.asarray(valid_mask, dtype=bool),
                "quality_scores": np.asarray(quality_scores, dtype=np.float32),
                "palm_pose_base": palm_pose_base,
            }
        return {
            "handedness": None,
            "keypoints_2d": np.full((21, 2), -1, dtype=np.int32),
            "points_3d_camera": np.full((21, 3), np.nan, dtype=np.float32),
            "points_3d_base": np.full((21, 3), np.nan, dtype=np.float32),
            "valid_mask": np.zeros(21, dtype=bool),
            "quality_scores": np.zeros(21, dtype=np.float32),
            "palm_pose_base": {"valid": False, "quality": 0.0},
        }

    @staticmethod
    def _smooth_vector_history(history: deque[np.ndarray]) -> np.ndarray | None:
        if not history:
            return None
        stacked = np.stack(history, axis=0).astype(np.float32)
        return np.mean(stacked, axis=0).astype(np.float32)

    @staticmethod
    def _normalize(vector: np.ndarray | None) -> np.ndarray | None:
        if vector is None:
            return None
        norm = float(np.linalg.norm(vector))
        if norm < 1e-6:
            return None
        return (vector / norm).astype(np.float32)

    @staticmethod
    def _extract_wrist(points_3d_base: np.ndarray, valid_mask: np.ndarray) -> np.ndarray | None:
        if points_3d_base.shape != (21, 3):
            return None
        if not bool(valid_mask[0]):
            return None
        wrist = np.asarray(points_3d_base[0], dtype=np.float32)
        if not np.isfinite(wrist).all():
            return None
        return wrist

    @staticmethod
    def _normalize_handedness_label(handedness: str | None) -> str:
        if handedness == "Right":
            return "right"
        if handedness == "Left":
            return "left"
        return HANDEDNESS_UNKNOWN


class HandWorkerCam0(HandWorker):
    @classmethod
    def from_config(cls, config_path: str | Path = DEFAULT_CONFIG_PATH) -> "HandWorkerCam0":
        worker = super().from_config(camera_id=0, config_path=config_path)
        return cls(worker.camera_id, worker.transform_chain, worker.config)


class HandWorkerCam1(HandWorker):
    @classmethod
    def from_config(cls, config_path: str | Path = DEFAULT_CONFIG_PATH) -> "HandWorkerCam1":
        worker = super().from_config(camera_id=1, config_path=config_path)
        return cls(worker.camera_id, worker.transform_chain, worker.config)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "HandWorkerDebug",
    "HandWorker",
    "HandWorkerCam0",
    "HandWorkerCam1",
]
