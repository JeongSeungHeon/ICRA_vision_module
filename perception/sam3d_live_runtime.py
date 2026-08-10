"""Runtime Hands23/FastSAM state machine for the standalone RTDE loop."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import threading
import time
from typing import Any

from perception.dynamic_bbox import DynamicFastSAMBBoxState
from perception.hands23_ipc import Hands23SidecarClient
from perception.sam3d_backend import REPO_ROOT, load_config, resolve_repo_path
from perception.sam3d_backend import build_runtime_shape_fitting_tracker
from perception.sam3d_runtime import (
    StableCaptureAccumulator,
    generate_runtime_template,
    validate_template_points,
    write_capture_artifacts,
)

READINESS_SILHOUETTE = "silhouette_ready"
READINESS_SHAPE_FIT = "shape_fit_ready"
READINESS_RAW_CLOUD = "raw_cloud_ready"
READINESS_STRATEGIES = {
    READINESS_SILHOUETTE,
    READINESS_SHAPE_FIT,
    READINESS_RAW_CLOUD,
}


@dataclass(frozen=True)
class Sam3DLiveDebug:
    active: bool
    gate_blocked: bool
    healthy: bool
    initialization_phase: str
    scaling_valid_frames: int
    scaling_required_frames: int
    silhouette_scale_frozen: bool
    readiness_strategy: str
    regeneration_enabled: bool
    cam0: Any
    cam1: Any
    last_inference_ms: float | None
    last_roundtrip_ms: float | None
    last_error: str | None


class Sam3DLiveRuntime:
    """Apply dynamic prompts before inference and consume results without blocking."""

    def __init__(
        self,
        pipeline: dict[str, Any],
        config_path: str | Path,
        *,
        ready_valid_frames: int | None = None,
        readiness_strategy: str = READINESS_SILHOUETTE,
        regeneration_enabled: bool = True,
        repo_root: str | Path = REPO_ROOT,
        client: Hands23SidecarClient | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.config_path = str(Path(config_path).expanduser().resolve())
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.config = load_config(self.config_path)
        self.readiness_strategy = str(readiness_strategy).strip().lower()
        if self.readiness_strategy not in READINESS_STRATEGIES:
            raise ValueError(
                f"Unsupported readiness_strategy={readiness_strategy!r}; "
                f"expected one of {sorted(READINESS_STRATEGIES)}"
            )
        self.regeneration_enabled = bool(regeneration_enabled)
        dynamic_cfg = (
            self.config.get("perception", {}).get("object", {}).get("hands23_bbox", {}) or {}
        )
        self.enabled = bool(dynamic_cfg.get("enabled", True))
        sam3d_cfg = (
            self.config.get("perception", {}).get("shape_fitting", {}).get("sam3d", {}) or {}
        )
        if self.readiness_strategy == READINESS_RAW_CLOUD:
            configured_scaling_frames = int(dynamic_cfg.get("raw_cloud_ready_frames", 4))
        else:
            configured_scaling_frames = int(
                sam3d_cfg.get("initial_silhouette_scaling_frames", 4)
            )
        self.ready_valid_frames = max(
            1,
            int(configured_scaling_frames if ready_valid_frames is None else ready_valid_frames),
        )
        self.state = DynamicFastSAMBBoxState(
            hold_timeout_s=float(dynamic_cfg.get("hold_timeout_s", 0.5)),
            result_age_timeout_s=float(dynamic_cfg.get("result_age_timeout_s", 1.0)),
        )
        self.client = client or self._build_client(dynamic_cfg)
        self.task_id = 0
        self._shape_ready_count = 0
        self._initialization_phase = self._fixed_initialization_phase()
        self._gate_blocked = bool(self.enabled)
        self._last_inference_ms: float | None = None
        self._last_roundtrip_ms: float | None = None
        self._force_gate = False
        self._regeneration_state = "idle"
        self._regeneration_error: str | None = None
        self._regeneration_thread: threading.Thread | None = None
        self._regeneration_result: tuple[Any, str] | None = None
        self._regeneration_lock = threading.Lock()
        self._capture_accumulator: StableCaptureAccumulator | None = None

    def _build_client(self, dynamic_cfg: dict[str, Any]) -> Hands23SidecarClient:
        runtime_cfg = self.config.get("runtime", {}).get("hands23_sidecar", {}) or {}
        return Hands23SidecarClient(
            python_interpreter=resolve_repo_path(
                runtime_cfg.get(
                    "python_interpreter",
                    "/home/ur5/miniforge3/envs/hands23_ros2/bin/python",
                ),
                repo_root=self.repo_root,
            ),
            sidecar_script=self.repo_root / "tools" / "hands23_sidecar.py",
            config_path=self.config_path,
            repo_path=resolve_repo_path(
                runtime_cfg.get("repo_path", "external/hands23_detector"),
                repo_root=self.repo_root,
            ),
            detector_config_path=resolve_repo_path(
                runtime_cfg.get(
                    "config_path",
                    "external/hands23_detector/faster_rcnn_X_101_32x8d_FPN_3x_Hands23.yaml",
                ),
                repo_root=self.repo_root,
            ),
            weights_path=resolve_repo_path(
                runtime_cfg.get(
                    "weights_path",
                    "external/hands23_detector/model_weights/model_hands23.pth",
                ),
                repo_root=self.repo_root,
            ),
            socket_path=runtime_cfg.get("socket", "/tmp/handover_hands23.sock"),
            startup_timeout_s=float(runtime_cfg.get("startup_timeout_s", 60.0)),
            request_timeout_s=float(runtime_cfg.get("request_timeout_s", 10.0)),
            max_input_hz=float(dynamic_cfg.get("max_input_hz", 10.0)),
        )

    def _fixed_initialization_phase(self) -> str:
        if self.readiness_strategy == READINESS_RAW_CLOUD:
            return "fixed_raw_cloud"
        if self.readiness_strategy == READINESS_SHAPE_FIT:
            return "fixed_shape_fit"
        return "fixed_scaling"

    def start(self) -> None:
        if self.enabled:
            self.client.start()
            self.client.reset(self.task_id)

    def close(self) -> None:
        if self.enabled:
            self.client.close()
        thread = self._regeneration_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def _engines(self):
        for camera_id in (0, 1):
            yield camera_id, self.pipeline[f"object_worker_cam{camera_id}"].segmentation_engine

    def reset(self, task_id: int) -> None:
        self.task_id = int(task_id)
        self._shape_ready_count = 0
        self._initialization_phase = self._fixed_initialization_phase()
        self._gate_blocked = bool(self.enabled)
        self.state.reset(self.task_id)
        if self.enabled:
            self.client.reset(self.task_id)
        for _camera_id, engine in self._engines():
            engine.reset_bbox()

    @property
    def regeneration_state(self) -> str:
        return self._regeneration_state

    @property
    def initialization_phase(self) -> str:
        return self._initialization_phase

    @property
    def silhouette_scale_frozen(self) -> bool:
        return bool(
            self.enabled
            and self.readiness_strategy == READINESS_SILHOUETTE
            and self._initialization_phase != self._fixed_initialization_phase()
        )

    @property
    def initial_scaling_progress(self) -> tuple[int, int]:
        return int(self._shape_ready_count), int(self.ready_valid_frames)

    def begin_regeneration(self, task_id: int) -> bool:
        """Start fixed-bbox stable capture; generation begins when capture is ready."""
        if (
            not self.enabled
            or not self.regeneration_enabled
            or self._regeneration_state not in {"idle", "ready", "failed"}
        ):
            return False
        fastsam_cfg = self.config.get("perception", {}).get("object", {}).get("fastsam", {}) or {}
        self.reset(task_id)
        self._force_gate = True
        self._gate_blocked = True
        self._regeneration_state = "capturing"
        self._regeneration_error = None
        self._capture_accumulator = StableCaptureAccumulator(
            stable_frames=int(fastsam_cfg.get("stable_frames", 3)),
            min_mask_pixels=int(fastsam_cfg.get("min_mask_pixels", 300)),
            min_temporal_iou=float(fastsam_cfg.get("min_temporal_iou", 0.8)),
        )
        return True

    def _start_generation(self, capture) -> None:
        runtime_cfg = self.config.get("runtime", {}).get("sam3d_server", {}) or {}
        sam_cfg = self.config.get("perception", {}).get("shape_fitting", {}).get("sam3d", {}) or {}
        artifact_root = resolve_repo_path(
            runtime_cfg.get("artifact_root", "output/sam3d"), repo_root=self.repo_root
        )
        artifact_dir = artifact_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        configured_bboxes = {
            f"cam{camera_id}": engine.initial_bbox
            for camera_id, engine in self._engines()
        }
        label = str(
            self.config.get("perception", {}).get("object", {}).get("fastsam", {}).get("label", "sam3d_object")
        )
        write_capture_artifacts(
            artifact_dir,
            capture,
            label=label,
            configured_bboxes=configured_bboxes,
        )
        self._regeneration_state = "generating"

        def _worker() -> None:
            try:
                template_path = generate_runtime_template(
                    config_path=self.config_path,
                    artifact_dir=artifact_dir,
                    repo_root=self.repo_root,
                    execution_mode=str(runtime_cfg.get("execution_mode", "server")),
                    sam3d_python=resolve_repo_path(
                        runtime_cfg.get(
                            "python_interpreter",
                            "/home/ur5/miniforge3/envs/sam3d-objects/bin/python",
                        ),
                        repo_root=self.repo_root,
                    ),
                    sam3d_repo=resolve_repo_path(
                        runtime_cfg.get("repo_path", "external/sam-3d-objects"),
                        repo_root=self.repo_root,
                    ),
                    server_socket=str(runtime_cfg.get("server_socket", "/tmp/handover_sam3d_stage1.sock")),
                    server_ready_timeout_s=float(runtime_cfg.get("server_ready_timeout_s", 120.0)),
                    request_timeout_s=float(runtime_cfg.get("request_timeout_s", 120.0)),
                )
                validate_template_points(
                    template_path,
                    min_template_points=int(sam_cfg.get("min_template_points", 100)),
                )
                tracker = build_runtime_shape_fitting_tracker(
                    self.config_path,
                    template_path,
                    silhouette_enabled_override=(
                        False if self.readiness_strategy == READINESS_SHAPE_FIT else None
                    ),
                )
                with self._regeneration_lock:
                    self._regeneration_result = (tracker, str(template_path))
            except Exception as exc:
                with self._regeneration_lock:
                    self._regeneration_error = f"{type(exc).__name__}: {exc}"

        self._regeneration_thread = threading.Thread(
            target=_worker,
            name="sam3d-regeneration",
            daemon=True,
        )
        self._regeneration_thread.start()

    def capture_regeneration_frame(self, snapshot) -> None:
        if self._regeneration_state != "capturing" or self._capture_accumulator is None:
            return
        worker = self.pipeline["object_worker_cam0"]
        debug = getattr(worker, "last_debug", None)
        mask = None if debug is None else getattr(debug, "combined_mask", None)
        selection = getattr(worker.segmentation_engine, "last_selection", None)
        capture = self._capture_accumulator.add(
            snapshot.cam0.color_image,
            mask,
            selection,
        )
        if capture is not None:
            self._capture_accumulator = None
            self._start_generation(capture)

    def poll_regeneration(self) -> str | None:
        """Install a completed tracker at a main-thread frame boundary."""
        if self._regeneration_state != "generating":
            return None
        with self._regeneration_lock:
            result = self._regeneration_result
            error = self._regeneration_error
            self._regeneration_result = None
        if result is not None:
            tracker, template_path = result
            self.pipeline["shape_fitting_tracker"] = tracker
            self.pipeline["runtime_template_path"] = template_path
            self._shape_ready_count = 0
            self._initialization_phase = self._fixed_initialization_phase()
            self.state.reset(self.task_id)
            self.client.reset(self.task_id)
            for _camera_id, engine in self._engines():
                engine.reset_bbox()
            self._regeneration_state = "fitting"
            return "installed"
        if error is not None:
            self.pipeline["shape_fitting_tracker"].reset()
            self.state.reset(self.task_id)
            self.client.reset(self.task_id)
            for _camera_id, engine in self._engines():
                engine.reset_bbox()
            self._force_gate = False
            self._regeneration_state = "failed"
            return "failed"
        return None

    def before_frame(self, snapshot, *, task_id: int, now_ros_s: float | None = None) -> bool:
        if not self.enabled:
            self._gate_blocked = False
            return False
        if int(task_id) != self.task_id:
            self.reset(int(task_id))
        now_ros_s = time.time() if now_ros_s is None else float(now_ros_s)
        for result in self.client.drain_results():
            self._last_inference_ms = result.inference_ms
            self._last_roundtrip_ms = result.roundtrip_ms
            self.state.accept(
                camera_id=result.camera_id,
                task_id=result.task_id,
                frame_seq=result.frame_seq,
                valid=result.valid,
                bbox_xyxy=result.bbox_xyxy,
                capture_time_s=result.capture_time_s,
                now_ros_s=now_ros_s,
                hand_side=result.hand_side,
                hand_score=result.hand_score,
                object_score=result.object_score,
                contact_state=result.contact_state,
                reason=result.reason,
            )

        if self._regeneration_state in {"capturing", "generating", "fitting"} and not self.state.active:
            for _camera_id, engine in self._engines():
                engine.reset_bbox()
            self._gate_blocked = True
            return True

        if not self.state.active:
            for _camera_id, engine in self._engines():
                engine.reset_bbox()
            # Fixed FastSAM prompts remain available for shape fitting, but no
            # measured/predicted robot target may escape during scale bootstrap.
            self._gate_blocked = True
            return True

        for camera_id, frame in ((0, snapshot.cam0), (1, snapshot.cam1)):
            capture_time_s = float(frame.timestamp_ms) / 1000.0
            # RealSense device timestamps are not Unix time. Use wall time for
            # cross-process freshness while retaining frame_seq for ordering.
            if capture_time_s < 1_000_000_000.0:
                capture_time_s = now_ros_s
            self.client.submit(
                camera_id,
                frame.color_image,
                task_id=self.task_id,
                frame_seq=int(snapshot.pair_index),
                capture_time_s=capture_time_s,
            )

        available = False
        now_monotonic = time.monotonic()
        for camera_id, engine in self._engines():
            bbox = self.state.bbox_for_camera(camera_id, now_monotonic_s=now_monotonic)
            engine.set_bbox(bbox)
            available = available or bbox is not None
        if available and self.client.healthy and self._initialization_phase == "waiting_hands23":
            self._initialization_phase = "dynamic"
        if (
            self._force_gate
            and self._initialization_phase == "dynamic"
            and self._regeneration_state == "fitting"
        ):
            self._force_gate = False
            self._regeneration_state = "ready"
        self._gate_blocked = bool(
            self._force_gate
            or self._initialization_phase != "dynamic"
            or not available
            or not self.client.healthy
        )
        if self._gate_blocked:
            for _camera_id, engine in self._engines():
                engine.set_bbox(None)
        return self._gate_blocked

    def after_shape_fit(self, shape_fitting_state) -> bool:
        if self.readiness_strategy == READINESS_RAW_CLOUD:
            return False
        if not self.enabled or self._initialization_phase != self._fixed_initialization_phase():
            return False
        if self._regeneration_state in {"capturing", "generating"}:
            return False
        ready = bool(
            getattr(shape_fitting_state, "valid", False)
            and getattr(shape_fitting_state, "initialized", False)
        )
        if self.readiness_strategy == READINESS_SILHOUETTE:
            ready = bool(
                ready
                and getattr(shape_fitting_state, "silhouette_enabled", False)
                and int(getattr(shape_fitting_state, "silhouette_candidate_count", 0)) > 0
                and int(getattr(shape_fitting_state, "silhouette_valid_camera_count", 0)) > 0
                and str(getattr(shape_fitting_state, "silhouette_reason", "")) in {"changed", "kept"}
            )
        return self._advance_initialization(ready)

    def after_raw_cloud(self, merged_object) -> bool:
        """Advance fixed-bbox initialization from consecutive valid raw clouds."""
        if self.readiness_strategy != READINESS_RAW_CLOUD:
            return False
        if not self.enabled or self._initialization_phase != self._fixed_initialization_phase():
            return False
        ready = bool(
            getattr(merged_object, "valid", False)
            and int(getattr(merged_object, "merged_point_count", 0)) > 0
        )
        return self._advance_initialization(ready)

    def _advance_initialization(self, ready: bool) -> bool:
        self._shape_ready_count = self._shape_ready_count + 1 if ready else 0
        if self._shape_ready_count < self.ready_valid_frames:
            return False
        self.state.activate(self.task_id)
        self._initialization_phase = "waiting_hands23"
        self._gate_blocked = True
        return True

    def after_object_perception(self, object_cam0, object_cam1) -> bool:
        """Fail closed when every available FastSAM observation is invalid."""
        if not self.enabled or not self.state.active:
            return self._gate_blocked
        any_valid = any(
            bool(getattr(state, "valid", False))
            for state in (object_cam0, object_cam1)
        )
        if not any_valid:
            self._gate_blocked = True
        return self._gate_blocked

    def debug(self) -> Sam3DLiveDebug:
        return Sam3DLiveDebug(
            active=self.state.active,
            gate_blocked=self._gate_blocked,
            healthy=self.client.healthy if self.enabled else True,
            initialization_phase=self._initialization_phase,
            scaling_valid_frames=int(self._shape_ready_count),
            scaling_required_frames=int(self.ready_valid_frames),
            silhouette_scale_frozen=self.silhouette_scale_frozen,
            readiness_strategy=self.readiness_strategy,
            regeneration_enabled=self.regeneration_enabled,
            cam0=self.state.debug(0),
            cam1=self.state.debug(1),
            last_inference_ms=self._last_inference_ms,
            last_roundtrip_ms=self._last_roundtrip_ms,
            last_error=(self._regeneration_error or self.client.last_error) if self.enabled else None,
        )


def enforce_sam3d_target_gate(shared_state, blocked: bool) -> bool:
    """Hard-clear measured, predicted, and armed targets when the gate closes."""
    if not bool(blocked):
        return False
    shared_state.clear_target(reset_prediction=True, reset_arm=True)
    return True


__all__ = [
    "READINESS_RAW_CLOUD",
    "READINESS_SHAPE_FIT",
    "READINESS_SILHOUETTE",
    "Sam3DLiveDebug",
    "Sam3DLiveRuntime",
    "enforce_sam3d_target_gate",
]
