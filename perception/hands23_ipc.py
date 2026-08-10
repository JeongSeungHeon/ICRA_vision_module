"""Standalone Hands23 sidecar protocol and latest-only asynchronous client."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import socket
import stat
import struct
import subprocess
import threading
import time
from typing import Any

import numpy as np
import yaml


_HEADER = struct.Struct("!I")
_MAX_HEADER_BYTES = 1 << 20


class Hands23IPCError(RuntimeError):
    pass


def validate_hands23_assets(
    *,
    python_interpreter: str | Path,
    sidecar_script: str | Path,
    config_path: str | Path,
    repo_path: str | Path,
    detector_config_path: str | Path,
    weights_path: str | Path,
) -> dict[str, Path]:
    """Validate both Hands23 data files and its importable source checkout."""
    paths = {
        "Hands23 Python": Path(python_interpreter).expanduser().resolve(),
        "sidecar script": Path(sidecar_script).expanduser().resolve(),
        "runtime config": Path(config_path).expanduser().resolve(),
        "Hands23 repo": Path(repo_path).expanduser().resolve(),
        "Hands23 config": Path(detector_config_path).expanduser().resolve(),
        "Hands23 weights": Path(weights_path).expanduser().resolve(),
    }
    missing = [f"{label}={path}" for label, path in paths.items() if not path.exists()]
    package_marker = (
        paths["Hands23 repo"]
        / "hodetector"
        / "modeling"
        / "roi_heads"
        / "__init__.py"
    )
    if not package_marker.is_file():
        missing.append(
            "Hands23 source package="
            f"{package_marker} (repo_path must point to the full Hands23 checkout, "
            "not a weights-only directory)"
        )
    detector_config = paths["Hands23 config"]
    if detector_config.is_file():
        try:
            detector_payload = yaml.safe_load(
                detector_config.read_text(encoding="utf-8")
            ) or {}
        except (OSError, yaml.YAMLError) as exc:
            missing.append(f"Hands23 config cannot be parsed={detector_config}: {exc}")
        else:
            base_value = (
                detector_payload.get("_BASE_")
                if isinstance(detector_payload, dict)
                else None
            )
            if base_value:
                base_path = (detector_config.parent / str(base_value)).resolve()
                if not base_path.is_file():
                    missing.append(f"Hands23 base config={base_path}")
                else:
                    paths["Hands23 base config"] = base_path
    if not paths["Hands23 Python"].is_file():
        missing.append(f"Hands23 Python is not a file={paths['Hands23 Python']}")
    elif not os.access(paths["Hands23 Python"], os.X_OK):
        missing.append(f"Hands23 Python is not executable={paths['Hands23 Python']}")
    if missing:
        raise FileNotFoundError(", ".join(missing))
    paths["Hands23 source package"] = package_marker
    return paths


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = int(size)
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("Hands23 socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_packet(connection: socket.socket, header: dict[str, Any], payload: bytes = b"") -> None:
    message = dict(header)
    message["payload_size"] = len(payload)
    encoded = json.dumps(message, separators=(",", ":")).encode("utf-8")
    if len(encoded) > _MAX_HEADER_BYTES:
        raise Hands23IPCError("Hands23 packet header is too large")
    connection.sendall(_HEADER.pack(len(encoded)) + encoded + payload)


def receive_packet(connection: socket.socket) -> tuple[dict[str, Any], bytes]:
    header_size = _HEADER.unpack(_recv_exact(connection, _HEADER.size))[0]
    if header_size <= 0 or header_size > _MAX_HEADER_BYTES:
        raise Hands23IPCError(f"invalid Hands23 header size: {header_size}")
    header = json.loads(_recv_exact(connection, header_size).decode("utf-8"))
    payload_size = int(header.pop("payload_size", 0))
    if payload_size < 0:
        raise Hands23IPCError(f"invalid Hands23 payload size: {payload_size}")
    return header, _recv_exact(connection, payload_size) if payload_size else b""


@dataclass(frozen=True)
class Hands23BBoxResult:
    task_id: int
    frame_seq: int
    camera_id: int
    capture_time_s: float
    valid: bool
    bbox_xyxy: tuple[float, float, float, float]
    hand_side: str
    hand_score: float
    object_score: float
    contact_state: str
    reason: str
    inference_ms: float
    roundtrip_ms: float


@dataclass
class _PendingFrame:
    task_id: int
    frame_seq: int
    camera_id: int
    capture_time_s: float
    image_bgr: np.ndarray
    submitted_monotonic_s: float


class Hands23SidecarClient:
    """Own the isolated predictor process and never queue more than one frame/camera."""

    def __init__(
        self,
        *,
        python_interpreter: str | Path,
        sidecar_script: str | Path,
        config_path: str | Path,
        repo_path: str | Path,
        detector_config_path: str | Path,
        weights_path: str | Path,
        socket_path: str | Path,
        startup_timeout_s: float = 60.0,
        request_timeout_s: float = 10.0,
        max_input_hz: float = 10.0,
    ) -> None:
        self.python_interpreter = Path(python_interpreter).expanduser().resolve()
        self.sidecar_script = Path(sidecar_script).expanduser().resolve()
        self.config_path = Path(config_path).expanduser().resolve()
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.detector_config_path = Path(detector_config_path).expanduser().resolve()
        self.weights_path = Path(weights_path).expanduser().resolve()
        self.socket_path = Path(socket_path).expanduser()
        self.startup_timeout_s = max(float(startup_timeout_s), 0.1)
        self.request_timeout_s = max(float(request_timeout_s), 0.1)
        self.input_period_s = 0.0 if float(max_input_hz) <= 0.0 else 1.0 / float(max_input_hz)

        self._condition = threading.Condition()
        self._pending: dict[int, _PendingFrame] = {}
        self._results: list[Hands23BBoxResult] = []
        self._last_submit_at = {0: float("-inf"), 1: float("-inf")}
        self._task_id = 0
        self._stopping = False
        self._process: subprocess.Popen | None = None
        self._connection: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._socket_inode: int | None = None
        self.last_error: str | None = None

    @property
    def healthy(self) -> bool:
        process = self._process
        thread = self._thread
        return bool(
            not self._stopping
            and process is not None
            and process.poll() is None
            and thread is not None
            and thread.is_alive()
            and self.last_error is None
        )

    def _environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        env_root = self.python_interpreter.parents[1]
        environment["PYTHONNOUSERSITE"] = "1"
        env_lib = env_root / "lib"
        if env_lib.is_dir():
            existing = environment.get("LD_LIBRARY_PATH", "")
            environment["LD_LIBRARY_PATH"] = f"{env_lib}:{existing}" if existing else str(env_lib)
        return environment

    def _prepare_socket_path(self) -> None:
        """Remove only a stale Unix socket; never replace files or listeners."""
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.socket_path.exists():
            return
        if not stat.S_ISSOCK(self.socket_path.lstat().st_mode):
            raise Hands23IPCError(
                f"refusing to replace non-socket path: {self.socket_path}"
            )
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(0.25)
        try:
            probe.connect(str(self.socket_path))
        except (ConnectionRefusedError, socket.timeout, OSError):
            self.socket_path.unlink()
        else:
            raise Hands23IPCError(
                f"a Hands23 sidecar is already listening at {self.socket_path}"
            )
        finally:
            probe.close()

    def start(self) -> None:
        validate_hands23_assets(
            python_interpreter=self.python_interpreter,
            sidecar_script=self.sidecar_script,
            config_path=self.config_path,
            repo_path=self.repo_path,
            detector_config_path=self.detector_config_path,
            weights_path=self.weights_path,
        )
        self._prepare_socket_path()
        command = [
            str(self.python_interpreter),
            str(self.sidecar_script),
            "--config", str(self.config_path),
            "--repo", str(self.repo_path),
            "--detector-config", str(self.detector_config_path),
            "--weights", str(self.weights_path),
            "--socket", str(self.socket_path),
        ]
        connection: socket.socket | None = None
        try:
            self._process = subprocess.Popen(command, env=self._environment())
            deadline = time.monotonic() + self.startup_timeout_s
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            while True:
                if self._process.poll() is not None:
                    return_code = self._process.returncode
                    raise Hands23IPCError(
                        f"Hands23 sidecar exited during startup with code {return_code}"
                    )
                try:
                    connection.connect(str(self.socket_path))
                    self._socket_inode = self.socket_path.stat().st_ino
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Hands23 sidecar did not become ready within {self.startup_timeout_s:.1f}s"
                        )
                    time.sleep(0.05)
            connection.settimeout(self.request_timeout_s)
            ready, _ = receive_packet(connection)
            if ready.get("type") != "ready":
                raise Hands23IPCError(f"unexpected Hands23 startup response: {ready}")
            self._connection = connection
            connection = None
            self._thread = threading.Thread(
                target=self._run,
                name="hands23-ipc",
                daemon=True,
            )
            self._thread.start()
        except BaseException:
            if connection is not None:
                connection.close()
            self.close()
            raise

    def reset(self, task_id: int) -> None:
        with self._condition:
            self._task_id = int(task_id)
            self._pending.clear()
            self._results.clear()
            self._last_submit_at = {0: float("-inf"), 1: float("-inf")}

    def submit(
        self,
        camera_id: int,
        image_bgr: np.ndarray,
        *,
        task_id: int,
        frame_seq: int,
        capture_time_s: float,
        now_monotonic_s: float | None = None,
    ) -> bool:
        camera_id = int(camera_id)
        if camera_id not in (0, 1) or not self.healthy or int(task_id) != self._task_id:
            return False
        now = time.monotonic() if now_monotonic_s is None else float(now_monotonic_s)
        with self._condition:
            if now - self._last_submit_at[camera_id] < self.input_period_s:
                return False
            self._last_submit_at[camera_id] = now
            self._pending[camera_id] = _PendingFrame(
                task_id=int(task_id),
                frame_seq=int(frame_seq),
                camera_id=camera_id,
                capture_time_s=float(capture_time_s),
                image_bgr=np.ascontiguousarray(image_bgr, dtype=np.uint8).copy(),
                submitted_monotonic_s=now,
            )
            self._condition.notify()
        return True

    def drain_results(self) -> list[Hands23BBoxResult]:
        with self._condition:
            results = list(self._results)
            self._results.clear()
            return results

    def _next_frame(self) -> _PendingFrame | None:
        with self._condition:
            self._condition.wait_for(lambda: self._stopping or bool(self._pending))
            if self._stopping:
                return None
            camera_id = min(self._pending, key=lambda key: self._pending[key].frame_seq)
            return self._pending.pop(camera_id)

    def _run(self) -> None:
        connection = self._connection
        if connection is None:
            return
        try:
            while True:
                frame = self._next_frame()
                if frame is None:
                    return
                height, width = frame.image_bgr.shape[:2]
                send_packet(
                    connection,
                    {
                        "type": "frame",
                        "task_id": frame.task_id,
                        "frame_seq": frame.frame_seq,
                        "camera_id": frame.camera_id,
                        "capture_time_s": frame.capture_time_s,
                        "height": height,
                        "width": width,
                        "channels": 3,
                        "dtype": "uint8",
                    },
                    frame.image_bgr.tobytes(order="C"),
                )
                response, _ = receive_packet(connection)
                if response.get("type") != "bbox":
                    raise Hands23IPCError(f"unexpected Hands23 response: {response}")
                bbox = tuple(float(value) for value in response.get("bbox_xyxy", ()))
                if len(bbox) != 4:
                    bbox = (0.0, 0.0, 0.0, 0.0)
                result = Hands23BBoxResult(
                    task_id=int(response["task_id"]),
                    frame_seq=int(response["frame_seq"]),
                    camera_id=int(response["camera_id"]),
                    capture_time_s=float(response["capture_time_s"]),
                    valid=bool(response.get("valid", False)),
                    bbox_xyxy=bbox,
                    hand_side=str(response.get("hand_side", "")),
                    hand_score=float(response.get("hand_score", 0.0)),
                    object_score=float(response.get("object_score", 0.0)),
                    contact_state=str(response.get("contact_state", "")),
                    reason=str(response.get("reason", "")),
                    inference_ms=float(response.get("inference_ms", 0.0)),
                    roundtrip_ms=max(
                        (time.monotonic() - frame.submitted_monotonic_s) * 1000.0,
                        0.0,
                    ),
                )
                with self._condition:
                    if result.task_id == self._task_id:
                        self._results.append(result)
        except Exception as exc:
            if not self._stopping:
                self.last_error = f"{type(exc).__name__}: {exc}"
                with self._condition:
                    self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        if self._connection is not None:
            try:
                self._connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._connection.close()
            self._connection = None
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2.0)
        self._process = None
        try:
            if (
                self._socket_inode is not None
                and self.socket_path.exists()
                and self.socket_path.stat().st_ino == self._socket_inode
            ):
                self.socket_path.unlink()
        except FileNotFoundError:
            pass
        self._socket_inode = None


__all__ = [
    "Hands23BBoxResult",
    "Hands23IPCError",
    "Hands23SidecarClient",
    "receive_packet",
    "send_packet",
    "validate_hands23_assets",
]
