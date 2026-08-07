#!/usr/bin/env python3
"""Long-lived stage1-only SAM3D model server over a local Unix socket."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import socketserver
import stat
import sys
import threading
import time
import traceback
from typing import Callable
import uuid

import yaml

if __package__:
    from .sam3d_generate import Sam3DStage1Engine
    from .sam3d_ipc import (
        DEFAULT_SOCKET_PATH,
        MAX_MESSAGE_BYTES,
        PROTOCOL_VERSION,
        build_model_spec,
    )
else:
    from sam3d_generate import Sam3DStage1Engine
    from sam3d_ipc import (
        DEFAULT_SOCKET_PATH,
        MAX_MESSAGE_BYTES,
        PROTOCOL_VERSION,
        build_model_spec,
    )


def _response(*, success: bool, **payload) -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "success": bool(success),
        **payload,
    }


class Sam3DServerRuntime:
    """Thread-safe model lifecycle and single-request GPU ownership."""

    def __init__(
        self,
        *,
        config_path: Path,
        sam3d_repo: Path,
        engine_factory: Callable = Sam3DStage1Engine,
    ):
        self.config_path = Path(config_path).resolve()
        self.sam3d_repo = Path(sam3d_repo).resolve()
        self.model_spec = build_model_spec(self.config_path, self.sam3d_repo)
        self.instance_id = uuid.uuid4().hex
        self.engine_factory = engine_factory
        self.engine = None
        self.state = "loading"
        self.message = "model loading"
        self.model_load_elapsed_s = 0.0
        self._state_lock = threading.Lock()
        self._generation_lock = threading.Lock()

    def load_model(self) -> None:
        started = time.perf_counter()
        try:
            with open(self.config_path, "r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}
            engine = self.engine_factory(config, sam3d_repo=self.sam3d_repo)
            with self._state_lock:
                self.engine = engine
                self.model_load_elapsed_s = float(
                    getattr(
                        engine,
                        "model_load_elapsed_s",
                        time.perf_counter() - started,
                    )
                )
                self.state = "ready"
                self.message = "model ready"
            print(
                "[sam3d_model_server] model ready: "
                f"instance={self.instance_id} "
                f"load={self.model_load_elapsed_s:.2f}s "
                f"signature={self.model_spec['signature'][:12]}",
                flush=True,
            )
        except Exception as exc:
            with self._state_lock:
                self.model_load_elapsed_s = time.perf_counter() - started
                self.state = "error"
                self.message = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

    def status_response(self) -> dict:
        with self._state_lock:
            return _response(
                success=True,
                operation="status",
                state=self.state,
                message=self.message,
                server_instance_id=self.instance_id,
                model_signature=self.model_spec["signature"],
                model_load_elapsed_s=float(self.model_load_elapsed_s),
            )

    def generate_response(self, request: dict) -> dict:
        with self._state_lock:
            state = self.state
            engine = self.engine
            message = self.message
        if state != "ready" or engine is None:
            return _response(
                success=False,
                operation="generate",
                state=state,
                message=(
                    "SAM3D server is busy with another template request"
                    if state == "busy"
                    else message
                ),
            )
        if not self._generation_lock.acquire(blocking=False):
            return _response(
                success=False,
                operation="generate",
                state="busy",
                message="SAM3D server is busy with another template request",
            )

        try:
            with self._state_lock:
                self.state = "busy"
                self.message = "template generation in progress"
            config_path = Path(str(request.get("config_path", ""))).expanduser()
            artifact_dir = Path(str(request.get("artifact_dir", ""))).expanduser()
            if not config_path.is_absolute() or not artifact_dir.is_absolute():
                raise ValueError("config_path and artifact_dir must be absolute paths")
            config_path = config_path.resolve()
            artifact_dir = artifact_dir.resolve()
            expected_signature = str(
                request.get("expected_model_signature", "")
            )
            request_spec = build_model_spec(config_path, self.sam3d_repo)
            if expected_signature != self.model_spec["signature"]:
                raise ValueError(
                    "request model signature does not match the resident server"
                )
            if request_spec["signature"] != self.model_spec["signature"]:
                raise ValueError(
                    "request config/repo differs from the resident model; "
                    "restart the SAM3D model server"
                )
            with open(config_path, "r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}

            request_id = uuid.uuid4().hex
            print(
                "[sam3d_model_server] request start: "
                f"id={request_id} artifacts={artifact_dir}",
                flush=True,
            )
            result = engine.generate_template(
                config,
                artifact_dir=artifact_dir,
                server_metadata={
                    "model_reused": True,
                    "server_instance_id": self.instance_id,
                    "model_signature": self.model_spec["signature"],
                },
            )
            print(
                "[sam3d_model_server] request complete: "
                f"id={request_id} "
                f"elapsed={float(result['request_elapsed_s']):.2f}s "
                f"points={int(result['template_point_count'])}",
                flush=True,
            )
            return _response(
                success=True,
                operation="generate",
                state="ready",
                message="template generated",
                request_id=request_id,
                server_instance_id=self.instance_id,
                model_signature=self.model_spec["signature"],
                template_path=result["template_path"],
                template_point_count=int(result["template_point_count"]),
                model_load_elapsed_s=float(self.model_load_elapsed_s),
                inference_elapsed_s=float(result["inference_elapsed_s"]),
                export_elapsed_s=float(result["export_elapsed_s"]),
                request_elapsed_s=float(result["request_elapsed_s"]),
            )
        except Exception as exc:
            traceback.print_exc()
            return _response(
                success=False,
                operation="generate",
                state="ready",
                message=f"{type(exc).__name__}: {exc}",
            )
        finally:
            with self._state_lock:
                if self.state != "error":
                    self.state = "ready"
                    self.message = "model ready"
            self._generation_lock.release()


class _Sam3DRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline(MAX_MESSAGE_BYTES + 1)
        if len(raw) > MAX_MESSAGE_BYTES:
            self._send(
                _response(success=False, message="request exceeded size limit")
            )
            return
        try:
            request = json.loads(raw.decode("utf-8"))
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            if int(request.get("protocol_version", -1)) != PROTOCOL_VERSION:
                self._send(
                    _response(
                        success=False,
                        message=(
                            "protocol mismatch: "
                            f"server={PROTOCOL_VERSION}, "
                            f"client={request.get('protocol_version')}"
                        ),
                    )
                )
                return
            operation = str(request.get("operation", ""))
            runtime: Sam3DServerRuntime = self.server.runtime
            if operation == "status":
                response = runtime.status_response()
            elif operation == "generate":
                response = runtime.generate_response(request)
            else:
                response = _response(
                    success=False,
                    message=f"unsupported operation: {operation!r}",
                )
        except Exception as exc:
            response = _response(
                success=False,
                message=f"invalid request: {type(exc).__name__}: {exc}",
            )
        self._send(response)

    def _send(self, response: dict) -> None:
        encoded = (
            json.dumps(response, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
            + b"\n"
        )
        try:
            self.wfile.write(encoded)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


class Sam3DUnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, socket_path: Path, runtime: Sam3DServerRuntime):
        self.runtime = runtime
        super().__init__(str(socket_path), _Sam3DRequestHandler)


def _prepare_socket_path(socket_path: Path) -> None:
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if not socket_path.exists():
        return
    mode = socket_path.lstat().st_mode
    if not stat.S_ISSOCK(mode):
        raise RuntimeError(
            f"refusing to replace non-socket path: {socket_path}"
        )
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(0.25)
    try:
        probe.connect(str(socket_path))
    except (ConnectionRefusedError, socket.timeout, OSError):
        socket_path.unlink()
    else:
        raise RuntimeError(f"a SAM3D server is already listening at {socket_path}")
    finally:
        probe.close()


def serve(
    *,
    config_path: Path,
    sam3d_repo: Path,
    socket_path: Path,
    engine_factory: Callable = Sam3DStage1Engine,
) -> None:
    socket_path = Path(socket_path).expanduser()
    if not socket_path.is_absolute():
        raise ValueError(f"server socket path must be absolute: {socket_path}")
    _prepare_socket_path(socket_path)
    runtime = Sam3DServerRuntime(
        config_path=Path(config_path).expanduser().resolve(),
        sam3d_repo=Path(sam3d_repo).expanduser().resolve(),
        engine_factory=engine_factory,
    )
    server = Sam3DUnixServer(socket_path, runtime)
    os.chmod(socket_path, 0o600)
    socket_inode = socket_path.stat().st_ino

    def _request_shutdown(signum, _frame):
        print(
            f"[sam3d_model_server] signal {signum}; shutting down",
            flush=True,
        )
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, _request_shutdown)
    signal.signal(signal.SIGTERM, _request_shutdown)
    loader = threading.Thread(
        target=runtime.load_model,
        name="sam3d-model-loader",
        daemon=True,
    )
    loader.start()
    print(
        "[sam3d_model_server] listening: "
        f"socket={socket_path} instance={runtime.instance_id} "
        f"signature={runtime.model_spec['signature'][:12]}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        try:
            if socket_path.exists() and socket_path.stat().st_ino == socket_inode:
                socket_path.unlink()
        except FileNotFoundError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sam3d-repo", required=True)
    parser.add_argument("--socket", default=DEFAULT_SOCKET_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    serve(
        config_path=Path(args.config),
        sam3d_repo=Path(args.sam3d_repo),
        socket_path=Path(args.socket),
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
