#!/usr/bin/env python3
"""Pure-Python IPC helpers shared by the standalone SAM3D server and bootstrap."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket
import time
from typing import Any

import yaml


PROTOCOL_VERSION = 1
DEFAULT_SOCKET_PATH = "/tmp/handover_sam3d_stage1.sock"
MAX_MESSAGE_BYTES = 1024 * 1024


class Sam3DIPCError(RuntimeError):
    """Base error for local SAM3D IPC failures."""


class Sam3DServerUnavailable(Sam3DIPCError):
    """The configured server socket cannot be reached."""


class Sam3DServerRejected(Sam3DIPCError):
    """The server returned a structured failure."""


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_checkpoint_config(config_path: str | Path, sam3d_repo: str | Path) -> Path:
    config = _load_yaml(Path(config_path).expanduser().resolve())
    sam_cfg = (
        config.get("perception", {})
        .get("shape_fitting", {})
        .get("sam3d", {})
        or {}
    )
    checkpoint_config = Path(
        sam_cfg.get("checkpoint_config", "checkpoints/hf/pipeline.yaml")
    ).expanduser()
    repo = Path(sam3d_repo).expanduser().resolve()
    if not checkpoint_config.is_absolute():
        checkpoint_config = repo / checkpoint_config
    return checkpoint_config.resolve()


def _add_file_content(hasher, label: str, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    hasher.update(label.encode("utf-8"))
    hasher.update(str(path.resolve()).encode("utf-8"))
    hasher.update(path.read_bytes())


def _add_large_file_identity(hasher, label: str, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    stat_result = path.stat()
    identity = {
        "label": label,
        "path": str(path.resolve()),
        "size": int(stat_result.st_size),
        "mtime_ns": int(stat_result.st_mtime_ns),
    }
    hasher.update(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def build_model_spec(
    config_path: str | Path,
    sam3d_repo: str | Path,
) -> dict[str, Any]:
    """Build a cheap signature for the exact resident stage1 implementation."""
    config_path = Path(config_path).expanduser().resolve()
    repo = Path(sam3d_repo).expanduser().resolve()
    config = _load_yaml(config_path)
    sam_cfg = (
        config.get("perception", {})
        .get("shape_fitting", {})
        .get("sam3d", {})
        or {}
    )
    checkpoint_config = resolve_checkpoint_config(config_path, repo)
    pipeline = _load_yaml(checkpoint_config)
    workspace = checkpoint_config.parent

    hasher = hashlib.sha256()
    hasher.update(f"sam3d-stage1-protocol:{PROTOCOL_VERSION}".encode("utf-8"))
    _add_file_content(hasher, "pipeline_config", checkpoint_config)
    for key in ("ss_generator_config_path", "ss_decoder_config_path"):
        relative = pipeline.get(key)
        if not relative:
            raise KeyError(f"SAM3D pipeline config is missing {key}")
        _add_file_content(hasher, key, (workspace / str(relative)).resolve())
    for key in ("ss_generator_ckpt_path", "ss_decoder_ckpt_path"):
        relative = pipeline.get(key)
        if not relative:
            raise KeyError(f"SAM3D pipeline config is missing {key}")
        _add_large_file_identity(hasher, key, (workspace / str(relative)).resolve())
    for relative in (
        "notebook/inference.py",
        "sam3d_objects/pipeline/inference_pipeline.py",
        "sam3d_objects/pipeline/inference_pipeline_pointmap.py",
    ):
        _add_file_content(hasher, f"implementation:{relative}", repo / relative)
    hasher.update(
        json.dumps(
            {
                "compile": bool(sam_cfg.get("compile", False)),
                "stage1_only_init": True,
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    return {
        "signature": hasher.hexdigest(),
        "checkpoint_config": str(checkpoint_config),
        "sam3d_repo": str(repo),
        "compile": bool(sam_cfg.get("compile", False)),
        "stage1_only_init": True,
    }


def request(
    socket_path: str | Path,
    payload: dict[str, Any],
    *,
    timeout_s: float,
) -> dict[str, Any]:
    path = Path(socket_path).expanduser()
    message = {
        "protocol_version": PROTOCOL_VERSION,
        **payload,
    }
    encoded = (
        json.dumps(message, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )
    if len(encoded) > MAX_MESSAGE_BYTES:
        raise Sam3DIPCError(f"SAM3D request is too large: {len(encoded)} bytes")

    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(float(timeout_s))
    try:
        client.connect(str(path))
        client.sendall(encoded)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = client.recv(min(65536, MAX_MESSAGE_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_MESSAGE_BYTES:
                raise Sam3DIPCError("SAM3D server response exceeded size limit")
            if b"\n" in chunk:
                break
    except (FileNotFoundError, ConnectionRefusedError, socket.timeout, OSError) as exc:
        raise Sam3DServerUnavailable(
            f"cannot reach SAM3D server at {path}: {exc}"
        ) from exc
    finally:
        client.close()

    raw = b"".join(chunks).split(b"\n", 1)[0]
    if not raw:
        raise Sam3DIPCError("SAM3D server returned an empty response")
    try:
        response = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Sam3DIPCError(f"invalid SAM3D server response: {exc}") from exc
    if not isinstance(response, dict):
        raise Sam3DIPCError("SAM3D server response must be a JSON object")
    if int(response.get("protocol_version", -1)) != PROTOCOL_VERSION:
        raise Sam3DServerRejected(
            "SAM3D server protocol mismatch: "
            f"client={PROTOCOL_VERSION}, server={response.get('protocol_version')}"
        )
    if not bool(response.get("success", False)):
        raise Sam3DServerRejected(
            str(response.get("message", "SAM3D server rejected the request"))
        )
    return response


def status(socket_path: str | Path, *, timeout_s: float = 2.0) -> dict[str, Any]:
    return request(socket_path, {"operation": "status"}, timeout_s=timeout_s)


def wait_until_ready(
    socket_path: str | Path,
    *,
    expected_signature: str,
    ready_timeout_s: float,
    poll_interval_s: float = 0.25,
) -> dict[str, Any]:
    path = Path(socket_path).expanduser()
    if not path.exists():
        raise Sam3DServerUnavailable(
            f"SAM3D server socket does not exist: {path}. "
            "Start run_sam3d_model_server.sh first."
        )

    deadline = time.monotonic() + float(ready_timeout_s)
    while True:
        response = status(path)
        actual_signature = str(response.get("model_signature", ""))
        if actual_signature != str(expected_signature):
            raise Sam3DServerRejected(
                "SAM3D server model signature mismatch; restart the model server "
                "with the same config and repo as the standalone application"
            )
        state = str(response.get("state", "error"))
        if state == "ready":
            return response
        if state == "error":
            raise Sam3DServerRejected(
                f"SAM3D server failed during model load: {response.get('message', '')}"
            )
        if state == "busy":
            raise Sam3DServerRejected(
                "SAM3D server is busy with another template request"
            )
        if state != "loading":
            raise Sam3DServerRejected(f"unknown SAM3D server state: {state}")
        if time.monotonic() >= deadline:
            raise Sam3DServerRejected(
                f"SAM3D server did not become ready within {ready_timeout_s:.1f}s"
            )
        time.sleep(min(float(poll_interval_s), max(0.0, deadline - time.monotonic())))


def generate(
    socket_path: str | Path,
    *,
    config_path: str | Path,
    artifact_dir: str | Path,
    expected_signature: str,
    timeout_s: float,
) -> dict[str, Any]:
    return request(
        socket_path,
        {
            "operation": "generate",
            "config_path": str(Path(config_path).expanduser().resolve()),
            "artifact_dir": str(Path(artifact_dir).expanduser().resolve()),
            "expected_model_signature": str(expected_signature),
        },
        timeout_s=timeout_s,
    )


__all__ = [
    "DEFAULT_SOCKET_PATH",
    "MAX_MESSAGE_BYTES",
    "PROTOCOL_VERSION",
    "Sam3DIPCError",
    "Sam3DServerRejected",
    "Sam3DServerUnavailable",
    "build_model_spec",
    "generate",
    "request",
    "resolve_checkpoint_config",
    "status",
    "wait_until_ready",
]
