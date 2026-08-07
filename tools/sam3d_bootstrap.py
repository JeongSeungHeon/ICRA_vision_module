#!/usr/bin/env python3
"""Run capture and SAM3D generation in isolated Python environments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

if __package__:
    from .sam3d_ipc import (
        Sam3DIPCError,
        build_model_spec,
        generate,
        wait_until_ready,
    )
else:
    from sam3d_ipc import (
        Sam3DIPCError,
        build_model_spec,
        generate,
        wait_until_ready,
    )


def _environment_for(interpreter: Path) -> dict[str, str]:
    environment = dict(os.environ)
    env_root = interpreter.resolve().parents[1]
    environment["CONDA_PREFIX"] = str(env_root)
    environment["CUDA_HOME"] = str(env_root)
    environment["LIDRA_SKIP_INIT"] = "true"
    env_lib = env_root / "lib"
    if env_lib.is_dir():
        existing = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = f"{env_lib}:{existing}" if existing else str(env_lib)
    return environment


def _run(command: list[str], *, cwd: Path, environment: dict[str, str], stage: str) -> None:
    print(f"[sam3d_bootstrap] starting {stage}", flush=True)
    result = subprocess.run(command, cwd=str(cwd), env=environment, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{stage} failed with exit code {result.returncode}")
    print(f"[sam3d_bootstrap] completed {stage}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--capture-python", required=True)
    parser.add_argument("--sam3d-python", required=True)
    parser.add_argument("--sam3d-repo", required=True)
    parser.add_argument(
        "--execution-mode",
        choices=("server", "oneshot"),
        default="server",
    )
    parser.add_argument("--server-socket", default="/tmp/handover_sam3d_stage1.sock")
    parser.add_argument("--server-ready-timeout-s", type=float, default=120.0)
    parser.add_argument("--request-timeout-s", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    capture_python = Path(args.capture_python).expanduser().resolve()
    sam3d_python = Path(args.sam3d_python).expanduser().resolve()
    sam3d_repo = Path(args.sam3d_repo).expanduser()
    if not sam3d_repo.is_absolute():
        sam3d_repo = repo_root / sam3d_repo
    sam3d_repo = sam3d_repo.resolve()

    required_paths = [
        (repo_root, "repo root"),
        (capture_python, "capture Python"),
        (sam3d_repo, "SAM3D repo"),
    ]
    if args.execution_mode == "oneshot":
        required_paths.append((sam3d_python, "SAM3D Python"))
    for path, description in required_paths:
        if not path.exists():
            raise SystemExit(f"{description} does not exist: {path}")

    artifact_dir = Path(args.artifact_dir).expanduser()
    if not artifact_dir.is_absolute():
        artifact_dir = repo_root / artifact_dir
    artifact_dir = artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=False)

    common = [
        "--config",
        str(Path(args.config).expanduser().resolve()),
        "--artifact-dir",
        str(artifact_dir),
        "--repo-root",
        str(repo_root),
    ]
    generation_command = [
        str(sam3d_python),
        str(repo_root / "tools" / "sam3d_generate.py"),
        *common,
        "--sam3d-repo",
        str(sam3d_repo),
    ]
    model_spec = None
    if args.execution_mode == "server":
        model_spec = build_model_spec(args.config, sam3d_repo)
        print(
            "[sam3d_bootstrap] checking persistent SAM3D server: "
            f"socket={args.server_socket}",
            flush=True,
        )
        server_status = wait_until_ready(
            args.server_socket,
            expected_signature=model_spec["signature"],
            ready_timeout_s=float(args.server_ready_timeout_s),
        )
        print(
            "[sam3d_bootstrap] persistent SAM3D server ready: "
            f"instance={server_status['server_instance_id']} "
            f"load={float(server_status['model_load_elapsed_s']):.2f}s",
            flush=True,
        )
    else:
        _run(
            [*generation_command, "--preflight"],
            cwd=repo_root,
            environment=_environment_for(sam3d_python),
            stage="SAM3D preflight",
        )
    _run(
        [str(capture_python), str(repo_root / "tools" / "sam3d_capture.py"), *common],
        cwd=repo_root,
        environment=_environment_for(capture_python),
        stage="cam0 FastSAM capture",
    )
    if args.execution_mode == "server":
        print("[sam3d_bootstrap] starting SAM3D server request", flush=True)
        response = generate(
            args.server_socket,
            config_path=args.config,
            artifact_dir=artifact_dir,
            expected_signature=model_spec["signature"],
            timeout_s=float(args.request_timeout_s),
        )
        print(
            "[sam3d_bootstrap] completed SAM3D server request: "
            f"instance={response['server_instance_id']} "
            f"elapsed={float(response['request_elapsed_s']):.2f}s",
            flush=True,
        )
    else:
        _run(
            generation_command,
            cwd=repo_root,
            environment=_environment_for(sam3d_python),
            stage="SAM3D stage1 generation",
        )
    template_path = artifact_dir / "template.npy"
    if not template_path.is_file():
        raise RuntimeError(f"SAM3D generation completed without template: {template_path}")
    print(f"[sam3d_bootstrap] ready: {template_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Sam3DIPCError as exc:
        print(f"[sam3d_bootstrap] ERROR: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
