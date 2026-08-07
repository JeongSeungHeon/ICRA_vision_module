"""Helpers for importing the existing non-ROS Python runtime.

All ROS2 code lives under ./ros2, but it intentionally reuses the current
repository modules by adding the repository root to sys.path at runtime.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT_ENV = "ICRA_VISION_REPO_ROOT"
ROOT_SENTINEL = "robot_control_rtde_fitting_final.py"


def infer_repo_root(explicit: str | Path | None = None) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not (root / ROOT_SENTINEL).exists():
            raise FileNotFoundError(f"{root} does not look like the ICRA vision repo root.")
        return root

    env_value = os.environ.get(REPO_ROOT_ENV)
    if env_value:
        return infer_repo_root(env_value)

    candidates: list[Path] = []
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, cwd.parent])

    module_path = Path(__file__).resolve()
    candidates.extend(module_path.parents)
    candidates.extend(parent.parent for parent in module_path.parents if parent.name == "ros2")

    for candidate in candidates:
        if (candidate / ROOT_SENTINEL).exists():
            return candidate

    raise FileNotFoundError(
        "Could not infer the ICRA vision repo root. Pass the repo_root ROS parameter "
        f"or set {REPO_ROOT_ENV}."
    )


def ensure_repo_on_path(repo_root: str | Path | None = None) -> Path:
    root = infer_repo_root(repo_root)
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root


def resolve_repo_path(repo_root: str | Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(repo_root).expanduser().resolve() / path).resolve()
