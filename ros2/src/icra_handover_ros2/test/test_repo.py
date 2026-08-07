from pathlib import Path

from icra_handover_ros2.repo import infer_repo_root, resolve_repo_path


def test_infer_repo_root_from_explicit_path():
    repo_root = Path(__file__).resolve().parents[4]

    inferred = infer_repo_root(repo_root)

    assert inferred == repo_root
    assert (inferred / "robot_control_rtde_fitting_final.py").exists()


def test_resolve_repo_path_keeps_absolute_paths():
    repo_root = Path(__file__).resolve().parents[4]
    absolute = repo_root / "configs" / "handover.yaml"

    assert resolve_repo_path(repo_root, absolute) == absolute


def test_resolve_repo_path_joins_relative_paths():
    repo_root = Path(__file__).resolve().parents[4]

    assert resolve_repo_path(repo_root, "configs/handover.yaml") == repo_root / "configs" / "handover.yaml"
