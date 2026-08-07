"""Build argument objects compatible with robot_control_rtde_fitting_final.py."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .repo import ensure_repo_on_path, resolve_repo_path


def build_runtime_args(
    *,
    repo_root: str,
    config_path: str,
    overrides: dict[str, Any] | None = None,
) -> SimpleNamespace:
    """Return a namespace matching the existing script's argparse output."""
    ensure_repo_on_path(repo_root)
    from robot_control_rtde_fitting_final import apply_config_defaults, load_yaml_config

    overrides = dict(overrides or {})
    config = str(resolve_repo_path(repo_root, config_path))
    model = overrides.get("model", "yoloe-26l-seg.pt")
    model_path = Path(str(model))
    repo_model_path = resolve_repo_path(repo_root, str(model))
    if not model_path.is_absolute() and repo_model_path.exists():
        model = str(repo_model_path)

    args = SimpleNamespace(
        model=model,
        prompt=overrides.get("prompt", None),
        serial=None,
        width=int(overrides.get("width", 640)),
        height=int(overrides.get("height", 480)),
        fps=int(overrides.get("fps", 30)),
        imgsz=int(overrides.get("imgsz", 640)),
        conf=float(overrides.get("conf", 0.25)),
        iou=float(overrides.get("iou", 0.45)),
        max_det=int(overrides.get("max_det", 100)),
        device=overrides.get("device", None),
        classes=overrides.get("classes", None),
        select_mode=str(overrides.get("select_mode", "highest_score")),
        select_class=overrides.get("select_class", None),
        half=bool(overrides.get("half", False)),
        depth_max_m=float(overrides.get("depth_max_m", 1.5)),
        config=config,
        enable_follow=bool(overrides.get("enable_follow", False)),
        robot_ip=overrides.get("robot_ip", None),
        min_valid_count=int(overrides.get("min_valid_count", 3)),
        target_timeout_s=float(overrides.get("target_timeout_s", 0.5)),
        move_to_base=False,
        open_gripper=bool(overrides.get("open_gripper", False)),
        verbose_robot=bool(overrides.get("verbose_robot", False)),
        follow_z=overrides.get("follow_z", None),
        control_hz=overrides.get("control_hz", None),
        position_tolerance_m=float(overrides.get("position_tolerance_m", 0.01)),
        move_timeout_s=float(overrides.get("move_timeout_s", 10.0)),
        gripper_close_timeout_s=float(overrides.get("gripper_close_timeout_s", 2.0)),
        gripper_release_dwell_s=float(overrides.get("gripper_release_dwell_s", 0.5)),
        pre_release_descend_before_open=overrides.get("pre_release_descend_before_open", None),
        pre_release_descend_m=overrides.get("pre_release_descend_m", None),
        follow_handoff_timeout_s=float(overrides.get("follow_handoff_timeout_s", 5.0)),
        enable_target_prediction=bool(overrides.get("enable_target_prediction", True)),
        prediction_max_horizon_s=float(overrides.get("prediction_max_horizon_s", 0.25)),
        prediction_process_noise_mm_s2=float(overrides.get("prediction_process_noise_mm_s2", 800.0)),
        prediction_measurement_noise_mm=float(overrides.get("prediction_measurement_noise_mm", 25.0)),
        prediction_max_xy_speed_mm_s=float(overrides.get("prediction_max_xy_speed_mm_s", 200.0)),
        prediction_reinit_jump_mm=float(overrides.get("prediction_reinit_jump_mm", 120.0)),
        debug_3d=bool(overrides.get("debug_3d", False)),
        save_image=bool(overrides.get("save_image", False)),
        debug_3d_dir=str(overrides.get("debug_3d_dir", "output/debug_3d")),
        debug_3d_max_object_points=int(overrides.get("debug_3d_max_object_points", 8000)),
        debug_3d_max_template_points=int(overrides.get("debug_3d_max_template_points", 8000)),
        debug_3d_template_axes=bool(overrides.get("debug_3d_template_axes", True)),
        disable_debug_3d_recording=bool(overrides.get("disable_debug_3d_recording", True)),
        record_video=bool(overrides.get("record_video", False)),
        profile_runtime=bool(overrides.get("profile_runtime", False)),
        profile_dir=str(overrides.get("profile_dir", "output/runtime_profile")),
        profile_sample_interval_s=float(overrides.get("profile_sample_interval_s", 1.0)),
        profile_print_every_s=float(overrides.get("profile_print_every_s", 0.0)),
    )

    return apply_config_defaults(args, load_yaml_config(config))
