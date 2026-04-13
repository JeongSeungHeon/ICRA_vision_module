"""CLI entry point for the CORSMAL s8 end-effector benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from benchmarks.s8_end_effector_utils import REPO_ROOT, load_s8_settings, run_s8_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CORSMAL s8 end-effector benchmark.")
    parser.add_argument("--config", default="configs/handover.yaml", help="Path to the main YAML config.")
    parser.add_argument("--robot-ip", default=None, help="Optional UR RTDE IP override.")
    parser.add_argument("--force-mock", action="store_true", help="Force mock RTDE mode for dry validation or rehearsal.")
    parser.add_argument(
        "--dry-run-score-only",
        action="store_true",
        help="Skip robot execution and recompute the validation summary from the existing submission CSV.",
    )
    parser.add_argument("--move-timeout-s", type=float, default=10.0, help="Maximum time to wait for any single leg.")
    parser.add_argument("--position-tolerance-m", type=float, default=0.01, help="Pose reach tolerance in meters.")
    return parser.parse_args()


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyYAML is required to load configs/handover.yaml for the s8 benchmark.") from exc

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_controller(config_path: Path, *, robot_ip: str | None = None, force_mock: bool = False) -> Any:
    try:
        from robot.rtde_controller import RtdeController
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("robot.rtde_controller could not be imported. Check the runtime environment.") from exc

    controller = RtdeController.from_config(config_path)
    if robot_ip:
        controller.robot_ip = str(robot_ip)
    # Match the fitting_final RTDE flow: when we pass the current TCP
    # orientation back into commands, treat it as rotvec rather than RPY.
    controller.fixed_orientation_format = "rotvec"
    if force_mock:
        controller.force_mock = True
    return controller


def main() -> None:
    args = parse_args()
    config_path = (REPO_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config).resolve()
    config = load_yaml_config(config_path)
    settings = load_s8_settings(
        config,
        repo_root=REPO_ROOT,
        config_path=config_path,
        move_timeout_s=args.move_timeout_s,
        position_tolerance_m=args.position_tolerance_m,
    )
    controller = build_controller(config_path, robot_ip=args.robot_ip, force_mock=args.force_mock)
    result = run_s8_benchmark(
        controller,
        settings,
        dry_run_score_only=bool(args.dry_run_score_only),
    )
    print(f"[s8] status={result.status}")
    print(f"[s8] manifest={result.manifest_path}")
    print(f"[s8] validation={result.validation_path}")
    if result.submission_csv_path is not None:
        print(f"[s8] submission_csv={result.submission_csv_path}")
    if result.metadata_json_path is not None:
        print(f"[s8] metadata_json={result.metadata_json_path}")


if __name__ == "__main__":
    main()
