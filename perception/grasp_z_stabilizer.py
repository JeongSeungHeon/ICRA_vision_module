"""Post-offset grasp-point z stabilization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_CONFIG_PATH = Path("configs/handover.yaml")


@dataclass
class GraspPointZStabilizerDebug:
    enabled: bool
    active: bool
    dist_xy_m: float | None = None
    raw_z_m: float | None = None
    limited_z_m: float | None = None
    filtered_z_m: float | None = None
    previous_filtered_z_m: float | None = None
    reset_reason: str | None = None


class GraspPointZStabilizer:
    """Clamp and smooth grasp-point z only in close object/grasp XY range."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        grasp_cfg = (config or {}).get("grasp", {})
        stabilizer_cfg = grasp_cfg.get("z_stabilization", {})
        self.enabled = bool(stabilizer_cfg.get("enabled", True))
        self.activation_dist_xy_m = float(stabilizer_cfg.get("activation_dist_xy_m", 0.065))
        self.max_step_z_m = max(float(stabilizer_cfg.get("max_step_z_m", 0.005)), 0.0)
        self.ema_alpha = float(np.clip(float(stabilizer_cfg.get("ema_alpha", 0.70)), 0.0, 1.0))

        self._previous_filtered_z: float | None = None
        self._was_active = False
        self.last_debug = GraspPointZStabilizerDebug(enabled=self.enabled, active=False)

    @classmethod
    def from_config(cls, config_path: str | Path = DEFAULT_CONFIG_PATH) -> "GraspPointZStabilizer":
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        return cls(config=config)

    def reset(self) -> None:
        self._previous_filtered_z = None
        self._was_active = False
        self.last_debug = GraspPointZStabilizerDebug(
            enabled=self.enabled,
            active=False,
            reset_reason="reset",
        )

    def process(
        self,
        measured_grasp_point_base,
        *,
        reference_xyz_mm=None,
        eef_xyz_mm=None,
    ) -> tuple[float, float, float] | None:
        """Return raw x/y plus stabilized z when reference target is close to EEF."""

        if measured_grasp_point_base is None:
            self._reset_history(reason="no_grasp_point")
            return None

        grasp_point = np.asarray(measured_grasp_point_base, dtype=np.float32).reshape(3)
        if not np.all(np.isfinite(grasp_point)):
            self._clear_history_with_raw_debug(grasp_point, reason="non_finite_grasp_point")
            return tuple(float(v) for v in grasp_point)

        raw_z = float(grasp_point[2])
        if not self.enabled:
            self._set_inactive_baseline(raw_z, reason="disabled")
            return tuple(float(v) for v in grasp_point)

        if reference_xyz_mm is None:
            self._clear_history_with_raw_debug(grasp_point, reason="no_reference_target")
            return tuple(float(v) for v in grasp_point)
        if eef_xyz_mm is None:
            self._clear_history_with_raw_debug(grasp_point, reason="no_eef_position")
            return tuple(float(v) for v in grasp_point)

        reference_xyz_mm = np.asarray(reference_xyz_mm, dtype=np.float32).reshape(3)
        eef_xyz_mm = np.asarray(eef_xyz_mm, dtype=np.float32).reshape(3)
        if not np.all(np.isfinite(reference_xyz_mm)):
            self._clear_history_with_raw_debug(grasp_point, reason="non_finite_reference_target")
            return tuple(float(v) for v in grasp_point)
        if not np.all(np.isfinite(eef_xyz_mm)):
            self._clear_history_with_raw_debug(grasp_point, reason="non_finite_eef_position")
            return tuple(float(v) for v in grasp_point)

        dist_xy = float(np.linalg.norm(reference_xyz_mm[:2] - eef_xyz_mm[:2]))
        dist_xy_m = dist_xy / 1000.0
        if dist_xy_m > self.activation_dist_xy_m:
            self._previous_filtered_z = raw_z
            self._was_active = False
            self.last_debug = GraspPointZStabilizerDebug(
                enabled=self.enabled,
                active=False,
                dist_xy_m=dist_xy_m,
                raw_z_m=raw_z,
                limited_z_m=raw_z,
                filtered_z_m=raw_z,
                previous_filtered_z_m=raw_z,
                reset_reason="outside_activation_distance",
            )
            return tuple(float(v) for v in grasp_point)

        previous_filtered_z = self._previous_filtered_z
        if previous_filtered_z is None or not self._was_active:
            self._previous_filtered_z = raw_z
            self._was_active = True
            self.last_debug = GraspPointZStabilizerDebug(
                enabled=self.enabled,
                active=True,
                dist_xy_m=dist_xy_m,
                raw_z_m=raw_z,
                limited_z_m=raw_z,
                filtered_z_m=raw_z,
                previous_filtered_z_m=previous_filtered_z,
                reset_reason="initialized",
            )
            return tuple(float(v) for v in grasp_point)

        delta_z = raw_z - previous_filtered_z
        if abs(delta_z) > self.max_step_z_m:
            limited_z = previous_filtered_z + float(np.sign(delta_z)) * self.max_step_z_m
        else:
            limited_z = raw_z
        filtered_z = (self.ema_alpha * previous_filtered_z) + ((1.0 - self.ema_alpha) * limited_z)

        self._previous_filtered_z = float(filtered_z)
        self._was_active = True
        self.last_debug = GraspPointZStabilizerDebug(
            enabled=self.enabled,
            active=True,
            dist_xy_m=dist_xy_m,
            raw_z_m=raw_z,
            limited_z_m=float(limited_z),
            filtered_z_m=float(filtered_z),
            previous_filtered_z_m=float(previous_filtered_z),
        )

        stabilized = grasp_point.copy()
        stabilized[2] = float(filtered_z)
        return tuple(float(v) for v in stabilized)

    def _reset_history(self, *, reason: str) -> None:
        self._previous_filtered_z = None
        self._was_active = False
        self.last_debug = GraspPointZStabilizerDebug(
            enabled=self.enabled,
            active=False,
            reset_reason=reason,
        )

    def _clear_history_with_raw_debug(self, grasp_point: np.ndarray, *, reason: str) -> None:
        self._previous_filtered_z = None
        self._was_active = False
        raw_z = None
        if grasp_point.size >= 3 and np.isfinite(grasp_point[2]):
            raw_z = float(grasp_point[2])
        self.last_debug = GraspPointZStabilizerDebug(
            enabled=self.enabled,
            active=False,
            raw_z_m=raw_z,
            limited_z_m=raw_z,
            filtered_z_m=raw_z,
            previous_filtered_z_m=None,
            reset_reason=reason,
        )

    def _set_inactive_baseline(self, raw_z: float, *, reason: str) -> None:
        self._previous_filtered_z = float(raw_z)
        self._was_active = False
        self.last_debug = GraspPointZStabilizerDebug(
            enabled=self.enabled,
            active=False,
            raw_z_m=float(raw_z),
            limited_z_m=float(raw_z),
            filtered_z_m=float(raw_z),
            previous_filtered_z_m=float(raw_z),
            reset_reason=reason,
        )


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "GraspPointZStabilizer",
    "GraspPointZStabilizerDebug",
]
