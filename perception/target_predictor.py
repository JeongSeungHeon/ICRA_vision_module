"""Kalman-based short-horizon target prediction for follow control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PredictedTarget:
    xyz_mm: np.ndarray
    source: str
    prediction_age_s: float
    dt_s: float
    velocity_xy_mm_s: np.ndarray
    valid: bool


class TargetPredictor:
    """2D constant-velocity Kalman filter with held Z for short dropouts."""

    def __init__(
        self,
        *,
        process_noise_mm_s2: float = 800.0,
        measurement_noise_mm: float = 25.0,
        max_velocity_xy_mm_s: float = 200.0,
        reinit_jump_mm: float = 120.0,
    ) -> None:
        self.process_noise_mm_s2 = max(float(process_noise_mm_s2), 1e-6)
        self.measurement_noise_mm = max(float(measurement_noise_mm), 1e-6)
        self.max_velocity_xy_mm_s = max(float(max_velocity_xy_mm_s), 1e-6)
        self.reinit_jump_mm = max(float(reinit_jump_mm), 1e-6)
        self.reset()

    def reset(self) -> None:
        self._state: np.ndarray | None = None
        self._cov: np.ndarray | None = None
        self._last_timestamp_perf: float | None = None
        self._last_measurement_timestamp_perf: float | None = None
        self._last_z_mm: float | None = None

    def has_state(self) -> bool:
        return self._state is not None and self._cov is not None and self._last_timestamp_perf is not None

    def update(self, measured_xyz_mm: np.ndarray, timestamp_perf: float) -> None:
        measured = np.asarray(measured_xyz_mm, dtype=np.float32).reshape(3)
        measured_xy = measured[:2].astype(np.float64)
        measured_z = float(measured[2])
        timestamp_perf = float(timestamp_perf)

        if not np.all(np.isfinite(measured_xy)) or not np.isfinite(measured_z):
            return

        if not self.has_state():
            self._initialize_state(measured_xy, measured_z, timestamp_perf)
            return

        dt_s = self._sanitize_dt(timestamp_perf - float(self._last_timestamp_perf))
        self._predict_inplace(dt_s)

        assert self._state is not None
        assert self._cov is not None

        innovation = measured_xy - self._state[:2]
        if float(np.linalg.norm(innovation)) > self.reinit_jump_mm:
            self._initialize_state(measured_xy, measured_z, timestamp_perf)
            return

        h = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        r = np.eye(2, dtype=np.float64) * (self.measurement_noise_mm ** 2)
        s = h @ self._cov @ h.T + r
        k = self._cov @ h.T @ np.linalg.inv(s)

        self._state = self._state + (k @ innovation)
        identity = np.eye(4, dtype=np.float64)
        self._cov = (identity - (k @ h)) @ self._cov
        self._clip_velocity()

        self._last_timestamp_perf = timestamp_perf
        self._last_measurement_timestamp_perf = timestamp_perf
        self._last_z_mm = measured_z

    def predict(self, timestamp_perf: float) -> PredictedTarget | None:
        if not self.has_state():
            return None

        timestamp_perf = float(timestamp_perf)
        dt_s = self._sanitize_dt(timestamp_perf - float(self._last_timestamp_perf))
        self._predict_inplace(dt_s)

        assert self._state is not None

        prediction_age_s = 0.0
        if self._last_measurement_timestamp_perf is not None:
            prediction_age_s = max(timestamp_perf - float(self._last_measurement_timestamp_perf), 0.0)

        z_mm = float("nan") if self._last_z_mm is None else float(self._last_z_mm)
        xyz_mm = np.array([self._state[0], self._state[1], z_mm], dtype=np.float32)
        velocity_xy_mm_s = self._state[2:4].astype(np.float32).copy()
        valid = bool(np.all(np.isfinite(xyz_mm)) and np.all(np.isfinite(velocity_xy_mm_s)))
        return PredictedTarget(
            xyz_mm=xyz_mm,
            source="predicted",
            prediction_age_s=float(prediction_age_s),
            dt_s=float(dt_s),
            velocity_xy_mm_s=velocity_xy_mm_s,
            valid=valid,
        )

    def _initialize_state(self, measured_xy: np.ndarray, measured_z: float, timestamp_perf: float) -> None:
        self._state = np.array([measured_xy[0], measured_xy[1], 0.0, 0.0], dtype=np.float64)
        self._cov = np.diag(
            [
                self.measurement_noise_mm ** 2,
                self.measurement_noise_mm ** 2,
                self.max_velocity_xy_mm_s ** 2,
                self.max_velocity_xy_mm_s ** 2,
            ]
        ).astype(np.float64)
        self._last_timestamp_perf = float(timestamp_perf)
        self._last_measurement_timestamp_perf = float(timestamp_perf)
        self._last_z_mm = float(measured_z)

    def _predict_inplace(self, dt_s: float) -> None:
        if not self.has_state() or dt_s <= 0.0:
            return

        assert self._state is not None
        assert self._cov is not None

        f = np.array(
            [
                [1.0, 0.0, dt_s, 0.0],
                [0.0, 1.0, 0.0, dt_s],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dt2 = dt_s * dt_s
        dt3 = dt2 * dt_s
        dt4 = dt2 * dt2
        q = self.process_noise_mm_s2 ** 2
        q_matrix = q * np.array(
            [
                [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt3 / 2.0, 0.0, dt2],
            ],
            dtype=np.float64,
        )

        self._state = f @ self._state
        self._cov = (f @ self._cov @ f.T) + q_matrix
        self._clip_velocity()
        self._last_timestamp_perf = float(self._last_timestamp_perf) + float(dt_s)

    def _clip_velocity(self) -> None:
        if self._state is None:
            return
        velocity = self._state[2:4]
        speed = float(np.linalg.norm(velocity))
        if speed <= self.max_velocity_xy_mm_s or speed <= 1e-9:
            return
        self._state[2:4] = velocity * (self.max_velocity_xy_mm_s / speed)

    @staticmethod
    def _sanitize_dt(dt_s: float) -> float:
        if not np.isfinite(dt_s):
            return 0.0
        return float(min(max(dt_s, 0.0), 1.0))

