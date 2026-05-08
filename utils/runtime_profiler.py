"""Lightweight runtime profiling for live robot/perception loops."""

from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


def _timestamp_for_filename() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S%f")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _read_proc_stat_total() -> tuple[int, int] | None:
    try:
        with open("/proc/stat", "r", encoding="utf-8") as handle:
            line = handle.readline()
    except OSError:
        return None
    parts = line.split()
    if not parts or parts[0] != "cpu":
        return None
    values = [int(float(value)) for value in parts[1:]]
    if not values:
        return None
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    total = sum(values)
    return total, idle


def _read_process_cpu_ticks() -> int | None:
    try:
        with open("/proc/self/stat", "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return None
    end_comm = text.rfind(")")
    if end_comm < 0:
        return None
    fields = text[end_comm + 2 :].split()
    if len(fields) < 15:
        return None
    try:
        return int(fields[11]) + int(fields[12])
    except (TypeError, ValueError):
        return None


def _read_process_memory_mb() -> dict[str, float | None]:
    page_size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
    result = {"process_rss_mb": None, "process_vms_mb": None}
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as handle:
            fields = handle.read().split()
    except OSError:
        return result
    try:
        if len(fields) >= 1:
            result["process_vms_mb"] = int(fields[0]) * page_size / (1024.0 * 1024.0)
        if len(fields) >= 2:
            result["process_rss_mb"] = int(fields[1]) * page_size / (1024.0 * 1024.0)
    except (TypeError, ValueError):
        return result
    return result


def _read_meminfo_mb() -> dict[str, float | None]:
    keys = {
        "MemTotal": "system_mem_total_mb",
        "MemAvailable": "system_mem_available_mb",
    }
    result = {value: None for value in keys.values()}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError:
        return result
    for line in lines:
        name, _, raw_value = line.partition(":")
        if name not in keys:
            continue
        parts = raw_value.strip().split()
        if not parts:
            continue
        try:
            result[keys[name]] = float(parts[0]) / 1024.0
        except (TypeError, ValueError):
            pass
    total = result.get("system_mem_total_mb")
    available = result.get("system_mem_available_mb")
    if total and available is not None:
        result["system_mem_used_percent"] = 100.0 * max(total - available, 0.0) / total
    else:
        result["system_mem_used_percent"] = None
    return result


def _percentiles(values: list[float]) -> dict[str, float | None]:
    finite_values = sorted(float(value) for value in values if value is not None)
    if not finite_values:
        return {"p50": None, "p90": None, "p95": None, "max": None, "mean": None, "min": None}

    def pick(percent: float) -> float:
        if len(finite_values) == 1:
            return finite_values[0]
        rank = (len(finite_values) - 1) * percent / 100.0
        lower = int(rank)
        upper = min(lower + 1, len(finite_values) - 1)
        weight = rank - lower
        return finite_values[lower] * (1.0 - weight) + finite_values[upper] * weight

    return {
        "p50": pick(50.0),
        "p90": pick(90.0),
        "p95": pick(95.0),
        "max": finite_values[-1],
        "mean": sum(finite_values) / len(finite_values),
        "min": finite_values[0],
    }


@dataclass
class RuntimeProfiler:
    """Collects loop timing and resource samples until explicitly saved."""

    enabled: bool = False
    output_dir: str | Path = "output/runtime_profile"
    sample_interval_s: float = 1.0
    print_every_s: float = 5.0
    run_context: dict[str, Any] | None = None
    nvidia_smi_path: str | None = None
    session_index: int = 0
    frames: list[dict[str, Any]] = field(default_factory=list)
    resources: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.sample_interval_s = max(float(self.sample_interval_s), 0.05)
        self.print_every_s = max(float(self.print_every_s), 0.0)
        self.run_context = dict(self.run_context or {})
        self.nvidia_smi_path = self.nvidia_smi_path or shutil.which("nvidia-smi")
        self._current_frame: dict[str, Any] | None = None
        self._current_stage_ms: dict[str, float] = {}
        self._session_started_wall = time.time()
        self._session_started_perf = time.perf_counter()
        self._last_resource_perf: float | None = None
        self._last_print_perf: float | None = None
        self._last_proc_ticks = _read_process_cpu_ticks()
        self._last_cpu_total_idle = _read_proc_stat_total()
        self._clock_ticks = os.sysconf(os.sysconf_names.get("SC_CLK_TCK", "SC_CLK_TCK")) if hasattr(os, "sysconf") else 100
        self._system_info = self._build_system_info()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add_stage_ms(name, (time.perf_counter() - started) * 1000.0)

    def add_stage_ms(self, name: str, elapsed_ms: float) -> None:
        if not self.enabled:
            return
        key = str(name).strip()
        if not key:
            return
        self._current_stage_ms[key] = self._current_stage_ms.get(key, 0.0) + float(elapsed_ms)

    def start_frame(self, *, timestamp_unix_s: float | None = None, task_epoch: int | None = None) -> None:
        if not self.enabled:
            return
        self._current_frame = {
            "timestamp_unix_s": time.time() if timestamp_unix_s is None else float(timestamp_unix_s),
            "task_epoch": "" if task_epoch is None else int(task_epoch),
        }
        self._current_stage_ms = {}

    def record_frame(
        self,
        *,
        frame_index: int,
        timestamp_unix_s: float | None = None,
        task_epoch: int | None = None,
        fps: float | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        if self._current_frame is None:
            self.start_frame(timestamp_unix_s=timestamp_unix_s, task_epoch=task_epoch)
        assert self._current_frame is not None
        row = dict(self._current_frame)
        row["session_index"] = int(self.session_index)
        row["frame_index"] = int(frame_index)
        row["elapsed_s"] = time.perf_counter() - self._session_started_perf
        if timestamp_unix_s is not None:
            row["timestamp_unix_s"] = float(timestamp_unix_s)
        if task_epoch is not None:
            row["task_epoch"] = int(task_epoch)
        if fps is not None:
            row["fps"] = float(fps)
        for name, elapsed_ms in sorted(self._current_stage_ms.items()):
            row[f"stage_{name}_ms"] = float(elapsed_ms)
        for key, value in (metrics or {}).items():
            row[str(key)] = _json_safe(value)
        self.frames.append(row)
        self._current_frame = None
        self._current_stage_ms = {}
        self._sample_resources_if_due()
        self._print_if_due(row)

    def save_current_session(self, *, reason: str = "manual_save") -> dict[str, str]:
        if not self.enabled:
            return {}
        if not self.frames and not self.resources:
            raise RuntimeError("No runtime profiling samples are buffered.")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = _timestamp_for_filename()
        base = self.output_dir / f"runtime_profile_{timestamp}_session{self.session_index:03d}"
        frames_path = base.with_name(base.name + "_frames.csv")
        resources_path = base.with_name(base.name + "_resources.csv")
        summary_path = base.with_name(base.name + "_summary.json")

        self._write_csv(frames_path, self.frames)
        self._write_csv(resources_path, self.resources)
        summary = self._build_summary(
            reason=reason,
            frames_path=frames_path,
            resources_path=resources_path,
            summary_path=summary_path,
        )
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)

        paths = {
            "frames": str(frames_path),
            "resources": str(resources_path),
            "summary": str(summary_path),
        }
        self.discard_current_session(reason=f"saved:{reason}", increment=True)
        return paths

    def discard_current_session(self, *, reason: str = "discard", increment: bool = True) -> None:
        if not self.enabled:
            return
        self.frames.clear()
        self.resources.clear()
        self._current_frame = None
        self._current_stage_ms = {}
        self._session_started_wall = time.time()
        self._session_started_perf = time.perf_counter()
        self._last_resource_perf = None
        self._last_print_perf = None
        self._last_proc_ticks = _read_process_cpu_ticks()
        self._last_cpu_total_idle = _read_proc_stat_total()
        if increment:
            self.session_index += 1
        print(f"[INFO] Runtime profiling session reset ({reason}). session={self.session_index:03d}")

    def close(self) -> None:
        self._current_frame = None
        self._current_stage_ms = {}

    @property
    def system_info(self) -> dict[str, Any]:
        return dict(self._system_info)

    def _sample_resources_if_due(self) -> None:
        now_perf = time.perf_counter()
        if self._last_resource_perf is not None and now_perf - self._last_resource_perf < self.sample_interval_s:
            return
        self.resources.append(self._sample_resources(now_perf))
        self._last_resource_perf = now_perf

    def _sample_resources(self, now_perf: float) -> dict[str, Any]:
        row: dict[str, Any] = {
            "session_index": int(self.session_index),
            "sample_index": len(self.resources),
            "timestamp_unix_s": time.time(),
            "elapsed_s": now_perf - self._session_started_perf,
        }
        row.update(_read_process_memory_mb())
        row.update(_read_meminfo_mb())
        row.update(self._cpu_percentages())
        row.update(self._torch_cuda_memory())
        row.update(self._nvidia_smi_sample())
        return row

    def _cpu_percentages(self) -> dict[str, float | None]:
        result = {"process_cpu_percent": None, "system_cpu_percent": None}
        proc_ticks = _read_process_cpu_ticks()
        cpu_total_idle = _read_proc_stat_total()
        now_perf = time.perf_counter()

        if proc_ticks is not None and self._last_proc_ticks is not None and self._last_resource_perf is not None:
            elapsed_s = max(now_perf - self._last_resource_perf, 1e-9)
            tick_delta = max(proc_ticks - self._last_proc_ticks, 0)
            result["process_cpu_percent"] = 100.0 * (tick_delta / float(self._clock_ticks)) / elapsed_s
        if cpu_total_idle is not None and self._last_cpu_total_idle is not None:
            total, idle = cpu_total_idle
            last_total, last_idle = self._last_cpu_total_idle
            total_delta = max(total - last_total, 0)
            idle_delta = max(idle - last_idle, 0)
            if total_delta > 0:
                result["system_cpu_percent"] = 100.0 * (1.0 - idle_delta / total_delta)

        self._last_proc_ticks = proc_ticks
        self._last_cpu_total_idle = cpu_total_idle
        return result

    def _torch_cuda_memory(self) -> dict[str, Any]:
        result = {
            "torch_cuda_available": False,
            "torch_cuda_device_name": None,
            "torch_cuda_allocated_mb": None,
            "torch_cuda_reserved_mb": None,
            "torch_cuda_max_allocated_mb": None,
        }
        try:
            import torch
        except Exception:
            return result
        try:
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            return result
        result["torch_cuda_available"] = cuda_available
        if not cuda_available:
            return result
        try:
            device_index = torch.cuda.current_device()
            result["torch_cuda_device_name"] = torch.cuda.get_device_name(device_index)
            result["torch_cuda_allocated_mb"] = torch.cuda.memory_allocated(device_index) / (1024.0 * 1024.0)
            result["torch_cuda_reserved_mb"] = torch.cuda.memory_reserved(device_index) / (1024.0 * 1024.0)
            result["torch_cuda_max_allocated_mb"] = torch.cuda.max_memory_allocated(device_index) / (1024.0 * 1024.0)
        except Exception:
            pass
        return result

    def _nvidia_smi_sample(self) -> dict[str, Any]:
        result = {
            "gpu_name": None,
            "gpu_util_percent": None,
            "gpu_memory_used_mb": None,
            "gpu_memory_total_mb": None,
            "gpu_temp_c": None,
            "gpu_power_w": None,
        }
        if not self.nvidia_smi_path:
            return result
        query = (
            "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
        )
        cmd = [
            self.nvidia_smi_path,
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=0.5,
            )
        except Exception:
            return result
        if completed.returncode != 0:
            return result
        line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            return result
        result["gpu_name"] = parts[0] or None
        numeric_keys = [
            "gpu_util_percent",
            "gpu_memory_used_mb",
            "gpu_memory_total_mb",
            "gpu_temp_c",
            "gpu_power_w",
        ]
        for key, raw_value in zip(numeric_keys, parts[1:6]):
            try:
                result[key] = float(raw_value)
            except (TypeError, ValueError):
                result[key] = None
        return result

    def _print_if_due(self, row: dict[str, Any]) -> None:
        if self.print_every_s <= 0.0:
            return
        now_perf = time.perf_counter()
        if self._last_print_perf is not None and now_perf - self._last_print_perf < self.print_every_s:
            return
        self._last_print_perf = now_perf
        resource = self.resources[-1] if self.resources else {}
        loop_ms = row.get("stage_loop_total_ms")
        fps = row.get("fps")
        rss = resource.get("process_rss_mb")
        gpu_mem = resource.get("gpu_memory_used_mb") or resource.get("torch_cuda_reserved_mb")
        gpu_util = resource.get("gpu_util_percent")
        print(
            "[PROFILE] "
            f"session={self.session_index:03d} frames={len(self.frames)} "
            f"fps={_fmt(fps, 1)} loop_ms={_fmt(loop_ms, 1)} "
            f"rss_mb={_fmt(rss, 0)} gpu_mem_mb={_fmt(gpu_mem, 0)} "
            f"gpu_util={_fmt(gpu_util, 0)}%"
        )

    def _write_csv(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = self._fieldnames(rows)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})

    @staticmethod
    def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
        preferred = [
            "session_index",
            "frame_index",
            "sample_index",
            "timestamp_unix_s",
            "elapsed_s",
            "task_epoch",
            "fps",
        ]
        keys = set()
        for row in rows:
            keys.update(row.keys())
        ordered = [key for key in preferred if key in keys]
        ordered.extend(sorted(key for key in keys if key not in ordered))
        return ordered

    def _build_summary(self, *, reason: str, frames_path: Path, resources_path: Path, summary_path: Path) -> dict[str, Any]:
        stage_keys = sorted(
            key for row in self.frames for key in row.keys() if key.startswith("stage_") and key.endswith("_ms")
        )
        stage_summary = {
            key[len("stage_") : -len("_ms")]: _percentiles(
                [float(row[key]) for row in self.frames if row.get(key) not in ("", None)]
            )
            for key in stage_keys
        }
        frame_metrics = {
            "fps": _percentiles([float(row["fps"]) for row in self.frames if row.get("fps") not in ("", None)]),
            "cam0_yolo_infer_ms": _percentiles([float(row["cam0_yolo_infer_ms"]) for row in self.frames if row.get("cam0_yolo_infer_ms") not in ("", None)]),
            "cam1_yolo_infer_ms": _percentiles([float(row["cam1_yolo_infer_ms"]) for row in self.frames if row.get("cam1_yolo_infer_ms") not in ("", None)]),
            "shape_fit_icp_time_ms": _percentiles([float(row["shape_fit_icp_time_ms"]) for row in self.frames if row.get("shape_fit_icp_time_ms") not in ("", None)]),
        }
        resource_summary = {
            "peak_process_rss_mb": _max_metric(self.resources, "process_rss_mb"),
            "peak_torch_cuda_reserved_mb": _max_metric(self.resources, "torch_cuda_reserved_mb"),
            "peak_torch_cuda_max_allocated_mb": _max_metric(self.resources, "torch_cuda_max_allocated_mb"),
            "peak_gpu_memory_used_mb": _max_metric(self.resources, "gpu_memory_used_mb"),
            "mean_gpu_util_percent": _mean_metric(self.resources, "gpu_util_percent"),
            "peak_gpu_util_percent": _max_metric(self.resources, "gpu_util_percent"),
            "mean_process_cpu_percent": _mean_metric(self.resources, "process_cpu_percent"),
            "peak_process_cpu_percent": _max_metric(self.resources, "process_cpu_percent"),
            "mean_system_cpu_percent": _mean_metric(self.resources, "system_cpu_percent"),
        }
        return {
            "schema_version": 1,
            "reason": reason,
            "session_index": int(self.session_index),
            "started_at": datetime.fromtimestamp(self._session_started_wall).isoformat(timespec="seconds"),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "duration_s": time.perf_counter() - self._session_started_perf,
            "frame_count": len(self.frames),
            "resource_sample_count": len(self.resources),
            "paths": {
                "frames": str(frames_path),
                "resources": str(resources_path),
                "summary": str(summary_path),
            },
            "system": _json_safe(self._system_info),
            "run_context": _json_safe(self.run_context),
            "stage_latency_ms": stage_summary,
            "frame_metrics": frame_metrics,
            "resource_metrics": resource_summary,
        }

    def _build_system_info(self) -> dict[str, Any]:
        info = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        }
        info.update(_read_meminfo_mb())
        try:
            import torch

            info["torch_version"] = getattr(torch, "__version__", None)
            info["torch_cuda_available"] = bool(torch.cuda.is_available())
            if info["torch_cuda_available"]:
                info["torch_cuda_device_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
        except Exception:
            info["torch_version"] = None
            info["torch_cuda_available"] = False
        return info


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(_json_safe(value), sort_keys=True)


def _fmt(value: Any, decimals: int) -> str:
    if value in ("", None):
        return "-"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "-"


def _numeric_metric(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value in ("", None):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _max_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = _numeric_metric(rows, key)
    return max(values) if values else None


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = _numeric_metric(rows, key)
    return sum(values) / len(values) if values else None


__all__ = ["RuntimeProfiler"]
