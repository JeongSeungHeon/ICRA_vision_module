#!/usr/bin/env python3
"""Rebuild a Rerun .rrd sidecar from a saved task mp4 and tactile CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import rerun as rr


def tactile_csv_path_for(video_path: Path) -> Path:
    return video_path.with_name(f"{video_path.stem}_tactile.csv")


def rrd_path_for(video_path: Path) -> Path:
    return video_path.with_suffix(".rrd")


def set_rerun_time(frame_index: int, elapsed_s: float, recording=None) -> None:
    kwargs = {}
    if recording is not None:
        kwargs["recording"] = recording
    if hasattr(rr, "set_time"):
        rr.set_time("frame", sequence=int(frame_index), **kwargs)
        rr.set_time("time", duration=float(elapsed_s), **kwargs)
    else:
        rr.set_time_sequence("frame", int(frame_index), **kwargs)
        rr.set_time_seconds("time", float(elapsed_s), **kwargs)


def rerun_scalar(value: float):
    scalar_cls = getattr(rr, "Scalars", None)
    if scalar_cls is None:
        scalar_cls = getattr(rr, "Scalar")
    return scalar_cls(float(value))


def log_frame(frame_bgr, row: dict[str, str], recording=None) -> None:
    kwargs = {}
    if recording is not None:
        kwargs["recording"] = recording

    frame_index = int(row["frame_index"])
    elapsed_s = float(row["elapsed_s"])
    set_rerun_time(frame_index, elapsed_s, recording=recording)

    image = rr.Image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    if hasattr(image, "compress"):
        image = image.compress()
    rr.log("camera/rgb", image, **kwargs)

    rr.log("tactile/total_norm", rerun_scalar(float(row["total_norm"])), **kwargs)
    if hasattr(rr, "TextLog"):
        rr.log("tactile/status", rr.TextLog(row.get("status", "")), **kwargs)

    mag_indices = sorted(
        int(key[3:-2])
        for key in row
        if key.startswith("mag") and key.endswith("_x")
    )
    for mag_idx in mag_indices:
        values = np.array(
            [
                float(row[f"mag{mag_idx}_x"]),
                float(row[f"mag{mag_idx}_y"]),
                float(row[f"mag{mag_idx}_z"]),
            ],
            dtype=np.float32,
        )
        rr.log(f"tactile/mag_{mag_idx}/x", rerun_scalar(float(values[0])), **kwargs)
        rr.log(f"tactile/mag_{mag_idx}/y", rerun_scalar(float(values[1])), **kwargs)
        rr.log(f"tactile/mag_{mag_idx}/z", rerun_scalar(float(values[2])), **kwargs)
        rr.log(f"tactile/mag_{mag_idx}/norm", rerun_scalar(float(np.linalg.norm(values))), **kwargs)


def rebuild(video_path: Path, csv_path: Path, output_path: Path, overwrite: bool) -> int:
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"tactile csv not found: {csv_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output exists, pass --overwrite: {output_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    recording = rr.new_recording("handover_tactile_recording") if hasattr(rr, "new_recording") else None
    if recording is not None:
        rr.save(str(output_path), recording=recording)
    else:
        rr.init("handover_tactile_recording", spawn=False)
        rr.save(str(output_path))

    rows_written = 0
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                ok, frame = capture.read()
                if not ok:
                    break
                log_frame(frame, row, recording=recording)
                rows_written += 1
    finally:
        capture.release()
        if hasattr(rr, "flush"):
            try:
                rr.flush(blocking=True, recording=recording)
            except TypeError:
                rr.flush(blocking=True)
        if hasattr(rr, "disconnect"):
            try:
                rr.disconnect(recording=recording)
            except TypeError:
                rr.disconnect()

    return rows_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild a tactile Rerun .rrd from task mp4 + tactile CSV.")
    parser.add_argument("video", type=Path, help="Task mp4 path, e.g. recordings/task_YYYYMMDD_HHMMSS.mp4")
    parser.add_argument("--csv", dest="csv_path", type=Path, default=None, help="Optional tactile CSV path.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output .rrd path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing .rrd.")
    args = parser.parse_args()

    video_path = args.video
    csv_path = args.csv_path or tactile_csv_path_for(video_path)
    output_path = args.output or rrd_path_for(video_path)
    rows_written = rebuild(video_path, csv_path, output_path, args.overwrite)
    print(f"Rebuilt {output_path} with {rows_written} frame(s).")


if __name__ == "__main__":
    main()
