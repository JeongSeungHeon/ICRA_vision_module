import argparse

import matplotlib.pyplot as plt
import numpy as np

FINGERS = [
    ([0, 1], [1, 2], [2, 3], [3, 4]),
    ([0, 5], [5, 6], [6, 7], [7, 8]),
    ([0, 9], [9, 10], [10, 11], [11, 12]),
    ([0, 13], [13, 14], [14, 15], [15, 16]),
    ([0, 17], [17, 18], [18, 19], [19, 20]),
]
FINGER_COLORS = ["orange", "black", "green", "blue", "red"]


def read_pose_file(filename):
    frames = []
    with open(filename, "r") as file_handle:
        for line in file_handle:
            values = [float(value) for value in line.split()]
            if not values:
                continue
            frame = np.reshape(values, (21, 3)).astype(np.float32)
            invalid_mask = np.all(frame == -1.0, axis=1)
            frame[invalid_mask] = np.nan
            frames.append(frame)

    if not frames:
        raise RuntimeError(f"No frames found in {filename}.")

    return np.array(frames, dtype=np.float32)


def camera_to_display_coordinates(points_xyz):
    display_points = points_xyz.copy()
    display_points[:, 0] = points_xyz[:, 0]
    display_points[:, 1] = points_xyz[:, 2]
    display_points[:, 2] = -points_xyz[:, 1]
    return display_points


def compute_axis_limits(points_xyz):
    valid_points = points_xyz[np.isfinite(points_xyz).all(axis=2)]
    if valid_points.size == 0:
        return (-0.2, 0.2), (0.0, 0.6), (-0.2, 0.2)

    valid_points = valid_points.reshape((-1, 3))
    mins = np.min(valid_points, axis=0)
    maxs = np.max(valid_points, axis=0)
    center = 0.5 * (mins + maxs)
    span = max(np.max(maxs - mins), 0.2)
    margin = 0.25 * span

    return (
        (center[0] - span / 2 - margin, center[0] + span / 2 + margin),
        (center[1] - span / 2 - margin, center[1] + span / 2 + margin),
        (center[2] - span / 2 - margin, center[2] + span / 2 + margin),
    )


def draw_frame(ax, frame_xyz, frame_index, axis_limits):
    ax.cla()
    valid_mask = np.isfinite(frame_xyz).all(axis=1)

    if np.any(valid_mask):
        ax.scatter(
            frame_xyz[valid_mask, 0],
            frame_xyz[valid_mask, 1],
            frame_xyz[valid_mask, 2],
            c="crimson",
            s=35,
        )

    for finger, color in zip(FINGERS, FINGER_COLORS):
        for start_idx, end_idx in finger:
            if not (valid_mask[start_idx] and valid_mask[end_idx]):
                continue
            ax.plot(
                [frame_xyz[start_idx, 0], frame_xyz[end_idx, 0]],
                [frame_xyz[start_idx, 1], frame_xyz[end_idx, 1]],
                [frame_xyz[start_idx, 2], frame_xyz[end_idx, 2]],
                linewidth=2.5,
                c=color,
            )

    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])
    ax.set_xlabel("x (right)")
    ax.set_ylabel("z (forward)")
    ax.set_zlabel("y (up)")
    ax.set_title(f"Depth 3D Hand Pose - frame {frame_index}")
    ax.view_init(elev=18, azim=-75)


def visualize_pose_file(filename, stride=1, pause=0.03):
    pose_frames = read_pose_file(filename)
    display_frames = np.array([camera_to_display_coordinates(frame) for frame in pose_frames], dtype=np.float32)
    axis_limits = compute_axis_limits(display_frames)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for frame_index in range(0, len(display_frames), max(1, stride)):
        draw_frame(ax, display_frames[frame_index], frame_index, axis_limits)
        plt.pause(pause)

    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("pose_file", help="Path to kpts_<serial>_3d_depth.dat")
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth frame")
    parser.add_argument("--pause", type=float, default=0.03, help="Seconds to pause between frames")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    visualize_pose_file(args.pose_file, stride=args.stride, pause=args.pause)
