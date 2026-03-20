import argparse

import numpy as np

FINGER_BONES = [
    ("thumb_01", 0, 1),
    ("thumb_12", 1, 2),
    ("thumb_23", 2, 3),
    ("thumb_34", 3, 4),
    ("index_05", 0, 5),
    ("index_56", 5, 6),
    ("index_67", 6, 7),
    ("index_78", 7, 8),
    ("middle_09", 0, 9),
    ("middle_910", 9, 10),
    ("middle_1011", 10, 11),
    ("middle_1112", 11, 12),
    ("ring_013", 0, 13),
    ("ring_1314", 13, 14),
    ("ring_1415", 14, 15),
    ("ring_1516", 15, 16),
    ("pinky_017", 0, 17),
    ("pinky_1718", 17, 18),
    ("pinky_1819", 18, 19),
    ("pinky_1920", 19, 20),
]


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


def slice_frames(pose_frames, start_frame=None, end_frame=None):
    start = 0 if start_frame is None else max(0, start_frame)
    end = len(pose_frames) if end_frame is None else min(len(pose_frames), end_frame)
    return pose_frames[start:end]


def summarize_validity(pose_frames):
    valid_mask = np.isfinite(pose_frames).all(axis=2)
    valid_counts = np.sum(valid_mask, axis=1)
    return {
        "num_frames": int(len(pose_frames)),
        "mean_valid_points": float(np.mean(valid_counts)),
        "min_valid_points": int(np.min(valid_counts)),
        "max_valid_points": int(np.max(valid_counts)),
        "mean_valid_ratio": float(np.mean(valid_counts / 21.0)),
    }


def summarize_landmark_stability(pose_frames):
    summaries = []
    for landmark_index in range(21):
        coords = pose_frames[:, landmark_index, :]
        valid_mask = np.isfinite(coords).all(axis=1)
        valid_coords = coords[valid_mask]
        if len(valid_coords) < 2:
            summaries.append({
                "landmark": landmark_index,
                "valid_frames": int(len(valid_coords)),
                "z_std_mm": np.nan,
                "frame_delta_mm": np.nan,
            })
            continue

        z_std_mm = float(np.std(valid_coords[:, 2]) * 1000.0)
        frame_delta_mm = float(np.mean(np.linalg.norm(np.diff(valid_coords, axis=0), axis=1)) * 1000.0)
        summaries.append({
            "landmark": landmark_index,
            "valid_frames": int(len(valid_coords)),
            "z_std_mm": z_std_mm,
            "frame_delta_mm": frame_delta_mm,
        })

    return summaries


def summarize_bone_lengths(pose_frames):
    summaries = []
    for bone_name, start_idx, end_idx in FINGER_BONES:
        start_points = pose_frames[:, start_idx, :]
        end_points = pose_frames[:, end_idx, :]
        valid_mask = np.isfinite(start_points).all(axis=1) & np.isfinite(end_points).all(axis=1)
        if np.count_nonzero(valid_mask) < 2:
            summaries.append({
                "bone": bone_name,
                "valid_frames": int(np.count_nonzero(valid_mask)),
                "mean_mm": np.nan,
                "std_mm": np.nan,
            })
            continue

        lengths = np.linalg.norm(end_points[valid_mask] - start_points[valid_mask], axis=1)
        summaries.append({
            "bone": bone_name,
            "valid_frames": int(len(lengths)),
            "mean_mm": float(np.mean(lengths) * 1000.0),
            "std_mm": float(np.std(lengths) * 1000.0),
        })

    return summaries


def print_summary(validity_summary, landmark_summary, bone_summary):
    print("Depth Pose Validation")
    print(f"frames: {validity_summary['num_frames']}")
    print(f"mean valid points/frame: {validity_summary['mean_valid_points']:.2f} / 21")
    print(f"valid ratio: {validity_summary['mean_valid_ratio'] * 100.0:.1f}%")
    print(f"min/max valid points: {validity_summary['min_valid_points']} / {validity_summary['max_valid_points']}")
    print("")

    ranked_landmarks = sorted(
        [item for item in landmark_summary if np.isfinite(item["z_std_mm"])],
        key=lambda item: item["z_std_mm"],
        reverse=True,
    )
    print("Most unstable landmarks by z std (mm)")
    for item in ranked_landmarks[:5]:
        print(
            f"  landmark {item['landmark']:2d}: "
            f"z_std={item['z_std_mm']:.2f} mm, "
            f"frame_delta={item['frame_delta_mm']:.2f} mm, "
            f"valid_frames={item['valid_frames']}"
        )
    print("")

    ranked_bones = sorted(
        [item for item in bone_summary if np.isfinite(item["std_mm"])],
        key=lambda item: item["std_mm"],
        reverse=True,
    )
    print("Most unstable bone lengths (mm)")
    for item in ranked_bones[:5]:
        print(
            f"  {item['bone']:12s}: "
            f"mean={item['mean_mm']:.2f} mm, "
            f"std={item['std_mm']:.2f} mm, "
            f"valid_frames={item['valid_frames']}"
        )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("pose_file", help="Path to kpts_<serial>_3d_depth.dat")
    parser.add_argument("--start-frame", type=int, default=None, help="First frame index to include")
    parser.add_argument("--end-frame", type=int, default=None, help="End frame index (exclusive)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    pose_frames = read_pose_file(args.pose_file)
    pose_frames = slice_frames(pose_frames, start_frame=args.start_frame, end_frame=args.end_frame)

    if len(pose_frames) == 0:
        raise RuntimeError("Selected frame range is empty.")

    validity_summary = summarize_validity(pose_frames)
    landmark_summary = summarize_landmark_stability(pose_frames)
    bone_summary = summarize_bone_lengths(pose_frames)
    print_summary(validity_summary, landmark_summary, bone_summary)
