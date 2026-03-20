import numpy as np

PALM_LANDMARK_INDICES = [0, 5, 9, 13, 17]
REQUIRED_POSE_INDICES = [0, 5, 9, 17]


def _normalize(vector, eps=1e-6):
    norm = np.linalg.norm(vector)
    if norm < eps:
        return None
    return vector / norm


def _handedness_to_id(handedness):
    if handedness == "Right":
        return 1
    if handedness == "Left":
        return -1
    return 0


def rotation_matrix_to_quaternion(rotation_matrix):
    rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64)
    trace = np.trace(rotation_matrix)

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
        qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
        qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
    else:
        diagonal = np.diag(rotation_matrix)
        max_index = int(np.argmax(diagonal))
        if max_index == 0:
            s = np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2.0
            qw = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            qx = 0.25 * s
            qy = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            qz = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        elif max_index == 1:
            s = np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2.0
            qw = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            qx = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            qy = 0.25 * s
            qz = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2.0
            qw = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
            qx = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            qy = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            qz = 0.25 * s

    quaternion = np.array([qx, qy, qz, qw], dtype=np.float32)
    quaternion_norm = np.linalg.norm(quaternion)
    if quaternion_norm > 0.0:
        quaternion /= quaternion_norm
    return quaternion


def estimate_palm_pose(points_3d, valid_mask, quality_scores=None, handedness=None, previous_rotation=None):
    points_3d = np.asarray(points_3d, dtype=np.float32).reshape((21, 3))
    valid_mask = np.asarray(valid_mask, dtype=bool).reshape((21,))
    if quality_scores is None:
        quality_scores = np.zeros(21, dtype=np.float32)
    else:
        quality_scores = np.asarray(quality_scores, dtype=np.float32).reshape((21,))

    used_landmarks_mask = np.zeros(21, dtype=bool)
    used_landmarks_mask[PALM_LANDMARK_INDICES] = valid_mask[PALM_LANDMARK_INDICES]

    if np.count_nonzero(valid_mask[PALM_LANDMARK_INDICES]) < 4:
        return {
            "position": np.full(3, np.nan, dtype=np.float32),
            "rotation_matrix": np.full((3, 3), np.nan, dtype=np.float32),
            "quaternion": np.full(4, np.nan, dtype=np.float32),
            "valid": False,
            "quality": 0.0,
            "handedness": handedness,
            "handedness_id": _handedness_to_id(handedness),
            "used_landmarks_mask": used_landmarks_mask,
        }

    if not np.all(valid_mask[REQUIRED_POSE_INDICES]):
        return {
            "position": np.full(3, np.nan, dtype=np.float32),
            "rotation_matrix": np.full((3, 3), np.nan, dtype=np.float32),
            "quaternion": np.full(4, np.nan, dtype=np.float32),
            "valid": False,
            "quality": 0.0,
            "handedness": handedness,
            "handedness_id": _handedness_to_id(handedness),
            "used_landmarks_mask": used_landmarks_mask,
        }

    palm_points = points_3d[PALM_LANDMARK_INDICES][valid_mask[PALM_LANDMARK_INDICES]]
    position = np.mean(palm_points, axis=0).astype(np.float32)

    wrist = points_3d[0]
    index_mcp = points_3d[5]
    middle_mcp = points_3d[9]
    pinky_mcp = points_3d[17]

    x_axis = _normalize(index_mcp - pinky_mcp)
    y_hint = middle_mcp - wrist
    y_hint_norm = np.linalg.norm(y_hint)
    if x_axis is None or y_hint_norm < 1e-6:
        return {
            "position": position,
            "rotation_matrix": np.full((3, 3), np.nan, dtype=np.float32),
            "quaternion": np.full(4, np.nan, dtype=np.float32),
            "valid": False,
            "quality": 0.0,
            "handedness": handedness,
            "handedness_id": _handedness_to_id(handedness),
            "used_landmarks_mask": used_landmarks_mask,
        }

    z_axis = _normalize(np.cross(x_axis, y_hint))
    if z_axis is None:
        return {
            "position": position,
            "rotation_matrix": np.full((3, 3), np.nan, dtype=np.float32),
            "quaternion": np.full(4, np.nan, dtype=np.float32),
            "valid": False,
            "quality": 0.0,
            "handedness": handedness,
            "handedness_id": _handedness_to_id(handedness),
            "used_landmarks_mask": used_landmarks_mask,
        }

    y_axis = _normalize(np.cross(z_axis, x_axis))
    if y_axis is None:
        return {
            "position": position,
            "rotation_matrix": np.full((3, 3), np.nan, dtype=np.float32),
            "quaternion": np.full(4, np.nan, dtype=np.float32),
            "valid": False,
            "quality": 0.0,
            "handedness": handedness,
            "handedness_id": _handedness_to_id(handedness),
            "used_landmarks_mask": used_landmarks_mask,
        }

    # previous frame과의 연속성을 유지해 normal 축이 갑자기 뒤집히는 현상을 줄인다.
    if previous_rotation is not None and np.isfinite(previous_rotation).all():
        previous_z = previous_rotation[:, 2]
        if np.dot(previous_z, z_axis) < 0.0:
            z_axis = -z_axis
            y_axis = -y_axis

    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32)
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)

    palm_quality = quality_scores[PALM_LANDMARK_INDICES][valid_mask[PALM_LANDMARK_INDICES]]
    mean_quality = float(np.mean(palm_quality)) if palm_quality.size else 0.0
    geometry_quality = float(np.clip(np.linalg.norm(np.cross(x_axis, y_hint / y_hint_norm)), 0.0, 1.0))
    visibility_quality = float(np.count_nonzero(valid_mask[PALM_LANDMARK_INDICES]) / len(PALM_LANDMARK_INDICES))
    quality = float(np.clip(0.5 * mean_quality + 0.25 * geometry_quality + 0.25 * visibility_quality, 0.0, 1.0))

    return {
        "position": position,
        "rotation_matrix": rotation_matrix,
        "quaternion": quaternion,
        "valid": True,
        "quality": quality,
        "handedness": handedness,
        "handedness_id": _handedness_to_id(handedness),
        "used_landmarks_mask": used_landmarks_mask,
    }
