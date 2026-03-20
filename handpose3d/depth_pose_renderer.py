import cv2 as cv
import numpy as np

FINGERS = [
    ([0, 1], [1, 2], [2, 3], [3, 4]),
    ([0, 5], [5, 6], [6, 7], [7, 8]),
    ([0, 9], [9, 10], [10, 11], [11, 12]),
    ([0, 13], [13, 14], [14, 15], [15, 16]),
    ([0, 17], [17, 18], [18, 19], [19, 20]),
]
FINGER_COLORS = [
    (0, 165, 255),
    (0, 0, 0),
    (0, 180, 0),
    (255, 0, 0),
    (0, 0, 255),
]


def camera_to_display_coordinates(points_xyz):
    # 카메라 좌표계(x right, y down, z forward)를
    # 화면에서 보기 쉬운 좌표계(x right, y forward, z up)로 바꾼다.
    display_points = points_xyz.copy()
    display_points[:, 0] = points_xyz[:, 0]
    display_points[:, 1] = points_xyz[:, 2]
    display_points[:, 2] = -points_xyz[:, 1]
    return display_points


def _build_projection_context(points_xyz, canvas_size, azimuth_deg=-65.0, elevation_deg=25.0):
    valid_mask = np.isfinite(points_xyz).all(axis=1)
    if not np.any(valid_mask):
        return None

    display_points = camera_to_display_coordinates(points_xyz)
    center_point = display_points[0] if valid_mask[0] else np.nanmean(display_points[valid_mask], axis=0)
    centered_points = display_points - center_point
    rotation = _rotation_matrix(azimuth_deg=azimuth_deg, elevation_deg=elevation_deg)
    rotated_points = centered_points @ rotation.T
    valid_rotated_points = rotated_points[valid_mask]
    max_extent = max(np.nanmax(np.abs(valid_rotated_points)), 0.12)
    scale = 0.38 * canvas_size / max_extent

    return {
        "center_point": center_point,
        "rotation": rotation,
        "scale": scale,
        "canvas_size": canvas_size,
    }


def _project_points_with_context(points_xyz, context):
    projected = np.full((len(points_xyz), 2), np.nan, dtype=np.float32)
    depth_order = np.full(len(points_xyz), np.nan, dtype=np.float32)

    if context is None:
        return projected, depth_order

    valid_mask = np.isfinite(points_xyz).all(axis=1)
    if not np.any(valid_mask):
        return projected, depth_order

    display_points = camera_to_display_coordinates(points_xyz)
    centered_points = display_points - context["center_point"]
    rotated_points = centered_points @ context["rotation"].T
    projected[valid_mask, 0] = rotated_points[valid_mask, 0] * context["scale"] + context["canvas_size"] / 2.0
    projected[valid_mask, 1] = context["canvas_size"] / 2.0 - rotated_points[valid_mask, 2] * context["scale"]
    depth_order[valid_mask] = rotated_points[valid_mask, 1]
    return projected, depth_order


def _rotation_matrix(azimuth_deg=-90.0, elevation_deg=25.0):
    # 3D hand를 어떤 시점에서 볼지 결정하는 가상 카메라 회전행렬이다.
    # azimuth는 좌우 회전, elevation은 위아래 회전에 해당한다.
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(elevation_deg)

    rotation_z = np.array(
        [
            [np.cos(azimuth), -np.sin(azimuth), 0.0],
            [np.sin(azimuth), np.cos(azimuth), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(elevation), -np.sin(elevation)],
            [0.0, np.sin(elevation), np.cos(elevation)],
        ],
        dtype=np.float32,
    )
    return rotation_x @ rotation_z


def _project_points(points_xyz, canvas_size, azimuth_deg=-65.0, elevation_deg=25.0):
    # 3D 점들을 2D canvas 위에 그릴 수 있도록 회전/이동/스케일링한다.
    context = _build_projection_context(points_xyz, canvas_size, azimuth_deg=azimuth_deg, elevation_deg=elevation_deg)
    return _project_points_with_context(points_xyz, context)


def render_live_pose(points_xyz, valid_mask, canvas_size=720, title="Live 3D Hand Pose"):
    # OpenCV canvas 위에 실시간 3D skeleton을 그려 반환한다.
    canvas = np.full((canvas_size, canvas_size, 3), 248, dtype=np.uint8)
    cv.rectangle(canvas, (0, 0), (canvas_size - 1, canvas_size - 1), (220, 220, 220), 2)
    cv.putText(canvas, title, (20, 35), cv.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv.LINE_AA)

    context = _build_projection_context(points_xyz, canvas_size)
    projected, depth_order = _project_points_with_context(points_xyz, context)
    finite_mask = np.isfinite(projected).all(axis=1) & valid_mask

    if not np.any(finite_mask):
        cv.putText(canvas, "No valid 3D landmarks", (20, canvas_size // 2), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 180), 2, cv.LINE_AA)
        return canvas

    # 점을 깊이 순서로 그리면 일부 겹침이 있을 때 조금 더 자연스럽게 보인다.
    order = np.argsort(np.nan_to_num(depth_order, nan=0.0))

    # 간단한 좌표축을 같이 그려서 현재 시점과 방향을 읽기 쉽게 만든다.
    origin = projected[0] if finite_mask[0] else np.array([70.0, canvas_size - 70.0], dtype=np.float32)
    axis_length = 55
    cv.arrowedLine(canvas, tuple(origin.astype(int)), tuple((origin + np.array([axis_length, 0])).astype(int)), (0, 0, 255), 2, tipLength=0.12)
    cv.arrowedLine(canvas, tuple(origin.astype(int)), tuple((origin + np.array([0, -axis_length])).astype(int)), (0, 180, 0), 2, tipLength=0.12)
    cv.putText(canvas, "x", tuple((origin + np.array([axis_length + 6, 0])).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
    cv.putText(canvas, "up", tuple((origin + np.array([-10, -axis_length - 8])).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 1, cv.LINE_AA)

    # 손가락 연결 정보를 이용해 뼈대를 선으로 그린다.
    for finger, color in zip(FINGERS, FINGER_COLORS):
        for start_idx, end_idx in finger:
            if not (finite_mask[start_idx] and finite_mask[end_idx]):
                continue
            cv.line(
                canvas,
                tuple(projected[start_idx].astype(int)),
                tuple(projected[end_idx].astype(int)),
                color,
                3,
                cv.LINE_AA,
            )

    # 관절 점은 depth 순서대로 찍어서 너무 뒤엉켜 보이지 않게 한다.
    for point_index in order:
        if not finite_mask[point_index]:
            continue
        point = tuple(projected[point_index].astype(int))
        cv.circle(canvas, point, 5, (40, 40, 40), -1, cv.LINE_AA)
        cv.circle(canvas, point, 3, (240, 80, 80), -1, cv.LINE_AA)

    # 화면 하단에는 현재 유효 landmark 개수와 대표 깊이값을 표시한다.
    valid_count = int(np.count_nonzero(finite_mask))
    valid_depths = points_xyz[finite_mask, 2]
    median_depth = float(np.median(valid_depths)) if valid_depths.size else float("nan")
    info_text = f"valid={valid_count}/21  median_z={median_depth:.3f}m"
    cv.putText(canvas, info_text, (20, canvas_size - 24), cv.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2, cv.LINE_AA)

    return canvas
