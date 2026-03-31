import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2 as cv
import mediapipe as mp
import numpy as np

from handpose3d.depth_pose_renderer import render_live_pose
from handpose3d.hand_pose_6d import estimate_palm_pose
from utils.calibration_utils import DLT, get_projection_matrix, write_keypoints_to_disk
from utils.depth_lifter import lift_hand_pose_3d
from utils.realsense_stream import RealSenseCamera, list_realsense_serials

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

TRIANGULATION_FRAME_SHAPE = [720, 1280]


def create_hand_detector():
    # MediaPipe Hands detector를 공통 설정으로 생성한다.
    return mp_hands.Hands(
        min_detection_confidence=0.5,
        max_num_hands=1,
        min_tracking_confidence=0.5,
    )


def extract_hand_keypoints(results, image_shape):
    # 검출이 없을 때는 기존 코드와 호환되도록 [-1, -1]로 채운다.
    image_height, image_width = image_shape[:2]
    frame_keypoints = [[-1, -1]] * 21

    if not results.multi_hand_landmarks:
        return np.array(frame_keypoints, dtype=np.int32)

    # 첫 번째 손의 21개 landmark를 픽셀 좌표로 변환한다.
    hand_landmarks = results.multi_hand_landmarks[0]
    frame_keypoints = []
    for point_index in range(21):
        landmark = hand_landmarks.landmark[point_index]
        pixel_x = int(round(image_width * landmark.x))
        pixel_y = int(round(image_height * landmark.y))
        pixel_x = int(np.clip(pixel_x, 0, image_width - 1))
        pixel_y = int(np.clip(pixel_y, 0, image_height - 1))
        frame_keypoints.append([pixel_x, pixel_y])

    return np.array(frame_keypoints, dtype=np.int32)


def detect_hand_keypoints(hands, bgr_frame):
    # MediaPipe는 RGB 입력을 기대하므로 변환 후 2D keypoint를 추출한다.
    rgb_frame = cv.cvtColor(bgr_frame, cv.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True
    keypoints = extract_hand_keypoints(results, bgr_frame.shape)
    handedness = None
    if results.multi_handedness:
        handedness = results.multi_handedness[0].classification[0].label
    return keypoints, results, handedness


def draw_landmarks(frame, results):
    # 시각화용으로 MediaPipe가 반환한 손 skeleton을 원본 프레임 위에 그린다.
    if not results.multi_hand_landmarks:
        return

    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


def crop_triangulation_frame(frame):
    # 기존 triangulation 경로는 정사각형 입력을 전제로 하므로 중앙 crop을 유지한다.
    frame_height, frame_width = frame.shape[:2]
    if frame_width == frame_height:
        return frame

    crop_size = min(frame_height, frame_width)
    width_center = frame_width // 2
    crop_half_width = crop_size // 2
    return frame[:, width_center - crop_half_width:width_center + crop_half_width]


def run_triangulation_pipeline(input_stream0, input_stream1, projection0, projection1):
    # 기존 stereo triangulation 파이프라인을 그대로 유지하는 경로다.
    cap0 = cv.VideoCapture(input_stream0)
    cap1 = cv.VideoCapture(input_stream1)
    caps = [cap0, cap1]

    for cap in caps:
        cap.set(3, TRIANGULATION_FRAME_SHAPE[1])
        cap.set(4, TRIANGULATION_FRAME_SHAPE[0])

    hands0 = create_hand_detector()
    hands1 = create_hand_detector()

    kpts_cam0 = []
    kpts_cam1 = []
    kpts_3d = []

    while True:
        # 두 카메라 프레임을 읽고 같은 순서의 2D keypoint를 맞춰서 삼각측량한다.
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            break

        frame0 = crop_triangulation_frame(frame0)
        frame1 = crop_triangulation_frame(frame1)

        frame0_keypoints, results0, _ = detect_hand_keypoints(hands0, frame0)
        frame1_keypoints, results1, _ = detect_hand_keypoints(hands1, frame1)

        kpts_cam0.append(frame0_keypoints)
        kpts_cam1.append(frame1_keypoints)

        frame_p3ds = []
        for uv0, uv1 in zip(frame0_keypoints, frame1_keypoints):
            # 한쪽이라도 검출 실패면 해당 3D 점도 invalid로 둔다.
            if uv0[0] == -1 or uv1[0] == -1:
                point_3d = [-1, -1, -1]
            else:
                point_3d = DLT(projection0, projection1, uv0, uv1)
            frame_p3ds.append(point_3d)

        frame_p3ds = np.array(frame_p3ds, dtype=np.float32).reshape((21, 3))
        kpts_3d.append(frame_p3ds)

        draw_landmarks(frame0, results0)
        draw_landmarks(frame1, results1)
        cv.imshow("cam0", frame0)
        cv.imshow("cam1", frame1)

        if cv.waitKey(1) & 0xFF == 27:
            break

    cv.destroyAllWindows()
    for cap in caps:
        cap.release()

    hands0.close()
    hands1.close()

    return np.array(kpts_cam0), np.array(kpts_cam1), np.array(kpts_3d)


def annotate_depth_pose(frame, keypoints_2d, points_3d, valid_mask):
    # depth lifting 결과를 2D 화면 위에 간단히 확인할 수 있도록 z값을 함께 표시한다.
    for point_index, (uv, xyz, is_valid) in enumerate(zip(keypoints_2d, points_3d, valid_mask)):
        if uv[0] < 0:
            continue

        center = (int(uv[0]), int(uv[1]))
        color = (0, 255, 0) if is_valid else (0, 0, 255)
        cv.circle(frame, center, 3, color, -1)

        if is_valid:
            label = f"{point_index}:{xyz[2]:.3f}m"
        else:
            label = f"{point_index}:NA"
        cv.putText(frame, label, (center[0] + 4, center[1] - 4), cv.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv.LINE_AA)


def annotate_palm_pose(frame, palm_pose):
    # 추정된 palm 6D pose의 중심점, handedness, quality를 2D 화면에도 표시한다.
    if not palm_pose.get("valid", False):
        cv.putText(frame, "Palm pose: invalid", (12, 24), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 180), 2, cv.LINE_AA)
        return

    handedness = palm_pose.get("handedness", "Unknown")
    quality = palm_pose.get("quality", 0.0)
    position = palm_pose["position"]
    cv.putText(
        frame,
        f"Palm pose {handedness} q={quality:.2f}",
        (12, 24),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        2,
        cv.LINE_AA,
    )
    cv.putText(
        frame,
        f"t=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})",
        (12, 52),
        cv.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        2,
        cv.LINE_AA,
    )


def project_camera_point_to_pixel(point_xyz, intrinsics):
    if not np.isfinite(point_xyz).all():
        return None
    if point_xyz[2] <= 1e-6:
        return None

    pixel_x = intrinsics["fx"] * point_xyz[0] / point_xyz[2] + intrinsics["cx"]
    pixel_y = intrinsics["fy"] * point_xyz[1] / point_xyz[2] + intrinsics["cy"]
    return np.array([pixel_x, pixel_y], dtype=np.float32)


def annotate_palm_axes_on_image(frame, palm_pose, intrinsics, axis_length_m=0.05):
    # palm 중심과 local x/y/z 축을 실제 컬러 영상 위에 투영해서 표시한다.
    if not palm_pose.get("valid", False):
        return

    center = palm_pose["position"]
    rotation_matrix = palm_pose["rotation_matrix"]
    pose_points = [
        center,
        center + axis_length_m * rotation_matrix[:, 0],
        center + axis_length_m * rotation_matrix[:, 1],
        center + axis_length_m * rotation_matrix[:, 2],
    ]
    projected_points = [project_camera_point_to_pixel(point, intrinsics) for point in pose_points]
    if any(point is None for point in projected_points):
        return

    projected_points = [point.astype(np.int32) for point in projected_points]
    center_pixel = tuple(projected_points[0])
    axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    axis_labels = ["x", "y", "z"]

    cv.circle(frame, center_pixel, 5, (255, 255, 255), -1, cv.LINE_AA)
    cv.circle(frame, center_pixel, 3, (30, 30, 30), -1, cv.LINE_AA)

    for projected_point, color, label in zip(projected_points[1:], axis_colors, axis_labels):
        axis_tip = tuple(projected_point)
        cv.arrowedLine(frame, center_pixel, axis_tip, color, 3, cv.LINE_AA, tipLength=0.16)
        cv.putText(frame, label, (axis_tip[0] + 4, axis_tip[1] - 4), cv.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv.LINE_AA)


def run_depth_pipeline(serials=None, width=640, height=480, fps=30, patch_radius=2, min_depth_m=0.1, max_depth_m=1.2):
    # RealSense color + aligned depth를 사용해 2D landmark를 카메라 좌표계 3D로 올린다.
    selected_serials = list(serials or [])
    if not selected_serials:
        available_serials = list_realsense_serials()
        if not available_serials:
            raise RuntimeError("No RealSense devices detected.")
        selected_serials = available_serials[:2]

    cameras = []
    hands_detectors = []
    keypoints_2d_logs = []
    keypoints_3d_logs = []
    palm_pose_logs = []
    previous_rotations = []

    try:
        for serial in selected_serials:
            cameras.append(RealSenseCamera(serial=serial, width=width, height=height, fps=fps))
            hands_detectors.append(create_hand_detector())
            keypoints_2d_logs.append([])
            keypoints_3d_logs.append([])
            palm_pose_logs.append([])
            previous_rotations.append(None)

        while True:
            for camera_index, camera in enumerate(cameras):
                # 각 카메라에서 color/depth 한 쌍을 읽고 2D 손 keypoint를 구한다.
                frame_bundle = camera.read()
                keypoints_2d, results, handedness = detect_hand_keypoints(hands_detectors[camera_index], frame_bundle.color_image)

                # depth patch를 robust하게 읽어 각 2D keypoint를 3D로 deprojection한다.
                points_3d, valid_mask, _, quality_scores = lift_hand_pose_3d(
                    keypoints_2d,
                    frame_bundle.depth_image_m,
                    frame_bundle.intrinsics,
                    patch_radius=patch_radius,
                    min_depth_m=min_depth_m,
                    max_depth_m=max_depth_m,
                )
                palm_pose = estimate_palm_pose(
                    points_3d,
                    valid_mask,
                    quality_scores=quality_scores,
                    handedness=handedness,
                    previous_rotation=previous_rotations[camera_index],
                )
                if palm_pose["valid"]:
                    previous_rotations[camera_index] = palm_pose["rotation_matrix"]

                points_3d_to_save = np.where(valid_mask[:, None], points_3d, -1.0)
                keypoints_2d_logs[camera_index].append(keypoints_2d.copy())
                keypoints_3d_logs[camera_index].append(points_3d_to_save.astype(np.float32))
                palm_pose_logs[camera_index].append(
                    serialize_palm_pose(
                        palm_pose,
                        timestamp_ms=frame_bundle.timestamp_ms,
                    )
                )

                # 컬러 영상 창과 실시간 3D skeleton 창을 함께 띄워서 즉시 검증할 수 있게 한다.
                display_frame = frame_bundle.color_image.copy()
                draw_landmarks(display_frame, results)
                #annotate_depth_pose(display_frame, keypoints_2d, points_3d, valid_mask)
                annotate_palm_pose(display_frame, palm_pose)
                annotate_palm_axes_on_image(display_frame, palm_pose, frame_bundle.intrinsics)
                cv.imshow(f"realsense_{camera_index}", display_frame)
                live_pose_canvas = render_live_pose(
                    points_3d,
                    valid_mask,
                    title=f"Live 3D Pose - cam{camera_index}",
                )
                cv.imshow(f"realsense_{camera_index}_3d", live_pose_canvas)

            if cv.waitKey(1) & 0xFF == 27:
                break

    finally:
        cv.destroyAllWindows()
        for hands in hands_detectors:
            hands.close()
        for camera in cameras:
            camera.stop()

    outputs_2d = [np.array(camera_log) for camera_log in keypoints_2d_logs]
    outputs_3d = [np.array(camera_log) for camera_log in keypoints_3d_logs]
    outputs_pose = [np.array(camera_log, dtype=np.float32) for camera_log in palm_pose_logs]
    return selected_serials, outputs_2d, outputs_3d, outputs_pose


def serialize_palm_pose(palm_pose, timestamp_ms):
    if palm_pose["valid"]:
        position = palm_pose["position"]
        quaternion = palm_pose["quaternion"]
        valid = 1.0
        quality = palm_pose["quality"]
        handedness_id = float(palm_pose["handedness_id"])
    else:
        position = np.full(3, -1.0, dtype=np.float32)
        quaternion = np.full(4, -1.0, dtype=np.float32)
        valid = 0.0
        quality = 0.0
        handedness_id = float(palm_pose["handedness_id"])

    return np.array(
        [
            position[0],
            position[1],
            position[2],
            quaternion[0],
            quaternion[1],
            quaternion[2],
            quaternion[3],
            valid,
            quality,
            handedness_id,
            float(timestamp_ms),
        ],
        dtype=np.float32,
    )


def write_pose_vectors_to_disk(filename, pose_vectors):
    with open(filename, "w") as output_file:
        for pose_vector in pose_vectors:
            output_file.write(" ".join(str(float(value)) for value in pose_vector))
            output_file.write("\n")


def save_depth_outputs(serials, keypoints_2d_list, keypoints_3d_list, palm_pose_list):
    # 카메라별 2D/3D 결과와 palm pose를 별도 파일로 저장한다.
    for camera_index, (serial, keypoints_2d, keypoints_3d, palm_poses) in enumerate(zip(serials, keypoints_2d_list, keypoints_3d_list, palm_pose_list)):
        serial_label = serial or f"cam{camera_index}"
        write_keypoints_to_disk(f"kpts_{serial_label}_2d.dat", keypoints_2d)
        write_keypoints_to_disk(f"kpts_{serial_label}_3d_depth.dat", keypoints_3d)
        write_pose_vectors_to_disk(f"palm_pose_{serial_label}.dat", palm_poses)


def parse_arguments(argv):
    # 기존 사용 방식인 `python handpose3d.py 0 1`도 계속 동작하도록 예외 처리한다.
    if len(argv) == 3 and all(not arg.startswith("-") for arg in argv[1:]):
        return argparse.Namespace(
            mode="triangulation",
            legacy_inputs=argv[1:],
            serials=None,
            width=640,
            height=480,
            fps=30,
            patch_radius=2,
            min_depth_m=0.1,
            max_depth_m=1.2,
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["triangulation", "depth"], default="triangulation")
    parser.add_argument("--serials", nargs="*", help="RealSense serials for depth mode")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--patch-radius", type=int, default=2)
    parser.add_argument("--min-depth-m", type=float, default=0.1)
    parser.add_argument("--max-depth-m", type=float, default=1.2)
    parser.add_argument("legacy_inputs", nargs="*", help="Legacy triangulation inputs: cam0 cam1")
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_arguments(sys.argv)

    if args.mode == "depth":
        # depth 모드에서는 RealSense 입력으로 카메라별 3D hand pose를 직접 복원한다.
        serials, keypoints_2d_list, keypoints_3d_list, palm_pose_list = run_depth_pipeline(
            serials=args.serials,
            width=args.width,
            height=args.height,
            fps=args.fps,
            patch_radius=args.patch_radius,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
        )
        # save_depth_outputs(serials, keypoints_2d_list, keypoints_3d_list, palm_pose_list)
    else:
        # triangulation 모드에서는 기존처럼 두 영상 입력과 calibration을 사용한다.
        input_stream0 = "media/cam0_test.mp4"
        input_stream1 = "media/cam1_test.mp4"

        if len(args.legacy_inputs) == 2:
            input_stream0 = int(args.legacy_inputs[0])
            input_stream1 = int(args.legacy_inputs[1])

        projection0 = get_projection_matrix(0)
        projection1 = get_projection_matrix(1)

        kpts_cam0, kpts_cam1, kpts_3d = run_triangulation_pipeline(input_stream0, input_stream1, projection0, projection1)
        write_keypoints_to_disk("kpts_cam0.dat", kpts_cam0)
        write_keypoints_to_disk("kpts_cam1.dat", kpts_cam1)
        write_keypoints_to_disk("kpts_3d.dat", kpts_3d)
