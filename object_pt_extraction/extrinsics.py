from dataclasses import dataclass
from pathlib import Path

import numpy as np


TRANSLATION_UNIT_SCALE = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "mm": 0.001,
    "millimeter": 0.001,
    "millimeters": 0.001,
}


@dataclass
class CameraExtrinsic:
    camera_id: int
    reference_camera_id: int
    rotation_ref_to_cam: np.ndarray
    translation_ref_to_cam: np.ndarray
    source_path: str

    @property
    def rotation_world_to_cam(self):
        # 하위 호환용 alias. 이 저장소에서는 world가 camera0 reference frame을 의미한다.
        return self.rotation_ref_to_cam

    @property
    def translation_world_to_cam(self):
        # 하위 호환용 alias. 이 저장소에서는 world가 camera0 reference frame을 의미한다.
        return self.translation_ref_to_cam


def _resolve_translation_scale(translation_unit):
    if isinstance(translation_unit, (int, float)):
        return float(translation_unit)

    normalized_unit = str(translation_unit).strip().lower()
    if normalized_unit not in TRANSLATION_UNIT_SCALE:
        raise ValueError(f"Unsupported translation unit: {translation_unit}")
    return TRANSLATION_UNIT_SCALE[normalized_unit]


def _normalize_points(points_xyz):
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    if points_xyz.ndim == 1:
        if points_xyz.shape[0] != 3:
            raise ValueError("1D point input must have shape (3,)")
        return points_xyz.reshape(1, 3), True
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("Point input must have shape (N, 3) or (3,)")
    return points_xyz, False


def _read_rotation_translation_text(file_path):
    rotation = []
    translation = []

    with open(file_path, "r", encoding="ascii") as input_file:
        header = input_file.readline().strip()
        if header != "R:":
            raise ValueError(f"Unexpected rotation header in {file_path}: {header}")

        for _ in range(3):
            rotation.append([float(value) for value in input_file.readline().split()])

        translation_header = input_file.readline().strip()
        if translation_header != "T:":
            raise ValueError(f"Unexpected translation header in {file_path}: {translation_header}")

        for _ in range(3):
            translation.append([float(value) for value in input_file.readline().split()])

    return np.asarray(rotation, dtype=np.float32), np.asarray(translation, dtype=np.float32).reshape(3)


def load_camera_extrinsic(camera_id, savefolder="camera_parameters", translation_unit="mm"):
    # 이 저장소의 rot_trans_c*.dat는 camera0 reference frame을 기준으로 저장된다.
    # 즉, rot_trans_c1.dat는 camera0 -> camera1 변환이다.
    file_path = Path(savefolder) / f"rot_trans_c{camera_id}.dat"
    rotation_ref_to_cam, translation_ref_to_cam = _read_rotation_translation_text(file_path)
    translation_scale = _resolve_translation_scale(translation_unit)
    return CameraExtrinsic(
        camera_id=int(camera_id),
        reference_camera_id=0,
        rotation_ref_to_cam=rotation_ref_to_cam,
        translation_ref_to_cam=translation_ref_to_cam * translation_scale,
        source_path=str(file_path),
    )


def invert_extrinsic(rotation_ref_to_cam, translation_ref_to_cam):
    # reference(camera0) -> camera 변환을 camera -> reference(camera0)로 뒤집는다.
    rotation_ref_to_cam = np.asarray(rotation_ref_to_cam, dtype=np.float32).reshape(3, 3)
    translation_ref_to_cam = np.asarray(translation_ref_to_cam, dtype=np.float32).reshape(3)
    rotation_cam_to_ref = rotation_ref_to_cam.T
    translation_cam_to_ref = -rotation_cam_to_ref @ translation_ref_to_cam
    return rotation_cam_to_ref.astype(np.float32), translation_cam_to_ref.astype(np.float32)


def transform_points(points_xyz, rotation, translation):
    # 행벡터 입력(Nx3)에 대해 X_dst = R @ X_src + t 를 적용한다.
    normalized_points, squeeze_output = _normalize_points(points_xyz)
    rotation = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float32).reshape(3)
    transformed_points = normalized_points @ rotation.T + translation.reshape(1, 3)
    if squeeze_output:
        return transformed_points[0].astype(np.float32)
    return transformed_points.astype(np.float32)


def convert_reference_to_camera(points_ref, rotation_ref_to_cam, translation_ref_to_cam):
    return transform_points(points_ref, rotation_ref_to_cam, translation_ref_to_cam)


def convert_camera_to_reference(points_cam, rotation_ref_to_cam, translation_ref_to_cam):
    rotation_cam_to_ref, translation_cam_to_ref = invert_extrinsic(
        rotation_ref_to_cam,
        translation_ref_to_cam,
    )
    return transform_points(points_cam, rotation_cam_to_ref, translation_cam_to_ref)


def convert_world_to_cam(points_world, rotation_world_to_cam, translation_world_to_cam):
    # 하위 호환 alias. 이 저장소에서는 world == camera0 reference frame.
    return convert_reference_to_camera(points_world, rotation_world_to_cam, translation_world_to_cam)


def convert_cam_to_world(points_cam, rotation_world_to_cam, translation_world_to_cam):
    # 하위 호환 alias. 이 저장소에서는 world == camera0 reference frame.
    return convert_camera_to_reference(points_cam, rotation_world_to_cam, translation_world_to_cam)


def convert_points_between_cameras(
    points_xyz,
    source_extrinsic,
    target_extrinsic,
):
    # source camera -> camera0 reference -> target camera 순서로 변환한다.
    if source_extrinsic.reference_camera_id != target_extrinsic.reference_camera_id:
        raise ValueError("Both extrinsics must share the same reference camera id")

    points_ref = convert_camera_to_reference(
        points_xyz,
        source_extrinsic.rotation_ref_to_cam,
        source_extrinsic.translation_ref_to_cam,
    )
    return convert_reference_to_camera(
        points_ref,
        target_extrinsic.rotation_ref_to_cam,
        target_extrinsic.translation_ref_to_cam,
    )


def convert_cam1_to_cam0(points_xyz_cam1, extrinsic_cam1=None, extrinsic_cam0=None, savefolder="camera_parameters", translation_unit="mm"):
    if extrinsic_cam1 is None:
        extrinsic_cam1 = load_camera_extrinsic(1, savefolder=savefolder, translation_unit=translation_unit)
    if extrinsic_cam0 is None:
        # camera0는 reference frame이므로 c1 -> c0 변환은 c1 extrinsic을 역변환하면 된다.
        return convert_camera_to_reference(
            points_xyz_cam1,
            extrinsic_cam1.rotation_ref_to_cam,
            extrinsic_cam1.translation_ref_to_cam,
        )

    return convert_points_between_cameras(
        points_xyz_cam1,
        source_extrinsic=extrinsic_cam1,
        target_extrinsic=extrinsic_cam0,
    )


def make_homogeneous_matrix(rotation, translation):
    rotation = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float32).reshape(3)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform
