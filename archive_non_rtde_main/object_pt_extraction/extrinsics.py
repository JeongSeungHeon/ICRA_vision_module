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
    rotation_cam_to_ref: np.ndarray
    translation_cam_to_ref: np.ndarray
    source_path: str

    @property
    def rotation_ref_to_cam(self):
        rotation_ref_to_cam, _ = invert_extrinsic(
            self.rotation_cam_to_ref,
            self.translation_cam_to_ref,
        )
        return rotation_ref_to_cam

    @property
    def translation_ref_to_cam(self):
        _, translation_ref_to_cam = invert_extrinsic(
            self.rotation_cam_to_ref,
            self.translation_cam_to_ref,
        )
        return translation_ref_to_cam

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


def load_camera_extrinsic(camera_id, savefolder="camera_parameters", translation_unit="m"):
    # 현재 calibration 파일은 camera -> camera0(reference) 형태의 [R|t]를 저장한다고 가정한다.
    # 즉, rot_trans_c1.dat는 camera1 -> camera0 변환이다.
    file_path = Path(savefolder) / f"rot_trans_c{camera_id}.dat"
    rotation_cam_to_ref, translation_cam_to_ref = _read_rotation_translation_text(file_path)
    translation_scale = _resolve_translation_scale(translation_unit)
    return CameraExtrinsic(
        camera_id=int(camera_id),
        reference_camera_id=0,
        rotation_cam_to_ref=rotation_cam_to_ref,
        translation_cam_to_ref=translation_cam_to_ref * translation_scale,
        source_path=str(file_path),
    )


def invert_extrinsic(rotation_cam_to_ref, translation_cam_to_ref):
    # camera -> reference(camera0) 변환을 reference(camera0) -> camera 변환으로 뒤집는다.
    rotation_cam_to_ref = np.asarray(rotation_cam_to_ref, dtype=np.float32).reshape(3, 3)
    translation_cam_to_ref = np.asarray(translation_cam_to_ref, dtype=np.float32).reshape(3)
    rotation_ref_to_cam = rotation_cam_to_ref.T
    translation_ref_to_cam = -rotation_ref_to_cam @ translation_cam_to_ref
    return rotation_ref_to_cam.astype(np.float32), translation_ref_to_cam.astype(np.float32)


def transform_points(points_xyz, rotation, translation):
    # 행벡터 입력(Nx3)에 대해 X_dst = R @ X_src + t 를 적용한다.
    normalized_points, squeeze_output = _normalize_points(points_xyz)
    rotation = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float32).reshape(3)
    transformed_points = normalized_points @ rotation.T + translation.reshape(1, 3)
    if squeeze_output:
        return transformed_points[0].astype(np.float32)
    return transformed_points.astype(np.float32)


def convert_reference_to_camera(points_ref, rotation_cam_to_ref, translation_cam_to_ref):
    rotation_ref_to_cam, translation_ref_to_cam = invert_extrinsic(
        rotation_cam_to_ref,
        translation_cam_to_ref,
    )
    return transform_points(points_ref, rotation_ref_to_cam, translation_ref_to_cam)


def convert_camera_to_reference(points_cam, rotation_cam_to_ref, translation_cam_to_ref):
    return transform_points(points_cam, rotation_cam_to_ref, translation_cam_to_ref)


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
        source_extrinsic.rotation_cam_to_ref,
        source_extrinsic.translation_cam_to_ref,
    )
    return convert_reference_to_camera(
        points_ref,
        target_extrinsic.rotation_cam_to_ref,
        target_extrinsic.translation_cam_to_ref,
    )


def convert_cam1_to_cam0(points_xyz_cam1, extrinsic_cam1=None, extrinsic_cam0=None, savefolder="camera_parameters", translation_unit="m"):
    if extrinsic_cam1 is None:
        extrinsic_cam1 = load_camera_extrinsic(1, savefolder=savefolder, translation_unit=translation_unit)
    if extrinsic_cam0 is None:
        # camera0는 reference frame이므로 c1 -> c0 변환은 c1 extrinsic을 그대로 적용하면 된다.
        return convert_camera_to_reference(
            points_xyz_cam1,
            extrinsic_cam1.rotation_cam_to_ref,
            extrinsic_cam1.translation_cam_to_ref,
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
