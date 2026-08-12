import argparse
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "input"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"
DEFAULT_IMAGE_NAMES = ("left2.png", "right2_.png")

MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5
KEYPOINT_RADIUS = 2
KEYPOINT_COLOR = (0, 0, 255)
LINE_COLOR = (0, 255, 0)
LINE_THICKNESS = 2
OPAQUE_ALPHA = 255


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def get_hand_connection_style():
    return {
        connection: mp_drawing.DrawingSpec(
            color=LINE_COLOR,
            thickness=LINE_THICKNESS,
        )
        for connection in mp_hands.HAND_CONNECTIONS
    }


def normalized_to_pixel_coordinates(normalized_x, normalized_y, image_width, image_height):
    if not (0 <= normalized_x <= 1 and 0 <= normalized_y <= 1):
        return None

    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def with_alpha(color_bgr):
    return (*color_bgr, OPAQUE_ALPHA)


def draw_keypoint_dots(image, hand_landmarks, color=KEYPOINT_COLOR):
    image_height, image_width = image.shape[:2]
    for landmark in hand_landmarks.landmark:
        landmark_px = normalized_to_pixel_coordinates(
            landmark.x,
            landmark.y,
            image_width,
            image_height,
        )
        if landmark_px:
            cv2.circle(
                image,
                landmark_px,
                KEYPOINT_RADIUS,
                color,
                thickness=-1,
            )


def draw_hand_connections(image, hand_landmarks, color=LINE_COLOR):
    image_height, image_width = image.shape[:2]
    for start_idx, end_idx in mp_hands.HAND_CONNECTIONS:
        start_landmark = hand_landmarks.landmark[start_idx]
        end_landmark = hand_landmarks.landmark[end_idx]
        start_px = normalized_to_pixel_coordinates(
            start_landmark.x,
            start_landmark.y,
            image_width,
            image_height,
        )
        end_px = normalized_to_pixel_coordinates(
            end_landmark.x,
            end_landmark.y,
            image_width,
            image_height,
        )
        if start_px and end_px:
            cv2.line(image, start_px, end_px, color, LINE_THICKNESS)


def draw_transparent_keypoints(image_shape, multi_hand_landmarks):
    image_height, image_width = image_shape[:2]
    output = np.zeros((image_height, image_width, 4), dtype=np.uint8)

    if not multi_hand_landmarks:
        return output

    for hand_landmarks in multi_hand_landmarks:
        draw_hand_connections(output, hand_landmarks, with_alpha(LINE_COLOR))
        draw_keypoint_dots(output, hand_landmarks, with_alpha(KEYPOINT_COLOR))

    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overlay MediaPipe 2D hand keypoints on two RGB images."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing the input images.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where keypoint overlay images will be saved.",
    )
    parser.add_argument(
        "--images",
        nargs=2,
        default=DEFAULT_IMAGE_NAMES,
        metavar=("IMAGE0", "IMAGE1"),
        help="Two image filenames inside --input-dir.",
    )
    parser.add_argument(
        "--max-num-hands",
        type=int,
        default=MAX_NUM_HANDS,
        help="Maximum number of hands to detect per image.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=MIN_DETECTION_CONFIDENCE,
        help="MediaPipe minimum detection confidence.",
    )
    return parser.parse_args()


def draw_keypoints(image_bgr, hands):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True

    output = image_bgr.copy()
    transparent_output = np.zeros((*image_bgr.shape[:2], 4), dtype=np.uint8)
    detected_hands = 0

    if results.multi_hand_landmarks:
        detected_hands = len(results.multi_hand_landmarks)
        transparent_output = draw_transparent_keypoints(
            image_bgr.shape,
            results.multi_hand_landmarks,
        )
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                output,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                None,
                get_hand_connection_style(),
            )
            draw_keypoint_dots(output, hand_landmarks)

    return output, transparent_output, detected_hands


def process_image(image_path, output_dir, hands):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    output, transparent_output, detected_hands = draw_keypoints(image, hands)

    output_path = output_dir / f"{image_path.stem}_keypoints{image_path.suffix}"
    transparent_output_path = output_dir / f"{image_path.stem}_keypoints_transparent.png"
    if not cv2.imwrite(str(output_path), output):
        raise RuntimeError(f"Failed to save output image: {output_path}")
    if not cv2.imwrite(str(transparent_output_path), transparent_output):
        raise RuntimeError(
            f"Failed to save transparent output image: {transparent_output_path}"
        )

    input_height, input_width = image.shape[:2]
    output_height, output_width = output.shape[:2]
    if (input_width, input_height) != (output_width, output_height):
        raise RuntimeError(
            "Output size changed: "
            f"input={input_width}x{input_height}, output={output_width}x{output_height}"
        )

    print(
        f"[SAVE] {output_path} "
        f"size={output_width}x{output_height}, detected_hands={detected_hands}"
    )
    print(
        f"[SAVE] {transparent_output_path} "
        f"size={output_width}x{output_height}, detected_hands={detected_hands}"
    )


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [input_dir / image_name for image_name in args.images]

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=args.max_num_hands,
        min_detection_confidence=args.min_detection_confidence,
    ) as hands:
        for image_path in image_paths:
            process_image(image_path, output_dir, hands)


if __name__ == "__main__":
    main()
