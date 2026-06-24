import argparse
from pathlib import Path

import cv2
import mediapipe as mp


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "input"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"
DEFAULT_IMAGE_NAMES = ("left.png", "right.png")

MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


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
    detected_hands = 0

    if results.multi_hand_landmarks:
        detected_hands = len(results.multi_hand_landmarks)
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                output,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

    return output, detected_hands


def process_image(image_path, output_dir, hands):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    output, detected_hands = draw_keypoints(image, hands)

    output_path = output_dir / f"{image_path.stem}_keypoints{image_path.suffix}"
    if not cv2.imwrite(str(output_path), output):
        raise RuntimeError(f"Failed to save output image: {output_path}")

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
