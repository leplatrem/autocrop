import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import pytesseract

INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]


def rotate_image_bound(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(rot_mat[0, 0])
    sin = abs(rot_mat[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(image, rot_mat, (new_w, new_h), borderValue=(255, 255, 255))


def detect_text_angle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

    xs = []
    ys = []
    widths = []
    heights = []

    for i in range(len(data["text"])):
        if int(data["conf"][i]) > 60 and data["text"][i].strip():
            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]
            xs.append(x + w // 2)
            ys.append(y + h // 2)
            widths.append(w)
            heights.append(h)

    if len(xs) < 5:
        return None  # Not enough text detected

    points = np.array(list(zip(xs, ys)))
    rect = cv2.minAreaRect(points)
    angle = rect[-1]

    # Normalize angle: OpenCV gives [-90, 0)
    if angle < -45:
        angle += 90

    return angle


def process_image(png_path, output_path):
    image = cv2.imread(str(png_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    angle = detect_text_angle(image)
    if angle is None or abs(angle) < 1:
        Image.fromarray(image).save(output_path)
        print(f"Skipped {png_path.name} (angle: {angle})")
        return

    rotated = rotate_image_bound(image, angle)
    Image.fromarray(rotated).save(output_path)
    print(f"Rotated {png_path.name} by {angle:.1f}° → {output_path.name}")


def process_folder():
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    for image_path in input_path.glob("*.png"):
        out_path = output_path / image_path.name
        if (
            not out_path.exists()
            or out_path.stat().st_mtime < image_path.stat().st_mtime
        ):
            process_image(image_path, out_path)


if __name__ == "__main__":
    process_folder()
