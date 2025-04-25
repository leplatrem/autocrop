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


import easyocr
import numpy as np
import cv2


from paddleocr import PaddleOCR
import numpy as np
import cv2

import paddle

print("PaddlePaddle version:", paddle.__version__)
print("PaddlePaddle is compiled with CUDA:", paddle.is_compiled_with_cuda())

# Load PaddleOCR model once
ocr = PaddleOCR(use_angle_cls=True, lang="en")  # 'en' or 'ch', etc.


def detect_text_angle(image):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result = ocr.ocr(image_bgr, cls=True)

    if not result or not result[0]:
        print("No text found.")
        return 0

    # Find longest text (based on width of box)
    max_len = 0
    best_box = None
    best_text = ""

    for line in result[0]:
        box, (text, conf) = line
        x1, y1 = box[0]
        x2, y2 = box[1]
        length = np.linalg.norm(np.array([x2 - x1, y2 - y1]))
        if length > max_len:
            max_len = length
            best_box = box
            best_text = text

    if best_box is None:
        return 0

    print(f"Longest text: '{best_text}'")

    # Get angle from top edge of rotated box
    (x1, y1), (x2, y2) = best_box[:2]
    dx = x2 - x1
    dy = y2 - y1
    angle = np.degrees(np.arctan2(dy, dx))

    print(f"Raw angle: {angle:.2f}°")

    # Normalize to range [-180, 180]
    angle = (angle + 180) % 360 - 180

    # Now rotate to make text appear upright and horizontal
    # Threshold to detect verticality
    if -120 < angle < -60:
        angle += 90
    elif 60 < angle < 120:
        angle -= 90
    elif angle <= -150 or angle >= 150:
        angle -= 180

    # Final normalization
    angle = (angle + 180) % 360 - 180
    print(f"Normalized angle: {angle:.2f}°")

    return angle


# reader = easyocr.Reader(['en'])  # load once globally

# def detect_text_angle(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     results = reader.readtext(gray, detail=1, paragraph=False)

#     if not results:
#         print("No text found")
#         return 0

#     # Sort by width (longest word-like)
#     results.sort(key=lambda x: x[0][1][0] - x[0][0][0], reverse=True)
#     box, text, conf = results[0]

#     print(f"Detected text: '{text}' with conf: {conf:.1f}")

#     # Compute angle from box
#     (x1, y1), (x2, y2), *_ = box
#     print(f"Box: {box}")
#     dx = x2 - x1
#     dy = y2 - y1
#     angle = np.degrees(np.arctan2(dy, dx))

#     # Normalize angle
#     if angle < -45:
#         angle += 90
#     elif angle > 45:
#         angle -= 90

#     print(f"Angle: {angle:.2f}°")
#     return angle

# def detect_text_angle(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

#     best_idx = None
#     best_width = 0

#     for i in range(len(data["text"])):
#         text = data["text"][i].strip()
#         if int(data["conf"][i]) > 60 and len(text) > 2:
#             w = data["width"][i]
#             if w > best_width:
#                 best_width = w
#                 best_idx = i

#     if best_idx is None:
#         return None

#     print("text=", data["text"][best_idx], "confidence=", data["conf"][best_idx])

#     # Get the angle of the longest word's bounding box using a rotated rect
#     x = data["left"][best_idx]
#     y = data["top"][best_idx]
#     w = data["width"][best_idx]
#     h = data["height"][best_idx]

#     cropped = gray[y:y+h, x:x+w]
#     _, binary = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#     # Find non-zero points (the ink)
#     coords = cv2.findNonZero(binary)
#     if coords is None or len(coords) < 5:
#         return 0

#     rect = cv2.minAreaRect(coords)
#     angle = rect[-1]

#     # Normalize angle to [-45, +45]
#     if angle < -45:
#         angle += 90

#     print(f"Detected angle from longest word '{data['text'][best_idx]}' = {angle:.2f}°")
#     return angle


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

    for image_path in input_path.glob("*.jpeg"):
        out_path = output_path / image_path.name
        if (
            not out_path.exists()
            or out_path.stat().st_mtime < image_path.stat().st_mtime
        ):
            process_image(image_path, out_path)


if __name__ == "__main__":
    process_folder()
