import sys
import cv2
import numpy as np
import json
from pathlib import Path
from PIL import Image

# --- Configuration ---
INPUT_IMAGE_DIR = sys.argv[1]
INPUT_JSON_DIR = sys.argv[2]
OUTPUT_DIR = sys.argv[3]
GROW_PIXELS = 20  # How much to grow the polygon outward


def grow_polygon_mask(image_shape, polygon, grow_pixels):
    # Create initial mask
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)

    # Create structuring element and dilate
    kernel_size = grow_pixels * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask, kernel, iterations=1)

    # Get new contour from the dilated mask
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour
    else:
        return np.array(polygon, dtype=np.int32)  # fallback


def extract_sticker(image_path, polygons, output_folder, base_name):
    image_bgr = cv2.imread(str(image_path))
    image_rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGBA)

    for sticker in polygons:
        original_poly = sticker["polygon"]

        # Grow the polygon
        grown_contour = grow_polygon_mask(image_rgba.shape, original_poly, GROW_PIXELS)

        # Create new mask
        mask = np.zeros(image_rgba.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [grown_contour], 255)

        # Apply mask
        sticker_rgba = image_rgba.copy()
        sticker_rgba[:, :, 3] = mask

        # Crop bounding box
        x, y, w, h = cv2.boundingRect(grown_contour)
        cropped = sticker_rgba[y : y + h, x : x + w]

        # Save
        out_path = output_folder / f"{base_name}_{sticker['sticker_id']}.png"
        Image.fromarray(cropped).save(out_path)
        print(f"Saved: {out_path.name}")


def process_folder():
    input_images = Path(INPUT_IMAGE_DIR)
    input_jsons = Path(INPUT_JSON_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    for json_file in input_jsons.glob("*.json"):
        base_name = json_file.stem.replace("_polygons", "")
        image_path = input_images / f"{base_name}.jpeg"

        if not image_path.exists():
            print(f"Image not found for {json_file.name}, skipping...")
            continue

        with open(json_file) as f:
            polygons = json.load(f)

        extract_sticker(image_path, polygons, output_path, base_name)


if __name__ == "__main__":
    process_folder()
