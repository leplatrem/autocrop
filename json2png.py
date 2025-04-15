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
FEATHER_SIZE = 7  # Feather size in pixels


def extract_sticker(image_path, polygons, output_folder, base_name):
    image_bgr = cv2.imread(str(image_path))
    image_rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGBA)

    for sticker in polygons:
        out_path = output_folder / f"{base_name}_{sticker['sticker_id']}.png"
        if out_path.exists():
            continue

        polygon = np.array(sticker["polygon"], dtype=np.int32)

        # Create mask from original polygon
        mask = np.zeros(image_rgba.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)

        # Apply feathering to the mask
        if FEATHER_SIZE > 0:
            # Blur the mask to create feathered edges
            mask_feathered = cv2.GaussianBlur(
                mask, (FEATHER_SIZE * 2 + 1, FEATHER_SIZE * 2 + 1), 0
            )
        else:
            mask_feathered = mask

        # Apply mask to alpha channel
        sticker_rgba = image_rgba.copy()
        sticker_rgba[:, :, 3] = mask_feathered

        # Crop bounding box
        x, y, w, h = cv2.boundingRect(polygon)
        cropped = sticker_rgba[y : y + h, x : x + w]

        # Save
        Image.fromarray(cropped).save(out_path)
        print(f"Saved: {out_path.name}")


def process_folder():
    input_images = Path(INPUT_IMAGE_DIR)
    input_jsons = Path(INPUT_JSON_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    for json_file in input_jsons.glob("*.json"):
        image_path = input_images / f"{json_file.stem}.jpeg"

        if not image_path.exists():
            print(f"Image not found for {json_file.name}, skipping...")
            continue

        with open(json_file) as f:
            polygons = json.load(f)

        extract_sticker(image_path, polygons, output_path, json_file.stem)


if __name__ == "__main__":
    process_folder()
