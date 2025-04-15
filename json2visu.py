import sys
import cv2
import numpy as np
import json
from pathlib import Path

# --- Configuration ---
INPUT_IMAGE_DIR = sys.argv[1]
INPUT_JSON_DIR = sys.argv[2]
OUTPUT_DIR = sys.argv[3]
FILL_COLOR = (0, 255, 0)
ALPHA = 0.5


def visualize_polygons(image_path, polygons, output_path):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"Could not read image: {image_path}")
        return

    overlay = image_bgr.copy()

    for sticker in polygons:
        polygon = np.array(sticker["polygon"], dtype=np.int32)
        cv2.fillPoly(overlay, [polygon], color=FILL_COLOR)
        cv2.polylines(
            image_bgr, [polygon], isClosed=True, color=(255, 0, 0), thickness=2
        )

    # Blend overlay with original image
    blended = cv2.addWeighted(overlay, ALPHA, image_bgr, 1 - ALPHA, 0)
    cv2.imwrite(str(output_path), blended)
    print(f"Saved visualization to: {output_path}")


def process_folder():
    input_images = Path(INPUT_IMAGE_DIR)
    input_jsons = Path(INPUT_JSON_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for json_file in input_jsons.glob("*.json"):
        base_name = json_file.stem.replace("_polygons", "")
        image_path = input_images / f"{base_name}.jpeg"
        output_path = output_dir / f"{base_name}.jpeg"

        if not image_path.exists():
            print(f"Image not found for {json_file.name}, skipping...")
            continue

        with open(json_file) as f:
            polygons = json.load(f)

        if (
            not output_path.exists()
            or output_path.stat().st_mtime < json_file.stat().st_mtime
        ):
            visualize_polygons(image_path, polygons, output_path)


if __name__ == "__main__":
    process_folder()
