import cv2
import sys
import numpy as np
from pathlib import Path
from PIL import Image

INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]


def find_average_white_point(image_rgb, top_percent=1):
    # Convert to HSV and extract brightness channel
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    value_channel = hsv[:, :, 2]

    # Flatten and sort brightness values
    flat_values = value_channel.flatten()
    threshold = np.percentile(flat_values, 100 - top_percent)

    # Create mask of top X% brightest pixels
    mask = value_channel >= threshold

    # Apply mask to original RGB image
    white_pixels = image_rgb[mask]

    # Compute average RGB of the brightest pixels
    average_white = np.mean(white_pixels, axis=0)
    return average_white.astype(np.uint8)


def stretch_levels(image_rgb, white_point):
    # Normalize each channel to 0–255 using white_point as max
    image_float = image_rgb.astype(np.float32)
    white_point = np.clip(white_point, 1, 255)  # Avoid division by zero
    stretched = np.clip((image_float / white_point) * 255.0, 0, 255)
    return stretched.astype(np.uint8)


def process_image(image_path, out_path):
    # Load image with OpenCV and convert to RGB
    image_bgr = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Step 1: Detect brightest point (white selector mimic)
    white_point = find_average_white_point(image_rgb)

    # Step 2: Apply GIMP-like level stretching
    adjusted = stretch_levels(image_rgb, white_point)

    # Save result
    Image.fromarray(adjusted).save(out_path)
    print(f"Saved: {out_path}")


def process_folder():
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    for image_path in input_path.glob("*.jpeg"):
        out_path = Path(OUTPUT_DIR) / image_path.name
        if (
            not out_path.exists()
            or out_path.stat().st_mtime < image_path.stat().st_mtime
        ):
            process_image(image_path, out_path)


if __name__ == "__main__":
    process_folder()
