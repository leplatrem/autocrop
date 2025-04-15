import sys

import cv2
import numpy as np
import torch
import json
import requests
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# --- Configuration ---
INPUT_DIR = Path(sys.argv[1])
OUTPUT_DIR = Path(sys.argv[2])

MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
MAX_EDGES = 1000
GROW_MARGIN = 30  # pixels radius: tune as needed
MIN_AREA = 0.01  # percent
MAX_AREA = 0.5  # percent
MAX_STICKERS_PER_SHEET = 30


# --- Download SAM model if not present ---
def download_checkpoint_if_missing(path, url):
    checkpoint_file = Path(path)
    if not checkpoint_file.exists():
        print(f"Downloading SAM checkpoint from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(checkpoint_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    else:
        print("Checkpoint already exists.")


# --- Main processing function ---
def process_image(image_path, json_path):
    print("Process", image_path)
    # Load image
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Download model if missing
    download_checkpoint_if_missing(CHECKPOINT_PATH, MODEL_URL)

    # Load SAM model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # Generate masks
    masks = mask_generator.generate(image_rgb)

    image_area = image_rgb.shape[0] * image_rgb.shape[1]
    min_area = MIN_AREA * image_area
    max_area = MAX_AREA * image_area

    # Sort by area (largest first)
    masks.sort(key=lambda m: m["area"], reverse=True)

    # Deduplicate overlapping masks using IoU
    def iou(mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0

    final_masks = []
    for m in masks:
        if m["area"] > max_area:
            continue
        if m["area"] < min_area:
            break
        if all(iou(m["segmentation"], f["segmentation"]) < 0.5 for f in final_masks):
            final_masks.append(m)
        if len(final_masks) >= MAX_STICKERS_PER_SHEET:  # limit to ~8-12 stickers
            break
    masks = final_masks

    # Extract polygons
    output_polygons = []
    for i, mask in enumerate(masks):
        # Step 1: Convert and dilate mask to grow it
        binary_mask = mask["segmentation"].astype(np.uint8) * 255
        # kernel_size = 2 * desired_margin_pixels + 1
        kernel_size = GROW_MARGIN * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        dilated = cv2.dilate(binary_mask, kernel, iterations=1)

        # Step 2: Find contour from the dilated mask
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)

        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) > MAX_EDGES:
            indices = np.linspace(0, len(approx) - 1, MAX_EDGES, dtype=int)
            approx = approx[indices]

        polygon = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]
        output_polygons.append({"sticker_id": f"sticker_{i + 1}", "polygon": polygon})

    # Save JSON
    with open(json_path, "w") as f:
        json.dump(output_polygons, f, indent=2)
    print(f"Saved {len(output_polygons)} polygons to {json_path}")


def process_folder():
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    for image_path in input_path.glob("*.jpeg"):
        out_path = OUTPUT_DIR / f"{image_path.stem}.json"
        if (
            not out_path.exists()
            or out_path.stat().st_mtime < image_path.stat().st_mtime
        ):
            process_image(image_path, out_path)


# --- Run ---
if __name__ == "__main__":
    process_folder()
