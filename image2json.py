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
GROW_MARGIN = 40  # pixels
MIN_AREA = 0.005  # % of image area
MAX_AREA = 0.5
MAX_STICKERS_PER_SHEET = 30
MAX_RES = 800


# --- Detect Best Available Device ---
if torch.backends.mps.is_available():
    DEVICE = "mps"
    """
    This will crash without this ugly in place fix:
    +++ .venv/lib/python3.12/site-packages/segment_anything/automatic_mask_generator.py	2025-04-25 20:02:08
@@ -274,7 +274,7 @@

         # Run model on this batch
         transformed_points = self.predictor.transform.apply_coords(points, im_size)
-        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
+        in_points = torch.as_tensor(transformed_points, dtype=torch.float32, device=self.predictor.device)
         in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
         masks, iou_preds, _ = self.predictor.predict_torch(
             in_points[:, None, :],
    """
    
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")


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


# --- Process one image ---
def process_image(image_path, json_path):
    print(f"Processing {image_path.name}")
    image_bgr = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ✅ Resize image if it's too big (MPS-safe)
    original_h, original_w = image_rgb.shape[:2]
    scale = 1.0
    if max(original_h, original_w) > MAX_RES:
        scale = MAX_RES / max(original_h, original_w)
        new_size = (int(original_w * scale), int(original_h * scale))
        image_rgb = cv2.resize(image_rgb, new_size, interpolation=cv2.INTER_AREA)
        print(f"Resized image to {new_size}, scale = {scale:.3f}")

    download_checkpoint_if_missing(CHECKPOINT_PATH, MODEL_URL)

    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image_rgb)

    image_area = image_rgb.shape[0] * image_rgb.shape[1]
    min_area = MIN_AREA * image_area
    max_area = MAX_AREA * image_area

    # Sort & filter masks
    masks.sort(key=lambda m: m["area"], reverse=True)

    def iou(mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0

    final_masks = []
    for m in masks:
        if m["area"] > max_area or m["area"] < min_area:
            continue
        if all(iou(m["segmentation"], f["segmentation"]) < 0.5 for f in final_masks):
            final_masks.append(m)
        if len(final_masks) >= MAX_STICKERS_PER_SHEET:
            break

    # Extract polygon contours
    output_polygons = []
    for i, mask in enumerate(final_masks):
        binary_mask = mask["segmentation"].astype(np.uint8) * 255
        kernel_size = int(GROW_MARGIN * scale) * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        dilated = cv2.dilate(binary_mask, kernel, iterations=1)

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

        # ✅ Rescale polygon back to original image coordinates
        polygon = [[int(pt[0][0] / scale), int(pt[0][1] / scale)] for pt in approx]
        output_polygons.append({"sticker_id": f"sticker_{i + 1}", "polygon": polygon})

    # Save JSON
    with open(json_path, "w") as f:
        json.dump(output_polygons, f, indent=2)
    print(f"Saved {len(output_polygons)} polygons to {json_path}")


# --- Batch runner ---
def process_folder():
    input_path = INPUT_DIR
    output_path = OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    for image_path in input_path.glob("*.jpeg"):
        out_path = output_path / f"{image_path.stem}.json"
        if (
            not out_path.exists()
            or out_path.stat().st_mtime < image_path.stat().st_mtime
        ):
            process_image(image_path, out_path)


if __name__ == "__main__":
    process_folder()
