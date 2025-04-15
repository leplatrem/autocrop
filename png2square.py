import sys
from PIL import Image, ImageFilter
from pathlib import Path

# --- Configuration ---
INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
TARGET_SIZE = 1280  # Set to a number like 1024 if you want fixed size; None = keep original max size
FEATHER_RADIUS = 0  # Pixels of feathering around edges


def add_feathered_mask(img, radius):
    """Apply a feathered alpha mask to soften edges."""
    alpha = img.split()[3]

    # Create a mask the same size, initially black (transparent)
    mask = Image.new("L", img.size, 0)

    # Paste original alpha into center of new mask
    mask.paste(alpha)

    # Feather the mask
    feathered = mask.filter(ImageFilter.GaussianBlur(radius))

    # Apply new feathered alpha to image
    img.putalpha(feathered)
    return img


def process_image(png_path, output_path):
    img = Image.open(png_path).convert("RGBA")
    bbox = img.getbbox()
    img = img.crop(bbox)

    # Apply feather effect to alpha
    if FEATHER_RADIUS:
        img = add_feathered_mask(img, FEATHER_RADIUS)

    width, height = img.size
    max_dim = max(max(width, height), TARGET_SIZE)

    # Create an empty image for compositing (RGBA)
    composite_img = Image.new("RGBA", (max_dim, max_dim), (255, 255, 255, 0))
    offset = ((max_dim - width) // 2, (max_dim - height) // 2)
    composite_img.paste(img, offset, img)

    # Convert to RGB (with white background)
    final_img = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    final_img.paste(composite_img, mask=composite_img.split()[3])

    # Save
    filename = output_path.with_suffix(".jpeg")
    final_img.save(filename, format="JPEG", quality=95)
    print(f"Saved: {filename}")


def process_folder():
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    for image_path in input_path.glob("*.png"):
        out_path = output_path / image_path.with_suffix(".jpg").name
        if (
            not out_path.exists()
            or out_path.stat().st_mtime < image_path.stat().st_mtime
        ):
            process_image(image_path, out_path)


if __name__ == "__main__":
    process_folder()
