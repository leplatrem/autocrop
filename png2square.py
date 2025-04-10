import sys

from PIL import Image
from pathlib import Path

# --- Configuration ---
INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
TARGET_SIZE = None  # Set to a number like 1024 if you want fixed size; None = keep original max size


def pad_to_square_white(png_path, output_path, target_size=None):
    img = Image.open(png_path).convert("RGBA")
    bbox = img.getbbox()
    img = img.crop(bbox)

    width, height = img.size
    max_dim = max(width, height) if target_size is None else target_size

    # Create white background
    square_bg = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))

    # Remove alpha (paste over white)
    rgb_img = Image.new("RGB", img.size, (255, 255, 255))
    rgb_img.paste(img, mask=img.split()[3])  # Use alpha channel as mask

    # Paste on center
    offset = ((max_dim - width) // 2, (max_dim - height) // 2)
    square_bg.paste(rgb_img, offset)

    # Save
    square_bg.save(output_path, format="JPEG", quality=95)


def process_folder(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for png_file in input_path.glob("*.png"):
        jpeg_file = output_path / (png_file.stem + ".jpg")
        pad_to_square_white(png_file, jpeg_file, target_size=TARGET_SIZE)
        print(f"Saved: {jpeg_file.name}")


if __name__ == "__main__":
    process_folder(INPUT_DIR, OUTPUT_DIR)
