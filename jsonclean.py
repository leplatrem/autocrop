import sys
import json
from pathlib import Path
from shapely.geometry import Polygon

INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]


def is_mostly_inside(poly_small, poly_large, threshold=0.8):
    """Return True if poly_small is mostly inside poly_large"""
    try:
        small = Polygon(poly_small)
        large = Polygon(poly_large)
        if not small.is_valid or not large.is_valid:
            return False
        intersection_area = small.intersection(large).area
        return (intersection_area / small.area) > threshold
    except Exception:
        return False


def filter_enclosed(polygons, threshold=0.8):
    keep = []
    for i, outer in enumerate(polygons):
        outer_poly = outer["polygon"]
        is_contained = False
        for j, inner in enumerate(polygons):
            if i == j:
                continue
            if is_mostly_inside(outer_poly, inner["polygon"], threshold):
                is_contained = True
                break
        if not is_contained:
            keep.append(outer)
    return keep


def process_json(json_path, out_path):
    with open(json_path) as f:
        polygons = json.load(f)

    filtered = filter_enclosed(polygons, threshold=0.8)

    with open(out_path, "w") as f:
        json.dump(filtered, f, indent=2)

    print(f"{json_path.name}: kept {len(filtered)} of {len(polygons)} polygons")


def process_folder():
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    for json_path in input_path.glob("*.json"):
        out_path = output_path / json_path.name
        if (
            not out_path.exists()
            or out_path.stat().st_mtime < json_path.stat().st_mtime
        ):
            process_json(json_path, out_path)


if __name__ == "__main__":
    process_folder()
