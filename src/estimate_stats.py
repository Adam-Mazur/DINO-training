from pathlib import Path
from PIL import Image
import numpy as np
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate simple RGB and size stats")
    parser.add_argument("root", type=Path, help="Directory to scan recursively")
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=100,
        help="Number of images to use (default: 100)",
    )
    return parser.parse_args()


def find_images(root: Path, limit: int) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    found: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            found.append(path)
            if len(found) >= limit:
                break
    return found


def main() -> None:
    args = parse_args()
    if not args.root.is_dir():
        raise SystemExit(f"Missing directory: {args.root}")
    if args.num <= 0:
        raise SystemExit("--num must be positive")

    images = find_images(args.root, args.num)
    if not images:
        raise SystemExit("No images found")

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    widths: list[int] = []
    heights: list[int] = []

    for path in images:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            arr = np.asarray(rgb, dtype=np.float32) / 255.0
        h, w, _ = arr.shape
        widths.append(w)
        heights.append(h)
        pixels = h * w
        total_pixels += pixels
        reshaped = arr.reshape(-1, 3)
        channel_sum += reshaped.sum(axis=0)
        channel_sq_sum += np.square(reshaped).sum(axis=0)

    mean = channel_sum / total_pixels
    variance = (channel_sq_sum / total_pixels) - mean**2
    std = np.sqrt(np.clip(variance, 0.0, None))

    print(f"Images used: {len(images)}")
    print(f"Mean  (R,G,B): {mean}")
    print(f"Std   (R,G,B): {std}")
    width_avg = sum(widths) / len(widths)
    height_avg = sum(heights) / len(heights)
    print(f"Width  min/avg/max: {min(widths)} / {width_avg:.2f} / {max(widths)}")
    print(f"Height min/avg/max: {min(heights)} / {height_avg:.2f} / {max(heights)}")


if __name__ == "__main__":
    main()
