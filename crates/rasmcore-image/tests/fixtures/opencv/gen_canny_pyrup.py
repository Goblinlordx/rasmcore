#!/usr/bin/env python3
"""Generate OpenCV reference fixtures for canny edge detection and pyrUp.

Uses canonical 128x128 grayscale test images.
"""

import numpy as np
import cv2
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_IMAGES = [
    "gradient_128",
    "checker_128",
    "noisy_flat_128",
    "sharp_edges_128",
    "photo_128",
    "flat_128",
    "highcontrast_128",
]

W, H = 128, 128


def load_gray(name: str) -> np.ndarray:
    path = os.path.join(SCRIPT_DIR, f"{name}_gray.raw")
    return np.fromfile(path, dtype=np.uint8).reshape((H, W))


def save_raw_u8(name: str, img: np.ndarray):
    path = os.path.join(SCRIPT_DIR, name)
    img.astype(np.uint8).tofile(path)
    print(f"  wrote {name} ({img.size} bytes)")


def main():
    # Canny edge detection with thresholds matching our test params
    low_threshold = 50
    high_threshold = 150

    for img_name in TEST_IMAGES:
        print(f"\n{img_name}:")
        img = load_gray(img_name)

        # Canny edge detection
        canny = cv2.Canny(img, low_threshold, high_threshold)
        save_raw_u8(f"{img_name}_canny_{low_threshold}_{high_threshold}.raw", canny)

    # pyrUp reference (64x64 -> 128x128)
    print("\npyrUp references (64x64 → 128x128):")
    for img_name in TEST_IMAGES:
        img = load_gray(img_name)
        # Downsample to 64x64 first, then pyrUp back to 128x128
        down = cv2.pyrDown(img)  # 128 → 64
        up = cv2.pyrUp(down)     # 64 → 128
        save_raw_u8(f"{img_name}_pyrup_from64.raw", up)
        # Also save the 64x64 source for the test
        save_raw_u8(f"{img_name}_pyrdown_64.raw", down)

    print("\nDone!")


if __name__ == "__main__":
    main()
