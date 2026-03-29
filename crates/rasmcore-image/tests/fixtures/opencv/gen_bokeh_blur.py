#!/usr/bin/env python3
"""Generate OpenCV cv2.filter2D reference fixtures for bokeh blur parity tests.

Uses the canonical 128×128 grayscale test images. Creates disc and hexagonal
kernels matching the Rust implementation, applies cv2.filter2D with
BORDER_REFLECT_101, and saves reference outputs.

Fixture naming: {image}_bokeh_{shape}_{radius}.raw (raw grayscale bytes)
"""

import numpy as np
import cv2
import os
import math

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


def make_disc_kernel(radius: int) -> np.ndarray:
    """Flat disc kernel matching the Rust make_disc_kernel()."""
    side = radius * 2 + 1
    center = float(radius)
    r2 = (radius + 0.5) ** 2
    kernel = np.zeros((side, side), dtype=np.float32)
    for y in range(side):
        for x in range(side):
            dx = x - center
            dy = y - center
            if dx * dx + dy * dy <= r2:
                kernel[y, x] = 1.0
    return kernel / kernel.sum()


def make_hex_kernel(radius: int) -> np.ndarray:
    """Flat hexagonal kernel matching the Rust make_hex_kernel()."""
    side = radius * 2 + 1
    center = float(radius)
    cr = radius + 0.5  # circumradius
    h = cr * (math.sqrt(3.0) / 2.0)
    kernel = np.zeros((side, side), dtype=np.float32)
    for y in range(side):
        for x in range(side):
            dx = abs(x - center)
            dy = abs(y - center)
            if dy <= h and dy * 0.5 + dx * (math.sqrt(3.0) / 2.0) <= h:
                kernel[y, x] = 1.0
    return kernel / kernel.sum()


def main():
    test_configs = [
        ("disc", 3, make_disc_kernel),
        ("disc", 7, make_disc_kernel),
        ("hex", 3, make_hex_kernel),
        ("hex", 7, make_hex_kernel),
    ]

    for img_name in TEST_IMAGES:
        print(f"\n{img_name}:")
        img = load_gray(img_name)

        for shape_name, radius, make_fn in test_configs:
            kernel = make_fn(radius)
            result = cv2.filter2D(
                img,
                -1,
                kernel,
                borderType=cv2.BORDER_REFLECT_101,
            )
            save_raw_u8(
                f"{img_name}_bokeh_{shape_name}_{radius}.raw", result
            )

    print("\nDone! All bokeh blur fixtures generated.")


if __name__ == "__main__":
    main()
