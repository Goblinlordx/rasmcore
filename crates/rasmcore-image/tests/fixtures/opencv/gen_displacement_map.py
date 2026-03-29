#!/usr/bin/env python3
"""Generate OpenCV cv2.remap reference fixtures for displacement map parity tests.

Uses the canonical 128×128 grayscale test images. Creates displacement maps
(radial barrel distortion and sinusoidal wave) and applies cv2.remap with
INTER_LINEAR + BORDER_CONSTANT to produce reference outputs.

Fixture naming: {image}_{warp_type}.raw  (raw grayscale bytes)
Map files:      displace_{warp_type}_map_x.raw, displace_{warp_type}_map_y.raw  (f32 LE)
"""

import numpy as np
import cv2
import os
import struct

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
    data = np.fromfile(path, dtype=np.uint8)
    return data.reshape((H, W))


def save_raw_u8(name: str, img: np.ndarray):
    path = os.path.join(SCRIPT_DIR, name)
    img.astype(np.uint8).tofile(path)
    print(f"  wrote {name} ({img.size} bytes)")


def save_raw_f32(name: str, data: np.ndarray):
    path = os.path.join(SCRIPT_DIR, name)
    data.astype(np.float32).tofile(path)
    print(f"  wrote {name} ({data.size * 4} bytes)")


def make_barrel_maps(w: int, h: int, k: float = 0.0005):
    """Radial barrel distortion: r' = r * (1 + k * r^2).

    Maps output (x,y) → source (sx,sy). Moderate distortion visible at edges.
    """
    cx, cy = w / 2.0, h / 2.0
    y_idx, x_idx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = x_idx - cx
    dy = y_idx - cy
    r2 = dx * dx + dy * dy
    factor = 1.0 + k * r2
    map_x = cx + dx * factor
    map_y = cy + dy * factor
    return map_x.astype(np.float32), map_y.astype(np.float32)


def make_wave_maps(w: int, h: int, amplitude: float = 5.0, freq: float = 0.08):
    """Sinusoidal wave distortion in x direction.

    map_x = x + amplitude * sin(freq * 2π * y)
    map_y = y
    """
    y_idx, x_idx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = x_idx + amplitude * np.sin(freq * 2.0 * np.pi * y_idx)
    map_y = y_idx.copy()
    return map_x.astype(np.float32), map_y.astype(np.float32)


def main():
    warp_types = {
        "barrel": make_barrel_maps,
        "wave": make_wave_maps,
    }

    # Save displacement maps (same for all images)
    for warp_name, make_fn in warp_types.items():
        map_x, map_y = make_fn(W, H)
        save_raw_f32(f"displace_{warp_name}_map_x.raw", map_x)
        save_raw_f32(f"displace_{warp_name}_map_y.raw", map_y)

    # Generate reference outputs for each image × warp
    for img_name in TEST_IMAGES:
        print(f"\n{img_name}:")
        img = load_gray(img_name)

        for warp_name, make_fn in warp_types.items():
            map_x, map_y = make_fn(W, H)
            result = cv2.remap(
                img,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            save_raw_u8(f"{img_name}_displace_{warp_name}.raw", result)

    print("\nDone! All displacement map fixtures generated.")


if __name__ == "__main__":
    main()
