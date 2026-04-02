#!/usr/bin/env python3
"""
Generate reference outputs for distortion filter parity tests.

Uses numpy + PIL to implement each distortion independently of the Rust code.
These reference implementations use bilinear interpolation with edge clamping.

Prerequisites:
  tests/fixtures/.venv/bin/pip install numpy Pillow

Usage:
  tests/fixtures/.venv/bin/python3 tests/fixtures/scripts/distortion_reference.py
"""

import os
import sys

try:
    import numpy as np
    from PIL import Image
except ImportError:
    print("ERROR: numpy and Pillow required. Install via:")
    print("  tests/fixtures/.venv/bin/pip install numpy Pillow")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_DIR = os.path.join(SCRIPT_DIR, "..", "generated")
INPUT_DIR = os.path.join(FIXTURES_DIR, "inputs")
OUTPUT_DIR = os.path.join(FIXTURES_DIR, "reference")


def _fetch_pixel(img, px, py, border="zero"):
    """Fetch pixel with border handling."""
    h, w = img.shape[:2]
    if border == "clamp":
        px = max(0, min(px, w - 1))
        py = max(0, min(py, h - 1))
        return img[py, px].astype(np.float64)
    else:  # zero
        if 0 <= px < w and 0 <= py < h:
            return img[py, px].astype(np.float64)
        return np.zeros(img.shape[2] if img.ndim == 3 else 1, dtype=np.float64)


def bilinear_sample(img: np.ndarray, sx: float, sy: float, border="zero") -> np.ndarray:
    """Bilinear interpolation with configurable border handling.
    border='zero' matches the Rust EwaSampler::bilinear (returns 0 for OOB).
    border='clamp' matches EwaSampler::fetch_clamp (edge repeat)."""
    x0 = int(np.floor(sx))
    y0 = int(np.floor(sy))
    fx = sx - x0
    fy = sy - y0
    val = (
        _fetch_pixel(img, x0, y0, border) * (1 - fx) * (1 - fy)
        + _fetch_pixel(img, x0 + 1, y0, border) * fx * (1 - fy)
        + _fetch_pixel(img, x0, y0 + 1, border) * (1 - fx) * fy
        + _fetch_pixel(img, x0 + 1, y0 + 1, border) * fx * fy
    )
    return val


def apply_distortion(img: np.ndarray, inverse_fn, border="zero") -> np.ndarray:
    """Apply distortion using inverse mapping with bilinear interpolation.
    border: 'zero' (default, matches Rust Bilinear/Ewa) or 'clamp' (matches EwaClamp)."""
    h, w = img.shape[:2]
    out = np.zeros_like(img, dtype=np.float64)
    for y in range(h):
        for x in range(w):
            sx, sy = inverse_fn(float(x), float(y), w, h)
            out[y, x] = bilinear_sample(img.astype(np.float64), sx, sy, border)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def spherize_inverse(x, y, w, h):
    """Spherize with amount=0.5, powf-based radius remapping."""
    cx = w * 0.5
    cy = h * 0.5
    radius = min(cx, cy)
    dx = (x - cx) / radius
    dy = (y - cy) / radius
    r = np.sqrt(dx * dx + dy * dy)
    if r >= 1.0 or r == 0.0:
        return (x, y)
    # amount=0.5 (positive = bulge): new_r = r^(1/(1+amount))
    amount = 0.5
    new_r = r ** (1.0 / (1.0 + amount))
    scale = new_r / r
    return (dx * scale * radius + cx, dy * scale * radius + cy)


def wave_inverse(x, y, w, h):
    """Horizontal wave: displaces y by amplitude*sin(2pi*x/wavelength).
    amplitude=10, wavelength=50, vertical=0 (horizontal wave)."""
    amplitude = 10.0
    wavelength = 50.0
    two_pi = 2.0 * np.pi
    # horizontal wave: shift y based on x
    return (x, y - amplitude * np.sin(two_pi * x / wavelength))


def ripple_inverse(x, y, w, h):
    """Concentric ripple: amplitude=8, wavelength=40, center=0.5,0.5."""
    amplitude = 8.0
    wavelength = 40.0
    cx = 0.5 * w
    cy = 0.5 * h
    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx * dx + dy * dy)
    if r < 1e-6:
        return (x, y)
    two_pi = 2.0 * np.pi
    disp = amplitude * np.sin(two_pi * r / wavelength)
    cos_a = dx / r
    sin_a = dy / r
    return (x + disp * cos_a, y + disp * sin_a)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_path = os.path.join(INPUT_DIR, "gradient_64x64_8bit.png")
    if not os.path.exists(input_path):
        print(f"ERROR: Input fixture not found: {input_path}")
        print("Run tests/fixtures/generate.sh first.")
        sys.exit(1)

    img = np.array(Image.open(input_path).convert("RGB"))

    # Spherize (amount=0.5)
    print("  Generating spherize reference...")
    spherize_out = apply_distortion(img, spherize_inverse)
    Image.fromarray(spherize_out).save(
        os.path.join(OUTPUT_DIR, "distort_spherize_05.png")
    )

    # Wave (amplitude=10, wavelength=50, horizontal)
    print("  Generating wave reference...")
    wave_out = apply_distortion(img, wave_inverse)
    Image.fromarray(wave_out).save(
        os.path.join(OUTPUT_DIR, "distort_wave_10x50.png")
    )

    # Ripple (amplitude=8, wavelength=40)
    print("  Generating ripple reference...")
    ripple_out = apply_distortion(img, ripple_inverse)
    Image.fromarray(ripple_out).save(
        os.path.join(OUTPUT_DIR, "distort_ripple_8_40.png")
    )

    print("  Distortion references generated.")


if __name__ == "__main__":
    main()
