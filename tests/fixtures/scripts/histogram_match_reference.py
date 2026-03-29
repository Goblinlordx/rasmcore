#!/usr/bin/env python3
"""
Generate reference outputs for histogram matching parity tests.

Reference: scikit-image exposure.match_histograms() v0.26.0
This validates our Rust implementation against an independent, established
library — NOT a reimplementation of the same algorithm.

Prerequisites:
  tests/fixtures/.venv/bin/pip install scikit-image

Usage:
  tests/fixtures/.venv/bin/python3 tests/fixtures/scripts/histogram_match_reference.py
"""

import json
import os
import sys

try:
    import numpy as np
    from skimage.exposure import match_histograms
    import skimage
except ImportError:
    print("ERROR: scikit-image required. Install via:")
    print("  tests/fixtures/.venv/bin/pip install scikit-image")
    sys.exit(1)


def generate_test_cases():
    cases = []

    # Case 1: Grayscale ramp (0-255) matched to itself -> identity
    src = np.arange(256, dtype=np.uint8).reshape(1, 256)
    tgt = np.arange(256, dtype=np.uint8).reshape(1, 256)
    expected = match_histograms(src, tgt).astype(np.uint8)
    cases.append({
        "name": "identity_ramp",
        "source": src.ravel().tolist(),
        "target": tgt.ravel().tolist(),
        "expected": expected.ravel().tolist(),
        "width_src": 256, "height_src": 1,
        "width_tgt": 256, "height_tgt": 1,
        "channels": 1,
    })

    # Case 2: Dark source (0-99) matched to bright target (155-254)
    src = np.arange(100, dtype=np.uint8).reshape(1, 100)
    tgt = np.arange(155, 255, dtype=np.uint8).reshape(1, 100)
    expected = match_histograms(src, tgt).astype(np.uint8)
    cases.append({
        "name": "dark_to_bright",
        "source": src.ravel().tolist(),
        "target": tgt.ravel().tolist(),
        "expected": expected.ravel().tolist(),
        "width_src": 100, "height_src": 1,
        "width_tgt": 100, "height_tgt": 1,
        "channels": 1,
    })

    # Case 3: Uniform random source matched to gaussian-clustered target
    np.random.seed(42)
    src = np.random.randint(0, 256, size=(1, 1024), dtype=np.uint8)
    tgt = np.clip(np.random.normal(128, 40, size=(1, 1024)), 0, 255).astype(np.uint8)
    expected = match_histograms(src, tgt).astype(np.uint8)
    cases.append({
        "name": "uniform_to_gaussian",
        "source": src.ravel().tolist(),
        "target": tgt.ravel().tolist(),
        "expected": expected.ravel().tolist(),
        "width_src": 1024, "height_src": 1,
        "width_tgt": 1024, "height_tgt": 1,
        "channels": 1,
    })

    # Case 4: RGB (3-channel) dark to bright
    np.random.seed(123)
    src_rgb = np.random.randint(0, 128, size=(8, 8, 3), dtype=np.uint8)
    tgt_rgb = np.random.randint(128, 256, size=(8, 8, 3), dtype=np.uint8)
    expected_rgb = match_histograms(src_rgb, tgt_rgb, channel_axis=-1).astype(np.uint8)
    cases.append({
        "name": "rgb_dark_to_bright",
        "source": src_rgb.ravel().tolist(),
        "target": tgt_rgb.ravel().tolist(),
        "expected": expected_rgb.ravel().tolist(),
        "width_src": 8, "height_src": 8,
        "width_tgt": 8, "height_tgt": 8,
        "channels": 3,
    })

    # Case 5: Larger image (64x64) for realistic histogram coverage
    np.random.seed(999)
    src_large = np.random.randint(20, 180, size=(64, 64), dtype=np.uint8)
    tgt_large = np.random.randint(50, 250, size=(64, 64), dtype=np.uint8)
    expected_large = match_histograms(src_large, tgt_large).astype(np.uint8)
    cases.append({
        "name": "gray_64x64_random",
        "source": src_large.ravel().tolist(),
        "target": tgt_large.ravel().tolist(),
        "expected": expected_large.ravel().tolist(),
        "width_src": 64, "height_src": 64,
        "width_tgt": 64, "height_tgt": 64,
        "channels": 1,
    })

    return cases


if __name__ == "__main__":
    cases = generate_test_cases()

    out_dir = "tests/fixtures/generated"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "histogram_match_reference.json")

    with open(out_path, "w") as f:
        json.dump(cases, f)
    print(f"Wrote {len(cases)} test cases to {out_path}")
    print(f"Reference: scikit-image {skimage.__version__}, numpy {np.__version__}")

    for c in cases:
        assert len(c["expected"]) == len(c["source"]), f"{c['name']}: length mismatch"
        print(f"  {c['name']}: {len(c['source'])} values, OK")
