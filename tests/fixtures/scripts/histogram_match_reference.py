#!/usr/bin/env python3
"""
Generate reference outputs for histogram matching parity tests.
Pure Python — no numpy dependency.

Algorithm: standard CDF inversion.
  1. Compute histogram (256 bins) of source and target
  2. Compute normalized CDFs
  3. For each source intensity, find target intensity with nearest CDF value
  4. Build LUT, apply
"""

import json
import os
import random


def compute_histogram(data):
    """Compute 256-bin histogram from a list of uint8 values."""
    h = [0] * 256
    for v in data:
        h[v] += 1
    return h


def compute_cdf(hist, total):
    """Compute normalized CDF (0.0 to 1.0) from histogram."""
    cdf = [0.0] * 256
    cdf[0] = hist[0] / total
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i] / total
    return cdf


def match_lut(src_data, tgt_data):
    """Build histogram matching LUT via CDF inversion."""
    src_hist = compute_histogram(src_data)
    tgt_hist = compute_histogram(tgt_data)
    src_cdf = compute_cdf(src_hist, len(src_data))
    tgt_cdf = compute_cdf(tgt_hist, len(tgt_data))

    lut = [0] * 256
    for s in range(256):
        val = src_cdf[s]
        best = 0
        best_diff = float('inf')
        for t in range(256):
            diff = abs(tgt_cdf[t] - val)
            if diff < best_diff:
                best_diff = diff
                best = t
        lut[s] = best
    return lut


def apply_lut(data, lut):
    """Apply LUT to data."""
    return [lut[v] for v in data]


def histogram_match_gray(source, target):
    """Match single-channel histogram."""
    lut = match_lut(source, target)
    return apply_lut(source, lut)


def histogram_match_rgb(source, target, width, height):
    """Match per-channel RGB histogram. source/target are flat [R,G,B,R,G,B,...] arrays."""
    # Split into channels
    n = width * height
    src_r = [source[i * 3] for i in range(n)]
    src_g = [source[i * 3 + 1] for i in range(n)]
    src_b = [source[i * 3 + 2] for i in range(n)]
    tgt_r = [target[i * 3] for i in range(n)]
    tgt_g = [target[i * 3 + 1] for i in range(n)]
    tgt_b = [target[i * 3 + 2] for i in range(n)]

    lut_r = match_lut(src_r, tgt_r)
    lut_g = match_lut(src_g, tgt_g)
    lut_b = match_lut(src_b, tgt_b)

    result = []
    for i in range(n):
        result.append(lut_r[source[i * 3]])
        result.append(lut_g[source[i * 3 + 1]])
        result.append(lut_b[source[i * 3 + 2]])
    return result


def generate_test_cases():
    cases = []

    # Case 1: Grayscale ramp (0-255) matched to itself -> identity
    src = list(range(256))
    tgt = list(range(256))
    expected = histogram_match_gray(src, tgt)
    cases.append({
        "name": "identity_ramp",
        "source": src, "target": tgt, "expected": expected,
        "width_src": 256, "height_src": 1,
        "width_tgt": 256, "height_tgt": 1,
        "channels": 1,
    })

    # Case 2: Dark source (0-99) matched to bright target (155-254)
    src = list(range(100))
    tgt = list(range(155, 255))
    expected = histogram_match_gray(src, tgt)
    cases.append({
        "name": "dark_to_bright",
        "source": src, "target": tgt, "expected": expected,
        "width_src": 100, "height_src": 1,
        "width_tgt": 100, "height_tgt": 1,
        "channels": 1,
    })

    # Case 3: Uniform random source matched to clustered target
    random.seed(42)
    src = [random.randint(0, 255) for _ in range(1024)]
    tgt = [min(255, max(0, int(random.gauss(128, 40)))) for _ in range(1024)]
    expected = histogram_match_gray(src, tgt)
    cases.append({
        "name": "uniform_to_gaussian",
        "source": src, "target": tgt, "expected": expected,
        "width_src": 1024, "height_src": 1,
        "width_tgt": 1024, "height_tgt": 1,
        "channels": 1,
    })

    # Case 4: RGB (3-channel) dark to bright
    random.seed(123)
    src_rgb = [random.randint(0, 127) for _ in range(8 * 8 * 3)]
    tgt_rgb = [random.randint(128, 255) for _ in range(8 * 8 * 3)]
    expected_rgb = histogram_match_rgb(src_rgb, tgt_rgb, 8, 8)
    cases.append({
        "name": "rgb_dark_to_bright",
        "source": src_rgb, "target": tgt_rgb, "expected": expected_rgb,
        "width_src": 8, "height_src": 8,
        "width_tgt": 8, "height_tgt": 8,
        "channels": 3,
    })

    return cases


if __name__ == "__main__":
    cases = generate_test_cases()

    out_dir = "tests/fixtures/generated"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "histogram_match_reference.json")

    with open(out_path, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"Wrote {len(cases)} test cases to {out_path}")

    # Quick validation
    for c in cases:
        assert len(c["expected"]) == len(c["source"]), f"Case {c['name']}: length mismatch"
        print(f"  {c['name']}: {len(c['source'])} pixels, OK")
