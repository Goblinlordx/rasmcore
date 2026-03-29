#!/usr/bin/env python3
"""
Validates that the Python-vs-OpenCV MergeMertens divergence is reproducible.

This script replicates the exact MergeMertens algorithm using only OpenCV's
own C++ primitives (cv2.pyrDown, cv2.pyrUp, cv2.Laplacian, cv2.split, cv2.merge)
and compares against cv2.createMergeMertens().process().

Expected result: checker_128 shows u8 max error ~17, all others ≤1.
This proves the divergence is not Rust-specific but inherent to f32 precision
differences between any reimplementation and OpenCV's internal C++ code.

Usage:
    python3 validate_mertens_python_vs_opencv.py

Requires: opencv-contrib-python-headless, numpy
"""
import sys
import os
import cv2
import numpy as np

print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Python version: {sys.version}")
print()

fixture_dir = os.path.dirname(os.path.abspath(__file__))

names = [
    "gradient_128", "checker_128", "noisy_flat_128", "sharp_edges_128",
    "photo_128", "flat_128", "highcontrast_128",
]

cv2.setNumThreads(1)

for name in names:
    gray_path = os.path.join(fixture_dir, f"{name}_gray.raw")
    if not os.path.exists(gray_path):
        print(f"SKIP {name}: fixture not found")
        continue

    gray = np.fromfile(gray_path, dtype=np.uint8).reshape(128, 128)
    h, w = gray.shape

    # Create color scene (must match Rust test fixture generation)
    scene = np.zeros((h, w, 3), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            v = gray[y, x].astype(np.float32)
            scene[y, x, 2] = v                          # R in BGR
            scene[y, x, 1] = v * 0.9 + (x % 7) * 2     # G
            scene[y, x, 0] = v * 0.7 + (y % 5) * 3     # B
    scene = np.clip(scene, 0, 255)
    brackets = [np.clip(scene * e, 0, 255).astype(np.uint8) for e in [0.25, 1.0, 4.0]]
    images_f = [img.astype(np.float32) / 255.0 for img in brackets]

    # --- Ground truth: OpenCV MergeMertens ---
    merger = cv2.createMergeMertens(1.0, 1.0, 1.0)
    result_cv = merger.process(brackets)

    # --- Manual replication using cv2 primitives only ---
    # Weight computation (matching merge.cpp lines 206-253)
    weights = []
    for img_f in images_f:
        gray_f = cv2.cvtColor(img_f, cv2.COLOR_RGB2GRAY)
        contrast = np.abs(cv2.Laplacian(gray_f, cv2.CV_32F))
        splitted = list(cv2.split(img_f))
        mean_img = (splitted[0] + splitted[1] + splitted[2]) / np.float32(3)
        sat_sq = np.zeros((h, w), dtype=np.float32)
        for c in range(3):
            dev = splitted[c] - mean_img
            sat_sq += dev * dev
        sat = np.sqrt(sat_sq)
        wellexp = np.ones((h, w), dtype=np.float32)
        for c in range(3):
            expo = splitted[c] - np.float32(0.5)
            expo = expo * expo
            expo = -expo / np.float32(0.08)
            wellexp *= np.exp(expo)
        wt = contrast * sat * wellexp + np.float32(1e-12)
        weights.append(wt)

    ws = weights[0] + weights[1] + weights[2]
    nw = [wt / ws for wt in weights]

    # Pyramid blending (matching merge.cpp lines 260-296)
    maxlevel = int(np.log(float(min(h, w))) / np.log(2.0))

    all_img_pyr = []
    for img_f in images_f:
        pyr = [img_f.copy()]
        for l in range(maxlevel):
            pyr.append(cv2.pyrDown(pyr[-1]))
        for lvl in range(maxlevel):
            up = cv2.pyrUp(pyr[lvl + 1], dstsize=(pyr[lvl].shape[1], pyr[lvl].shape[0]))
            pyr[lvl] = pyr[lvl] - up
        all_img_pyr.append(pyr)

    all_w_pyr = [[wt.copy()] for wt in nw]
    for i in range(3):
        for l in range(maxlevel):
            all_w_pyr[i].append(cv2.pyrDown(all_w_pyr[i][-1]))

    res_pyr = [None] * (maxlevel + 1)
    for i in range(3):
        for lvl in range(maxlevel + 1):
            splitted = list(cv2.split(all_img_pyr[i][lvl]))
            for c in range(len(splitted)):
                splitted[c] = splitted[c] * all_w_pyr[i][lvl]
            weighted = cv2.merge(splitted)
            if res_pyr[lvl] is None:
                res_pyr[lvl] = weighted.copy()
            else:
                res_pyr[lvl] = res_pyr[lvl] + weighted

    r = res_pyr[maxlevel].copy()
    for lvl in range(maxlevel - 1, -1, -1):
        up = cv2.pyrUp(r, dstsize=(res_pyr[lvl].shape[1], res_pyr[lvl].shape[0]))
        r = up + res_pyr[lvl]

    # --- Compare ---
    f32_diff = np.abs(r.astype(np.float64) - result_cv.astype(np.float64))
    u8_manual = np.clip(r * 255, 0, 255).astype(np.uint8)
    u8_cv = np.clip(result_cv * 255, 0, 255).astype(np.uint8)
    u8_diff = np.abs(u8_manual.astype(np.int16) - u8_cv.astype(np.int16))

    print(
        f"{name:20}  f32 max={f32_diff.max():.6f} mean={f32_diff.mean():.8f}"
        f"  |  u8 max={u8_diff.max():2d} MAE={u8_diff.mean():.4f}"
    )

# --- Threading validation ---
print()
gray = np.fromfile(os.path.join(fixture_dir, "checker_128_gray.raw"), dtype=np.uint8).reshape(128, 128)
scene = np.zeros((128, 128, 3), dtype=np.float32)
for y in range(128):
    for x in range(128):
        v = gray[y, x].astype(np.float32)
        scene[y, x, 2] = v
        scene[y, x, 1] = v * 0.9 + (x % 7) * 2
        scene[y, x, 0] = v * 0.7 + (y % 5) * 3
scene = np.clip(scene, 0, 255)
brackets = [np.clip(scene * e, 0, 255).astype(np.uint8) for e in [0.25, 1.0, 4.0]]

results = {}
for threads in [1, 4, 0]:
    cv2.setNumThreads(threads)
    merger = cv2.createMergeMertens(1.0, 1.0, 1.0)
    results[threads] = merger.process(brackets).copy()

d_1_4 = np.abs(results[1].astype(np.float64) - results[4].astype(np.float64)).max()
d_1_0 = np.abs(results[1].astype(np.float64) - results[0].astype(np.float64)).max()
print(f"Threading test (checker_128):")
print(f"  1-thread vs 4-thread: max_diff={d_1_4:.2e}")
print(f"  1-thread vs auto:     max_diff={d_1_0:.2e}")
print(f"  Conclusion: {'DETERMINISTIC' if d_1_4 == 0 and d_1_0 == 0 else 'NON-DETERMINISTIC'}")
