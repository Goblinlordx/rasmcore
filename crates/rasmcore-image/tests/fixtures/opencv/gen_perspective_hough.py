#!/usr/bin/env python3
"""Generate OpenCV reference fixtures for perspective warp and Hough line parity tests.

Generates:
- Perspective warp: warpPerspective with known 3x3 matrix on canonical test images
- Hough lines: HoughLinesP on synthetic edge images with known geometry
- Homography: getPerspectiveTransform for known 4-point correspondences

Fixture naming: {name}.raw (raw bytes), {name}.json (metadata/expected values)
"""

import numpy as np
import cv2
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
W, H = 128, 128


def load_gray(name: str) -> np.ndarray:
    path = os.path.join(SCRIPT_DIR, f"{name}_gray.raw")
    return np.fromfile(path, dtype=np.uint8).reshape((H, W))


def save_raw_u8(name: str, img: np.ndarray):
    path = os.path.join(SCRIPT_DIR, name)
    img.astype(np.uint8).tofile(path)
    print(f"  wrote {name} ({img.size} bytes)")


def save_json(name: str, data):
    path = os.path.join(SCRIPT_DIR, name)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  wrote {name}")


# ── Perspective Warp Fixtures ────────────────────────────────────────────────

def gen_perspective_warp():
    """Generate warpPerspective references with a known translation matrix."""
    print("\n=== Perspective Warp ===")

    # Simple translation: shift by (10, 5)
    M_translate = np.float64([[1, 0, 10], [0, 1, 5], [0, 0, 1]])

    # Perspective transform: mild keystone
    src = np.float32([[0, 0], [127, 0], [127, 127], [0, 127]])
    dst = np.float32([[10, 5], [117, 0], [122, 127], [5, 122]])
    M_perspective = cv2.getPerspectiveTransform(src, dst)

    for img_name in ["gradient_128", "checker_128", "sharp_edges_128"]:
        img = load_gray(img_name)

        # Translation warp
        result_t = cv2.warpPerspective(img, M_translate, (W, H),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0)
        save_raw_u8(f"{img_name}_warp_translate.raw", result_t)

        # Perspective warp
        result_p = cv2.warpPerspective(img, M_perspective, (W, H),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0)
        save_raw_u8(f"{img_name}_warp_perspective.raw", result_p)

    # Save matrices as JSON for Rust test to load
    save_json("warp_translate_matrix.json", {
        "matrix": M_translate.flatten().tolist(),
        "description": "Translation by (10, 5)"
    })
    save_json("warp_perspective_matrix.json", {
        "matrix": M_perspective.flatten().tolist(),
        "src": src.tolist(),
        "dst": dst.tolist(),
        "description": "Mild keystone perspective"
    })


# ── Homography Solver Fixtures ───────────────────────────────────────────────

def gen_homography():
    """Generate getPerspectiveTransform references for known point correspondences."""
    print("\n=== Homography Solver ===")

    test_cases = [
        {
            "name": "identity",
            "src": [[0, 0], [100, 0], [100, 100], [0, 100]],
            "dst": [[0, 0], [100, 0], [100, 100], [0, 100]],
        },
        {
            "name": "translation",
            "src": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "dst": [[10, 20], [11, 20], [11, 21], [10, 21]],
        },
        {
            "name": "perspective",
            "src": [[0, 0], [127, 0], [127, 127], [0, 127]],
            "dst": [[10, 5], [117, 0], [122, 127], [5, 122]],
        },
    ]

    results = []
    for tc in test_cases:
        src = np.float32(tc["src"])
        dst = np.float32(tc["dst"])
        M = cv2.getPerspectiveTransform(src, dst)
        results.append({
            "name": tc["name"],
            "src": tc["src"],
            "dst": tc["dst"],
            "matrix": M.flatten().tolist(),
        })
        print(f"  {tc['name']}: M[8]={M[2][2]:.15e}")

    save_json("homography_reference.json", results)


# ── Hough Lines Fixtures ─────────────────────────────────────────────────────

def gen_hough_lines():
    """Generate HoughLinesP references on synthetic edge images."""
    print("\n=== Hough Lines ===")

    # Image with a single horizontal line at y=32
    img_h = np.zeros((64, 64), dtype=np.uint8)
    img_h[32, 5:60] = 255
    save_raw_u8("hough_horizontal_64.raw", img_h)

    lines_h = cv2.HoughLinesP(img_h, rho=1, theta=np.pi/180,
                               threshold=20, minLineLength=20, maxLineGap=5)
    h_result = []
    if lines_h is not None:
        for l in lines_h:
            h_result.append(l[0].tolist())
    print(f"  horizontal: {len(h_result)} lines")

    # Image with a single vertical line at x=32
    img_v = np.zeros((64, 64), dtype=np.uint8)
    img_v[5:60, 32] = 255
    save_raw_u8("hough_vertical_64.raw", img_v)

    lines_v = cv2.HoughLinesP(img_v, rho=1, theta=np.pi/180,
                               threshold=20, minLineLength=20, maxLineGap=5)
    v_result = []
    if lines_v is not None:
        for l in lines_v:
            v_result.append(l[0].tolist())
    print(f"  vertical: {len(v_result)} lines")

    # Cross: both horizontal and vertical
    img_cross = np.zeros((64, 64), dtype=np.uint8)
    img_cross[32, 5:60] = 255
    img_cross[5:60, 32] = 255
    save_raw_u8("hough_cross_64.raw", img_cross)

    lines_cross = cv2.HoughLinesP(img_cross, rho=1, theta=np.pi/180,
                                   threshold=15, minLineLength=20, maxLineGap=5)
    cross_result = []
    if lines_cross is not None:
        for l in lines_cross:
            cross_result.append(l[0].tolist())
    print(f"  cross: {len(cross_result)} lines")

    save_json("hough_reference.json", {
        "horizontal": {"lines": h_result, "params": {"threshold": 20, "minLineLength": 20, "maxLineGap": 5}},
        "vertical": {"lines": v_result, "params": {"threshold": 20, "minLineLength": 20, "maxLineGap": 5}},
        "cross": {"lines": cross_result, "params": {"threshold": 15, "minLineLength": 20, "maxLineGap": 5}},
        "note": "OpenCV HoughLinesP is non-deterministic (RNG-dependent). These are single-run reference values."
    })


if __name__ == "__main__":
    gen_perspective_warp()
    gen_homography()
    gen_hough_lines()
    print("\nDone.")
