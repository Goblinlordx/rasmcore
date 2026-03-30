#!/usr/bin/env python3
"""Generate colour-science 0.4.7 reference values for color space parity tests.

Generates JSON fixtures with exact reference conversions for:
- sRGB ↔ XYZ (D65)
- ProPhoto RGB ↔ XYZ (D50) ↔ sRGB (via Bradford D50↔D65)
- Adobe RGB ↔ XYZ (D65) ↔ sRGB
- Bradford chromatic adaptation D65↔D50

Each fixture contains input values, expected output, and the max acceptable error.

Usage: python3 gen_color_space_reference.py
Requires: pip install colour-science numpy
"""

import json
import os

try:
    import colour
    import numpy as np
except ImportError:
    print("ERROR: requires 'colour-science' and 'numpy'")
    print("  pip install colour-science numpy")
    exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def save_json(name, data):
    path = os.path.join(SCRIPT_DIR, name)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  wrote {name}")


# Standard test colors (sRGB, [0,1] range)
TEST_COLORS = [
    {"name": "red",    "srgb": [0.8, 0.1, 0.1]},
    {"name": "green",  "srgb": [0.1, 0.7, 0.2]},
    {"name": "blue",   "srgb": [0.1, 0.1, 0.8]},
    {"name": "gray",   "srgb": [0.5, 0.5, 0.5]},
    {"name": "custom", "srgb": [0.6, 0.4, 0.2]},
    {"name": "white",  "srgb": [1.0, 1.0, 1.0]},
    {"name": "black",  "srgb": [0.0, 0.0, 0.0]},
]


def gen_srgb_xyz():
    """sRGB ↔ XYZ roundtrip reference values."""
    print("\n=== sRGB ↔ XYZ ===")
    results = []
    for tc in TEST_COLORS:
        rgb = np.array(tc["srgb"])
        xyz = colour.sRGB_to_XYZ(rgb)
        back = colour.XYZ_to_sRGB(xyz)
        results.append({
            "name": tc["name"],
            "srgb_input": tc["srgb"],
            "xyz": xyz.tolist(),
            "srgb_roundtrip": back.tolist(),
            "roundtrip_max_error": float(np.max(np.abs(back - rgb))),
        })
    save_json("color_ref_srgb_xyz.json", {
        "reference": "colour-science " + colour.__version__,
        "illuminant": "D65",
        "values": results,
    })


def gen_prophoto():
    """ProPhoto RGB ↔ sRGB reference values."""
    print("\n=== ProPhoto RGB ===")
    pp = colour.RGB_COLOURSPACES['ProPhoto RGB']
    results = []
    for tc in TEST_COLORS:
        srgb = np.array(tc["srgb"])
        # sRGB → XYZ → ProPhoto
        xyz = colour.sRGB_to_XYZ(srgb)
        pp_rgb = colour.XYZ_to_RGB(xyz, pp)
        # ProPhoto → XYZ → sRGB
        xyz_back = colour.RGB_to_XYZ(pp_rgb, pp)
        srgb_back = colour.XYZ_to_sRGB(xyz_back)
        results.append({
            "name": tc["name"],
            "srgb_input": tc["srgb"],
            "prophoto": pp_rgb.tolist(),
            "srgb_roundtrip": srgb_back.tolist(),
            "roundtrip_max_error": float(np.max(np.abs(srgb_back - srgb))),
        })
        print(f"  {tc['name']}: roundtrip err={np.max(np.abs(srgb_back - srgb)):.2e}")

    save_json("color_ref_prophoto.json", {
        "reference": "colour-science " + colour.__version__,
        "illuminant": "D50 (ProPhoto) via Bradford to D65 (sRGB)",
        "matrix_rgb_to_xyz": pp.matrix_RGB_to_XYZ.tolist(),
        "matrix_xyz_to_rgb": pp.matrix_XYZ_to_RGB.tolist(),
        "values": results,
    })


def gen_adobe():
    """Adobe RGB ↔ sRGB reference values."""
    print("\n=== Adobe RGB ===")
    ar = colour.RGB_COLOURSPACES['Adobe RGB (1998)']
    results = []
    for tc in TEST_COLORS:
        srgb = np.array(tc["srgb"])
        xyz = colour.sRGB_to_XYZ(srgb)
        a_rgb = colour.XYZ_to_RGB(xyz, ar)
        xyz_back = colour.RGB_to_XYZ(a_rgb, ar)
        srgb_back = colour.XYZ_to_sRGB(xyz_back)
        results.append({
            "name": tc["name"],
            "srgb_input": tc["srgb"],
            "adobe": a_rgb.tolist(),
            "srgb_roundtrip": srgb_back.tolist(),
            "roundtrip_max_error": float(np.max(np.abs(srgb_back - srgb))),
        })
        print(f"  {tc['name']}: roundtrip err={np.max(np.abs(srgb_back - srgb)):.2e}")

    save_json("color_ref_adobe.json", {
        "reference": "colour-science " + colour.__version__,
        "illuminant": "D65",
        "matrix_rgb_to_xyz": ar.matrix_RGB_to_XYZ.tolist(),
        "matrix_xyz_to_rgb": ar.matrix_XYZ_to_RGB.tolist(),
        "values": results,
    })


def gen_bradford():
    """Bradford chromatic adaptation reference values."""
    print("\n=== Bradford ===")
    d65 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
    d50 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50']

    test_xyz = [
        {"name": "d65_white", "xyz": [0.9504559270516716, 1.0, 1.0890577507598784]},
        {"name": "mid_gray", "xyz": [0.5, 0.4, 0.3]},
        {"name": "red_ish", "xyz": [0.4, 0.2, 0.05]},
    ]

    results = []
    for tc in test_xyz:
        xyz = np.array(tc["xyz"])
        adapted = colour.chromatic_adaptation(
            xyz, colour.xy_to_XYZ(d65), colour.xy_to_XYZ(d50),
            method='Von Kries', transform='Bradford'
        )
        back = colour.chromatic_adaptation(
            adapted, colour.xy_to_XYZ(d50), colour.xy_to_XYZ(d65),
            method='Von Kries', transform='Bradford'
        )
        results.append({
            "name": tc["name"],
            "xyz_d65": tc["xyz"],
            "xyz_d50": adapted.tolist(),
            "xyz_d65_roundtrip": back.tolist(),
            "roundtrip_max_error": float(np.max(np.abs(back - xyz))),
        })
        print(f"  {tc['name']}: roundtrip err={np.max(np.abs(back - xyz)):.2e}")

    save_json("color_ref_bradford.json", {
        "reference": "colour-science " + colour.__version__,
        "values": results,
    })


if __name__ == "__main__":
    gen_srgb_xyz()
    gen_prophoto()
    gen_adobe()
    gen_bradford()
    print("\nDone.")
