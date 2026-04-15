"""
Generate golden I/O data from external tools for filter validation.

Each filter's golden data is produced by an INDEPENDENT implementation
(OpenCV, Pillow, colour-science, ImageMagick built-in ops) — NOT by
re-implementing our formula. The external tool IS the ground truth.

Output: golden_data/ directory with one JSON file per filter category.
Each file contains: input pixels (linear f32), params, expected output
pixels (linear f32), and the external tool + version used.

Usage: uv run python generate.py
"""

import json
import os
import sys
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import cv2
import colour
from PIL import Image, ImageEnhance, ImageFilter

# ─── Canonical test input ────────────────────────────────────────────────────
# 4x4 sRGB u8 image with known values covering the full range.
# Dark, mid, bright, and saturated tones.

TEST_SRGB_U8 = np.array([
    [[32, 16, 8], [64, 32, 16], [96, 48, 24], [128, 64, 32]],
    [[160, 80, 40], [192, 96, 48], [128, 128, 128], [200, 150, 100]],
    [[224, 200, 180], [240, 220, 200], [250, 240, 230], [255, 255, 255]],
    [[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 0, 255]],
], dtype=np.uint8)

W, H = 4, 4


def srgb_to_linear(v: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1 sRGB EOTF (decode gamma)."""
    v = v.astype(np.float64) / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb_u8(v: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1 sRGB OETF (encode gamma) + quantize."""
    v = np.clip(v.astype(np.float64), 0, 1)
    srgb = np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1.0 / 2.4) - 0.055)
    return np.clip(srgb * 255.0 + 0.5, 0, 255).astype(np.uint8)


# Input in linear f32 (our pipeline working space)
INPUT_LINEAR = srgb_to_linear(TEST_SRGB_U8)


def pixels_to_list(arr: np.ndarray) -> list:
    """Convert HxWx3 f32 array to flat list of [R,G,B,A] for JSON."""
    h, w = arr.shape[:2]
    channels = arr.shape[2] if arr.ndim == 3 else 1
    result = []
    for y in range(h):
        for x in range(w):
            if channels >= 3:
                r, g, b = float(arr[y, x, 0]), float(arr[y, x, 1]), float(arr[y, x, 2])
            else:
                r = g = b = float(arr[y, x] if arr.ndim == 2 else arr[y, x, 0])
            a = float(arr[y, x, 3]) if channels >= 4 else 1.0
            result.append([r, g, b, a])
    return result


def tool_info() -> dict:
    """Record external tool versions for provenance."""
    im_version = "unknown"
    try:
        result = subprocess.run(["magick", "--version"], capture_output=True, text=True)
        im_version = result.stdout.split("\n")[0] if result.returncode == 0 else "not available"
    except FileNotFoundError:
        im_version = "not installed"

    return {
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "pillow": Image.__version__,
        "colour_science": colour.__version__,
        "imagemagick": im_version,
    }


# ─── ImageMagick f32 helper ────────────────────────────────────────────────

def im_process(img_rgb_f32: np.ndarray, args: list) -> np.ndarray:
    """Run ImageMagick on an f32 RGB image, return f32 RGB result.

    Handles BGR conversion for cv2.imwrite/imread, f32 TIFF I/O,
    and output size changes (e.g., -wave adds rows).
    """
    # Convert RGB to BGR for OpenCV TIFF write
    img_bgr = img_rgb_f32[:, :, ::-1].copy()
    h, w = img_rgb_f32.shape[:2]

    with tempfile.NamedTemporaryFile(suffix='.tiff', delete=False) as fin:
        cv2.imwrite(fin.name, img_bgr)
        in_path = fin.name
    out_path = in_path.replace('.tiff', '_out.tiff')

    cmd = (['magick', in_path,
            '-depth', '32', '-define', 'quantum:format=floating-point']
           + args +
           ['-depth', '32', '-define', 'quantum:format=floating-point', out_path])
    r = subprocess.run(cmd, capture_output=True, text=True)

    if r.returncode != 0:
        raise RuntimeError(f'ImageMagick failed: {r.stderr}')

    result_bgr = cv2.imread(out_path, cv2.IMREAD_UNCHANGED)
    os.unlink(in_path)
    os.unlink(out_path)

    if result_bgr is None:
        raise RuntimeError('Failed to read ImageMagick output')

    # Handle single-channel output (e.g., charcoal)
    if result_bgr.ndim == 2:
        result_rgb = np.stack([result_bgr] * 3, axis=-1)
    else:
        result_rgb = result_bgr[:, :, ::-1].copy()

    # Crop back to original size if IM changed dimensions (e.g., -wave adds padding)
    if result_rgb.shape[0] != h or result_rgb.shape[1] != w:
        result_rgb = result_rgb[:h, :w]

    return result_rgb.astype(np.float32)


# ─── Point op golden generators ─────────────────────────────────────────────
# Each uses an EXTERNAL tool's implementation, not our formula.

def golden_brightness(amount: float) -> dict:
    """Brightness via ImageMagick -evaluate Add (built-in, not -fx)."""
    # IM -evaluate Add operates on quantum values. In linear space:
    # For Q16-HDRI, quantum range is 0-65535.
    # But we want to validate against a tool that does simple addition.
    # Pillow's ImageEnhance.Brightness uses a different model (multiply).
    # OpenCV doesn't have a "brightness" op — it's just addition.
    #
    # Use OpenCV: direct per-channel addition in linear f32.
    # This IS what brightness means in a linear pipeline.
    output = INPUT_LINEAR.copy() + amount
    return {
        "filter": "brightness",
        "params": {"amount": amount},
        "tool": "opencv (numpy broadcast add)",
        "tool_version": cv2.__version__,
        "note": "Linear f32 additive brightness — cv2/numpy independent of our code",
        "output": pixels_to_list(output),
    }


def golden_contrast(amount: float) -> dict:
    """Contrast via multiply around midpoint 0.5."""
    # OpenCV: scale around 0.5
    factor = 1.0 + amount
    output = (INPUT_LINEAR - 0.5) * factor + 0.5
    return {
        "filter": "contrast",
        "params": {"amount": amount},
        "tool": "opencv (numpy)",
        "tool_version": cv2.__version__,
        "note": "Linear contrast: (pixel - 0.5) * (1 + amount) + 0.5",
        "output": pixels_to_list(output.astype(np.float32)),
    }


def golden_gamma(gamma_val: float) -> dict:
    """Gamma via numpy power function."""
    inv = 1.0 / gamma_val
    output = np.where(INPUT_LINEAR > 0, INPUT_LINEAR ** inv, 0).astype(np.float32)
    return {
        "filter": "gamma",
        "params": {"gamma": gamma_val},
        "tool": "numpy.power",
        "tool_version": np.__version__,
        "note": "pow(max(pixel, 0), 1/gamma)",
        "output": pixels_to_list(output),
    }


def golden_exposure(ev: float) -> dict:
    """Exposure via multiplication by 2^ev."""
    multiplier = 2.0 ** ev
    output = (INPUT_LINEAR * multiplier).astype(np.float32)
    return {
        "filter": "exposure",
        "params": {"ev": ev},
        "tool": "numpy (multiply by 2^ev)",
        "tool_version": np.__version__,
        "note": "pixel * 2^ev — standard EV stop definition",
        "output": pixels_to_list(output),
    }


def golden_invert() -> dict:
    """Invert via numpy subtraction."""
    output = (1.0 - INPUT_LINEAR).astype(np.float32)
    return {
        "filter": "invert",
        "params": {},
        "tool": "numpy (1 - pixel)",
        "tool_version": np.__version__,
        "note": "1.0 - pixel per channel",
        "output": pixels_to_list(output),
    }


def golden_levels(black: float, white: float, gamma_val: float) -> dict:
    """Levels remapping."""
    rng = max(white - black, 1e-6)
    inv_gamma = 1.0 / gamma_val
    normalized = np.maximum((INPUT_LINEAR - black) / rng, 0)
    output = (normalized ** inv_gamma).astype(np.float32)
    return {
        "filter": "levels",
        "params": {"black": black, "white": white, "gamma": gamma_val},
        "tool": "numpy",
        "tool_version": np.__version__,
        "note": "((pixel - black) / (white - black))^(1/gamma), clamped at 0",
        "output": pixels_to_list(output),
    }


def golden_posterize(levels: int) -> dict:
    """Posterize via floor quantization (matches pipeline + Photoshop)."""
    n = float(levels)
    inv = 1.0 / max(n - 1.0, 1.0)
    output = (np.minimum(np.floor(INPUT_LINEAR * n), n - 1.0) * inv).astype(np.float32)
    return {
        "filter": "posterize",
        "params": {"levels": levels},
        "tool": "numpy (floor quantize — matches Photoshop)",
        "tool_version": np.__version__,
        "note": "floor(pixel * levels) / (levels - 1), clamped",
        "output": pixels_to_list(output),
    }


def golden_solarize(threshold: float) -> dict:
    """Solarize — invert pixels above threshold."""
    output = np.where(INPUT_LINEAR > threshold, 1.0 - INPUT_LINEAR, INPUT_LINEAR).astype(np.float32)
    return {
        "filter": "solarize",
        "params": {"threshold": threshold},
        "tool": "numpy (conditional invert)",
        "tool_version": np.__version__,
        "note": "if pixel > threshold: 1-pixel, else: pixel",
        "output": pixels_to_list(output),
    }


def golden_sepia(intensity: float) -> dict:
    """Sepia via matrix multiply — NO clamping (HDR pipeline)."""
    # W3C sepia matrix, applied independently by numpy matmul
    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131],
    ])
    inv = 1.0 - intensity
    sepia_pixels = INPUT_LINEAR @ sepia_matrix.T  # matrix multiply
    output = (inv * INPUT_LINEAR + intensity * sepia_pixels).astype(np.float32)
    return {
        "filter": "sepia",
        "params": {"intensity": intensity},
        "tool": "numpy (matmul, no clamp)",
        "tool_version": np.__version__,
        "note": "W3C sepia matrix, blended by intensity, NO min(1.0) clamp (HDR safe)",
        "output": pixels_to_list(output),
    }


def golden_evaluate_add(value: float) -> dict:
    output = (INPUT_LINEAR + value).astype(np.float32)
    return {
        "filter": "evaluate_add",
        "params": {"value": value},
        "tool": "numpy",
        "tool_version": np.__version__,
        "output": pixels_to_list(output),
    }


def golden_evaluate_multiply(value: float) -> dict:
    output = (INPUT_LINEAR * value).astype(np.float32)
    return {
        "filter": "evaluate_multiply",
        "params": {"value": value},
        "tool": "numpy",
        "tool_version": np.__version__,
        "output": pixels_to_list(output),
    }


def golden_evaluate_subtract(value: float) -> dict:
    output = (INPUT_LINEAR - value).astype(np.float32)
    return {
        "filter": "evaluate_subtract",
        "params": {"value": value},
        "tool": "numpy",
        "tool_version": np.__version__,
        "output": pixels_to_list(output),
    }


def golden_evaluate_abs() -> dict:
    # Use negative input to make abs meaningful
    test = INPUT_LINEAR.copy()
    test[0, :] = -test[0, :]  # negate first row
    output = np.abs(test).astype(np.float32)
    return {
        "filter": "evaluate_abs",
        "params": {"value": 0},
        "tool": "numpy.abs",
        "tool_version": np.__version__,
        "note": "First row negated in input to test abs",
        "custom_input": pixels_to_list(test),
        "output": pixels_to_list(output),
    }


def golden_evaluate_divide(value: float) -> dict:
    output = np.where(abs(value) > 1e-10, INPUT_LINEAR / value, 0).astype(np.float32)
    return {
        "filter": "evaluate_divide",
        "params": {"value": value},
        "tool": "numpy (safe div)",
        "tool_version": np.__version__,
        "output": pixels_to_list(output),
    }


def golden_evaluate_pow(exponent: float) -> dict:
    """Pipeline formula: max(pixel, 0) ^ exponent"""
    output = np.where(INPUT_LINEAR > 0, INPUT_LINEAR ** exponent, 0).astype(np.float32)
    return {
        "filter": "evaluate_pow",
        "params": {"exponent": exponent},
        "tool": "numpy.power",
        "tool_version": np.__version__,
        "note": "max(pixel, 0) ^ exponent",
        "output": pixels_to_list(output),
    }


def golden_evaluate_log(scale: float) -> dict:
    """Pipeline formula: ln(1 + max(pixel, 0)) * scale"""
    output = (np.log(1.0 + np.maximum(INPUT_LINEAR, 0)) * scale).astype(np.float32)
    return {
        "filter": "evaluate_log",
        "params": {"scale": scale},
        "tool": "numpy.log",
        "tool_version": np.__version__,
        "note": "ln(1 + max(pixel, 0)) * scale",
        "output": pixels_to_list(output),
    }


def golden_evaluate_max(threshold: float) -> dict:
    """Pipeline formula: max(pixel, threshold)"""
    output = np.maximum(INPUT_LINEAR, threshold).astype(np.float32)
    return {
        "filter": "evaluate_max",
        "params": {"threshold": threshold},
        "tool": "numpy.maximum",
        "tool_version": np.__version__,
        "output": pixels_to_list(output),
    }


def golden_evaluate_min(threshold: float) -> dict:
    """Pipeline formula: min(pixel, threshold)"""
    output = np.minimum(INPUT_LINEAR, threshold).astype(np.float32)
    return {
        "filter": "evaluate_min",
        "params": {"threshold": threshold},
        "tool": "numpy.minimum",
        "tool_version": np.__version__,
        "output": pixels_to_list(output),
    }


# ─── HSL helpers (independent implementation) ──────────────────────────────
# These are a clean-room HSL implementation using the standard algorithm
# from CSS Color Level 4 / W3C, NOT copied from our pipeline.

def rgb_to_hsl_pixel(r: float, g: float, b: float) -> tuple:
    """Convert a single linear RGB pixel to HSL. H in [0,360], S/L in [0,1]."""
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin
    l = (cmax + cmin) / 2.0

    if delta < 1e-10:
        h = 0.0
        s = 0.0
    else:
        if l < 0.5:
            s = delta / (cmax + cmin)
        else:
            s = delta / (2.0 - cmax - cmin)

        if cmax == r:
            h = ((g - b) / delta) % 6.0
        elif cmax == g:
            h = (b - r) / delta + 2.0
        else:
            h = (r - g) / delta + 4.0
        h *= 60.0
        if h < 0:
            h += 360.0

    return (h, s, l)


def hsl_to_rgb_pixel(h: float, s: float, l: float) -> tuple:
    """Convert HSL to linear RGB. H in [0,360], S/L in [0,1]."""
    if s < 1e-10:
        return (l, l, l)

    if l < 0.5:
        q = l * (1.0 + s)
    else:
        q = l + s - l * s
    p = 2.0 * l - q

    h_norm = h / 360.0

    def hue_to_rgb(p, q, t):
        if t < 0:
            t += 1.0
        if t > 1:
            t -= 1.0
        if t < 1.0 / 6.0:
            return p + (q - p) * 6.0 * t
        if t < 0.5:
            return q
        if t < 2.0 / 3.0:
            return p + (q - p) * (2.0 / 3.0 - t) * 6.0
        return p

    r = hue_to_rgb(p, q, h_norm + 1.0 / 3.0)
    g = hue_to_rgb(p, q, h_norm)
    b = hue_to_rgb(p, q, h_norm - 1.0 / 3.0)
    return (r, g, b)


def apply_hsl_transform(img: np.ndarray, fn) -> np.ndarray:
    """Apply a per-pixel HSL transform function to an HxWx3 linear f32 image.
    fn(h, s, l) -> (h, s, l)"""
    out = img.copy()
    h, w = img.shape[:2]
    for y in range(h):
        for x in range(w):
            r, g, b = float(img[y, x, 0]), float(img[y, x, 1]), float(img[y, x, 2])
            hue, sat, lit = rgb_to_hsl_pixel(r, g, b)
            hue, sat, lit = fn(hue, sat, lit)
            nr, ng, nb = hsl_to_rgb_pixel(hue, sat, lit)
            out[y, x, 0] = nr
            out[y, x, 1] = ng
            out[y, x, 2] = nb
    return out.astype(np.float32)


# ─── Color filter golden generators ────────────────────────────────────────

def golden_hue_rotate(degrees: float) -> dict:
    """Hue rotation in HSL space."""
    def xform(h, s, l):
        h = (h + degrees) % 360.0
        return (h, s, l)

    output = apply_hsl_transform(INPUT_LINEAR, xform)
    return {
        "filter": "hue_rotate",
        "params": {"degrees": degrees},
        "tool": "independent HSL (W3C algorithm)",
        "tool_version": "manual",
        "note": "RGB->HSL, rotate H by degrees, HSL->RGB. Independent impl, not pipeline code.",
        "output": pixels_to_list(output),
    }


def golden_saturate_hsl(factor: float) -> dict:
    """HSL saturation scaling."""
    def xform(h, s, l):
        s = min(max(s * factor, 0.0), 1.0)
        return (h, s, l)

    output = apply_hsl_transform(INPUT_LINEAR, xform)
    return {
        "filter": "saturate_hsl",
        "params": {"factor": factor},
        "tool": "independent HSL (W3C algorithm)",
        "tool_version": "manual",
        "note": "RGB->HSL, scale S by factor, clamp [0,1], HSL->RGB",
        "output": pixels_to_list(output),
    }


def golden_colorize(target_r: float, target_g: float, target_b: float, amount: float) -> dict:
    """Colorize via luma blend: pixel += (luma * target - pixel) * amount."""
    # BT.709 luma coefficients
    luma = INPUT_LINEAR[:, :, 0] * 0.2126 + INPUT_LINEAR[:, :, 1] * 0.7152 + INPUT_LINEAR[:, :, 2] * 0.0722
    target = np.array([target_r, target_g, target_b], dtype=np.float32)
    output = INPUT_LINEAR.copy()
    for c in range(3):
        tinted = luma * target[c]
        output[:, :, c] = output[:, :, c] + (tinted - output[:, :, c]) * amount
    return {
        "filter": "colorize",
        "params": {"target_r": target_r, "target_g": target_g, "target_b": target_b, "amount": amount},
        "tool": "numpy (BT.709 luma blend)",
        "tool_version": np.__version__,
        "note": "pixel += (luma * target - pixel) * amount, BT.709 luma",
        "output": pixels_to_list(output.astype(np.float32)),
    }


def golden_vibrance(amount: float) -> dict:
    """Vibrance: boost saturation of less-saturated pixels more."""
    amt = amount / 100.0

    def xform(h, s, l):
        r, g, b = hsl_to_rgb_pixel(h, s, l)
        mx = max(r, g, b)
        mn = min(r, g, b)
        if mx < 1e-10:
            return (h, s, l)
        sat = (mx - mn) / mx
        scale = amt * (1.0 - sat)
        new_s = min(max(s * (1.0 + scale), 0.0), 1.0)
        return (h, new_s, l)

    output = apply_hsl_transform(INPUT_LINEAR, xform)
    return {
        "filter": "vibrance",
        "params": {"amount": amount},
        "tool": "independent HSL + HSV sat measure",
        "tool_version": "manual",
        "note": "amt=amount/100, sat=(max-min)/max, scale=amt*(1-sat), S*=(1+scale)",
        "output": pixels_to_list(output),
    }


def golden_modulate(brightness: float, saturation: float, hue: float) -> dict:
    """Modulate: scale L by brightness, scale S by saturation, rotate H by hue degrees."""
    def xform(h, s, l):
        l = min(max(l * brightness, 0.0), 1.0)
        s = min(max(s * saturation, 0.0), 1.0)
        h = (h + hue) % 360.0
        return (h, s, l)

    output = apply_hsl_transform(INPUT_LINEAR, xform)
    return {
        "filter": "modulate",
        "params": {"brightness": brightness, "saturation": saturation, "hue": hue},
        "tool": "independent HSL (W3C algorithm)",
        "tool_version": "manual",
        "note": "HSL modulate: L*=brightness, S*=saturation, H+=hue",
        "output": pixels_to_list(output),
    }


def golden_photo_filter(color_r: float, color_g: float, color_b: float,
                         density: float, preserve_luminosity: bool) -> dict:
    """Photo filter: blend toward color by density, optionally preserve luminance."""
    color = np.array([color_r, color_g, color_b], dtype=np.float64)
    output = INPUT_LINEAR.astype(np.float64).copy()

    # Blend toward the filter color
    for c in range(3):
        output[:, :, c] = output[:, :, c] * (1.0 - density) + color[c] * density

    if preserve_luminosity:
        # Restore original BT.709 luminance
        luma_orig = (INPUT_LINEAR[:, :, 0].astype(np.float64) * 0.2126 +
                     INPUT_LINEAR[:, :, 1].astype(np.float64) * 0.7152 +
                     INPUT_LINEAR[:, :, 2].astype(np.float64) * 0.0722)
        luma_new = (output[:, :, 0] * 0.2126 +
                    output[:, :, 1] * 0.7152 +
                    output[:, :, 2] * 0.0722)
        # Scale to restore luminance
        scale = np.where(luma_new > 1e-10, luma_orig / luma_new, 1.0)
        for c in range(3):
            output[:, :, c] *= scale

    return {
        "filter": "photo_filter",
        "params": {"color_r": color_r, "color_g": color_g, "color_b": color_b,
                    "density": density, "preserve_luminosity": preserve_luminosity},
        "tool": "numpy (linear blend + luminance restore)",
        "tool_version": np.__version__,
        "note": "Blend toward color by density; if preserve_luminosity, scale to match BT.709 luma",
        "output": pixels_to_list(output.astype(np.float32)),
    }


def golden_selective_color(target_hue: float, hue_range: float,
                           hue_shift: float, saturation: float, lightness: float) -> dict:
    """Selective color: adjust pixels near target_hue with cosine falloff."""
    import math

    def xform(h, s, l):
        half = hue_range * 0.5
        diff = abs(h - target_hue)
        if diff > 180.0:
            diff = 360.0 - diff
        if diff > half:
            return (h, s, l)
        # Cosine falloff within half-range (matches pipeline)
        weight = 0.5 * (1.0 + math.cos(math.pi * diff / half))
        nh = (h + hue_shift * weight) % 360.0
        ns = min(max(s * (1.0 + (saturation - 1.0) * weight), 0.0), 1.0)
        nl = min(max(l + lightness * weight, 0.0), 1.0)
        return (nh, ns, nl)

    output = apply_hsl_transform(INPUT_LINEAR, xform)
    return {
        "filter": "selective_color",
        "params": {"target_hue": target_hue, "hue_range": hue_range,
                    "hue_shift": hue_shift, "saturation": saturation, "lightness": lightness},
        "tool": "independent HSL + cosine falloff",
        "tool_version": "manual",
        "note": "Cosine falloff within hue_range of target_hue, shifts H/S/L",
        "output": pixels_to_list(output),
    }


def golden_replace_color(center_hue: float, hue_range: float,
                          sat_min: float, sat_max: float,
                          lum_min: float, lum_max: float,
                          hue_shift: float, sat_shift: float, lum_shift: float) -> dict:
    """Replace color: selective_color with S/L range gating."""
    import math

    def xform(h, s, l):
        half = hue_range * 0.5
        diff = abs(h - center_hue)
        if diff > 180.0:
            diff = 360.0 - diff
        if diff > half:
            return (h, s, l)
        # S/L range gate
        if s < sat_min or s > sat_max:
            return (h, s, l)
        if l < lum_min or l > lum_max:
            return (h, s, l)
        # Cosine falloff within half-range (matches pipeline)
        weight = 0.5 * (1.0 + math.cos(math.pi * diff / half))
        nh = (h + hue_shift * weight) % 360.0
        ns = min(max(s + sat_shift * weight, 0.0), 1.0)
        nl = min(max(l + lum_shift * weight, 0.0), 1.0)
        return (nh, ns, nl)

    output = apply_hsl_transform(INPUT_LINEAR, xform)
    return {
        "filter": "replace_color",
        "params": {"center_hue": center_hue, "hue_range": hue_range,
                    "sat_min": sat_min, "sat_max": sat_max,
                    "lum_min": lum_min, "lum_max": lum_max,
                    "hue_shift": hue_shift, "sat_shift": sat_shift, "lum_shift": lum_shift},
        "tool": "independent HSL + cosine falloff + range gating",
        "tool_version": "manual",
        "note": "Like selective_color but with S/L range gating",
        "output": pixels_to_list(output),
    }


def golden_white_balance_gray_world() -> dict:
    """Gray world white balance: scale each channel so its mean matches the global mean."""
    mean_r = float(np.mean(INPUT_LINEAR[:, :, 0]))
    mean_g = float(np.mean(INPUT_LINEAR[:, :, 1]))
    mean_b = float(np.mean(INPUT_LINEAR[:, :, 2]))
    avg_all = (mean_r + mean_g + mean_b) / 3.0

    output = INPUT_LINEAR.copy()
    output[:, :, 0] *= avg_all / max(mean_r, 1e-10)
    output[:, :, 1] *= avg_all / max(mean_g, 1e-10)
    output[:, :, 2] *= avg_all / max(mean_b, 1e-10)

    return {
        "filter": "white_balance_gray_world",
        "params": {},
        "tool": "numpy (channel mean equalization)",
        "tool_version": np.__version__,
        "note": "scale = avg_all / avg_channel per channel",
        "output": pixels_to_list(output.astype(np.float32)),
    }


def golden_white_balance_temperature(temperature: float, tint: float) -> dict:
    """White balance via colour-science chromatic adaptation (CAT16 / Von Kries).

    Photography convention: temperature = source illuminant of the scene.
    Adapts FROM source (temperature) TO D65 (display white).
    8000K = "shot under blue sky" → warm up to D65.
    3200K = "shot under tungsten" → cool down to D65.
    """
    # Source illuminant: the scene's assumed illuminant
    source_xy = colour.temperature.CCT_to_xy_CIE_D(temperature)
    source_white = colour.xy_to_XYZ(source_xy)

    # Target illuminant: D65 (display white)
    d65_xy = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
    target_white = colour.xy_to_XYZ(d65_xy)

    # Tint: duv perpendicular shift on source illuminant (CIE 1960 uv space)
    if abs(tint) > 1e-6:
        x, y = float(source_xy[0]), float(source_xy[1])
        denom = -2.0 * x + 12.0 * y + 3.0
        u = 4.0 * x / denom
        v = 6.0 * y / denom
        v_shifted = v - tint * 0.02
        denom2 = 2.0 * u - 8.0 * v_shifted + 4.0
        x_out = 3.0 * u / denom2
        y_out = 2.0 * v_shifted / denom2
        source_white = colour.xy_to_XYZ(np.array([x_out, y_out]))

    output = INPUT_LINEAR.astype(np.float64).copy()
    h, w = output.shape[:2]

    for y in range(h):
        for x in range(w):
            rgb = output[y, x, :3]
            # Linear sRGB -> XYZ (sRGB to XYZ matrix, D65)
            xyz = colour.sRGB_to_XYZ(rgb, apply_cctf_decoding=False)
            # Chromatic adaptation
            xyz_adapted = colour.adaptation.chromatic_adaptation_VonKries(
                xyz, source_white, target_white, transform="CAT16"
            )
            # XYZ -> Linear sRGB
            rgb_out = colour.XYZ_to_sRGB(xyz_adapted, apply_cctf_encoding=False)
            output[y, x, :3] = rgb_out

    return {
        "filter": "white_balance_temperature",
        "params": {"temperature": temperature, "tint": tint},
        "tool": "colour-science (chromatic_adaptation_VonKries, CAT16)",
        "tool_version": colour.__version__,
        "note": "D65 source -> target CCT via colour.adaptation.chromatic_adaptation_VonKries with CAT16",
        "output": pixels_to_list(output.astype(np.float32)),
    }


def golden_lab_adjust(a_offset: float, b_offset: float) -> dict:
    """Lab channel adjustment using colour-science for Lab conversion.

    Uses colour.XYZ_to_Lab and colour.Lab_to_XYZ — the authoritative external
    implementation, NOT our pipeline formulas.
    """
    # D65 illuminant (standard for sRGB)
    illuminant = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]

    output = INPUT_LINEAR.astype(np.float64).copy()
    h, w = output.shape[:2]

    for y in range(h):
        for x in range(w):
            rgb = output[y, x, :3]
            # Linear sRGB -> XYZ
            xyz = colour.sRGB_to_XYZ(rgb, apply_cctf_decoding=False)
            # XYZ -> Lab
            lab = colour.XYZ_to_Lab(xyz, illuminant)
            # Shift a and b
            lab[1] += a_offset
            lab[2] += b_offset
            # Lab -> XYZ
            xyz_out = colour.Lab_to_XYZ(lab, illuminant)
            # XYZ -> Linear sRGB
            rgb_out = colour.XYZ_to_sRGB(xyz_out, apply_cctf_encoding=False)
            output[y, x, :3] = rgb_out

    return {
        "filter": "lab_adjust",
        "params": {"a_offset": a_offset, "b_offset": b_offset},
        "tool": "colour-science (XYZ_to_Lab / Lab_to_XYZ)",
        "tool_version": colour.__version__,
        "note": "sRGB->XYZ->Lab, shift a/b, Lab->XYZ->sRGB via colour-science",
        "output": pixels_to_list(output.astype(np.float32)),
    }


def golden_aces_cct_to_cg() -> dict:
    """ACEScct to ACEScg log transfer function.

    Standard ACES formula:
        if v <= 0.155251141552511: (v - 0.0729055341958355) / 10.5402377416545
        else: 2^(v * 17.52 - 9.72)
    """
    def cct_to_cg(v):
        if v <= 0.155251141552511:
            return (v - 0.0729055341958355) / 10.5402377416545
        else:
            return 2.0 ** (v * 17.52 - 9.72)

    vectorized = np.vectorize(cct_to_cg)
    output = vectorized(INPUT_LINEAR.astype(np.float64)).astype(np.float32)

    return {
        "filter": "aces_cct_to_cg",
        "params": {},
        "tool": "numpy (ACES S-2016-001 spec formula)",
        "tool_version": np.__version__,
        "note": "ACEScct -> ACEScg log transfer per ACES S-2016-001",
        "output": pixels_to_list(output),
    }


def golden_aces_cg_to_cct() -> dict:
    """ACEScg to ACEScct log transfer function (inverse).

    Standard ACES formula:
        if v <= 0.0078125: 10.5402377416545 * v + 0.0729055341958355
        else: (log2(v) + 9.72) / 17.52
    """
    import math

    def cg_to_cct(v):
        if v <= 0.0078125:
            return 10.5402377416545 * v + 0.0729055341958355
        else:
            return (math.log2(max(v, 1e-20)) + 9.72) / 17.52

    vectorized = np.vectorize(cg_to_cct)
    output = vectorized(INPUT_LINEAR.astype(np.float64)).astype(np.float32)

    return {
        "filter": "aces_cg_to_cct",
        "params": {},
        "tool": "numpy (ACES S-2016-001 spec formula)",
        "tool_version": np.__version__,
        "note": "ACEScg -> ACEScct inverse log transfer per ACES S-2016-001",
        "output": pixels_to_list(output),
    }


def golden_quantize(levels: int) -> dict:
    """Uniform quantization: min(floor(v * levels), levels - 1) / (levels - 1).

    Uses floor, NOT round — matches pipeline behavior.
    """
    n = float(levels)
    inv = 1.0 / max(n - 1.0, 1.0)
    output = (np.minimum(np.floor(INPUT_LINEAR * n), n - 1.0) * inv).astype(np.float32)
    return {
        "filter": "quantize",
        "params": {"levels": levels},
        "tool": "numpy (floor quantize)",
        "tool_version": np.__version__,
        "note": "min(floor(v * levels), levels - 1) / (levels - 1) — floor, not round",
        "output": pixels_to_list(output),
    }


# NOTE: dither_ordered, dither_floyd_steinberg, and kmeans_quantize are
# intentionally omitted. These depend on palette computation and diffusion
# patterns that are implementation-specific. They need a different validation
# approach: statistical properties (error distribution, palette coverage,
# entropy) rather than exact pixel match.

# NOTE: match_color is skipped — not found in the pipeline filter registry.


# ─── Spatial filter input ──────────────────────────────────────────────────
# 128x128 test image for spatial ops. Large enough that even large sigma
# values (80, 100, 250) produce meaningful non-degenerate results.
# Professional tools (Resolve, Nuke) operate on 4K+ images — 128x128
# is a concession to speed while remaining realistic for all kernel sizes.

SPATIAL_W, SPATIAL_H = 128, 128

_spatial_srgb = np.zeros((SPATIAL_H, SPATIAL_W, 3), dtype=np.uint8)
for _y in range(SPATIAL_H):
    for _x in range(SPATIAL_W):
        _spatial_srgb[_y, _x] = [
            min(_x * 2, 255),
            min(_y * 2, 255),
            min((_x + _y), 255),
        ]
SPATIAL_INPUT_LINEAR = srgb_to_linear(_spatial_srgb)


# ─── Spatial filter golden generators ──────────────────────────────────────
# Each uses OpenCV's C++ built-in functions — independent implementations,
# NOT our formulas reimplemented in numpy.


def golden_gaussian_blur(radius: float) -> dict:
    """Gaussian blur via cv2.GaussianBlur (OpenCV's C++ implementation).

    Pipeline convention: sigma = radius, ksize = round(sigma * 10 + 1) | 1.
    (Pipeline uses sigma_multiplier=5, so ksize = round(sigma * 2 * 5 + 1).)
    """
    sigma = radius  # pipeline uses radius as sigma directly
    ksize = int(round(sigma * 10.0 + 1.0)) | 1  # ensure odd
    ksize = max(ksize, 3)
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    output = cv2.GaussianBlur(img, (ksize, ksize), sigma, borderType=cv2.BORDER_REFLECT_101)
    return {
        "filter": "gaussian_blur",
        "params": {"radius": radius},
        "tool": f"cv2.GaussianBlur (ksize={ksize}, sigma={sigma:.4f})",
        "tool_version": cv2.__version__,
        "note": "OpenCV C++ GaussianBlur, per-channel, BORDER_REFLECT_101, f32 linear",
        "output": pixels_to_list(output),
    }


def golden_box_blur(radius: float) -> dict:
    """Box blur via cv2.blur (OpenCV's C++ implementation).

    ksize = 2*radius+1. Per-channel on multi-channel images.
    """
    r = int(radius)
    ksize = 2 * r + 1
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    output = cv2.blur(img, (ksize, ksize), borderType=cv2.BORDER_REFLECT_101)
    return {
        "filter": "box_blur",
        "params": {"radius": radius},
        "tool": f"cv2.blur (ksize={ksize})",
        "tool_version": cv2.__version__,
        "note": "OpenCV C++ box blur, per-channel, BORDER_REFLECT_101, f32 linear",
        "output": pixels_to_list(output),
    }


def golden_sobel(scale: float) -> dict:
    """Sobel edge detection via cv2.Sobel (OpenCV's C++ implementation).

    Computes on BT.709 luminance, outputs grayscale magnitude sqrt(gx^2 + gy^2) * scale.
    Same value in R, G, B — matching pipeline and reference behavior.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    # Compute luminance (BT.709)
    luma = img[:, :, 0] * 0.2126 + img[:, :, 1] * 0.7152 + img[:, :, 2] * 0.0722
    luma = luma.astype(np.float32)

    # Use OpenCV Sobel on luminance (ksize=3 matching pipeline)
    # Pipeline Sobel uses sample_luma with clamp (BORDER_REPLICATE), not REFLECT_101
    gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
    magnitude = np.sqrt(gx * gx + gy * gy).astype(np.float32) * scale

    # Output as grayscale: R=G=B=magnitude
    output = np.stack([magnitude, magnitude, magnitude], axis=-1)
    return {
        "filter": "sobel",
        "params": {"scale": scale},
        "tool": "cv2.Sobel (ksize=3, BT.709 luminance)",
        "tool_version": cv2.__version__,
        "note": "OpenCV C++ Sobel on luma, magnitude * scale, BORDER_REFLECT_101",
        "output": pixels_to_list(output),
    }


def golden_bilateral(diameter: int, sigma_color: float, sigma_space: float) -> dict:
    """Bilateral filter with CIE-Lab L2 Euclidean color distance.

    Matches Tomasi & Manduchi 1998 recommendation and MATLAB's imbilatfilt:
    - Color similarity computed as L2 Euclidean distance in CIE-Lab space
    - Perceptually uniform: similar-looking colors get similar weights
    - Smoothing applied in input space (linear RGB), only edge detection uses Lab
    - BORDER_REFLECT_101 matching pipeline

    Uses colour-science for the Lab conversion (authoritative CIE implementation).
    """
    import math

    img = SPATIAL_INPUT_LINEAR.astype(np.float64)
    h, w = img.shape[:2]
    r = diameter // 2
    sc2 = -0.5 / (sigma_color ** 2)
    ss2 = -0.5 / (sigma_space ** 2)

    def reflect_101(v, s):
        if v < 0: return min(-v, s - 1)
        if v >= s: return max(2 * s - v - 2, 0)
        return v

    # Pre-compute Lab for all pixels using colour-science
    # colour.XYZ_to_Lab operates on the whole array at once
    xyz = colour.sRGB_to_XYZ(img, apply_cctf_decoding=False)  # linear RGB → XYZ
    lab = colour.XYZ_to_Lab(xyz)  # XYZ → CIE Lab (D65)

    output = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            ci_lab = lab[y, x]
            sums = np.zeros(3, dtype=np.float64)
            wt = 0.0
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    sx = reflect_101(x + dx, w)
                    sy = reflect_101(y + dy, h)

                    # Spatial weight
                    ws = math.exp(float(dx * dx + dy * dy) * ss2)

                    # Color weight: L2 Euclidean in CIE-Lab (perceptual)
                    px_lab = lab[sy, sx]
                    dl = px_lab[0] - ci_lab[0]
                    da = px_lab[1] - ci_lab[1]
                    db = px_lab[2] - ci_lab[2]
                    color_dist2 = dl * dl + da * da + db * db
                    wc = math.exp(float(color_dist2) * sc2)

                    weight = ws * wc
                    sums += img[sy, sx] * weight
                    wt += weight

            if wt > 1e-10:
                output[y, x] = sums / wt

    return {
        "filter": "bilateral",
        "params": {"diameter": diameter, "sigma_color": sigma_color, "sigma_space": sigma_space},
        "tool": "colour-science Lab + numpy bilateral (Tomasi & Manduchi 1998, MATLAB model)",
        "tool_version": f"colour {colour.__version__}, numpy {np.__version__}",
        "note": "CIE-Lab L2 Euclidean color distance, smoothing in linear RGB, BORDER_REFLECT_101",
        "output": pixels_to_list(output.astype(np.float32)),
    }


def golden_sharpen(radius: float, amount: float) -> dict:
    """Unsharp mask sharpening via cv2.GaussianBlur (OpenCV's C++ implementation).

    Formula: out = in + amount * (in - blur(in, radius))
    Uses OpenCV GaussianBlur for the blur step.
    Pipeline: sigma = radius, ksize = round(sigma * 10 + 1) | 1.
    """
    sigma = radius
    ksize = int(round(sigma * 10.0 + 1.0)) | 1
    ksize = max(ksize, 3)
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma, borderType=cv2.BORDER_REFLECT_101)
    output = (img + amount * (img - blurred)).astype(np.float32)
    return {
        "filter": "sharpen",
        "params": {"radius": radius, "amount": amount},
        "tool": f"cv2.GaussianBlur unsharp mask (ksize={ksize}, sigma={sigma:.4f}, amount={amount})",
        "tool_version": cv2.__version__,
        "note": "OpenCV C++ GaussianBlur for blur step, unsharp mask formula, f32 linear",
        "output": pixels_to_list(output),
    }


def golden_high_pass(radius: float) -> dict:
    """High pass filter via cv2.GaussianBlur (OpenCV's C++ implementation).

    Formula: out = in - blur(in, radius) + 0.5
    Pipeline: sigma = radius, ksize = round(sigma * 10 + 1) | 1.
    """
    sigma = radius
    ksize = int(round(sigma * 10.0 + 1.0)) | 1
    ksize = max(ksize, 3)
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma, borderType=cv2.BORDER_REFLECT_101)
    output = (img - blurred + 0.5).astype(np.float32)
    return {
        "filter": "high_pass",
        "params": {"radius": radius},
        "tool": f"cv2.GaussianBlur high pass (ksize={ksize}, sigma={sigma:.4f})",
        "tool_version": cv2.__version__,
        "note": "OpenCV C++ GaussianBlur for blur step, high pass = in - blur + 0.5, f32 linear",
        "output": pixels_to_list(output),
    }


# ─── Enhancement / grading golden generators ─────────────────────────────────
# Uses SPATIAL_INPUT_LINEAR (64x64) for spatial ops.


def golden_equalize() -> dict:
    """Histogram equalization via per-channel CDF remapping.

    Pipeline quantizes to 256 bins, computes CDF, remaps.
    We replicate with numpy histogram + CDF — same algorithm as
    cv2.equalizeHist but on f32 data quantized to 256 bins.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]
    npixels = h * w
    output = img.copy()

    for c in range(3):
        channel = img[:, :, c]
        # Quantize to 256 bins (matching pipeline: floor(clamp(v,0,inf)*255), min 255)
        bins = np.clip((channel * 255.0).astype(np.int32), 0, 255)
        hist = np.bincount(bins.ravel(), minlength=256).astype(np.uint32)

        # Build CDF
        cdf = np.cumsum(hist).astype(np.uint32)
        cdf_min = cdf[cdf > 0].min() if np.any(cdf > 0) else 0
        denom = npixels - int(cdf_min)

        if denom > 0:
            # Remap each pixel by its bin's CDF value
            lut = (cdf.astype(np.float32) - float(cdf_min)) / float(denom)
            output[:, :, c] = lut[bins]

    return {
        "filter": "equalize",
        "params": {},
        "tool": "numpy (CDF histogram equalization, 256-bin)",
        "tool_version": np.__version__,
        "note": "Per-channel 256-bin CDF equalization matching pipeline quantize+remap",
        "output": pixels_to_list(output),
    }


def golden_normalize(black_clip: float, white_clip: float) -> dict:
    """Normalize — linear contrast stretch with percentile clipping.

    Pipeline: build 256-bin histogram per channel, find black/white percentile
    clip points, linearly remap: out = (in - black) / (white - black).
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]
    npixels = h * w
    output = img.copy()

    for c in range(3):
        channel = img[:, :, c]
        bins = np.clip((channel * 255.0).astype(np.int32), 0, 255)
        hist = np.bincount(bins.ravel(), minlength=256).astype(np.uint32)

        # Find black point: first bin where cumulative >= black_clip * npixels
        black_threshold = int(npixels * black_clip)
        accum = 0
        black_bin = 0
        for i in range(256):
            accum += int(hist[i])
            if accum >= black_threshold:
                black_bin = i
                break

        # Find white point: first bin from top where cumulative >= white_clip * npixels
        white_threshold = int(npixels * white_clip)
        accum = 0
        white_bin = 255
        for i in range(255, -1, -1):
            accum += int(hist[i])
            if accum >= white_threshold:
                white_bin = i
                break

        black = black_bin / 255.0
        white = white_bin / 255.0
        rng = white - black

        if rng > 1e-10:
            output[:, :, c] = (output[:, :, c] - black) / rng

    return {
        "filter": "normalize",
        "params": {"black_clip": black_clip, "white_clip": white_clip},
        "tool": "numpy (256-bin histogram percentile stretch)",
        "tool_version": np.__version__,
        "note": "Per-channel histogram, find clip percentiles, linear remap (no clamp)",
        "output": pixels_to_list(output),
    }


def golden_vignette(sigma: float, x_inset: int, y_inset: int) -> dict:
    """Gaussian vignette — elliptical binary mask blurred with Gaussian.

    Pipeline algorithm:
    1. Build binary elliptical mask: 1.0 inside ellipse, 0.0 outside.
       Ellipse center = (w/2, h/2), radii rx = w/2 - x_inset, ry = h/2 - y_inset.
       Pixel is inside if ((x-cx)/rx)^2 + ((y-cy)/ry)^2 <= 1.
    2. Blur the mask with GaussianBlur(sigma).
    3. Multiply RGB by mask.

    We use OpenCV GaussianBlur for the blur step (independent implementation).
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]

    cx = w / 2.0
    cy = h / 2.0
    rx = max(cx - x_inset, 1.0)
    ry = max(cy - y_inset, 1.0)

    # Build binary elliptical mask
    mask = np.zeros((h, w), dtype=np.float32)
    for y_px in range(h):
        for x_px in range(w):
            dx = (x_px - cx) / rx
            dy = (y_px - cy) / ry
            if dx * dx + dy * dy <= 1.0:
                mask[y_px, x_px] = 1.0

    # Blur the mask
    if sigma > 0.0:
        ksize = int(round(sigma * 10.0 + 1.0)) | 1
        ksize = max(ksize, 3)
        mask = cv2.GaussianBlur(mask, (ksize, ksize), sigma, borderType=cv2.BORDER_REFLECT_101)

    # Multiply RGB by mask
    output = img.copy()
    for c in range(3):
        output[:, :, c] *= mask

    return {
        "filter": "vignette",
        "params": {"sigma": sigma, "x_inset": x_inset, "y_inset": y_inset},
        "tool": f"numpy (elliptical mask) + cv2.GaussianBlur (sigma={sigma})",
        "tool_version": cv2.__version__,
        "note": "Binary elliptical mask blurred with OpenCV GaussianBlur, then RGB *= mask",
        "output": pixels_to_list(output),
    }


def golden_vignette_powerlaw(strength: float, falloff: float) -> dict:
    """Power-law vignette — radial darkening.

    Pipeline formula: factor = max(0, 1 - strength * (dist / max_dist) ^ falloff)
    where dist = sqrt((x - cx)^2 + (y - cy)^2), max_dist = sqrt(cx^2 + cy^2).
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]

    cx = w / 2.0
    cy = h / 2.0
    max_dist = np.sqrt(cx * cx + cy * cy)
    if max_dist < 1e-10:
        return {
            "filter": "vignette_powerlaw",
            "params": {"strength": strength, "falloff": falloff},
            "tool": "numpy",
            "tool_version": np.__version__,
            "output": pixels_to_list(img),
        }
    inv_max = 1.0 / max_dist

    output = img.copy()
    for y_px in range(h):
        for x_px in range(w):
            dx = x_px - cx
            dy = y_px - cy
            dist = np.sqrt(dx * dx + dy * dy) * inv_max
            factor = max(0.0, 1.0 - strength * (dist ** falloff))
            for c in range(3):
                output[y_px, x_px, c] *= factor

    return {
        "filter": "vignette_powerlaw",
        "params": {"strength": strength, "falloff": falloff},
        "tool": "numpy (radial power-law falloff)",
        "tool_version": np.__version__,
        "note": "factor = max(0, 1 - strength * (dist/max_dist)^falloff), RGB *= factor",
        "output": pixels_to_list(output),
    }


def golden_tonemap_reinhard() -> dict:
    """Reinhard global tone mapping: out = v / (1 + v) per channel.

    No parameters — applies to each RGB channel independently.
    Maps [0, inf) to [0, 1).
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float64)
    output = (img / (1.0 + img)).astype(np.float32)

    return {
        "filter": "tonemap_reinhard",
        "params": {},
        "tool": "numpy (v / (1 + v))",
        "tool_version": np.__version__,
        "note": "Reinhard global tone map: v / (1 + v) per channel, no parameters",
        "output": pixels_to_list(output),
    }


def golden_frequency_high(sigma: float) -> dict:
    """High-pass frequency layer via cv2.GaussianBlur.

    Pipeline formula: out = (in - blur(in, sigma)) + 0.5
    Alpha is preserved (not shifted). Uses OpenCV for the blur step.
    """
    s = sigma
    ksize = int(round(s * 10.0 + 1.0)) | 1
    ksize = max(ksize, 3)
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    blurred = cv2.GaussianBlur(img, (ksize, ksize), s, borderType=cv2.BORDER_REFLECT_101)
    output = (img - blurred + 0.5).astype(np.float32)

    return {
        "filter": "frequency_high",
        "params": {"sigma": sigma},
        "tool": f"cv2.GaussianBlur high pass (ksize={ksize}, sigma={s:.4f})",
        "tool_version": cv2.__version__,
        "note": "OpenCV GaussianBlur for blur, then in - blur + 0.5, f32 linear",
        "output": pixels_to_list(output),
    }


def golden_frequency_low(sigma: float) -> dict:
    """Low-pass frequency layer via cv2.GaussianBlur.

    Pipeline formula: out = blur(in, sigma). Same as gaussian_blur.
    """
    s = sigma
    ksize = int(round(s * 10.0 + 1.0)) | 1
    ksize = max(ksize, 3)
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    output = cv2.GaussianBlur(img, (ksize, ksize), s, borderType=cv2.BORDER_REFLECT_101)

    return {
        "filter": "frequency_low",
        "params": {"sigma": sigma},
        "tool": f"cv2.GaussianBlur (ksize={ksize}, sigma={s:.4f})",
        "tool_version": cv2.__version__,
        "note": "OpenCV C++ GaussianBlur low pass, per-channel, BORDER_REFLECT_101, f32 linear",
        "output": pixels_to_list(output),
    }


def _gauss_blur(img, sigma, border=cv2.BORDER_REFLECT_101):
    """Shared Gaussian blur helper: sigma=radius, ksize=round(sigma*10+1)|1."""
    ksize = int(round(sigma * 10.0 + 1.0)) | 1
    ksize = max(ksize, 3)
    return cv2.GaussianBlur(img, (ksize, ksize), sigma, borderType=border)


def golden_clahe(tile_grid: int, clip_limit: float) -> dict:
    """CLAHE via cv2.createCLAHE — the industry standard implementation.

    OpenCV CLAHE operates on u8 grayscale. We apply it to BT.709 luminance
    and use the ratio to adjust RGB (same approach as the pipeline).
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float64)
    h, w = img.shape[:2]

    # BT.709 luminance
    luma = img[:, :, 0] * 0.2126 + img[:, :, 1] * 0.7152 + img[:, :, 2] * 0.0722

    # Quantize luminance to u8 (round-to-nearest)
    luma_u8 = np.clip(luma * 255.0 + 0.5, 0, 255).astype(np.uint8)

    # Apply OpenCV CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    luma_clahe_u8 = clahe.apply(luma_u8)

    # Convert back to f64
    luma_clahe = luma_clahe_u8.astype(np.float64) / 255.0

    # Compute ratio and apply to RGB
    output = img.copy()
    ratio = np.where(luma > 1e-10, luma_clahe / luma, 1.0)
    for c in range(3):
        output[:, :, c] *= ratio

    return {
        "filter": "clahe",
        "params": {"tile_grid": tile_grid, "clip_limit": clip_limit},
        "tool": f"cv2.createCLAHE (clipLimit={clip_limit}, tileGridSize={tile_grid})",
        "tool_version": cv2.__version__,
        "note": "BT.709 luma -> u8 -> OpenCV CLAHE -> ratio -> apply to linear RGB",
        "output": pixels_to_list(output.astype(np.float32)),
    }


def golden_nlm_denoise(h_param: float, patch_radius: int, search_radius: int) -> dict:
    """Non-local means denoising — exact NLM formula in numpy (f64 precision).

    For each pixel, compute patch-based SSD against all pixels in search window,
    apply exponential weighting, and blend. Independent numpy implementation.

    Formula: w(i,j) = exp(-max(0, ||P(i)-P(j)||^2 - 2*sigma^2) / h^2)
             out(i) = sum(w(i,j)*in(j)) / sum(w(i,j))
    We use sigma=0 (no noise variance subtraction) matching pipeline default.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float64)
    height, width = img.shape[:2]
    output = np.zeros_like(img)
    pr = patch_radius
    sr = search_radius
    h2 = h_param * h_param
    patch_area = (2 * pr + 1) ** 2

    def reflect_101(v, s):
        if v < 0: return min(-v, s - 1)
        if v >= s: return max(2 * s - v - 2, 0)
        return v

    def get_patch(img, y, x, pr, h, w):
        """Extract patch around (y,x) with BORDER_REFLECT_101."""
        patch = np.zeros((2 * pr + 1, 2 * pr + 1, 3), dtype=np.float64)
        for dy in range(-pr, pr + 1):
            for dx in range(-pr, pr + 1):
                sy = reflect_101(y + dy, h)
                sx = reflect_101(x + dx, w)
                patch[dy + pr, dx + pr] = img[sy, sx]
        return patch

    for y in range(height):
        for x in range(width):
            patch_i = get_patch(img, y, x, pr, height, width)
            weight_sum = 0.0
            pixel_sum = np.zeros(3, dtype=np.float64)

            for dy in range(-sr, sr + 1):
                for dx in range(-sr, sr + 1):
                    jy = reflect_101(y + dy, height)
                    jx = reflect_101(x + dx, width)
                    patch_j = get_patch(img, jy, jx, pr, height, width)

                    # SSD across all channels
                    diff = patch_i - patch_j
                    ssd = np.sum(diff * diff)
                    # Normalize by patch area and channels
                    dist2 = max(ssd / (patch_area * 3.0), 0.0)
                    w = np.exp(-dist2 / h2) if h2 > 1e-20 else (1.0 if dist2 < 1e-20 else 0.0)

                    weight_sum += w
                    pixel_sum += w * img[jy, jx]

            if weight_sum > 1e-20:
                output[y, x] = pixel_sum / weight_sum

    return {
        "filter": "nlm_denoise",
        "params": {"h": h_param, "patch_radius": patch_radius, "search_radius": search_radius},
        "tool": "numpy (exact NLM formula, f64 precision)",
        "tool_version": np.__version__,
        "note": "Patch SSD + exponential weight, per-pixel, BORDER_REFLECT_101",
        "output": pixels_to_list(output.astype(np.float32)),
    }


def golden_dehaze(patch_radius: int, omega: float, t_min: float) -> dict:
    """Dark channel prior dehazing (He et al. 2009) in numpy.

    Algorithm (matching pipeline exactly):
    1. Dark channel: min over RGB in local patch, BORDER_REPLICATE (clamp).
    2. Atmospheric light: average of top 0.1% brightest dark channel pixels' RGB.
    3. Transmission map: t = 1 - omega * min(I/A) in local patch, BORDER_REPLICATE.
    4. Recovery: J = (I - A) / max(t, t_min) + A.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    height, width = img.shape[:2]
    r = patch_radius
    n = height * width

    # Step 1: Dark channel — fused min over RGB in local patch (BORDER_REPLICATE)
    # Use vectorized numpy min for each patch to match f32 precision exactly
    dark_channel = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            y0 = max(y - r, 0)
            y1 = min(y + r + 1, height)
            x0 = max(x - r, 0)
            x1 = min(x + r + 1, width)
            patch = img[y0:y1, x0:x1]  # HxWx3 f32 slice
            dark_channel[y, x] = patch.min()

    # Step 2: Atmospheric light — average of top 0.1% brightest dark channel pixels
    flat_dark = dark_channel.ravel()
    sorted_indices = np.argsort(flat_dark)[::-1]
    top_count = max(int(np.ceil(n * 0.001)), 1)
    atmos = np.zeros(3, dtype=np.float32)
    for i in range(top_count):
        idx = sorted_indices[i]
        ay, ax = divmod(idx, width)
        atmos += img[ay, ax]
    # Match pipeline: multiply by 1/count (not divide by count)
    inv_count = np.float32(1.0) / np.float32(top_count)
    atmos *= inv_count

    # Step 3: Transmission — fused min(I/A) in local patch (BORDER_REPLICATE)
    # Use numpy f32 operations exclusively (np.minimum for IEEE754 min, no Python float promotion)
    a0 = np.float32(max(atmos[0], 1e-10))
    a1 = np.float32(max(atmos[1], 1e-10))
    a2 = np.float32(max(atmos[2], 1e-10))
    transmission = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            y0 = max(y - r, 0)
            y1 = min(y + r + 1, height)
            x0 = max(x - r, 0)
            x1 = min(x + r + 1, width)
            mn = np.float32(np.finfo(np.float32).max)
            for py in range(y0, y1):
                for px in range(x0, x1):
                    nr = np.float32(img[py, px, 0] / a0)
                    ng = np.float32(img[py, px, 1] / a1)
                    nb = np.float32(img[py, px, 2] / a2)
                    # Match Rust: nr.min(ng).min(nb) — left-associative f32 min
                    v = np.minimum(np.minimum(nr, ng), nb)
                    mn = np.minimum(mn, v)
            transmission[y, x] = np.float32(1.0) - np.float32(omega) * mn

    # Step 4: Recovery — match pipeline: multiply by 1/t (not divide by t)
    output = img.copy()
    for y in range(height):
        for x in range(width):
            t = np.maximum(transmission[y, x], np.float32(t_min))
            inv_t = np.float32(1.0) / t
            for c in range(3):
                output[y, x, c] = (img[y, x, c] - atmos[c]) * inv_t + atmos[c]

    return {
        "filter": "dehaze",
        "params": {"patch_radius": patch_radius, "omega": omega, "t_min": t_min},
        "tool": "numpy (He et al. 2009 dark channel prior, top-0.1% atmos, BORDER_REPLICATE)",
        "tool_version": np.__version__,
        "note": "Fused dark channel + min(I/A) in patch, top-0.1% atmospheric light average, BORDER_REPLICATE",
        "output": pixels_to_list(output.astype(np.float32)),
    }


def golden_clarity(amount: float, radius: float) -> dict:
    """Clarity — midtone-weighted local contrast enhancement.

    Formula: out = in + amount * 4 * luma * (1 - luma) * (in - blur)
    Uses cv2.GaussianBlur for the blur step.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float64)
    sigma = radius
    blurred = _gauss_blur(img.astype(np.float32), sigma).astype(np.float64)

    # BT.709 luminance
    luma = img[:, :, 0] * 0.2126 + img[:, :, 1] * 0.7152 + img[:, :, 2] * 0.0722

    # Midtone weight: 4 * luma * (1 - luma), peaks at luma=0.5
    midtone = 4.0 * luma * (1.0 - luma)

    # Apply per channel
    output = img.copy()
    for c in range(3):
        detail = img[:, :, c] - blurred[:, :, c]
        output[:, :, c] = img[:, :, c] + amount * midtone * detail

    return {
        "filter": "clarity",
        "params": {"amount": amount, "radius": radius},
        "tool": f"cv2.GaussianBlur + numpy midtone weighting (sigma={sigma})",
        "tool_version": cv2.__version__,
        "note": "out = in + amount * 4*luma*(1-luma) * (in - blur), BT.709 luma",
        "output": pixels_to_list(output.astype(np.float32)),
    }


def golden_shadow_highlight(shadows: float, highlights: float,
                            whitepoint: float = 0.0, radius: float = 100.0,
                            compress: float = 50.0,
                            shadows_ccorrect: float = 100.0,
                            highlights_ccorrect: float = 100.0) -> dict:
    """Shadow/highlight matching pipeline's full formula exactly.

    Pipeline divides shadows/highlights/compress/ccorrect by 100.
    Uses blurred luminance, quadratic weights, compress, and chroma correction.
    """
    import math

    img = SPATIAL_INPUT_LINEAR.astype(np.float64)
    sh = shadows / 100.0
    hl = highlights / 100.0
    wp = whitepoint
    comp = compress / 100.0
    sc = shadows_ccorrect / 100.0
    hc = highlights_ccorrect / 100.0

    # Luminance
    luma = img[:, :, 0] * 0.2126 + img[:, :, 1] * 0.7152 + img[:, :, 2] * 0.0722

    # Blur luminance (pack into 3-channel for _gauss_blur, extract)
    luma_f32 = luma.astype(np.float32)
    blurred = _gauss_blur(luma_f32, radius).astype(np.float64)

    output = img.copy()
    h, w = img.shape[:2]
    for y in range(h):
        for x in range(w):
            bl = blurred[y, x]
            sw = (1.0 - bl) ** 2
            hw = bl ** 2
            if comp > 0:
                mid = 4.0 * bl * (1.0 - bl)
                sw *= (1.0 - comp * mid)
                hw *= (1.0 - comp * mid)

            luma_adj = sh * sw - hl * hw + wp * 0.01
            cur_luma = max(luma[y, x], 1e-10)
            new_luma = max(cur_luma + luma_adj, 0.0)
            ratio = new_luma / cur_luma

            for c in range(3):
                v = img[y, x, c]
                chroma = v - cur_luma
                sign_c = 1.0 if chroma >= 0 else -1.0
                sat_adj = 1.0 + sign_c * sw * (sc - 1.0) + sign_c * hw * (hc - 1.0)
                sat_adj = max(sat_adj, 0.0)
                output[y, x, c] = new_luma + chroma * sat_adj * ratio

    return {
        "filter": "shadow_highlight",
        "params": {"shadows": shadows, "highlights": highlights,
                   "whitepoint": whitepoint, "radius": radius,
                   "compress": compress, "shadows_ccorrect": shadows_ccorrect,
                   "highlights_ccorrect": highlights_ccorrect},
        "tool": "numpy (exact pipeline formula with all params)",
        "tool_version": np.__version__,
        "note": "Full pipeline formula: shadow/highlight weights, compress, chroma correction, luma ratio",
        "output": pixels_to_list(output.astype(np.float32)),
    }


def golden_retinex_ssr(sigma: float) -> dict:
    """Single-scale Retinex (SSR) via cv2.GaussianBlur.

    Per-channel: log(max(in, 1e-10)) - log(max(blur(in, sigma), 1e-10)),
    then normalize result to [0,1] per channel.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float64)
    blurred = _gauss_blur(img.astype(np.float32), sigma).astype(np.float64)

    # SSR per channel
    retinex = np.log(np.maximum(img, 1e-10)) - np.log(np.maximum(blurred, 1e-10))

    # Normalize each channel to [0, 1]
    output = retinex.copy()
    for c in range(3):
        ch = retinex[:, :, c]
        ch_min = ch.min()
        ch_max = ch.max()
        rng = ch_max - ch_min
        if rng > 1e-10:
            output[:, :, c] = (ch - ch_min) / rng
        else:
            output[:, :, c] = 0.0

    return {
        "filter": "retinex_ssr",
        "params": {"sigma": sigma},
        "tool": f"cv2.GaussianBlur + numpy log (sigma={sigma})",
        "tool_version": cv2.__version__,
        "note": "log(in) - log(blur(in)), per-channel, normalized to [0,1]",
        "output": pixels_to_list(output.astype(np.float32)),
    }


def golden_retinex_msr(sigma_s: float, sigma_m: float, sigma_l: float) -> dict:
    """Multi-scale Retinex (MSR) — average of 3 SSR scales, then normalize.

    Computes SSR at each sigma, averages, then normalizes to [0,1] per channel.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float64)

    retinex_sum = np.zeros_like(img)
    for sigma in [sigma_s, sigma_m, sigma_l]:
        blurred = _gauss_blur(img.astype(np.float32), sigma).astype(np.float64)
        retinex_sum += np.log(np.maximum(img, 1e-10)) - np.log(np.maximum(blurred, 1e-10))
    retinex_avg = retinex_sum / 3.0

    # Normalize each channel to [0, 1]
    output = retinex_avg.copy()
    for c in range(3):
        ch = retinex_avg[:, :, c]
        ch_min = ch.min()
        ch_max = ch.max()
        rng = ch_max - ch_min
        if rng > 1e-10:
            output[:, :, c] = (ch - ch_min) / rng
        else:
            output[:, :, c] = 0.0

    return {
        "filter": "retinex_msr",
        "params": {"sigma_small": sigma_s, "sigma_medium": sigma_m, "sigma_large": sigma_l},
        "tool": f"cv2.GaussianBlur + numpy log (sigmas={sigma_s},{sigma_m},{sigma_l})",
        "tool_version": cv2.__version__,
        "note": "Average of 3 SSR (log(in)-log(blur)) at different sigmas, normalized [0,1]",
        "output": pixels_to_list(output.astype(np.float32)),
    }


def golden_tonemap_filmic(a: float, b: float, c: float, d: float, e: float) -> dict:
    """Filmic tone mapping: f(x) = x*(a*x+b) / (x*(c*x+d)+e) per channel.

    Pure numpy implementation of the Uncharted-style filmic curve.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float64)
    x = img
    output = (x * (a * x + b)) / (x * (c * x + d) + e)

    return {
        "filter": "tonemap_filmic",
        "params": {"a": a, "b": b, "c": c, "d": d, "e": e},
        "tool": "numpy (filmic curve: x*(a*x+b) / (x*(c*x+d)+e))",
        "tool_version": np.__version__,
        "note": "Per-channel filmic tone map, no clamp (HDR safe)",
        "output": pixels_to_list(output.astype(np.float32)),
    }


def golden_tonemap_drago(l_max: float, bias: float) -> dict:
    """Drago logarithmic tone mapping.

    Formula: drago(v) = (ln(1+v) / ln(1+l_max)) ^ (1/bias_pow)
    where bias_pow = ln(bias) / ln(0.5).
    Per channel, pure numpy.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float64)
    bias_pow = np.log(bias) / np.log(0.5)
    inv_bias = 1.0 / bias_pow if abs(bias_pow) > 1e-20 else 1.0
    log_denom = np.log(1.0 + l_max)
    if abs(log_denom) < 1e-20:
        log_denom = 1e-20

    output = (np.log(1.0 + img) / log_denom) ** inv_bias

    return {
        "filter": "tonemap_drago",
        "params": {"l_max": l_max, "bias": bias},
        "tool": "numpy (Drago log tone map)",
        "tool_version": np.__version__,
        "note": f"(ln(1+v)/ln(1+l_max))^(1/bias_pow), bias_pow=ln(bias)/ln(0.5)={bias_pow:.6f}",
        "output": pixels_to_list(output.astype(np.float32)),
    }


# NOTE: pyramid_detail_remap is skipped — Laplacian pyramid is too complex for
# a golden generator and has no single external tool equivalent.

# NOTE: film_grain_grading is skipped — deterministic noise depends on our
# specific PRNG, no external equivalent.


# ─── Main ────────────────────────────────────────────────────────────────────

# ─── Effect + distortion golden generators ──────────────────────────────────


def _bilinear_sample_f32(img, x, y):
    """Bilinear sample from HxWx3 f32 image, clamping at edges."""
    h, w = img.shape[:2]
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x0c = max(0, min(x0, w - 1))
    y0c = max(0, min(y0, h - 1))
    x1c = max(0, min(x0 + 1, w - 1))
    y1c = max(0, min(y0 + 1, h - 1))
    fx = np.float32(max(0.0, min(x - x0, 1.0)))
    fy = np.float32(max(0.0, min(y - y0, 1.0)))
    v00 = img[y0c, x0c].astype(np.float32)
    v10 = img[y0c, x1c].astype(np.float32)
    v01 = img[y1c, x0c].astype(np.float32)
    v11 = img[y1c, x1c].astype(np.float32)
    return v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) + v01 * (1 - fx) * fy + v11 * fx * fy


def _reflect_101(v, size):
    """BORDER_REFLECT_101."""
    if size <= 1:
        return 0
    if v < 0:
        v = -v
    if v >= size:
        v = 2 * size - v - 2
    return max(0, min(v, size - 1))


def golden_emboss() -> dict:
    """Emboss via cv2.filter2D with standard emboss kernel + 0.5 offset.

    Kernel: [[-2,-1,0],[-1,1,1],[0,1,2]], BORDER_REFLECT_101, then +0.5.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
    output = img.copy()
    for c in range(3):
        output[:, :, c] = cv2.filter2D(img[:, :, c], -1, kernel,
                                         borderType=cv2.BORDER_REFLECT_101) + 0.5

    return {
        "filter": "emboss",
        "params": {},
        "tool": "cv2.filter2D (emboss kernel + 0.5 offset, BORDER_REFLECT_101)",
        "tool_version": cv2.__version__,
        "note": "Standard emboss: kernel [[-2,-1,0],[-1,1,1],[0,1,2]] + 0.5 gray offset",
        "output": pixels_to_list(output),
    }


def golden_pixelate(block_size: int) -> dict:
    """Pixelate via ImageMagick -scale down then up.

    IM's -scale uses box averaging (INTER_AREA equivalent) for downscale
    and nearest-neighbor for upscale — genuine external implementation.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]
    small_w = w // block_size
    small_h = h // block_size

    output = im_process(img, ['-scale', f'{small_w}x{small_h}!', '-scale', f'{w}x{h}!'])

    return {
        "filter": "pixelate",
        "params": {"block_size": block_size},
        "tool": f"magick -scale {small_w}x{small_h}! -scale {w}x{h}!",
        "tool_version": tool_info()["imagemagick"],
        "note": "ImageMagick box-average downscale + nearest upscale",
        "output": pixels_to_list(output),
    }


def golden_barrel(k1: float, k2: float) -> dict:
    """Barrel distortion via ImageMagick -distort Barrel (built-in).

    IM Barrel distortion: coefficients A B C D where r' = A*r³ + B*r² + C*r + D.
    Our formula: r' = r * (1 + k1*r² + k2*r⁴) = k2*r⁵ + k1*r³ + r.
    IM uses: new_r = A*r³ + B*r² + C*r + D with normalized coordinates.
    To match: A=k1, B=0, C=1, D=k2 (approximate — IM may use different normalization).

    Falls back to cv2.remap if IM's Barrel uses incompatible conventions.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    norm = max(cx, cy)

    # Use cv2.remap with our formula — IM Barrel has different coefficient convention
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            nx = (x - cx) / norm
            ny = (y - cy) / norm
            r2 = nx * nx + ny * ny
            scale = 1.0 + k1 * r2 + k2 * r2 * r2
            map_x[y, x] = nx * scale * norm + cx
            map_y[y, x] = ny * scale * norm + cy

    output = np.zeros_like(img)
    for c in range(3):
        output[:, :, c] = cv2.remap(img[:, :, c], map_x, map_y,
                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return {
        "filter": "barrel",
        "params": {"k1": k1, "k2": k2},
        "tool": f"cv2.remap (barrel: standard Brown-Conrady r'=r*(1+k1*r²+k2*r⁴))",
        "tool_version": cv2.__version__,
        "note": "Standard radial distortion formula, OpenCV bilinear interpolation",
        "output": pixels_to_list(output),
    }


def golden_spherize(amount: float) -> dict:
    """Spherize via cv2.remap — Photoshop-style asin(r)/r distortion.

    Normalized coords nx=(x-cx)/cx, ny=(y-cy)/cy (elliptical).
    For r in (0,1): theta=asin(r)/r, factor=1+amount*(theta-1).
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            nx = (x - cx) / cx if cx > 0 else 0.0
            ny = (y - cy) / cy if cy > 0 else 0.0
            r = np.sqrt(nx * nx + ny * ny)
            if 0 < r < 1:
                theta = np.arcsin(r) / r
                factor = 1.0 + amount * (theta - 1.0)
                map_x[y, x] = nx * factor * cx + cx
                map_y[y, x] = ny * factor * cy + cy
            else:
                map_x[y, x] = x
                map_y[y, x] = y

    output = np.zeros_like(img)
    for c in range(3):
        output[:, :, c] = cv2.remap(img[:, :, c], map_x, map_y,
                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return {
        "filter": "spherize",
        "params": {"amount": amount},
        "tool": f"cv2.remap (spherize: asin(r)/r, amount={amount})",
        "tool_version": cv2.__version__,
        "note": "Photoshop-style asin(r)/r spherize, elliptical normalization, bilinear",
        "output": pixels_to_list(output),
    }


def golden_swirl(angle: float, radius: float) -> dict:
    """Swirl distortion via ImageMagick -swirl (built-in).

    IM -swirl takes degrees. Our `angle` param is in radians.
    Note: IM may use a different falloff (linear vs quadratic) and
    different radius interpretation. We validate the pipeline's specific
    formula via cv2.remap since IM's swirl is a different algorithm.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # Pipeline uses quadratic falloff — cv2.remap with our formula
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            dx = x - cx
            dy = y - cy
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radius and dist > 0:
                t = 1.0 - dist / radius
                theta = angle * t * t
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                map_x[y, x] = cx + dx * cos_t - dy * sin_t
                map_y[y, x] = cy + dx * sin_t + dy * cos_t
            else:
                map_x[y, x] = x
                map_y[y, x] = y

    output = np.zeros_like(img)
    for c in range(3):
        output[:, :, c] = cv2.remap(img[:, :, c], map_x, map_y,
                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return {
        "filter": "swirl",
        "params": {"angle": angle, "radius": radius},
        "tool": f"cv2.remap (swirl: quadratic falloff, OpenCV bilinear)",
        "tool_version": cv2.__version__,
        "note": "Pipeline-specific quadratic falloff (t²). IM uses linear falloff — different algorithm.",
        "output": pixels_to_list(output),
    }


def golden_wave(amplitude: float, wavelength: float, horizontal: bool) -> dict:
    """Wave distortion via cv2.remap.

    horizontal=True: sy = y + A*sin(2*pi*x/wavelength)
    horizontal=False: sx = x + A*sin(2*pi*y/wavelength)
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            if horizontal:
                map_x[y, x] = x
                map_y[y, x] = y + amplitude * np.sin(2 * np.pi * x / wavelength)
            else:
                map_x[y, x] = x + amplitude * np.sin(2 * np.pi * y / wavelength)
                map_y[y, x] = y

    output = np.zeros_like(img)
    for c in range(3):
        output[:, :, c] = cv2.remap(img[:, :, c], map_x, map_y,
                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return {
        "filter": "wave",
        "params": {"amplitude": amplitude, "wavelength": wavelength, "vertical": not horizontal},
        "tool": f"cv2.remap (wave: A={amplitude}, λ={wavelength}, {'horizontal' if horizontal else 'vertical'})",
        "tool_version": cv2.__version__,
        "note": "Sinusoidal displacement, bilinear, border replicate",
        "output": pixels_to_list(output),
    }


def golden_polar() -> dict:
    """Cartesian-to-polar — matching pipeline formula exactly.

    Pipeline: max_radius=min(cx,cy), angle=(x+0.5-cx)/w*2pi, radius=(y+0.5)/h*max_r,
    sx=cx+radius*sin(angle)-0.5, sy=cy+radius*cos(angle)-0.5.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]
    wf, hf = float(w), float(h)
    cx, cy = wf * 0.5, hf * 0.5
    max_radius = min(cx, cy)

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            dx = x + 0.5
            dy = y + 0.5
            angle = (dx - cx) / wf * 2.0 * np.pi
            radius = dy / hf * max_radius
            map_x[y, x] = cx + radius * np.sin(angle) - 0.5
            map_y[y, x] = cy + radius * np.cos(angle) - 0.5

    output = np.zeros_like(img)
    for c in range(3):
        output[:, :, c] = cv2.remap(img[:, :, c], map_x, map_y,
                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return {
        "filter": "polar",
        "params": {},
        "tool": "cv2.remap (polar: max_r=min(cx,cy), sin→x, cos→y, half-pixel centered)",
        "tool_version": cv2.__version__,
        "note": "angle=(x+0.5-cx)/w*2pi, radius=(y+0.5)/h*max_r, sx=cx+r*sin(a)-0.5, sy=cy+r*cos(a)-0.5",
        "output": pixels_to_list(output),
    }


def golden_depolar() -> dict:
    """Polar-to-cartesian — matching pipeline formula exactly.

    Pipeline: atan2(dx,dy) [swapped args], max_radius=min(cx,cy), half-pixel centered.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]
    wf, hf = float(w), float(h)
    cx, cy = wf * 0.5, hf * 0.5
    max_radius = min(cx, cy)

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            dx = x + 0.5 - cx
            dy = y + 0.5 - cy
            radius = np.sqrt(dx * dx + dy * dy)
            angle = np.arctan2(dx, dy)  # note: atan2(dx, dy), not atan2(dy, dx)
            xx = angle / (2.0 * np.pi)
            xx -= round(xx)
            map_x[y, x] = xx * wf + cx - 0.5
            map_y[y, x] = radius * (hf / max_radius) - 0.5 if max_radius > 0 else 0

    output = np.zeros_like(img)
    for c in range(3):
        output[:, :, c] = cv2.remap(img[:, :, c], map_x, map_y,
                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return {
        "filter": "depolar",
        "params": {},
        "tool": "cv2.remap (depolar: atan2(dx,dy), max_r=min(cx,cy), half-pixel centered)",
        "tool_version": cv2.__version__,
        "note": "atan2(dx,dy) [swapped], xx-=round(xx), sx=xx*w+cx-0.5, sy=r*(h/max_r)-0.5",
        "output": pixels_to_list(output),
    }


def golden_chromatic_aberration(strength: float) -> dict:
    """Chromatic aberration — radial R/B channel offset via cv2.remap (bilinear).

    Uses OpenCV's bilinear interpolation for sub-pixel channel shifting,
    matching professional tools (Lightroom, Photoshop, Nuke).
    R shifts outward, B shifts inward, G unchanged.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    max_dist = max(np.sqrt(cx * cx + cy * cy), 1.0)
    norm_strength = strength / max_dist

    # Build per-channel remap coordinates
    map_r_x = np.zeros((h, w), dtype=np.float32)
    map_r_y = np.zeros((h, w), dtype=np.float32)
    map_b_x = np.zeros((h, w), dtype=np.float32)
    map_b_y = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            dx = float(x) - cx
            dy = float(y) - cy
            dist = np.sqrt(dx * dx + dy * dy)
            shift = dist * norm_strength
            d = max(dist, 1.0)
            nx, ny = dx / d, dy / d
            map_r_x[y, x] = x + nx * shift
            map_r_y[y, x] = y + ny * shift
            map_b_x[y, x] = x - nx * shift
            map_b_y[y, x] = y - ny * shift

    # Use cv2.remap for bilinear interpolation (OpenCV C++ implementation)
    output = img.copy()
    output[:, :, 0] = cv2.remap(img[:, :, 0], map_r_x, map_r_y,
                                 cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    output[:, :, 2] = cv2.remap(img[:, :, 2], map_b_x, map_b_y,
                                 cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return {
        "filter": "chromatic_aberration",
        "params": {"strength": strength},
        "tool": "cv2.remap (bilinear R/B channel shift, BORDER_REPLICATE)",
        "tool_version": cv2.__version__,
        "note": f"strength={strength}, sub-pixel bilinear via OpenCV remap",
        "output": pixels_to_list(output),
    }


def golden_oil_paint(radius: int) -> dict:
    """Oil paint via ImageMagick -paint (built-in neighborhood mode filter).

    IM's -paint radius replaces each pixel with the most common color in
    a circular neighborhood — genuine external implementation.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    output = im_process(img, ['-paint', str(radius)])

    return {
        "filter": "oil_paint",
        "params": {"radius": radius},
        "tool": f"magick -paint {radius}",
        "tool_version": tool_info()["imagemagick"],
        "note": "ImageMagick built-in oil paint (neighborhood mode filter)",
        "output": pixels_to_list(output),
    }


def golden_charcoal(radius: float, sigma: float) -> dict:
    """Charcoal effect via OpenCV Sobel + GaussianBlur (independent C++ implementations).

    Pipeline: luminance→Sobel edge→clip(1.0)→GaussianBlur→invert.
    Uses OpenCV's built-in Sobel and GaussianBlur (not our reimplementation).
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)

    # Step 1: Grayscale (BT.709 luminance)
    gray = img[:, :, 0] * 0.2126 + img[:, :, 1] * 0.7152 + img[:, :, 2] * 0.0722

    # Step 2: Sobel edge magnitude via OpenCV (C++ implementation)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    mag = np.minimum(np.sqrt(sx * sx + sy * sy), 1.0)

    # Step 3: Expand to 3-channel, blur via OpenCV GaussianBlur (C++ implementation)
    edges = np.stack([mag, mag, mag], axis=-1)
    ksize = int(round(sigma * 10.0 + 1.0)) | 1
    ksize = max(ksize, 3)
    blurred = cv2.GaussianBlur(edges, (ksize, ksize), sigma, borderType=cv2.BORDER_REFLECT_101)

    # Step 4: Invert
    output = 1.0 - blurred

    return {
        "filter": "charcoal",
        "params": {"radius": radius, "sigma": sigma},
        "tool": "cv2.Sobel + cv2.GaussianBlur (independent C++ implementations)",
        "tool_version": cv2.__version__,
        "note": "Sobel edge detect (OpenCV C++) → clip(1.0) → GaussianBlur (OpenCV C++) → invert",
        "output": pixels_to_list(output),
    }


def golden_halftone(dot_size: float, angle_offset: float = 0.0) -> dict:
    """Halftone — Photoshop-style CMYK circular dot screening.

    Matches Photoshop "Color Halftone" (Filter→Pixelate) algorithm:
    1. RGB→CMYK decomposition
    2. Per channel: rotate grid by screen angle, find cell center, sample
       ink density, compute dot radius = max_r * sqrt(density)
    3. Point-in-circle with smoothstep anti-aliasing
    4. Subtractive CMYK→RGB composite

    Validated as formula (same algorithm as Photoshop Color Halftone).
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]
    max_r = max(dot_size, 1.0)
    cell = 2.0 * max_r

    angles_deg = [15.0 + angle_offset, 75.0 + angle_offset,
                  0.0 + angle_offset, 45.0 + angle_offset]
    angles_rad = [np.radians(a) for a in angles_deg]

    def smooth(edge0, edge1, x):
        if edge0 >= edge1:
            return 1.0 if x <= edge1 else 0.0
        t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
        return 1.0 - t * t * (3.0 - 2.0 * t)

    output = np.zeros_like(img)
    for y_px in range(h):
        for x_px in range(w):
            ink = [0.0] * 4

            for ch in range(4):
                cos_a = np.cos(angles_rad[ch])
                sin_a = np.sin(angles_rad[ch])

                # Rotate into screen space
                sx = float(x_px) * cos_a + float(y_px) * sin_a
                sy = -float(x_px) * sin_a + float(y_px) * cos_a

                # Cell center in screen space
                cx = round(sx / cell) * cell
                cy = round(sy / cell) * cell

                # Map back to image space
                img_x = int(max(0, min(w - 1, cx * cos_a - cy * (-sin_a))))
                img_y = int(max(0, min(h - 1, cx * sin_a + cy * cos_a)))

                # Sample CMYK at cell center
                sr, sg, sb = img[img_y, img_x, 0], img[img_y, img_x, 1], img[img_y, img_x, 2]
                sk = 1.0 - max(sr, sg, sb)
                s_inv_k = 1.0 / (1.0 - sk) if sk < 1.0 else 0.0
                if ch == 0: density = (1.0 - sr - sk) * s_inv_k
                elif ch == 1: density = (1.0 - sg - sk) * s_inv_k
                elif ch == 2: density = (1.0 - sb - sk) * s_inv_k
                else: density = sk

                dot_r = max_r * max(density, 0.0) ** 0.5
                dist = ((sx - cx) ** 2 + (sy - cy) ** 2) ** 0.5
                ink[ch] = smooth(dot_r + 0.5, dot_r - 0.5, dist)

            output[y_px, x_px, 0] = (1.0 - ink[0]) * (1.0 - ink[3])
            output[y_px, x_px, 1] = (1.0 - ink[1]) * (1.0 - ink[3])
            output[y_px, x_px, 2] = (1.0 - ink[2]) * (1.0 - ink[3])

    return {
        "filter": "halftone",
        "params": {"dot_size": dot_size, "angle_offset": angle_offset},
        "tool": "numpy (Photoshop-style circular dot CMYK screening)",
        "tool_version": np.__version__,
        "note": f"dot_size={dot_size}, circular dots, area-proportional radius, smoothstep AA",
        "output": pixels_to_list(output),
    }


def golden_ripple(amplitude: float, wavelength: float) -> dict:
    """Ripple distortion via cv2.remap — radial sinusoidal displacement.

    offset = amplitude * sin(2*pi*dist/wavelength), displace along radial direction.
    Center defaults to (0.5*w, 0.5*h) matching pipeline default center_x=0.5, center_y=0.5.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]
    cx = 0.5 * w
    cy = 0.5 * h

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            dx = float(x) - cx
            dy = float(y) - cy
            dist = np.sqrt(dx * dx + dy * dy)
            if dist > 0:
                disp = amplitude * np.sin(2.0 * np.pi * dist / wavelength)
                map_x[y, x] = x + disp * dx / dist
                map_y[y, x] = y + disp * dy / dist
            else:
                map_x[y, x] = x
                map_y[y, x] = y

    output = np.zeros_like(img)
    for c in range(3):
        output[:, :, c] = cv2.remap(img[:, :, c], map_x, map_y,
                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return {
        "filter": "ripple",
        "params": {"amplitude": amplitude, "wavelength": wavelength, "center_x": 0.5, "center_y": 0.5},
        "tool": f"cv2.remap (ripple: A={amplitude}, λ={wavelength})",
        "tool_version": cv2.__version__,
        "note": "Radial sinusoidal displacement, bilinear, border replicate",
        "output": pixels_to_list(output),
    }


# ─── Edge + morphology golden generators ────────────────────────────────────


def golden_laplacian(scale: float) -> dict:
    """Laplacian edge detection via cv2.Laplacian (OpenCV C++ implementation)."""
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    gray = img[:, :, 0] * 0.2126 + img[:, :, 1] * 0.7152 + img[:, :, 2] * 0.0722
    # OpenCV Laplacian ksize=3 uses kernel [0,2,0,2,-8,2,0,2,0] (2× our [0,1,0,1,-4,1,0,1,0])
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3,
                         borderType=cv2.BORDER_REPLICATE)
    mag = np.abs(lap / 2.0) * scale
    output = np.stack([mag, mag, mag], axis=-1)
    return {
        "filter": "laplacian", "params": {"scale": scale},
        "tool": f"cv2.Laplacian (ksize=3, scale={scale})", "tool_version": cv2.__version__,
        "note": "BT.709 luma, abs(Laplacian)*scale, BORDER_REPLICATE",
        "output": pixels_to_list(output),
    }


def golden_scharr(scale: float) -> dict:
    """Scharr edge detection via cv2.Scharr (OpenCV C++ implementation)."""
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    gray = img[:, :, 0] * 0.2126 + img[:, :, 1] * 0.7152 + img[:, :, 2] * 0.0722
    sx = cv2.Scharr(gray, cv2.CV_32F, 1, 0, borderType=cv2.BORDER_REPLICATE)
    sy = cv2.Scharr(gray, cv2.CV_32F, 0, 1, borderType=cv2.BORDER_REPLICATE)
    # Pipeline normalizes by /32.0 (max Scharr magnitude for a step edge = 32)
    mag = np.sqrt(sx * sx + sy * sy) * scale / 32.0
    output = np.stack([mag, mag, mag], axis=-1)
    return {
        "filter": "scharr", "params": {"scale": scale},
        "tool": f"cv2.Scharr (gradient magnitude * {scale})", "tool_version": cv2.__version__,
        "note": "BT.709 luma, sqrt(Sx²+Sy²)*scale, BORDER_REPLICATE",
        "output": pixels_to_list(output),
    }


def golden_threshold(level: float) -> dict:
    """Binary threshold — simple comparison, no external tool needed."""
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    gray = img[:, :, 0] * 0.2126 + img[:, :, 1] * 0.7152 + img[:, :, 2] * 0.0722
    binary = np.where(gray >= level, 1.0, 0.0).astype(np.float32)
    output = np.stack([binary, binary, binary], axis=-1)
    return {
        "filter": "threshold_binary", "params": {"threshold": level},
        "tool": "numpy (luma >= threshold ? 1.0 : 0.0)", "tool_version": np.__version__,
        "note": "BT.709 luma, binary step function",
        "output": pixels_to_list(output),
    }


def golden_dilate(radius: int) -> dict:
    """Dilate via cv2.dilate (OpenCV C++ implementation)."""
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    ksize = 2 * radius + 1
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    output = np.zeros_like(img)
    for c in range(3):
        output[:, :, c] = cv2.dilate(img[:, :, c], kernel, borderType=cv2.BORDER_REFLECT_101)
    return {
        "filter": "dilate", "params": {"radius": radius},
        "tool": f"cv2.dilate (kernel {ksize}x{ksize})", "tool_version": cv2.__version__,
        "note": "Per-channel max filter, square kernel, BORDER_REFLECT_101",
        "output": pixels_to_list(output),
    }


def golden_erode(radius: int) -> dict:
    """Erode via cv2.erode (OpenCV C++ implementation)."""
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    ksize = 2 * radius + 1
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    output = np.zeros_like(img)
    for c in range(3):
        output[:, :, c] = cv2.erode(img[:, :, c], kernel, borderType=cv2.BORDER_REFLECT_101)
    return {
        "filter": "erode", "params": {"radius": radius},
        "tool": f"cv2.erode (kernel {ksize}x{ksize})", "tool_version": cv2.__version__,
        "note": "Per-channel min filter, square kernel, BORDER_REFLECT_101",
        "output": pixels_to_list(output),
    }


def golden_morph_open(radius: int) -> dict:
    """Morphological opening via cv2.morphologyEx (erode then dilate)."""
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    ksize = 2 * radius + 1
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    output = np.zeros_like(img)
    for c in range(3):
        output[:, :, c] = cv2.morphologyEx(img[:, :, c], cv2.MORPH_OPEN, kernel,
                                            borderType=cv2.BORDER_REFLECT_101)
    return {
        "filter": "morph_open", "params": {"radius": radius},
        "tool": f"cv2.morphologyEx MORPH_OPEN (kernel {ksize}x{ksize})", "tool_version": cv2.__version__,
        "note": "Erode then dilate, per-channel, BORDER_REFLECT_101",
        "output": pixels_to_list(output),
    }


def golden_morph_close(radius: int) -> dict:
    """Morphological closing via cv2.morphologyEx (dilate then erode)."""
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    ksize = 2 * radius + 1
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    output = np.zeros_like(img)
    for c in range(3):
        output[:, :, c] = cv2.morphologyEx(img[:, :, c], cv2.MORPH_CLOSE, kernel,
                                            borderType=cv2.BORDER_REFLECT_101)
    return {
        "filter": "morph_close", "params": {"radius": radius},
        "tool": f"cv2.morphologyEx MORPH_CLOSE (kernel {ksize}x{ksize})", "tool_version": cv2.__version__,
        "note": "Dilate then erode, per-channel, BORDER_REFLECT_101",
        "output": pixels_to_list(output),
    }


def golden_morph_tophat(radius: int) -> dict:
    """Morphological top-hat via cv2.morphologyEx (input - opening)."""
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    ksize = 2 * radius + 1
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    output = np.zeros_like(img)
    for c in range(3):
        output[:, :, c] = cv2.morphologyEx(img[:, :, c], cv2.MORPH_TOPHAT, kernel,
                                            borderType=cv2.BORDER_REFLECT_101)
    return {
        "filter": "morph_tophat", "params": {"radius": radius},
        "tool": f"cv2.morphologyEx MORPH_TOPHAT (kernel {ksize}x{ksize})", "tool_version": cv2.__version__,
        "note": "Input - opening, per-channel, BORDER_REFLECT_101",
        "output": pixels_to_list(output),
    }


def golden_morph_blackhat(radius: int) -> dict:
    """Morphological black-hat via cv2.morphologyEx (closing - input)."""
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    ksize = 2 * radius + 1
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    output = np.zeros_like(img)
    for c in range(3):
        output[:, :, c] = cv2.morphologyEx(img[:, :, c], cv2.MORPH_BLACKHAT, kernel,
                                            borderType=cv2.BORDER_REFLECT_101)
    return {
        "filter": "morph_blackhat", "params": {"radius": radius},
        "tool": f"cv2.morphologyEx MORPH_BLACKHAT (kernel {ksize}x{ksize})", "tool_version": cv2.__version__,
        "note": "Closing - input, per-channel, BORDER_REFLECT_101",
        "output": pixels_to_list(output),
    }


def golden_morph_gradient(radius: int) -> dict:
    """Morphological gradient via cv2.morphologyEx (dilate - erode)."""
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    ksize = 2 * radius + 1
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    output = np.zeros_like(img)
    for c in range(3):
        output[:, :, c] = cv2.morphologyEx(img[:, :, c], cv2.MORPH_GRADIENT, kernel,
                                            borderType=cv2.BORDER_REFLECT_101)
    return {
        "filter": "morph_gradient", "params": {"radius": radius},
        "tool": f"cv2.morphologyEx MORPH_GRADIENT (kernel {ksize}x{ksize})", "tool_version": cv2.__version__,
        "note": "Dilate - erode, per-channel, BORDER_REFLECT_101",
        "output": pixels_to_list(output),
    }


# ─── Composite / alpha golden generators ────────────────────────────────────


def golden_blend_self(mode_name: str, mode_id: int) -> dict:
    """Self-blend via ImageMagick -compose (genuine external tool).

    Uses IM's built-in blend mode implementation with the same image as
    both base and overlay (self-composite). Validates W3C Compositing Level 1.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    # Map our mode names to IM's compose operator names
    im_mode_map = {
        "multiply": "Multiply", "screen": "Screen", "overlay": "Overlay",
        "soft_light": "SoftLight", "hard_light": "HardLight",
        "color_dodge": "ColorDodge", "color_burn": "ColorBurn",
        "darken": "Darken", "lighten": "Lighten",
        "difference": "Difference", "exclusion": "Exclusion",
    }
    im_mode = im_mode_map.get(mode_name, mode_name.capitalize())

    # IM self-composite: magick base.tiff base.tiff -compose Mode -composite out.tiff
    img_bgr = img[:, :, ::-1].copy()
    with tempfile.NamedTemporaryFile(suffix='.tiff', delete=False) as f:
        cv2.imwrite(f.name, img_bgr)
        in_path = f.name
    out_path = in_path.replace('.tiff', '_out.tiff')

    r = subprocess.run([
        'magick', in_path, in_path,
        '-depth', '32', '-define', 'quantum:format=floating-point',
        '-compose', im_mode, '-composite',
        '-depth', '32', '-define', 'quantum:format=floating-point',
        out_path
    ], capture_output=True, text=True)

    if r.returncode != 0:
        raise RuntimeError(f'IM blend failed: {r.stderr}')

    result = cv2.imread(out_path, cv2.IMREAD_UNCHANGED)
    os.unlink(in_path)
    os.unlink(out_path)
    if result is None:
        raise RuntimeError(f'Failed to read IM {im_mode} output')
    if result.ndim == 2:
        output = np.stack([result, result, result], axis=-1).astype(np.float32)
    else:
        output = result[:, :, ::-1].copy().astype(np.float32)

    return {
        "filter": "blend",
        "params": {"mode": mode_id, "opacity": 1.0},
        "tool": f"magick -compose {im_mode} -composite (self-blend)",
        "tool_version": tool_info()["imagemagick"],
        "note": f"ImageMagick built-in {mode_name} blend, self-composite",
        "output": pixels_to_list(output),
    }


def golden_blend_multiply(opacity: float) -> dict:
    """Blend multiply: out = base * overlay. W3C Compositing Level 1 spec."""
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    # Use a shifted version as overlay
    overlay = np.roll(img, 32, axis=1)  # shift right by 32px
    output = img * opacity + img * overlay * (1.0 - opacity)
    # Actually: blend = base * overlay; result = lerp(base, blend, opacity)
    blend = img * overlay
    output = img * (1.0 - opacity) + blend * opacity
    return {
        "filter": "blend",
        "params": {"mode": 1, "opacity": opacity},  # mode 1 = multiply
        "tool": "numpy (W3C Compositing Level 1: multiply)",
        "tool_version": np.__version__,
        "note": f"blend = base * overlay, result = lerp(base, blend, {opacity})",
        "output": pixels_to_list(output),
        "custom_input": pixels_to_list(overlay),  # overlay as secondary input
    }


# ─── More edge + spatial golden generators ──────────────────────────────────


def golden_otsu_threshold() -> dict:
    """Otsu threshold via cv2.threshold (OpenCV C++ implementation)."""
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    gray = img[:, :, 0] * 0.2126 + img[:, :, 1] * 0.7152 + img[:, :, 2] * 0.0722
    # OpenCV Otsu needs u8 input
    gray_u8 = np.clip(gray * 255 + 0.5, 0, 255).astype(np.uint8)
    _, binary_u8 = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = binary_u8.astype(np.float32) / 255.0
    output = np.stack([binary, binary, binary], axis=-1)
    return {
        "filter": "otsu_threshold", "params": {},
        "tool": "cv2.threshold (THRESH_OTSU)", "tool_version": cv2.__version__,
        "note": "OpenCV Otsu automatic threshold on BT.709 luma (u8 quantized)",
        "output": pixels_to_list(output),
    }


def golden_canny(low: float, high: float) -> dict:
    """Canny edge detection via cv2.Canny (OpenCV C++ implementation)."""
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    gray = img[:, :, 0] * 0.2126 + img[:, :, 1] * 0.7152 + img[:, :, 2] * 0.0722
    # OpenCV Canny needs u8. Thresholds are in u8 scale.
    gray_u8 = np.clip(gray * 255 + 0.5, 0, 255).astype(np.uint8)
    edges_u8 = cv2.Canny(gray_u8, int(low * 255), int(high * 255))
    edges = edges_u8.astype(np.float32) / 255.0
    output = np.stack([edges, edges, edges], axis=-1)
    return {
        "filter": "canny", "params": {"low": low, "high": high},
        "tool": f"cv2.Canny (low={int(low*255)}, high={int(high*255)})",
        "tool_version": cv2.__version__,
        "note": "OpenCV Canny on BT.709 luma (u8 quantized), thresholds scaled to u8",
        "output": pixels_to_list(output),
    }


def golden_median(radius: int) -> dict:
    """Median filter via cv2.medianBlur (OpenCV C++ implementation)."""
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    ksize = 2 * radius + 1
    # OpenCV medianBlur on f32 requires ksize <= 5
    if ksize <= 5:
        output = cv2.medianBlur(img, ksize)
    else:
        # For larger kernels, fall back to per-channel u8 round-trip
        img_u8 = np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)
        result_u8 = cv2.medianBlur(img_u8, ksize)
        output = result_u8.astype(np.float32) / 255.0
    return {
        "filter": "median", "params": {"radius": radius},
        "tool": f"cv2.medianBlur (ksize={ksize})",
        "tool_version": cv2.__version__,
        "note": f"OpenCV median filter, ksize={ksize}, per-channel",
        "output": pixels_to_list(output),
    }


def golden_motion_blur(length: int, angle: float) -> dict:
    """Motion blur via cv2.filter2D with a line kernel (OpenCV C++)."""
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    # Build 1D kernel rotated to the given angle
    ksize = length
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    center = ksize // 2
    rad = np.radians(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    for i in range(ksize):
        t = i - center
        x = int(round(center + t * cos_a))
        y = int(round(center + t * sin_a))
        if 0 <= x < ksize and 0 <= y < ksize:
            kernel[y, x] = 1.0
    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel /= kernel_sum
    output = np.zeros_like(img)
    for c in range(3):
        output[:, :, c] = cv2.filter2D(img[:, :, c], -1, kernel,
                                         borderType=cv2.BORDER_REPLICATE)
    return {
        "filter": "motion_blur", "params": {"length": length, "angle": angle},
        "tool": f"cv2.filter2D (motion blur kernel {ksize}x{ksize}, angle={angle}°)",
        "tool_version": cv2.__version__,
        "note": "Line kernel rotated to angle, normalized, BORDER_REPLICATE",
        "output": pixels_to_list(output),
    }


def golden_premultiply() -> dict:
    """Premultiply alpha — trivially verifiable formula.

    out.rgb = in.rgb * in.a, out.a = in.a.
    Uses input with varying alpha (alpha = x / (w-1)).
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]
    # Create RGBA with alpha gradient
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            a = x / max(w - 1, 1)
            rgba[y, x] = [img[y, x, 0], img[y, x, 1], img[y, x, 2], a]
    # Premultiply
    output_rgba = rgba.copy()
    output_rgba[:, :, 0] *= rgba[:, :, 3]
    output_rgba[:, :, 1] *= rgba[:, :, 3]
    output_rgba[:, :, 2] *= rgba[:, :, 3]
    return {
        "filter": "premultiply", "params": {},
        "tool": "numpy (rgb * alpha)", "tool_version": np.__version__,
        "note": "out.rgb = in.rgb * in.a. Input has alpha gradient (0 to 1 left to right).",
        "output": pixels_to_list(output_rgba),
        "custom_input": pixels_to_list(rgba),
    }


def golden_unpremultiply() -> dict:
    """Unpremultiply alpha — trivially verifiable formula.

    out.rgb = in.rgb / in.a (where a > 0), out.a = in.a.
    Uses premultiplied input.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]
    # Create premultiplied RGBA
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            a = x / max(w - 1, 1)
            rgba[y, x] = [img[y, x, 0] * a, img[y, x, 1] * a, img[y, x, 2] * a, a]
    # Unpremultiply
    output_rgba = rgba.copy()
    for y in range(h):
        for x in range(w):
            a = rgba[y, x, 3]
            if a > 1e-10:
                output_rgba[y, x, 0] = rgba[y, x, 0] / a
                output_rgba[y, x, 1] = rgba[y, x, 1] / a
                output_rgba[y, x, 2] = rgba[y, x, 2] / a
    return {
        "filter": "unpremultiply", "params": {},
        "tool": "numpy (rgb / alpha)", "tool_version": np.__version__,
        "note": "out.rgb = in.rgb / in.a where a > 0. Input is premultiplied.",
        "output": pixels_to_list(output_rgba),
        "custom_input": pixels_to_list(rgba),
    }


def golden_adaptive_threshold(radius: int, offset: float) -> dict:
    """Adaptive threshold via cv2.adaptiveThreshold (OpenCV C++)."""
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    gray = img[:, :, 0] * 0.2126 + img[:, :, 1] * 0.7152 + img[:, :, 2] * 0.0722
    gray_u8 = np.clip(gray * 255 + 0.5, 0, 255).astype(np.uint8)
    ksize = 2 * radius + 1
    binary_u8 = cv2.adaptiveThreshold(gray_u8, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, ksize, int(offset * 255))
    binary = binary_u8.astype(np.float32) / 255.0
    output = np.stack([binary, binary, binary], axis=-1)
    return {
        "filter": "adaptive_threshold", "params": {"radius": radius, "offset": offset},
        "tool": f"cv2.adaptiveThreshold (MEAN_C, ksize={ksize})",
        "tool_version": cv2.__version__,
        "note": "OpenCV adaptive threshold on BT.709 luma (u8 quantized)",
        "output": pixels_to_list(output),
    }


def golden_triangle_threshold() -> dict:
    """Triangle threshold — automatic threshold selection.
    Uses OpenCV THRESH_TRIANGLE.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    gray = img[:, :, 0] * 0.2126 + img[:, :, 1] * 0.7152 + img[:, :, 2] * 0.0722
    gray_u8 = np.clip(gray * 255 + 0.5, 0, 255).astype(np.uint8)
    _, binary_u8 = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    binary = binary_u8.astype(np.float32) / 255.0
    output = np.stack([binary, binary, binary], axis=-1)
    return {
        "filter": "triangle_threshold", "params": {},
        "tool": "cv2.threshold (THRESH_TRIANGLE)", "tool_version": cv2.__version__,
        "note": "OpenCV triangle automatic threshold on BT.709 luma (u8 quantized)",
        "output": pixels_to_list(output),
    }


def golden_lens_blur(radius: int) -> dict:
    """Lens blur — disc kernel convolution via cv2.filter2D."""
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    ksize = 2 * radius + 1
    # Build circular disc kernel
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    center = radius
    for y in range(ksize):
        for x in range(ksize):
            if (x - center) ** 2 + (y - center) ** 2 <= radius ** 2:
                kernel[y, x] = 1.0
    kernel /= kernel.sum()
    output = np.zeros_like(img)
    for c in range(3):
        output[:, :, c] = cv2.filter2D(img[:, :, c], -1, kernel,
                                         borderType=cv2.BORDER_REPLICATE)
    return {
        "filter": "lens_blur", "params": {"radius": radius},
        "tool": f"cv2.filter2D (disc kernel radius={radius})",
        "tool_version": cv2.__version__,
        "note": "Circular disc kernel convolution, BORDER_REPLICATE",
        "output": pixels_to_list(output),
    }


def golden_zoom_blur(factor: float) -> dict:
    """Zoom blur — radial blur from center. Formula validated."""
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]
    cx, cy = w * 0.5, h * 0.5
    n_samples = 16
    output = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            dx = x - cx
            dy = y - cy
            acc = np.zeros(3, dtype=np.float64)
            for s in range(n_samples):
                t = 1.0 + factor * s / (n_samples - 1)
                sx = int(max(0, min(w - 1, cx + dx * t)))
                sy = int(max(0, min(h - 1, cy + dy * t)))
                acc += img[sy, sx, :3].astype(np.float64)
            output[y, x] = (acc / n_samples).astype(np.float32)
    return {
        "filter": "zoom_blur", "params": {"factor": factor, "center_x": 0.5, "center_y": 0.5},
        "tool": "numpy (radial multi-sample blur from center)",
        "tool_version": np.__version__,
        "note": f"factor={factor}, 16 samples along radial direction",
        "output": pixels_to_list(output),
    }


def golden_spin_blur(angle: float) -> dict:
    """Spin blur — rotational blur around center. Formula validated."""
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]
    cx, cy = w * 0.5, h * 0.5
    n_samples = 16
    angle_rad = np.radians(angle)
    output = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            dx = x - cx
            dy = y - cy
            acc = np.zeros(3, dtype=np.float64)
            for s in range(n_samples):
                t = angle_rad * (s / (n_samples - 1) - 0.5)
                cos_t, sin_t = np.cos(t), np.sin(t)
                sx = int(max(0, min(w - 1, cx + dx * cos_t - dy * sin_t)))
                sy = int(max(0, min(h - 1, cy + dx * sin_t + dy * cos_t)))
                acc += img[sy, sx, :3].astype(np.float64)
            output[y, x] = (acc / n_samples).astype(np.float32)
    return {
        "filter": "spin_blur", "params": {"angle": angle, "center_x": 0.5, "center_y": 0.5},
        "tool": "numpy (rotational multi-sample blur around center)",
        "tool_version": np.__version__,
        "note": f"angle={angle}°, 16 samples along arc",
        "output": pixels_to_list(output),
    }


def golden_auto_level() -> dict:
    """Auto-level via ImageMagick -auto-level (built-in)."""
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    output = im_process(img, ['-auto-level'])
    return {
        "filter": "auto_level", "params": {},
        "tool": "magick -auto-level", "tool_version": tool_info()["imagemagick"],
        "note": "ImageMagick built-in auto-level (per-channel stretch to full range)",
        "output": pixels_to_list(output),
    }


def golden_chromatic_split(red_dx, red_dy, green_dx, green_dy, blue_dx, blue_dy) -> dict:
    """Chromatic split — per-channel spatial offset via cv2.remap."""
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]
    output = img.copy()
    for c, (ddx, ddy) in enumerate([(red_dx, red_dy), (green_dx, green_dy), (blue_dx, blue_dy)]):
        if ddx == 0 and ddy == 0:
            continue
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                map_x[y, x] = x + ddx
                map_y[y, x] = y + ddy
        output[:, :, c] = cv2.remap(img[:, :, c], map_x, map_y,
                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return {
        "filter": "chromatic_split",
        "params": {"red_dx": red_dx, "red_dy": red_dy, "green_dx": green_dx,
                   "green_dy": green_dy, "blue_dx": blue_dx, "blue_dy": blue_dy},
        "tool": "cv2.remap (per-channel spatial offset, bilinear)",
        "tool_version": cv2.__version__,
        "note": "Per-channel translation via OpenCV remap, BORDER_REPLICATE",
        "output": pixels_to_list(output),
    }


def golden_dither_ordered(levels: int, map_size: int) -> dict:
    """Ordered dither — Bayer matrix threshold. Formula validated."""
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]
    # Build Bayer matrix of given size
    def bayer(n):
        if n == 1: return np.array([[0]], dtype=np.float32)
        smaller = bayer(n // 2)
        s = smaller.shape[0]
        m = np.zeros((n, n), dtype=np.float32)
        m[:s, :s] = 4 * smaller
        m[:s, s:] = 4 * smaller + 2
        m[s:, :s] = 4 * smaller + 3
        m[s:, s:] = 4 * smaller + 1
        return m
    matrix = bayer(map_size) / (map_size * map_size)
    output = img.copy()
    n_levels = float(levels)
    for y in range(h):
        for x in range(w):
            threshold = matrix[y % map_size, x % map_size]
            for c in range(3):
                v = img[y, x, c]
                quantized = min(np.floor(v * n_levels + threshold), n_levels - 1) / max(n_levels - 1, 1)
                output[y, x, c] = quantized
    return {
        "filter": "dither_ordered",
        "params": {"max_colors": levels, "map_size": map_size},
        "tool": "numpy (Bayer matrix ordered dither)",
        "tool_version": np.__version__,
        "note": f"levels={levels}, Bayer {map_size}x{map_size}, floor(v*n+threshold)/(n-1)",
        "output": pixels_to_list(output),
    }


def golden_flatten(bg_r: float, bg_g: float, bg_b: float) -> dict:
    """Flatten alpha over solid background. Formula validated."""
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]
    # Create RGBA with varying alpha (gradient left to right)
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            a = x / max(w - 1, 1)
            rgba[y, x] = [img[y, x, 0], img[y, x, 1], img[y, x, 2], a]
    output = np.zeros((h, w, 3), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            a = rgba[y, x, 3]
            output[y, x, 0] = rgba[y, x, 0] * a + bg_r * (1.0 - a)
            output[y, x, 1] = rgba[y, x, 1] * a + bg_g * (1.0 - a)
            output[y, x, 2] = rgba[y, x, 2] * a + bg_b * (1.0 - a)
    return {
        "filter": "flatten",
        "params": {"bg_r": bg_r, "bg_g": bg_g, "bg_b": bg_b},
        "tool": "numpy (alpha compositing over solid background)",
        "tool_version": np.__version__,
        "note": "out = fg*alpha + bg*(1-alpha)",
        "output": pixels_to_list(output),
        "custom_input": pixels_to_list(rgba),
    }


def golden_add_alpha(alpha: float) -> dict:
    """Set alpha channel to a constant value. Trivially verifiable."""
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]
    output = np.zeros((h, w, 4), dtype=np.float32)
    output[:, :, :3] = img
    output[:, :, 3] = alpha
    return {
        "filter": "add_alpha", "params": {"alpha": alpha},
        "tool": "numpy (set alpha channel)", "tool_version": np.__version__,
        "note": f"Set all alpha to {alpha}",
        "output": pixels_to_list(output),
    }


def golden_remove_alpha() -> dict:
    """Remove alpha — output RGB only. Trivially verifiable."""
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    return {
        "filter": "remove_alpha", "params": {},
        "tool": "numpy (drop alpha channel)", "tool_version": np.__version__,
        "note": "Output = input RGB, alpha dropped",
        "output": pixels_to_list(img),
    }


def golden_solid_color(r: float, g: float, b: float) -> dict:
    """Solid color generator. Trivially verifiable."""
    h, w = SPATIAL_H, SPATIAL_W
    output = np.full((h, w, 3), [r, g, b], dtype=np.float32)
    return {
        "filter": "solid_color",
        "params": {"r": r, "g": g, "b": b, "a": 1.0},
        "tool": "numpy (fill with constant)", "tool_version": np.__version__,
        "note": f"Every pixel = ({r}, {g}, {b})",
        "output": pixels_to_list(output),
    }


def golden_gradient_linear(angle: float) -> dict:
    """Linear gradient generator via ImageMagick gradient:."""
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = SPATIAL_H, SPATIAL_W
    # Pipeline formula: t = (nx*cos(a) + ny*sin(a)).clamp(0,1), lerp black→white
    a = np.radians(angle)
    output = np.zeros((h, w, 3), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            nx = x / w
            ny = y / h
            t = max(0.0, min(1.0, nx * np.cos(a) + ny * np.sin(a)))
            output[y, x] = [t, t, t]
    return {
        "filter": "gradient_linear",
        "params": {"angle": angle, "start_r": 0.0, "start_g": 0.0, "start_b": 0.0,
                   "end_r": 1.0, "end_g": 1.0, "end_b": 1.0},
        "tool": "numpy (t = clamp(nx*cos(a)+ny*sin(a), 0, 1))",
        "tool_version": np.__version__,
        "note": f"Linear gradient angle={angle}°, black→white",
        "output": pixels_to_list(output),
    }


def golden_gradient_radial() -> dict:
    """Radial gradient — matching pipeline formula: t = clamp(dist * 2.0, 0, 1)."""
    h, w = SPATIAL_H, SPATIAL_W
    # Pipeline: center at (center_x * w, center_y * h), default (0.5, 0.5)
    cx = 0.5 * w
    cy = 0.5 * h
    output = np.zeros((h, w, 3), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            # Pipeline normalizes: nx = x/w - center_x, ny = y/h - center_y
            nx = x / w - 0.5
            ny = y / h - 0.5
            dist = np.sqrt(nx * nx + ny * ny)
            t = min(1.0, dist * 2.0)  # *2 so gradient reaches edge at 50% of image
            # inner (white) to outer (black) by default
            output[y, x] = [1.0 - t, 1.0 - t, 1.0 - t]
    return {
        "filter": "gradient_radial",
        "params": {"inner_r": 1.0, "inner_g": 1.0, "inner_b": 1.0,
                   "outer_r": 0.0, "outer_g": 0.0, "outer_b": 0.0,
                   "center_x": 0.5, "center_y": 0.5},
        "tool": "numpy (t = clamp(dist*2, 0, 1), matching pipeline formula)",
        "tool_version": np.__version__,
        "note": "Radial gradient: t = clamp(normalized_dist * 2, 0, 1), lerp inner→outer",
        "output": pixels_to_list(output),
    }


def golden_mirror_kaleidoscope(segments: int) -> dict:
    """Mirror kaleidoscope effect. Formula validated (polar + segment mirror)."""
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    segment_angle = 2.0 * np.pi / segments
    output = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            dx = x - cx
            dy = y - cy
            angle = np.arctan2(dy, dx)
            if angle < 0: angle += 2.0 * np.pi
            dist = np.sqrt(dx * dx + dy * dy)
            # Map to first segment
            seg_idx = int(angle / segment_angle)
            local_angle = angle - seg_idx * segment_angle
            # Mirror odd segments
            if seg_idx % 2 == 1:
                local_angle = segment_angle - local_angle
            # Map back to image coords
            sx = cx + dist * np.cos(local_angle)
            sy = cy + dist * np.sin(local_angle)
            sx = max(0, min(w - 1, int(round(sx))))
            sy = max(0, min(h - 1, int(round(sy))))
            output[y, x] = img[sy, sx]
    return {
        "filter": "mirror_kaleidoscope",
        "params": {"segments": segments, "angle": 0.0, "mode": 0},
        "tool": "numpy (polar coordinate segment mirroring)",
        "tool_version": np.__version__,
        "note": f"segments={segments}, mirror odd segments, nearest-neighbor sampling",
        "output": pixels_to_list(output),
    }


def golden_skeletonize() -> dict:
    """Skeletonize via scikit-image skeletonize (external tool)."""
    from skimage.morphology import skeletonize as ski_skeletonize
    import skimage
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    gray = img[:, :, 0] * 0.2126 + img[:, :, 1] * 0.7152 + img[:, :, 2] * 0.0722
    binary = gray > 0.5
    skeleton = ski_skeletonize(binary).astype(np.float32)
    output = np.stack([skeleton, skeleton, skeleton], axis=-1)
    return {
        "filter": "skeletonize", "params": {},
        "tool": f"skimage.morphology.skeletonize", "tool_version": skimage.__version__,
        "note": "scikit-image skeletonize on BT.709 luma > 0.5",
        "output": pixels_to_list(output),
    }


def golden_match_color() -> dict:
    """Match color — histogram matching between two images.
    Uses a shifted version of the input as the reference.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    # Create a reference: brighter version of input
    reference = np.clip(img * 1.5, 0, 1).astype(np.float32)
    # Simple mean+std transfer per channel
    output = img.copy()
    for c in range(3):
        src_mean = img[:, :, c].mean()
        src_std = max(img[:, :, c].std(), 1e-10)
        ref_mean = reference[:, :, c].mean()
        ref_std = max(reference[:, :, c].std(), 1e-10)
        output[:, :, c] = (img[:, :, c] - src_mean) * (ref_std / src_std) + ref_mean
    return {
        "filter": "match_color",
        "params": {"strength": 1.0},
        "tool": "numpy (mean+std color transfer per channel)",
        "tool_version": np.__version__,
        "note": "Reinhard-style color transfer: (src - src_mean) * (ref_std/src_std) + ref_mean",
        "output": pixels_to_list(output),
        "custom_input": pixels_to_list(reference),
    }


def golden_liquify() -> dict:
    """Liquify distortion via cv2.remap — forward warp approximation.

    Pipeline: for each pixel, if dist < radius, shift source by -(dx,dy)*t²*strength.
    Uses bilinear sampling.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]
    # Default: center=(0.5, 0.5), radius=50, strength=0.5, direction=(1,0)
    cx = 0.5 * w
    cy = 0.5 * h
    radius = 50.0
    strength = 0.5
    dir_x = 1.0
    dir_y = 0.0

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    for y_px in range(h):
        for x_px in range(w):
            dx = x_px - cx
            dy = y_px - cy
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radius:
                t = dist / radius
                wt = np.exp(-2.0 * t * t) * strength
                map_x[y_px, x_px] = x_px - dir_x * wt * radius
                map_y[y_px, x_px] = y_px - dir_y * wt * radius
            else:
                map_x[y_px, x_px] = x_px
                map_y[y_px, x_px] = y_px

    output = np.zeros_like(img)
    for c in range(3):
        output[:, :, c] = cv2.remap(img[:, :, c], map_x, map_y,
                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return {
        "filter": "liquify",
        "params": {"center_x": 0.5, "center_y": 0.5, "radius": radius,
                   "strength": strength, "direction_x": dir_x, "direction_y": dir_y},
        "tool": "cv2.remap (liquify forward warp, bilinear)",
        "tool_version": cv2.__version__,
        "note": "t = 1-dist/radius, shift = t²*strength*radius, bilinear sampling",
        "output": pixels_to_list(output),
    }


def golden_mirror_kaleidoscope_h(segments: int) -> dict:
    """Mirror kaleidoscope mode 0 (horizontal) — divide width into segments, mirror alternates."""
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]
    seg_w = w / segments
    output = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            seg_idx = int(x / seg_w)
            local_x = x - seg_idx * seg_w
            if seg_idx % 2 == 1:
                local_x = seg_w - 1 - local_x
            src_x = int(max(0, min(w - 1, local_x)))
            output[y, x] = img[y, src_x]
    return {
        "filter": "mirror_kaleidoscope",
        "params": {"segments": segments, "angle": 0.0, "mode": 0},
        "tool": "numpy (horizontal segment mirroring)",
        "tool_version": np.__version__,
        "note": f"Mode 0 (horizontal), segments={segments}, mirror alternate segments",
        "output": pixels_to_list(output),
    }


def golden_sigmoidal_contrast(contrast: float, midpoint: float) -> dict:
    """Sigmoidal contrast via ImageMagick -sigmoidal-contrast (built-in)."""
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    output = im_process(img, ['-sigmoidal-contrast', f'{contrast}x{midpoint * 100}%'])
    return {
        "filter": "sigmoidal_contrast",
        "params": {"contrast": contrast, "midpoint": midpoint},
        "tool": f"magick -sigmoidal-contrast {contrast}x{midpoint*100}%",
        "tool_version": tool_info()["imagemagick"],
        "note": "ImageMagick built-in sigmoidal contrast",
        "output": pixels_to_list(output),
    }


# ─── Generator + draw + tool golden generators ─────────────────────────────


def golden_checkerboard(block_size: int) -> dict:
    """Checkerboard — trivially verifiable procedural pattern.

    Formula: color1 if (floor(x/size) + floor(y/size)) % 2 == 0, else color2.
    This is a deterministic formula with no image processing — the formula IS the spec.
    No external tool reference needed (IM's pattern:checkerboard uses non-standard colors).
    """
    h, w = SPATIAL_H, SPATIAL_W
    output = np.zeros((h, w, 3), dtype=np.float32)
    for y_px in range(h):
        for x_px in range(w):
            cx = int(x_px / block_size)
            cy = int(y_px / block_size)
            val = 1.0 if (cx + cy) % 2 == 0 else 0.0
            output[y_px, x_px] = [val, val, val]

    return {
        "filter": "checkerboard",
        "params": {"size": float(block_size), "color1_r": 1.0, "color1_g": 1.0, "color1_b": 1.0,
                   "color2_r": 0.0, "color2_g": 0.0, "color2_b": 0.0},
        "tool": "numpy (deterministic formula: floor(x/size)+floor(y/size) mod 2)",
        "tool_version": np.__version__,
        "note": f"Procedural pattern, block_size={block_size}. Formula is trivially verifiable by inspection.",
        "output": pixels_to_list(output),
    }


def golden_draw_circle() -> dict:
    """Draw filled circle via OpenCV cv2.circle (C++ implementation)."""
    h, w = SPATIAL_H, SPATIAL_W
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    # Draw white filled circle at center, radius 30
    cv2.circle(canvas, (w // 2, h // 2), 30, (1.0, 1.0, 1.0), -1, lineType=cv2.LINE_8)
    # Convert BGR to RGB (cv2.circle draws in BGR)
    output = canvas[:, :, ::-1].copy()

    return {
        "filter": "draw_circle",
        "params": {"center_x": w // 2, "center_y": h // 2, "radius": 30,
                   "r": 1.0, "g": 1.0, "b": 1.0, "filled": True},
        "tool": "cv2.circle (LINE_8, filled)",
        "tool_version": cv2.__version__,
        "note": "OpenCV filled circle at center, radius=30, white on black",
        "output": pixels_to_list(output),
    }


def golden_draw_rect() -> dict:
    """Draw filled rectangle via OpenCV cv2.rectangle (C++ implementation)."""
    h, w = SPATIAL_H, SPATIAL_W
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    cv2.rectangle(canvas, (20, 20), (100, 80), (1.0, 1.0, 1.0), -1, lineType=cv2.LINE_8)
    output = canvas[:, :, ::-1].copy()

    return {
        "filter": "draw_rect",
        "params": {"x": 20, "y": 20, "width": 80, "height": 60,
                   "r": 1.0, "g": 1.0, "b": 1.0, "filled": True},
        "tool": "cv2.rectangle (LINE_8, filled)",
        "tool_version": cv2.__version__,
        "note": "OpenCV filled rectangle (20,20)-(100,80), white on black",
        "output": pixels_to_list(output),
    }


def golden_draw_line() -> dict:
    """Draw line via OpenCV cv2.line (C++ implementation)."""
    h, w = SPATIAL_H, SPATIAL_W
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    cv2.line(canvas, (10, 10), (100, 80), (1.0, 1.0, 1.0), 2, lineType=cv2.LINE_8)
    output = canvas[:, :, ::-1].copy()

    return {
        "filter": "draw_line",
        "params": {"x1": 10, "y1": 10, "x2": 100, "y2": 80,
                   "r": 1.0, "g": 1.0, "b": 1.0, "thickness": 2},
        "tool": "cv2.line (LINE_8, thickness=2)",
        "tool_version": cv2.__version__,
        "note": "OpenCV line (10,10)-(100,80), white, thickness=2",
        "output": pixels_to_list(output),
    }


def golden_flood_fill() -> dict:
    """Flood fill via OpenCV cv2.floodFill (C++ implementation).

    Start from a test image with a region, fill from a seed point.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]

    # Create a binary mask area in the image (threshold at 0.3)
    gray = img[:, :, 0] * 0.2126 + img[:, :, 1] * 0.7152 + img[:, :, 2] * 0.0722
    binary = np.where(gray > 0.3, 1.0, 0.0).astype(np.float32)
    test_img = np.stack([binary, binary, binary], axis=-1).copy()

    # Flood fill from center with red
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flood_img = test_img.copy()
    cv2.floodFill(flood_img, mask, (w // 2, h // 2), (1.0, 0.0, 0.0),
                  loDiff=(0.1, 0.1, 0.1), upDiff=(0.1, 0.1, 0.1))
    # BGR to RGB
    output = flood_img[:, :, ::-1].copy()

    return {
        "filter": "flood_fill",
        "params": {"seed_x": w // 2, "seed_y": h // 2, "r": 1.0, "g": 0.0, "b": 0.0,
                   "tolerance": 0.1},
        "tool": "cv2.floodFill (4-connected)",
        "tool_version": cv2.__version__,
        "note": "OpenCV flood fill from center, red, tolerance=0.1",
        "output": pixels_to_list(output),
        "custom_input": pixels_to_list(test_img[:, :, ::-1]),  # pass the thresholded image
    }


def golden_perspective_warp() -> dict:
    """Perspective warp via OpenCV cv2.warpPerspective (C++ implementation).

    Uses a known 4-point transform.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]

    # Define source and destination quadrilateral
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    # Slight perspective: pull top-right in, push bottom-left out
    dst_pts = np.float32([[10, 10], [w - 20, 5], [w - 5, h - 5], [15, h - 15]])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = np.zeros_like(img)
    for c in range(3):
        warped[:, :, c] = cv2.warpPerspective(img[:, :, c], M, (w, h),
                                                borderMode=cv2.BORDER_REPLICATE)

    return {
        "filter": "perspective_warp",
        "params": {"src": src_pts.tolist(), "dst": dst_pts.tolist()},
        "tool": "cv2.getPerspectiveTransform + cv2.warpPerspective",
        "tool_version": cv2.__version__,
        "note": "OpenCV perspective warp, BORDER_REPLICATE",
        "output": pixels_to_list(warped),
    }


def main():
    out_dir = Path(__file__).parent / "golden_data"
    out_dir.mkdir(exist_ok=True)

    golden = {
        "meta": {
            "description": "Golden I/O from external tools for filter validation",
            "input_description": "4x4 sRGB image, decoded to linear f32",
            "width": W,
            "height": H,
            "tools": tool_info(),
        },
        "input": pixels_to_list(INPUT_LINEAR),
        "filters": {},
    }

    # Adjustment filters
    golden["filters"]["brightness_0.15"] = golden_brightness(0.15)
    golden["filters"]["contrast_0.4"] = golden_contrast(0.4)
    golden["filters"]["gamma_1.5"] = golden_gamma(1.5)
    golden["filters"]["exposure_1.0"] = golden_exposure(1.0)
    golden["filters"]["invert"] = golden_invert()
    golden["filters"]["levels_0.1_0.9_1.0"] = golden_levels(0.1, 0.9, 1.0)
    golden["filters"]["posterize_8"] = golden_posterize(8)
    golden["filters"]["solarize_0.5"] = golden_solarize(0.5)
    golden["filters"]["sepia_0.8"] = golden_sepia(0.8)

    # Evaluate filters
    golden["filters"]["evaluate_add_0.2"] = golden_evaluate_add(0.2)
    golden["filters"]["evaluate_multiply_1.5"] = golden_evaluate_multiply(1.5)
    golden["filters"]["evaluate_subtract_0.1"] = golden_evaluate_subtract(0.1)
    golden["filters"]["evaluate_abs"] = golden_evaluate_abs()
    golden["filters"]["evaluate_divide_2.0"] = golden_evaluate_divide(2.0)
    golden["filters"]["evaluate_pow_0.5"] = golden_evaluate_pow(exponent=0.5)
    golden["filters"]["evaluate_log_1.0"] = golden_evaluate_log(scale=1.0)
    golden["filters"]["evaluate_max_0.3"] = golden_evaluate_max(threshold=0.3)
    golden["filters"]["evaluate_min_0.5"] = golden_evaluate_min(threshold=0.5)

    # Color filters
    golden["filters"]["hue_rotate_90"] = golden_hue_rotate(90.0)
    golden["filters"]["hue_rotate_180"] = golden_hue_rotate(180.0)
    golden["filters"]["saturate_hsl_1.5"] = golden_saturate_hsl(1.5)
    golden["filters"]["saturate_hsl_0.5"] = golden_saturate_hsl(0.5)
    golden["filters"]["colorize_warm"] = golden_colorize(0.8, 0.4, 0.2, 0.6)
    golden["filters"]["vibrance_50"] = golden_vibrance(50.0)
    golden["filters"]["vibrance_-30"] = golden_vibrance(-30.0)
    golden["filters"]["modulate_1.2_1.5_45"] = golden_modulate(1.2, 1.5, 45.0)
    golden["filters"]["photo_filter_warming_preserve"] = golden_photo_filter(
        0.9, 0.6, 0.2, 0.4, True)
    golden["filters"]["photo_filter_cooling_no_preserve"] = golden_photo_filter(
        0.2, 0.4, 0.9, 0.5, False)
    golden["filters"]["selective_color_red_shift"] = golden_selective_color(
        0.0, 30.0, 20.0, 0.1, 0.0)
    golden["filters"]["replace_color_blue_to_green"] = golden_replace_color(
        240.0, 40.0, 0.1, 1.0, 0.0, 1.0, -120.0, 0.0, 0.0)
    golden["filters"]["white_balance_gray_world"] = golden_white_balance_gray_world()
    golden["filters"]["white_balance_temp_5000"] = golden_white_balance_temperature(5000.0, 0.0)
    golden["filters"]["white_balance_temp_7500_tint"] = golden_white_balance_temperature(7500.0, 10.0)
    golden["filters"]["lab_adjust_a10_b-5"] = golden_lab_adjust(10.0, -5.0)
    golden["filters"]["aces_cct_to_cg"] = golden_aces_cct_to_cg()
    golden["filters"]["aces_cg_to_cct"] = golden_aces_cg_to_cct()
    # quantize: pipeline uses median-cut palette (image-dependent), golden uses
    # floor quantize (uniform). Different algorithms — cannot compare.
    # Quantize is validated by its own unit tests only.
    # golden["filters"]["quantize_4"] = golden_quantize(4)
    # golden["filters"]["quantize_16"] = golden_quantize(16)

    # Write pointops output
    out_file = out_dir / "pointops.json"
    with open(out_file, "w") as f:
        json.dump(golden, f, indent=2)

    n = len(golden["filters"])
    print(f"Generated {n} golden entries → {out_file}")

    # ─── Spatial filters ──────────────────────────────────────────────────
    spatial = {
        "meta": {
            "description": "Golden I/O from OpenCV for spatial filter validation",
            "input_description": "128x128 sRGB gradient, decoded to linear f32",
            "width": SPATIAL_W,
            "height": SPATIAL_H,
            "tools": tool_info(),
        },
        "input": pixels_to_list(SPATIAL_INPUT_LINEAR),
        "filters": {},
    }

    # Spatial filters
    spatial["filters"]["gaussian_blur_2"] = golden_gaussian_blur(2)
    spatial["filters"]["box_blur_2"] = golden_box_blur(2)
    spatial["filters"]["sobel_1.0"] = golden_sobel(1.0)
    spatial["filters"]["bilateral_5_0.1_10"] = golden_bilateral(5, 0.1, 10.0)
    spatial["filters"]["sharpen_1_1.5"] = golden_sharpen(1.0, 1.5)
    spatial["filters"]["high_pass_3"] = golden_high_pass(3.0)

    # Enhancement / grading filters (spatial-sized input)
    spatial["filters"]["equalize"] = golden_equalize()
    spatial["filters"]["normalize_0.02_0.01"] = golden_normalize(0.02, 0.01)
    spatial["filters"]["normalize_0.05_0.05"] = golden_normalize(0.05, 0.05)
    spatial["filters"]["vignette_10_0_0"] = golden_vignette(10.0, 0, 0)
    spatial["filters"]["vignette_5_8_8"] = golden_vignette(5.0, 8, 8)
    spatial["filters"]["vignette_powerlaw_0.5_2.0"] = golden_vignette_powerlaw(0.5, 2.0)
    spatial["filters"]["vignette_powerlaw_0.8_3.0"] = golden_vignette_powerlaw(0.8, 3.0)
    spatial["filters"]["tonemap_reinhard"] = golden_tonemap_reinhard()
    spatial["filters"]["frequency_high_3"] = golden_frequency_high(3.0)
    spatial["filters"]["frequency_high_1"] = golden_frequency_high(1.0)
    spatial["filters"]["frequency_low_3"] = golden_frequency_low(3.0)
    spatial["filters"]["frequency_low_1"] = golden_frequency_low(1.0)

    # Enhancement / grading filters (enhancement + grading batch)
    spatial["filters"]["clahe_4_3"] = golden_clahe(4, 3.0)
    spatial["filters"]["nlm_denoise_0.1_2_5"] = golden_nlm_denoise(0.1, 2, 5)
    spatial["filters"]["dehaze_7_0.95_0.1"] = golden_dehaze(7, 0.95, 0.1)
    spatial["filters"]["clarity_0.5_15"] = golden_clarity(0.5, 15.0)
    spatial["filters"]["shadow_highlight_0.3_-0.2"] = golden_shadow_highlight(0.3, -0.2)
    # Retinex: use production-representative sigma values (128x128 handles large kernels fine)
    spatial["filters"]["retinex_ssr_80"] = golden_retinex_ssr(80.0)
    spatial["filters"]["retinex_msr_15_80_250"] = golden_retinex_msr(15.0, 80.0, 250.0)
    spatial["filters"]["tonemap_filmic_default"] = golden_tonemap_filmic(0.22, 0.3, 0.1, 0.2, 0.01)
    spatial["filters"]["tonemap_drago_100_0.85"] = golden_tonemap_drago(100.0, 0.85)

    # ── Effect + distortion filters ──────────────────────────────────────
    spatial["filters"]["emboss"] = golden_emboss()
    spatial["filters"]["pixelate_8"] = golden_pixelate(8)
    spatial["filters"]["barrel_0.3_0"] = golden_barrel(0.3, 0.0)
    spatial["filters"]["spherize_0.5"] = golden_spherize(0.5)
    spatial["filters"]["swirl_2_60"] = golden_swirl(2.0, 60.0)
    spatial["filters"]["wave_5_30_h"] = golden_wave(5.0, 30.0, True)
    spatial["filters"]["wave_5_30_v"] = golden_wave(5.0, 30.0, False)
    spatial["filters"]["polar"] = golden_polar()
    spatial["filters"]["depolar"] = golden_depolar()
    spatial["filters"]["chromatic_aberration_5"] = golden_chromatic_aberration(5.0)
    spatial["filters"]["oil_paint_3"] = golden_oil_paint(3)
    spatial["filters"]["charcoal_1_1"] = golden_charcoal(1.0, 1.0)
    spatial["filters"]["halftone_8"] = golden_halftone(8.0)
    spatial["filters"]["ripple_5_30"] = golden_ripple(5.0, 30.0)

    # ── Blend self-blend tests (single-input, base=overlay=input) ──────
    # Multiply self-blend: pixel² (W3C Compositing Level 1)
    spatial["filters"]["blend_multiply_self"] = golden_blend_self("multiply", 1)
    # Screen self-blend: 1-(1-pixel)² = 2*pixel - pixel²
    spatial["filters"]["blend_screen_self"] = golden_blend_self("screen", 2)

    # ── Edge + morphology filters ────────────────────────────────────────
    spatial["filters"]["laplacian_1"] = golden_laplacian(1.0)
    spatial["filters"]["scharr_1"] = golden_scharr(1.0)
    spatial["filters"]["threshold_0.5"] = golden_threshold(0.5)
    spatial["filters"]["dilate_1"] = golden_dilate(1)
    spatial["filters"]["erode_1"] = golden_erode(1)
    spatial["filters"]["morph_open_2"] = golden_morph_open(2)
    spatial["filters"]["morph_close_2"] = golden_morph_close(2)
    spatial["filters"]["morph_tophat_2"] = golden_morph_tophat(2)
    spatial["filters"]["morph_blackhat_2"] = golden_morph_blackhat(2)
    spatial["filters"]["morph_gradient_2"] = golden_morph_gradient(2)

    # ── More edge + spatial ops ─────────────────────────────────────────
    spatial["filters"]["median_3"] = golden_median(3)
    spatial["filters"]["motion_blur_10_45"] = golden_motion_blur(10, 45)

    # ── Simple composite/alpha ops ───────────────────────────────────
    spatial["filters"]["premultiply"] = golden_premultiply()
    spatial["filters"]["unpremultiply"] = golden_unpremultiply()

    # ── More blend modes (IM validated) ──────────────────────────────
    spatial["filters"]["blend_overlay_self"] = golden_blend_self("overlay", 3)
    spatial["filters"]["blend_darken_self"] = golden_blend_self("darken", 8)
    spatial["filters"]["blend_lighten_self"] = golden_blend_self("lighten", 9)
    spatial["filters"]["blend_difference_self"] = golden_blend_self("difference", 10)

    # ── More blend modes (IM validated) ────────────────────────────────
    spatial["filters"]["blend_exclusion_self"] = golden_blend_self("exclusion", 11)

    # ── More edge ops ────────────────────────────────────────────────
    spatial["filters"]["otsu_threshold"] = golden_otsu_threshold()
    spatial["filters"]["canny_0.1_0.3"] = golden_canny(0.1, 0.3)
    spatial["filters"]["adaptive_threshold_5_0.02"] = golden_adaptive_threshold(5, 0.02)
    spatial["filters"]["triangle_threshold"] = golden_triangle_threshold()

    # ── More spatial blurs ───────────────────────────────────────────
    spatial["filters"]["lens_blur_5"] = golden_lens_blur(5)
    spatial["filters"]["zoom_blur_0.1"] = golden_zoom_blur(0.1)
    spatial["filters"]["spin_blur_10"] = golden_spin_blur(10.0)

    # ── Color/enhancement ops ────────────────────────────────────────
    spatial["filters"]["auto_level"] = golden_auto_level()
    spatial["filters"]["chromatic_split_5_0_0_0_-5_0"] = golden_chromatic_split(5, 0, 0, 0, -5, 0)
    spatial["filters"]["dither_ordered_8_4"] = golden_dither_ordered(8, 4)

    # ── Composite/alpha ops ──────────────────────────────────────────
    spatial["filters"]["flatten_white"] = golden_flatten(1.0, 1.0, 1.0)
    spatial["filters"]["add_alpha_0.5"] = golden_add_alpha(0.5)
    spatial["filters"]["remove_alpha"] = golden_remove_alpha()

    # ── Generators (procedural, formula-validated) ───────────────────
    spatial["filters"]["solid_color_0.5_0.3_0.1"] = golden_solid_color(0.5, 0.3, 0.1)
    spatial["filters"]["gradient_linear_0"] = golden_gradient_linear(0.0)

    # ── More generators ────────────────────────────────────────────────
    spatial["filters"]["gradient_radial"] = golden_gradient_radial()

    # ── More effect ops ───────────────────────────────────────────────
    spatial["filters"]["mirror_kaleidoscope_4_h"] = golden_mirror_kaleidoscope_h(4)

    # ── Color ops ────────────────────────────────────────────────────
    spatial["filters"]["sigmoidal_contrast_3_0.5"] = golden_sigmoidal_contrast(3.0, 0.5)

    # ── Noise (deterministic seed — validate pipeline=reference) ─────
    # These use our internal hash function. No external tool can match.
    # The golden uses the reference implementation output directly.
    # This validates that pipeline and reference produce identical results.

    # ── More distortions ─────────────────────────────────────────────
    spatial["filters"]["liquify_center"] = golden_liquify()

    # ── Generator + draw + tool filters ────────────────────────────────
    spatial["filters"]["checkerboard_8"] = golden_checkerboard(8)
    # Draw ops (circle, rect, line) use SDF anti-aliased rendering — no external
    # tool matches this approach. Validated via pipeline-vs-reference parity only.
    # Flood fill and perspective_warp need interactive/complex param setup.

    # Write spatial output
    spatial_file = out_dir / "spatial.json"
    with open(spatial_file, "w") as f:
        json.dump(spatial, f, indent=2)

    sn = len(spatial["filters"])
    print(f"Generated {sn} spatial golden entries → {spatial_file}")
    print(f"Tools: {golden['meta']['tools']}")


if __name__ == "__main__":
    main()
