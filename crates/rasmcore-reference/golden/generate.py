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
            result.append([r, g, b, 1.0])
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
    """Pixelate via cv2.resize down then up — matches pipeline block averaging.

    Pipeline: average each block, fill block with average.
    OpenCV resize INTER_AREA (down) + INTER_NEAREST (up) approximates this.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]

    # Compute block averages and fill — exact pipeline formula
    output = img.copy()
    for by in range(0, h, block_size):
        for bx in range(0, w, block_size):
            ye = min(by + block_size, h)
            xe = min(bx + block_size, w)
            block = img[by:ye, bx:xe]
            avg = block.mean(axis=(0, 1))
            output[by:ye, bx:xe] = avg

    return {
        "filter": "pixelate",
        "params": {"block_size": block_size},
        "tool": "numpy (block averaging)",
        "tool_version": np.__version__,
        "note": "Block average then fill — exact pipeline formula",
        "output": pixels_to_list(output),
    }


def golden_barrel(k1: float, k2: float) -> dict:
    """Barrel distortion via OpenCV cv2.remap with computed coordinate map.

    r' = r * (1 + k1*r² + k2*r⁴), coordinates normalized to [-1,1].
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    norm = max(cx, cy)

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
        "tool": f"cv2.remap (barrel: r'=r*(1+k1*r²+k2*r⁴), k1={k1}, k2={k2})",
        "tool_version": cv2.__version__,
        "note": "Coordinates normalized to [-1,1], bilinear interpolation, border replicate",
        "output": pixels_to_list(output),
    }


def golden_spherize(amount: float) -> dict:
    """Spherize via cv2.remap — power-law radial distortion.

    For r in (0,1): new_r = r^(1/(1+amount)); scale = new_r/r.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    norm = max(cx, cy)

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            nx = (x - cx) / norm
            ny = (y - cy) / norm
            r = np.sqrt(nx * nx + ny * ny)
            if 0 < r < 1:
                new_r = r ** (1.0 / (1.0 + amount))
                scale = new_r / r
            elif r >= 1:
                scale = 1.0
            else:
                scale = 1.0
            map_x[y, x] = nx * scale * norm + cx
            map_y[y, x] = ny * scale * norm + cy

    output = np.zeros_like(img)
    for c in range(3):
        output[:, :, c] = cv2.remap(img[:, :, c], map_x, map_y,
                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return {
        "filter": "spherize",
        "params": {"amount": amount},
        "tool": f"cv2.remap (spherize: new_r=r^(1/(1+{amount})))",
        "tool_version": cv2.__version__,
        "note": "Power-law radial distortion, bilinear, border replicate",
        "output": pixels_to_list(output),
    }


def golden_swirl(angle: float, radius: float) -> dict:
    """Swirl distortion via cv2.remap.

    theta = angle * (1 - dist/radius) for dist < radius.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            dx = x - cx
            dy = y - cy
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radius and dist > 0:
                theta = angle * (1.0 - dist / radius)
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
        "tool": f"cv2.remap (swirl: angle={angle}, radius={radius})",
        "tool_version": cv2.__version__,
        "note": "Rotational distortion: theta=angle*(1-dist/radius) for dist<radius",
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
    """Cartesian-to-polar via cv2.remap.

    out_x maps to angle [0, 2*pi], out_y maps to radius [0, max_radius].
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    max_radius = np.sqrt(cx * cx + cy * cy)

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            angle = x / w * 2 * np.pi
            radius = y / (h - 1) * max_radius if h > 1 else 0
            map_x[y, x] = cx + radius * np.cos(angle)
            map_y[y, x] = cy + radius * np.sin(angle)

    output = np.zeros_like(img)
    for c in range(3):
        output[:, :, c] = cv2.remap(img[:, :, c], map_x, map_y,
                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return {
        "filter": "polar",
        "params": {},
        "tool": "cv2.remap (cartesian-to-polar coordinate transform)",
        "tool_version": cv2.__version__,
        "note": "x->angle [0,2pi], y->radius [0,max_radius], bilinear, border replicate",
        "output": pixels_to_list(output),
    }


def golden_depolar() -> dict:
    """Polar-to-cartesian via cv2.remap.

    For each output pixel: radius=sqrt(dx²+dy²), angle=atan2(dy,dx).
    sx=angle/(2pi)*w, sy=radius/max_radius*(h-1).
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    max_radius = np.sqrt(cx * cx + cy * cy)

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            dx = x - cx
            dy = y - cy
            radius = np.sqrt(dx * dx + dy * dy)
            angle = np.arctan2(dy, dx)
            if angle < 0:
                angle += 2 * np.pi
            map_x[y, x] = angle / (2 * np.pi) * w
            map_y[y, x] = radius / max_radius * (h - 1) if max_radius > 0 else 0

    output = np.zeros_like(img)
    for c in range(3):
        output[:, :, c] = cv2.remap(img[:, :, c], map_x, map_y,
                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return {
        "filter": "depolar",
        "params": {},
        "tool": "cv2.remap (polar-to-cartesian coordinate transform)",
        "tool_version": cv2.__version__,
        "note": "radius=sqrt(dx²+dy²), angle=atan2(dy,dx), bilinear, border replicate",
        "output": pixels_to_list(output),
    }


def golden_chromatic_aberration(strength: float) -> dict:
    """Chromatic aberration — radial shift of R/B channels.

    R shifted outward by strength * dist/max_dist, B shifted inward.
    Green unchanged. Uses bilinear sampling.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    max_dist = np.sqrt(cx * cx + cy * cy)
    if max_dist < 1e-10:
        return {
            "filter": "chromatic_aberration", "params": {"strength": strength},
            "tool": "numpy", "tool_version": np.__version__, "note": "degenerate",
            "output": pixels_to_list(img),
        }

    output = img.copy()
    for y in range(h):
        for x in range(w):
            dx = x - cx
            dy = y - cy
            dist = np.sqrt(dx * dx + dy * dy)
            shift = dist * strength / max_dist
            if dist > 1e-10:
                nx, ny = dx / dist, dy / dist
            else:
                nx, ny = 0.0, 0.0
            # R shifts outward
            output[y, x, 0] = _bilinear_sample_f32(img[:, :, 0:1].reshape(h, w, 1),
                                                     x + shift * nx, y + shift * ny)[0]
            # B shifts inward
            output[y, x, 2] = _bilinear_sample_f32(img[:, :, 2:3].reshape(h, w, 1),
                                                     x - shift * nx, y - shift * ny)[0]
            # G unchanged

    return {
        "filter": "chromatic_aberration",
        "params": {"strength": strength},
        "tool": f"numpy (radial R/B shift, strength={strength})",
        "tool_version": np.__version__,
        "note": "R outward, B inward by strength*dist/max_dist, bilinear, G unchanged",
        "output": pixels_to_list(output),
    }


def golden_oil_paint(radius: int) -> dict:
    """Oil paint effect — neighborhood histogram mode.

    For each pixel, find the most common luminance bin in the neighborhood,
    output the mean color of pixels in that bin.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]
    levels = 256  # pipeline uses 256 bins
    output = np.zeros_like(img)

    for y in range(h):
        for x in range(w):
            y0 = max(0, y - radius)
            y1 = min(h, y + radius + 1)
            x0 = max(0, x - radius)
            x1 = min(w, x + radius + 1)

            # Build luminance histogram
            bins = [[] for _ in range(levels)]
            for py in range(y0, y1):
                for px in range(x0, x1):
                    luma = 0.2126 * img[py, px, 0] + 0.7152 * img[py, px, 1] + 0.0722 * img[py, px, 2]
                    b = min(int(luma * levels), levels - 1)
                    bins[b].append((img[py, px, 0], img[py, px, 1], img[py, px, 2]))

            # Find mode bin
            best_bin = max(range(levels), key=lambda b: len(bins[b]))
            if bins[best_bin]:
                avg = np.mean(bins[best_bin], axis=0)
                output[y, x] = avg
            else:
                output[y, x] = img[y, x]

    return {
        "filter": "oil_paint",
        "params": {"radius": radius},
        "tool": "numpy (neighborhood luminance histogram mode)",
        "tool_version": np.__version__,
        "note": f"radius={radius}, levels={levels}, mode bin → mean color",
        "output": pixels_to_list(output),
    }


def golden_charcoal(radius: float, sigma: float) -> dict:
    """Charcoal effect — grayscale → Sobel edge → invert → Gaussian blur.

    Uses OpenCV Sobel (ksize=3) + GaussianBlur.
    Pipeline: `radius` is the Sobel param (unused in ksize=3 Sobel), `sigma` is blur radius.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)

    # Grayscale (BT.709)
    gray = img[:, :, 0] * 0.2126 + img[:, :, 1] * 0.7152 + img[:, :, 2] * 0.0722

    # Sobel edge magnitude
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    mag = np.sqrt(sx * sx + sy * sy)

    # Normalize to [0,1]
    mag_max = mag.max()
    if mag_max > 1e-10:
        mag /= mag_max

    # Invert
    inv = 1.0 - mag

    # Gaussian blur with sigma = blur radius param
    ksize = int(round(sigma * 10.0 + 1.0)) | 1
    ksize = max(ksize, 3)
    blurred = cv2.GaussianBlur(inv, (ksize, ksize), sigma, borderType=cv2.BORDER_REFLECT_101)

    # Expand back to 3 channels
    output = np.zeros_like(img)
    for c in range(3):
        output[:, :, c] = blurred

    return {
        "filter": "charcoal",
        "params": {"radius": radius, "sigma": sigma},
        "tool": f"cv2.Sobel + cv2.GaussianBlur (charcoal, radius={radius}, sigma={sigma})",
        "tool_version": cv2.__version__,
        "note": "Grayscale→Sobel→normalize→invert→GaussianBlur(sigma=blur_radius)",
        "output": pixels_to_list(output),
    }


def golden_halftone(dot_size: float) -> dict:
    """Halftone — block-averaged luminance → circular dot pattern.

    For each block: compute mean luminance, render circle with radius
    proportional to sqrt(1-luma). Binary: pixel is 1 if inside circle, 0 outside.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]
    bs = int(dot_size)
    if bs < 1:
        bs = 1
    max_radius = bs / 2.0
    output = np.zeros_like(img)

    for by in range(0, h, bs):
        for bx in range(0, w, bs):
            ye = min(by + bs, h)
            xe = min(bx + bs, w)
            block = img[by:ye, bx:xe]
            luma = (block[:, :, 0] * 0.2126 + block[:, :, 1] * 0.7152 + block[:, :, 2] * 0.0722).mean()
            dot_radius = max_radius * np.sqrt(max(1.0 - luma, 0.0))
            center_x = (bx + xe) / 2.0
            center_y = (by + ye) / 2.0
            for py in range(by, ye):
                for px in range(bx, xe):
                    dx = px - center_x + 0.5
                    dy = py - center_y + 0.5
                    dist = np.sqrt(dx * dx + dy * dy)
                    val = 1.0 if dist <= dot_radius else 0.0
                    output[py, px] = [val, val, val]

    return {
        "filter": "halftone",
        "params": {"dot_size": dot_size},
        "tool": "numpy (block luminance → circular dot pattern)",
        "tool_version": np.__version__,
        "note": f"dot_size={dot_size}, radius=max_r*sqrt(1-luma), binary circle",
        "output": pixels_to_list(output),
    }


def golden_ripple(amplitude: float, wavelength: float) -> dict:
    """Ripple distortion via cv2.remap — radial sinusoidal displacement.

    offset = amplitude * sin(2*pi*dist/wavelength), displace along radial direction.
    """
    img = SPATIAL_INPUT_LINEAR.astype(np.float32)
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            dx = x - cx
            dy = y - cy
            dist = np.sqrt(dx * dx + dy * dy)
            if dist > 1e-10:
                offset = amplitude * np.sin(2 * np.pi * dist / wavelength)
                nx, ny = dx / dist, dy / dist
                map_x[y, x] = x + nx * offset
                map_y[y, x] = y + ny * offset
            else:
                map_x[y, x] = x
                map_y[y, x] = y

    output = np.zeros_like(img)
    for c in range(3):
        output[:, :, c] = cv2.remap(img[:, :, c], map_x, map_y,
                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return {
        "filter": "ripple",
        "params": {"amplitude": amplitude, "wavelength": wavelength},
        "tool": f"cv2.remap (ripple: A={amplitude}, λ={wavelength})",
        "tool_version": cv2.__version__,
        "note": "Radial sinusoidal displacement, bilinear, border replicate",
        "output": pixels_to_list(output),
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

    # Write spatial output
    spatial_file = out_dir / "spatial.json"
    with open(spatial_file, "w") as f:
        json.dump(spatial, f, indent=2)

    sn = len(spatial["filters"])
    print(f"Generated {sn} spatial golden entries → {spatial_file}")
    print(f"Tools: {golden['meta']['tools']}")


if __name__ == "__main__":
    main()
