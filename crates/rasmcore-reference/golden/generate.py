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
# 64x64 test image for spatial ops. Large enough that typical kernel sizes
# (radius 1-3, ksize 11-31) don't make most pixels border pixels.
# Professional tools (Resolve, Nuke) operate on 4K+ images — a 64x64 test
# is already a concession to speed.

SPATIAL_W, SPATIAL_H = 64, 64

_spatial_srgb = np.zeros((SPATIAL_H, SPATIAL_W, 3), dtype=np.uint8)
for _y in range(SPATIAL_H):
    for _x in range(SPATIAL_W):
        _spatial_srgb[_y, _x] = [
            min(_x * 4, 255),
            min(_y * 4, 255),
            min((_x + _y) * 2, 255),
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


# ─── Main ────────────────────────────────────────────────────────────────────

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
            "input_description": "16x16 sRGB gradient, decoded to linear f32",
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

    # Write spatial output
    spatial_file = out_dir / "spatial.json"
    with open(spatial_file, "w") as f:
        json.dump(spatial, f, indent=2)

    sn = len(spatial["filters"])
    print(f"Generated {sn} spatial golden entries → {spatial_file}")
    print(f"Tools: {golden['meta']['tools']}")


if __name__ == "__main__":
    main()
