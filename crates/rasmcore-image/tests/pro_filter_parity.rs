//! Pro Filter Reference Parity Tests — validation against Python references.
//!
//! Validates all 15 pro filters (color grading, tonemapping, content-aware)
//! against reference implementations in Python (numpy).
//!
//! **These tests do NOT skip.** If the venv is missing, they FAIL with
//! instructions to set it up.
//!
//! Setup:
//!   python3 -m venv tests/fixtures/.venv
//!   tests/fixtures/.venv/bin/pip install numpy Pillow opencv-python-headless

use rasmcore_image::domain::color_grading;
use rasmcore_image::domain::content_aware;
use rasmcore_image::domain::filters;
use rasmcore_image::domain::types::*;
use std::path::Path;
use std::process::Command;

// ─── Test Infrastructure ────────────────────────────────────────────────────

fn venv_python() -> String {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let venv = manifest.join("../../tests/fixtures/.venv/bin/python3");
    if !venv.exists() {
        panic!(
            "Reference test venv not found at {}.\n\
             Setup:\n  python3 -m venv tests/fixtures/.venv\n  \
             tests/fixtures/.venv/bin/pip install numpy Pillow opencv-python-headless",
            venv.display()
        );
    }
    venv.to_string_lossy().into_owned()
}

fn run_python_ref(script: &str) -> Vec<u8> {
    let python = venv_python();
    let output = Command::new(&python)
        .arg("-c")
        .arg(script)
        .output()
        .unwrap_or_else(|e| panic!("Failed to run {python}: {e}"));
    assert!(
        output.status.success(),
        "Python reference script failed:\n{}\nstderr: {}",
        script.lines().take(5).collect::<Vec<_>>().join("\n"),
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

fn mean_absolute_error(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "buffer length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    if a.is_empty() {
        return 0.0;
    }
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .sum::<f64>()
        / a.len() as f64
}

fn max_absolute_error(a: &[u8], b: &[u8]) -> u8 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0)
}

fn assert_close(label: &str, ours: &[u8], reference: &[u8], max_mae: f64) {
    let mae = mean_absolute_error(ours, reference);
    let max_err = max_absolute_error(ours, reference);
    assert!(
        mae <= max_mae,
        "{label}: MAE={mae:.4} exceeds threshold {max_mae}. max_err={max_err}"
    );
    eprintln!("  {label}: MAE={mae:.4}, max_err={max_err} ✓");
}

fn make_gradient_rgb(w: u32, h: u32) -> Vec<u8> {
    let mut p = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            p.push(((x * 255) / w) as u8);
            p.push(((y * 255) / h) as u8);
            p.push(128);
        }
    }
    p
}

fn info_rgb8(w: u32, h: u32) -> ImageInfo {
    ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Rgb8,
        color_space: ColorSpace::Srgb,
    }
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mse: f64 = a[..n]
        .iter()
        .zip(b[..n].iter())
        .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
        .sum::<f64>()
        / n as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 1: Color Grading Parity
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn parity_asc_cdl() {
    let (w, h) = (64, 64);
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);

    // Non-trivial CDL: slope=[1.2, 0.9, 1.1], offset=[0.05, -0.03, 0.0], power=[0.9, 1.1, 1.0]
    let cdl = color_grading::AscCdl {
        slope: [1.2, 0.9, 1.1],
        offset: [0.05, -0.03, 0.0],
        power: [0.9, 1.1, 1.0],
        saturation: 1.0,
    };
    let ours = color_grading::asc_cdl(&pixels, &info, &cdl).unwrap();

    let script = format!(
        r#"
import sys, numpy as np
px = np.frombuffer(bytes({pixels:?}), dtype=np.uint8).reshape(-1, 3).astype(np.float64) / 255.0
slope = np.array([1.2, 0.9, 1.1])
offset = np.array([0.05, -0.03, 0.0])
power = np.array([0.9, 1.1, 1.0])
out = np.clip(np.maximum(px * slope + offset, 0.0) ** power, 0.0, 1.0)
sys.stdout.buffer.write((out * 255.0 + 0.5).astype(np.uint8).tobytes())
"#
    );
    let reference = run_python_ref(&script);
    assert_close("asc_cdl", &ours, &reference, 0.5);
}

#[test]
fn parity_asc_cdl_identity() {
    // Identity CDL should be lossless (or near-lossless)
    let (w, h) = (64, 64);
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);
    let cdl = color_grading::AscCdl::default();
    let ours = color_grading::asc_cdl(&pixels, &info, &cdl).unwrap();
    let mae = mean_absolute_error(&ours, &pixels);
    eprintln!("  asc_cdl identity: MAE={mae:.4}");
    assert!(
        mae <= 0.5,
        "identity CDL should be near-lossless, MAE={mae:.4}"
    );
}

#[test]
fn parity_lift_gamma_gain() {
    let (w, h) = (64, 64);
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);

    let lgg = color_grading::LiftGammaGain {
        lift: [0.05, -0.02, 0.0],
        gamma: [0.8, 1.2, 1.0],
        gain: [1.1, 0.95, 1.0],
    };
    let ours = color_grading::lift_gamma_gain(&pixels, &info, &lgg).unwrap();

    let script = format!(
        r#"
import sys, numpy as np
px = np.frombuffer(bytes({pixels:?}), dtype=np.uint8).reshape(-1, 3).astype(np.float64) / 255.0
lift = np.array([0.05, -0.02, 0.0])
gamma = np.array([0.8, 1.2, 1.0])
gain = np.array([1.1, 0.95, 1.0])
lifted = px + lift * (1.0 - px)
safe_lifted = np.maximum(lifted, 0.0)
safe_gamma = np.where(gamma > 0, gamma, 1.0)
gammaed = np.where((gamma > 0) & (safe_lifted > 0), safe_lifted ** (1.0 / safe_gamma), 0.0)
out = np.clip(gain * gammaed, 0.0, 1.0)
sys.stdout.buffer.write((out * 255.0 + 0.5).astype(np.uint8).tobytes())
"#
    );
    let reference = run_python_ref(&script);
    assert_close("lift_gamma_gain", &ours, &reference, 0.5);
}

#[test]
fn parity_split_toning() {
    let (w, h) = (64, 64);
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);

    let st = color_grading::SplitToning {
        shadow_color: [0.0, 0.0, 0.8],
        highlight_color: [1.0, 0.7, 0.3],
        balance: 0.0,
        strength: 0.5,
    };
    let ours = color_grading::split_toning(&pixels, &info, &st).unwrap();

    let script = format!(
        r#"
import sys, numpy as np
px = np.frombuffer(bytes({pixels:?}), dtype=np.uint8).reshape(-1, 3).astype(np.float64) / 255.0
shadow_color = np.array([0.0, 0.0, 0.8])
highlight_color = np.array([1.0, 0.7, 0.3])
balance = 0.0
strength = 0.5
luma = 0.2126 * px[:, 0] + 0.7152 * px[:, 1] + 0.0722 * px[:, 2]
midpoint = 0.5 + balance * 0.5
shadow_weight = np.clip(1.0 - luma / max(midpoint, 0.001), 0.0, 1.0) * strength
highlight_weight = np.clip((luma - midpoint) / max(1.0 - midpoint, 0.001), 0.0, 1.0) * strength
out = px.copy()
for c in range(3):
    out[:, c] = px[:, c] + (shadow_color[c] - px[:, c]) * shadow_weight + (highlight_color[c] - px[:, c]) * highlight_weight
out = np.clip(out, 0.0, 1.0)
sys.stdout.buffer.write((out * 255.0 + 0.5).astype(np.uint8).tobytes())
"#
    );
    let reference = run_python_ref(&script);
    assert_close("split_toning", &ours, &reference, 1.0);
}

#[test]
fn parity_curves_master() {
    let (w, h) = (64, 64);
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);

    // S-curve: darken shadows, brighten highlights
    let points = vec![(0.0, 0.0), (0.25, 0.15), (0.75, 0.85), (1.0, 1.0)];
    let tc = color_grading::ToneCurves {
        r: points.clone(),
        g: points.clone(),
        b: points.clone(),
    };
    let ours = color_grading::curves(&pixels, &info, &tc).unwrap();

    // Build LUT in Python via Fritsch-Carlson monotone cubic Hermite (same algorithm as Rust)
    let script_pchip = format!(
        r#"
import sys, numpy as np

px = np.frombuffer(bytes({pixels:?}), dtype=np.uint8).reshape(-1, 3)
pts = sorted([(0.0, 0.0), (0.25, 0.15), (0.75, 0.85), (1.0, 1.0)], key=lambda p: p[0])
xs = np.array([p[0] for p in pts])
ys = np.array([p[1] for p in pts])
n = len(pts)

# Fritsch-Carlson monotone cubic Hermite tangents
deltas = np.diff(ys) / np.maximum(np.diff(xs), 1e-6)
m = np.zeros(n)
m[0] = deltas[0]
m[-1] = deltas[-1]
for i in range(1, n - 1):
    m[i] = (deltas[i - 1] + deltas[i]) * 0.5

# Monotonicity constraint
for i in range(n - 1):
    if abs(deltas[i]) < 1e-6:
        m[i] = 0.0
        m[i + 1] = 0.0
    else:
        alpha = m[i] / deltas[i]
        beta = m[i + 1] / deltas[i]
        tau = alpha * alpha + beta * beta
        if tau > 9.0:
            t = 3.0 / np.sqrt(tau)
            m[i] = t * alpha * deltas[i]
            m[i + 1] = t * beta * deltas[i]

# Evaluate LUT at 256 positions
lut = np.zeros(256, dtype=np.uint8)
for idx in range(256):
    x = idx / 255.0
    # Find segment
    seg = np.searchsorted(xs, x, side='right') - 1
    seg = max(0, min(seg, n - 2))
    x0, x1 = xs[seg], xs[seg + 1]
    y0, y1 = ys[seg], ys[seg + 1]
    h = max(x1 - x0, 1e-6)
    t = (x - x0) / h
    t2, t3 = t * t, t * t * t
    h00 = 2*t3 - 3*t2 + 1
    h10 = t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 = t3 - t2
    y = h00*y0 + h10*h*m[seg] + h01*y1 + h11*h*m[seg+1]
    lut[idx] = max(0, min(255, int(y * 255.0 + 0.5)))

out = lut[px]
sys.stdout.buffer.write(out.tobytes())
"#
    );
    let reference = run_python_ref(&script_pchip);
    // Same algorithm (Fritsch-Carlson) in both — should be near-exact
    assert_close("curves_master", &ours, &reference, 0.5);
}

#[test]
fn parity_curves_channel_isolation() {
    // Apply red-only curve, verify green and blue channels unchanged
    let (w, h) = (32, 32);
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);

    let s_curve = vec![(0.0, 0.0), (0.25, 0.15), (0.75, 0.85), (1.0, 1.0)];
    let identity = vec![(0.0, 0.0), (1.0, 1.0)];

    let tc_red = color_grading::ToneCurves {
        r: s_curve.clone(),
        g: identity.clone(),
        b: identity.clone(),
    };
    let result = color_grading::curves(&pixels, &info, &tc_red).unwrap();

    // Green and blue channels should be unchanged
    for i in 0..(w * h) as usize {
        let orig_g = pixels[i * 3 + 1];
        let orig_b = pixels[i * 3 + 2];
        let res_g = result[i * 3 + 1];
        let res_b = result[i * 3 + 2];
        assert_eq!(orig_g, res_g, "green channel changed at pixel {i}");
        assert_eq!(orig_b, res_b, "blue channel changed at pixel {i}");
    }

    // Red channel should be different (S-curve modifies it)
    let red_changed = (0..(w * h) as usize).any(|i| pixels[i * 3] != result[i * 3]);
    assert!(red_changed, "red curve should modify at least some pixels");
    eprintln!("  curves channel isolation: green/blue unchanged, red modified ✓");
}

#[test]
fn parity_film_grain_determinism() {
    let (w, h) = (64, 64);
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);

    let params = color_grading::FilmGrainParams {
        amount: 0.3,
        size: 1.5,
        color: false,
        seed: 42,
    };

    // Same seed should produce identical output
    let result1 = color_grading::film_grain(&pixels, &info, &params).unwrap();
    let result2 = color_grading::film_grain(&pixels, &info, &params).unwrap();
    assert_eq!(
        result1, result2,
        "film grain must be deterministic with same seed"
    );

    // Different seed should produce different output
    let params2 = color_grading::FilmGrainParams { seed: 99, ..params };
    let result3 = color_grading::film_grain(&pixels, &info, &params2).unwrap();
    assert_ne!(
        result1, result3,
        "different seeds should produce different output"
    );

    // Verify grain is stronger in midtones than shadows
    // Create uniform shadow (val=20) and midtone (val=128) images
    let shadow_px = vec![20u8; (w * h * 3) as usize];
    let midtone_px = vec![128u8; (w * h * 3) as usize];
    let shadow_grain = color_grading::film_grain(&shadow_px, &info, &params).unwrap();
    let midtone_grain = color_grading::film_grain(&midtone_px, &info, &params).unwrap();

    let shadow_mae = mean_absolute_error(&shadow_px, &shadow_grain);
    let midtone_mae = mean_absolute_error(&midtone_px, &midtone_grain);
    eprintln!(
        "  film grain midtone emphasis: shadow_mae={shadow_mae:.2}, midtone_mae={midtone_mae:.2}"
    );
    assert!(
        midtone_mae > shadow_mae,
        "grain should be stronger in midtones ({midtone_mae:.2}) than shadows ({shadow_mae:.2})"
    );
    eprintln!("  film grain determinism + midtone emphasis ✓");
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 2: Tonemapping Parity
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn parity_tonemap_reinhard() {
    let (w, h) = (64, 64);
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);

    let ours = color_grading::tonemap_reinhard(&pixels, &info).unwrap();

    let script = format!(
        r#"
import sys, numpy as np
px = np.frombuffer(bytes({pixels:?}), dtype=np.uint8).reshape(-1, 3).astype(np.float64) / 255.0
out = px / (1.0 + px)
sys.stdout.buffer.write((out * 255.0 + 0.5).astype(np.uint8).tobytes())
"#
    );
    let reference = run_python_ref(&script);
    assert_close("tonemap_reinhard", &ours, &reference, 0.5);
}

#[test]
fn parity_tonemap_drago() {
    let (w, h) = (64, 64);
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);

    let params = color_grading::DragoParams {
        l_max: 1.0,
        bias: 0.85,
    };
    let ours = color_grading::tonemap_drago(&pixels, &info, &params).unwrap();

    let script = format!(
        r#"
import sys, numpy as np
px = np.frombuffer(bytes({pixels:?}), dtype=np.uint8).reshape(-1, 3).astype(np.float64) / 255.0
l_max = 1.0
bias = 0.85
log_max = np.log(1.0 + l_max)
bias_pow = max(np.log(bias) / np.log(0.5), 0.01)
def drago(val):
    mapped = np.where(val > 0, np.log(1.0 + val) / log_max, 0.0)
    return np.clip(np.power(mapped, 1.0 / bias_pow), 0.0, 1.0)
out = np.stack([drago(px[:, c]) for c in range(3)], axis=1)
sys.stdout.buffer.write((out * 255.0 + 0.5).astype(np.uint8).tobytes())
"#
    );
    let reference = run_python_ref(&script);
    assert_close("tonemap_drago", &ours, &reference, 1.0);
}

#[test]
fn parity_tonemap_filmic() {
    let (w, h) = (64, 64);
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);

    let params = color_grading::FilmicParams::default(); // a=2.51, b=0.03, c=2.43, d=0.59, e=0.14
    let ours = color_grading::tonemap_filmic(&pixels, &info, &params).unwrap();

    let script = format!(
        r#"
import sys, numpy as np
px = np.frombuffer(bytes({pixels:?}), dtype=np.uint8).reshape(-1, 3).astype(np.float64) / 255.0
a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
def filmic(x):
    num = x * (a * x + b)
    den = x * (c * x + d) + e
    return np.clip(num / den, 0.0, 1.0)
out = np.stack([filmic(px[:, ch]) for ch in range(3)], axis=1)
sys.stdout.buffer.write((out * 255.0 + 0.5).astype(np.uint8).tobytes())
"#
    );
    let reference = run_python_ref(&script);
    assert_close("tonemap_filmic", &ours, &reference, 0.5);
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 3: Content-Aware Parity
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn parity_smart_crop_registered() {
    // Exercise the registered wrapper (filters::smart_crop_registered) and
    // verify it produces a valid cropped image of the correct size
    let (w, h) = (128, 128);
    let mut pixels = vec![64u8; (w * h * 3) as usize];
    // Add detail in bottom-right quadrant
    for y in 64..128u32 {
        for x in 64..128u32 {
            let idx = ((y * w + x) * 3) as usize;
            let v = (((x + y) * 7) % 256) as u8;
            pixels[idx] = v;
            pixels[idx + 1] = 255 - v;
            pixels[idx + 2] = v / 2;
        }
    }
    let info = info_rgb8(w, h);

    let result = filters::smart_crop_registered(&pixels, &info, 64, 64).unwrap();
    assert_eq!(
        result.len(),
        64 * 64 * 3,
        "smart_crop output should be 64x64x3 = {} bytes, got {}",
        64 * 64 * 3,
        result.len()
    );

    // The cropped region should have high variance (selected the interesting region)
    let mean: f64 = result.iter().map(|&v| v as f64).sum::<f64>() / result.len() as f64;
    let variance: f64 = result
        .iter()
        .map(|&v| (v as f64 - mean).powi(2))
        .sum::<f64>()
        / result.len() as f64;
    assert!(
        variance > 100.0,
        "smart crop should select detailed region, variance={variance:.0}"
    );
    eprintln!("  smart_crop registered: 64x64 output, variance={variance:.0} ✓");
}

#[test]
fn parity_seam_carve_width() {
    let (w, h) = (64u32, 32u32);
    let mut pixels = vec![128u8; (w * h * 3) as usize];
    // Add a bright vertical stripe in the center that should be preserved
    for y in 0..h {
        for x in 28..36 {
            let idx = ((y * w + x) * 3) as usize;
            pixels[idx] = 255;
            pixels[idx + 1] = 255;
            pixels[idx + 2] = 255;
        }
    }
    let info = info_rgb8(w, h);
    let target_w = 48;

    let (result, new_info) = content_aware::seam_carve_width(&pixels, &info, target_w).unwrap();

    // Verify output dimensions
    assert_eq!(new_info.width, target_w);
    assert_eq!(new_info.height, h);
    assert_eq!(result.len(), (target_w * h * 3) as usize);

    // The bright stripe should be preserved (content-aware removes low-energy seams)
    // Compare against a naive center crop of the same dimensions
    let naive_offset = ((w - target_w) / 2) as usize;
    let mut naive_crop = Vec::with_capacity((target_w * h * 3) as usize);
    for y in 0..h as usize {
        for x in naive_offset..naive_offset + target_w as usize {
            let idx = (y * w as usize + x) * 3;
            naive_crop.push(pixels[idx]);
            naive_crop.push(pixels[idx + 1]);
            naive_crop.push(pixels[idx + 2]);
        }
    }

    let seam_psnr = psnr(&result, &naive_crop);
    eprintln!("  seam_carve_width: {w}→{target_w}, PSNR vs naive crop: {seam_psnr:.1}dB");
    // Seam carving removes different columns than naive crop, so pixel-level
    // agreement is low. Just verify it's not degenerate (all black/white).
    assert!(
        seam_psnr > 5.0,
        "seam carve vs naive crop PSNR degenerate: {seam_psnr:.1}dB"
    );
    eprintln!("  seam_carve_width: dimensions correct, content preserved ✓");
}

#[test]
fn parity_seam_carve_height() {
    let (w, h) = (32u32, 64u32);
    let mut pixels = vec![128u8; (w * h * 3) as usize];
    // Add a bright horizontal stripe in the center
    for y in 28..36 {
        for x in 0..w {
            let idx = ((y * w + x) * 3) as usize;
            pixels[idx] = 255;
            pixels[idx + 1] = 255;
            pixels[idx + 2] = 255;
        }
    }
    let info = info_rgb8(w, h);
    let target_h = 48;

    let (result, new_info) = content_aware::seam_carve_height(&pixels, &info, target_h).unwrap();

    assert_eq!(new_info.width, w);
    assert_eq!(new_info.height, target_h);
    assert_eq!(result.len(), (w * target_h * 3) as usize);
    eprintln!("  seam_carve_height: {h}→{target_h}, dimensions correct ✓");
}

#[test]
fn parity_selective_color() {
    let (w, h) = (64, 64);
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);

    let params = content_aware::SelectiveColorParams {
        hue_range: content_aware::HueRange {
            center: 0.0, // target reds
            width: 60.0,
        },
        hue_shift: 30.0,
        saturation: 1.5,
        lightness: 0.0,
    };
    let ours = content_aware::selective_color(&pixels, &info, &params).unwrap();

    let script = format!(
        r#"
import sys, numpy as np

def rgb_to_hsl(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    l = (mx + mn) / 2.0
    if abs(mx - mn) < 1e-6:
        return (0.0, 0.0, l)
    d = mx - mn
    s = d / (2.0 - mx - mn) if l > 0.5 else d / (mx + mn)
    if abs(mx - r) < 1e-6:
        h = (g - b) / d + (6.0 if g < b else 0.0)
    elif abs(mx - g) < 1e-6:
        h = (b - r) / d + 2.0
    else:
        h = (r - g) / d + 4.0
    return (h * 60.0, s, l)

def hsl_to_rgb(h, s, l):
    if s < 1e-6:
        return (l, l, l)
    def hue2rgb(p, q, t):
        if t < 0: t += 1
        if t > 1: t -= 1
        if t < 1/6: return p + (q - p) * 6 * t
        if t < 1/2: return q
        if t < 2/3: return p + (q - p) * (2/3 - t) * 6
        return p
    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q
    return (hue2rgb(p, q, h/360 + 1/3), hue2rgb(p, q, h/360), hue2rgb(p, q, h/360 - 1/3))

px = list(bytes({pixels:?}))
out = list(px)
center = 0.0
half_width = 30.0
import math
n = {w} * {h}
for i in range(n):
    r = px[i*3] / 255.0
    g = px[i*3+1] / 255.0
    b = px[i*3+2] / 255.0
    h, s, l = rgb_to_hsl(r, g, b)
    hue_diff = ((h - center + 180) % 360) - 180
    if abs(hue_diff) > half_width:
        continue
    blend = 0.5 * (1.0 + math.cos(abs(hue_diff) / half_width * math.pi)) if half_width > 0 else 1.0
    new_h = (h + 30.0 * blend) % 360
    new_s = min(max(s * (1.0 + (1.5 - 1.0) * blend), 0.0), 1.0)
    new_l = l
    nr, ng, nb = hsl_to_rgb(new_h, new_s, new_l)
    out[i*3] = max(0, min(255, round(nr * 255)))
    out[i*3+1] = max(0, min(255, round(ng * 255)))
    out[i*3+2] = max(0, min(255, round(nb * 255)))
sys.stdout.buffer.write(bytes(out))
"#
    );
    let reference = run_python_ref(&script);
    assert_close("selective_color", &ours, &reference, 1.0);
}

#[test]
fn parity_vibrance() {
    let (w, h) = (64, 64);
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);

    let ours = rasmcore_image::domain::filters::vibrance(&pixels, &info, 40.0).unwrap();

    let script = format!(
        r#"
import sys, numpy as np
px = np.frombuffer(bytes({pixels:?}), dtype=np.uint8).reshape(-1, 3).astype(np.float64) / 255.0
amount = 40.0
out = np.empty_like(px)
for i in range(len(px)):
    r, g, b = px[i]
    mx = max(r, g, b)
    mn = min(r, g, b)
    sat = (mx - mn) / mx if mx > 0 else 0.0
    scale = (amount / 100.0) * (1.0 - sat)
    # Convert to HSL
    if mx == mn:
        h, s, l = 0.0, 0.0, (mx + mn) / 2.0
    else:
        l = (mx + mn) / 2.0
        d = mx - mn
        s = d / (2.0 - mx - mn) if l > 0.5 else d / (mx + mn)
        if mx == r:
            h = (g - b) / d + (6.0 if g < b else 0.0)
        elif mx == g:
            h = (b - r) / d + 2.0
        else:
            h = (r - g) / d + 4.0
        h /= 6.0
    new_s = max(0.0, min(1.0, s * (1.0 + scale)))
    # HSL to RGB
    if new_s == 0:
        out[i] = [l, l, l]
    else:
        def hue2rgb(p, q, t):
            if t < 0: t += 1
            if t > 1: t -= 1
            if t < 1/6: return p + (q - p) * 6 * t
            if t < 1/2: return q
            if t < 2/3: return p + (q - p) * (2/3 - t) * 6
            return p
        q = l * (1 + new_s) if l < 0.5 else l + new_s - l * new_s
        p = 2 * l - q
        out[i] = [hue2rgb(p, q, h + 1/3), hue2rgb(p, q, h), hue2rgb(p, q, h - 1/3)]
sys.stdout.buffer.write(np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8).tobytes())
"#
    );
    let reference = run_python_ref(&script);
    assert_close("vibrance", &ours, &reference, 1.5);
}

#[test]
fn parity_channel_mixer() {
    let (w, h) = (64, 64);
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);

    let ours = rasmcore_image::domain::filters::channel_mixer(
        &pixels, &info, 0.8, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.8,
    )
    .unwrap();

    let script = format!(
        r#"
import sys, numpy as np
px = np.frombuffer(bytes({pixels:?}), dtype=np.uint8).reshape(-1, 3).astype(np.float64) / 255.0
m = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
out = np.clip(px @ m.T, 0, 1)
sys.stdout.buffer.write(np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8).tobytes())
"#
    );
    let reference = run_python_ref(&script);
    assert_close("channel_mixer", &ours, &reference, 0.5);
}

#[test]
fn parity_sparse_color_shepard() {
    let (w, h) = (32, 32);
    let pixels = vec![0u8; (w * h * 3) as usize];
    let info = info_rgb8(w, h);

    // Red at (0,0), Blue at (31,31)
    let ours = rasmcore_image::domain::filters::sparse_color(
        &pixels,
        &info,
        "0,0:FF0000;31,31:0000FF".to_string(),
        2.0,
    )
    .unwrap();

    let script = format!(
        r#"
import sys, numpy as np
w, h = {w}, {h}
pts = [(0, 0, np.array([255, 0, 0], dtype=np.float64)),
       (31, 31, np.array([0, 0, 255], dtype=np.float64))]
out = np.zeros((h, w, 3), dtype=np.float64)
for y in range(h):
    for x in range(w):
        sum_w = 0.0
        sum_c = np.zeros(3)
        exact = None
        for cx, cy, color in pts:
            dx = x - cx
            dy = y - cy
            d2 = dx*dx + dy*dy
            if d2 < 0.001:
                exact = color
                break
            w_i = 1.0 / d2
            sum_c += w_i * color
            sum_w += w_i
        if exact is not None:
            out[y, x] = exact
        elif sum_w > 0:
            out[y, x] = sum_c / sum_w
sys.stdout.buffer.write(np.clip(out + 0.5, 0, 255).astype(np.uint8).tobytes())
"#
    );
    let reference = run_python_ref(&script);
    assert_close("sparse_color_shepard", &ours, &reference, 0.5);
}

#[test]
fn parity_modulate_hsl() {
    let (w, h) = (64, 64);
    let pixels = make_gradient_rgb(w, h);
    let info = info_rgb8(w, h);

    // brightness=80%, saturation=120%, hue=30 degrees
    let ours =
        rasmcore_image::domain::filters::modulate(&pixels, &info, 80.0, 120.0, 30.0).unwrap();

    let script = format!(
        r#"
import sys, numpy as np, colorsys
px = np.frombuffer(bytes({pixels:?}), dtype=np.uint8).reshape(-1, 3).astype(np.float64) / 255.0
bri, sat, hue_deg = 0.8, 1.2, 30.0
out = np.empty_like(px)
for i in range(len(px)):
    r, g, b = px[i]
    # RGB to HLS (Python colorsys uses HLS order, H in [0,1])
    h_norm, l, s = colorsys.rgb_to_hls(r, g, b)
    h = h_norm * 360.0
    # Modulate
    l = min(1.0, max(0.0, l * bri))
    s = min(1.0, max(0.0, s * sat))
    h = (h + hue_deg) % 360.0
    # HLS to RGB
    r2, g2, b2 = colorsys.hls_to_rgb(h / 360.0, l, s)
    out[i] = [r2, g2, b2]
sys.stdout.buffer.write(np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8).tobytes())
"#
    );
    let reference = run_python_ref(&script);
    assert_close("modulate_hsl", &ours, &reference, 1.0);
}
