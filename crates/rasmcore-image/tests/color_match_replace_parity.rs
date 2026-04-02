//! Parity tests for match_color and replace_color filters.
//!
//! match_color is validated against a numpy LAB mean/std transfer reference.
//! replace_color is validated with targeted hue shift behavior.
//!
//! Setup:
//!   python3 -m venv tests/fixtures/.venv
//!   tests/fixtures/.venv/bin/pip install numpy Pillow opencv-python-headless

use rasmcore_image::domain::filters;
use rasmcore_image::domain::pipeline::Rect;
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
        script.lines().take(10).collect::<Vec<_>>().join("\n"),
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

fn mean_absolute_error(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "buffer length mismatch: {} vs {}", a.len(), b.len());
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

fn info_rgb8(w: u32, h: u32) -> ImageInfo {
    ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Rgb8,
        color_space: ColorSpace::Srgb,
    }
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

// ═══════════════════════════════════════════════════════════════════════════
// match_color parity against numpy LAB statistics transfer
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn match_color_parity_numpy_lab_transfer() {
    let w = 16u32;
    let h = 16;
    // Target: gradient image
    let target = make_gradient_rgb(w, h);
    // Reference: warm-shifted gradient (flip R and B, shift green)
    let reference: Vec<u8> = (0..w * h)
        .flat_map(|i| {
            let x = (i % w) as u8;
            let y = (i / w) as u8;
            [
                200u8.saturating_add(x / 8),
                100u8.saturating_add(y / 4),
                50,
            ]
        })
        .collect();

    let target_info = info_rgb8(w, h);
    let ref_info = info_rgb8(w, h);

    let ours = filters::match_color(&target, &target_info, &reference, &ref_info, 1.0).unwrap();

    // Python reference: Reinhard 2001 LAB transfer using numpy + skimage-style conversion
    let script = format!(
        r#"
import sys
import numpy as np

def srgb_to_linear(v):
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(v):
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(np.maximum(v, 0), 1.0/2.4) - 0.055)

def rgb_to_xyz(rgb):
    lin = srgb_to_linear(rgb)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    return lin @ M.T

D65 = np.array([0.9504559270516716, 1.0, 1.0890577507598784])

def lab_f(t):
    delta = 6.0 / 29.0
    return np.where(t > delta**3, np.cbrt(t), t / (3 * delta**2) + 4.0/29.0)

def lab_f_inv(t):
    delta = 6.0 / 29.0
    return np.where(t > delta, t**3, 3 * delta**2 * (t - 4.0/29.0))

def xyz_to_lab(xyz):
    scaled = xyz / D65
    f = lab_f(scaled)
    L = 116.0 * f[:, 1] - 16.0
    a = 500.0 * (f[:, 0] - f[:, 1])
    b = 200.0 * (f[:, 1] - f[:, 2])
    return np.stack([L, a, b], axis=1)

def lab_to_xyz(lab):
    fy = (lab[:, 0] + 16.0) / 116.0
    fx = lab[:, 1] / 500.0 + fy
    fz = fy - lab[:, 2] / 200.0
    x = D65[0] * lab_f_inv(fx)
    y = D65[1] * lab_f_inv(fy)
    z = D65[2] * lab_f_inv(fz)
    return np.stack([x, y, z], axis=1)

def xyz_to_rgb(xyz):
    M_inv = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ])
    lin = xyz @ M_inv.T
    return linear_to_srgb(np.clip(lin, 0, None))

target = np.array({target:?}, dtype=np.uint8).reshape(-1, 3).astype(np.float64) / 255.0
ref_img = np.array({reference:?}, dtype=np.uint8).reshape(-1, 3).astype(np.float64) / 255.0

# Convert to LAB
target_lab = xyz_to_lab(rgb_to_xyz(target))
ref_lab = xyz_to_lab(rgb_to_xyz(ref_img))

# Compute statistics (population std)
t_mean = target_lab.mean(axis=0)
t_std = target_lab.std(axis=0)
r_mean = ref_lab.mean(axis=0)
r_std = ref_lab.std(axis=0)

# Reinhard transfer
scale = np.where(t_std > 1e-10, r_std / t_std, 1.0)
result_lab = (target_lab - t_mean) * scale + r_mean

# Convert back to RGB
result_xyz = lab_to_xyz(result_lab)
result_rgb = xyz_to_rgb(result_xyz)
result_u8 = np.clip(result_rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)

sys.stdout.buffer.write(result_u8.tobytes())
"#
    );

    let reference_result = run_python_ref(&script);

    let mae = mean_absolute_error(&ours, &reference_result);
    let max_err = max_absolute_error(&ours, &reference_result);
    eprintln!("  match_color LAB parity: MAE={mae:.4}, max_err={max_err}");
    assert!(
        mae <= 1.0,
        "match_color parity: MAE={mae:.4} exceeds 1.0, max_err={max_err}"
    );
    assert!(
        max_err <= 2,
        "match_color parity: max_err={max_err} exceeds 2"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// replace_color targeted hue shift test
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn replace_color_targeted_hue_shift() {
    // Build a test image: 4 red pixels, 4 green pixels, 4 blue pixels
    let pixels: Vec<u8> = [
        [255u8, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], // red row
        [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0],     // green row
        [0, 0, 255], [0, 0, 255], [0, 0, 255], [0, 0, 255],     // blue row
    ]
    .concat();
    let info = ImageInfo {
        width: 4,
        height: 3,
        format: PixelFormat::Rgb8,
        color_space: ColorSpace::Srgb,
    };
    let config = filters::ReplaceColorParams {
        center_hue: 0.0,
        hue_range: 60.0,
        sat_min: 0.0,
        sat_max: 1.0,
        lum_min: 0.0,
        lum_max: 1.0,
        hue_shift: 120.0,
        sat_shift: 0.0,
        lum_shift: 0.0,
    };
    let result = filters::replace_color(
        Rect::new(0, 0, info.width, info.height),
        &mut |_| Ok(pixels.clone()),
        &info,
        &config,
    )
    .unwrap();

    // Red pixels (first 12 bytes) should have shifted to green (hue 0 + 120 = 120)
    for i in 0..4 {
        let pi = i * 3;
        assert!(
            result[pi + 1] > result[pi],
            "red pixel {i} should shift green: R={}, G={}",
            result[pi],
            result[pi + 1]
        );
    }

    // Green pixels (bytes 12-23) should be UNCHANGED (hue ~120, outside range)
    for i in 4..8 {
        let pi = i * 3;
        assert_eq!(result[pi], 0, "green pixel {}: R should stay 0, got {}", i, result[pi]);
        assert_eq!(result[pi + 1], 255, "green pixel {}: G should stay 255, got {}", i, result[pi + 1]);
        assert_eq!(result[pi + 2], 0, "green pixel {}: B should stay 0, got {}", i, result[pi + 2]);
    }

    // Blue pixels (bytes 24-35) should be UNCHANGED (hue ~240, outside range)
    for i in 8..12 {
        let pi = i * 3;
        assert_eq!(result[pi], 0, "blue pixel {}: R should stay 0, got {}", i, result[pi]);
        assert_eq!(result[pi + 1], 0, "blue pixel {}: G should stay 0, got {}", i, result[pi + 1]);
        assert_eq!(result[pi + 2], 255, "blue pixel {}: B should stay 255, got {}", i, result[pi + 2]);
    }

    eprintln!("  replace_color targeted hue shift: all assertions passed");
}
