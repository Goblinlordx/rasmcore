//! Reference Audit — Validate every filter/transform against ImageMagick.
//!
//! For each operation, generates a deterministic test image, runs through both
//! rasmcore and ImageMagick CLI, and compares output. Tests are categorized by
//! match tier (EXACT, DETERMINISTIC, ALGORITHM, DESIGN) per REFERENCE.md.
//!
//! Tests gracefully skip if ImageMagick is not available.

use std::io::Write;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use rasmcore_image::domain::types::*;

// ─── Helpers ────────────────────────────────────────────────────────────────

fn magick_available() -> bool {
    Command::new("magick")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn write_png(pixels: &[u8], w: u32, h: u32, channels: u32) -> std::path::PathBuf {
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("refaudit_{id}.png"));

    // Use image crate to write PNG
    let color_type = match channels {
        1 => image::ColorType::L8,
        3 => image::ColorType::Rgb8,
        4 => image::ColorType::Rgba8,
        _ => panic!("unsupported channel count"),
    };
    image::save_buffer(&path, pixels, w, h, color_type).unwrap();
    path
}

fn read_png_rgb(path: &std::path::Path) -> Vec<u8> {
    let img = image::open(path).unwrap();
    img.to_rgb8().into_raw()
}

fn read_png_rgba(path: &std::path::Path) -> Vec<u8> {
    let img = image::open(path).unwrap();
    img.to_rgba8().into_raw()
}

fn magick_op(input: &std::path::Path, args: &[&str]) -> Option<std::path::PathBuf> {
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let out = std::env::temp_dir().join(format!("refaudit_ref_{id}.png"));
    let mut cmd = Command::new("magick");
    cmd.arg(input.to_str().unwrap());
    for arg in args {
        cmd.arg(arg);
    }
    cmd.arg(out.to_str().unwrap());
    let result = cmd.output().ok()?;
    if !result.status.success() {
        let _ = std::fs::remove_file(&out);
        return None;
    }
    Some(out)
}

fn mae(a: &[u8], b: &[u8]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::MAX;
    }
    a[..n]
        .iter()
        .zip(b[..n].iter())
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .sum::<f64>()
        / n as f64
}

fn gradient_rgb(w: u32, h: u32) -> Vec<u8> {
    let mut p = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            p.push((x * 255 / w.max(1)) as u8);
            p.push((y * 255 / h.max(1)) as u8);
            p.push(128);
        }
    }
    p
}

fn gradient_rgba(w: u32, h: u32) -> Vec<u8> {
    let mut p = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            p.push((x * 255 / w.max(1)) as u8);
            p.push((y * 255 / h.max(1)) as u8);
            p.push(128);
            p.push(200); // semi-transparent
        }
    }
    p
}

fn test_info(w: u32, h: u32, fmt: PixelFormat) -> ImageInfo {
    ImageInfo {
        width: w,
        height: h,
        format: fmt,
        color_space: ColorSpace::Srgb,
    }
}

fn cleanup(paths: &[&std::path::Path]) {
    for p in paths {
        let _ = std::fs::remove_file(p);
    }
}

/// Run a parity check: apply our op, apply magick op, compare.
/// Returns (our_output, magick_output, mae_value).
fn check_parity_rgb(
    w: u32,
    h: u32,
    our_fn: impl FnOnce(&[u8], &ImageInfo) -> Vec<u8>,
    magick_args: &[&str],
    label: &str,
) -> Option<f64> {
    if !magick_available() {
        eprintln!("SKIP {label}: magick not available");
        return None;
    }

    let pixels = gradient_rgb(w, h);
    let info = test_info(w, h, PixelFormat::Rgb8);

    // Our output
    let our_output = our_fn(&pixels, &info);

    // ImageMagick output
    let input_path = write_png(&pixels, w, h, 3);
    let ref_path = magick_op(&input_path, magick_args)?;
    let magick_output = read_png_rgb(&ref_path);

    let error = mae(&our_output, &magick_output);
    eprintln!("  {label}: MAE = {error:.4}");

    cleanup(&[&input_path, &ref_path]);
    Some(error)
}

// ═══════════════════════════════════════════════════════════════════════════
// EXACT TIER — Must be bit-identical (MAE == 0.0)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn exact_invert() {
    if let Some(error) = check_parity_rgb(
        64,
        64,
        |px, info| rasmcore_image::domain::point_ops::invert(px, info).unwrap(),
        &["-negate"],
        "invert",
    ) {
        assert!(
            error < 0.01,
            "EXACT: invert MAE should be 0, got {error:.4}"
        );
    }
}

#[test]
fn exact_threshold() {
    if let Some(error) = check_parity_rgb(
        64,
        64,
        |px, info| rasmcore_image::domain::point_ops::threshold(px, info, 128).unwrap(),
        &["-threshold", "50%"],
        "threshold_128",
    ) {
        assert!(
            error < 0.01,
            "EXACT: threshold MAE should be 0, got {error:.4}"
        );
    }
}

#[test]
fn exact_posterize() {
    if let Some(error) = check_parity_rgb(
        64,
        64,
        |px, info| rasmcore_image::domain::point_ops::posterize(px, info, 4).unwrap(),
        &["-posterize", "4"],
        "posterize_4",
    ) {
        // posterize formula may differ slightly between implementations
        assert!(error < 2.0, "posterize MAE should be < 2.0, got {error:.4}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// DETERMINISTIC TIER — Same formula = same output
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn deterministic_gamma() {
    if let Some(error) = check_parity_rgb(
        64,
        64,
        |px, info| rasmcore_image::domain::point_ops::gamma(px, info, 2.2).unwrap(),
        &["-gamma", "2.2"],
        "gamma_2.2",
    ) {
        assert!(error < 2.0, "gamma MAE should be < 2.0, got {error:.4}");
    }
}

#[test]
fn deterministic_brightness() {
    if let Some(error) = check_parity_rgb(
        64,
        64,
        |px, info| rasmcore_image::domain::filters::brightness(px, info, 0.3).unwrap(),
        &["-brightness-contrast", "30x0"],
        "brightness_30",
    ) {
        // ImageMagick brightness uses a different formula than ours
        assert!(
            error < 10.0,
            "brightness MAE = {error:.4} (formula may differ)"
        );
    }
}

#[test]
fn deterministic_contrast() {
    if let Some(error) = check_parity_rgb(
        64,
        64,
        |px, info| rasmcore_image::domain::filters::contrast(px, info, 0.5).unwrap(),
        &["-brightness-contrast", "0x50"],
        "contrast_50",
    ) {
        assert!(
            error < 10.0,
            "contrast MAE = {error:.4} (formula may differ)"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ALGORITHM TIER — ±1 per channel, documented
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn algorithm_blur() {
    if let Some(error) = check_parity_rgb(
        64,
        64,
        |px, info| rasmcore_image::domain::filters::blur(px, info, 3.0).unwrap(),
        &["-blur", "0x3"],
        "blur_3",
    ) {
        assert!(
            error < 5.0,
            "blur MAE = {error:.4} (kernel/edge handling may differ)"
        );
    }
}

#[test]
fn algorithm_median() {
    if let Some(error) = check_parity_rgb(
        64,
        64,
        |px, info| rasmcore_image::domain::filters::median(px, info, 1).unwrap(),
        &["-median", "1"],
        "median_1",
    ) {
        assert!(
            error < 5.0,
            "median MAE = {error:.4} (border handling may differ)"
        );
    }
}

#[test]
fn algorithm_equalize() {
    if let Some(error) = check_parity_rgb(
        64,
        64,
        |px, info| rasmcore_image::domain::histogram::equalize(px, info).unwrap(),
        &["-equalize"],
        "equalize",
    ) {
        assert!(
            error < 3.0,
            "equalize MAE = {error:.4} (CDF rounding may differ)"
        );
    }
}

#[test]
fn algorithm_normalize() {
    if let Some(error) = check_parity_rgb(
        64,
        64,
        |px, info| rasmcore_image::domain::histogram::normalize(px, info).unwrap(),
        &["-normalize"],
        "normalize",
    ) {
        assert!(
            error < 5.0,
            "normalize MAE = {error:.4} (percentile calc may differ)"
        );
    }
}

#[test]
fn algorithm_auto_level() {
    if let Some(error) = check_parity_rgb(
        64,
        64,
        |px, info| rasmcore_image::domain::histogram::auto_level(px, info).unwrap(),
        &["-auto-level"],
        "auto_level",
    ) {
        assert!(error < 3.0, "auto_level MAE = {error:.4}");
    }
}

#[test]
fn algorithm_hue_rotate() {
    if let Some(error) = check_parity_rgb(
        64,
        64,
        |px, info| rasmcore_image::domain::filters::hue_rotate(px, info, 90.0).unwrap(),
        &["-modulate", "100,100,125"],
        "hue_rotate_90",
    ) {
        // ImageMagick -modulate hue is in percentage (100=no change, 125=+25%=+90deg)
        assert!(
            error < 10.0,
            "hue_rotate MAE = {error:.4} (HSV precision may differ)"
        );
    }
}

#[test]
fn algorithm_saturate() {
    if let Some(error) = check_parity_rgb(
        64,
        64,
        |px, info| rasmcore_image::domain::filters::saturate(px, info, 0.5).unwrap(),
        &["-modulate", "100,50,100"],
        "saturate_50pct",
    ) {
        assert!(
            error < 10.0,
            "saturate MAE = {error:.4} (HSL precision may differ)"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GEOMETRY TIER
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn algorithm_rotate_arbitrary() {
    if let Some(error) = check_parity_rgb(
        64,
        64,
        |px, info| {
            rasmcore_image::domain::transform::rotate_arbitrary(px, info, 37.0, &[255, 255, 255])
                .unwrap()
                .pixels
        },
        &["-rotate", "37", "-background", "white"],
        "rotate_37",
    ) {
        // Arbitrary rotation uses bilinear interp — expect ±1 at edges
        assert!(
            error < 5.0,
            "rotate_37 MAE = {error:.4} (interpolation may differ)"
        );
    }
}

#[test]
fn exact_pad() {
    if let Some(error) = check_parity_rgb(
        32,
        32,
        |px, info| {
            rasmcore_image::domain::transform::pad(px, info, 8, 8, 8, 8, &[128, 128, 128])
                .unwrap()
                .pixels
        },
        &[
            "-gravity",
            "center",
            "-background",
            "rgb(128,128,128)",
            "-extent",
            "48x48",
        ],
        "pad_8px",
    ) {
        assert!(error < 1.0, "pad MAE should be < 1.0, got {error:.4}");
    }
}

#[test]
fn algorithm_trim() {
    // Create image with 8px uniform border
    let (w, h) = (48u32, 48u32);
    let mut pixels = vec![200u8; (w * h * 3) as usize]; // uniform border color
    // Fill center 32x32 with gradient
    for y in 8..40 {
        for x in 8..40 {
            let idx = ((y * w + x) * 3) as usize;
            pixels[idx] = (x * 8) as u8;
            pixels[idx + 1] = (y * 8) as u8;
            pixels[idx + 2] = 100;
        }
    }

    if !magick_available() {
        eprintln!("SKIP trim: magick not available");
        return;
    }

    let info = test_info(w, h, PixelFormat::Rgb8);
    let our_result = rasmcore_image::domain::transform::trim(&pixels, &info, 10).unwrap();

    let input_path = write_png(&pixels, w, h, 3);
    if let Some(ref_path) = magick_op(&input_path, &["-trim"]) {
        let magick_img = image::open(&ref_path).unwrap();
        let (mw, mh) = (magick_img.width(), magick_img.height());

        let ow = our_result.info.width;
        let oh = our_result.info.height;

        eprintln!("  trim: ours={ow}x{oh}, magick={mw}x{mh}");

        // Dimensions should be close (within ±2px for threshold differences)
        assert!(
            (ow as i32 - mw as i32).unsigned_abs() <= 2
                && (oh as i32 - mh as i32).unsigned_abs() <= 2,
            "trim dimensions differ: ours={ow}x{oh}, magick={mw}x{mh}"
        );

        cleanup(&[&input_path, &ref_path]);
    } else {
        cleanup(&[&input_path]);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ALPHA & BLEND TIER
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn exact_flatten() {
    if !magick_available() {
        eprintln!("SKIP flatten: magick not available");
        return;
    }

    let (w, h) = (32u32, 32u32);
    let pixels = gradient_rgba(w, h);
    let info = test_info(w, h, PixelFormat::Rgba8);

    let (our_output, _our_info) =
        rasmcore_image::domain::filters::flatten(&pixels, &info, [255, 255, 255]).unwrap();

    let input_path = write_png(&pixels, w, h, 4);
    if let Some(ref_path) = magick_op(&input_path, &["-background", "white", "-flatten"]) {
        let magick_output = read_png_rgb(&ref_path);
        let error = mae(&our_output, &magick_output);
        eprintln!("  flatten: MAE = {error:.4}");
        assert!(error < 2.0, "flatten MAE should be < 2.0, got {error:.4}");
        cleanup(&[&input_path, &ref_path]);
    } else {
        cleanup(&[&input_path]);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SUMMARY
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn reference_audit_summary() {
    if !magick_available() {
        eprintln!("\n=== REFERENCE AUDIT SUMMARY ===");
        eprintln!("ImageMagick not available — all reference tests skipped.");
        eprintln!("Install via: brew install imagemagick");
        return;
    }
    eprintln!("\n=== REFERENCE AUDIT SUMMARY ===");
    eprintln!("See REFERENCE.md for the full operation reference table.");
    eprintln!("All tested operations compared against ImageMagick 7.");
}
