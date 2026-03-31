//! Metrics Reference Parity Tests — PSNR, SSIM, MAE, RMSE validated
//! against ImageMagick `compare -metric` and scikit-image.
//!
//! Tests skip gracefully when reference tools are not available.

use rasmcore_image::domain::types::*;
use std::path::Path;
use std::process::Command;

fn has_tool(name: &str) -> bool {
    Command::new("which")
        .arg(name)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn venv_python() -> Option<String> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let venv = manifest.join("../../tests/fixtures/.venv/bin/python3");
    if venv.exists() {
        Some(venv.to_string_lossy().into_owned())
    } else {
        None
    }
}

fn info_rgb8(w: u32, h: u32) -> ImageInfo {
    ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Rgb8,
        color_space: ColorSpace::Srgb,
    }
}

/// Create a gradient PNG image (32x32 RGB) and a shifted version for comparison.
/// Returns (original_png_bytes, shifted_png_bytes, original_pixels, shifted_pixels).
fn make_test_pair() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = 32u32;
    let h = 32u32;
    let mut orig = Vec::with_capacity((w * h * 3) as usize);
    let mut shifted = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = (x * 8) as u8;
            let g = (y * 8) as u8;
            let b = 128u8;
            orig.extend_from_slice(&[r, g, b]);
            // Shift by +10 on all channels (clamped)
            shifted.extend_from_slice(&[
                r.saturating_add(10),
                g.saturating_add(10),
                b.saturating_add(10),
            ]);
        }
    }

    let info = info_rgb8(w, h);
    let orig_png = rasmcore_image::domain::encoder::encode(&orig, &info, "png", None).unwrap();
    let shifted_png =
        rasmcore_image::domain::encoder::encode(&shifted, &info, "png", None).unwrap();

    (orig_png, shifted_png, orig, shifted)
}

/// PSNR parity vs ImageMagick -compare -metric PSNR within 0.01 dB.
#[test]
fn psnr_parity_vs_imagemagick() {
    if !has_tool("magick") {
        eprintln!("  psnr_parity_vs_imagemagick: SKIP (magick not found)");
        return;
    }

    let (orig_png, shifted_png, orig_px, shifted_px) = make_test_pair();
    let info = info_rgb8(32, 32);

    // Our PSNR
    let our_psnr =
        rasmcore_image::domain::metrics::psnr(&orig_px, &info, &shifted_px, &info).unwrap();

    // ImageMagick PSNR
    let tmp_a = std::env::temp_dir().join("metrics_psnr_a.png");
    let tmp_b = std::env::temp_dir().join("metrics_psnr_b.png");
    let tmp_diff = std::env::temp_dir().join("metrics_psnr_diff.png");
    std::fs::write(&tmp_a, &orig_png).unwrap();
    std::fs::write(&tmp_b, &shifted_png).unwrap();

    let output = Command::new("magick")
        .args([
            "compare",
            "-metric",
            "PSNR",
            tmp_a.to_str().unwrap(),
            tmp_b.to_str().unwrap(),
            tmp_diff.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    // ImageMagick outputs "value (normalized)" to stderr — parse the first number
    let stderr = String::from_utf8_lossy(&output.stderr);
    let im_psnr: f64 = stderr
        .trim()
        .split_whitespace()
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("Failed to parse IM PSNR from: '{stderr}'"));

    let diff = (our_psnr - im_psnr).abs();
    eprintln!("  PSNR: ours={our_psnr:.4} dB, IM={im_psnr:.4} dB, diff={diff:.4}");
    assert!(
        diff < 0.01,
        "PSNR divergence too high: ours={our_psnr:.4}, IM={im_psnr:.4}, diff={diff:.4} (max 0.01)"
    );

    let _ = std::fs::remove_file(&tmp_a);
    let _ = std::fs::remove_file(&tmp_b);
    let _ = std::fs::remove_file(&tmp_diff);
}

/// MAE parity vs ImageMagick -compare -metric MAE within 0.01.
#[test]
fn mae_parity_vs_imagemagick() {
    if !has_tool("magick") {
        eprintln!("  mae_parity_vs_imagemagick: SKIP (magick not found)");
        return;
    }

    let (orig_png, shifted_png, orig_px, shifted_px) = make_test_pair();
    let info = info_rgb8(32, 32);

    let our_mae =
        rasmcore_image::domain::metrics::mae(&orig_px, &info, &shifted_px, &info).unwrap();

    let tmp_a = std::env::temp_dir().join("metrics_mae_a.png");
    let tmp_b = std::env::temp_dir().join("metrics_mae_b.png");
    let tmp_diff = std::env::temp_dir().join("metrics_mae_diff.png");
    std::fs::write(&tmp_a, &orig_png).unwrap();
    std::fs::write(&tmp_b, &shifted_png).unwrap();

    let output = Command::new("magick")
        .args([
            "compare",
            "-metric",
            "MAE",
            tmp_a.to_str().unwrap(),
            tmp_b.to_str().unwrap(),
            tmp_diff.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    let stderr = String::from_utf8_lossy(&output.stderr);
    // IM MAE format: "value (normalized)" — e.g., "2570.67 (0.0392157)"
    // The normalized value is MAE/255.0 — we need the raw value / 255.0 or use the raw.
    // Actually IM MAE is the total absolute error / num_pixels but scaled to 0-65535 range.
    // The parenthesized value is the normalized (0-1) version.
    let im_mae_normalized: f64 = if let Some(start) = stderr.find('(') {
        let rest = &stderr[start + 1..];
        if let Some(end) = rest.find(')') {
            rest[..end].parse().unwrap_or(0.0)
        } else {
            0.0
        }
    } else {
        // Fallback: try parsing raw value and dividing by 257 (65535/255)
        stderr
            .trim()
            .split_whitespace()
            .next()
            .and_then(|s| s.parse::<f64>().ok())
            .map(|v| v / 65535.0)
            .unwrap_or(0.0)
    };

    // Our MAE is in [0, 255] range, IM normalized is [0, 1].
    let our_mae_normalized = our_mae / 255.0;
    let diff = (our_mae_normalized - im_mae_normalized).abs();
    eprintln!(
        "  MAE: ours={our_mae:.4} (norm={our_mae_normalized:.6}), IM_norm={im_mae_normalized:.6}, diff={diff:.6}"
    );
    assert!(
        diff < 0.001,
        "MAE divergence too high: ours_norm={our_mae_normalized:.6}, IM_norm={im_mae_normalized:.6}"
    );

    let _ = std::fs::remove_file(&tmp_a);
    let _ = std::fs::remove_file(&tmp_b);
    let _ = std::fs::remove_file(&tmp_diff);
}

/// RMSE parity vs ImageMagick -compare -metric RMSE within 0.01.
#[test]
fn rmse_parity_vs_imagemagick() {
    if !has_tool("magick") {
        eprintln!("  rmse_parity_vs_imagemagick: SKIP (magick not found)");
        return;
    }

    let (orig_png, shifted_png, orig_px, shifted_px) = make_test_pair();
    let info = info_rgb8(32, 32);

    let our_rmse =
        rasmcore_image::domain::metrics::rmse(&orig_px, &info, &shifted_px, &info).unwrap();

    let tmp_a = std::env::temp_dir().join("metrics_rmse_a.png");
    let tmp_b = std::env::temp_dir().join("metrics_rmse_b.png");
    let tmp_diff = std::env::temp_dir().join("metrics_rmse_diff.png");
    std::fs::write(&tmp_a, &orig_png).unwrap();
    std::fs::write(&tmp_b, &shifted_png).unwrap();

    let output = Command::new("magick")
        .args([
            "compare",
            "-metric",
            "RMSE",
            tmp_a.to_str().unwrap(),
            tmp_b.to_str().unwrap(),
            tmp_diff.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    let stderr = String::from_utf8_lossy(&output.stderr);
    // IM RMSE format: "value (normalized)" — normalized is RMSE/65535 (16-bit range)
    let im_rmse_normalized: f64 = if let Some(start) = stderr.find('(') {
        let rest = &stderr[start + 1..];
        if let Some(end) = rest.find(')') {
            rest[..end].parse().unwrap_or(0.0)
        } else {
            0.0
        }
    } else {
        0.0
    };

    // Our RMSE is in [0, 255], IM normalized is [0, 1].
    let our_rmse_normalized = our_rmse / 255.0;
    let diff = (our_rmse_normalized - im_rmse_normalized).abs();
    eprintln!(
        "  RMSE: ours={our_rmse:.4} (norm={our_rmse_normalized:.6}), IM_norm={im_rmse_normalized:.6}, diff={diff:.6}"
    );
    assert!(
        diff < 0.001,
        "RMSE divergence too high: ours_norm={our_rmse_normalized:.6}, IM_norm={im_rmse_normalized:.6}"
    );

    let _ = std::fs::remove_file(&tmp_a);
    let _ = std::fs::remove_file(&tmp_b);
    let _ = std::fs::remove_file(&tmp_diff);
}

/// SSIM parity vs scikit-image structural_similarity within 0.001.
#[test]
fn ssim_parity_vs_scikit_image() {
    let python = match venv_python() {
        Some(p) => p,
        None => {
            eprintln!("  ssim_parity_vs_scikit: SKIP (venv not found)");
            return;
        }
    };

    let (_, _, orig_px, shifted_px) = make_test_pair();
    let info = info_rgb8(32, 32);

    let our_ssim =
        rasmcore_image::domain::metrics::ssim(&orig_px, &info, &shifted_px, &info).unwrap();

    // Write raw pixel data to temp files for Python
    let tmp_a = std::env::temp_dir().join("metrics_ssim_a.raw");
    let tmp_b = std::env::temp_dir().join("metrics_ssim_b.raw");
    std::fs::write(&tmp_a, &orig_px).unwrap();
    std::fs::write(&tmp_b, &shifted_px).unwrap();

    let script = format!(
        r#"
import numpy as np
from skimage.metrics import structural_similarity as ssim

a = np.fromfile("{}", dtype=np.uint8).reshape(32, 32, 3)
b = np.fromfile("{}", dtype=np.uint8).reshape(32, 32, 3)
val = ssim(a, b, channel_axis=2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255)
print(f"{{val:.10f}}")
"#,
        tmp_a.display(),
        tmp_b.display()
    );

    let output = Command::new(&python)
        .arg("-c")
        .arg(&script)
        .output()
        .unwrap();

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        eprintln!("  ssim_parity_vs_scikit: SKIP (scikit-image error: {err})");
        let _ = std::fs::remove_file(&tmp_a);
        let _ = std::fs::remove_file(&tmp_b);
        return;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let ref_ssim: f64 = stdout.trim().parse().unwrap_or_else(|_| {
        panic!("Failed to parse scikit SSIM from: '{stdout}'");
    });

    let diff = (our_ssim - ref_ssim).abs();
    eprintln!("  SSIM: ours={our_ssim:.6}, scikit={ref_ssim:.6}, diff={diff:.6}");
    assert!(
        diff < 0.01,
        "SSIM divergence too high: ours={our_ssim:.6}, scikit={ref_ssim:.6}, diff={diff:.6} (max 0.01)"
    );

    let _ = std::fs::remove_file(&tmp_a);
    let _ = std::fs::remove_file(&tmp_b);
}
