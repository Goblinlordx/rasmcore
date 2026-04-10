//! f32 filter wave 1 verification tests.
//!
//! For each migrated filter:
//! 1. f32 input produces f32 output (no format downgrade)
//! 2. f32 result matches u8 result within precision bounds
//!    (f32 should be equal or better — no quantization artifacts)

use rasmcore_image::domain::types::{ColorSpace, ImageInfo, PixelFormat};

fn make_info(w: u32, h: u32, fmt: PixelFormat) -> ImageInfo {
    ImageInfo {
        width: w,
        height: h,
        format: fmt,
        color_space: ColorSpace::Srgb,
    }
}

/// Create a test gradient as u8 RGB pixels (3 bytes per pixel).
fn gradient_rgb8(w: u32, h: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            pixels.push(((x * 255) / w.max(1)) as u8);
            pixels.push(((y * 255) / h.max(1)) as u8);
            pixels.push(128u8);
        }
    }
    pixels
}

/// Convert u8 RGB to f32 RGB (Rgb32f format) — 12 bytes per pixel.
fn u8_to_f32_rgb(pixels: &[u8]) -> Vec<u8> {
    pixels
        .iter()
        .flat_map(|&v| (v as f32 / 255.0).to_le_bytes())
        .collect()
}

/// Read f32 samples from byte buffer.
fn read_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Convert f32 samples back to u8 for comparison.
fn f32_to_u8_samples(f32_bytes: &[u8]) -> Vec<u8> {
    read_f32(f32_bytes)
        .iter()
        .map(|&v| (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8)
        .collect()
}

/// Helper: run a filter on u8 and f32 versions of same data, compare results.
/// Returns (max_diff, mean_diff) between f32→u8 and native u8 results.
fn compare_u8_f32<F>(w: u32, h: u32, filter_fn: F) -> (u8, f32)
where
    F: Fn(&[u8], &ImageInfo) -> Vec<u8>,
{
    let u8_pixels = gradient_rgb8(w, h);
    let f32_pixels = u8_to_f32_rgb(&u8_pixels);

    let info_u8 = make_info(w, h, PixelFormat::Rgb8);
    let info_f32 = make_info(w, h, PixelFormat::Rgb32f);

    let result_u8 = filter_fn(&u8_pixels, &info_u8);
    let result_f32 = filter_fn(&f32_pixels, &info_f32);

    // f32 output should still be f32 sized (12 bytes per pixel)
    let expected_f32_len = (w * h * 3) as usize * 4;
    assert_eq!(
        result_f32.len(),
        expected_f32_len,
        "f32 output should be f32 format ({} bytes), got {}",
        expected_f32_len,
        result_f32.len()
    );

    // Convert f32 result to u8 for comparison
    let result_f32_as_u8 = f32_to_u8_samples(&result_f32);
    assert_eq!(result_u8.len(), result_f32_as_u8.len());

    let mut max_diff = 0u8;
    let mut total_diff = 0u64;
    for (a, b) in result_u8.iter().zip(result_f32_as_u8.iter()) {
        let diff = (*a as i16 - *b as i16).unsigned_abs() as u8;
        if diff > max_diff {
            max_diff = diff;
        }
        total_diff += diff as u64;
    }
    let mean_diff = total_diff as f32 / result_u8.len() as f32;
    (max_diff, mean_diff)
}

// ── Point Op Filters ────────────────────────────────────────────────────

#[test]
fn f32_brightness_parity() {
    let (max_diff, mean_diff) = compare_u8_f32(64, 64, |px, info| {
        use rasmcore_image::domain::point_ops::{PointOp, apply_op};
        apply_op(px, info, &PointOp::Brightness(0.2)).unwrap()
    });
    assert!(max_diff <= 1, "brightness max_diff={max_diff}");
    assert!(mean_diff < 0.5, "brightness mean_diff={mean_diff}");
}

#[test]
fn f32_contrast_parity() {
    let (max_diff, mean_diff) = compare_u8_f32(64, 64, |px, info| {
        use rasmcore_image::domain::point_ops::{PointOp, apply_op};
        apply_op(px, info, &PointOp::Contrast(0.5)).unwrap()
    });
    assert!(max_diff <= 1, "contrast max_diff={max_diff}");
    // Contrast midpoint 128/255 vs 0.5 causes ≤1 LSB rounding diff on most pixels
    assert!(mean_diff < 1.0, "contrast mean_diff={mean_diff}");
}

#[test]
fn f32_gamma_parity() {
    let (max_diff, mean_diff) = compare_u8_f32(64, 64, |px, info| {
        use rasmcore_image::domain::point_ops::{PointOp, apply_op};
        apply_op(px, info, &PointOp::Gamma(2.2)).unwrap()
    });
    assert!(max_diff <= 1, "gamma max_diff={max_diff}");
    assert!(mean_diff < 0.5, "gamma mean_diff={mean_diff}");
}

#[test]
fn f32_exposure_parity() {
    let (max_diff, mean_diff) = compare_u8_f32(64, 64, |px, info| {
        use rasmcore_image::domain::point_ops::{PointOp, apply_op};
        apply_op(
            px,
            info,
            &PointOp::Exposure {
                ev: 1.0,
                offset: 0.0,
                gamma_correction: 1.0,
            },
        )
        .unwrap()
    });
    assert!(max_diff <= 1, "exposure max_diff={max_diff}");
    assert!(mean_diff < 0.5, "exposure mean_diff={mean_diff}");
}

#[test]
fn f32_invert_parity() {
    let (max_diff, _) = compare_u8_f32(64, 64, |px, info| {
        use rasmcore_image::domain::point_ops::{PointOp, apply_op};
        apply_op(px, info, &PointOp::Invert).unwrap()
    });
    assert!(max_diff <= 1, "invert max_diff={max_diff}");
}

#[test]
fn f32_levels_parity() {
    let (max_diff, mean_diff) = compare_u8_f32(64, 64, |px, info| {
        use rasmcore_image::domain::point_ops::{PointOp, apply_op};
        apply_op(
            px,
            info,
            &PointOp::Levels {
                black: 0.1,
                white: 0.9,
                gamma: 1.2,
            },
        )
        .unwrap()
    });
    assert!(max_diff <= 1, "levels max_diff={max_diff}");
    assert!(mean_diff < 0.5, "levels mean_diff={mean_diff}");
}

#[test]
fn f32_posterize_parity() {
    let (max_diff, _) = compare_u8_f32(64, 64, |px, info| {
        use rasmcore_image::domain::point_ops::{PointOp, apply_op};
        apply_op(px, info, &PointOp::Posterize(4)).unwrap()
    });
    // Posterize with few levels can have larger diffs at quantization boundaries
    assert!(max_diff <= 2, "posterize max_diff={max_diff}");
}

#[test]
fn f32_solarize_parity() {
    let (max_diff, _) = compare_u8_f32(64, 64, |px, info| {
        use rasmcore_image::domain::point_ops::{PointOp, apply_op};
        apply_op(px, info, &PointOp::Solarize(128)).unwrap()
    });
    assert!(max_diff <= 1, "solarize max_diff={max_diff}");
}

// ── Evaluate Filters ────────────────────────────────────────────────────

#[test]
fn f32_eval_add_parity() {
    let (max_diff, _) = compare_u8_f32(64, 64, |px, info| {
        use rasmcore_image::domain::point_ops::{PointOp, apply_op};
        apply_op(px, info, &PointOp::EvalAdd(50)).unwrap()
    });
    assert!(max_diff <= 1, "eval_add max_diff={max_diff}");
}

#[test]
fn f32_eval_multiply_parity() {
    let (max_diff, _) = compare_u8_f32(64, 64, |px, info| {
        use rasmcore_image::domain::point_ops::{PointOp, apply_op};
        apply_op(px, info, &PointOp::EvalMultiply(1.5)).unwrap()
    });
    assert!(max_diff <= 1, "eval_multiply max_diff={max_diff}");
}

#[test]
fn f32_eval_pow_parity() {
    let (max_diff, _) = compare_u8_f32(64, 64, |px, info| {
        use rasmcore_image::domain::point_ops::{PointOp, apply_op};
        apply_op(px, info, &PointOp::EvalPow(0.5)).unwrap()
    });
    assert!(max_diff <= 1, "eval_pow max_diff={max_diff}");
}

// ── Color Op Filters ────────────────────────────────────────────────────

#[test]
fn f32_hue_rotate_parity() {
    let (max_diff, mean_diff) = compare_u8_f32(64, 64, |px, info| {
        use rasmcore_image::domain::color_lut::ColorOp;
        use rasmcore_image::domain::filters::common::color::apply_color_op;
        apply_color_op(px, info, &ColorOp::HueRotate(90.0)).unwrap()
    });
    assert!(max_diff <= 1, "hue_rotate max_diff={max_diff}");
    assert!(mean_diff < 0.5, "hue_rotate mean_diff={mean_diff}");
}

#[test]
fn f32_saturate_parity() {
    let (max_diff, mean_diff) = compare_u8_f32(64, 64, |px, info| {
        use rasmcore_image::domain::color_lut::ColorOp;
        use rasmcore_image::domain::filters::common::color::apply_color_op;
        apply_color_op(px, info, &ColorOp::Saturate(1.5)).unwrap()
    });
    assert!(max_diff <= 1, "saturate max_diff={max_diff}");
    assert!(mean_diff < 0.5, "saturate mean_diff={mean_diff}");
}

#[test]
fn f32_vibrance_parity() {
    let (max_diff, mean_diff) = compare_u8_f32(64, 64, |px, info| {
        use rasmcore_image::domain::color_lut::ColorOp;
        use rasmcore_image::domain::filters::common::color::apply_color_op;
        apply_color_op(px, info, &ColorOp::Vibrance(0.5)).unwrap()
    });
    assert!(max_diff <= 1, "vibrance max_diff={max_diff}");
    assert!(mean_diff < 0.5, "vibrance mean_diff={mean_diff}");
}

#[test]
fn f32_sepia_parity() {
    let (max_diff, mean_diff) = compare_u8_f32(64, 64, |px, info| {
        use rasmcore_image::domain::color_lut::ColorOp;
        use rasmcore_image::domain::filters::common::color::apply_color_op;
        apply_color_op(px, info, &ColorOp::Sepia(0.8)).unwrap()
    });
    assert!(max_diff <= 1, "sepia max_diff={max_diff}");
    assert!(mean_diff < 0.5, "sepia mean_diff={mean_diff}");
}

// ── RGB Transform Filters ───────────────────────────────────────────────

#[test]
fn f32_tonemap_reinhard_parity() {
    let (max_diff, mean_diff) = compare_u8_f32(64, 64, |px, info| {
        rasmcore_image::domain::color_grading::tonemap_reinhard(px, info).unwrap()
    });
    assert!(max_diff <= 1, "tonemap_reinhard max_diff={max_diff}");
    assert!(mean_diff < 0.5, "tonemap_reinhard mean_diff={mean_diff}");
}

#[test]
fn f32_tonemap_filmic_parity() {
    let (max_diff, mean_diff) = compare_u8_f32(64, 64, |px, info| {
        rasmcore_image::domain::color_grading::tonemap_filmic(px, info, &Default::default())
            .unwrap()
    });
    assert!(max_diff <= 1, "tonemap_filmic max_diff={max_diff}");
    assert!(mean_diff < 0.5, "tonemap_filmic mean_diff={mean_diff}");
}

#[test]
fn f32_tonemap_drago_parity() {
    let (max_diff, mean_diff) = compare_u8_f32(64, 64, |px, info| {
        rasmcore_image::domain::color_grading::tonemap_drago(px, info, &Default::default()).unwrap()
    });
    assert!(max_diff <= 1, "tonemap_drago max_diff={max_diff}");
    assert!(mean_diff < 0.5, "tonemap_drago mean_diff={mean_diff}");
}

#[test]
fn f32_curves_master_parity() {
    let tc = rasmcore_image::domain::color_grading::ToneCurves {
        r: vec![(0.0, 0.0), (0.5, 0.7), (1.0, 1.0)],
        g: vec![(0.0, 0.0), (0.5, 0.7), (1.0, 1.0)],
        b: vec![(0.0, 0.0), (0.5, 0.7), (1.0, 1.0)],
    };
    let (max_diff, mean_diff) = compare_u8_f32(64, 64, |px, info| {
        rasmcore_image::domain::color_grading::curves(px, info, &tc).unwrap()
    });
    // Curves use 256-entry LUT internally, so f32 values get quantized to LUT indices
    assert!(max_diff <= 2, "curves max_diff={max_diff}");
    assert!(mean_diff < 1.0, "curves mean_diff={mean_diff}");
}

// ── Enhancement Filters ─────────────────────────────────────────────────

#[test]
fn f32_dodge_parity() {
    let (max_diff, mean_diff) = compare_u8_f32(64, 64, |px, info| {
        use rasmcore_image::domain::filters::common::blending::dodge_burn_impl;
        dodge_burn_impl(px, info, 0.5, 1, true).unwrap()
    });
    assert!(max_diff <= 1, "dodge max_diff={max_diff}");
    assert!(mean_diff < 0.5, "dodge mean_diff={mean_diff}");
}

#[test]
fn f32_burn_parity() {
    let (max_diff, mean_diff) = compare_u8_f32(64, 64, |px, info| {
        use rasmcore_image::domain::filters::common::blending::dodge_burn_impl;
        dodge_burn_impl(px, info, 0.75, 2, false).unwrap()
    });
    assert!(max_diff <= 1, "burn max_diff={max_diff}");
    assert!(mean_diff < 0.5, "burn mean_diff={mean_diff}");
}

// ── Verify f32 output format preservation ───────────────────────────────

#[test]
fn f32_output_preserves_format() {
    use rasmcore_image::domain::point_ops::{PointOp, apply_op};

    let info = make_info(4, 4, PixelFormat::Rgb32f);
    let samples: Vec<f32> = (0..48).map(|i| i as f32 / 48.0).collect();
    let pixels: Vec<u8> = samples.iter().flat_map(|v| v.to_le_bytes()).collect();

    let result = apply_op(&pixels, &info, &PointOp::Brightness(0.1)).unwrap();
    // Output should be f32 format: 4 bytes per sample, 3 samples per pixel, 16 pixels
    assert_eq!(result.len(), 48 * 4, "f32 output should be 48 f32 samples");

    // All values should be valid f32 in [0, 1]
    let out = read_f32(&result);
    for (i, &v) in out.iter().enumerate() {
        assert!(v >= 0.0 && v <= 1.0, "f32 sample {i} out of range: {v}");
    }
}

#[test]
fn f32_rgba32f_alpha_preserved() {
    use rasmcore_image::domain::point_ops::{PointOp, apply_op};

    let info = make_info(2, 1, PixelFormat::Rgba32f);
    // 2 pixels: [0.5, 0.3, 0.7, 0.9] [0.1, 0.8, 0.4, 0.6]
    let samples: Vec<f32> = vec![0.5, 0.3, 0.7, 0.9, 0.1, 0.8, 0.4, 0.6];
    let pixels: Vec<u8> = samples.iter().flat_map(|v| v.to_le_bytes()).collect();

    let result = apply_op(&pixels, &info, &PointOp::Invert).unwrap();
    let out = read_f32(&result);

    // Alpha channels should be unchanged
    assert!(
        (out[3] - 0.9).abs() < 1e-6,
        "alpha pixel 0 changed: {}",
        out[3]
    );
    assert!(
        (out[7] - 0.6).abs() < 1e-6,
        "alpha pixel 1 changed: {}",
        out[7]
    );
}
