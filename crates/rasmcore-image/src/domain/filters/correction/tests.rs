//! Tests for lens correction filters.

use crate::domain::filters::common::*;
use crate::domain::types::ColorSpace;

fn rgb_info(w: u32, h: u32) -> ImageInfo {
    ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Rgb8,
        color_space: ColorSpace::Srgb,
    }
}

fn gradient_rgb(w: u32, h: u32) -> Vec<u8> {
    let mut px = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            px.push(((x * 255) / w.max(1)) as u8);
            px.push(((y * 255) / h.max(1)) as u8);
            px.push(128u8);
        }
    }
    px
}

// ── Registration ──

#[test]
fn correction_filters_registered() {
    let filters = crate::domain::filter_registry::registered_filters();
    let names: Vec<&str> = filters.iter().map(|f| f.name).collect();
    assert!(names.contains(&"ca_remove"), "ca_remove should be registered");
    assert!(names.contains(&"red_eye_remove"), "red_eye_remove should be registered");
}

// ── CA Remove ──

#[test]
fn ca_remove_zero_shift_is_identity() {
    let pixels = gradient_rgb(32, 32);
    let info = rgb_info(32, 32);
    let r = Rect::new(0, 0, 32, 32);
    let mut u = |_: Rect| Ok(pixels.clone());
    let result = super::ca_remove(r, &mut u, &info, &super::CaRemoveParams {
        red_shift: 0.0,
        blue_shift: 0.0,
    }).unwrap();
    assert_eq!(result, pixels);
}

#[test]
fn ca_remove_output_size() {
    let pixels = gradient_rgb(64, 64);
    let info = rgb_info(64, 64);
    let r = Rect::new(0, 0, 64, 64);
    let mut u = |_: Rect| Ok(pixels.clone());
    let result = super::ca_remove(r, &mut u, &info, &super::CaRemoveParams {
        red_shift: 0.005,
        blue_shift: -0.003,
    }).unwrap();
    assert_eq!(result.len(), 64 * 64 * 3);
}

#[test]
fn ca_remove_green_unchanged() {
    let pixels = gradient_rgb(64, 64);
    let info = rgb_info(64, 64);
    let r = Rect::new(0, 0, 64, 64);
    let mut u = |_: Rect| Ok(pixels.clone());
    let result = super::ca_remove(r, &mut u, &info, &super::CaRemoveParams {
        red_shift: 0.005,
        blue_shift: -0.003,
    }).unwrap();
    // Green channel should be identical
    for i in (1..pixels.len()).step_by(3) {
        assert_eq!(result[i], pixels[i], "green channel should be unchanged at byte {i}");
    }
}

#[test]
fn ca_remove_synthetic_ca_correction() {
    // Create an image with synthetic CA: R shifted outward, B shifted inward
    // Then apply ca_remove to correct it. The corrected image should be
    // closer to the original than the CA'd version.
    let w = 64u32;
    let h = 64u32;
    let original = gradient_rgb(w, h);
    let info = rgb_info(w, h);

    // Apply CA effect (from our chromatic_aberration filter)
    let r_ca = Rect::new(0, 0, w, h);
    let mut u_ca = |_: Rect| Ok(original.clone());
    let ca_image = crate::domain::filters::effect::chromatic_aberration(
        r_ca, &mut u_ca, &info,
        &crate::domain::filters::effect::ChromaticAberrationParams { strength: 3.0 },
    ).unwrap();

    // Apply CA removal with similar magnitude
    let r_fix = Rect::new(0, 0, w, h);
    let mut u_fix = |_: Rect| Ok(ca_image.clone());
    let corrected = super::ca_remove(r_fix, &mut u_fix, &info, &super::CaRemoveParams {
        red_shift: -0.004,
        blue_shift: 0.004,
    }).unwrap();

    // Corrected should be closer to original than CA'd version
    let ca_error: u64 = original.iter().zip(ca_image.iter())
        .map(|(&a, &b)| (a as i64 - b as i64).unsigned_abs())
        .sum();
    let corrected_error: u64 = original.iter().zip(corrected.iter())
        .map(|(&a, &b)| (a as i64 - b as i64).unsigned_abs())
        .sum();
    assert!(
        corrected_error < ca_error,
        "CA removal should reduce error: ca_error={ca_error}, corrected_error={corrected_error}"
    );
}

// ── Red Eye Remove ──

#[test]
fn red_eye_remove_output_size() {
    let pixels = gradient_rgb(64, 64);
    let info = rgb_info(64, 64);
    let r = Rect::new(0, 0, 64, 64);
    let mut u = |_: Rect| Ok(pixels.clone());
    let result = super::red_eye_remove(r, &mut u, &info, &super::RedEyeRemoveParams {
        center_x: 32, center_y: 32, radius: 10, darken: 0.5, threshold: 0.3,
    }).unwrap();
    assert_eq!(result.len(), 64 * 64 * 3);
}

#[test]
fn red_eye_desaturates_red_pixels() {
    // Create an image with a red circle (simulating red-eye)
    let w = 32u32;
    let h = 32u32;
    let cx = 16f32;
    let cy = 16f32;
    let mut pixels = vec![128u8; (w * h * 3) as usize]; // gray background
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            if (dx * dx + dy * dy).sqrt() <= 8.0 {
                let idx = (y * w + x) as usize * 3;
                pixels[idx] = 220;   // high R
                pixels[idx + 1] = 30; // low G
                pixels[idx + 2] = 30; // low B
            }
        }
    }

    let info = rgb_info(w, h);
    let r = Rect::new(0, 0, w, h);
    let mut u = |_: Rect| Ok(pixels.clone());
    let result = super::red_eye_remove(r, &mut u, &info, &super::RedEyeRemoveParams {
        center_x: 16, center_y: 16, radius: 10, darken: 0.3, threshold: 0.2,
    }).unwrap();

    // Check that the center red pixel was desaturated
    let center_idx = (16 * w + 16) as usize * 3;
    let r_val = result[center_idx];
    let g_val = result[center_idx + 1];
    let b_val = result[center_idx + 2];
    // After desaturation, R should be much closer to G and B
    let orig_spread = 220u32 - 30;
    let new_spread = r_val.max(g_val).max(b_val) as u32 - r_val.min(g_val).min(b_val) as u32;
    assert!(
        new_spread < orig_spread / 2,
        "red-eye should be desaturated: orig_spread={orig_spread}, new_spread={new_spread}"
    );
}

#[test]
fn red_eye_leaves_non_red_unchanged() {
    // Blue pixel should not be affected
    let w = 16u32;
    let h = 16u32;
    let mut pixels = vec![0u8; (w * h * 3) as usize];
    // Fill with blue
    for i in (0..pixels.len()).step_by(3) {
        pixels[i] = 30;
        pixels[i + 1] = 30;
        pixels[i + 2] = 220;
    }

    let info = rgb_info(w, h);
    let r = Rect::new(0, 0, w, h);
    let mut u = |_: Rect| Ok(pixels.clone());
    let result = super::red_eye_remove(r, &mut u, &info, &super::RedEyeRemoveParams {
        center_x: 8, center_y: 8, radius: 20, darken: 0.5, threshold: 0.3,
    }).unwrap();
    assert_eq!(result, pixels, "blue pixels should be unchanged by red-eye removal");
}

// ── GPU ops ──

#[test]
fn ca_remove_gpu_ops_generated() {
    let params = super::CaRemoveParams { red_shift: 0.005, blue_shift: -0.003 };
    use rasmcore_pipeline::GpuCapable;
    let ops = params.gpu_ops(100, 100).unwrap();
    assert_eq!(ops.len(), 1);
}

#[test]
fn ca_remove_gpu_none_at_zero() {
    let params = super::CaRemoveParams { red_shift: 0.0, blue_shift: 0.0 };
    use rasmcore_pipeline::GpuCapable;
    assert!(params.gpu_ops(100, 100).is_none());
}

#[test]
fn red_eye_gpu_ops_generated() {
    let params = super::RedEyeRemoveParams {
        center_x: 50, center_y: 50, radius: 10, darken: 0.5, threshold: 0.3,
    };
    use rasmcore_pipeline::GpuCapable;
    let ops = params.gpu_ops(100, 100).unwrap();
    assert_eq!(ops.len(), 1);
}
