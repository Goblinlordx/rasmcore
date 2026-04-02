//! Tests for mask generators, operations, and masked blend.

use super::*;
use crate::domain::types::{ColorSpace, ImageInfo, PixelFormat};
use rasmcore_pipeline::Rect;

fn info_rgb8(w: u32, h: u32) -> ImageInfo {
    ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Rgb8,
        color_space: ColorSpace::Srgb,
    }
}

fn info_gray8(w: u32, h: u32) -> ImageInfo {
    ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Gray8,
        color_space: ColorSpace::Srgb,
    }
}

fn solid_rgb(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
    (0..(w * h)).flat_map(|_| [r, g, b]).collect()
}

// ── Gradient Linear ─────────────────────────────────────────────────────

#[test]
fn gradient_linear_produces_correct_size() {
    let mask = mask_gradient_linear(64, 32, 0.0, 0.5, 1.0);
    assert_eq!(mask.len(), 64 * 32 * 3);
}

#[test]
fn gradient_linear_horizontal_gradient() {
    let mask = mask_gradient_linear(100, 10, 0.0, 0.5, 1.0);
    // Left side should be darker, right side lighter
    let left = mask[0]; // First pixel R
    let right = mask[(99) * 3]; // Last pixel in first row
    assert!(right > left, "right={right} should be > left={left}");
}

// ── Gradient Radial ─────────────────────────────────────────────────────

#[test]
fn gradient_radial_center_is_white() {
    let mask = mask_gradient_radial(64, 64, 0.5, 0.5, 0.5, 0.5, 0.5);
    // Center pixel (32, 32)
    let center_idx = (32 * 64 + 32) * 3;
    assert!(
        mask[center_idx] > 200,
        "center should be bright, got {}",
        mask[center_idx]
    );
}

#[test]
fn gradient_radial_corner_is_dark() {
    let mask = mask_gradient_radial(64, 64, 0.5, 0.5, 0.3, 0.3, 0.2);
    // Corner pixel (0, 0) should be dark (outside ellipse)
    assert!(mask[0] < 50, "corner should be dark, got {}", mask[0]);
}

// ── Luminance Range ─────────────────────────────────────────────────────

#[test]
fn luminance_range_highlights_only() {
    // Create image: dark pixel (50,50,50) and bright pixel (200,200,200)
    let pixels = vec![50, 50, 50, 200, 200, 200];
    let info = info_rgb8(2, 1);
    let (mask, mask_info) = mask_luminance_range(&pixels, &info, 0.7, 1.0, 0.0).unwrap();
    assert_eq!(mask_info.format, PixelFormat::Gray8);
    assert_eq!(mask.len(), 2);
    // Dark pixel should be masked out
    assert_eq!(mask[0], 0, "dark pixel should be 0, got {}", mask[0]);
    // Bright pixel should be included
    assert!(
        mask[1] > 200,
        "bright pixel should be ~255, got {}",
        mask[1]
    );
}

#[test]
fn luminance_range_shadows_only() {
    let pixels = vec![50, 50, 50, 200, 200, 200];
    let info = info_rgb8(2, 1);
    let (mask, _) = mask_luminance_range(&pixels, &info, 0.0, 0.3, 0.0).unwrap();
    // Dark pixel should be included
    assert!(mask[0] > 200, "dark pixel should be ~255, got {}", mask[0]);
    // Bright pixel should be masked out
    assert_eq!(mask[1], 0, "bright pixel should be 0, got {}", mask[1]);
}

// ── Color Range ─────────────────────────────────────────────────────────

#[test]
fn color_range_isolates_red() {
    // Red, green, blue pixels
    let pixels = vec![255, 0, 0, 0, 255, 0, 0, 0, 255];
    let info = info_rgb8(3, 1);
    let (mask, mask_info) = mask_color_range(&pixels, &info, 0.0, 60.0, 0.3, 0.0).unwrap();
    assert_eq!(mask_info.format, PixelFormat::Gray8);
    // Red should be selected
    assert_eq!(mask[0], 255, "red should be 255, got {}", mask[0]);
    // Green and blue should not
    assert_eq!(mask[1], 0, "green should be 0, got {}", mask[1]);
    assert_eq!(mask[2], 0, "blue should be 0, got {}", mask[2]);
}

// ── From Path ───────────────────────────────────────────────────────────

#[test]
fn from_path_single_point() {
    let mask = mask_from_path(32, 32, 1, 16.0, 16.0, 8.0, 1.0);
    assert_eq!(mask.len(), 32 * 32 * 3);
    // Center should be bright
    let center_idx = (16 * 32 + 16) * 3;
    assert!(
        mask[center_idx] > 200,
        "center should be bright, got {}",
        mask[center_idx]
    );
    // Far corner should be dark
    assert!(mask[0] < 10, "corner should be dark, got {}", mask[0]);
}

// ── Mask Combine ────────────────────────────────────────────────────────

#[test]
fn mask_combine_add() {
    let a = vec![100, 100, 100, 200, 200, 200];
    let b = vec![100, 100, 100, 100, 100, 100];
    let info = info_rgb8(2, 1);
    let result = mask_combine(&a, &info, &b, &info, 0).unwrap();
    // First pixel: 100+100 = 200
    assert_eq!(result[0], 200);
    // Second pixel: 200+100 = 255 (clamped)
    assert_eq!(result[3], 255);
}

#[test]
fn mask_combine_subtract() {
    let a = vec![200, 200, 200, 50, 50, 50];
    let b = vec![100, 100, 100, 100, 100, 100];
    let info = info_rgb8(2, 1);
    let result = mask_combine(&a, &info, &b, &info, 1).unwrap();
    assert_eq!(result[0], 100); // 200-100
    assert_eq!(result[3], 0); // 50-100 clamped to 0
}

#[test]
fn mask_combine_intersect() {
    let a = vec![200, 200, 200, 50, 50, 50];
    let b = vec![100, 100, 100, 100, 100, 100];
    let info = info_rgb8(2, 1);
    let result = mask_combine(&a, &info, &b, &info, 2).unwrap();
    assert_eq!(result[0], 100); // min(200, 100)
    assert_eq!(result[3], 50); // min(50, 100)
}

// ── Mask Invert ─────────────────────────────────────────────────────────

#[test]
fn mask_invert_gray8() {
    let pixels = vec![0u8, 128, 255];
    let info = info_gray8(3, 1);
    let result = mask_invert(
        Rect::new(0, 0, 3, 1),
        &mut |_| Ok(pixels.clone()),
        &info,
        &MaskInvertParams {},
    )
    .unwrap();
    assert_eq!(result, vec![255, 127, 0]);
}

#[test]
fn mask_invert_rgb8() {
    let pixels = vec![100, 100, 100, 200, 200, 200];
    let info = info_rgb8(2, 1);
    let result = mask_invert(
        Rect::new(0, 0, 2, 1),
        &mut |_| Ok(pixels.clone()),
        &info,
        &MaskInvertParams {},
    )
    .unwrap();
    assert_eq!(result[0], 155); // 255 - 100
    assert_eq!(result[3], 55); // 255 - 200
}

// ── Mask Feather ────────────────────────────────────────────────────────

#[test]
fn mask_feather_blurs_sharp_edge() {
    // Create a sharp edge mask: left half = 0, right half = 255
    let w = 32u32;
    let h = 8u32;
    let mut pixels = vec![0u8; (w * h) as usize];
    for y in 0..h as usize {
        for x in (w / 2) as usize..w as usize {
            pixels[y * w as usize + x] = 255;
        }
    }
    let info = info_gray8(w, h);
    let result = mask_feather(
        Rect::new(0, 0, w, h),
        &mut |_| Ok(pixels.clone()),
        &info,
        &MaskFeatherParams { radius: 3.0 },
    )
    .unwrap();
    // Middle row, at boundary: should be intermediate (not 0 or 255)
    let mid_row = 4;
    let boundary = (w / 2) as usize;
    let val = result[mid_row * w as usize + boundary];
    assert!(
        val > 20 && val < 235,
        "boundary should be intermediate after blur, got {val}"
    );
}

// ── Masked Blend ────────────────────────────────────────────────────────

#[test]
fn masked_blend_white_mask_gives_adjusted() {
    let original = solid_rgb(2, 2, 100, 100, 100);
    let adjusted = solid_rgb(2, 2, 200, 200, 200);
    let mask = vec![255u8; 2 * 2]; // White Gray8 mask
    let info = info_rgb8(2, 2);
    let result = masked_blend(&original, &info, &adjusted, &info, &mask, 2, 2).unwrap();
    // Should be all adjusted
    for &v in &result {
        assert_eq!(v, 200);
    }
}

#[test]
fn masked_blend_black_mask_gives_original() {
    let original = solid_rgb(2, 2, 100, 100, 100);
    let adjusted = solid_rgb(2, 2, 200, 200, 200);
    let mask = vec![0u8; 2 * 2]; // Black Gray8 mask
    let info = info_rgb8(2, 2);
    let result = masked_blend(&original, &info, &adjusted, &info, &mask, 2, 2).unwrap();
    // Should be all original
    for &v in &result {
        assert_eq!(v, 100);
    }
}

#[test]
fn masked_blend_half_mask_gives_midpoint() {
    let original = solid_rgb(1, 1, 0, 0, 0);
    let adjusted = solid_rgb(1, 1, 200, 200, 200);
    let mask = vec![128u8]; // ~50% Gray8 mask
    let info = info_rgb8(1, 1);
    let result = masked_blend(&original, &info, &adjusted, &info, &mask, 1, 1).unwrap();
    // 200 * 128/255 + 0 * 127/255 ≈ 100
    assert!(
        (result[0] as i32 - 100).abs() <= 1,
        "expected ~100, got {}",
        result[0]
    );
}

#[test]
fn masked_blend_selective_exposure() {
    // Simulate selective exposure: brighten only where mask is white
    let original = solid_rgb(4, 1, 100, 100, 100);
    let adjusted = solid_rgb(4, 1, 200, 200, 200); // "brightened"
                                                   // Mask: first 2 pixels white, last 2 black
    let mask = vec![255, 255, 0, 0]; // Gray8 4 pixels
    let info = info_rgb8(4, 1);
    let result = masked_blend(&original, &info, &adjusted, &info, &mask, 4, 1).unwrap();
    // First 2 pixels should be adjusted (200)
    assert_eq!(result[0], 200);
    assert_eq!(result[3], 200);
    // Last 2 pixels should be original (100)
    assert_eq!(result[6], 100);
    assert_eq!(result[9], 100);
}
