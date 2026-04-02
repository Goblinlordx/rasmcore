//! Tests for brush tool filters.

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

/// Uniform color image.
fn solid_rgb(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
    let count = (w * h) as usize;
    let mut px = Vec::with_capacity(count * 3);
    for _ in 0..count {
        px.push(r);
        px.push(g);
        px.push(b);
    }
    px
}

/// Grayscale mask (1 byte per pixel).
fn solid_mask(w: u32, h: u32, val: u8) -> Vec<u8> {
    vec![val; (w * h) as usize]
}

fn gray_info(w: u32, h: u32) -> ImageInfo {
    ImageInfo {
        width: w,
        height: h,
        format: PixelFormat::Gray8,
        color_space: ColorSpace::Srgb,
    }
}

// ── Registration ──

#[test]
fn brush_tools_registered() {
    let compositors = crate::domain::filter_registry::registered_compositors();
    let names: Vec<&str> = compositors.iter().map(|c| c.name).collect();
    for expected in &["clone_stamp", "healing_brush", "smudge", "sponge"] {
        assert!(
            names.contains(expected),
            "{expected} should be in the compositor registry"
        );
    }
}

// ── Clone Stamp ──

#[test]
fn clone_stamp_exact_copy_with_full_mask() {
    // Red image with one blue pixel at (2,2). Clone offset (-2,-2) should
    // copy the blue pixel to (0,0) when mask is fully white.
    let w = 4u32;
    let h = 4u32;
    let mut pixels = solid_rgb(w, h, 255, 0, 0);
    // Set pixel at (2,2) to blue
    let idx = (2 * w as usize + 2) * 3;
    pixels[idx] = 0;
    pixels[idx + 1] = 0;
    pixels[idx + 2] = 255;

    let mask = solid_mask(w, h, 255);
    let fg_info = rgb_info(w, h);
    let mask_info = gray_info(w, h);

    let result = super::clone_stamp(&pixels, &fg_info, &mask, &mask_info, -2, -2).unwrap();

    // Pixel at (0,0) should now be blue (cloned from (2,2) via offset (-2,-2) → src=(0+(-2), 0+(-2))… wait,
    // offset means src_x = x + offset_x. To clone FROM (2,2) TO (0,0), we need offset = +2,+2.
    // Let me re-check: src = (x + offset_x, y + offset_y). At dst=(0,0), src=(2,2) → offset=(2,2).
    let result2 = super::clone_stamp(&pixels, &fg_info, &mask, &mask_info, 2, 2).unwrap();
    // Pixel at (0,0) should be blue
    assert_eq!(result2[0], 0, "R should be 0 (blue pixel cloned)");
    assert_eq!(result2[1], 0, "G should be 0");
    assert_eq!(result2[2], 255, "B should be 255");
}

#[test]
fn clone_stamp_zero_mask_is_identity() {
    let pixels = solid_rgb(8, 8, 128, 64, 32);
    let mask = solid_mask(8, 8, 0);
    let fg_info = rgb_info(8, 8);
    let mask_info = gray_info(8, 8);

    let result = super::clone_stamp(&pixels, &fg_info, &mask, &mask_info, 3, 3).unwrap();
    assert_eq!(result, pixels, "zero mask should produce identity");
}

// ── Healing Brush ──

#[test]
fn healing_brush_output_size() {
    let pixels = solid_rgb(16, 16, 100, 100, 100);
    let mask = solid_mask(16, 16, 128);
    let fg_info = rgb_info(16, 16);
    let mask_info = gray_info(16, 16);

    let result = super::healing_brush(&pixels, &fg_info, &mask, &mask_info, 0, 0).unwrap();
    assert_eq!(result.len(), pixels.len());
}

#[test]
fn healing_brush_zero_mask_is_identity() {
    let pixels = solid_rgb(8, 8, 100, 200, 50);
    let mask = solid_mask(8, 8, 0);
    let fg_info = rgb_info(8, 8);
    let mask_info = gray_info(8, 8);

    let result = super::healing_brush(&pixels, &fg_info, &mask, &mask_info, 2, 2).unwrap();
    assert_eq!(result, pixels);
}

// ── Smudge ──

#[test]
fn smudge_zero_mask_is_identity() {
    let pixels = solid_rgb(8, 8, 100, 200, 50);
    let mask = solid_mask(8, 8, 0);
    let fg_info = rgb_info(8, 8);
    let mask_info = gray_info(8, 8);

    let result = super::smudge(&pixels, &fg_info, &mask, &mask_info, 1.0, 0.0, 0.5).unwrap();
    assert_eq!(result, pixels);
}

#[test]
fn smudge_zero_strength_is_identity() {
    let pixels = solid_rgb(8, 8, 100, 200, 50);
    let mask = solid_mask(8, 8, 255);
    let fg_info = rgb_info(8, 8);
    let mask_info = gray_info(8, 8);

    let result = super::smudge(&pixels, &fg_info, &mask, &mask_info, 1.0, 0.0, 0.0).unwrap();
    assert_eq!(result, pixels);
}

#[test]
fn smudge_output_differs_with_gradient() {
    // Create a horizontal gradient
    let w = 16u32;
    let h = 4u32;
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for _y in 0..h {
        for x in 0..w {
            let v = (x * 255 / (w - 1)) as u8;
            pixels.push(v);
            pixels.push(v);
            pixels.push(v);
        }
    }
    let mask = solid_mask(w, h, 255);
    let fg_info = rgb_info(w, h);
    let mask_info = gray_info(w, h);

    let result = super::smudge(&pixels, &fg_info, &mask, &mask_info, 1.0, 0.0, 1.0).unwrap();
    assert_ne!(result, pixels, "smudge on gradient should change pixels");
}

// ── Sponge ──

#[test]
fn sponge_zero_mask_is_identity() {
    let pixels = solid_rgb(8, 8, 200, 100, 50);
    let mask = solid_mask(8, 8, 0);
    let fg_info = rgb_info(8, 8);
    let mask_info = gray_info(8, 8);

    let result = super::sponge(&pixels, &fg_info, &mask, &mask_info, 0, 0.5).unwrap();
    assert_eq!(result, pixels);
}

#[test]
fn sponge_saturate_increases_color() {
    // Start with a muted color
    let pixels = solid_rgb(4, 4, 180, 150, 150);
    let mask = solid_mask(4, 4, 255);
    let fg_info = rgb_info(4, 4);
    let mask_info = gray_info(4, 4);

    let result = super::sponge(&pixels, &fg_info, &mask, &mask_info, 0, 1.0).unwrap();
    // After saturating, the dominant channel (R=180) should be more different from others
    let r = result[0] as i32;
    let g = result[1] as i32;
    // Saturation increased → R should increase relative to G/B, or G/B should decrease
    let orig_diff = 180i32 - 150;
    let new_diff = r - g;
    assert!(new_diff >= orig_diff, "saturate should increase color separation: orig_diff={orig_diff}, new_diff={new_diff}");
}

#[test]
fn sponge_desaturate_reduces_color() {
    let pixels = solid_rgb(4, 4, 255, 100, 50);
    let mask = solid_mask(4, 4, 255);
    let fg_info = rgb_info(4, 4);
    let mask_info = gray_info(4, 4);

    let result = super::sponge(&pixels, &fg_info, &mask, &mask_info, 1, 1.0).unwrap();
    // After desaturating, channels should converge toward gray (luminance)
    let r = result[0];
    let g = result[1];
    let b = result[2];
    let range = r.max(g).max(b) - r.min(g).min(b);
    let orig_range = 255u8 - 50;
    assert!(range < orig_range, "desaturate should reduce channel range: {range} vs orig {orig_range}");
}
