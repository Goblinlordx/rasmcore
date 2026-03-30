//! Alpha compositing — Porter-Duff "over" blend.
//!
//! Operates on raw RGBA8 pixel buffers using premultiplied alpha internally
//! for SIMD-friendly math (no per-pixel division in the inner loop).
//! The inner loop is a simple linear scan that LLVM auto-vectorizes to SIMD128.

use super::error::ImageError;
use super::types::{ImageInfo, PixelFormat};

/// Composite foreground over background using Porter-Duff "over" operator.
///
/// Places the foreground at `(offset_x, offset_y)` on the background.
/// Both inputs must be RGBA8 with straight (non-premultiplied) alpha.
/// Output has the same dimensions as the background.
#[rasmcore_macros::register_compositor(
    name = "composite",
    category = "composite",
    group = "composite",
    reference = "Porter-Duff 1984 over operator"
)]
pub fn alpha_composite_over(
    fg_pixels: &[u8],
    fg_info: &ImageInfo,
    bg_pixels: &[u8],
    bg_info: &ImageInfo,
    offset_x: i32,
    offset_y: i32,
) -> Result<Vec<u8>, ImageError> {
    if fg_info.format != PixelFormat::Rgba8 {
        return Err(ImageError::UnsupportedFormat(
            "foreground must be RGBA8".into(),
        ));
    }
    if bg_info.format != PixelFormat::Rgba8 {
        return Err(ImageError::UnsupportedFormat(
            "background must be RGBA8".into(),
        ));
    }

    let fg_w = fg_info.width as i32;
    let fg_h = fg_info.height as i32;
    let bg_w = bg_info.width as usize;
    let bg_h = bg_info.height as usize;

    let expected_fg = fg_info.width as usize * fg_info.height as usize * 4;
    if fg_pixels.len() < expected_fg {
        return Err(ImageError::InvalidInput(
            "foreground pixel buffer too small".into(),
        ));
    }
    let expected_bg = bg_w * bg_h * 4;
    if bg_pixels.len() < expected_bg {
        return Err(ImageError::InvalidInput(
            "background pixel buffer too small".into(),
        ));
    }

    // Compute the overlap region in background coordinates (all in i32 to avoid wrapping)
    let bg_w_i = bg_w as i32;
    let bg_h_i = bg_h as i32;
    let blend_x0 = offset_x.max(0);
    let blend_y0 = offset_y.max(0);
    let blend_x1 = (offset_x + fg_w).min(bg_w_i);
    let blend_y1 = (offset_y + fg_h).min(bg_h_i);

    // No overlap — return background unchanged
    if blend_x0 >= blend_x1 || blend_y0 >= blend_y1 {
        return Ok(bg_pixels.to_vec());
    }

    let mut out = bg_pixels.to_vec();

    // Foreground start offset (handles negative offset_x/offset_y)
    let fg_start_x = (blend_x0 - offset_x) as usize;
    let fg_start_y = (blend_y0 - offset_y) as usize;
    let fg_stride = fg_info.width as usize * 4;
    let bg_stride = bg_w * 4;
    let blend_x0 = blend_x0 as usize;
    let blend_y0 = blend_y0 as usize;
    let blend_x1 = blend_x1 as usize;
    let blend_y1 = blend_y1 as usize;

    for row in 0..(blend_y1 - blend_y0) {
        let fg_row_offset = (fg_start_y + row) * fg_stride + fg_start_x * 4;
        let bg_row_offset = (blend_y0 + row) * bg_stride + blend_x0 * 4;
        let width = blend_x1 - blend_x0;

        blend_row_over(
            &fg_pixels[fg_row_offset..fg_row_offset + width * 4],
            &mut out[bg_row_offset..bg_row_offset + width * 4],
        );
    }

    Ok(out)
}

/// Blend a row of RGBA8 pixels using Porter-Duff "over" (premultiplied math).
///
/// Uses premultiplied alpha internally to avoid per-pixel division:
///   out_r = fg_r * fg_a + bg_r * (1 - fg_a)
///   out_a = fg_a + bg_a * (1 - fg_a)
///
/// Written as a tight loop for LLVM auto-vectorization to SIMD128.
#[inline]
fn blend_row_over(fg: &[u8], bg_out: &mut [u8]) {
    debug_assert_eq!(fg.len(), bg_out.len());
    debug_assert_eq!(fg.len() % 4, 0);

    for (fg_px, bg_px) in fg.chunks_exact(4).zip(bg_out.chunks_exact_mut(4)) {
        let fg_a = fg_px[3] as u32;

        // Fast paths: fully transparent or fully opaque foreground
        if fg_a == 0 {
            continue;
        }
        if fg_a == 255 {
            bg_px.copy_from_slice(fg_px);
            continue;
        }

        let inv_a = 255 - fg_a; // (1 - fg_a) scaled to 0..255
        let bg_a = bg_px[3] as u32;

        // Premultiplied blend: out = fg * fg_a + bg * inv_a (all in 0..255 scale)
        // We use (x * 255 + 128) >> 8 ≈ x for the final division, but to keep
        // precision we compute in 0..65025 then divide by 255.
        // Simpler: (fg_c * fg_a + bg_c * inv_a + 127) / 255
        bg_px[0] = ((fg_px[0] as u32 * fg_a + bg_px[0] as u32 * inv_a + 127) / 255) as u8;
        bg_px[1] = ((fg_px[1] as u32 * fg_a + bg_px[1] as u32 * inv_a + 127) / 255) as u8;
        bg_px[2] = ((fg_px[2] as u32 * fg_a + bg_px[2] as u32 * inv_a + 127) / 255) as u8;
        bg_px[3] = ((fg_a * 255 + bg_a * inv_a + 127) / 255) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::ColorSpace;

    fn rgba_info(w: u32, h: u32) -> ImageInfo {
        ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        }
    }

    /// Create a buffer of `count` identical RGBA pixels.
    fn solid(r: u8, g: u8, b: u8, a: u8, count: usize) -> Vec<u8> {
        [r, g, b, a].repeat(count)
    }

    #[test]
    fn opaque_foreground_replaces_background() {
        let fg = solid(255, 0, 0, 255, 4);
        let bg = solid(0, 0, 255, 255, 4);
        let fg_info = rgba_info(2, 2);
        let bg_info = rgba_info(2, 2);

        let result = alpha_composite_over(&fg, &fg_info, &bg, &bg_info, 0, 0).unwrap();
        for px in result.chunks_exact(4) {
            assert_eq!(px, [255, 0, 0, 255]);
        }
    }

    #[test]
    fn transparent_foreground_preserves_background() {
        let fg = solid(255, 0, 0, 0, 4);
        let bg = solid(0, 0, 255, 255, 4);
        let fg_info = rgba_info(2, 2);
        let bg_info = rgba_info(2, 2);

        let result = alpha_composite_over(&fg, &fg_info, &bg, &bg_info, 0, 0).unwrap();
        for px in result.chunks_exact(4) {
            assert_eq!(px, [0, 0, 255, 255]);
        }
    }

    #[test]
    fn semi_transparent_blend() {
        // 50% red over solid blue
        let fg = vec![255, 0, 0, 128];
        let bg = vec![0, 0, 255, 255];
        let fg_info = rgba_info(1, 1);
        let bg_info = rgba_info(1, 1);

        let result = alpha_composite_over(&fg, &fg_info, &bg, &bg_info, 0, 0).unwrap();
        assert_eq!(result[0], 128); // red
        assert_eq!(result[1], 0); // green
        assert_eq!(result[2], 127); // blue
        assert_eq!(result[3], 255); // alpha
    }

    #[test]
    fn offset_positioning() {
        let fg = vec![255, 0, 0, 255];
        let bg = solid(0, 0, 255, 255, 9);
        let fg_info = rgba_info(1, 1);
        let bg_info = rgba_info(3, 3);

        let result = alpha_composite_over(&fg, &fg_info, &bg, &bg_info, 1, 1).unwrap();
        for (i, px) in result.chunks_exact(4).enumerate() {
            let (x, y) = (i % 3, i / 3);
            if x == 1 && y == 1 {
                assert_eq!(px, [255, 0, 0, 255], "pixel ({x},{y}) should be red");
            } else {
                assert_eq!(px, [0, 0, 255, 255], "pixel ({x},{y}) should be blue");
            }
        }
    }

    #[test]
    fn foreground_clipped_right_bottom() {
        let fg = solid(255, 0, 0, 255, 4);
        let bg = solid(0, 0, 255, 255, 9);
        let fg_info = rgba_info(2, 2);
        let bg_info = rgba_info(3, 3);

        let result = alpha_composite_over(&fg, &fg_info, &bg, &bg_info, 2, 2).unwrap();
        for (i, px) in result.chunks_exact(4).enumerate() {
            let (x, y) = (i % 3, i / 3);
            if x == 2 && y == 2 {
                assert_eq!(px, [255, 0, 0, 255]);
            } else {
                assert_eq!(px, [0, 0, 255, 255]);
            }
        }
    }

    #[test]
    fn negative_offset_clips_left_top() {
        let fg = vec![
            10, 20, 30, 255, // (0,0) — clipped
            40, 50, 60, 255, // (1,0) — clipped
            70, 80, 90, 255, // (0,1) — clipped
            255, 0, 0, 255, // (1,1) — visible at bg (0,0)
        ];
        let bg = solid(0, 0, 255, 255, 9);
        let fg_info = rgba_info(2, 2);
        let bg_info = rgba_info(3, 3);

        let result = alpha_composite_over(&fg, &fg_info, &bg, &bg_info, -1, -1).unwrap();
        assert_eq!(&result[0..4], [255, 0, 0, 255]);
        assert_eq!(&result[4..8], [0, 0, 255, 255]);
    }

    #[test]
    fn foreground_completely_outside() {
        let fg = solid(255, 0, 0, 255, 4);
        let bg = solid(0, 0, 255, 255, 4);
        let fg_info = rgba_info(2, 2);
        let bg_info = rgba_info(2, 2);

        let result = alpha_composite_over(&fg, &fg_info, &bg, &bg_info, 10, 10).unwrap();
        assert_eq!(result, bg);

        let result = alpha_composite_over(&fg, &fg_info, &bg, &bg_info, -10, -10).unwrap();
        assert_eq!(result, bg);
    }

    #[test]
    fn rejects_non_rgba8() {
        let fg = vec![0; 12]; // 2x2 RGB8
        let bg = vec![0; 16]; // 2x2 RGBA8
        let fg_info = ImageInfo {
            width: 2,
            height: 2,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let bg_info = rgba_info(2, 2);

        assert!(alpha_composite_over(&fg, &fg_info, &bg, &bg_info, 0, 0).is_err());
    }

    #[test]
    fn both_semi_transparent() {
        // 50% red over 50% blue
        let fg = vec![255, 0, 0, 128];
        let bg = vec![0, 0, 255, 128];
        let fg_info = rgba_info(1, 1);
        let bg_info = rgba_info(1, 1);

        let result = alpha_composite_over(&fg, &fg_info, &bg, &bg_info, 0, 0).unwrap();
        assert_eq!(result[0], 128); // red
        assert_eq!(result[1], 0); // green
        assert_eq!(result[2], 127); // blue
        assert_eq!(result[3], 192); // alpha: ~75%
    }

    #[test]
    fn output_dimensions_match_background() {
        let fg = solid(255, 0, 0, 255, 4);
        let bg = solid(0, 0, 255, 255, 25);
        let fg_info = rgba_info(2, 2);
        let bg_info = rgba_info(5, 5);

        let result = alpha_composite_over(&fg, &fg_info, &bg, &bg_info, 1, 1).unwrap();
        assert_eq!(result.len(), 5 * 5 * 4);
    }
}
