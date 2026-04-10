//! Drawing primitives — line, rectangle, circle, and text rendering.
//!
//! Uses `tiny_skia` (re-exported by `resvg`) for anti-aliased shape rendering.
//! Text uses an embedded 8×16 bitmap font for zero-dependency WASM-compatible
//! rendering.

mod arc;
mod circle;
mod ellipse;
mod line;
mod polygon;
mod rect;
mod text;

pub use arc::*;
pub use circle::*;
pub use ellipse::*;
pub use line::*;
pub use polygon::*;
pub use rect::*;
pub use text::*;

use super::error::ImageError;
use super::types::{ImageInfo, PixelFormat};

// ─── Pixel Buffer ↔ tiny_skia Pixmap ─────────────────────────────────────

/// Ensure input is RGBA8; convert RGB8 → RGBA8 if needed.
/// Returns (pixels, info) where info.format is always Rgba8.
pub(crate) fn ensure_rgba8(
    pixels: &[u8],
    info: &ImageInfo,
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    match info.format {
        PixelFormat::Rgba8 => Ok((pixels.to_vec(), info.clone())),
        PixelFormat::Rgb8 => {
            let expected = info.width as usize * info.height as usize * 3;
            if pixels.len() < expected {
                return Err(ImageError::InvalidInput(format!(
                    "draw: expected {} bytes for {}x{} RGB8, got {}",
                    expected,
                    info.width,
                    info.height,
                    pixels.len()
                )));
            }
            let mut rgba = Vec::with_capacity(info.width as usize * info.height as usize * 4);
            for chunk in pixels.chunks_exact(3) {
                rgba.extend_from_slice(chunk);
                rgba.push(255);
            }
            let out_info = ImageInfo {
                format: PixelFormat::Rgba8,
                ..info.clone()
            };
            Ok((rgba, out_info))
        }
        other => Err(ImageError::UnsupportedFormat(format!(
            "draw operations require RGB8 or RGBA8, got {other:?}"
        ))),
    }
}

/// Convert straight-alpha RGBA8 pixels to a tiny_skia Pixmap (premultiplied).
pub(crate) fn pixels_to_pixmap(
    pixels: &[u8],
    width: u32,
    height: u32,
) -> Result<resvg::tiny_skia::Pixmap, ImageError> {
    let mut pixmap = resvg::tiny_skia::Pixmap::new(width, height)
        .ok_or_else(|| ImageError::ProcessingFailed("draw: failed to create pixmap".into()))?;

    let pm_data = pixmap.data_mut();
    for (i, chunk) in pixels.chunks_exact(4).enumerate() {
        let r = chunk[0];
        let g = chunk[1];
        let b = chunk[2];
        let a = chunk[3];
        let base = i * 4;
        if a == 255 {
            pm_data[base] = r;
            pm_data[base + 1] = g;
            pm_data[base + 2] = b;
            pm_data[base + 3] = a;
        } else if a == 0 {
            pm_data[base] = 0;
            pm_data[base + 1] = 0;
            pm_data[base + 2] = 0;
            pm_data[base + 3] = 0;
        } else {
            // Premultiply
            let af = a as f32 / 255.0;
            pm_data[base] = (r as f32 * af + 0.5) as u8;
            pm_data[base + 1] = (g as f32 * af + 0.5) as u8;
            pm_data[base + 2] = (b as f32 * af + 0.5) as u8;
            pm_data[base + 3] = a;
        }
    }

    Ok(pixmap)
}

/// Convert a tiny_skia Pixmap (premultiplied) back to straight-alpha RGBA8.
pub(crate) fn pixmap_to_pixels(pixmap: &resvg::tiny_skia::Pixmap) -> Vec<u8> {
    pixmap
        .pixels()
        .iter()
        .flat_map(|px| {
            let r = px.red();
            let g = px.green();
            let b = px.blue();
            let a = px.alpha();
            if a == 0 {
                [0, 0, 0, 0]
            } else if a == 255 {
                [r, g, b, a]
            } else {
                // Demultiply
                [
                    ((r as u16 * 255 + a as u16 / 2) / a as u16) as u8,
                    ((g as u16 * 255 + a as u16 / 2) / a as u16) as u8,
                    ((b as u16 * 255 + a as u16 / 2) / a as u16) as u8,
                    a,
                ]
            }
        })
        .collect()
}

/// Create a tiny_skia Paint from RGBA color components.
pub(crate) fn make_paint(r: u8, g: u8, b: u8, a: u8) -> resvg::tiny_skia::Paint<'static> {
    let mut paint = resvg::tiny_skia::Paint::default();
    paint.set_color_rgba8(r, g, b, a);
    paint.anti_alias = true;
    paint
}

/// Create a tiny_skia Stroke with given width.
pub(crate) fn make_stroke(width: f32) -> resvg::tiny_skia::Stroke {
    resvg::tiny_skia::Stroke {
        width,
        ..resvg::tiny_skia::Stroke::default()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::ColorSpace;

    fn white_rgba(w: u32, h: u32) -> (Vec<u8>, ImageInfo) {
        let pixels = vec![255u8; (w * h * 4) as usize];
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        (pixels, info)
    }

    fn white_rgb(w: u32, h: u32) -> (Vec<u8>, ImageInfo) {
        let pixels = vec![255u8; (w * h * 3) as usize];
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        (pixels, info)
    }

    #[test]
    fn draw_line_produces_output() {
        let (px, info) = white_rgba(100, 100);
        let (result, out_info) =
            draw_line(&px, &info, 10.0, 10.0, 90.0, 90.0, [255, 0, 0, 255], 2.0).unwrap();
        assert_eq!(out_info.format, PixelFormat::Rgba8);
        assert_eq!(result.len(), 100 * 100 * 4);
        // The diagonal line should have modified some pixels from white to red
        assert_ne!(result, px, "line should modify pixels");
    }

    #[test]
    fn draw_rect_filled() {
        let (px, info) = white_rgba(100, 100);
        let (result, _) = draw_rect(
            &px,
            &info,
            20.0,
            20.0,
            40.0,
            30.0,
            [0, 0, 255, 255],
            1.0,
            true,
        )
        .unwrap();
        // Check center pixel of rect (40, 35) is blue
        let idx = (35 * 100 + 40) * 4;
        assert_eq!(result[idx], 0, "red channel should be 0 (blue fill)");
        assert_eq!(result[idx + 2], 255, "blue channel should be 255");
    }

    #[test]
    fn draw_rect_outline() {
        let (px, info) = white_rgba(100, 100);
        let (result, _) = draw_rect(
            &px,
            &info,
            20.0,
            20.0,
            40.0,
            30.0,
            [255, 0, 0, 255],
            2.0,
            false,
        )
        .unwrap();
        // Center of rect (40, 35) should still be white (outline only)
        let idx = (35 * 100 + 40) * 4;
        assert_eq!(result[idx], 255, "center should remain white (outline)");
        // Edge pixel (20, 20) should be red
        let edge = (20 * 100 + 20) * 4;
        assert_ne!(
            result[edge..edge + 3],
            [255, 255, 255],
            "edge should be red"
        );
    }

    #[test]
    fn draw_circle_filled() {
        let (px, info) = white_rgba(100, 100);
        let (result, _) =
            draw_circle(&px, &info, 50.0, 50.0, 20.0, [0, 255, 0, 255], 1.0, true).unwrap();
        // Center (50, 50) should be green
        let idx = (50 * 100 + 50) * 4;
        assert_eq!(result[idx + 1], 255, "green channel at center");
    }

    #[test]
    fn draw_line_rgb8_auto_converts() {
        let (px, info) = white_rgb(100, 100);
        let (result, out_info) =
            draw_line(&px, &info, 0.0, 0.0, 99.0, 99.0, [255, 0, 0, 255], 1.0).unwrap();
        assert_eq!(out_info.format, PixelFormat::Rgba8);
        assert_eq!(result.len(), 100 * 100 * 4);
    }

    #[test]
    fn draw_text_renders_chars() {
        let (px, info) = white_rgba(200, 50);
        let (result, _) = draw_text(&px, &info, 10, 10, "Hello", 1, [0, 0, 0, 255]).unwrap();
        // Text should have modified some pixels
        assert_ne!(result, px);
    }

    #[test]
    fn draw_text_scaled() {
        let (px, info) = white_rgba(200, 100);
        let (result, _) = draw_text(&px, &info, 10, 10, "AB", 2, [0, 0, 0, 255]).unwrap();
        assert_ne!(result, px);
    }

    // ── TrueType text tests ──

    fn load_system_font() -> Option<Vec<u8>> {
        // Try common system font paths (macOS, Linux)
        for path in &[
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNSMono.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
        ] {
            if let Ok(data) = std::fs::read(path) {
                return Some(data);
            }
        }
        None
    }

    #[test]
    fn draw_text_ttf_renders_glyphs() {
        let Some(font_data) = load_system_font() else {
            eprintln!("SKIP draw_text_ttf: no system font found");
            return;
        };
        let (px, info) = white_rgba(400, 100);
        let (result, out_info) = draw_text_ttf(
            &px,
            &info,
            10,
            10,
            "Hello World",
            &font_data,
            24.0,
            [0, 0, 0, 255],
        )
        .unwrap();
        assert_eq!(out_info.format, PixelFormat::Rgba8);
        assert_ne!(result, px, "TTF text should modify pixels");

        // Verify anti-aliased pixels exist (values between 0 and 255)
        let has_antialiased = result
            .chunks_exact(4)
            .any(|rgba| rgba[0] > 0 && rgba[0] < 255 && rgba[3] > 0);
        assert!(
            has_antialiased,
            "TTF rendering should produce anti-aliased pixels"
        );
    }

    #[test]
    fn draw_text_ttf_multiline() {
        let Some(font_data) = load_system_font() else {
            eprintln!("SKIP draw_text_ttf_multiline: no system font found");
            return;
        };
        let (px, info) = white_rgba(400, 200);
        let (result_single, _) = draw_text_ttf(
            &px,
            &info,
            10,
            10,
            "Hello",
            &font_data,
            24.0,
            [0, 0, 0, 255],
        )
        .unwrap();
        let (result_multi, _) = draw_text_ttf(
            &px,
            &info,
            10,
            10,
            "Hello\nWorld",
            &font_data,
            24.0,
            [0, 0, 0, 255],
        )
        .unwrap();

        // Multi-line should modify more pixels (two lines vs one)
        let modified_single = result_single
            .iter()
            .zip(px.iter())
            .filter(|(a, b)| a != b)
            .count();
        let modified_multi = result_multi
            .iter()
            .zip(px.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            modified_multi > modified_single,
            "multi-line ({modified_multi}) should modify more pixels than single ({modified_single})"
        );
    }

    #[test]
    fn draw_text_auto_bitmap_fallback() {
        let (px, info) = white_rgba(200, 50);
        // No font data → bitmap fallback
        let (result, _) =
            draw_text_auto(&px, &info, 10, 10, "Test", None, 16.0, [0, 0, 0, 255]).unwrap();
        assert_ne!(result, px, "bitmap fallback should render text");
    }

    #[test]
    fn draw_text_auto_ttf_when_available() {
        let Some(font_data) = load_system_font() else {
            eprintln!("SKIP draw_text_auto_ttf: no system font found");
            return;
        };
        let (px, info) = white_rgba(400, 100);
        let (result, _) = draw_text_auto(
            &px,
            &info,
            10,
            10,
            "TTF Text",
            Some(&font_data),
            24.0,
            [0, 0, 0, 255],
        )
        .unwrap();
        assert_ne!(result, px, "TTF auto should render text");
    }

    // ── Polygon tests ──

    #[test]
    fn draw_polygon_triangle_filled() {
        use crate::domain::param_types::Point2D;
        let (px, info) = white_rgba(100, 100);
        let points = [
            Point2D { x: 50.0, y: 10.0 },
            Point2D { x: 10.0, y: 90.0 },
            Point2D { x: 90.0, y: 90.0 },
        ];
        let (result, _) = draw_polygon(
            &px,
            &info,
            &points,
            [255, 0, 0, 255],
            [0, 0, 0, 255],
            0.0,
            true,
        )
        .unwrap();
        // Center of triangle (~50, 60) should be red
        let idx = (60 * 100 + 50) * 4;
        assert_eq!(result[idx], 255, "red fill at center");
        assert_eq!(result[idx + 1], 0);
    }

    #[test]
    fn draw_polygon_square_stroked() {
        use crate::domain::param_types::Point2D;
        let (px, info) = white_rgba(100, 100);
        let points = [
            Point2D { x: 20.0, y: 20.0 },
            Point2D { x: 80.0, y: 20.0 },
            Point2D { x: 80.0, y: 80.0 },
            Point2D { x: 20.0, y: 80.0 },
        ];
        let (result, _) = draw_polygon(
            &px,
            &info,
            &points,
            [0, 0, 0, 0],
            [0, 0, 255, 255],
            3.0,
            false,
        )
        .unwrap();
        // Center (50, 50) should still be white (stroke only, no fill)
        let center = (50 * 100 + 50) * 4;
        assert_eq!(result[center], 255, "center white");
        assert_eq!(result[center + 1], 255);
        // Edge (20, 20) should have blue stroke
        let edge = (20 * 100 + 20) * 4;
        assert_ne!(
            result[edge..edge + 3],
            [255, 255, 255],
            "edge should be stroked"
        );
    }

    #[test]
    fn draw_polygon_too_few_points_errors() {
        use crate::domain::param_types::Point2D;
        let (px, info) = white_rgba(100, 100);
        let result = draw_polygon(
            &px,
            &info,
            &[Point2D { x: 10.0, y: 10.0 }, Point2D { x: 20.0, y: 20.0 }],
            [0, 0, 0, 255],
            [0, 0, 0, 255],
            1.0,
            true,
        );
        assert!(result.is_err());
    }

    // ── Ellipse tests ──

    #[test]
    fn draw_ellipse_filled() {
        let (px, info) = white_rgba(100, 100);
        let (result, _) = draw_ellipse(
            &px,
            &info,
            50.0,
            50.0,
            30.0,
            15.0,
            [0, 255, 0, 255],
            1.0,
            true,
        )
        .unwrap();
        // Center (50, 50) should be green
        let idx = (50 * 100 + 50) * 4;
        assert_eq!(result[idx + 1], 255, "green at center");
    }

    #[test]
    fn draw_ellipse_outline() {
        let (px, info) = white_rgba(100, 100);
        let (result, _) = draw_ellipse(
            &px,
            &info,
            50.0,
            50.0,
            30.0,
            15.0,
            [255, 0, 0, 255],
            2.0,
            false,
        )
        .unwrap();
        // Center should remain white
        let center = (50 * 100 + 50) * 4;
        assert_eq!(result[center], 255, "center white for outline");
        // A point on the ellipse edge (50+30, 50) = (80, 50) should be red
        let edge = (50 * 100 + 80) * 4;
        assert_ne!(result[edge..edge + 3], [255, 255, 255], "edge stroked");
    }

    // ── Arc tests ──

    #[test]
    fn draw_arc_semicircle() {
        let (px, info) = white_rgba(100, 100);
        let (result, _) = draw_arc(
            &px,
            &info,
            50.0,
            50.0,
            30.0,
            30.0,
            0.0,
            180.0,
            [255, 0, 0, 255],
            2.0,
        )
        .unwrap();
        assert_ne!(result, px, "arc should modify pixels");
    }

    #[test]
    fn draw_arc_invalid_radius_errors() {
        let (px, info) = white_rgba(100, 100);
        assert!(
            draw_arc(
                &px,
                &info,
                50.0,
                50.0,
                0.0,
                30.0,
                0.0,
                90.0,
                [0, 0, 0, 255],
                1.0
            )
            .is_err()
        );
    }

    #[test]
    fn draw_text_ttf_invalid_font_returns_error() {
        let (px, info) = white_rgba(100, 50);
        let result = draw_text_ttf(
            &px,
            &info,
            0,
            0,
            "test",
            b"not a font",
            16.0,
            [0, 0, 0, 255],
        );
        assert!(result.is_err(), "invalid font data should return error");
    }
}
