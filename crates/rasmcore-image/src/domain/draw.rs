//! Drawing primitives — line, rectangle, circle, and text rendering.
//!
//! Uses `tiny_skia` (re-exported by `resvg`) for anti-aliased shape rendering.
//! Text uses an embedded 8×16 bitmap font for zero-dependency WASM-compatible
//! rendering.

use super::error::ImageError;
use super::types::{ImageInfo, PixelFormat};

// ─── Pixel Buffer ↔ tiny_skia Pixmap ─────────────────────────────────────

/// Ensure input is RGBA8; convert RGB8 → RGBA8 if needed.
/// Returns (pixels, info) where info.format is always Rgba8.
fn ensure_rgba8(pixels: &[u8], info: &ImageInfo) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    match info.format {
        PixelFormat::Rgba8 => Ok((pixels.to_vec(), info.clone())),
        PixelFormat::Rgb8 => {
            let expected = info.width as usize * info.height as usize * 3;
            if pixels.len() < expected {
                return Err(ImageError::InvalidInput(format!(
                    "draw: expected {} bytes for {}x{} RGB8, got {}",
                    expected, info.width, info.height, pixels.len()
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
fn pixels_to_pixmap(
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
fn pixmap_to_pixels(pixmap: &resvg::tiny_skia::Pixmap) -> Vec<u8> {
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
fn make_paint(r: u8, g: u8, b: u8, a: u8) -> resvg::tiny_skia::Paint<'static> {
    let mut paint = resvg::tiny_skia::Paint::default();
    paint.set_color_rgba8(r, g, b, a);
    paint.anti_alias = true;
    paint
}

/// Create a tiny_skia Stroke with given width.
fn make_stroke(width: f32) -> resvg::tiny_skia::Stroke {
    let mut stroke = resvg::tiny_skia::Stroke::default();
    stroke.width = width;
    stroke
}

// ─── Shape Drawing ───────────────────────────────────────────────────────

/// Draw a line on the image.
///
/// Coordinates are in pixels. Color is RGBA. Line width in pixels.
pub fn draw_line(
    pixels: &[u8],
    info: &ImageInfo,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    color: [u8; 4],
    width: f32,
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    let (rgba, out_info) = ensure_rgba8(pixels, info)?;
    let mut pixmap = pixels_to_pixmap(&rgba, out_info.width, out_info.height)?;

    let mut pb = resvg::tiny_skia::PathBuilder::new();
    pb.move_to(x1, y1);
    pb.line_to(x2, y2);
    let path = pb.finish().ok_or_else(|| {
        ImageError::InvalidParameters("draw_line: invalid path coordinates".into())
    })?;

    let paint = make_paint(color[0], color[1], color[2], color[3]);
    let stroke = make_stroke(width);

    pixmap.stroke_path(
        &path,
        &paint,
        &stroke,
        resvg::tiny_skia::Transform::identity(),
        None,
    );

    Ok((pixmap_to_pixels(&pixmap), out_info))
}

/// Draw a rectangle on the image.
///
/// If `filled` is true, the rectangle is filled. Otherwise only the outline
/// is drawn with the given `stroke_width`.
pub fn draw_rect(
    pixels: &[u8],
    info: &ImageInfo,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    color: [u8; 4],
    stroke_width: f32,
    filled: bool,
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    let (rgba, out_info) = ensure_rgba8(pixels, info)?;
    let mut pixmap = pixels_to_pixmap(&rgba, out_info.width, out_info.height)?;

    let rect = resvg::tiny_skia::Rect::from_xywh(x, y, w, h).ok_or_else(|| {
        ImageError::InvalidParameters(format!("draw_rect: invalid rect ({x},{y},{w},{h})"))
    })?;

    let paint = make_paint(color[0], color[1], color[2], color[3]);

    if filled {
        let mut pb = resvg::tiny_skia::PathBuilder::new();
        pb.push_rect(rect);
        if let Some(path) = pb.finish() {
            pixmap.fill_path(
                &path,
                &paint,
                resvg::tiny_skia::FillRule::Winding,
                resvg::tiny_skia::Transform::identity(),
                None,
            );
        }
    } else {
        let mut pb = resvg::tiny_skia::PathBuilder::new();
        pb.push_rect(rect);
        if let Some(path) = pb.finish() {
            let stroke = make_stroke(stroke_width);
            pixmap.stroke_path(
                &path,
                &paint,
                &stroke,
                resvg::tiny_skia::Transform::identity(),
                None,
            );
        }
    }

    Ok((pixmap_to_pixels(&pixmap), out_info))
}

/// Draw a circle on the image.
///
/// If `filled` is true, the circle is filled. Otherwise only the outline
/// is drawn with the given `stroke_width`.
pub fn draw_circle(
    pixels: &[u8],
    info: &ImageInfo,
    cx: f32,
    cy: f32,
    radius: f32,
    color: [u8; 4],
    stroke_width: f32,
    filled: bool,
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    let (rgba, out_info) = ensure_rgba8(pixels, info)?;
    let mut pixmap = pixels_to_pixmap(&rgba, out_info.width, out_info.height)?;

    let path = resvg::tiny_skia::PathBuilder::from_circle(cx, cy, radius).ok_or_else(|| {
        ImageError::InvalidParameters(format!(
            "draw_circle: invalid circle ({cx},{cy},r={radius})"
        ))
    })?;

    let paint = make_paint(color[0], color[1], color[2], color[3]);

    if filled {
        pixmap.fill_path(
            &path,
            &paint,
            resvg::tiny_skia::FillRule::Winding,
            resvg::tiny_skia::Transform::identity(),
            None,
        );
    } else {
        let stroke = make_stroke(stroke_width);
        pixmap.stroke_path(
            &path,
            &paint,
            &stroke,
            resvg::tiny_skia::Transform::identity(),
            None,
        );
    }

    Ok((pixmap_to_pixels(&pixmap), out_info))
}

// ─── Text Rendering (embedded bitmap font) ───────────────────────────────

/// Embedded 8×16 bitmap font glyph data for ASCII 32–126 (95 glyphs).
///
/// Each glyph is 16 bytes (one byte per row, 8 pixels wide, MSB-first).
/// This is a minimal fixed-width font suitable for annotations and debugging.
/// Based on the classic VGA 8×16 bitmap font (public domain).
const FONT_8X16: &[u8] = include_bytes!("font_8x16.bin");
const GLYPH_WIDTH: u32 = 8;
const GLYPH_HEIGHT: u32 = 16;
const GLYPH_FIRST: u8 = 32; // space
const GLYPH_LAST: u8 = 126; // tilde
const GLYPH_BYTES: usize = 16; // 16 rows × 1 byte/row

/// Draw text on the image using the embedded 8×16 bitmap font.
///
/// `x`, `y` are the top-left corner of the text baseline. `scale` multiplies
/// the native 8×16 size (1 = native, 2 = 16×32 pixels, etc.). Color is RGBA.
pub fn draw_text(
    pixels: &[u8],
    info: &ImageInfo,
    x: u32,
    y: u32,
    text: &str,
    scale: u32,
    color: [u8; 4],
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    let (rgba, out_info) = ensure_rgba8(pixels, info)?;
    let mut output = rgba;
    let w = out_info.width as usize;
    let h = out_info.height as usize;
    let scale = scale.max(1) as usize;
    let gw = GLYPH_WIDTH as usize * scale;
    let _gh = GLYPH_HEIGHT as usize * scale;

    let mut cursor_x = x as usize;
    let cursor_y = y as usize;

    for ch in text.bytes() {
        if ch == b'\n' {
            // Newlines not supported in this simple impl — skip
            continue;
        }
        let glyph_index = if ch >= GLYPH_FIRST && ch <= GLYPH_LAST {
            (ch - GLYPH_FIRST) as usize
        } else {
            // Unknown chars render as '?'
            (b'?' - GLYPH_FIRST) as usize
        };

        let glyph_offset = glyph_index * GLYPH_BYTES;
        if glyph_offset + GLYPH_BYTES > FONT_8X16.len() {
            continue; // Safety: skip if font data too short
        }

        for row in 0..GLYPH_HEIGHT as usize {
            let byte = FONT_8X16[glyph_offset + row];
            for col in 0..GLYPH_WIDTH as usize {
                if byte & (0x80 >> col) != 0 {
                    // Pixel is "on" — draw at scaled position
                    for sy in 0..scale {
                        for sx in 0..scale {
                            let px = cursor_x + col * scale + sx;
                            let py = cursor_y + row * scale + sy;
                            if px < w && py < h {
                                let idx = (py * w + px) * 4;
                                // Alpha blend the text color onto the background
                                let ta = color[3] as u16;
                                let inv_a = 255 - ta;
                                output[idx] =
                                    ((color[0] as u16 * ta + output[idx] as u16 * inv_a) / 255)
                                        as u8;
                                output[idx + 1] =
                                    ((color[1] as u16 * ta + output[idx + 1] as u16 * inv_a) / 255)
                                        as u8;
                                output[idx + 2] =
                                    ((color[2] as u16 * ta + output[idx + 2] as u16 * inv_a) / 255)
                                        as u8;
                                output[idx + 3] = output[idx + 3].max(color[3]);
                            }
                        }
                    }
                }
            }
        }

        cursor_x += gw;
        if cursor_x >= w {
            break; // Stop if text runs off-screen
        }
    }

    Ok((output, out_info))
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
        let (result, out_info) = draw_line(&px, &info, 10.0, 10.0, 90.0, 90.0, [255, 0, 0, 255], 2.0).unwrap();
        assert_eq!(out_info.format, PixelFormat::Rgba8);
        assert_eq!(result.len(), 100 * 100 * 4);
        // The diagonal line should have modified some pixels from white to red
        assert_ne!(result, px, "line should modify pixels");
    }

    #[test]
    fn draw_rect_filled() {
        let (px, info) = white_rgba(100, 100);
        let (result, _) = draw_rect(&px, &info, 20.0, 20.0, 40.0, 30.0, [0, 0, 255, 255], 1.0, true).unwrap();
        // Check center pixel of rect (40, 35) is blue
        let idx = (35 * 100 + 40) * 4;
        assert_eq!(result[idx], 0, "red channel should be 0 (blue fill)");
        assert_eq!(result[idx + 2], 255, "blue channel should be 255");
    }

    #[test]
    fn draw_rect_outline() {
        let (px, info) = white_rgba(100, 100);
        let (result, _) = draw_rect(&px, &info, 20.0, 20.0, 40.0, 30.0, [255, 0, 0, 255], 2.0, false).unwrap();
        // Center of rect (40, 35) should still be white (outline only)
        let idx = (35 * 100 + 40) * 4;
        assert_eq!(result[idx], 255, "center should remain white (outline)");
        // Edge pixel (20, 20) should be red
        let edge = (20 * 100 + 20) * 4;
        assert_ne!(result[edge..edge + 3], [255, 255, 255], "edge should be red");
    }

    #[test]
    fn draw_circle_filled() {
        let (px, info) = white_rgba(100, 100);
        let (result, _) = draw_circle(&px, &info, 50.0, 50.0, 20.0, [0, 255, 0, 255], 1.0, true).unwrap();
        // Center (50, 50) should be green
        let idx = (50 * 100 + 50) * 4;
        assert_eq!(result[idx + 1], 255, "green channel at center");
    }

    #[test]
    fn draw_line_rgb8_auto_converts() {
        let (px, info) = white_rgb(100, 100);
        let (result, out_info) = draw_line(&px, &info, 0.0, 0.0, 99.0, 99.0, [255, 0, 0, 255], 1.0).unwrap();
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
}
