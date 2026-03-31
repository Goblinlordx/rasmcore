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
    resvg::tiny_skia::Stroke {
        width,
        ..resvg::tiny_skia::Stroke::default()
    }
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

/// Draw a polygon on the image from a list of vertices.
///
/// Vertices are (x, y) pairs. The path is closed automatically.
/// If `filled` is true, the polygon is filled. Otherwise only the outline
/// is drawn with the given `stroke_width`.
pub fn draw_polygon(
    pixels: &[u8],
    info: &ImageInfo,
    vertices: &[(f32, f32)],
    fill_color: [u8; 4],
    stroke_color: [u8; 4],
    stroke_width: f32,
    filled: bool,
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    if vertices.len() < 3 {
        return Err(ImageError::InvalidParameters(
            "draw_polygon: need at least 3 vertices".into(),
        ));
    }
    let (rgba, out_info) = ensure_rgba8(pixels, info)?;
    let mut pixmap = pixels_to_pixmap(&rgba, out_info.width, out_info.height)?;

    let mut pb = resvg::tiny_skia::PathBuilder::new();
    pb.move_to(vertices[0].0, vertices[0].1);
    for &(x, y) in &vertices[1..] {
        pb.line_to(x, y);
    }
    pb.close();
    let path = pb.finish().ok_or_else(|| {
        ImageError::InvalidParameters("draw_polygon: invalid path coordinates".into())
    })?;

    if filled {
        let paint = make_paint(fill_color[0], fill_color[1], fill_color[2], fill_color[3]);
        pixmap.fill_path(
            &path,
            &paint,
            resvg::tiny_skia::FillRule::Winding,
            resvg::tiny_skia::Transform::identity(),
            None,
        );
    }

    if stroke_width > 0.0 {
        let paint = make_paint(
            stroke_color[0],
            stroke_color[1],
            stroke_color[2],
            stroke_color[3],
        );
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

/// Draw an ellipse on the image.
///
/// `cx`, `cy` are the center, `rx` and `ry` are the radii along the X and Y
/// axes. Uses `tiny_skia::PathBuilder::from_oval` for native-quality bezier
/// approximation.
pub fn draw_ellipse(
    pixels: &[u8],
    info: &ImageInfo,
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    color: [u8; 4],
    stroke_width: f32,
    filled: bool,
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    if rx <= 0.0 || ry <= 0.0 {
        return Err(ImageError::InvalidParameters(format!(
            "draw_ellipse: radii must be positive (rx={rx}, ry={ry})"
        )));
    }
    let (rgba, out_info) = ensure_rgba8(pixels, info)?;
    let mut pixmap = pixels_to_pixmap(&rgba, out_info.width, out_info.height)?;

    let oval_rect =
        resvg::tiny_skia::Rect::from_xywh(cx - rx, cy - ry, rx * 2.0, ry * 2.0).ok_or_else(
            || {
                ImageError::InvalidParameters(format!(
                    "draw_ellipse: invalid oval ({cx},{cy},{rx},{ry})"
                ))
            },
        )?;
    let path = resvg::tiny_skia::PathBuilder::from_oval(oval_rect).ok_or_else(|| {
        ImageError::InvalidParameters("draw_ellipse: failed to build oval path".into())
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

/// Draw an arc (partial ellipse outline) on the image.
///
/// `cx`, `cy` are the center, `rx` and `ry` are the radii.
/// `start_angle` and `end_angle` are in degrees (0 = right, counter-clockwise).
/// The arc is always stroked (not filled).
///
/// Uses cubic bezier segments to approximate each quadrant of the arc.
pub fn draw_arc(
    pixels: &[u8],
    info: &ImageInfo,
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    start_angle: f32,
    end_angle: f32,
    color: [u8; 4],
    stroke_width: f32,
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    if rx <= 0.0 || ry <= 0.0 {
        return Err(ImageError::InvalidParameters(format!(
            "draw_arc: radii must be positive (rx={rx}, ry={ry})"
        )));
    }
    let (rgba, out_info) = ensure_rgba8(pixels, info)?;
    let mut pixmap = pixels_to_pixmap(&rgba, out_info.width, out_info.height)?;

    let path = build_arc_path(cx, cy, rx, ry, start_angle, end_angle).ok_or_else(|| {
        ImageError::InvalidParameters("draw_arc: failed to build arc path".into())
    })?;

    let paint = make_paint(color[0], color[1], color[2], color[3]);
    let stroke = make_stroke(stroke_width);
    pixmap.stroke_path(
        &path,
        &paint,
        &stroke,
        resvg::tiny_skia::Transform::identity(),
        None,
    );

    Ok((pixmap_to_pixels(&pixmap), out_info))
}

/// Build a cubic bezier arc path. Splits the angular range into segments of
/// at most 90 degrees each, using the standard bezier arc approximation.
fn build_arc_path(
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    start_deg: f32,
    end_deg: f32,
) -> Option<resvg::tiny_skia::Path> {
    let mut pb = resvg::tiny_skia::PathBuilder::new();

    let start_rad = start_deg.to_radians();
    let end_rad = end_deg.to_radians();

    // Normalize: ensure we sweep in the positive direction
    let mut sweep = end_rad - start_rad;
    if sweep.abs() < 1e-6 {
        return None;
    }
    if sweep < 0.0 {
        sweep += std::f32::consts::TAU;
    }

    // Split into segments of at most PI/2 (90 degrees)
    let max_segment = std::f32::consts::FRAC_PI_2;
    let n_segments = (sweep / max_segment).ceil() as usize;
    let segment_angle = sweep / n_segments as f32;

    // Start point
    let sx = cx + rx * start_rad.cos();
    let sy = cy - ry * start_rad.sin();
    pb.move_to(sx, sy);

    let mut angle = start_rad;
    for _ in 0..n_segments {
        let next_angle = angle + segment_angle;
        arc_bezier_segment(&mut pb, cx, cy, rx, ry, angle, next_angle);
        angle = next_angle;
    }

    pb.finish()
}

/// Append a single cubic bezier segment approximating an elliptical arc.
///
/// Uses the standard parametric bezier approximation for circular arcs
/// scaled by (rx, ry). The magic factor `alpha = 4/3 * tan(da/4)` ensures
/// the bezier curve passes through the arc endpoints with correct tangents.
fn arc_bezier_segment(
    pb: &mut resvg::tiny_skia::PathBuilder,
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    a1: f32,
    a2: f32,
) {
    let da = a2 - a1;
    let alpha = (da / 4.0).tan() * 4.0 / 3.0;

    let cos1 = a1.cos();
    let sin1 = a1.sin();
    let cos2 = a2.cos();
    let sin2 = a2.sin();

    // Control point 1: tangent at start
    let cp1x = cx + rx * (cos1 - alpha * sin1);
    let cp1y = cy - ry * (sin1 + alpha * cos1);

    // Control point 2: tangent at end
    let cp2x = cx + rx * (cos2 + alpha * sin2);
    let cp2y = cy - ry * (sin2 - alpha * cos2);

    // End point
    let ex = cx + rx * cos2;
    let ey = cy - ry * sin2;

    pb.cubic_to(cp1x, cp1y, cp2x, cp2y, ex, ey);
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
        let glyph_index = if (GLYPH_FIRST..=GLYPH_LAST).contains(&ch) {
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
                                output[idx] = ((color[0] as u16 * ta + output[idx] as u16 * inv_a)
                                    / 255) as u8;
                                output[idx + 1] = ((color[1] as u16 * ta
                                    + output[idx + 1] as u16 * inv_a)
                                    / 255) as u8;
                                output[idx + 2] = ((color[2] as u16 * ta
                                    + output[idx + 2] as u16 * inv_a)
                                    / 255) as u8;
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

// ─── TrueType / OpenType Font Rendering ─────────────────────────────────
//
// Full pipeline: unicode-bidi → unicode-linebreak → rustybuzz → fontdue
//   1. Bidi: detect paragraph direction, split into directional runs
//   2. Line break: find UAX#14 break opportunities for word wrap
//   3. Shape: rustybuzz produces glyph IDs + positions (ligatures, kerning)
//   4. Rasterize: fontdue renders each glyph bitmap, alpha-blend onto image

/// Shape a text run using rustybuzz and return positioned glyphs.
///
/// Returns Vec<(glyph_id, x_advance, x_offset, y_offset)> in font units.
fn shape_run(
    face: &rustybuzz::Face<'_>,
    text: &str,
    direction: rustybuzz::Direction,
) -> Vec<(u16, i32, i32, i32)> {
    let mut buffer = rustybuzz::UnicodeBuffer::new();
    buffer.push_str(text);
    buffer.set_direction(direction);
    let output = rustybuzz::shape(face, &[], buffer);
    let positions = output.glyph_positions();
    let infos = output.glyph_infos();
    infos
        .iter()
        .zip(positions.iter())
        .map(|(info, pos)| {
            (
                info.glyph_id as u16,
                pos.x_advance,
                pos.x_offset,
                pos.y_offset,
            )
        })
        .collect()
}

/// Blend a single glyph bitmap onto the output buffer.
#[allow(clippy::too_many_arguments)]
fn blend_glyph(
    output: &mut [u8],
    w: usize,
    h: usize,
    gx: i32,
    gy: i32,
    bitmap: &[u8],
    glyph_w: usize,
    glyph_h: usize,
    color_r: u16,
    color_g: u16,
    color_b: u16,
    color_a: u16,
) {
    for row in 0..glyph_h {
        for col in 0..glyph_w {
            let px_x = gx + col as i32;
            let px_y = gy + row as i32;
            if px_x < 0 || px_y < 0 || px_x >= w as i32 || px_y >= h as i32 {
                continue;
            }
            let coverage = bitmap[row * glyph_w + col] as u16;
            if coverage == 0 {
                continue;
            }
            let alpha = (color_a * coverage) / 255;
            let inv_alpha = 255 - alpha;
            let idx = (px_y as usize * w + px_x as usize) * 4;
            output[idx] = ((color_r * alpha + output[idx] as u16 * inv_alpha) / 255) as u8;
            output[idx + 1] = ((color_g * alpha + output[idx + 1] as u16 * inv_alpha) / 255) as u8;
            output[idx + 2] = ((color_b * alpha + output[idx + 2] as u16 * inv_alpha) / 255) as u8;
            output[idx + 3] = output[idx + 3].max(alpha as u8);
        }
    }
}

/// Draw text using a TrueType/OpenType font with full shaping pipeline.
///
/// Pipeline: unicode-bidi → rustybuzz (shaping) → fontdue (rasterize).
/// Supports ligatures, kerning, RTL/bidi, multi-line via '\n'.
/// `font_data`: raw TTF/OTF bytes. `font_size_pt`: point size at 96 DPI.
pub fn draw_text_ttf(
    pixels: &[u8],
    info: &ImageInfo,
    x: u32,
    y: u32,
    text: &str,
    font_data: &[u8],
    font_size_pt: f32,
    color: [u8; 4],
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    let (rgba, out_info) = ensure_rgba8(pixels, info)?;
    let mut output = rgba;
    let w = out_info.width as usize;
    let h = out_info.height as usize;

    // Parse font for rasterization (fontdue)
    let fd_settings = fontdue::FontSettings::default();
    let fd_font = fontdue::Font::from_bytes(font_data, fd_settings)
        .map_err(|e| ImageError::InvalidInput(format!("fontdue parse error: {e}")))?;

    // Parse font for shaping (rustybuzz)
    let rb_face = rustybuzz::Face::from_slice(font_data, 0)
        .ok_or_else(|| ImageError::InvalidInput("rustybuzz: failed to parse font".into()))?;

    // Convert pt to px
    let px_size = font_size_pt * (96.0 / 72.0);

    // Line metrics
    let line_metrics = fd_font
        .horizontal_line_metrics(px_size)
        .unwrap_or(fontdue::LineMetrics {
            ascent: px_size * 0.8,
            descent: px_size * -0.2,
            line_gap: 0.0,
            new_line_size: px_size * 1.2,
        });
    let line_height = line_metrics.new_line_size;

    // Scale factor: rustybuzz works in font units, we need pixels
    let upem = rb_face.units_per_em() as f32;
    let scale = px_size / upem;

    let color_r = color[0] as u16;
    let color_g = color[1] as u16;
    let color_b = color[2] as u16;
    let color_a = color[3] as u16;

    let mut cursor_y = y as f32 + line_metrics.ascent;

    for paragraph in text.split('\n') {
        if paragraph.is_empty() {
            cursor_y += line_height;
            continue;
        }

        // Bidi: detect paragraph direction and get visual runs
        let bidi_info = unicode_bidi::BidiInfo::new(paragraph, None);
        let para = &bidi_info.paragraphs[0];
        let line_range = para.range.clone();
        let (levels, runs) = bidi_info.visual_runs(para, line_range);

        let mut cursor_x = x as f32;

        // Process each visual run (already in display order)
        for run in &runs {
            let run_text = &paragraph[run.clone()];
            if run_text.is_empty() {
                continue;
            }

            // Detect direction from bidi level at run start
            let first_level = levels[run.start];
            let direction = if first_level.is_rtl() {
                rustybuzz::Direction::RightToLeft
            } else {
                rustybuzz::Direction::LeftToRight
            };

            // Shape the run
            let glyphs = shape_run(&rb_face, run_text, direction);

            // Render each shaped glyph
            for (glyph_id, x_advance, x_offset, y_offset) in &glyphs {
                let (metrics, bitmap) = fd_font.rasterize_indexed(*glyph_id, px_size);

                let gx = cursor_x as i32 + ((*x_offset as f32) * scale) as i32 + metrics.xmin;
                let gy = cursor_y as i32 + ((*y_offset as f32) * scale) as i32
                    - metrics.height as i32
                    - metrics.ymin;

                blend_glyph(
                    &mut output,
                    w,
                    h,
                    gx,
                    gy,
                    &bitmap,
                    metrics.width,
                    metrics.height,
                    color_r,
                    color_g,
                    color_b,
                    color_a,
                );

                cursor_x += (*x_advance as f32) * scale;
            }
        }

        cursor_y += line_height;
    }

    Ok((output, out_info))
}

/// Draw text with optional TrueType font. Falls back to bitmap when font_data is None.
pub fn draw_text_auto(
    pixels: &[u8],
    info: &ImageInfo,
    x: u32,
    y: u32,
    text: &str,
    font_data: Option<&[u8]>,
    font_size_pt: f32,
    color: [u8; 4],
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    match font_data {
        Some(data) => draw_text_ttf(pixels, info, x, y, text, data, font_size_pt, color),
        None => {
            let scale = (font_size_pt / 12.0).round().max(1.0) as u32;
            draw_text(pixels, info, x, y, text, scale, color)
        }
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
        let (px, info) = white_rgba(100, 100);
        let verts = vec![(50.0, 10.0), (10.0, 90.0), (90.0, 90.0)];
        let (result, _) = draw_polygon(
            &px,
            &info,
            &verts,
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
        let (px, info) = white_rgba(100, 100);
        let verts = vec![(20.0, 20.0), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0)];
        let (result, _) = draw_polygon(
            &px,
            &info,
            &verts,
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
        assert_ne!(result[edge..edge + 3], [255, 255, 255], "edge should be stroked");
    }

    #[test]
    fn draw_polygon_too_few_vertices_errors() {
        let (px, info) = white_rgba(100, 100);
        let result = draw_polygon(
            &px,
            &info,
            &[(10.0, 10.0), (20.0, 20.0)],
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
        let (result, _) =
            draw_ellipse(&px, &info, 50.0, 50.0, 30.0, 15.0, [0, 255, 0, 255], 1.0, true)
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
        assert!(draw_arc(&px, &info, 50.0, 50.0, 0.0, 30.0, 0.0, 90.0, [0, 0, 0, 255], 1.0).is_err());
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
