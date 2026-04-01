use super::super::error::ImageError;
use super::super::types::ImageInfo;
use super::ensure_rgba8;

// ─── Text Rendering (embedded bitmap font) ───────────────────────────────

/// Embedded 8×16 bitmap font glyph data for ASCII 32–126 (95 glyphs).
///
/// Each glyph is 16 bytes (one byte per row, 8 pixels wide, MSB-first).
/// This is a minimal fixed-width font suitable for annotations and debugging.
/// Based on the classic VGA 8×16 bitmap font (public domain).
const FONT_8X16: &[u8] = include_bytes!("../font_8x16.bin");
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
