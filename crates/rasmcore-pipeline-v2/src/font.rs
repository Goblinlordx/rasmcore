//! Font resource — TTF/OTF parsing, glyph rasterization, and atlas caching.
//!
//! Consumer provides raw font bytes. The pipeline parses glyph outlines
//! and caches rasterized glyphs in an atlas. No fonts are embedded in the binary.

use std::collections::HashMap;

/// Parsed font data with glyph atlas cache.
pub struct Font {
    inner: fontdue::Font,
    /// Cached rasterized glyphs: (char, size_px as u32) -> GlyphBitmap
    atlas: std::cell::RefCell<HashMap<(char, u32), GlyphBitmap>>,
}

/// Rasterized glyph bitmap (f32 alpha coverage).
#[derive(Clone)]
pub struct GlyphBitmap {
    pub width: u32,
    pub height: u32,
    /// f32 alpha values (width * height). Coverage map for the glyph.
    pub alpha: Vec<f32>,
    /// Metrics: horizontal offset from pen position.
    pub x_offset: f32,
    /// Metrics: vertical offset from baseline.
    pub y_offset: f32,
    /// Advance width (distance to next glyph).
    pub advance: f32,
}

/// Font info returned via WIT.
pub struct FontInfo {
    pub units_per_em: u16,
    pub ascender: i16,
    pub descender: i16,
    pub num_glyphs: u16,
}

impl Font {
    /// Parse a TTF/OTF font from raw bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let settings = fontdue::FontSettings::default();
        let inner = fontdue::Font::from_bytes(data, settings)
            .map_err(|e| format!("font parse error: {e}"))?;
        Ok(Self {
            inner,
            atlas: std::cell::RefCell::new(HashMap::new()),
        })
    }

    /// Get font metadata.
    pub fn info(&self) -> FontInfo {
        FontInfo {
            units_per_em: self.inner.units_per_em() as u16,
            ascender: self.inner.horizontal_line_metrics(1.0)
                .map(|m| m.ascent as i16)
                .unwrap_or(0),
            descender: self.inner.horizontal_line_metrics(1.0)
                .map(|m| m.descent as i16)
                .unwrap_or(0),
            num_glyphs: 0,
        }
    }

    /// Rasterize a glyph at the given pixel size. Cached per (char, size).
    pub fn rasterize(&self, ch: char, size_px: f32) -> GlyphBitmap {
        let key = (ch, size_px as u32);
        {
            let atlas = self.atlas.borrow();
            if let Some(cached) = atlas.get(&key) {
                return cached.clone();
            }
        }

        let (metrics, bitmap) = self.inner.rasterize(ch, size_px);
        let alpha: Vec<f32> = bitmap.iter().map(|&b| b as f32 / 255.0).collect();

        let glyph = GlyphBitmap {
            width: metrics.width as u32,
            height: metrics.height as u32,
            alpha,
            x_offset: metrics.xmin as f32,
            y_offset: metrics.ymin as f32,
            advance: metrics.advance_width,
        };

        self.atlas.borrow_mut().insert(key, glyph.clone());
        glyph
    }

    /// Render a text string into the given f32 RGBA pixel buffer.
    ///
    /// Composites glyph alpha coverage onto the existing pixel data.
    /// `x`, `y` are the pen position (baseline origin).
    /// `color` is [R, G, B, A] in the pipeline's working color space.
    pub fn render_text(
        &self,
        pixels: &mut [f32],
        img_w: u32,
        img_h: u32,
        text: &str,
        x: f32,
        y: f32,
        size_px: f32,
        color: [f32; 4],
    ) {
        let mut pen_x = x;
        let line_metrics = self.inner.horizontal_line_metrics(size_px);
        let ascent = line_metrics.map(|m| m.ascent).unwrap_or(size_px * 0.8);

        for ch in text.chars() {
            if ch == ' ' {
                let glyph = self.rasterize(' ', size_px);
                pen_x += glyph.advance;
                continue;
            }

            let glyph = self.rasterize(ch, size_px);
            let gx = (pen_x + glyph.x_offset).round() as i32;
            let gy = (y + ascent - glyph.y_offset - glyph.height as f32).round() as i32;

            // Blit glyph alpha onto pixel buffer
            for gy_off in 0..glyph.height as i32 {
                let py = gy + gy_off;
                if py < 0 || py >= img_h as i32 { continue; }
                for gx_off in 0..glyph.width as i32 {
                    let px = gx + gx_off;
                    if px < 0 || px >= img_w as i32 { continue; }

                    let alpha_idx = (gy_off as u32 * glyph.width + gx_off as u32) as usize;
                    let coverage = glyph.alpha[alpha_idx] * color[3];
                    if coverage < 1e-6 { continue; }

                    let pi = ((py as u32 * img_w + px as u32) * 4) as usize;
                    // Pre-multiplied alpha blend
                    let inv = 1.0 - coverage;
                    pixels[pi]     = pixels[pi]     * inv + color[0] * coverage;
                    pixels[pi + 1] = pixels[pi + 1] * inv + color[1] * coverage;
                    pixels[pi + 2] = pixels[pi + 2] * inv + color[2] * coverage;
                    pixels[pi + 3] = pixels[pi + 3] * inv + coverage;
                }
            }

            pen_x += glyph.advance;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Minimal valid TTF — we can't bundle a real font, but we can test error handling
    #[test]
    fn font_from_invalid_bytes_returns_error() {
        let result = Font::from_bytes(b"not a font");
        assert!(result.is_err());
    }
}
