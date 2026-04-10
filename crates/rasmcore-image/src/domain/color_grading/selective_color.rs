use super::{ImageError, ImageInfo, apply_rgb_transform};

// ─── Color Balance (Photoshop-style) ─────────────────────────────────────

/// Photoshop-style color balance — per-tonal-range CMY-RGB shifts.
#[derive(Debug, Clone, Copy)]
pub struct ColorBalance {
    /// Shadow [cyan_red, magenta_green, yellow_blue] in [-1, 1]
    pub shadow: [f32; 3],
    /// Midtone [cyan_red, magenta_green, yellow_blue] in [-1, 1]
    pub midtone: [f32; 3],
    /// Highlight [cyan_red, magenta_green, yellow_blue] in [-1, 1]
    pub highlight: [f32; 3],
    /// Preserve luminosity after color shifts (PS default: true)
    pub preserve_luminosity: bool,
}

impl Default for ColorBalance {
    fn default() -> Self {
        Self {
            shadow: [0.0; 3],
            midtone: [0.0; 3],
            highlight: [0.0; 3],
            preserve_luminosity: true,
        }
    }
}

/// Apply color balance to a single pixel.
#[inline]
pub fn color_balance_pixel(r: f32, g: f32, b: f32, cb: &ColorBalance) -> (f32, f32, f32) {
    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;

    let shadow_w = ((1.0 - luma) * (1.0 - luma) * 1.5).min(1.0);
    let highlight_w = (luma * luma * 1.5).min(1.0);
    let midtone_w = (1.0 - shadow_w - highlight_w).max(0.0);

    let dr = cb.shadow[0] * shadow_w + cb.midtone[0] * midtone_w + cb.highlight[0] * highlight_w;
    let dg = cb.shadow[1] * shadow_w + cb.midtone[1] * midtone_w + cb.highlight[1] * highlight_w;
    let db = cb.shadow[2] * shadow_w + cb.midtone[2] * midtone_w + cb.highlight[2] * highlight_w;

    let mut out_r = (r + dr).clamp(0.0, 1.0);
    let mut out_g = (g + dg).clamp(0.0, 1.0);
    let mut out_b = (b + db).clamp(0.0, 1.0);

    if cb.preserve_luminosity {
        let new_luma = 0.2126 * out_r + 0.7152 * out_g + 0.0722 * out_b;
        if new_luma > 1e-6 {
            let scale = luma / new_luma;
            out_r = (out_r * scale).clamp(0.0, 1.0);
            out_g = (out_g * scale).clamp(0.0, 1.0);
            out_b = (out_b * scale).clamp(0.0, 1.0);
        }
    }

    (out_r, out_g, out_b)
}

/// Apply color balance to an image pixel buffer.
pub fn color_balance(
    pixels: &[u8],
    info: &ImageInfo,
    cb: &ColorBalance,
) -> Result<Vec<u8>, ImageError> {
    apply_rgb_transform(pixels, info, |r, g, b| color_balance_pixel(r, g, b, cb))
}
