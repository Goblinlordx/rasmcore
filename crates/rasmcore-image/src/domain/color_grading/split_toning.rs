use super::{ImageError, ImageInfo, apply_rgb_transform};

/// Split toning parameters — tint shadows and highlights with different colors.
///
/// `shadow_color` and `highlight_color` are RGB tint values in [0, 1].
/// `balance` shifts the crossover point: negative = more shadow tint, positive = more highlight tint.
/// `strength` controls overall intensity (0 = no tinting, 1 = full).
#[derive(Debug, Clone, Copy)]
pub struct SplitToning {
    pub shadow_color: [f32; 3],
    pub highlight_color: [f32; 3],
    /// Balance: -1.0 (all shadow) to +1.0 (all highlight). Default: 0.0.
    pub balance: f32,
    /// Strength: 0.0 (none) to 1.0 (full). Default: 0.5.
    pub strength: f32,
}

impl Default for SplitToning {
    fn default() -> Self {
        Self {
            shadow_color: [0.0, 0.0, 0.5],    // Blue shadows
            highlight_color: [1.0, 0.8, 0.4], // Warm highlights
            balance: 0.0,
            strength: 0.5,
        }
    }
}

/// Apply split toning to a single pixel.
#[inline]
pub fn split_toning_pixel(r: f32, g: f32, b: f32, st: &SplitToning) -> (f32, f32, f32) {
    // Luminance (Rec. 709)
    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;

    // Crossover point shifted by balance
    let midpoint = 0.5 + st.balance * 0.5;

    // Shadow/highlight blend factor
    let shadow_weight = (1.0 - luma / midpoint.max(0.001)).clamp(0.0, 1.0) * st.strength;
    let highlight_weight =
        ((luma - midpoint) / (1.0 - midpoint).max(0.001)).clamp(0.0, 1.0) * st.strength;

    // Blend: tint toward shadow/highlight color
    let out_r = r
        + (st.shadow_color[0] - r) * shadow_weight
        + (st.highlight_color[0] - r) * highlight_weight;
    let out_g = g
        + (st.shadow_color[1] - g) * shadow_weight
        + (st.highlight_color[1] - g) * highlight_weight;
    let out_b = b
        + (st.shadow_color[2] - b) * shadow_weight
        + (st.highlight_color[2] - b) * highlight_weight;

    (
        out_r.clamp(0.0, 1.0),
        out_g.clamp(0.0, 1.0),
        out_b.clamp(0.0, 1.0),
    )
}

/// Apply split toning to an image pixel buffer.
pub fn split_toning(
    pixels: &[u8],
    info: &ImageInfo,
    st: &SplitToning,
) -> Result<Vec<u8>, ImageError> {
    apply_rgb_transform(pixels, info, |r, g, b| split_toning_pixel(r, g, b, st))
}
