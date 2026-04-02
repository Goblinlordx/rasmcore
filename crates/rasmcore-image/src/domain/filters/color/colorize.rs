//! Filter: colorize (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Colorize using the Photoshop/W3C Color blend mode algorithm.
///
/// Replaces hue and chroma of each pixel with those of `target_color`
/// while preserving the pixel's original luminance (BT.601 luma).
/// `amount` controls blend strength: 0=none, 1=full colorize.
///
/// Reference: W3C Compositing and Blending Level 1, Section 10.2
/// (SetLum/ClipColor with Lum = 0.299R + 0.587G + 0.114B).

#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct ColorizeParams {
    /// Target color to blend toward
    pub target: crate::domain::param_types::ColorRgb,
    /// Blend amount (0=none, 1=full tint)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub amount: f32,
}
impl ColorLutOp for ColorizeParams {
    fn build_clut(&self) -> ColorLut3D {
        ColorOp::Colorize(
            [
                self.target.r as f32 / 255.0,
                self.target.g as f32 / 255.0,
                self.target.b as f32 / 255.0,
            ],
            self.amount,
        )
        .to_clut(DEFAULT_CLUT_GRID)
    }
}

#[rasmcore_macros::register_filter(
    name = "colorize",
    category = "color",
    reference = "W3C Compositing Level 1 / Photoshop Color blend mode",
    color_op = "true"
)]
pub fn colorize(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &ColorizeParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let target_r = config.target.r;
    let target_g = config.target.g;
    let target_b = config.target.b;
    let amount = config.amount;

    let target_norm = [
        target_r as f32 / 255.0,
        target_g as f32 / 255.0,
        target_b as f32 / 255.0,
    ];
    apply_color_op(
        pixels,
        info,
        &ColorOp::Colorize(target_norm, amount.clamp(0.0, 1.0)),
    )
}
