//! Filter: colorize (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Colorize with selectable method.
///
/// - `"w3c"` (default): Photoshop/W3C Color blend mode — SetLum/ClipColor
///   with BT.601 luma. Industry standard. Collapses to neutral at L=0/1.
/// - `"lab"`: CIELAB perceptual — replaces a*b* chrominance with parabolic
///   weighting by L*. Preserves subtle tint at highlights/shadows.
///   Based on the libvips/sharp tint() approach.

#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "colorize", category = "color", reference = "W3C Compositing Level 1 / Photoshop Color blend mode", color_op = "true")]
pub struct ColorizeParams {
    /// Target color to blend toward
    pub target: crate::domain::param_types::ColorRgb,
    /// Blend amount (0=none, 1=full tint)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub amount: f32,
    /// Colorize method
    #[param(
        default = "w3c",
        hint = "rc.enum",
        options = "w3c:Photoshop/W3C standard — SetLum/ClipColor with BT.601 luma|lab:CIELAB perceptual — parabolic weighting, natural tint at extremes"
    )]
    pub method: String,
}
impl ColorLutOp for ColorizeParams {
    fn build_clut(&self) -> ColorLut3D {
        let target_norm = [
            self.target.r as f32 / 255.0,
            self.target.g as f32 / 255.0,
            self.target.b as f32 / 255.0,
        ];
        let amt = self.amount;
        // Empty method defaults to "w3c" (ConfigParams Default gives "" for String)
        let op = if self.method == "lab" {
            ColorOp::ColorizeLab(target_norm, amt)
        } else {
            ColorOp::Colorize(target_norm, amt)
        };
        op.to_clut(DEFAULT_CLUT_GRID)
    }
}

impl CpuFilter for ColorizeParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();

    let target_norm = [
        self.target.r as f32 / 255.0,
        self.target.g as f32 / 255.0,
        self.target.b as f32 / 255.0,
    ];
    let amount = self.amount.clamp(0.0, 1.0);

    // Empty method defaults to "w3c" (ConfigParams Default gives "" for String)
    let op = if self.method == "lab" {
        ColorOp::ColorizeLab(target_norm, amount)
    } else {
        ColorOp::Colorize(target_norm, amount)
    };
    apply_color_op(pixels, info, &op)
}
}

