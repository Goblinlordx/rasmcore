//! Filter: split_toning (category: grading)

#[allow(unused_imports)]
use crate::domain::filters::common::*;


#[derive(rasmcore_macros::ConfigParams, Clone)]
/// Split toning — tint shadows and highlights with different hues
pub struct SplitToningParams {
    /// Highlight hue (degrees)
    #[param(
        min = 0.0,
        max = 360.0,
        step = 1.0,
        default = 40.0,
        hint = "rc.angle_deg"
    )]
    pub highlight_hue: f32,
    /// Shadow hue (degrees)
    #[param(
        min = 0.0,
        max = 360.0,
        step = 1.0,
        default = 220.0,
        hint = "rc.angle_deg"
    )]
    pub shadow_hue: f32,
    /// Balance (-1 = all shadow, +1 = all highlight)
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.0, hint = "rc.signed_slider")]
    pub balance: f32,
}
impl ColorLutOp for SplitToningParams {
    fn build_clut(&self) -> ColorLut3D {
        let st = crate::domain::color_grading::SplitToning {
            shadow_color: hue_to_rgb_tint(self.shadow_hue),
            highlight_color: hue_to_rgb_tint(self.highlight_hue),
            balance: self.balance,
            strength: 0.5,
        };
        ColorLut3D::from_fn(DEFAULT_CLUT_GRID, move |r, g, b| {
            crate::domain::color_grading::split_toning_pixel(r, g, b, &st)
        })
    }
}

#[rasmcore_macros::register_filter(
    name = "split_toning",
    category = "grading",
    reference = "shadow/highlight hue tinting",
    color_op = "true"
)]
pub fn split_toning_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &SplitToningParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let highlight_hue = config.highlight_hue;
    let shadow_hue = config.shadow_hue;
    let balance = config.balance;

    let st = crate::domain::color_grading::SplitToning {
        highlight_color: hue_to_rgb_tint(highlight_hue),
        shadow_color: hue_to_rgb_tint(shadow_hue),
        balance,
        strength: 0.5,
    };
    crate::domain::color_grading::split_toning(pixels, info, &st)
}
