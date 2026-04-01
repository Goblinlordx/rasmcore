//! Filter: color_balance (category: adjustment)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Photoshop-style color balance — per-tonal-range CMY-RGB adjustment.
///
/// Shifts colors along Cyan-Red, Magenta-Green, and Yellow-Blue axes
/// independently for shadows, midtones, and highlights. Tonal ranges use
/// smooth luminance-based weighting with Rec. 709 luma coefficients.

/// Parameters for Photoshop-style color balance adjustment.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct ColorBalanceParams {
    /// Shadow Cyan-Red (-100 to 100)
    #[param(min = -100.0, max = 100.0, step = 1.0, default = 0.0, hint = "rc.signed_slider")]
    pub shadow_cyan_red: f32,
    /// Shadow Magenta-Green (-100 to 100)
    #[param(min = -100.0, max = 100.0, step = 1.0, default = 0.0, hint = "rc.signed_slider")]
    pub shadow_magenta_green: f32,
    /// Shadow Yellow-Blue (-100 to 100)
    #[param(min = -100.0, max = 100.0, step = 1.0, default = 0.0, hint = "rc.signed_slider")]
    pub shadow_yellow_blue: f32,
    /// Midtone Cyan-Red (-100 to 100)
    #[param(min = -100.0, max = 100.0, step = 1.0, default = 0.0, hint = "rc.signed_slider")]
    pub midtone_cyan_red: f32,
    /// Midtone Magenta-Green (-100 to 100)
    #[param(min = -100.0, max = 100.0, step = 1.0, default = 0.0, hint = "rc.signed_slider")]
    pub midtone_magenta_green: f32,
    /// Midtone Yellow-Blue (-100 to 100)
    #[param(min = -100.0, max = 100.0, step = 1.0, default = 0.0, hint = "rc.signed_slider")]
    pub midtone_yellow_blue: f32,
    /// Highlight Cyan-Red (-100 to 100)
    #[param(min = -100.0, max = 100.0, step = 1.0, default = 0.0, hint = "rc.signed_slider")]
    pub highlight_cyan_red: f32,
    /// Highlight Magenta-Green (-100 to 100)
    #[param(min = -100.0, max = 100.0, step = 1.0, default = 0.0, hint = "rc.signed_slider")]
    pub highlight_magenta_green: f32,
    /// Highlight Yellow-Blue (-100 to 100)
    #[param(min = -100.0, max = 100.0, step = 1.0, default = 0.0, hint = "rc.signed_slider")]
    pub highlight_yellow_blue: f32,
    /// Preserve luminosity after color shifts
    #[param(default = true)]
    pub preserve_luminosity: bool,
}
impl ColorLutOp for ColorBalanceParams {
    fn build_clut(&self) -> ColorLut3D {
        let cb = crate::domain::color_grading::ColorBalance {
            shadow: [
                self.shadow_cyan_red / 100.0,
                self.shadow_magenta_green / 100.0,
                self.shadow_yellow_blue / 100.0,
            ],
            midtone: [
                self.midtone_cyan_red / 100.0,
                self.midtone_magenta_green / 100.0,
                self.midtone_yellow_blue / 100.0,
            ],
            highlight: [
                self.highlight_cyan_red / 100.0,
                self.highlight_magenta_green / 100.0,
                self.highlight_yellow_blue / 100.0,
            ],
            preserve_luminosity: self.preserve_luminosity,
        };
        ColorLut3D::from_fn(DEFAULT_CLUT_GRID, move |r, g, b| {
            crate::domain::color_grading::color_balance_pixel(r, g, b, &cb)
        })
    }
}

#[rasmcore_macros::register_filter(
    name = "color_balance",
    category = "adjustment",
    reference = "Photoshop color balance (shadow/midtone/highlight CMY-RGB)",
    color_op = "true"
)]
pub fn color_balance(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &ColorBalanceParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    validate_format(info.format)?;
    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            color_balance(r, &mut u, i8, config)
        });
    }
    let cb = crate::domain::color_grading::ColorBalance {
        shadow: [
            config.shadow_cyan_red / 100.0,
            config.shadow_magenta_green / 100.0,
            config.shadow_yellow_blue / 100.0,
        ],
        midtone: [
            config.midtone_cyan_red / 100.0,
            config.midtone_magenta_green / 100.0,
            config.midtone_yellow_blue / 100.0,
        ],
        highlight: [
            config.highlight_cyan_red / 100.0,
            config.highlight_magenta_green / 100.0,
            config.highlight_yellow_blue / 100.0,
        ],
        preserve_luminosity: config.preserve_luminosity,
    };
    crate::domain::color_grading::color_balance(pixels, info, &cb)
}
