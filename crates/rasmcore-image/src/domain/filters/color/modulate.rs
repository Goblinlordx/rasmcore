//! Filter: modulate (category: color)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Combined brightness/saturation/hue adjustment in HSB color space.
///
/// IM equivalent: -modulate brightness,saturation,hue
/// Uses HSB (same as HSV where B=V=max(R,G,B)), not HSL.
/// Identity at (100, 100, 0).

#[derive(rasmcore_macros::Filter, Clone)]
/// HSB modulate — combined brightness, saturation, hue adjustment.
#[filter(name = "modulate", category = "color", reference = "luma-preserving HSL modulation", color_op = "true")]
pub struct ModulateParams {
    /// Brightness percentage (100 = unchanged, 0 = black, 200 = 2x bright)
    #[param(min = 0.0, max = 200.0, step = 1.0, default = 100.0)]
    pub brightness: f32,
    /// Saturation percentage (100 = unchanged, 0 = grayscale, 200 = 2x saturated)
    #[param(min = 0.0, max = 200.0, step = 1.0, default = 100.0)]
    pub saturation: f32,
    /// Hue rotation in degrees (0 = unchanged)
    #[param(min = 0.0, max = 360.0, step = 1.0, default = 0.0)]
    pub hue: f32,
}
impl ColorLutOp for ModulateParams {
    fn build_clut(&self) -> ColorLut3D {
        ColorOp::Modulate {
            brightness: self.brightness,
            saturation: self.saturation,
            hue: self.hue,
        }
        .to_clut(DEFAULT_CLUT_GRID)
    }
}

impl CpuFilter for ModulateParams {
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
    let brightness = self.brightness;
    let saturation = self.saturation;
    let hue = self.hue;

    apply_color_op(
        pixels,
        info,
        &ColorOp::Modulate {
            brightness: brightness / 100.0,
            saturation: saturation / 100.0,
            hue,
        },
    )
}
}

