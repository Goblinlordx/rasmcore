//! Filter: frequency_low (category: enhancement)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Frequency separation — low-pass (structure) layer.
///
/// Returns the low-frequency component of the image: large-scale color and
/// tonal structure with fine detail removed. Computed as a Gaussian blur of
/// the input at the given sigma.
///
/// The low-pass and high-pass layers satisfy: `original = low + high - 128`
/// (per channel, for 8-bit images).
///
/// - `sigma`: Gaussian blur radius controlling the separation frequency.
///   Higher sigma puts more detail into the low-pass (smoother high-pass).
///   Typical values: 2-10 for skin retouching, 10-30 for artistic effects.
///
/// Parameters for frequency separation — low-pass (structure) layer.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "frequency_low", category = "enhancement", group = "frequency", variant = "low", reference = "Gaussian low-pass separation")]
pub struct FrequencyLowParams {
    /// Gaussian sigma controlling separation frequency (higher = more in low-pass)
    #[param(
        min = 0.5,
        max = 50.0,
        step = 0.5,
        default = 4.0,
        hint = "rc.log_slider"
    )]
    pub sigma: f32,
}

impl CpuFilter for FrequencyLowParams {
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
    let sigma = self.sigma;

    validate_format(info.format)?;
    if sigma <= 0.0 {
        return Ok(pixels.to_vec());
    }
    blur_impl(pixels, info, &BlurParams { radius: sigma })
}
}

