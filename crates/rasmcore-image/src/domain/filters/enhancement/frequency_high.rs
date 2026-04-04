//! Filter: frequency_high (category: enhancement)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Frequency separation — high-pass (detail) layer.
///
/// Returns the high-frequency component of the image: fine texture and detail
/// with large-scale color/tone removed. Computed as `original - blur + 128`
/// per channel, where 128 is the neutral mid-gray offset for u8 storage.
///
/// The low-pass and high-pass layers satisfy: `original = low + high - 128`
/// (per channel, for 8-bit images).
///
/// - `sigma`: Gaussian blur radius controlling the separation frequency.
///   Higher sigma captures finer detail in the high-pass.
///   Typical values: 2-10 for skin retouching, 10-30 for artistic effects.
///
/// Parameters for frequency separation — high-pass (detail) layer.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "frequency_high", category = "enhancement", group = "frequency", variant = "high", reference = "Gaussian high-pass separation")]
pub struct FrequencyHighParams {
    /// Gaussian sigma controlling separation frequency (higher = finer detail in high-pass)
    #[param(
        min = 0.5,
        max = 50.0,
        step = 0.5,
        default = 4.0,
        hint = "rc.log_slider"
    )]
    pub sigma: f32,
}

impl CpuFilter for FrequencyHighParams {
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
        // No blur means no low-pass content → high-pass is all-128 (neutral)
        return Ok(vec![128u8; pixels.len()]);
    }

    // 16-bit path: compute in f32 for precision
    if is_16bit(info.format) {
        let orig_f32 = u16_pixels_to_f32(pixels);
        let blurred = blur_impl(pixels, info, &BlurParams { radius: sigma })?;
        let blur_f32 = u16_pixels_to_f32(&blurred);
        // high = orig - blur + 0.5 (mid-gray in normalized [0,1])
        let result_f32: Vec<f32> = orig_f32
            .iter()
            .zip(blur_f32.iter())
            .map(|(&o, &b)| (o - b + 0.5).clamp(0.0, 1.0))
            .collect();
        return Ok(f32_to_u16_pixels(&result_f32));
    }

    let blurred = blur_impl(pixels, info, &BlurParams { radius: sigma })?;
    let ch = channels(info.format);
    let n = pixels.len();
    let mut result = vec![0u8; n];

    // SIMD-friendly loop: simple per-sample arithmetic that LLVM
    // auto-vectorizes to SIMD128 when compiled with +simd128.
    // high = clamp(original - blur + 128, 0, 255)
    for i in 0..n {
        // Alpha channel: preserve from original
        if ch == 4 && i % 4 == 3 {
            result[i] = pixels[i];
        } else {
            let diff = pixels[i] as i16 - blurred[i] as i16 + 128;
            result[i] = diff.clamp(0, 255) as u8;
        }
    }

    Ok(result)
}
}

