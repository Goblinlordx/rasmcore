//! Filter: clarity (category: enhancement)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Clarity — midtone-weighted local contrast enhancement.
///
/// Applies a large-radius unsharp mask but weights the effect by a midtone curve:
/// shadows and highlights get less enhancement, midtones (luminance 25-75%) get full.
/// This matches Lightroom/Photoshop "Clarity" slider behavior.
///
/// - `amount`: enhancement strength (0.0-2.0 typical, 1.0 = full effect)
/// - `sigma`: blur radius for local contrast (30-50 typical)

/// Parameters for clarity (midtone local contrast).
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct ClarityParams {
    /// Enhancement strength (0.0-2.0 typical)
    #[param(min = 0.0, max = 3.0, step = 0.1, default = 1.0)]
    pub amount: f32,
    /// Blur radius for local contrast (30-50 typical)
    #[param(
        min = 5.0,
        max = 100.0,
        step = 1.0,
        default = 40.0,
        hint = "rc.log_slider"
    )]
    pub sigma: f32,
}

#[rasmcore_macros::register_filter(
    name = "clarity",
    category = "enhancement",
    reference = "midtone-weighted local contrast"
)]
pub fn clarity(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &ClarityParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let amount = config.amount;
    let sigma = config.sigma;

    validate_format(info.format)?;
    let channels = match info.format {
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "clarity requires RGB8 or RGBA8".into(),
            ));
        }
    };
    let n = (info.width as usize) * (info.height as usize);

    // Compute luminance for midtone weighting
    let mut luma = vec![0.0f32; n];
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        let pi = i * channels;
        luma[i] = (0.2126 * pixels[pi] as f32
            + 0.7152 * pixels[pi + 1] as f32
            + 0.0722 * pixels[pi + 2] as f32)
            / 255.0;
    }

    // Apply large-radius blur
    let blurred = blur_impl(pixels, info, &BlurParams { radius: sigma })?;

    // Midtone weight function: bell curve centered at 0.5, zero at 0 and 1
    // w(l) = 4 * l * (1 - l) — parabola peaking at 0.5 with w(0.5) = 1.0
    let mut result = vec![0u8; pixels.len()];
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        let weight = 4.0 * luma[i] * (1.0 - luma[i]) * amount;
        let pi = i * channels;
        for c in 0..3 {
            let orig = pixels[pi + c] as f32;
            let blur_val = blurred[pi + c] as f32;
            let detail = orig - blur_val; // high-frequency detail
            let enhanced = orig + detail * weight;
            result[pi + c] = enhanced.round().clamp(0.0, 255.0) as u8;
        }
        if channels == 4 {
            result[pi + 3] = pixels[pi + 3]; // alpha
        }
    }

    Ok(result)
}
