//! Filter: poisson_noise (category: effect)

#[allow(unused_imports)]
use crate::domain::filters::common::*;


#[derive(rasmcore_macros::ConfigParams, Clone)]
/// Poisson noise — signal-dependent noise (brighter regions get more noise)
pub struct PoissonNoiseParams {
    /// Scale factor for Poisson lambda (higher = more visible noise)
    #[param(min = 0.0, max = 100.0, step = 0.5, default = 10.0)]
    pub scale: f32,
    /// Random seed for reproducibility
    #[param(
        min = 0,
        max = 18446744073709551615,
        step = 1,
        default = 42,
        hint = "rc.seed"
    )]
    pub seed: u64,
}

#[rasmcore_macros::register_filter(
    name = "poisson_noise",
    category = "effect",
    group = "noise",
    variant = "poisson",
    reference = "signal-dependent Poisson noise"
)]
pub fn poisson_noise(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &PoissonNoiseParams,
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
            poisson_noise(r, &mut u, i8, config)
        });
    }

    let scale = config.scale as f64;
    if scale == 0.0 {
        return Ok(pixels.to_vec());
    }

    let ch = channels(info.format);
    let has_alpha = matches!(info.format, PixelFormat::Rgba8);
    let mut rng = config.seed.max(1);

    let mut out = pixels.to_vec();
    for pixel in out.chunks_exact_mut(ch) {
        let color_ch = if has_alpha { ch - 1 } else { ch };
        for c in &mut pixel[..color_ch] {
            // Lambda is proportional to pixel value — brighter pixels get more noise
            let lambda = *c as f64 * scale;
            let noisy = poisson_random(lambda, &mut rng) / scale;
            *c = noisy.clamp(0.0, 255.0) as u8;
        }
    }
    Ok(out)
}
