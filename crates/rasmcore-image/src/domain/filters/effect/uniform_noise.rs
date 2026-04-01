//! Filter: uniform_noise (category: effect)

#[allow(unused_imports)]
use crate::domain::filters::common::*;


#[derive(rasmcore_macros::ConfigParams, Clone)]
/// Uniform noise — adds uniformly distributed random noise
pub struct UniformNoiseParams {
    /// Noise range: values are added in [-range, +range]
    #[param(min = 0.0, max = 128.0, step = 0.5, default = 20.0)]
    pub range: f32,
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
    name = "uniform_noise",
    category = "effect",
    group = "noise",
    variant = "uniform",
    reference = "additive uniform noise"
)]
pub fn uniform_noise(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &UniformNoiseParams,
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
            uniform_noise(r, &mut u, i8, config)
        });
    }

    let range = config.range as f64;
    if range == 0.0 {
        return Ok(pixels.to_vec());
    }

    let ch = channels(info.format);
    let has_alpha = matches!(info.format, PixelFormat::Rgba8);
    let mut rng = config.seed.max(1);

    let mut out = pixels.to_vec();
    for pixel in out.chunks_exact_mut(ch) {
        let color_ch = if has_alpha { ch - 1 } else { ch };
        for c in &mut pixel[..color_ch] {
            // Uniform in [-range, +range]
            let noise = (xorshift64_f64(&mut rng) * 2.0 - 1.0) * range;
            let v = *c as f64 + noise;
            *c = v.clamp(0.0, 255.0) as u8;
        }
    }
    Ok(out)
}
