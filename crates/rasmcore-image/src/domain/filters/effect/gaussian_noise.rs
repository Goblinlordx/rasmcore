//! Filter: gaussian_noise (category: effect)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

#[rasmcore_macros::register_filter(
    name = "gaussian_noise",
    category = "effect",
    group = "noise",
    variant = "gaussian",
    reference = "additive Gaussian noise"
)]
pub fn gaussian_noise(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &GaussianNoiseParams,
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
            gaussian_noise(r, &mut u, i8, config)
        });
    }

    let amount = config.amount as f64 / 100.0;
    if amount == 0.0 {
        return Ok(pixels.to_vec());
    }

    let mean = config.mean as f64;
    let sigma = config.sigma as f64;
    let ch = channels(info.format);
    let has_alpha = matches!(info.format, PixelFormat::Rgba8);
    let mut rng = config.seed.max(1); // avoid zero state

    let mut out = pixels.to_vec();
    for pixel in out.chunks_exact_mut(ch) {
        let color_ch = if has_alpha { ch - 1 } else { ch };
        for c in &mut pixel[..color_ch] {
            let noise = box_muller(&mut rng) * sigma + mean;
            let v = *c as f64 + noise * amount;
            *c = v.clamp(0.0, 255.0) as u8;
        }
    }
    Ok(out)
}
