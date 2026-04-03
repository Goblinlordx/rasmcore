//! Filter: salt_pepper_noise (category: effect)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;


#[derive(rasmcore_macros::Filter, Clone)]
/// Salt-and-pepper noise — randomly sets pixels to black or white
#[filter(name = "salt_pepper_noise", category = "effect", group = "noise", variant = "salt_pepper", reference = "impulse noise (salt and pepper)")]
pub struct SaltPepperNoiseParams {
    /// Density of noise pixels (0 = none, 1 = all replaced)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.05)]
    pub density: f32,
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

impl CpuFilter for SaltPepperNoiseParams {
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
    validate_format(info.format)?;
    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            self.compute(r, &mut u, i8)
        });
    }

    let density = self.density as f64;
    if density == 0.0 {
        return Ok(pixels.to_vec());
    }

    let ch = channels(info.format);
    let has_alpha = matches!(info.format, PixelFormat::Rgba8);
    let mut rng = self.seed.max(1);

    let mut out = pixels.to_vec();
    for pixel in out.chunks_exact_mut(ch) {
        let r = xorshift64_f64(&mut rng);
        if r < density {
            let color_ch = if has_alpha { ch - 1 } else { ch };
            // First half → salt (white), second half → pepper (black)
            let val = if xorshift64_f64(&mut rng) < 0.5 {
                255u8
            } else {
                0u8
            };
            for c in &mut pixel[..color_ch] {
                *c = val;
            }
        }
    }
    Ok(out)
}
}

