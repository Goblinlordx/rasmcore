//! Filter: wave (category: distortion)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Wave: sinusoidal displacement along one axis.
///
/// Displaces pixels sinusoidally: horizontal wave shifts rows up/down,
/// vertical wave shifts columns left/right.
///
/// Equivalent to ImageMagick `-wave {amplitude}x{wavelength}`.
#[rasmcore_macros::register_filter(
    name = "wave",
    category = "distortion",
    reference = "sinusoidal wave displacement"
)]
pub fn wave(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &WaveParams,
) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;

    if is_16bit(info.format) {
        let full = Rect::new(0, 0, info.width, info.height);
        let pixels = upstream(full)?;
        let info16 = &ImageInfo { width: info.width, height: info.height, ..*info };
        return process_via_8bit(&pixels, info16, |px, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(px.to_vec());
            wave(r, &mut u, i8, config)
        });
    }

    let amplitude = config.amplitude;
    let wl = if config.wavelength.abs() < 1e-6 { 1.0 } else { config.wavelength };
    let is_vert = config.vertical >= 0.5;
    let overlap = amplitude.ceil() as u32 + 1;
    let dummy_j = crate::domain::ewa::JACOBIAN_IDENTITY;

    apply_distortion(
        request, upstream, info,
        DistortionOverlap::Uniform(overlap),
        DistortionSampling::Bilinear,
        &|xf, yf| {
            let two_pi = std::f32::consts::TAU;
            if is_vert {
                (xf - amplitude * (two_pi * yf / wl).sin(), yf)
            } else {
                (xf, yf - amplitude * (two_pi * xf / wl).sin())
            }
        },
        &|_xf, _yf| dummy_j,
    )
}
