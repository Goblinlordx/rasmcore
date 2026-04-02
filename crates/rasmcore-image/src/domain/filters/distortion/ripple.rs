//! Filter: ripple (category: distortion)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Ripple: concentric sinusoidal distortion radiating from a center point.
///
/// Displaces pixels radially based on their distance from center:
/// each pixel moves along its radial direction by `amplitude * sin(2π * r / wavelength)`.
///
/// Equivalent to ImageMagick concentric wave effect.

#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct RippleParams {
    /// Displacement amplitude in pixels
    #[param(min = 0.0, max = 100.0, step = 1.0, default = 8.0)]
    pub amplitude: f32,
    /// Wavelength in pixels
    #[param(min = 1.0, max = 500.0, step = 5.0, default = 40.0)]
    pub wavelength: f32,
    /// Center X (0.0-1.0 normalized, 0.5 = center)
    #[param(min = 0.0, max = 1.0, step = 0.05, default = 0.5)]
    pub center_x: f32,
    /// Center Y (0.0-1.0 normalized, 0.5 = center)
    #[param(min = 0.0, max = 1.0, step = 0.05, default = 0.5)]
    pub center_y: f32,
}

#[rasmcore_macros::register_filter(
    name = "ripple",
    category = "distortion",
    reference = "concentric ripple displacement"
)]
pub fn ripple(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &RippleParams,
) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;

    if is_16bit(info.format) {
        let full = Rect::new(0, 0, info.width, info.height);
        let pixels = upstream(full)?;
        let info16 = &ImageInfo { width: info.width, height: info.height, ..*info };
        return process_via_8bit(&pixels, info16, |px, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(px.to_vec());
            ripple(r, &mut u, i8, config)
        });
    }

    let amplitude = config.amplitude;
    let wl = if config.wavelength.abs() < 1e-6 { 1.0 } else { config.wavelength };
    let cx = config.center_x * info.width as f32;
    let cy = config.center_y * info.height as f32;
    let overlap = amplitude.ceil() as u32 + 1;

    // Sampling: Ewa — no direct IM equivalent. EWA suits the radial sinusoidal
    // displacement which creates varying anisotropy depending on distance from center.
    apply_distortion(
        request, upstream, info,
        DistortionOverlap::Uniform(overlap),
        DistortionSampling::Ewa,
        &|xf, yf| {
            let dx = xf - cx;
            let dy = yf - cy;
            let r = (dx * dx + dy * dy).sqrt();
            if r < 1e-6 {
                (xf, yf)
            } else {
                let two_pi = std::f32::consts::TAU;
                let disp = amplitude * (two_pi * r / wl).sin();
                let cos_a = dx / r;
                let sin_a = dy / r;
                (xf + disp * cos_a, yf + disp * sin_a)
            }
        },
        &|xf, yf| {
            crate::domain::ewa::jacobian_ripple(xf, yf, cx, cy, amplitude, wl)
        },
    )
}
