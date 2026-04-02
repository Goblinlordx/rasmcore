//! Filter: depolar (category: distortion)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// DePolar: convert polar-coordinate image back to Cartesian projection.
///
/// Inverse of `polar`: maps a polar representation back to rectangular.
/// For each output pixel, compute radius and angle from center,
/// then look up in the polar-space input image.
///
/// Equivalent to ImageMagick `-distort DePolar "max_radius"`.
#[rasmcore_macros::register_filter(
    name = "depolar", gpu = "true",
    category = "distortion",
    group = "distort_polar",
    variant = "from_polar",
    reference = "polar to Cartesian coordinate transform"
)]
pub fn depolar(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;

    if is_16bit(info.format) {
        let full = Rect::new(0, 0, info.width, info.height);
        let pixels = upstream(full)?;
        let info16 = &ImageInfo { width: info.width, height: info.height, ..*info };
        return process_via_8bit(&pixels, info16, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            depolar(r, &mut u, i8)
        });
    }

    // Use f64 throughout to match IM's double-precision pipeline
    let wf = info.width as f64;
    let hf = info.height as f64;
    // IM uses pixel-center convention: d.x = i + 0.5, center = w/2
    let cx = wf * 0.5;
    let cy = hf * 0.5;
    let max_radius = (wf * 0.5).min(hf * 0.5);
    let two_pi = std::f64::consts::TAU;
    let c6 = wf / two_pi;
    let c7 = hf / max_radius;
    let half_w = wf * 0.5;

    // Sampling: Ewa — matches IM Polar (distort.c EWA engine, MAE 2.55).
    // The Cartesian-to-polar mapping benefits from EWA's anisotropic filtering
    // near the center where radial lines converge.
    apply_distortion(
        request, upstream, info,
        DistortionOverlap::FullImage,
        DistortionSampling::Ewa,
        &|xf, yf| {
            // IM pixel-center: d.x = i + 0.5
            let xf64 = xf as f64 + 0.5;
            let yf64 = yf as f64 + 0.5;
            let ii = xf64 - cx;
            let jj = yf64 - cy;
            let radius = (ii * ii + jj * jj).sqrt();
            let angle = ii.atan2(jj);
            let mut xx = angle / two_pi;
            xx -= xx.round();
            let sx = (xx * two_pi * c6 + half_w - 0.5) as f32;
            let sy = (radius * c7 - 0.5) as f32;
            (sx, sy)
        },
        &|xf, yf| {
            let xf64 = xf as f64 + 0.5;
            let yf64 = yf as f64 + 0.5;
            let ii = xf64 - cx;
            let jj = yf64 - cy;
            let r2 = ii * ii + jj * jj;
            if r2 < 1e-10 {
                crate::domain::ewa::JACOBIAN_IDENTITY
            } else {
                let r = r2.sqrt();
                [
                    [(jj / r2 * c6) as f32, (-ii / r2 * c6) as f32],
                    [(ii / r * c7) as f32, (jj / r * c7) as f32],
                ]
            }
        },
    )
}
