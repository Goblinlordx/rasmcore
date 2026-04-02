//! Filter: spherize (category: distortion)

#[allow(unused_imports)]
use crate::domain::filters::common::*;


/// Spherize: apply spherical projection for bulge/pinch effect.
/// `amount > 0` = bulge (fisheye), `amount < 0` = pinch.
/// `amount = 0` is identity.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct SpherizeParams {
    /// Bulge/pinch amount (-1 to 1, positive = bulge)
    #[param(min = -1.0, max = 1.0, step = 0.05, default = 0.5, hint = "rc.signed_slider")]
    pub amount: f32,
}

#[rasmcore_macros::register_filter(
    name = "spherize", gpu = "true",
    category = "distortion",
    reference = "spherical bulge distortion"
)]
pub fn spherize(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &SpherizeParams,
) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;

    if is_16bit(info.format) {
        let full = Rect::new(0, 0, info.width, info.height);
        let pixels = upstream(full)?;
        let info16 = &ImageInfo { width: info.width, height: info.height, ..*info };
        return process_via_8bit(&pixels, info16, |px, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(px.to_vec());
            spherize(r, &mut u, i8, config)
        });
    }

    let w = info.width as f32;
    let h = info.height as f32;
    let cx = w * 0.5;
    let cy = h * 0.5;
    let radius = cx.min(cy);
    let amt = config.amount.clamp(-1.0, 1.0);

    // Sampling: Ewa — no direct IM equivalent. EWA suits the nonlinear radial
    // mapping (powf-based bulge/pinch has anisotropic stretching near the edge).
    apply_distortion(
        request, upstream, info,
        DistortionOverlap::FullImage,
        DistortionSampling::Ewa,
        &|xf, yf| {
            let dx = (xf - cx) / radius;
            let dy = (yf - cy) / radius;
            let r = (dx * dx + dy * dy).sqrt();
            if r >= 1.0 || r == 0.0 {
                // Outside effect radius or at center: identity mapping
                (xf, yf)
            } else {
                let new_r = if amt >= 0.0 {
                    r.powf(1.0 / (1.0 + amt))
                } else {
                    r.powf(1.0 + amt.abs())
                };
                let scale = new_r / r;
                (dx * scale * radius + cx, dy * scale * radius + cy)
            }
        },
        &|xf, yf| {
            crate::domain::ewa::jacobian_spherize(xf, yf, cx, cy, amt, radius)
        },
    )
}
