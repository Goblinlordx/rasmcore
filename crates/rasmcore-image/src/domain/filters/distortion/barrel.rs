//! Filter: barrel (category: distortion)

#[allow(unused_imports)]
use crate::domain::filters::common::*;


/// Barrel distortion: apply radial polynomial distortion.
/// `r_distorted = r * (1 + k1*r² + k2*r⁴)`.
/// `k1 > 0` = barrel, `k1 < 0` = pincushion.
/// This is the inverse of the `undistort` correction filter.
/// Matches ImageMagick `-distort Barrel` normalization: `rscale = 2/min(w,h)`.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct BarrelParams {
    /// Radial distortion coefficient (positive = barrel, negative = pincushion)
    #[param(min = -1.0, max = 1.0, step = 0.05, default = 0.3, hint = "rc.signed_slider")]
    pub k1: f32,
    /// Higher-order radial coefficient
    #[param(min = -1.0, max = 1.0, step = 0.05, default = 0.0, hint = "rc.signed_slider")]
    pub k2: f32,
}

#[rasmcore_macros::register_filter(
    name = "barrel",
    category = "distortion",
    reference = "Brown-Conrady radial distortion model"
)]
pub fn barrel(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &BarrelParams,
) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;

    if is_16bit(info.format) {
        let full = Rect::new(0, 0, info.width, info.height);
        let pixels = upstream(full)?;
        let info16 = &ImageInfo { width: info.width, height: info.height, ..*info };
        return process_via_8bit(&pixels, info16, |px, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(px.to_vec());
            barrel(r, &mut u, i8, config)
        });
    }

    let k1 = config.k1;
    let k2 = config.k2;
    // IM: center = w/2, rscale = 2/min(w,h), pixel-center convention (i+0.5)
    let wf = info.width as f64;
    let hf = info.height as f64;
    let cx = wf * 0.5;
    let cy = hf * 0.5;
    let rscale = 2.0 / wf.min(hf);
    // IM denormalizes coefficients: A *= rscale³, B *= rscale²
    let a_coeff = k1 as f64 * rscale * rscale * rscale;
    let b_coeff = k2 as f64 * rscale * rscale;

    let distort_f64 = move |ox: f64, oy: f64| -> (f64, f64) {
        let di = ox - cx;
        let dj = oy - cy;
        let dr = (di * di + dj * dj).sqrt();
        let df = a_coeff * dr * dr * dr + b_coeff * dr * dr + 1.0;
        (di * df + cx, dj * df + cy)
    };

    apply_distortion(
        request, upstream, info,
        DistortionOverlap::FullImage,
        DistortionSampling::EwaClamp,
        &|xf, yf| {
            // IM pixel-center: d.x = i + 0.5
            let xf64 = xf as f64 + 0.5;
            let yf64 = yf as f64 + 0.5;
            let (sx, sy) = distort_f64(xf64, yf64);
            (sx as f32, sy as f32)
        },
        &|xf, yf| {
            let xf64 = xf as f64 + 0.5;
            let yf64 = yf as f64 + 0.5;
            let h_step = 0.5;
            let (sx_px, sy_px) = distort_f64(xf64 + h_step, yf64);
            let (sx_mx, sy_mx) = distort_f64(xf64 - h_step, yf64);
            let (sx_py, sy_py) = distort_f64(xf64, yf64 + h_step);
            let (sx_my, sy_my) = distort_f64(xf64, yf64 - h_step);
            let inv_2h = 1.0 / (2.0 * h_step);
            [
                [
                    ((sx_px - sx_mx) * inv_2h) as f32,
                    ((sx_py - sx_my) * inv_2h) as f32,
                ],
                [
                    ((sy_px - sy_mx) * inv_2h) as f32,
                    ((sy_py - sy_my) * inv_2h) as f32,
                ],
            ]
        },
    )
}
