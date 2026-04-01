//! Filter: polar (category: distortion)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Polar: convert Cartesian image to polar-coordinate projection.
///
/// Maps the rectangular image into a polar representation where:
/// - Output x-axis represents angle (0 to 2π across width)
/// - Output y-axis represents radius (0 to max_radius across height)
///
/// Equivalent to ImageMagick `-distort Polar "max_radius"`.
#[rasmcore_macros::register_filter(
    name = "polar",
    category = "distortion",
    group = "distort_polar",
    variant = "to_polar",
    reference = "Cartesian to polar coordinate transform"
)]
pub fn polar(
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
            polar(r, &mut u, i8)
        });
    }

    let wf = info.width as f32;
    let hf = info.height as f32;
    let cx = wf * 0.5;
    let cy = hf * 0.5;
    let max_radius = cx.min(cy);

    apply_distortion(
        request, upstream, info,
        DistortionOverlap::FullImage,
        DistortionSampling::Ewa,
        &|xf, yf| {
            let two_pi = std::f32::consts::TAU;
            let radius = yf / hf * max_radius;
            let angle = xf / wf * two_pi - std::f32::consts::PI;
            (cx + radius * angle.sin(), cy - radius * angle.cos())
        },
        &|xf, yf| {
            crate::domain::ewa::jacobian_polar(xf, yf, wf, hf, max_radius)
        },
    )
}
