//! Filter: swirl (category: distortion)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

#[rasmcore_macros::register_filter(
    name = "swirl",
    category = "distortion",
    reference = "vortex rotation distortion"
)]
pub fn swirl(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &SwirlParams,
) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;

    if is_16bit(info.format) {
        let full = Rect::new(0, 0, info.width, info.height);
        let pixels = upstream(full)?;
        let info16 = &ImageInfo { width: info.width, height: info.height, ..*info };
        return process_via_8bit(&pixels, info16, |px, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(px.to_vec());
            swirl(r, &mut u, i8, config)
        });
    }

    let w = info.width as f32;
    let h = info.height as f32;
    let cx = w * 0.5;
    let cy = h * 0.5;
    let rad = if config.radius <= 0.0 { cx.max(cy) } else { config.radius };
    let angle_rad = config.angle.to_radians();
    let (scale_x, scale_y) = if info.width > info.height {
        (1.0f32, w / h)
    } else if info.height > info.width {
        (h / w, 1.0f32)
    } else {
        (1.0, 1.0)
    };

    apply_distortion(
        request, upstream, info,
        DistortionOverlap::FullImage,
        DistortionSampling::Ewa,
        &|xf, yf| {
            let dx = scale_x * (xf - cx);
            let dy = scale_y * (yf - cy);
            let dist = (dx * dx + dy * dy).sqrt();
            let t = (1.0 - dist / rad).max(0.0);
            let rot = angle_rad * t * t;
            let (cos_r, sin_r) = (rot.cos(), rot.sin());
            ((cos_r * dx - sin_r * dy) / scale_x + cx,
             (sin_r * dx + cos_r * dy) / scale_y + cy)
        },
        &|xf, yf| {
            crate::domain::ewa::jacobian_swirl(xf, yf, cx, cy, angle_rad, rad, scale_x, scale_y)
        },
    )
}
