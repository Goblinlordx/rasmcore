//! Filter: motion_blur (category: spatial)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

#[rasmcore_macros::register_filter(
    name = "motion_blur",
    category = "spatial",
    group = "blur",
    variant = "motion",
    reference = "linear kernel simulating camera motion"
)]
pub fn motion_blur(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &MotionBlurParams,
) -> Result<Vec<u8>, ImageError> {
    let overlap = config.length;
    let expanded = request.expand_uniform(overlap, info.width, info.height);
    let pixels = upstream(expanded)?;
    let info = &ImageInfo {
        width: expanded.width,
        height: expanded.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let length = config.length;
    let angle_degrees = config.angle_degrees;

    if length == 0 {
        return Ok(pixels.to_vec());
    }
    validate_format(info.format)?;

    let side = (2 * length + 1) as usize;
    let center = length as f32;
    let angle = angle_degrees.to_radians();
    let dx = angle.cos();
    let dy = -angle.sin(); // negative because Y increases downward in image coords

    // Rasterize the line: walk from -length to +length along the direction vector,
    // marking each pixel in the kernel. Use Bresenham-style: step in 0.5-pixel increments
    // for smooth coverage.
    let mut kernel = vec![0.0f32; side * side];
    let steps = (length as f32 * 2.0).ceil() as usize * 2 + 1;
    let mut count = 0u32;
    for i in 0..steps {
        let t = (i as f32 / (steps - 1) as f32) * 2.0 - 1.0; // -1..1
        let px = center + t * length as f32 * dx;
        let py = center + t * length as f32 * dy;
        let ix = px.round() as usize;
        let iy = py.round() as usize;
        if ix < side && iy < side {
            let idx = iy * side + ix;
            if kernel[idx] == 0.0 {
                kernel[idx] = 1.0;
                count += 1;
            }
        }
    }

    if count == 0 {
        return Ok(pixels.to_vec());
    }

    let full_rect = Rect::new(0, 0, info.width, info.height);
    let result = {
        let mut u = |_: Rect| Ok(pixels.to_vec());
        convolve(
            full_rect,
            &mut u,
            info,
            &kernel,
            &ConvolveParams {
                kw: side as u32,
                kh: side as u32,
                divisor: count as f32,
            },
        )
    }?;
    Ok(crop_to_request(&result, expanded, request, info.format))
}
