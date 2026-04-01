//! Filter: lens_blur (category: spatial)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Lens blur — depth-of-field simulation with disc or polygon bokeh kernel.
///
/// With blade_count=0, uses a circular disc (same as bokeh_blur).
/// With blade_count=5-12, uses a regular polygon simulating a camera aperture
/// with that many blades, rotated by the rotation parameter.

/// Lens blur config.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct LensBlurParams {
    /// Blur radius in pixels
    #[param(min = 0, max = 50, step = 1, default = 5, hint = "rc.log_slider")]
    pub radius: u32,
    /// Aperture blade count (0=disc, 5-8=polygon)
    #[param(min = 0, max = 12, step = 1, default = 0)]
    pub blade_count: u32,
    /// Blade rotation angle in degrees
    #[param(
        min = 0.0,
        max = 360.0,
        step = 1.0,
        default = 0.0,
        hint = "rc.angle_deg"
    )]
    pub rotation: f32,
}

#[rasmcore_macros::register_filter(
    name = "lens_blur",
    category = "spatial",
    group = "blur",
    variant = "lens",
    reference = "shaped bokeh kernel depth-of-field simulation"
)]
pub fn lens_blur(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &LensBlurParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let radius = config.radius;
    let blade_count = config.blade_count;
    let rotation = config.rotation;

    if radius == 0 {
        return Ok(pixels.to_vec());
    }

    let (kernel, side) = if blade_count == 0 || blade_count < 3 {
        make_disc_kernel(radius)
    } else {
        make_polygon_kernel(radius, blade_count, rotation)
    };

    let divisor: f32 = kernel.iter().sum();
    if divisor == 0.0 {
        return Ok(pixels.to_vec());
    }

    {
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(pixels.to_vec());
        convolve(
            r,
            &mut u,
            info,
            &kernel,
            &ConvolveParams {
                kw: side as u32,
                kh: side as u32,
                divisor,
            },
        )
    }
}
