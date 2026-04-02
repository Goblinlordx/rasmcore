//! Filter: tilt_shift (category: spatial)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Tilt-shift blur — selective focus with graduated blur band.
///
/// Keeps a band of the image sharp while progressively blurring toward the edges.
/// Creates a miniature/diorama effect. The focus band can be rotated via the angle param.

/// Tilt-shift blur config.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(
    name = "tilt_shift",
    category = "spatial",
    group = "blur",
    variant = "tilt_shift",
    reference = "graduated blur mask with Gaussian blur"
)]
pub struct TiltShiftParams {
    /// Focus band center position (0.0=top, 0.5=middle, 1.0=bottom)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub focus_position: f32,
    /// Focus band size as fraction of image height (0.0=none, 1.0=full)
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.2)]
    pub band_size: f32,
    /// Maximum blur radius in pixels
    #[param(
        min = 0.0,
        max = 100.0,
        step = 0.5,
        default = 8.0,
        hint = "rc.log_slider"
    )]
    pub blur_radius: f32,
    /// Focus band angle in degrees (0=horizontal, 90=vertical)
    #[param(
        min = 0.0,
        max = 360.0,
        step = 1.0,
        default = 0.0,
        hint = "rc.angle_deg"
    )]
    pub angle: f32,
}

impl CpuFilter for TiltShiftParams {
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
        let focus_position = self.focus_position;
        let band_size = self.band_size;
        let blur_radius = self.blur_radius;
        let angle = self.angle;

        validate_format(info.format)?;

        if blur_radius <= 0.0 || band_size >= 1.0 {
            return Ok(pixels.to_vec());
        }

        let w = info.width as usize;
        let h = info.height as usize;
        let bpp = channels(info.format);

        // Generate the fully blurred version
        let blurred = blur_impl(
            pixels,
            info,
            &BlurParams {
                radius: blur_radius,
            },
        )?;

        // Compute per-pixel blur mask based on distance from focus band
        let angle_rad = angle.to_radians();
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();

        let half_band = band_size * 0.5;
        // Transition zone: from band edge to full blur
        let transition = (0.5 - half_band).max(0.01);

        let mut output = Vec::with_capacity(pixels.len());

        for y in 0..h {
            for x in 0..w {
                // Compute position along the focus axis (perpendicular to the band)
                let nx = x as f32 / w as f32 - 0.5;
                let ny = y as f32 / h as f32 - focus_position;

                // Rotate to align with band angle
                let dist = (-nx * sin_a + ny * cos_a).abs();

                // Compute blur amount: 0 inside band, ramps to 1 outside
                let t = if dist <= half_band {
                    0.0
                } else {
                    ((dist - half_band) / transition).clamp(0.0, 1.0)
                };
                // Smoothstep for gradual falloff
                let t = t * t * (3.0 - 2.0 * t);

                let idx = (y * w + x) * bpp;
                for c in 0..bpp {
                    let orig = pixels[idx + c] as f32;
                    let blur_val = blurred[idx + c] as f32;
                    output.push((orig + (blur_val - orig) * t) as u8);
                }
            }
        }

        Ok(output)
    }
}
