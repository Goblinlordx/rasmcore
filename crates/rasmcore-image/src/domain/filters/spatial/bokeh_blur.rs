//! Filter: bokeh_blur (category: spatial)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Apply lens bokeh blur with a disc or hexagonal kernel.
///
/// Generates a uniform kernel of the specified shape and radius, then applies
/// it via 2D convolution. Matches `cv2.filter2D` with the same kernel and
/// `BORDER_REFLECT_101`.
///
/// `radius` is the kernel half-size in pixels (kernel side = 2*radius+1).
/// Minimum radius is 1.
/// Parameters for bokeh (lens) blur.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(
    name = "bokeh_blur",
    category = "spatial",
    group = "blur",
    variant = "bokeh",
    reference = "physical optic disc/polygon aperture model"
)]
pub struct BokehBlurParams {
    /// Kernel half-size in pixels (kernel side = 2*radius+1)
    #[param(min = 1, max = 50, step = 1, default = 5, hint = "rc.log_slider")]
    pub radius: u32,
    /// Kernel shape: 0=disc, 1=hexagon
    #[param(min = 0, max = 1, step = 1, default = 0)]
    pub shape: u32,
}

impl CpuFilter for BokehBlurParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
        let overlap = self.radius;
        let expanded = request.expand_uniform(overlap, info.width, info.height);
        let pixels = upstream(expanded)?;
        let info = &ImageInfo {
            width: expanded.width,
            height: expanded.height,
            ..*info
        };
        let pixels = pixels.as_slice();
        let radius = self.radius;
        let shape = self.shape;

        let shape = match shape {
            1 => 1,
            _ => 0,
        };
        if radius == 0 {
            return Ok(pixels.to_vec());
        }
        let (kernel, side) = match shape {
            1 => make_hex_kernel(radius),
            _ => make_disc_kernel(radius),
        };
        let divisor: f32 = kernel.iter().sum();
        if divisor == 0.0 {
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
                    divisor,
                },
            )
        }?;
        Ok(crop_to_request(&result, expanded, request, info.format))
    }

    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> Rect {
        let overlap = self.radius;
        output.expand_uniform(overlap, bounds_w, bounds_h)
    }
}
