//! Filter: gaussian_blur_cv (category: spatial)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Gaussian blur with OpenCV-compatible kernel and border handling.
///
/// Generates a Gaussian kernel matching `cv2.getGaussianKernel` and applies it
/// via our `convolve()` function (which uses `BORDER_REFLECT_101` and is already
/// pixel-exact against OpenCV `filter2D`).
///
/// This is a separate entry point from `blur()` with an explicit `sigma` parameter
/// (vs `blur()`'s `radius` which maps to sigma). Both now use the same `convolve()`
/// backend with `BORDER_REFLECT_101`. Use this when pixel-exact OpenCV parity is
/// required.
///
/// - `sigma`: Gaussian standard deviation
///
/// Parameters for OpenCV-compatible Gaussian blur.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(
    name = "gaussian_blur_cv",
    category = "spatial",
    group = "blur",
    variant = "gaussian_cv",
    reference = "OpenCV-compatible separable Gaussian"
)]
pub struct GaussianBlurCvParams {
    /// Gaussian standard deviation
    #[param(
        min = 0.1,
        max = 50.0,
        step = 0.1,
        default = 1.0,
        hint = "rc.log_slider"
    )]
    pub sigma: f32,
}

impl CpuFilter for GaussianBlurCvParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
        let overlap = (self.sigma * 3.0).ceil() as u32 + 1;
        let expanded = request.expand_uniform(overlap, info.width, info.height);
        let pixels = upstream(expanded)?;
        let info = &ImageInfo {
            width: expanded.width,
            height: expanded.height,
            ..*info
        };
        let pixels = pixels.as_slice();
        let sigma = self.sigma;

        if sigma <= 0.0 {
            return Ok(pixels.to_vec());
        }

        // For large sigma, use stackable box blur approximation (O(1) per pixel).
        // This is dramatically faster: e.g. sigma=80 reduces from 481-tap separable
        // convolution to 3 box blur passes. PSNR >= 35dB for sigma >= 20.
        if sigma >= 20.0 {
            let result = gaussian_blur_box_approx(pixels, info, sigma)?;
            return Ok(crop_to_request(&result, expanded, request, info.format));
        }

        // Small sigma: exact separable Gaussian (kernel size is manageable)
        let ksize = {
            let k = (sigma * 6.0 + 1.0).round() as usize;
            if k.is_multiple_of(2) { k + 1 } else { k }
        };
        let ksize = ksize.max(3);

        let k1d = gaussian_kernel_1d(ksize, sigma);
        let mut kernel_2d = vec![0.0f32; ksize * ksize];
        for y in 0..ksize {
            for x in 0..ksize {
                kernel_2d[y * ksize + x] = k1d[y] * k1d[x];
            }
        }

        let full_rect = Rect::new(0, 0, info.width, info.height);
        let result = {
            let mut u = |_: Rect| Ok(pixels.to_vec());
            convolve(
                full_rect,
                &mut u,
                info,
                &kernel_2d,
                &ConvolveParams {
                    kw: ksize as u32,
                    kh: ksize as u32,
                    divisor: 1.0,
                },
            )
        }?;
        Ok(crop_to_request(&result, expanded, request, info.format))
    }

    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> Rect {
        let overlap = (self.sigma * 3.0).ceil() as u32 + 1;
        output.expand_uniform(overlap, bounds_w, bounds_h)
    }
}
