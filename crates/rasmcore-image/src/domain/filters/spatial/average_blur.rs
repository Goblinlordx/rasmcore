//! Filter: average_blur (category: spatial)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Average blur — compute the mean color of the entire image and fill.
///
/// Computes per-channel mean of all pixels and returns a solid-color image.
/// Matches Photoshop's Average blur behavior: reduces an image to its
/// dominant color.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(
    name = "average_blur",
    category = "spatial",
    group = "blur",
    variant = "average",
    reference = "Photoshop Average blur"
)]
pub struct AverageBlurParams {}

impl CpuFilter for AverageBlurParams {
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
        validate_format(info.format)?;

        if is_16bit(info.format) {
            let cfg = self.clone();
            return process_via_8bit(pixels, info, |p8, i8| {
                let r = Rect::new(0, 0, i8.width, i8.height);
                let mut u = |_: Rect| Ok(p8.to_vec());
                cfg.compute(r, &mut u, i8)
            });
        }

        let ch = crate::domain::types::bytes_per_pixel(info.format) as usize;
        let pixel_count = pixels.len() / ch;
        if pixel_count == 0 {
            return Ok(pixels.to_vec());
        }

        // Sum each channel
        let mut sums = vec![0u64; ch];
        for i in 0..pixel_count {
            for c in 0..ch {
                sums[c] += pixels[i * ch + c] as u64;
            }
        }

        // Compute mean per channel
        let means: Vec<u8> = sums
            .iter()
            .map(|&s| (s / pixel_count as u64) as u8)
            .collect();

        // Fill output with mean color
        let mut out = vec![0u8; pixels.len()];
        for i in 0..pixel_count {
            for c in 0..ch {
                out[i * ch + c] = means[c];
            }
        }

        Ok(out)
    }
}
