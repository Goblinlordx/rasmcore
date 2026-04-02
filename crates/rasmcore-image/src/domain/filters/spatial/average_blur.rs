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

        let ch = channels(info.format);
        let samples = pixels_to_f32_samples(pixels, info.format);
        let pixel_count = samples.len() / ch;
        if pixel_count == 0 {
            return Ok(pixels.to_vec());
        }

        // Sum each channel in f64 for precision
        let mut sums = vec![0.0f64; ch];
        for i in 0..pixel_count {
            for c in 0..ch {
                sums[c] += samples[i * ch + c] as f64;
            }
        }

        // Compute mean per channel
        let inv_count = 1.0 / pixel_count as f64;
        let means: Vec<f32> = sums.iter().map(|&s| (s * inv_count) as f32).collect();

        // Fill output with mean color
        let mut out = vec![0.0f32; samples.len()];
        for i in 0..pixel_count {
            for c in 0..ch {
                out[i * ch + c] = means[c];
            }
        }

        Ok(f32_samples_to_pixels(&out, info.format))
    }
}
