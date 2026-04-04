//! Filter: mask_feather (category: mask)
//!
//! Apply gaussian blur to mask edges for smooth transitions.

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

#[derive(rasmcore_macros::Filter, Clone)]
/// Mask feather parameters.
#[filter(name = "mask_feather", category = "mask", reference = "gaussian blur on mask for feathered edges")]
pub struct MaskFeatherParams {
    /// Feather radius (gaussian blur sigma)
    #[param(min = 0.1, max = 100.0, step = 0.1, default = 5.0)]
    pub radius: f32,
}

/// Apply gaussian blur to a mask for feathered edges.
///
/// Works on Gray8 (extracts channel, blurs, returns Gray8).
/// For RGB8: converts to gray, blurs, returns as RGB8 with R=G=B.
impl CpuFilter for MaskFeatherParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let w = request.width as usize;
    let h = request.height as usize;
    let sigma = self.radius;

    match info.format {
        PixelFormat::Gray8 => {
            // Convert to f32, blur, convert back
            let f32_data: Vec<f32> = pixels.iter().map(|&b| b as f32).collect();
            let blurred = blur_1ch_f32(&f32_data, w, h, sigma);
            Ok(blurred
                .iter()
                .map(|&v| v.round().clamp(0.0, 255.0) as u8)
                .collect())
        }
        PixelFormat::Rgb8 => {
            // Extract luminance to gray, blur, output as RGB
            let n = w * h;
            let gray: Vec<f32> = (0..n)
                .map(|i| {
                    let r = pixels[i * 3] as f32;
                    let g = pixels[i * 3 + 1] as f32;
                    let b = pixels[i * 3 + 2] as f32;
                    r * 0.2126 + g * 0.7152 + b * 0.0722
                })
                .collect();
            let blurred = blur_1ch_f32(&gray, w, h, sigma);
            let mut result = vec![0u8; n * 3];
            for i in 0..n {
                let v = blurred[i].round().clamp(0.0, 255.0) as u8;
                result[i * 3] = v;
                result[i * 3 + 1] = v;
                result[i * 3 + 2] = v;
            }
            Ok(result)
        }
        _ => Err(ImageError::UnsupportedFormat(
            "mask_feather requires Gray8 or RGB8".into(),
        )),
    }
}
}

