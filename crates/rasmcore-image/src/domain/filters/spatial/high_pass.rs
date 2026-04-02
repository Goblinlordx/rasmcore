//! Filter: high_pass (category: spatial)
//!
//! Spatial high-pass filter for frequency separation retouching.
//! Subtracts a Gaussian-blurred version from the original, producing
//! a detail/texture layer centered at mid-gray (128).
//!
//! Formula: high_pass[c] = clamp((original[c] - blurred[c]) / 2 + 128, 0, 255)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Parameters for high-pass filter.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct HighPassParams {
    /// Blur radius for frequency separation (larger = more detail preserved)
    #[param(min = 0.0, max = 100.0, step = 0.5, default = 10.0, hint = "rc.log_slider")]
    pub radius: f32,
}

impl InputRectProvider for HighPassParams {
    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> Rect {
        let radius = self.radius;
        let overlap = if radius > 0.0 { ((radius * 3.0).ceil() as u32).max(1) } else { 0 };
        output.expand_uniform(overlap, bounds_w, bounds_h)
    }
}

#[rasmcore_macros::register_filter(
    name = "high_pass", gpu = "true",
    category = "spatial",
    group = "detail",
    reference = "frequency separation high-pass"
)]
pub fn high_pass(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &HighPassParams,
) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;

    if is_16bit(info.format) {
        let full = Rect::new(0, 0, info.width, info.height);
        let pixels = upstream(full)?;
        let info16 = &ImageInfo { width: info.width, height: info.height, ..*info };
        return process_via_8bit(&pixels, info16, |px, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(px.to_vec());
            high_pass(r, &mut u, i8, config)
        });
    }

    let radius = config.radius;

    // Get original pixels for the requested region (with blur overlap)
    let overlap = if radius > 0.0 { ((radius * 3.0).ceil() as u32).max(1) } else { 0 };
    let expanded = request.expand_uniform(overlap, info.width, info.height);
    let pixels = upstream(expanded)?;
    let expanded_info = ImageInfo { width: expanded.width, height: expanded.height, ..*info };

    // Blur the expanded region
    let blur_config = BlurParams { radius };
    let blurred = blur_impl(&pixels, &expanded_info, &blur_config)?;

    // Crop both to the requested region
    let original = crop_to_request(&pixels, expanded, request, info.format);
    let blurred = crop_to_request(&blurred, expanded, request, info.format);

    let channels = crate::domain::types::bytes_per_pixel(info.format) as usize;
    let n = (request.width * request.height) as usize * channels;
    let mut output = vec![0u8; n];

    for i in 0..n {
        // Preserve alpha channel unchanged
        if channels == 4 && i % 4 == 3 {
            output[i] = original[i];
        } else {
            // high_pass = (original - blurred) / 2 + 128
            let diff = original[i] as i16 - blurred[i] as i16;
            output[i] = ((diff / 2) + 128).clamp(0, 255) as u8;
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::{ColorSpace, ImageInfo, PixelFormat};

    fn solid_rgba(w: u32, h: u32, r: u8, g: u8, b: u8) -> (Vec<u8>, ImageInfo) {
        let mut pixels = vec![0u8; (w * h * 4) as usize];
        for i in 0..(w * h) as usize {
            pixels[i * 4] = r;
            pixels[i * 4 + 1] = g;
            pixels[i * 4 + 2] = b;
            pixels[i * 4 + 3] = 255;
        }
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        (pixels, info)
    }

    #[test]
    fn solid_color_produces_mid_gray() {
        // A solid color image has no detail — high pass should be ~128
        let (pixels, info) = solid_rgba(32, 32, 200, 100, 50);
        let config = HighPassParams { radius: 5.0 };
        let request = Rect::new(0, 0, 32, 32);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = high_pass(request, &mut u, &info, &config).unwrap();

        // All RGB channels should be 128 (mid-gray) since original == blurred
        for i in 0..(32 * 32) as usize {
            for c in 0..3 {
                let val = result[i * 4 + c];
                assert!(
                    (val as i16 - 128).unsigned_abs() <= 1,
                    "pixel {i} channel {c}: expected ~128, got {val}"
                );
            }
            // Alpha preserved
            assert_eq!(result[i * 4 + 3], 255);
        }
    }

    #[test]
    fn zero_radius_produces_mid_gray() {
        // Zero radius = blur is identity, so original - original = 0, output = 128
        let (pixels, info) = solid_rgba(16, 16, 255, 0, 128);
        let config = HighPassParams { radius: 0.0 };
        let request = Rect::new(0, 0, 16, 16);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = high_pass(request, &mut u, &info, &config).unwrap();

        for i in 0..(16 * 16) as usize {
            for c in 0..3 {
                let val = result[i * 4 + c];
                assert_eq!(val, 128, "pixel {i} ch {c}: expected 128, got {val}");
            }
        }
    }

    #[test]
    fn high_pass_plus_blur_reconstructs_original() {
        // Create a test image with some variation
        let w = 32u32;
        let h = 32u32;
        let mut pixels = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                pixels[i] = (x * 8) as u8;
                pixels[i + 1] = (y * 8) as u8;
                pixels[i + 2] = ((x + y) * 4) as u8;
                pixels[i + 3] = 255;
            }
        }
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };

        let radius = 3.0;
        let request = Rect::new(0, 0, w, h);

        // Get high-pass
        let config = HighPassParams { radius };
        let mut u = |_: Rect| Ok(pixels.clone());
        let hp = high_pass(request, &mut u, &info, &config).unwrap();

        // Get blur
        let blur_config = super::BlurParams { radius };
        let mut u = |_: Rect| Ok(pixels.clone());
        let blurred = super::blur(request, &mut u, &info, &blur_config).unwrap();

        // Reconstruct: original ≈ blurred + 2 * (high_pass - 128)
        // This is the Linear Light blend mode reconstruction
        let mut max_err = 0i16;
        let n = (w * h) as usize;
        for i in 0..n {
            for c in 0..3 {
                let idx = i * 4 + c;
                let reconstructed = (blurred[idx] as i16 + 2 * (hp[idx] as i16 - 128)).clamp(0, 255);
                let err = (reconstructed - pixels[idx] as i16).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }
        // Allow small rounding error from integer arithmetic
        assert!(
            max_err <= 3,
            "reconstruction max error was {max_err}, expected <= 3"
        );
    }

    #[test]
    fn large_radius_preserves_detail() {
        // With very large radius, blur ≈ flat, so high_pass ≈ (original - flat)/2 + 128
        // Detail should be visible (not all mid-gray)
        let w = 32u32;
        let h = 32u32;
        let mut pixels = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                // Checkerboard pattern
                let v = if (x + y) % 2 == 0 { 255u8 } else { 0u8 };
                pixels[i] = v;
                pixels[i + 1] = v;
                pixels[i + 2] = v;
                pixels[i + 3] = 255;
            }
        }
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };

        let config = HighPassParams { radius: 50.0 };
        let request = Rect::new(0, 0, w, h);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = high_pass(request, &mut u, &info, &config).unwrap();

        // Result should NOT be all mid-gray — there should be variance
        let mut min_val = 255u8;
        let mut max_val = 0u8;
        for i in 0..(w * h) as usize {
            let v = result[i * 4]; // R channel
            if v < min_val { min_val = v; }
            if v > max_val { max_val = v; }
        }
        let range = max_val as i16 - min_val as i16;
        assert!(
            range > 50,
            "large-radius high-pass should preserve detail, range was {range}"
        );
    }
}
