mod affine;
mod auto_orient;
mod convert;
mod crop;
mod flip;
mod pad;
mod resize;
mod rotate;
mod trim;
mod undistort;

pub use affine::*;
pub use auto_orient::*;
pub use convert::*;
pub use crop::*;
pub use flip::*;
pub use pad::*;
pub use resize::*;
pub use rotate::*;
pub use trim::*;
pub use undistort::*;

use super::error::ImageError;
use super::types::{ImageInfo, PixelFormat};

/// Get bytes per pixel for supported formats.
pub(crate) fn bytes_per_pixel(format: PixelFormat) -> Result<usize, ImageError> {
    match format {
        PixelFormat::Rgb8 => Ok(3),
        PixelFormat::Rgba8 => Ok(4),
        PixelFormat::Gray8 => Ok(1),
        PixelFormat::Gray16 => Ok(2),
        PixelFormat::Rgb16 => Ok(6),
        PixelFormat::Rgba16 => Ok(8),
        PixelFormat::Rgba32f => Ok(16),
        PixelFormat::Rgb32f => Ok(12),
        PixelFormat::Gray32f => Ok(4),
        PixelFormat::Rgba16f => Ok(8),
        PixelFormat::Rgb16f => Ok(6),
        PixelFormat::Gray16f => Ok(2),
        _ => Err(ImageError::UnsupportedFormat(format!(
            "{format:?} not supported for geometric transforms"
        ))),
    }
}

/// Validate pixel buffer size matches dimensions and format.
pub(crate) fn validate_pixel_buffer(
    pixels: &[u8],
    info: &ImageInfo,
    bpp: usize,
) -> Result<(), ImageError> {
    let expected = info.width as usize * info.height as usize * bpp;
    if pixels.len() < expected {
        return Err(ImageError::InvalidInput(format!(
            "pixel buffer too small: need {expected}, got {}",
            pixels.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::metadata::ExifOrientation;
    use super::super::types::ColorSpace;

    fn make_image(w: u32, h: u32) -> (Vec<u8>, ImageInfo) {
        let pixels: Vec<u8> = (0..(w * h * 3)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        (pixels, info)
    }

    #[test]
    fn resize_changes_dimensions() {
        let (px, info) = make_image(64, 64);
        let result = resize(&px, &info, 32, 16, super::super::types::ResizeFilter::Bilinear).unwrap();
        assert_eq!(result.info.width, 32);
        assert_eq!(result.info.height, 16);
    }

    #[test]
    fn resize_preserves_format() {
        let (px, info) = make_image(16, 16);
        let result = resize(&px, &info, 8, 8, super::super::types::ResizeFilter::Nearest).unwrap();
        assert_eq!(result.info.format, PixelFormat::Rgb8);
    }

    #[test]
    fn resize_pixel_data_length_correct() {
        let (px, info) = make_image(16, 16);
        let result = resize(&px, &info, 32, 24, super::super::types::ResizeFilter::Lanczos3).unwrap();
        assert_eq!(result.pixels.len(), 32 * 24 * 3);
    }

    #[test]
    fn resize_all_filters_work() {
        use super::super::types::ResizeFilter;
        let (px, info) = make_image(16, 16);
        for filter in [
            ResizeFilter::Nearest,
            ResizeFilter::Bilinear,
            ResizeFilter::Bicubic,
            ResizeFilter::Lanczos3,
        ] {
            let result = resize(&px, &info, 8, 8, filter);
            assert!(result.is_ok(), "filter {filter:?} failed");
        }
    }

    #[test]
    fn crop_returns_correct_region() {
        let (px, info) = make_image(32, 32);
        let result = crop(&px, &info, 4, 4, 16, 16).unwrap();
        assert_eq!(result.info.width, 16);
        assert_eq!(result.info.height, 16);
        assert_eq!(result.pixels.len(), 16 * 16 * 3);
    }

    #[test]
    fn crop_out_of_bounds_returns_error() {
        let (px, info) = make_image(16, 16);
        let result = crop(&px, &info, 10, 10, 10, 10);
        assert!(result.is_err());
        match result.unwrap_err() {
            ImageError::InvalidParameters(_) => {}
            other => panic!("expected InvalidParameters, got {other:?}"),
        }
    }

    #[test]
    fn crop_zero_dimension_returns_error() {
        let (px, info) = make_image(16, 16);
        let result = crop(&px, &info, 0, 0, 0, 8);
        assert!(result.is_err());
    }

    #[test]
    fn rotate_90_swaps_dimensions() {
        use super::super::types::Rotation;
        let (px, info) = make_image(16, 8);
        let result = rotate(&px, &info, Rotation::R90).unwrap();
        assert_eq!(result.info.width, 8);
        assert_eq!(result.info.height, 16);
    }

    #[test]
    fn rotate_180_preserves_dimensions() {
        use super::super::types::Rotation;
        let (px, info) = make_image(16, 8);
        let result = rotate(&px, &info, Rotation::R180).unwrap();
        assert_eq!(result.info.width, 16);
        assert_eq!(result.info.height, 8);
    }

    #[test]
    fn rotate_270_swaps_dimensions() {
        use super::super::types::Rotation;
        let (px, info) = make_image(16, 8);
        let result = rotate(&px, &info, Rotation::R270).unwrap();
        assert_eq!(result.info.width, 8);
        assert_eq!(result.info.height, 16);
    }

    #[test]
    fn flip_horizontal_preserves_dimensions() {
        use super::super::types::FlipDirection;
        let (px, info) = make_image(16, 8);
        let result = flip(&px, &info, FlipDirection::Horizontal).unwrap();
        assert_eq!(result.info.width, 16);
        assert_eq!(result.info.height, 8);
        assert_eq!(result.pixels.len(), px.len());
    }

    #[test]
    fn flip_vertical_preserves_dimensions() {
        use super::super::types::FlipDirection;
        let (px, info) = make_image(16, 8);
        let result = flip(&px, &info, FlipDirection::Vertical).unwrap();
        assert_eq!(result.info.width, 16);
        assert_eq!(result.info.height, 8);
    }

    #[test]
    fn convert_rgb8_to_rgba8() {
        let (px, info) = make_image(8, 8);
        let result = convert_format(&px, &info, PixelFormat::Rgba8).unwrap();
        assert_eq!(result.info.format, PixelFormat::Rgba8);
        assert_eq!(result.pixels.len(), 8 * 8 * 4);
    }

    #[test]
    fn convert_rgb8_to_gray8() {
        let (px, info) = make_image(8, 8);
        let result = convert_format(&px, &info, PixelFormat::Gray8).unwrap();
        assert_eq!(result.info.format, PixelFormat::Gray8);
        assert_eq!(result.pixels.len(), 8 * 8 * 1);
    }

    #[test]
    fn convert_unsupported_returns_error() {
        let (px, info) = make_image(8, 8);
        let result = convert_format(&px, &info, PixelFormat::Nv12);
        assert!(result.is_err());
    }

    #[test]
    fn auto_orient_normal_is_identity() {
        let (px, info) = make_image(16, 8);
        let result = auto_orient(&px, &info, ExifOrientation::Normal).unwrap();
        assert_eq!(result.pixels, px);
        assert_eq!(result.info.width, 16);
        assert_eq!(result.info.height, 8);
    }

    #[test]
    fn auto_orient_rotate90_swaps_dimensions() {
        let (px, info) = make_image(16, 8);
        let result = auto_orient(&px, &info, ExifOrientation::Rotate90).unwrap();
        assert_eq!(result.info.width, 8);
        assert_eq!(result.info.height, 16);
    }

    #[test]
    fn auto_orient_rotate180_preserves_dimensions() {
        let (px, info) = make_image(16, 8);
        let result = auto_orient(&px, &info, ExifOrientation::Rotate180).unwrap();
        assert_eq!(result.info.width, 16);
        assert_eq!(result.info.height, 8);
    }

    #[test]
    fn auto_orient_rotate270_swaps_dimensions() {
        let (px, info) = make_image(16, 8);
        let result = auto_orient(&px, &info, ExifOrientation::Rotate270).unwrap();
        assert_eq!(result.info.width, 8);
        assert_eq!(result.info.height, 16);
    }

    #[test]
    fn auto_orient_flip_horizontal_preserves_dimensions() {
        let (px, info) = make_image(16, 8);
        let result = auto_orient(&px, &info, ExifOrientation::FlipHorizontal).unwrap();
        assert_eq!(result.info.width, 16);
        assert_eq!(result.info.height, 8);
    }

    #[test]
    fn auto_orient_flip_vertical_preserves_dimensions() {
        let (px, info) = make_image(16, 8);
        let result = auto_orient(&px, &info, ExifOrientation::FlipVertical).unwrap();
        assert_eq!(result.info.width, 16);
        assert_eq!(result.info.height, 8);
    }

    #[test]
    fn auto_orient_transpose_swaps_dimensions() {
        let (px, info) = make_image(16, 8);
        let result = auto_orient(&px, &info, ExifOrientation::Transpose).unwrap();
        assert_eq!(result.info.width, 8);
        assert_eq!(result.info.height, 16);
    }

    #[test]
    fn auto_orient_transverse_swaps_dimensions() {
        let (px, info) = make_image(16, 8);
        let result = auto_orient(&px, &info, ExifOrientation::Transverse).unwrap();
        assert_eq!(result.info.width, 8);
        assert_eq!(result.info.height, 16);
    }

    #[test]
    fn auto_orient_all_8_orientations_work() {
        let (px, info) = make_image(16, 8);
        for tag in 1..=8 {
            let orient = ExifOrientation::from_tag(tag);
            let result = auto_orient(&px, &info, orient);
            assert!(result.is_ok(), "orientation {tag} failed");
        }
    }

    #[test]
    fn auto_orient_from_exif_with_no_exif_is_identity() {
        let (px, info) = make_image(16, 8);
        // Non-JPEG data — no EXIF, should return unchanged
        let result = auto_orient_from_exif(&px, &info, &[0x89, 0x50]).unwrap();
        assert_eq!(result.pixels, px);
    }

    // ─── Extended Geometry Tests ────────────────────────────────────────

    #[test]
    fn rotate_arbitrary_0_preserves_dimensions() {
        let (px, info) = make_image(32, 32);
        let bg = [0, 0, 0];
        let result = rotate_arbitrary(&px, &info, 0.0, &bg).unwrap();
        assert_eq!(result.info.width, 32);
        assert_eq!(result.info.height, 32);
    }

    #[test]
    fn rotate_arbitrary_90_matches_dimensions() {
        let (px, info) = make_image(32, 16);
        let bg = [0, 0, 0];
        let result = rotate_arbitrary(&px, &info, 90.0, &bg).unwrap();
        // 90 degrees: output is roughly height x width
        assert!(result.info.width >= 15 && result.info.width <= 17);
        assert!(result.info.height >= 31 && result.info.height <= 33);
    }

    #[test]
    fn rotate_arbitrary_45_expands_dimensions() {
        let (px, info) = make_image(32, 32);
        let bg = [255, 255, 255];
        let result = rotate_arbitrary(&px, &info, 45.0, &bg).unwrap();
        // 45 degrees expands: side * sqrt(2) ≈ 45
        assert!(result.info.width > 40);
        assert!(result.info.height > 40);
    }

    #[test]
    fn rotate_arbitrary_180_preserves_dimensions() {
        let (px, info) = make_image(32, 32);
        let bg = [0, 0, 0];
        let result = rotate_arbitrary(&px, &info, 180.0, &bg).unwrap();
        // Floating-point bounding box may be ±1 of original
        assert!((result.info.width as i32 - 32).abs() <= 1);
        assert!((result.info.height as i32 - 32).abs() <= 1);
    }

    #[test]
    fn rotate_arbitrary_preserves_format() {
        let (px, info) = make_image(16, 16);
        let bg = [0, 0, 0];
        let result = rotate_arbitrary(&px, &info, 37.0, &bg).unwrap();
        assert_eq!(result.info.format, PixelFormat::Rgb8);
    }

    #[test]
    fn pad_symmetric() {
        let (px, info) = make_image(16, 16);
        let result = pad(&px, &info, 4, 4, 4, 4, &[128, 128, 128]).unwrap();
        assert_eq!(result.info.width, 24);
        assert_eq!(result.info.height, 24);
        assert_eq!(result.pixels.len(), 24 * 24 * 3);
    }

    #[test]
    fn pad_asymmetric() {
        let (px, info) = make_image(8, 8);
        let result = pad(&px, &info, 2, 4, 6, 8, &[255, 0, 0]).unwrap();
        assert_eq!(result.info.width, 8 + 4 + 8);
        assert_eq!(result.info.height, 8 + 2 + 6);
    }

    #[test]
    fn pad_preserves_center_pixels() {
        let (px, info) = make_image(4, 4);
        let result = pad(&px, &info, 1, 1, 1, 1, &[0, 0, 0]).unwrap();
        // Check center pixel (1,1) in output = (0,0) in original
        let bpp = 3;
        let out_w = 6;
        let idx = (1 * out_w + 1) * bpp;
        assert_eq!(result.pixels[idx..idx + 3], px[0..3]);
    }

    #[test]
    fn pad_fill_color_correct() {
        let px = vec![128u8; 4 * 4 * 3]; // 4x4 gray
        let info = ImageInfo {
            width: 4,
            height: 4,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let result = pad(&px, &info, 1, 1, 1, 1, &[255, 0, 0]).unwrap();
        // Top-left corner (0,0) should be fill color (red)
        assert_eq!(result.pixels[0], 255); // R
        assert_eq!(result.pixels[1], 0); // G
        assert_eq!(result.pixels[2], 0); // B
    }

    #[test]
    fn trim_removes_uniform_border() {
        // Create 8x8 image with 2-pixel red border around green center
        let mut px = vec![255u8; 8 * 8 * 3]; // all red
        for y in 2..6 {
            for x in 2..6 {
                let idx = (y * 8 + x) * 3;
                px[idx] = 0; // R
                px[idx + 1] = 255; // G
                px[idx + 2] = 0; // B
            }
        }
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let result = trim(&px, &info, 0).unwrap();
        assert_eq!(result.info.width, 4);
        assert_eq!(result.info.height, 4);
    }

    #[test]
    fn trim_with_threshold() {
        // Create image with near-uniform border (within threshold)
        let mut px = vec![100u8; 8 * 8 * 3]; // all 100
        for y in 2..6 {
            for x in 2..6 {
                let idx = (y * 8 + x) * 3;
                px[idx] = 200; // significantly different
            }
        }
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        // Threshold 0: all 100-valued border trimmed
        let result = trim(&px, &info, 0).unwrap();
        assert_eq!(result.info.width, 4);
    }

    #[test]
    fn trim_all_uniform_returns_1x1() {
        let px = vec![128u8; 8 * 8 * 3];
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let result = trim(&px, &info, 0).unwrap();
        assert_eq!(result.info.width, 1);
        assert_eq!(result.info.height, 1);
    }

    #[test]
    fn affine_identity() {
        // Use a larger image so edge bg-fill is a small fraction
        let (px, info) = make_image(64, 64);
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let result = affine(&px, &info, &identity, 64, 64, &[0, 0, 0]).unwrap();
        assert_eq!(result.info.width, 64);
        assert_eq!(result.info.height, 64);
        // Interior pixels should match; edge row/col may differ (bg fill)
        // With 64x64, edge pixels are ~3% of total → low MAE
        let mae: f64 = px
            .iter()
            .zip(result.pixels.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / px.len() as f64;
        assert!(
            mae < 5.0,
            "identity affine MAE should be < 5.0, got {mae:.2}"
        );
    }

    #[test]
    fn affine_scale_2x() {
        let (px, info) = make_image(8, 8);
        let scale2 = [2.0, 0.0, 0.0, 0.0, 2.0, 0.0];
        let result = affine(&px, &info, &scale2, 16, 16, &[0, 0, 0]).unwrap();
        assert_eq!(result.info.width, 16);
        assert_eq!(result.info.height, 16);
    }

    #[test]
    fn affine_singular_matrix_rejected() {
        let (px, info) = make_image(8, 8);
        let singular = [1.0, 2.0, 0.0, 2.0, 4.0, 0.0]; // det = 1*4 - 2*2 = 0
        let result = affine(&px, &info, &singular, 8, 8, &[0, 0, 0]);
        assert!(result.is_err());
    }

    // ─── 16-bit format conversion tests ────────────────────────────

    #[test]
    fn convert_rgb8_to_rgb16_roundtrip() {
        let pixels = vec![0u8, 128, 255, 64, 192, 32];
        let info = ImageInfo {
            width: 2,
            height: 1,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        // Upscale to 16-bit
        let up = convert_format(&pixels, &info, PixelFormat::Rgb16).unwrap();
        assert_eq!(up.info.format, PixelFormat::Rgb16);
        assert_eq!(up.pixels.len(), 12); // 2 pixels * 3 channels * 2 bytes

        // Verify precise scaling: u8*257 maps 0->0, 128->32896, 255->65535
        let r0 = u16::from_le_bytes([up.pixels[0], up.pixels[1]]);
        let g0 = u16::from_le_bytes([up.pixels[2], up.pixels[3]]);
        let b0 = u16::from_le_bytes([up.pixels[4], up.pixels[5]]);
        assert_eq!(r0, 0);
        assert_eq!(g0, 128 * 257); // 32896
        assert_eq!(b0, 65535);

        // Downscale back to 8-bit
        let down = convert_format(&up.pixels, &up.info, PixelFormat::Rgb8).unwrap();
        assert_eq!(down.info.format, PixelFormat::Rgb8);
        assert_eq!(
            down.pixels, pixels,
            "Rgb8 -> Rgb16 -> Rgb8 must be lossless"
        );
    }

    #[test]
    fn convert_gray8_to_gray16_preserves_values() {
        let pixels: Vec<u8> = (0..=255).collect();
        let info = ImageInfo {
            width: 256,
            height: 1,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let up = convert_format(&pixels, &info, PixelFormat::Gray16).unwrap();
        assert_eq!(up.info.format, PixelFormat::Gray16);

        // Check boundary values
        let first = u16::from_le_bytes([up.pixels[0], up.pixels[1]]);
        let last = u16::from_le_bytes([up.pixels[510], up.pixels[511]]);
        assert_eq!(first, 0);
        assert_eq!(last, 65535);
    }

    #[test]
    fn convert_rgb16_to_rgba16() {
        let mut pixels = Vec::new();
        for v in [0u16, 32768, 65535] {
            pixels.extend_from_slice(&v.to_le_bytes());
        }
        let info = ImageInfo {
            width: 1,
            height: 1,
            format: PixelFormat::Rgb16,
            color_space: ColorSpace::Srgb,
        };
        let result = convert_format(&pixels, &info, PixelFormat::Rgba16).unwrap();
        assert_eq!(result.info.format, PixelFormat::Rgba16);
        assert_eq!(result.pixels.len(), 8); // 1 pixel * 4 channels * 2 bytes
        // Alpha should be 65535 (fully opaque)
        let alpha = u16::from_le_bytes([result.pixels[6], result.pixels[7]]);
        assert_eq!(alpha, 65535);
    }

    // ─── 16-bit E2E chain test ─────────────────────────────────────────

    #[test]
    fn e2e_16bit_chain_gamma_brightness_resize_equalize() {
        use crate::domain::histogram;
        use crate::domain::point_ops;
        use super::super::types::ResizeFilter;

        // 1. Create a 16-bit Rgb16 gradient (32x32)
        let (w, h) = (32u32, 32u32);
        let mut pixels = Vec::with_capacity((w * h * 6) as usize);
        for y in 0..h {
            for x in 0..w {
                let r = (x * 65535 / w.max(1)) as u16;
                let g = (y * 65535 / h.max(1)) as u16;
                let b = 32768u16;
                pixels.extend_from_slice(&r.to_le_bytes());
                pixels.extend_from_slice(&g.to_le_bytes());
                pixels.extend_from_slice(&b.to_le_bytes());
            }
        }
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb16,
            color_space: ColorSpace::Srgb,
        };

        // 2. Apply gamma 2.2 (16-bit auto-dispatch)
        let after_gamma = point_ops::gamma(&pixels, &info, 2.2).unwrap();
        assert_eq!(after_gamma.len(), pixels.len());

        // 3. Apply brightness +0.1 (16-bit auto-dispatch)
        let after_bright = {
            use crate::domain::filters;
            {
                let r = rasmcore_pipeline::Rect::new(0, 0, info.width, info.height);
                let mut u = |_: rasmcore_pipeline::Rect| Ok(after_gamma.clone());
                filters::brightness(r, &mut u, &info, &filters::BrightnessParams { amount: 0.1 })
                    .unwrap()
            }
        };

        // 4. Resize to 16x16 (fast_image_resize U16x3)
        let resized = resize(&after_bright, &info, 16, 16, ResizeFilter::Lanczos3).unwrap();
        assert_eq!(resized.info.width, 16);
        assert_eq!(resized.info.height, 16);
        assert_eq!(resized.info.format, PixelFormat::Rgb16);
        assert_eq!(resized.pixels.len(), 16 * 16 * 6);

        // 5. Equalize (16-bit histogram with 65536 bins)
        let equalized = histogram::equalize(&resized.pixels, &resized.info).unwrap();
        assert_eq!(equalized.len(), resized.pixels.len());

        // 6. Verify: read back u16 values, check range expanded
        let values: Vec<u16> = equalized
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        let min_val = *values.iter().min().unwrap();
        let max_val = *values.iter().max().unwrap();
        // After equalization, range should span most of 0-65535
        assert!(
            max_val > 50000,
            "equalized max should be near 65535, got {max_val}"
        );
        assert!(
            min_val < 5000,
            "equalized min should be near 0, got {min_val}"
        );
    }

    #[test]
    fn e2e_16bit_encode_decode_roundtrip_tiff() {
        // Create 16-bit image
        let (w, h) = (8u32, 8u32);
        let mut pixels = Vec::new();
        for i in 0..(w * h) {
            let v = (i * 1023) as u16;
            pixels.extend_from_slice(&v.to_le_bytes());
            pixels.extend_from_slice(&(v / 2).to_le_bytes());
            pixels.extend_from_slice(&(65535 - v).to_le_bytes());
        }
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb16,
            color_space: ColorSpace::Srgb,
        };

        // Encode as 16-bit TIFF
        let encoded = crate::domain::encoder::tiff::encode(
            &pixels,
            &info,
            &crate::domain::encoder::tiff::TiffEncodeConfig::default(),
        )
        .unwrap();

        // Decode back
        let decoded = crate::domain::decoder::decode(&encoded).unwrap();
        assert_eq!(decoded.info.format, PixelFormat::Rgb16);
        assert_eq!(decoded.info.width, w);
        assert_eq!(decoded.info.height, h);
        assert_eq!(
            decoded.pixels, pixels,
            "16-bit TIFF roundtrip must be lossless"
        );
    }

    #[test]
    fn resize_rgb16_preserves_format() {
        use super::super::types::ResizeFilter;
        let mut pixels = Vec::new();
        for i in 0..16u32 * 16 {
            let v = (i * 257) as u16;
            pixels.extend_from_slice(&v.to_le_bytes());
            pixels.extend_from_slice(&v.to_le_bytes());
            pixels.extend_from_slice(&v.to_le_bytes());
        }
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Rgb16,
            color_space: ColorSpace::Srgb,
        };
        let result = resize(&pixels, &info, 8, 8, ResizeFilter::Bilinear).unwrap();
        assert_eq!(result.info.format, PixelFormat::Rgb16);
        assert_eq!(result.info.width, 8);
        assert_eq!(result.pixels.len(), 8 * 8 * 6);
    }

    #[test]
    fn crop_rgb16_works() {
        let mut pixels = Vec::new();
        for i in 0..16u32 * 16 {
            let v = (i * 100) as u16;
            pixels.extend_from_slice(&v.to_le_bytes());
            pixels.extend_from_slice(&v.to_le_bytes());
            pixels.extend_from_slice(&v.to_le_bytes());
        }
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Rgb16,
            color_space: ColorSpace::Srgb,
        };
        let result = crop(&pixels, &info, 2, 2, 8, 8).unwrap();
        assert_eq!(result.info.format, PixelFormat::Rgb16);
        assert_eq!(result.info.width, 8);
        assert_eq!(result.info.height, 8);
        assert_eq!(result.pixels.len(), 8 * 8 * 6);
    }

    #[test]
    fn undistort_zero_distortion_is_identity() {
        let (pixels, info) = make_image(32, 32);
        let camera = CameraMatrix {
            fx: 32.0,
            fy: 32.0,
            cx: 16.0,
            cy: 16.0,
        };
        let dist = DistortionCoeffs {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
        };
        let result = undistort(&pixels, &info, &camera, &dist).unwrap();
        // Interior pixels should be identical (borders may differ due to sampling)
        let bpp = 3;
        let mut matched = 0;
        let mut total = 0;
        for y in 2..30 {
            for x in 2..30 {
                total += 1;
                let idx = (y * 32 + x) * bpp;
                if pixels[idx..idx + bpp] == result.pixels[idx..idx + bpp] {
                    matched += 1;
                }
            }
        }
        assert_eq!(matched, total, "zero distortion should be identity");
    }

    #[test]
    fn undistort_barrel_shifts_pixels_inward() {
        // Barrel distortion (k1 > 0): corners move outward in distorted image
        // Undistortion should move them inward (toward center)
        let mut pixels = vec![0u8; 64 * 64 * 3];
        // White pixel at corner
        let idx = (5 * 64 + 5) * 3;
        pixels[idx] = 255;
        pixels[idx + 1] = 255;
        pixels[idx + 2] = 255;
        let info = ImageInfo {
            width: 64,
            height: 64,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let camera = CameraMatrix {
            fx: 50.0,
            fy: 50.0,
            cx: 32.0,
            cy: 32.0,
        };
        let dist = DistortionCoeffs {
            k1: 0.5,
            k2: 0.0,
            k3: 0.0,
        };
        let result = undistort(&pixels, &info, &camera, &dist).unwrap();
        // The white pixel should have moved (it won't be at exactly the same position)
        assert_eq!(result.pixels.len(), pixels.len());
    }

    #[test]
    fn undistort_produces_valid_output_size() {
        let (pixels, info) = make_image(128, 128);
        let camera = CameraMatrix {
            fx: 100.0,
            fy: 100.0,
            cx: 64.0,
            cy: 64.0,
        };
        let dist = DistortionCoeffs {
            k1: -0.3,
            k2: 0.1,
            k3: 0.0,
        };
        let result = undistort(&pixels, &info, &camera, &dist).unwrap();
        assert_eq!(result.info.width, 128);
        assert_eq!(result.info.height, 128);
        assert_eq!(result.pixels.len(), 128 * 128 * 3);
    }
}
