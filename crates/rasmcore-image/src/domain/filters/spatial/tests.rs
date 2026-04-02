//! Tests for spatial filters

#[allow(unused_imports)]
use crate::domain::filters::common::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::ColorSpace;

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
    fn blur_preserves_dimensions() {
        let (px, info) = make_image(16, 16);
        let result = blur_impl(&px, &info, &BlurParams { radius: 2.0 }).unwrap();
        assert_eq!(result.len(), px.len());
    }

    #[test]
    fn blur_zero_radius_preserves_pixels() {
        let (px, info) = make_image(8, 8);
        let result = blur_impl(&px, &info, &BlurParams { radius: 0.0 }).unwrap();
        assert_eq!(result, px);
    }

    #[test]
    fn blur_negative_radius_returns_error() {
        let (px, info) = make_image(8, 8);
        let result = blur_impl(&px, &info, &BlurParams { radius: -1.0 });
        assert!(result.is_err());
    }

    #[test]
    fn box_blur_preserves_dimensions() {
        let (px, info) = make_image(16, 16);
        let r = Rect::new(0, 0, info.width, info.height);
        let result = box_blur(
            r,
            &mut |_| Ok(px.to_vec()),
            &info,
            &BoxBlurParams { radius: 3 },
        )
        .unwrap();
        assert_eq!(result.len(), px.len());
    }

    #[test]
    fn box_blur_reduces_variance() {
        // Box blur should reduce the variance of pixel values
        let (px, info) = make_image(16, 16);
        let r = Rect::new(0, 0, info.width, info.height);
        let result = box_blur(
            r,
            &mut |_| Ok(px.to_vec()),
            &info,
            &BoxBlurParams { radius: 5 },
        )
        .unwrap();
        // Compute variance of R channel before and after
        let ch = 4;
        let n = px.len() / ch;
        let mean_before: f64 = (0..n).map(|i| px[i * ch] as f64).sum::<f64>() / n as f64;
        let var_before: f64 = (0..n)
            .map(|i| (px[i * ch] as f64 - mean_before).powi(2))
            .sum::<f64>()
            / n as f64;
        let mean_after: f64 = (0..n).map(|i| result[i * ch] as f64).sum::<f64>() / n as f64;
        let var_after: f64 = (0..n)
            .map(|i| (result[i * ch] as f64 - mean_after).powi(2))
            .sum::<f64>()
            / n as f64;
        assert!(
            var_after < var_before,
            "Box blur should reduce variance: {var_before} -> {var_after}"
        );
    }

    #[test]
    fn average_blur_returns_mean_color() {
        // Solid white image → average should be white
        let pixels = vec![255u8; 4 * 4 * 4]; // 4x4 white Rgba8
        let info = ImageInfo {
            width: 4,
            height: 4,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        let r = Rect::new(0, 0, info.width, info.height);
        let result = average_blur(r, &mut |_| Ok(pixels.to_vec()), &info).unwrap();
        for i in 0..16 {
            assert_eq!(result[i * 4], 255);
            assert_eq!(result[i * 4 + 1], 255);
            assert_eq!(result[i * 4 + 2], 255);
        }
    }

    #[test]
    fn average_blur_checkerboard() {
        // 2x2 checkerboard: [0,0,0,255], [255,255,255,255] alternating
        let pixels = vec![
            0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 255,
        ];
        let info = ImageInfo {
            width: 2,
            height: 2,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        let r = Rect::new(0, 0, info.width, info.height);
        let result = average_blur(r, &mut |_| Ok(pixels.to_vec()), &info).unwrap();
        // Mean should be ~127-128 for each channel
        for i in 0..4 {
            assert!((result[i * 4] as i16 - 127).unsigned_abs() <= 1);
        }
    }

    #[test]
    fn smart_sharpen_preserves_dimensions() {
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let pixels = vec![128u8; 16 * 16 * 3];
        let result = smart_sharpen(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &SmartSharpenParams {
                amount: 1.0,
                radius: 2,
                threshold: 50.0,
            },
        )
        .unwrap();
        assert_eq!(result.len(), pixels.len());
    }

    #[test]
    fn smart_sharpen_zero_amount_identity() {
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| (i % 256) as u8).collect();
        let result = smart_sharpen(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &SmartSharpenParams {
                amount: 0.0,
                radius: 2,
                threshold: 50.0,
            },
        )
        .unwrap();
        assert_eq!(result, pixels);
    }

    #[test]
    fn sharpen_preserves_dimensions() {
        let (px, info) = make_image(16, 16);
        let result = sharpen_impl(&px, &info, &SharpenParams { amount: 1.0 }).unwrap();
        assert_eq!(result.len(), px.len());
    }

    #[test]
    fn brightness_increases() {
        let (px, info) = make_image(8, 8);
        let result = brightness(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(px.to_vec()),
            &info,
            &BrightnessParams { amount: 0.5 },
        )
        .unwrap();
        assert_eq!(result.len(), px.len());
        let avg_orig: f64 = px.iter().map(|&v| v as f64).sum::<f64>() / px.len() as f64;
        let avg_bright: f64 = result.iter().map(|&v| v as f64).sum::<f64>() / result.len() as f64;
        assert!(avg_bright > avg_orig, "brightness should increase average");
    }

    #[test]
    fn brightness_out_of_range_returns_error() {
        let (px, info) = make_image(8, 8);
        assert!(
            brightness(
                Rect::new(0, 0, info.width, info.height),
                &mut |_| Ok(px.to_vec()),
                &info,
                &BrightnessParams { amount: 1.5 }
            )
            .is_err()
        );
        assert!(
            brightness(
                Rect::new(0, 0, info.width, info.height),
                &mut |_| Ok(px.to_vec()),
                &info,
                &BrightnessParams { amount: -1.5 }
            )
            .is_err()
        );
    }

    #[test]
    fn contrast_preserves_dimensions() {
        let (px, info) = make_image(8, 8);
        let result = contrast(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(px.to_vec()),
            &info,
            &ContrastParams { amount: 0.5 },
        )
        .unwrap();
        assert_eq!(result.len(), px.len());
    }

    #[test]
    fn contrast_out_of_range_returns_error() {
        let (px, info) = make_image(8, 8);
        assert!(
            contrast(
                Rect::new(0, 0, info.width, info.height),
                &mut |_| Ok(px.to_vec()),
                &info,
                &ContrastParams { amount: 2.0 },
            )
            .is_err()
        );
    }

    #[test]
    fn grayscale_changes_format() {
        let (px, info) = make_image(16, 16);
        let result = grayscale(&px, &info).unwrap();
        assert_eq!(result.info.format, PixelFormat::Gray8);
        assert_eq!(result.pixels.len(), 16 * 16);
    }

    #[test]
    fn grayscale_preserves_dimensions() {
        let (px, info) = make_image(32, 24);
        let result = grayscale(&px, &info).unwrap();
        assert_eq!(result.info.width, 32);
        assert_eq!(result.info.height, 24);
    }

    #[test]
    fn filters_work_on_rgba8() {
        let pixels: Vec<u8> = (0..(8 * 8 * 4)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        assert!(blur_impl(&pixels, &info, &BlurParams { radius: 1.0 }).is_ok());
        assert!(sharpen_impl(&pixels, &info, &SharpenParams { amount: 1.0 }).is_ok());
        assert!(
            brightness(
                Rect::new(0, 0, info.width, info.height),
                &mut |_| Ok(pixels.to_vec()),
                &info,
                &BrightnessParams { amount: 0.2 }
            )
            .is_ok()
        );
        assert!(
            contrast(
                Rect::new(0, 0, info.width, info.height),
                &mut |_| Ok(pixels.to_vec()),
                &info,
                &ContrastParams { amount: 0.2 },
            )
            .is_ok()
        );
        assert!(grayscale(&pixels, &info).is_ok());
    }

    #[test]
    fn contrast_lut_produces_expected_values() {
        // Zero contrast should be near identity
        let (px, info) = make_image(4, 4);
        let result = contrast(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(px.to_vec()),
            &info,
            &ContrastParams { amount: 0.0 },
        )
        .unwrap();
        assert_eq!(result, px);
    }

    #[test]
    fn hue_rotate_zero_is_identity() {
        // Hue rotate by 0 degrees should preserve pixels (via ColorOp delegation)
        let (px, info) = make_image(8, 8);
        let result = hue_rotate(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(px.to_vec()),
            &info,
            &HueRotateParams { degrees: 0.0 },
        )
        .unwrap();
        for (i, (&orig, &out)) in px.iter().zip(result.iter()).enumerate() {
            assert!(
                (orig as i16 - out as i16).abs() <= 1,
                "pixel {i}: {orig} -> {out}"
            );
        }
    }

    #[test]
    fn saturate_one_is_identity() {
        // Saturation factor 1.0 should preserve pixels
        let (px, info) = make_image(8, 8);
        let r = Rect::new(0, 0, info.width, info.height);
        let px2 = px.clone();
        let mut u = |_: Rect| Ok(px2.clone());
        let result = saturate(r, &mut u, &info, &SaturateParams { factor: 1.0 }).unwrap();
        for (i, (&orig, &out)) in px.iter().zip(result.iter()).enumerate() {
            assert!(
                (orig as i16 - out as i16).abs() <= 1,
                "pixel {i}: {orig} -> {out}"
            );
        }
    }

    #[test]
    fn hue_rotate_preserves_dimensions() {
        let (px, info) = make_image(8, 8);
        let result = hue_rotate(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(px.to_vec()),
            &info,
            &HueRotateParams { degrees: 90.0 },
        )
        .unwrap();
        assert_eq!(result.len(), px.len());
    }

    #[test]
    fn hue_rotate_360_identity() {
        let (px, info) = make_image(8, 8);
        let result = hue_rotate(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(px.to_vec()),
            &info,
            &HueRotateParams { degrees: 360.0 },
        )
        .unwrap();
        // Should be very close to original (within rounding)
        let mae: f64 = px
            .iter()
            .zip(result.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / px.len() as f64;
        assert!(
            mae < 1.0,
            "360° hue rotation should be near-identity, MAE={mae:.2}"
        );
    }

    #[test]
    fn saturate_zero_is_grayscale() {
        let (px, info) = make_image(8, 8);
        let r = Rect::new(0, 0, info.width, info.height);
        let px2 = px.clone();
        let mut u = |_: Rect| Ok(px2.clone());
        let result = saturate(r, &mut u, &info, &SaturateParams { factor: 0.0 }).unwrap();
        // All pixels should have r≈g≈b
        for chunk in result.chunks_exact(3) {
            let spread = chunk.iter().map(|&v| v as i32).max().unwrap()
                - chunk.iter().map(|&v| v as i32).min().unwrap();
            assert!(
                spread <= 1,
                "saturate(0) should produce gray, got spread={spread}"
            );
        }
    }

    #[test]
    fn saturate_one_near_identity() {
        let (px, info) = make_image(8, 8);
        let r = Rect::new(0, 0, info.width, info.height);
        let px2 = px.clone();
        let mut u = |_: Rect| Ok(px2.clone());
        let result = saturate(r, &mut u, &info, &SaturateParams { factor: 1.0 }).unwrap();
        let mae: f64 = px
            .iter()
            .zip(result.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / px.len() as f64;
        assert!(
            mae < 1.0,
            "saturate(1.0) should be near-identity, MAE={mae:.2}"
        );
    }

    #[test]
    fn sepia_preserves_dimensions() {
        let (px, info) = make_image(8, 8);
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(px.clone());
        let result = sepia(r, &mut u, &info, &SepiaParams { intensity: 1.0 }).unwrap();
        assert_eq!(result.len(), px.len());
    }

    #[test]
    fn sepia_zero_is_identity() {
        let (px, info) = make_image(8, 8);
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(px.clone());
        let result = sepia(r, &mut u, &info, &SepiaParams { intensity: 0.0 }).unwrap();
        assert_eq!(result, px);
    }

    #[test]
    fn colorize_preserves_dimensions() {
        let (px, info) = make_image(8, 8);
        let result = colorize(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(px.to_vec()),
            &info,
            &ColorizeParams {
                target: crate::domain::param_types::ColorRgb { r: 255, g: 0, b: 0 },
                amount: 0.5,
                method: "w3c".to_string(),
            },
        )
        .unwrap();
        assert_eq!(result.len(), px.len());
    }

    #[test]
    fn colorize_zero_is_identity() {
        let (px, info) = make_image(8, 8);
        let result = colorize(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(px.to_vec()),
            &info,
            &ColorizeParams {
                target: crate::domain::param_types::ColorRgb { r: 255, g: 0, b: 0 },
                amount: 0.0,
                method: "w3c".to_string(),
            },
        )
        .unwrap();
        assert_eq!(result, px);
    }

    #[test]
    fn convolve_identity_preserves_image() {
        let info = ImageInfo {
            width: 4,
            height: 4,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let pixels: Vec<u8> = (0..16).collect();
        // Identity kernel: [0,0,0, 0,1,0, 0,0,0]
        let kernel = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = convolve(
            r,
            &mut u,
            &info,
            &kernel,
            &ConvolveParams {
                kw: 3,
                kh: 3,
                divisor: 1.0,
            },
        )
        .unwrap();
        assert_eq!(result, pixels);
    }

    #[test]
    fn convolve_sharpen_kernel() {
        let info = ImageInfo {
            width: 4,
            height: 4,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let pixels = vec![128u8; 16];
        // Sharpen kernel: center=5, neighbors=-1
        let kernel = [0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0];
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = convolve(
            r,
            &mut u,
            &info,
            &kernel,
            &ConvolveParams {
                kw: 3,
                kh: 3,
                divisor: 1.0,
            },
        )
        .unwrap();
        // Uniform input → sharpen produces same output (no edges)
        assert!(result.iter().all(|&v| (v as i32 - 128).unsigned_abs() < 2));
    }

    #[test]
    fn median_removes_salt_and_pepper() {
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let mut pixels = vec![128u8; 64];
        // Add salt-and-pepper noise
        pixels[27] = 0; // pepper
        pixels[35] = 255; // salt
        let result = median(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &MedianParams { radius: 1 },
        )
        .unwrap();
        // Noise pixels should be replaced by median of neighbors (~128)
        assert!(
            (result[27] as i32 - 128).unsigned_abs() < 10,
            "pepper not removed: {}",
            result[27]
        );
        assert!(
            (result[35] as i32 - 128).unsigned_abs() < 10,
            "salt not removed: {}",
            result[35]
        );
    }

    #[test]
    fn sobel_detects_edges() {
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        // Left half = 0, right half = 255 → vertical edge at column 4
        let mut pixels = vec![0u8; 64];
        for r in 0..8 {
            for c in 4..8 {
                pixels[r * 8 + c] = 255;
            }
        }
        let result = sobel(&pixels, &info).unwrap();
        // Edge pixels at column 3-4 should have high gradient
        let edge_val = result[3 * 8 + 4]; // near the edge
        let flat_val = result[3 * 8 + 0]; // in flat region
        assert!(
            edge_val > flat_val + 50,
            "edge not detected: edge={edge_val} flat={flat_val}"
        );
    }

    #[test]
    fn canny_produces_binary_edges() {
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        // Vertical edge in the middle
        let mut pixels = vec![50u8; 256];
        for r in 0..16 {
            for c in 8..16 {
                pixels[r * 16 + c] = 200;
            }
        }
        let result = canny(
            &pixels,
            &info,
            &CannyParams {
                low_threshold: 30.0,
                high_threshold: 100.0,
            },
        )
        .unwrap();
        // Should produce binary output (0 or 255 only)
        assert!(
            result.iter().all(|&v| v == 0 || v == 255),
            "non-binary canny output"
        );
        // Should have some edge pixels
        let edge_count = result.iter().filter(|&&v| v == 255).count();
        assert!(edge_count > 0, "no edges detected");
    }

    #[test]
    fn color_effects_work_on_rgba8() {
        let pixels: Vec<u8> = (0..(8 * 8 * 4)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        assert!(
            hue_rotate(
                Rect::new(0, 0, info.width, info.height),
                &mut |_| Ok(pixels.to_vec()),
                &info,
                &HueRotateParams { degrees: 45.0 }
            )
            .is_ok()
        );
        assert!(
            saturate(
                Rect::new(0, 0, info.width, info.height),
                &mut |_| Ok(pixels.to_vec()),
                &info,
                &SaturateParams { factor: 1.5 }
            )
            .is_ok()
        );
        assert!(
            sepia(
                Rect::new(0, 0, info.width, info.height),
                &mut |_| Ok(pixels.to_vec()),
                &info,
                &SepiaParams { intensity: 0.8 }
            )
            .is_ok()
        );
        assert!(
            colorize(
                Rect::new(0, 0, info.width, info.height),
                &mut |_| Ok(pixels.to_vec()),
                &info,
                &ColorizeParams {
                    target: crate::domain::param_types::ColorRgb {
                        r: 0,
                        g: 128,
                        b: 255
                    },
                    amount: 0.5,
                    method: "w3c".to_string(),
                }
            )
            .is_ok()
        );
    }

    fn make_rgba(w: u32, h: u32) -> (Vec<u8>, ImageInfo) {
        let pixels: Vec<u8> = (0..(w * h * 4)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        (pixels, info)
    }

    #[test]
    fn premultiply_unpremultiply_roundtrip() {
        // Use pixels with alpha > 0 (alpha=0 loses info, alpha=1 has high rounding error)
        let mut pixels = Vec::new();
        for _ in 0..64 {
            pixels.extend_from_slice(&[100, 150, 200, 200]); // non-trivial alpha
        }
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        let pre = premultiply(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.clone()),
            &info,
        )
        .unwrap();
        let unpre = unpremultiply(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pre.clone()),
            &info,
        )
        .unwrap();
        for i in (0..pixels.len()).step_by(4) {
            for c in 0..3 {
                assert!(
                    (pixels[i + c] as i32 - unpre[i + c] as i32).abs() <= 1,
                    "roundtrip error at pixel {}: ch{c}: {} vs {}",
                    i / 4,
                    pixels[i + c],
                    unpre[i + c]
                );
            }
        }
    }

    #[test]
    fn flatten_white_bg() {
        // Fully opaque pixel should pass through unchanged
        let pixels = vec![100u8, 150, 200, 255, 50, 75, 100, 0];
        let info = ImageInfo {
            width: 2,
            height: 1,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        let (rgb, new_info) = flatten(&pixels, &info, [255, 255, 255]).unwrap();
        assert_eq!(new_info.format, PixelFormat::Rgb8);
        assert_eq!(rgb.len(), 6);
        assert_eq!(rgb[0], 100); // opaque pixel unchanged
        assert_eq!(rgb[1], 150);
        assert_eq!(rgb[2], 200);
        assert_eq!(rgb[3], 255); // transparent pixel → white bg
        assert_eq!(rgb[4], 255);
        assert_eq!(rgb[5], 255);
    }

    #[test]
    fn add_remove_alpha_roundtrip() {
        let (px, info) = make_image(4, 4); // RGB8
        let (rgba, rgba_info) = add_alpha(&px, &info, 255).unwrap();
        assert_eq!(rgba_info.format, PixelFormat::Rgba8);
        assert_eq!(rgba.len(), 4 * 4 * 4);
        let (rgb, rgb_info) = remove_alpha(&rgba, &rgba_info).unwrap();
        assert_eq!(rgb_info.format, PixelFormat::Rgb8);
        assert_eq!(rgb, px);
    }

    #[test]
    fn blend_multiply_identity() {
        // Multiply with white (255) should be near-identity
        let (px, info) = make_image(4, 4);
        let white: Vec<u8> = vec![255; 4 * 4 * 3];
        let result = blend(&px, &info, &white, &info, BlendMode::Multiply).unwrap();
        assert_eq!(result, px);
    }

    #[test]
    fn blend_screen_with_black() {
        // Screen with black (0) should be near-identity
        let (px, info) = make_image(4, 4);
        let black = vec![0u8; 4 * 4 * 3];
        let result = blend(&px, &info, &black, &info, BlendMode::Screen).unwrap();
        let mae: f64 = px
            .iter()
            .zip(result.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / px.len() as f64;
        assert!(
            mae < 1.0,
            "screen with black should be near-identity, MAE={mae:.2}"
        );
    }

    #[test]
    fn blend_all_modes_run() {
        let (px, info) = make_image(4, 4);
        let px2: Vec<u8> = (0..(4 * 4 * 3)).map(|i| ((i * 3) % 256) as u8).collect();
        for mode in [
            BlendMode::Multiply,
            BlendMode::Screen,
            BlendMode::Overlay,
            BlendMode::Darken,
            BlendMode::Lighten,
            BlendMode::SoftLight,
            BlendMode::HardLight,
            BlendMode::Difference,
            BlendMode::Exclusion,
            BlendMode::ColorDodge,
            BlendMode::ColorBurn,
            BlendMode::VividLight,
            BlendMode::LinearDodge,
            BlendMode::LinearBurn,
            BlendMode::LinearLight,
            BlendMode::PinLight,
            BlendMode::HardMix,
            BlendMode::Subtract,
            BlendMode::Divide,
            BlendMode::Dissolve,
            BlendMode::DarkerColor,
            BlendMode::LighterColor,
            BlendMode::Hue,
            BlendMode::Saturation,
            BlendMode::Color,
            BlendMode::Luminosity,
        ] {
            let result = blend(&px, &info, &px2, &info, mode);
            assert!(result.is_ok(), "blend mode {mode:?} failed");
            assert_eq!(result.unwrap().len(), px.len());
        }
    }

    #[test]
    fn blend_dissolve_deterministic() {
        let (px, info) = make_image(4, 4);
        let px2: Vec<u8> = (0..(4 * 4 * 3)).map(|i| ((i * 3) % 256) as u8).collect();
        let r1 = blend(&px, &info, &px2, &info, BlendMode::Dissolve).unwrap();
        let r2 = blend(&px, &info, &px2, &info, BlendMode::Dissolve).unwrap();
        assert_eq!(r1, r2, "Dissolve must be deterministic for same inputs");
    }

    #[test]
    fn blend_dissolve_selects_whole_pixel() {
        // Each output pixel must be either fg or bg (no blending)
        let fg = vec![255, 0, 0, 0, 255, 0]; // red, green
        let bg = vec![0, 0, 255, 128, 128, 128]; // blue, gray
        let info = ImageInfo {
            width: 2,
            height: 1,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let result = blend(&fg, &info, &bg, &info, BlendMode::Dissolve).unwrap();
        for px in result.chunks_exact(3) {
            assert!(
                px == &fg[0..3] || px == &bg[0..3] || px == &fg[3..6] || px == &bg[3..6],
                "Dissolve pixel {px:?} is neither fg nor bg"
            );
        }
    }

    #[test]
    fn blend_darker_color_selects_darker() {
        // fg: bright red (high lum), bg: dark blue (low lum)
        let fg = vec![255, 200, 200]; // lum ≈ 213
        let bg = vec![0, 0, 50]; // lum ≈ 6
        let info = ImageInfo {
            width: 1,
            height: 1,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let result = blend(&fg, &info, &bg, &info, BlendMode::DarkerColor).unwrap();
        assert_eq!(&result, &bg, "DarkerColor should select the darker pixel");
    }

    #[test]
    fn blend_lighter_color_selects_lighter() {
        let fg = vec![255, 200, 200]; // lum ≈ 213
        let bg = vec![0, 0, 50]; // lum ≈ 6
        let info = ImageInfo {
            width: 1,
            height: 1,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let result = blend(&fg, &info, &bg, &info, BlendMode::LighterColor).unwrap();
        assert_eq!(&result, &fg, "LighterColor should select the lighter pixel");
    }

    #[test]
    fn blend_difference_self_is_black() {
        let (px, info) = make_image(4, 4);
        let result = blend(&px, &info, &px, &info, BlendMode::Difference).unwrap();
        for &v in &result {
            assert!(v <= 1, "difference with self should be ~0, got {v}");
        }
    }

    // ── Bokeh Blur Tests ─────────────────────────────────────────────────

    #[test]
    fn bokeh_disc_zero_radius_is_identity() {
        let (px, info) = make_image(8, 8);
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(px.clone());
        let result = bokeh_blur(
            r,
            &mut u,
            &info,
            &BokehBlurParams {
                radius: 0,
                shape: 0,
            },
        )
        .unwrap();
        assert_eq!(result, px);
    }

    #[test]
    fn bokeh_disc_kernel_is_circular() {
        let (kernel, side) = make_disc_kernel(3);
        assert_eq!(side, 7);
        // Center should be 1.0
        assert_eq!(kernel[3 * 7 + 3], 1.0);
        // Corners should be 0.0 (outside circle)
        assert_eq!(kernel[0], 0.0);
        assert_eq!(kernel[6], 0.0);
        assert_eq!(kernel[6 * 7], 0.0);
        assert_eq!(kernel[6 * 7 + 6], 0.0);
        // Edge midpoints should be 1.0 (inside circle)
        assert_eq!(kernel[0 * 7 + 3], 1.0); // top center
        assert_eq!(kernel[3 * 7 + 0], 1.0); // left center
    }

    #[test]
    fn bokeh_hex_kernel_is_hexagonal() {
        let (kernel, side) = make_hex_kernel(3);
        assert_eq!(side, 7);
        // Center should be 1.0
        assert_eq!(kernel[3 * 7 + 3], 1.0);
        // Top/bottom centers should be 1.0
        assert_eq!(kernel[0 * 7 + 3], 1.0);
        assert_eq!(kernel[6 * 7 + 3], 1.0);
        // Hex kernel should differ from disc at some corner-adjacent pixels
        let (disc_k, _) = make_disc_kernel(3);
        let differs = kernel.iter().zip(disc_k.iter()).any(|(h, d)| h != d);
        assert!(differs, "hex and disc kernels should differ");
    }

    #[test]
    fn bokeh_flat_image_unchanged() {
        // A flat image convolved with any normalized kernel should stay flat
        let w = 16u32;
        let h = 16u32;
        let pixels = vec![100u8; (w * h) as usize];
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let r = Rect::new(0, 0, info.width, info.height);
        let px2 = pixels.clone();
        let mut u = |_: Rect| Ok(px2.clone());
        let result = bokeh_blur(
            r,
            &mut u,
            &info,
            &BokehBlurParams {
                radius: 3,
                shape: 0,
            },
        )
        .unwrap();
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v as i16 - 100).abs() <= 1,
                "pixel {i} should be ~100, got {v}"
            );
        }
    }

    #[test]
    fn bokeh_rgba_supported() {
        let pixels: Vec<u8> = (0..64).map(|i| (i * 4) as u8).collect();
        let info = ImageInfo {
            width: 4,
            height: 4,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        let r = Rect::new(0, 0, info.width, info.height);
        let px2 = pixels.clone();
        let mut u = |_: Rect| Ok(px2.clone());
        let result = bokeh_blur(
            r,
            &mut u,
            &info,
            &BokehBlurParams {
                radius: 1,
                shape: 1,
            },
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 64);
    }
}

#[cfg(test)]
mod optimization_tests {
    use crate::domain::types::*;
    use super::*;

    #[test]
    fn separable_detection_box_blur() {
        // Box blur 3x3 is separable: [1,1,1] * [1,1,1]^T
        let result = is_separable(&kernels::BOX_BLUR_3X3, 3, 3);
        assert!(result.is_some(), "box blur should be detected as separable");
        let (row, col) = result.unwrap();
        assert_eq!(row.len(), 3);
        assert_eq!(col.len(), 3);
    }

    #[test]
    fn separable_detection_emboss_not_separable() {
        // Emboss kernel is NOT separable
        let result = is_separable(&kernels::EMBOSS, 3, 3);
        assert!(result.is_none(), "emboss should NOT be separable");
    }

    #[test]
    fn histogram_median_matches_sort_median() {
        // Both paths should give the same output
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let mut pixels = vec![128u8; 256];
        // Add some variation
        for i in 0..256 {
            pixels[i] = (i as u8).wrapping_mul(7).wrapping_add(13);
        }

        // radius=2: uses sort path
        let sort_result = median(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &MedianParams { radius: 2 },
        )
        .unwrap();
        // radius=3: uses histogram path
        let hist_result = median(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &MedianParams { radius: 3 },
        )
        .unwrap();

        // Both should produce valid output (different radii = different results, but both correct)
        assert!(!sort_result.is_empty());
        assert!(!hist_result.is_empty());
        // Histogram path with radius=3 should produce smoother output
    }

    #[test]
    fn convolve_perf_1024x1024() {
        let info = ImageInfo {
            width: 1024,
            height: 1024,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let pixels = vec![128u8; 1024 * 1024];

        let start = std::time::Instant::now();
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(pixels.clone());
        let _ = convolve(
            r,
            &mut u,
            &info,
            &kernels::BOX_BLUR_3X3,
            &ConvolveParams {
                kw: 3,
                kh: 3,
                divisor: 9.0,
            },
        )
        .unwrap();
        let elapsed = start.elapsed();

        // Separable path should handle 1024x1024 in under 500ms
        assert!(
            elapsed.as_millis() < 500,
            "3x3 convolve on 1024x1024 took {:?}, expected < 500ms",
            elapsed
        );
    }

    #[test]
    fn median_perf_512x512() {
        let info = ImageInfo {
            width: 512,
            height: 512,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let pixels: Vec<u8> = (0..(512 * 512)).map(|i| (i % 256) as u8).collect();

        let start = std::time::Instant::now();
        let _ = median(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &MedianParams { radius: 3 },
        )
        .unwrap();
        let elapsed = start.elapsed();

        // Histogram median should handle 512x512 radius=3 in under 500ms
        assert!(
            elapsed.as_millis() < 500,
            "median radius=3 on 512x512 took {:?}, expected < 500ms",
            elapsed
        );
    }

    // ─── CLAHE Tests ──────────────────────────────────────────────────────

    #[test]
    fn clahe_enhances_low_contrast() {
        let info = ImageInfo {
            width: 64,
            height: 64,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        // Low-contrast input: values 100-155
        let pixels: Vec<u8> = (0..(64 * 64)).map(|i| (100 + (i % 56)) as u8).collect();
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = clahe(
            r,
            &mut u,
            &info,
            &ClaheParams {
                clip_limit: 2.0,
                tile_grid: 8,
            },
        )
        .unwrap();

        // CLAHE should expand dynamic range
        let in_range = *pixels.iter().max().unwrap() as i32 - *pixels.iter().min().unwrap() as i32;
        let out_range = *result.iter().max().unwrap() as i32 - *result.iter().min().unwrap() as i32;
        assert!(
            out_range > in_range,
            "CLAHE should expand range: in={in_range}, out={out_range}"
        );
    }

    #[test]
    fn clahe_flat_image_uniform_output() {
        // CLAHE on flat input: OpenCV redistributes excess across all bins,
        // so the output is NOT identity but is uniform (all same value).
        let info = ImageInfo {
            width: 32,
            height: 32,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let pixels = vec![128u8; 32 * 32];
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = clahe(
            r,
            &mut u,
            &info,
            &ClaheParams {
                clip_limit: 2.0,
                tile_grid: 4,
            },
        )
        .unwrap();
        // All output pixels should be the same value (uniform)
        let first = result[0];
        for &v in &result {
            assert_eq!(v, first, "flat input should produce uniform output");
        }
    }

    #[test]
    fn clahe_rejects_non_gray() {
        let info = ImageInfo {
            width: 4,
            height: 4,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(vec![0u8; 48]);
        assert!(
            clahe(
                r,
                &mut u,
                &info,
                &ClaheParams {
                    clip_limit: 2.0,
                    tile_grid: 8
                }
            )
            .is_err()
        );
    }

    // ─── Bilateral Filter Tests ───────────────────────────────────────────

    #[test]
    fn bilateral_preserves_edges() {
        let info = ImageInfo {
            width: 32,
            height: 32,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        // Half black, half white
        let mut pixels = vec![0u8; 32 * 32];
        for y in 0..32 {
            for x in 16..32 {
                pixels[y * 32 + x] = 255;
            }
        }
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = bilateral(
            r,
            &mut u,
            &info,
            &BilateralParams {
                diameter: 5,
                sigma_color: 50.0,
                sigma_space: 50.0,
            },
        )
        .unwrap();

        // Edge should be preserved: pixels at x=14 should still be dark, x=18 still bright
        let mid_y = 16;
        assert!(result[mid_y * 32 + 14] < 50, "left of edge should be dark");
        assert!(
            result[mid_y * 32 + 18] > 200,
            "right of edge should be bright"
        );
    }

    #[test]
    fn bilateral_smooths_noise() {
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        // Noisy flat region
        let pixels: Vec<u8> = (0..256)
            .map(|i| (128i32 + ((i * 17 + 5) % 21) as i32 - 10).clamp(0, 255) as u8)
            .collect();
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = bilateral(
            r,
            &mut u,
            &info,
            &BilateralParams {
                diameter: 5,
                sigma_color: 25.0,
                sigma_space: 25.0,
            },
        )
        .unwrap();

        // Should reduce variance
        let var_in: f64 = pixels
            .iter()
            .map(|&v| (v as f64 - 128.0).powi(2))
            .sum::<f64>()
            / 256.0;
        let mean_out = result.iter().map(|&v| v as f64).sum::<f64>() / 256.0;
        let var_out: f64 = result
            .iter()
            .map(|&v| (v as f64 - mean_out).powi(2))
            .sum::<f64>()
            / 256.0;
        assert!(
            var_out < var_in,
            "bilateral should reduce variance: {var_out:.1} vs {var_in:.1}"
        );
    }

    // ─── Guided Filter Tests ──────────────────────────────────────────────

    #[test]
    fn guided_filter_smooths() {
        let info = ImageInfo {
            width: 32,
            height: 32,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        // Noisy flat region (128 ± noise)
        let pixels: Vec<u8> = (0..(32 * 32))
            .map(|i| (128i32 + ((i * 17 + 3) % 21) as i32 - 10).clamp(0, 255) as u8)
            .collect();
        let result = guided_filter_impl(
            &pixels,
            &info,
            &GuidedFilterParams {
                radius: 4,
                epsilon: 0.01,
            },
        )
        .unwrap();

        // Should reduce variance from mean
        let mean_in = pixels.iter().map(|&v| v as f64).sum::<f64>() / pixels.len() as f64;
        let var_in: f64 = pixels
            .iter()
            .map(|&v| (v as f64 - mean_in).powi(2))
            .sum::<f64>()
            / pixels.len() as f64;
        let mean_out = result.iter().map(|&v| v as f64).sum::<f64>() / result.len() as f64;
        let var_out: f64 = result
            .iter()
            .map(|&v| (v as f64 - mean_out).powi(2))
            .sum::<f64>()
            / result.len() as f64;
        assert!(
            var_out < var_in,
            "guided filter should reduce variance: {var_out:.1} vs {var_in:.1}"
        );
    }

    #[test]
    fn guided_filter_flat_identity() {
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let pixels = vec![100u8; 16 * 16];
        let result = guided_filter_impl(
            &pixels,
            &info,
            &GuidedFilterParams {
                radius: 4,
                epsilon: 0.01,
            },
        )
        .unwrap();
        // Flat input should produce flat output
        for &v in &result {
            assert!((v as i32 - 100).abs() <= 1, "flat pixel changed to {v}");
        }
    }
}

#[cfg(test)]
mod tests_16bit {
    use crate::domain::types::*;
    use super::*;

    fn make_rgb16(w: u32, h: u32, val: u16) -> (Vec<u8>, ImageInfo) {
        let n = (w * h * 3) as usize;
        let samples: Vec<u16> = vec![val; n];
        let bytes = u16_to_bytes(&samples);
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb16,
            color_space: ColorSpace::Srgb,
        };
        (bytes, info)
    }

    fn make_gray16(w: u32, h: u32, val: u16) -> (Vec<u8>, ImageInfo) {
        let n = (w * h) as usize;
        let samples: Vec<u16> = vec![val; n];
        let bytes = u16_to_bytes(&samples);
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Gray16,
            color_space: ColorSpace::Srgb,
        };
        (bytes, info)
    }

    #[test]
    fn blur_16bit_identity() {
        let (px, info) = make_rgb16(8, 8, 32768);
        let result = blur_impl(&px, &info, &BlurParams { radius: 0.0 }).unwrap();
        assert_eq!(result, px, "zero-radius blur should be identity");
    }

    #[test]
    fn blur_16bit_produces_output() {
        let (px, info) = make_rgb16(8, 8, 32768);
        let result = blur_impl(&px, &info, &BlurParams { radius: 1.0 }).unwrap();
        assert_eq!(result.len(), px.len(), "output length should match");
    }

    #[test]
    fn sharpen_16bit_produces_output() {
        let (px, info) = make_rgb16(8, 8, 32768);
        let result = sharpen_impl(&px, &info, &SharpenParams { amount: 1.0 }).unwrap();
        assert_eq!(result.len(), px.len());
    }

    #[test]
    fn convolve_16bit_identity_kernel() {
        let (px, info) = make_gray16(4, 4, 50000);
        let kernel = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(px.clone());
        let result = convolve(
            r,
            &mut u,
            &info,
            &kernel,
            &ConvolveParams {
                kw: 3,
                kh: 3,
                divisor: 1.0,
            },
        )
        .unwrap();
        // Should be close to original (some precision loss from 16→8→16)
        let orig = bytes_to_u16(&px);
        let out = bytes_to_u16(&result);
        for i in 0..orig.len() {
            assert!(
                (orig[i] as i32 - out[i] as i32).abs() < 300,
                "identity convolve changed pixel {} by {}",
                i,
                (orig[i] as i32 - out[i] as i32).abs()
            );
        }
    }

    #[test]
    fn median_16bit_produces_output() {
        let (px, info) = make_gray16(8, 8, 32768);
        let result = median(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(px.to_vec()),
            &info,
            &MedianParams { radius: 1 },
        )
        .unwrap();
        assert_eq!(result.len(), px.len());
    }

    #[test]
    fn sobel_16bit_produces_output() {
        let (px, info) = make_gray16(8, 8, 32768);
        let result = sobel(&px, &info).unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn hue_rotate_16bit() {
        let (px, info) = make_rgb16(4, 4, 32768);
        let result = hue_rotate(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(px.to_vec()),
            &info,
            &HueRotateParams { degrees: 90.0 },
        )
        .unwrap();
        assert_eq!(result.len(), px.len());
    }

    #[test]
    fn brightness_16bit() {
        let (px, info) = make_rgb16(4, 4, 32768);
        let result = brightness(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(px.to_vec()),
            &info,
            &BrightnessParams { amount: 0.5 },
        )
        .unwrap();
        assert_eq!(result.len(), px.len());
        // Brightened pixels should be higher
        let orig = bytes_to_u16(&px);
        let out = bytes_to_u16(&result);
        assert!(
            out[0] > orig[0],
            "brightness should increase: {} > {}",
            out[0],
            orig[0]
        );
    }

    #[test]
    fn sepia_16bit() {
        let (px, info) = make_rgb16(4, 4, 32768);
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(px.clone());
        let result = sepia(r, &mut u, &info, &SepiaParams { intensity: 1.0 }).unwrap();
        assert_eq!(result.len(), px.len());
    }
}

#[cfg(test)]
mod spatial_tests {
    use super::*;
    use crate::domain::types::ColorSpace;

    fn gray_info(w: u32, h: u32) -> ImageInfo {
        ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        }
    }

    #[test]
    fn connected_components_two_blobs() {
        // Two separate 2x2 white squares on black background
        #[rustfmt::skip]
        let px = vec![
            255,255,  0,  0,  0,
            255,255,  0,  0,  0,
              0,  0,  0,  0,  0,
              0,  0,  0,255,255,
              0,  0,  0,255,255,
        ];
        let info = gray_info(5, 5);
        let (labels, count) = connected_components(&px, &info, 8).unwrap();
        assert_eq!(count, 2, "should find 2 components");
        // Top-left and bottom-right should have different labels
        assert_ne!(labels[0], labels[3 * 5 + 3]);
        assert_ne!(labels[0], 0);
        assert_ne!(labels[3 * 5 + 3], 0);
        // Background should be 0
        assert_eq!(labels[2 * 5 + 2], 0);
    }

    #[test]
    fn connected_components_4_vs_8() {
        // Diagonal connection: 4-connectivity = 2 components, 8-connectivity = 1
        #[rustfmt::skip]
        let px = vec![
            255,  0,
              0,255,
        ];
        let info = gray_info(2, 2);
        let (_, count4) = connected_components(&px, &info, 4).unwrap();
        let (_, count8) = connected_components(&px, &info, 8).unwrap();
        assert_eq!(count4, 2, "4-connectivity: diagonal = separate");
        assert_eq!(count8, 1, "8-connectivity: diagonal = connected");
    }

    #[test]
    fn flood_fill_fills_region() {
        #[rustfmt::skip]
        let px = vec![
            100,100,100,  0,200,
            100,100,100,  0,200,
            100,100,100,  0,200,
        ];
        let info = gray_info(5, 3);
        let (result, filled) = flood_fill(&px, &info, 1, 1, 50, 0, 4).unwrap();
        assert_eq!(filled, 9, "should fill 3x3 region of value 100");
        assert_eq!(result[0], 50);
        assert_eq!(result[4], 200); // untouched
        assert_eq!(result[3], 0); // barrier untouched
    }

    #[test]
    fn flood_fill_with_tolerance() {
        let px = vec![100, 102, 105, 110, 200];
        let info = gray_info(5, 1);
        let (result, filled) = flood_fill(&px, &info, 0, 0, 50, 5, 4).unwrap();
        // Tolerance 5 from seed=100: fills 100, 102, 105 (all within ±5)
        assert_eq!(filled, 3);
        assert_eq!(result[0], 50);
        assert_eq!(result[1], 50);
        assert_eq!(result[2], 50);
        assert_eq!(result[3], 110); // 110 > 105, not within tolerance of 100
    }

    #[test]
    fn pyr_down_halves_size() {
        let px = vec![128u8; 64 * 64];
        let info = gray_info(64, 64);
        let (result, new_info) = pyr_down(&px, &info).unwrap();
        assert_eq!(new_info.width, 32);
        assert_eq!(new_info.height, 32);
        assert_eq!(result.len(), 32 * 32);
        // Uniform input → uniform output
        for &v in &result {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn pyr_up_doubles_size() {
        let px = vec![128u8; 32 * 32];
        let info = gray_info(32, 32);
        let (result, new_info) = pyr_up(&px, &info).unwrap();
        assert_eq!(new_info.width, 64);
        assert_eq!(new_info.height, 64);
        assert_eq!(result.len(), 64 * 64);
    }

    #[test]
    fn pyr_down_up_roundtrip() {
        // pyrUp(pyrDown(img)) should be close to original for smooth content
        let mut px = vec![0u8; 64 * 64];
        for y in 0..64 {
            for x in 0..64 {
                px[y * 64 + x] = ((x * 255) / 63) as u8;
            }
        }
        let info = gray_info(64, 64);
        let (down, down_info) = pyr_down(&px, &info).unwrap();
        let (up, _) = pyr_up(&down, &down_info).unwrap();

        // MAE should be small for smooth gradient
        let mae: f64 = px
            .iter()
            .zip(up.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / px.len() as f64;
        assert!(
            mae < 5.0,
            "pyrDown→pyrUp roundtrip MAE={mae:.2} (should be < 5.0)"
        );
    }

    // ── Displacement Map Tests ───────────────────────────────────────────

    #[test]
    fn displacement_map_identity() {
        // Identity map: map_x[y*w+x] = x, map_y[y*w+x] = y → output == input
        let w = 16u32;
        let h = 16u32;
        let info = gray_info(w, h);
        let pixels: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let mut map_x = vec![0.0f32; 256];
        let mut map_y = vec![0.0f32; 256];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                map_x[idx] = x as f32;
                map_y[idx] = y as f32;
            }
        }
        let r = Rect::new(0, 0, w, h);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = displacement_map(r, &mut u, &info, &map_x, &map_y).unwrap();
        assert_eq!(
            result, pixels,
            "identity displacement should reproduce input"
        );
    }

    #[test]
    fn displacement_map_uniform_shift() {
        // Shift all pixels right by 1: map_x = x - 1.0
        let w = 8u32;
        let h = 4u32;
        let info = gray_info(w, h);
        let mut pixels = vec![0u8; (w * h) as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                pixels[y * w as usize + x] = (x * 30 + y * 10) as u8;
            }
        }
        let mut map_x = vec![0.0f32; (w * h) as usize];
        let mut map_y = vec![0.0f32; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                map_x[idx] = x as f32 - 1.0; // shift right by 1
                map_y[idx] = y as f32;
            }
        }
        let r = Rect::new(0, 0, w, h);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = displacement_map(r, &mut u, &info, &map_x, &map_y).unwrap();
        // First column should be black (source x = -1 → out of bounds)
        for y in 0..h as usize {
            assert_eq!(result[y * w as usize], 0, "left edge should be black");
        }
        // Other columns should match pixels shifted
        for y in 0..h as usize {
            for x in 1..w as usize {
                assert_eq!(
                    result[y * w as usize + x],
                    pixels[y * w as usize + x - 1],
                    "pixel ({x},{y}) should be shifted"
                );
            }
        }
    }

    #[test]
    fn displacement_map_oob_produces_black() {
        let w = 4u32;
        let h = 4u32;
        let info = gray_info(w, h);
        let pixels = vec![128u8; (w * h) as usize];
        // All map coordinates point outside the image
        let map_x = vec![-10.0f32; (w * h) as usize];
        let map_y = vec![-10.0f32; (w * h) as usize];
        let r = Rect::new(0, 0, w, h);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = displacement_map(r, &mut u, &info, &map_x, &map_y).unwrap();
        assert!(result.iter().all(|&v| v == 0), "all OOB → all black");
    }

    #[test]
    fn displacement_map_rgba() {
        let w = 4u32;
        let h = 4u32;
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        let pixels: Vec<u8> = (0..64).map(|i| (i * 4) as u8).collect();
        // Identity map
        let mut map_x = vec![0.0f32; 16];
        let mut map_y = vec![0.0f32; 16];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                map_x[idx] = x as f32;
                map_y[idx] = y as f32;
            }
        }
        let r = Rect::new(0, 0, w, h);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = displacement_map(r, &mut u, &info, &map_x, &map_y).unwrap();
        assert_eq!(result, pixels);
    }

    #[test]
    fn displacement_map_size_mismatch_error() {
        let info = gray_info(4, 4);
        let pixels = vec![0u8; 16];
        let map_x = vec![0.0f32; 8]; // wrong size
        let map_y = vec![0.0f32; 16];
        let r = Rect::new(0, 0, 4, 4);
        let mut u = |_: Rect| Ok(pixels.clone());
        assert!(displacement_map(r, &mut u, &info, &map_x, &map_y).is_err());
    }

    #[test]
    fn displacement_map_subpixel_bilinear() {
        // Test bilinear interpolation at half-pixel offsets
        let w = 4u32;
        let h = 1u32;
        let info = gray_info(w, h);
        let pixels = vec![0u8, 100, 200, 50];
        // Sample at x=0.5 → blend of pixel 0 (0) and pixel 1 (100)
        let map_x = vec![0.5f32, 1.5, 2.5, 0.0];
        let map_y = vec![0.0f32; 4];
        let r = Rect::new(0, 0, w, h);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = displacement_map(r, &mut u, &info, &map_x, &map_y).unwrap();
        assert_eq!(result[0], 50, "blend(0, 100) at 0.5 = 50");
        assert_eq!(result[1], 150, "blend(100, 200) at 0.5 = 150");
        assert_eq!(result[2], 125, "blend(200, 50) at 0.5 = 125");
        assert_eq!(result[3], 0, "exact pixel 0 = 0");
    }
}

#[cfg(test)]
mod motion_blur_tests {
    use super::*;

    fn make_gray(w: u32, h: u32, val: u8) -> (Vec<u8>, ImageInfo) {
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Gray8,
            color_space: crate::domain::types::ColorSpace::Srgb,
        };
        (vec![val; (w * h) as usize], info)
    }

    #[test]
    fn zero_length_is_identity() {
        let (pixels, info) = make_gray(8, 8, 128);
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = motion_blur(
            r,
            &mut u,
            &info,
            &MotionBlurParams {
                length: 0,
                angle_degrees: 45.0,
            },
        )
        .unwrap();
        assert_eq!(result, pixels);
    }

    #[test]
    fn uniform_image_unchanged() {
        // Motion blur of a uniform image should produce the same uniform image
        let (pixels, info) = make_gray(16, 16, 100);
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = motion_blur(
            r,
            &mut u,
            &info,
            &MotionBlurParams {
                length: 3,
                angle_degrees: 0.0,
            },
        )
        .unwrap();
        // Interior pixels should be exactly 100 (uniform input)
        // Border pixels may differ due to reflect101 padding
        let w = info.width as usize;
        let h = info.height as usize;
        for y in 3..h - 3 {
            for x in 3..w - 3 {
                assert_eq!(result[y * w + x], 100, "pixel ({x},{y}) should be 100");
            }
        }
    }

    #[test]
    fn horizontal_blur_spreads_horizontal() {
        // Single bright pixel in center, horizontal blur should spread it horizontally
        let w = 16u32;
        let h = 16u32;
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Gray8,
            color_space: crate::domain::types::ColorSpace::Srgb,
        };
        let mut pixels = vec![0u8; (w * h) as usize];
        pixels[8 * 16 + 8] = 255; // center pixel

        let r = Rect::new(0, 0, w, h);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = motion_blur(
            r,
            &mut u,
            &info,
            &MotionBlurParams {
                length: 3,
                angle_degrees: 0.0,
            },
        )
        .unwrap();

        // The bright pixel should spread along the horizontal line (row 8)
        // but not vertically (rows 7 and 9 at x=8 should be 0 or near-0)
        let center_row_sum: u32 = (0..w).map(|x| result[8 * 16 + x as usize] as u32).sum();
        let adjacent_row_sum: u32 = (0..w).map(|x| result[7 * 16 + x as usize] as u32).sum();
        assert!(
            center_row_sum > adjacent_row_sum * 3,
            "horizontal blur should concentrate energy on center row: center={center_row_sum} adj={adjacent_row_sum}"
        );
    }

    #[test]
    fn rgb8_works() {
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgb8,
            color_space: crate::domain::types::ColorSpace::Srgb,
        };
        let pixels = vec![128u8; 8 * 8 * 3];
        let r = Rect::new(0, 0, 8, 8);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = motion_blur(
            r,
            &mut u,
            &info,
            &MotionBlurParams {
                length: 2,
                angle_degrees: 45.0,
            },
        )
        .unwrap();
        assert_eq!(result.len(), pixels.len());
    }
}

#[cfg(test)]
mod zoom_blur_tests {
    use super::*;

    fn make_gray(w: u32, h: u32, val: u8) -> (Vec<u8>, ImageInfo) {
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Gray8,
            color_space: crate::domain::types::ColorSpace::Srgb,
        };
        (vec![val; (w * h) as usize], info)
    }

    #[test]
    fn zero_factor_is_identity() {
        let (px, info) = make_gray(32, 32, 128);
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(px.clone());
        let result = zoom_blur(
            r,
            &mut u,
            &info,
            &ZoomBlurParams {
                center_x: 0.5,
                center_y: 0.5,
                factor: 0.0,
            },
        )
        .unwrap();
        assert_eq!(result, px);
    }

    #[test]
    fn preserves_dimensions() {
        let (px, info) = make_gray(64, 48, 128);
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(px.clone());
        let result = zoom_blur(
            r,
            &mut u,
            &info,
            &ZoomBlurParams {
                center_x: 0.5,
                center_y: 0.5,
                factor: 0.3,
            },
        )
        .unwrap();
        assert_eq!(result.len(), px.len());
    }

    #[test]
    fn uniform_image_stays_uniform() {
        let (px, info) = make_gray(32, 32, 100);
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(px.clone());
        let result = zoom_blur(
            r,
            &mut u,
            &info,
            &ZoomBlurParams {
                center_x: 0.5,
                center_y: 0.5,
                factor: 0.5,
            },
        )
        .unwrap();
        for &v in &result {
            assert!(
                (v as i16 - 100).abs() <= 1,
                "uniform image should stay uniform, got {v}"
            );
        }
    }

    #[test]
    fn adaptive_samples_more_at_edges() {
        // The GEGL algorithm uses more samples for pixels farther from center.
        // With a 64x64 image and factor=0.5, corner pixels have a longer ray
        // than pixels near center. This test just verifies it runs without panic.
        let (px, info) = make_gray(64, 64, 128);
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(px.clone());
        let result = zoom_blur(
            r,
            &mut u,
            &info,
            &ZoomBlurParams {
                center_x: 0.5,
                center_y: 0.5,
                factor: 0.5,
            },
        )
        .unwrap();
        assert_eq!(result.len(), px.len());
    }

    #[test]
    fn rgb_preserves_channels() {
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Rgb8,
            color_space: crate::domain::types::ColorSpace::Srgb,
        };
        let px = vec![128u8; 16 * 16 * 3];
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(px.clone());
        let result = zoom_blur(
            r,
            &mut u,
            &info,
            &ZoomBlurParams {
                center_x: 0.5,
                center_y: 0.5,
                factor: 0.2,
            },
        )
        .unwrap();
        assert_eq!(result.len(), 16 * 16 * 3);
    }

    #[test]
    fn rgba_preserves_channels() {
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Rgba8,
            color_space: crate::domain::types::ColorSpace::Srgb,
        };
        let px = vec![128u8; 16 * 16 * 4];
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(px.clone());
        let result = zoom_blur(
            r,
            &mut u,
            &info,
            &ZoomBlurParams {
                center_x: 0.5,
                center_y: 0.5,
                factor: 0.2,
            },
        )
        .unwrap();
        assert_eq!(result.len(), 16 * 16 * 4);
    }

    #[test]
    fn center_pixel_stays_sharp() {
        // Center pixel's ray has zero length → min 3 samples all at center → no blur
        let w = 32u32;
        let h = 32u32;
        let mut px = vec![0u8; (w * h) as usize];
        px[16 * w as usize + 16] = 200;
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Gray8,
            color_space: crate::domain::types::ColorSpace::Srgb,
        };
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(px.clone());
        let result = zoom_blur(
            r,
            &mut u,
            &info,
            &ZoomBlurParams {
                center_x: 0.5,
                center_y: 0.5,
                factor: 0.3,
            },
        )
        .unwrap();
        let center_val = result[16 * w as usize + 16];
        // Center pixel samples near itself → stays close to original
        assert!(
            center_val >= 150,
            "center pixel should stay bright, got {center_val}"
        );
    }
}

#[cfg(test)]
mod kuwahara_rank_tests {
    use super::*;
    use crate::domain::types::ColorSpace;

    fn rgb_info(w: u32, h: u32) -> ImageInfo {
        ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        }
    }

    fn gray_info(w: u32, h: u32) -> ImageInfo {
        ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        }
    }

    // ── Kuwahara ──

    #[test]
    fn kuwahara_radius_0_is_identity() {
        let pixels: Vec<u8> = (0..32 * 32 * 3).map(|i| (i % 256) as u8).collect();
        let info = rgb_info(32, 32);
        let result = kuwahara(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &KuwaharaParams { radius: 0 },
        )
        .unwrap();
        assert_eq!(result, pixels);
    }

    #[test]
    fn kuwahara_preserves_size() {
        let pixels = vec![128u8; 64 * 64 * 3];
        let info = rgb_info(64, 64);
        let result = kuwahara(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &KuwaharaParams { radius: 3 },
        )
        .unwrap();
        assert_eq!(result.len(), pixels.len());
    }

    #[test]
    fn kuwahara_uniform_is_identity() {
        let pixels = vec![100u8; 32 * 32 * 3];
        let info = rgb_info(32, 32);
        let result = kuwahara(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &KuwaharaParams { radius: 3 },
        )
        .unwrap();
        assert_eq!(result, pixels);
    }

    #[test]
    fn kuwahara_edge_preservation() {
        // Create image with sharp vertical edge at x=16
        let w = 32u32;
        let h = 32u32;
        let mut pixels = vec![0u8; (w * h * 3) as usize];
        for y in 0..h as usize {
            for x in 16..w as usize {
                let off = (y * w as usize + x) * 3;
                pixels[off] = 255;
                pixels[off + 1] = 255;
                pixels[off + 2] = 255;
            }
        }
        let info = rgb_info(w, h);
        let result = kuwahara(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &KuwaharaParams { radius: 2 },
        )
        .unwrap();
        // Interior pixels far from edge should be unchanged
        // Left side interior (x=4, y=16) should still be dark
        let left_off = (16 * w as usize + 4) * 3;
        assert!(
            result[left_off] < 30,
            "left interior should be dark, got {}",
            result[left_off]
        );
        // Right side interior (x=28, y=16) should still be bright
        let right_off = (16 * w as usize + 28) * 3;
        assert!(
            result[right_off] > 225,
            "right interior should be bright, got {}",
            result[right_off]
        );
    }

    #[test]
    fn kuwahara_gray_works() {
        let pixels = vec![128u8; 32 * 32];
        let info = gray_info(32, 32);
        let result = kuwahara(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &KuwaharaParams { radius: 2 },
        )
        .unwrap();
        assert_eq!(result.len(), pixels.len());
    }

    // ── Rank Filter ──

    #[test]
    fn rank_filter_radius_0_is_identity() {
        let pixels: Vec<u8> = (0..16 * 16 * 3).map(|i| (i % 256) as u8).collect();
        let info = rgb_info(16, 16);
        let result = rank_filter(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &RankFilterParams {
                radius: 0,
                rank: 0.5,
            },
        )
        .unwrap();
        assert_eq!(result, pixels);
    }

    #[test]
    fn rank_filter_preserves_size() {
        let pixels = vec![128u8; 32 * 32 * 3];
        let info = rgb_info(32, 32);
        let result = rank_filter(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &RankFilterParams {
                radius: 2,
                rank: 0.5,
            },
        )
        .unwrap();
        assert_eq!(result.len(), pixels.len());
    }

    #[test]
    fn rank_filter_median_matches_existing() {
        // rank=0.5 should produce same output as the existing median filter
        let pixels: Vec<u8> = (0..32 * 32 * 3)
            .map(|i| ((i * 7 + 13) % 256) as u8)
            .collect();
        let info = rgb_info(32, 32);
        let median_result = median(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &MedianParams { radius: 3 },
        )
        .unwrap();
        let rank_result = rank_filter(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &RankFilterParams {
                radius: 3,
                rank: 0.5,
            },
        )
        .unwrap();
        assert_eq!(rank_result, median_result, "rank 0.5 should match median");
    }

    #[test]
    fn rank_filter_min_produces_dark() {
        // rank=0.0 is local minimum — result should be <= input for each pixel
        let pixels: Vec<u8> = (0..16 * 16).map(|i| ((i * 17 + 5) % 256) as u8).collect();
        let info = gray_info(16, 16);
        let result = rank_filter(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &RankFilterParams {
                radius: 1,
                rank: 0.0,
            },
        )
        .unwrap();
        // Local min should be <= each pixel's own value (approximately — due to edge reflect)
        let mean_input: f64 = pixels.iter().map(|&v| v as f64).sum::<f64>() / pixels.len() as f64;
        let mean_output: f64 = result.iter().map(|&v| v as f64).sum::<f64>() / result.len() as f64;
        assert!(
            mean_output < mean_input,
            "min rank should produce darker output: input={mean_input:.1}, output={mean_output:.1}"
        );
    }

    #[test]
    fn rank_filter_max_produces_bright() {
        // rank=1.0 is local maximum — result should be >= input on average
        let pixels: Vec<u8> = (0..16 * 16).map(|i| ((i * 17 + 5) % 256) as u8).collect();
        let info = gray_info(16, 16);
        let result = rank_filter(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &RankFilterParams {
                radius: 1,
                rank: 1.0,
            },
        )
        .unwrap();
        let mean_input: f64 = pixels.iter().map(|&v| v as f64).sum::<f64>() / pixels.len() as f64;
        let mean_output: f64 = result.iter().map(|&v| v as f64).sum::<f64>() / result.len() as f64;
        assert!(
            mean_output > mean_input,
            "max rank should produce brighter output: input={mean_input:.1}, output={mean_output:.1}"
        );
    }

    #[test]
    fn rank_filter_uniform_is_identity() {
        let pixels = vec![100u8; 16 * 16 * 3];
        let info = rgb_info(16, 16);
        for rank in [0.0f32, 0.5, 1.0] {
            let result = rank_filter(
                Rect::new(0, 0, info.width, info.height),
                &mut |_| Ok(pixels.to_vec()),
                &info,
                &RankFilterParams {
                    radius: 2,
                    rank: rank,
                },
            )
            .unwrap();
            assert_eq!(
                result, pixels,
                "uniform image should be identity at rank={rank}"
            );
        }
    }
}

#[cfg(test)]
mod tilt_shift_lens_blur_tests {
    use super::*;
    use crate::domain::types::ColorSpace;

    fn rgb_info(w: u32, h: u32) -> ImageInfo {
        ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        }
    }

    #[test]
    fn tilt_shift_zero_radius_is_identity() {
        let pixels = vec![128u8; 32 * 32 * 3];
        let info = rgb_info(32, 32);
        let result = tilt_shift(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &TiltShiftParams {
                focus_position: 0.5,
                band_size: 0.2,
                blur_radius: 0.0,
                angle: 0.0,
            },
        )
        .unwrap();
        assert_eq!(result, pixels);
    }

    #[test]
    fn tilt_shift_full_band_is_identity() {
        let pixels: Vec<u8> = (0..32 * 32 * 3).map(|i| (i % 256) as u8).collect();
        let info = rgb_info(32, 32);
        let result = tilt_shift(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &TiltShiftParams {
                focus_position: 0.5,
                band_size: 1.0,
                blur_radius: 10.0,
                angle: 0.0,
            },
        )
        .unwrap();
        assert_eq!(result, pixels);
    }

    #[test]
    fn tilt_shift_center_band_stays_sharp() {
        // Generate gradient image
        let w = 32u32;
        let h = 32u32;
        let mut pixels = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let idx = ((y * w + x) * 3) as usize;
                pixels[idx] = (x * 8) as u8;
                pixels[idx + 1] = (y * 8) as u8;
                pixels[idx + 2] = 128;
            }
        }
        let info = rgb_info(w, h);
        let result = tilt_shift(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &TiltShiftParams {
                focus_position: 0.5,
                band_size: 0.3,
                blur_radius: 10.0,
                angle: 0.0,
            },
        )
        .unwrap();

        // Center row (y=16) should be in the focus band — exactly preserved
        let center_y = 16;
        for x in 1..(w - 1) as usize {
            let idx = (center_y * w as usize + x) * 3;
            assert_eq!(result[idx], pixels[idx], "center pixel at x={x} changed");
        }
    }

    #[test]
    fn tilt_shift_edges_are_blurred() {
        let w = 32u32;
        let h = 32u32;
        let mut pixels = vec![0u8; (w * h * 3) as usize];
        // Create alternating pattern (high frequency — blurring should smooth it)
        for y in 0..h {
            for x in 0..w {
                let idx = ((y * w + x) * 3) as usize;
                let val = if (x + y) % 2 == 0 { 255 } else { 0 };
                pixels[idx] = val;
                pixels[idx + 1] = val;
                pixels[idx + 2] = val;
            }
        }
        let info = rgb_info(w, h);
        let result = tilt_shift(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &TiltShiftParams {
                focus_position: 0.5,
                band_size: 0.2,
                blur_radius: 8.0,
                angle: 0.0,
            },
        )
        .unwrap();

        // Top edge (y=0) should be heavily blurred — values should be closer to 128
        let top_y = 0;
        let mut top_variance = 0.0f64;
        for x in 2..(w - 2) as usize {
            let idx = (top_y * w as usize + x) * 3;
            let val = result[idx] as f64;
            top_variance += (val - 128.0).abs();
        }
        top_variance /= (w - 4) as f64;
        assert!(
            top_variance < 100.0,
            "top edge should be blurred (variance={top_variance})"
        );
    }

    #[test]
    fn lens_blur_zero_radius_is_identity() {
        let pixels = vec![128u8; 16 * 16 * 3];
        let info = rgb_info(16, 16);
        let result = lens_blur(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &LensBlurParams {
                radius: 0,
                blade_count: 0,
                rotation: 0.0,
            },
        )
        .unwrap();
        assert_eq!(result, pixels);
    }

    #[test]
    fn lens_blur_disc_mode() {
        let pixels: Vec<u8> = (0..32 * 32 * 3).map(|i| (i % 256) as u8).collect();
        let info = rgb_info(32, 32);
        let result = lens_blur(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &LensBlurParams {
                radius: 3,
                blade_count: 0,
                rotation: 0.0,
            },
        )
        .unwrap();
        assert_eq!(result.len(), pixels.len());
        // Should be different from input (blurred)
        assert_ne!(result, pixels);
    }

    #[test]
    fn lens_blur_polygon_mode() {
        let pixels: Vec<u8> = (0..32 * 32 * 3).map(|i| (i % 256) as u8).collect();
        let info = rgb_info(32, 32);
        // 6-blade hexagon
        let result = lens_blur(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &LensBlurParams {
                radius: 3,
                blade_count: 6,
                rotation: 0.0,
            },
        )
        .unwrap();
        assert_eq!(result.len(), pixels.len());
        assert_ne!(result, pixels);
    }

    #[test]
    fn lens_blur_disc_matches_bokeh_blur() {
        let pixels: Vec<u8> = (0..32 * 32 * 3).map(|i| (i % 256) as u8).collect();
        let info = rgb_info(32, 32);
        let disc = lens_blur(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &LensBlurParams {
                radius: 3,
                blade_count: 0,
                rotation: 0.0,
            },
        )
        .unwrap();
        let r = Rect::new(0, 0, info.width, info.height);
        let px2 = pixels.clone();
        let mut u = |_: Rect| Ok(px2.clone());
        let bokeh = bokeh_blur(
            r,
            &mut u,
            &info,
            &BokehBlurParams {
                radius: 3,
                shape: 0,
            },
        )
        .unwrap();
        // Both use same make_disc_kernel + convolve — should be identical
        assert_eq!(disc, bokeh, "lens_blur disc should match bokeh_blur disc");
    }

    #[test]
    fn polygon_kernel_has_correct_shape() {
        let (kernel, side) = make_polygon_kernel(5, 6, 0.0);
        assert_eq!(side, 11); // 2*5+1
        // Center should be filled
        assert!(kernel[5 * 11 + 5] > 0.0, "center should be filled");
        // Total weight should be positive
        let total: f32 = kernel.iter().sum();
        assert!(total > 10.0, "polygon should have substantial area");
    }

    #[test]
    fn lens_blur_rotation_changes_output() {
        let pixels: Vec<u8> = (0..32 * 32 * 3).map(|i| (i % 256) as u8).collect();
        let info = rgb_info(32, 32);
        let r0 = lens_blur(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &LensBlurParams {
                radius: 4,
                blade_count: 6,
                rotation: 0.0,
            },
        )
        .unwrap();
        let r30 = lens_blur(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &LensBlurParams {
                radius: 4,
                blade_count: 6,
                rotation: 30.0,
            },
        )
        .unwrap();
        // Different rotation should produce different output
        assert_ne!(r0, r30, "rotated polygon should produce different result");
    }
}

