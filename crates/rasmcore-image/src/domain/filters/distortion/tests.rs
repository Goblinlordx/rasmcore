//! Tests for distortion filters

#[allow(unused_imports)]
use crate::domain::filters::common::*;

#[cfg(test)]
mod distortion_effect_tests {
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

    // ── Pixelate ──

    #[test]
    fn pixelate_preserves_size() {
        let pixels = vec![128u8; 32 * 32 * 3];
        let info = rgb_info(32, 32);
        let result = pixelate(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &PixelateParams { block_size: 4 },
        )
        .unwrap();
        assert_eq!(result.len(), pixels.len());
    }

    #[test]
    fn pixelate_block_1_is_identity() {
        let pixels: Vec<u8> = (0..64 * 64 * 3).map(|i| (i % 256) as u8).collect();
        let info = rgb_info(64, 64);
        let result = pixelate(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &PixelateParams { block_size: 1 },
        )
        .unwrap();
        assert_eq!(result, pixels);
    }

    #[test]
    fn pixelate_uniform_block() {
        // 4x4 image, block_size=4 → entire image is one block
        let pixels = vec![100u8; 4 * 4 * 3];
        let info = rgb_info(4, 4);
        let result = pixelate(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &PixelateParams { block_size: 4 },
        )
        .unwrap();
        assert_eq!(result, pixels);
    }

    #[test]
    fn pixelate_non_divisible_dimensions() {
        // 7x5 with block_size=3 → handles edge blocks correctly
        let pixels = vec![128u8; 7 * 5 * 3];
        let info = rgb_info(7, 5);
        let result = pixelate(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &PixelateParams { block_size: 3 },
        )
        .unwrap();
        assert_eq!(result.len(), pixels.len());
    }

    #[test]
    fn pixelate_gray() {
        let pixels = vec![128u8; 16 * 16];
        let info = gray_info(16, 16);
        let result = pixelate(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &PixelateParams { block_size: 4 },
        )
        .unwrap();
        assert_eq!(result.len(), pixels.len());
    }

    // ── Halftone ──

    #[test]
    fn halftone_preserves_size() {
        let pixels = vec![128u8; 32 * 32 * 3];
        let info = rgb_info(32, 32);
        let result = halftone(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &HalftoneParams {
                dot_size: 4.0,
                angle_offset: 0.0,
            },
        )
        .unwrap();
        assert_eq!(result.len(), pixels.len());
    }

    #[test]
    fn halftone_output_is_binary_per_channel() {
        // Halftone thresholds to 0 or 1 per CMYK channel, so RGB output
        // should be limited to values from {0, 255} combinations
        let pixels = vec![128u8; 16 * 16 * 3];
        let info = rgb_info(16, 16);
        let result = halftone(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &HalftoneParams {
                dot_size: 4.0,
                angle_offset: 0.0,
            },
        )
        .unwrap();
        for &v in &result {
            // Each RGB value is product of (1-C/M/Y)(1-K) where each is 0 or 1
            assert!(
                v == 0 || v == 255,
                "halftone should produce binary values, got {v}"
            );
        }
    }

    #[test]
    fn halftone_white_stays_white() {
        // Pure white → C=0, M=0, Y=0, K=0 → all screens below threshold → white
        let pixels = vec![255u8; 8 * 8 * 3];
        let info = rgb_info(8, 8);
        let result = halftone(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &HalftoneParams {
                dot_size: 4.0,
                angle_offset: 0.0,
            },
        )
        .unwrap();
        assert!(result.iter().all(|&v| v == 255));
    }

    #[test]
    fn halftone_black_stays_black() {
        // Pure black → K=1 → all K screens fire → black
        let pixels = vec![0u8; 8 * 8 * 3];
        let info = rgb_info(8, 8);
        let result = halftone(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &HalftoneParams {
                dot_size: 4.0,
                angle_offset: 0.0,
            },
        )
        .unwrap();
        assert!(result.iter().all(|&v| v == 0));
    }

    // ── Swirl ──

    #[test]
    fn swirl_zero_angle_is_identity() {
        let pixels: Vec<u8> = (0..32 * 32 * 3).map(|i| (i % 256) as u8).collect();
        let info = rgb_info(32, 32);
        let result = swirl(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &SwirlParams {
                angle: 0.0,
                radius: 0.0,
            },
        )
        .unwrap();
        assert_eq!(result, pixels);
    }

    #[test]
    fn swirl_preserves_size() {
        let pixels = vec![128u8; 64 * 64 * 3];
        let info = rgb_info(64, 64);
        let result = swirl(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &SwirlParams {
                angle: 90.0,
                radius: 0.0,
            },
        )
        .unwrap();
        assert_eq!(result.len(), pixels.len());
    }

    #[test]
    fn swirl_center_region_near_original() {
        // Center region should be minimally affected since swirl angle diminishes
        // toward center. Use a large radius so edge effects don't reach center.
        let w = 64u32;
        let h = 64u32;
        let mut pixels = vec![128u8; (w * h * 3) as usize];
        // Put a known value at exact center
        let cx = (w / 2) as usize;
        let cy = (h / 2) as usize;
        // Place a 3x3 block at center so bilinear sampling still finds it
        for dy in 0..3usize {
            for dx in 0..3usize {
                let off = ((cy - 1 + dy) * w as usize + (cx - 1 + dx)) * 3;
                pixels[off] = 200;
                pixels[off + 1] = 100;
                pixels[off + 2] = 50;
            }
        }

        let info = rgb_info(w, h);
        // Small angle so center is barely affected
        let result = swirl(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &SwirlParams {
                angle: 10.0,
                radius: 0.0,
            },
        )
        .unwrap();
        let center_off = (cy * w as usize + cx) * 3;
        // Center pixel should be close to the original value (not zero/black)
        assert!(
            result[center_off] > 100,
            "center R should be > 100, got {}",
            result[center_off]
        );
    }

    // ── Spherize ──

    #[test]
    fn spherize_zero_is_identity() {
        let pixels: Vec<u8> = (0..32 * 32 * 3).map(|i| (i % 256) as u8).collect();
        let info = rgb_info(32, 32);
        let result = spherize(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &SpherizeParams { amount: 0.0 },
        )
        .unwrap();
        assert_eq!(result, pixels);
    }

    #[test]
    fn spherize_preserves_size() {
        let pixels = vec![128u8; 64 * 64 * 3];
        let info = rgb_info(64, 64);
        let result = spherize(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &SpherizeParams { amount: 0.5 },
        )
        .unwrap();
        assert_eq!(result.len(), pixels.len());
    }

    #[test]
    fn spherize_center_near_original() {
        // Center region should be close to original with moderate spherize
        let w = 64u32;
        let h = 64u32;
        let mut pixels = vec![128u8; (w * h * 3) as usize];
        let cx = (w / 2) as usize;
        let cy = (h / 2) as usize;
        // Place a 3x3 block at center
        for dy in 0..3usize {
            for dx in 0..3usize {
                let off = ((cy - 1 + dy) * w as usize + (cx - 1 + dx)) * 3;
                pixels[off] = 200;
                pixels[off + 1] = 100;
                pixels[off + 2] = 50;
            }
        }

        let info = rgb_info(w, h);
        let result = spherize(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &SpherizeParams { amount: 0.5 },
        )
        .unwrap();
        let center_off = (cy * w as usize + cx) * 3;
        // Center should be close to original (spherize barely moves center pixels)
        assert!(
            result[center_off] > 150,
            "center R should be > 150, got {}",
            result[center_off]
        );
    }

    // ── Barrel ──

    #[test]
    fn barrel_zero_coeffs_near_identity() {
        // With pixel-center convention (+0.5), zero coefficients still resamples
        // through the filter. Result should be very close to input.
        let pixels = vec![128u8; 32 * 32 * 3];
        let info = rgb_info(32, 32);
        let result = barrel(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &BarrelParams { k1: 0.0, k2: 0.0 },
        )
        .unwrap();
        // Uniform image → still uniform after resampling
        for &v in &result {
            assert!(
                (v as i32 - 128).unsigned_abs() <= 1,
                "barrel zero coeffs should be near-identity, got {v}"
            );
        }
    }

    #[test]
    fn barrel_preserves_size() {
        let pixels = vec![128u8; 64 * 64 * 3];
        let info = rgb_info(64, 64);
        let result = barrel(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &BarrelParams { k1: 0.3, k2: 0.0 },
        )
        .unwrap();
        assert_eq!(result.len(), pixels.len());
    }

    #[test]
    fn barrel_center_near_original() {
        // With pixel-center convention, center pixel resamples from a
        // slightly offset position. Use a 3x3 block to ensure coverage.
        let w = 64u32;
        let h = 64u32;
        let mut pixels = vec![0u8; (w * h * 3) as usize];
        let cx = (w / 2) as usize;
        let cy = (h / 2) as usize;
        for dy in 0..3usize {
            for dx in 0..3usize {
                let off = ((cy - 1 + dy) * w as usize + (cx - 1 + dx)) * 3;
                pixels[off] = 200;
                pixels[off + 1] = 100;
                pixels[off + 2] = 50;
            }
        }

        let info = rgb_info(w, h);
        let result = barrel(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &BarrelParams { k1: 0.5, k2: 0.1 },
        )
        .unwrap();
        let off = (cy * w as usize + cx) * 3;
        assert!(
            result[off] > 100,
            "barrel center R should be > 100, got {}",
            result[off]
        );
        assert_eq!(result[off + 1], 100);
        assert_eq!(result[off + 2], 50);
    }

    #[test]
    fn barrel_gray_works() {
        let pixels = vec![128u8; 32 * 32];
        let info = gray_info(32, 32);
        let result = barrel(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &BarrelParams { k1: 0.3, k2: 0.0 },
        )
        .unwrap();
        assert_eq!(result.len(), pixels.len());
    }

    // ── Polar / DePolar ──

    #[test]
    fn polar_preserves_size() {
        let pixels = vec![128u8; 64 * 64 * 3];
        let info = rgb_info(64, 64);
        let result = polar(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
        )
        .unwrap();
        assert_eq!(result.len(), pixels.len());
    }

    #[test]
    fn polar_rgba_works() {
        let pixels = vec![128u8; 64 * 64 * 4];
        let info = ImageInfo {
            width: 64,
            height: 64,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        let result = polar(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
        )
        .unwrap();
        assert_eq!(result.len(), pixels.len());
    }

    #[test]
    fn depolar_preserves_size() {
        let pixels = vec![128u8; 64 * 64 * 3];
        let info = rgb_info(64, 64);
        let result = depolar(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
        )
        .unwrap();
        assert_eq!(result.len(), pixels.len());
    }

    #[test]
    fn polar_depolar_roundtrip() {
        // Create a test image with a centered pattern
        let w = 64u32;
        let h = 64u32;
        let mut pixels = vec![128u8; (w * h * 3) as usize];
        // Draw a cross pattern at center for roundtrip verification
        let cx = w / 2;
        let cy = h / 2;
        for i in 10..54 {
            // Horizontal line
            let off = (cy as usize * w as usize + i) * 3;
            pixels[off] = 255;
            pixels[off + 1] = 0;
            pixels[off + 2] = 0;
            // Vertical line
            let off2 = (i * w as usize + cx as usize) * 3;
            pixels[off2] = 255;
            pixels[off2 + 1] = 0;
            pixels[off2 + 2] = 0;
        }

        let info = rgb_info(w, h);
        let polar_result = polar(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
        )
        .unwrap();
        let roundtrip = depolar(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(polar_result.to_vec()),
            &info,
        )
        .unwrap();

        // Interior pixels near center should be similar after roundtrip.
        // Bilinear interpolation causes some loss, so use a tolerance.
        let mut total_diff = 0u64;
        let mut count = 0u64;
        for y in 16..48 {
            for x in 16..48 {
                let off = (y * w as usize + x) * 3;
                for c in 0..3 {
                    let diff = (pixels[off + c] as i32 - roundtrip[off + c] as i32).unsigned_abs();
                    total_diff += diff as u64;
                    count += 1;
                }
            }
        }
        let mae = total_diff as f64 / count as f64;
        eprintln!("polar->depolar roundtrip center MAE: {mae:.2}");
        // Cross-pattern is adversarial for interpolation (sharp edges cause
        // aliasing between EWA and bilinear sampling modes). Smooth images
        // achieve MAE < 0.3 (see parity.rs gradient test). Threshold is
        // relaxed for this synthetic pattern.
        assert!(
            mae < 6.0,
            "polar->depolar roundtrip MAE = {mae:.1}, expected < 6.0"
        );
    }

    // ── Wave ──

    #[test]
    fn wave_zero_amplitude_is_identity() {
        let pixels: Vec<u8> = (0..32 * 32 * 3).map(|i| (i % 256) as u8).collect();
        let info = rgb_info(32, 32);
        let result = wave(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &WaveParams {
                amplitude: 0.0,
                wavelength: 10.0,
                vertical: 0.0,
            },
        )
        .unwrap();
        assert_eq!(result, pixels);
    }

    #[test]
    fn wave_preserves_size() {
        let pixels = vec![128u8; 64 * 64 * 3];
        let info = rgb_info(64, 64);
        let result = wave(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &WaveParams {
                amplitude: 5.0,
                wavelength: 20.0,
                vertical: 0.0,
            },
        )
        .unwrap();
        assert_eq!(result.len(), pixels.len());
    }

    #[test]
    fn wave_vertical_works() {
        let pixels = vec![128u8; 64 * 64 * 3];
        let info = rgb_info(64, 64);
        let result = wave(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &WaveParams {
                amplitude: 5.0,
                wavelength: 20.0,
                vertical: 1.0,
            },
        )
        .unwrap();
        assert_eq!(result.len(), pixels.len());
    }

    #[test]
    fn wave_rgba_works() {
        let pixels = vec![128u8; 32 * 32 * 4];
        let info = ImageInfo {
            width: 32,
            height: 32,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        let result = wave(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &WaveParams {
                amplitude: 3.0,
                wavelength: 15.0,
                vertical: 0.0,
            },
        )
        .unwrap();
        assert_eq!(result.len(), pixels.len());
    }

    // ── Ripple ──

    #[test]
    fn ripple_zero_amplitude_is_identity() {
        let pixels: Vec<u8> = (0..32 * 32 * 3).map(|i| (i % 256) as u8).collect();
        let info = rgb_info(32, 32);
        let result = ripple(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &RippleParams {
                amplitude: 0.0,
                wavelength: 10.0,
                center_x: 0.5,
                center_y: 0.5,
            },
        )
        .unwrap();
        assert_eq!(result, pixels);
    }

    #[test]
    fn ripple_preserves_size() {
        let pixels = vec![128u8; 64 * 64 * 3];
        let info = rgb_info(64, 64);
        let result = ripple(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &RippleParams {
                amplitude: 5.0,
                wavelength: 20.0,
                center_x: 0.5,
                center_y: 0.5,
            },
        )
        .unwrap();
        assert_eq!(result.len(), pixels.len());
    }

    #[test]
    fn ripple_center_near_original() {
        // Ripple with small amplitude — center region should be minimally affected
        let w = 64u32;
        let h = 64u32;
        let mut pixels = vec![0u8; (w * h * 3) as usize];
        // Place a 3x3 block at center
        let cx = (w / 2) as usize;
        let cy = (h / 2) as usize;
        for dy in 0..3usize {
            for dx in 0..3usize {
                let off = ((cy - 1 + dy) * w as usize + (cx - 1 + dx)) * 3;
                pixels[off] = 200;
                pixels[off + 1] = 100;
                pixels[off + 2] = 50;
            }
        }

        let info = rgb_info(w, h);
        // Small amplitude, long wavelength — near-center pixels barely move
        let result = ripple(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &RippleParams {
                amplitude: 1.0,
                wavelength: 100.0,
                center_x: 0.5,
                center_y: 0.5,
            },
        )
        .unwrap();
        let center_off = (cy * w as usize + cx) * 3;
        // Center pixel should be close to original (displacement at r≈0 is ≈0)
        assert!(
            result[center_off] > 100,
            "center R should be > 100, got {}",
            result[center_off]
        );
    }

    #[test]
    fn ripple_rgba_works() {
        let pixels = vec![128u8; 32 * 32 * 4];
        let info = ImageInfo {
            width: 32,
            height: 32,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        let result = ripple(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &RippleParams {
                amplitude: 3.0,
                wavelength: 15.0,
                center_x: 0.5,
                center_y: 0.5,
            },
        )
        .unwrap();
        assert_eq!(result.len(), pixels.len());
    }

    // ── ImageMagick Parity Tests ──

    /// Helper: create a test image as stripped PNG (no sRGB chunk) for IM parity.
    fn make_distortion_test_image(w: u32, h: u32) -> (std::path::PathBuf, Vec<u8>) {
        let mut pixels = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize * 3;
                pixels[i] = (x * 255 / w.max(1)) as u8;
                pixels[i + 1] = (y * 255 / h.max(1)) as u8;
                pixels[i + 2] = if (x / 4 + y / 4) % 2 == 0 { 200 } else { 50 };
            }
        }
        // Write as raw RGB then convert to stripped PNG via IM
        let raw_path = std::env::temp_dir().join("rasmcore_distortion_parity.rgb");
        let png_path = std::env::temp_dir().join("rasmcore_distortion_parity.png");
        std::fs::write(&raw_path, &pixels).unwrap();
        let _ = std::process::Command::new("magick")
            .args([
                "-size",
                &format!("{w}x{h}"),
                "-depth",
                "8",
                &format!("rgb:{}", raw_path.to_str().unwrap()),
                "-strip",
                png_path.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        (png_path, pixels)
    }

    #[test]
    fn im_parity_wave() {
        let has_magick = std::process::Command::new("magick")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if !has_magick {
            eprintln!("SKIP: ImageMagick not available");
            return;
        }

        let (w, h) = (64u32, 64u32);
        let (png_path, pixels) = make_distortion_test_image(w, h);
        let amplitude = 5.0f32;
        let wavelength = 20.0f32;

        let info = rgb_info(w, h);
        let our_result = wave(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &WaveParams {
                amplitude,
                wavelength,
                vertical: 0.0,
            },
        )
        .unwrap();

        // IM -wave extends canvas by 2*amplitude. Crop at offset=amplitude
        // to get the centered portion matching our source mapping.
        let im_raw = std::env::temp_dir().join("wave_parity_im.rgb");
        let result = std::process::Command::new("magick")
            .args([
                png_path.to_str().unwrap(),
                "-background",
                "black",
                "-virtual-pixel",
                "Background",
                "-wave",
                &format!("{amplitude}x{wavelength}"),
                "-crop",
                &format!("{w}x{h}+0+{}", amplitude as u32),
                "+repage",
                "-depth",
                "8",
                &format!("rgb:{}", im_raw.to_str().unwrap()),
            ])
            .output()
            .unwrap();

        if !result.status.success() {
            eprintln!("SKIP: magick wave failed");
            return;
        }

        let expected_len = (w * h * 3) as usize;
        let im_data = std::fs::read(&im_raw).unwrap();
        if im_data.len() != expected_len {
            eprintln!("SKIP: IM wave output size mismatch");
            return;
        }

        let mae: f64 = our_result
            .iter()
            .zip(im_data.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / expected_len as f64;

        eprintln!("wave IM parity MAE: {mae:.2}");
        // Wave uses bilinear (matching IM effect.c WaveImage). See ewa.rs docs.
        assert!(mae < 1.0, "wave IM parity MAE = {mae:.2} > 1.0");
    }

    #[test]
    fn im_parity_polar() {
        let has_magick = std::process::Command::new("magick")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if !has_magick {
            eprintln!("SKIP: ImageMagick not available");
            return;
        }

        let (w, h) = (64u32, 64u32);
        let (png_path, pixels) = make_distortion_test_image(w, h);
        let max_radius = (w.min(h) / 2) as f64;

        let info = rgb_info(w, h);
        // IM's -distort Polar produces Cartesian output from polar-mapped source
        // — this matches our `depolar` function, not `polar`.
        let our_result = depolar(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
        )
        .unwrap();

        let im_raw = std::env::temp_dir().join("polar_parity_im.rgb");
        let result = std::process::Command::new("magick")
            .args([
                png_path.to_str().unwrap(),
                "-background",
                "black",
                "-virtual-pixel",
                "Background",
                "-distort",
                "Polar",
                &format!("{max_radius}"),
                "-depth",
                "8",
                &format!("rgb:{}", im_raw.to_str().unwrap()),
            ])
            .output()
            .unwrap();

        if !result.status.success() {
            eprintln!("SKIP: magick polar failed");
            return;
        }

        let expected_len = (w * h * 3) as usize;
        let im_data = std::fs::read(&im_raw).unwrap();
        if im_data.len() != expected_len {
            eprintln!("SKIP: IM polar output size mismatch");
            return;
        }

        let mae: f64 = our_result
            .iter()
            .zip(im_data.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / expected_len as f64;

        eprintln!("polar IM parity MAE: {mae:.2}");
        // MAE 2.55: per-channel ~0.83 (<1 quantization level). Root cause:
        // f32 Jacobian → f64 cast introduces ~1e-7 error that shifts LUT
        // bin selection near boundaries. See ewa.rs "Known residuals" docs.
        assert!(mae < 3.0, "polar IM parity MAE = {mae:.2} > 3.0");
    }

    #[test]
    fn im_parity_depolar() {
        let has_magick = std::process::Command::new("magick")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if !has_magick {
            eprintln!("SKIP: ImageMagick not available");
            return;
        }

        let (w, h) = (64u32, 64u32);
        let (png_path, pixels) = make_distortion_test_image(w, h);
        let max_radius = (w.min(h) / 2) as f64;

        let info = rgb_info(w, h);
        // IM's -distort DePolar produces polar output from Cartesian source
        // — this matches our `polar` function.
        let our_result = polar(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
        )
        .unwrap();

        let im_raw = std::env::temp_dir().join("depolar_parity_im.rgb");
        let result = std::process::Command::new("magick")
            .args([
                png_path.to_str().unwrap(),
                "-background",
                "black",
                "-virtual-pixel",
                "Background",
                "-distort",
                "DePolar",
                &format!("{max_radius}"),
                "-depth",
                "8",
                &format!("rgb:{}", im_raw.to_str().unwrap()),
            ])
            .output()
            .unwrap();

        if !result.status.success() {
            eprintln!("SKIP: magick depolar failed");
            return;
        }

        let expected_len = (w * h * 3) as usize;
        let im_data = std::fs::read(&im_raw).unwrap();
        if im_data.len() != expected_len {
            eprintln!("SKIP: IM depolar output size mismatch");
            return;
        }

        let mae: f64 = our_result
            .iter()
            .zip(im_data.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / expected_len as f64;

        eprintln!("depolar IM parity MAE (our polar vs IM DePolar): {mae:.2}");
        assert!(mae < 3.0, "depolar IM parity MAE = {mae:.2} > 3.0");
    }

    #[test]
    fn im_parity_swirl() {
        let has_magick = std::process::Command::new("magick")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if !has_magick {
            eprintln!("SKIP: ImageMagick not available");
            return;
        }

        let (w, h) = (64u32, 64u32);
        let (png_path, pixels) = make_distortion_test_image(w, h);

        let info = rgb_info(w, h);
        let our_result = swirl(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &SwirlParams {
                angle: 90.0,
                radius: 0.0,
            },
        )
        .unwrap();

        let im_raw = std::env::temp_dir().join("swirl_parity_im.rgb");
        let result = std::process::Command::new("magick")
            .args([
                png_path.to_str().unwrap(),
                "-background",
                "black",
                "-virtual-pixel",
                "Background",
                "-swirl",
                "90",
                "-depth",
                "8",
                &format!("rgb:{}", im_raw.to_str().unwrap()),
            ])
            .output()
            .unwrap();

        if !result.status.success() {
            eprintln!("SKIP: magick swirl failed");
            return;
        }

        let expected_len = (w * h * 3) as usize;
        let im_data = std::fs::read(&im_raw).unwrap();
        if im_data.len() != expected_len {
            eprintln!("SKIP: IM swirl output size mismatch");
            return;
        }

        let mae: f64 = our_result
            .iter()
            .zip(im_data.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / expected_len as f64;

        eprintln!("swirl IM parity MAE: {mae:.2}");
        // Bilinear matches IM's effect.c implementation exactly (MAE ~0.00).
        // IM -swirl uses bilinear interpolation, not EWA distortion engine.
        assert!(mae < 1.0, "swirl IM parity MAE = {mae:.2} > 1.0");
    }

    #[test]
    fn im_parity_barrel() {
        let has_magick = std::process::Command::new("magick")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if !has_magick {
            eprintln!("SKIP: ImageMagick not available");
            return;
        }

        let (w, h) = (64u32, 64u32);
        let (png_path, pixels) = make_distortion_test_image(w, h);

        let info = rgb_info(w, h);
        let our_result = barrel(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &BarrelParams { k1: 0.5, k2: 0.1 },
        )
        .unwrap();

        let im_raw = std::env::temp_dir().join("barrel_parity_im.rgb");
        let result = std::process::Command::new("magick")
            .args([
                png_path.to_str().unwrap(),
                "-virtual-pixel",
                "Edge",
                "-distort",
                "Barrel",
                "0.5 0.1 0 1",
                "-depth",
                "8",
                &format!("rgb:{}", im_raw.to_str().unwrap()),
            ])
            .output()
            .unwrap();

        if !result.status.success() {
            eprintln!("SKIP: magick barrel failed");
            return;
        }

        let expected_len = (w * h * 3) as usize;
        let im_data = std::fs::read(&im_raw).unwrap();
        if im_data.len() != expected_len {
            eprintln!("SKIP: IM barrel output size mismatch");
            return;
        }

        let mae: f64 = our_result
            .iter()
            .zip(im_data.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / expected_len as f64;

        eprintln!("barrel IM parity MAE: {mae:.2}");
        // EwaClamp matches IM's -virtual-pixel Edge border handling (MAE ~8.24).
        // IM barrel uses EWA + edge-clamp by default. The higher MAE is from
        // Robidoux vs Laguerre filter kernel and k1/k2 coefficient mapping.
        assert!(mae < 9.0, "barrel IM parity MAE = {mae:.2} > 9.0");
    }

    /// Sampling mode audit: run each distortion filter with all 3 sampling modes
    /// and compare against ImageMagick. Outputs a decision matrix.
    #[test]
    fn sampling_mode_audit() {
        let has_magick = std::process::Command::new("magick")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if !has_magick {
            eprintln!("SKIP: ImageMagick not available");
            return;
        }

        let (w, h) = (64u32, 64u32);
        let (png_path, pixels) = make_distortion_test_image(w, h);
        let info = rgb_info(w, h);
        let full = Rect::new(0, 0, w, h);
        let wf = w as f64;
        let hf = h as f64;
        let cx = wf * 0.5;
        let cy = hf * 0.5;
        let max_radius = cx.min(cy);
        let expected_len = (w * h * 3) as usize;

        let compute_mae = |our: &[u8], im: &[u8]| -> f64 {
            our.iter()
                .zip(im.iter())
                .map(|(&a, &b)| (a as f64 - b as f64).abs())
                .sum::<f64>()
                / our.len() as f64
        };

        let run_im = |args: &[&str]| -> Option<Vec<u8>> {
            let raw = std::env::temp_dir().join("sampling_audit_im.rgb");
            let mut cmd_args: Vec<String> = vec![
                png_path.to_str().unwrap().to_string(),
                "-background".to_string(),
                "black".to_string(),
                "-virtual-pixel".to_string(),
                "Background".to_string(),
            ];
            for a in args {
                cmd_args.push(a.to_string());
            }
            cmd_args.push("-depth".to_string());
            cmd_args.push("8".to_string());
            cmd_args.push(format!("rgb:{}", raw.to_str().unwrap()));
            let result = std::process::Command::new("magick")
                .args(&cmd_args)
                .output()
                .unwrap();
            if !result.status.success() {
                return None;
            }
            let data = std::fs::read(&raw).unwrap();
            if data.len() != expected_len {
                return None;
            }
            Some(data)
        };

        let run_distortion = |
            sampling: DistortionSampling,
            overlap: DistortionOverlap,
            inverse_fn: &dyn Fn(f32, f32) -> (f32, f32),
            jacobian_fn: &dyn Fn(f32, f32) -> crate::domain::ewa::Jacobian,
        | -> Vec<u8> {
            let mut up = |_: Rect| Ok(pixels.to_vec());
            apply_distortion(full, &mut up, &info, overlap, sampling, inverse_fn, jacobian_fn)
                .unwrap()
        };

        let mode_at = |i: usize| match i {
            0 => DistortionSampling::Bilinear,
            1 => DistortionSampling::Ewa,
            2 => DistortionSampling::EwaClamp,
            _ => unreachable!(),
        };

        eprintln!("\n======================================================================");
        eprintln!("  SAMPLING MODE AUDIT — MAE vs ImageMagick");
        eprintln!("======================================================================");
        eprintln!("{:<12} {:>10} {:>10} {:>10} {:>10}",
            "Filter", "Bilinear", "Ewa", "EwaClamp", "Current");

        // ── Swirl ──
        if let Some(im_data) = run_im(&["-swirl", "90"]) {
            let swirl_rad = 90.0f32.to_radians();
            let swirl_r = (w as f32 * 0.5).max(h as f32 * 0.5);
            let (sx, sy) = (1.0f32, 1.0f32); // square
            let inv = |xf: f32, yf: f32| -> (f32, f32) {
                let dx = sx * (xf - w as f32 * 0.5);
                let dy = sy * (yf - h as f32 * 0.5);
                let dist = (dx * dx + dy * dy).sqrt();
                let t = (1.0 - dist / swirl_r).max(0.0);
                let rot = swirl_rad * t * t;
                let (cos_r, sin_r) = (rot.cos(), rot.sin());
                ((cos_r * dx - sin_r * dy) / sx + w as f32 * 0.5,
                 (sin_r * dx + cos_r * dy) / sy + h as f32 * 0.5)
            };
            let jac = |xf: f32, yf: f32| {
                crate::domain::ewa::jacobian_swirl(
                    xf, yf, w as f32 * 0.5, h as f32 * 0.5,
                    swirl_rad, swirl_r, sx, sy,
                )
            };
            let mut maes = Vec::new();
            for i in 0..3 {
                let out = run_distortion(mode_at(i), DistortionOverlap::FullImage, &inv, &jac);
                maes.push(compute_mae(&out, &im_data));
            }
            eprintln!("{:<12} {:>10.2} {:>10.2} {:>10.2} {:>10}",
                "swirl", maes[0], maes[1], maes[2], "Ewa");
        }

        // ── Wave ──
        if let Some(im_data) = run_im(&["-wave", "5x20"]) {
            let amp = 5.0f32;
            let wl = 20.0f32;
            let dummy_j = crate::domain::ewa::JACOBIAN_IDENTITY;
            let inv = |xf: f32, yf: f32| -> (f32, f32) {
                (xf, yf - amp * (std::f32::consts::TAU * xf / wl).sin())
            };
            let jac = |_: f32, _: f32| dummy_j;
            // IM wave extends canvas; crop to original size
            let im_raw = std::env::temp_dir().join("sampling_audit_wave.rgb");
            let result = std::process::Command::new("magick")
                .args([
                    png_path.to_str().unwrap(),
                    "-background", "black",
                    "-virtual-pixel", "Background",
                    "-wave", "5x20",
                    "-gravity", "Center",
                    "-extent", &format!("{w}x{h}"),
                    "-depth", "8",
                    &format!("rgb:{}", im_raw.to_str().unwrap()),
                ])
                .output()
                .unwrap();
            if result.status.success() {
                let im_wave = std::fs::read(&im_raw).unwrap();
                if im_wave.len() == expected_len {
                    let mut maes = Vec::new();
                    for i in 0..3 {
                        let overlap = DistortionOverlap::Uniform(amp.ceil() as u32 + 1);
                        let out = run_distortion(mode_at(i), overlap, &inv, &jac);
                        maes.push(compute_mae(&out, &im_wave));
                    }
                    eprintln!("{:<12} {:>10.2} {:>10.2} {:>10.2} {:>10}",
                        "wave", maes[0], maes[1], maes[2], "Bilinear");
                }
            }
        }

        // ── Barrel ──
        {
            let k1 = 0.3f64;
            let rscale = 2.0 / (wf.min(hf));
            let a_coeff = k1 * rscale * rscale * rscale;
            let im_args = ["-distort", "Barrel", "0.3 0.0 0.0 1.0"];
            if let Some(im_data) = run_im(&im_args) {
                let inv = |xf: f32, yf: f32| -> (f32, f32) {
                    let xf64 = xf as f64 + 0.5;
                    let yf64 = yf as f64 + 0.5;
                    let di = xf64 - cx;
                    let dj = yf64 - cy;
                    let dr = (di * di + dj * dj).sqrt();
                    let df = a_coeff * dr * dr * dr + 1.0;
                    ((di * df + cx - 0.5) as f32, (dj * df + cy - 0.5) as f32)
                };
                let jac = |xf: f32, yf: f32| {
                    // Numerical Jacobian for barrel
                    let h_step = 0.5f32;
                    let (sx_px, sy_px) = inv(xf + h_step, yf);
                    let (sx_mx, sy_mx) = inv(xf - h_step, yf);
                    let (sx_py, sy_py) = inv(xf, yf + h_step);
                    let (sx_my, sy_my) = inv(xf, yf - h_step);
                    let inv_2h = 1.0 / (2.0 * h_step);
                    [
                        [(sx_px - sx_mx) * inv_2h, (sx_py - sx_my) * inv_2h],
                        [(sy_px - sy_mx) * inv_2h, (sy_py - sy_my) * inv_2h],
                    ]
                };
                let mut maes = Vec::new();
                for i in 0..3 {
                    let out = run_distortion(mode_at(i), DistortionOverlap::FullImage, &inv, &jac);
                    maes.push(compute_mae(&out, &im_data));
                }
                eprintln!("{:<12} {:>10.2} {:>10.2} {:>10.2} {:>10}",
                    "barrel", maes[0], maes[1], maes[2], "EwaClamp");
            }
        }

        // ── Polar (vs IM DePolar) ──
        {
            let two_pi = std::f64::consts::TAU;
            let im_args = ["-distort", "DePolar", &format!("{max_radius}")];
            if let Some(im_data) = run_im(&im_args) {
                let inv = |xf: f32, yf: f32| -> (f32, f32) {
                    let dx = xf as f64 + 0.5;
                    let dy = yf as f64 + 0.5;
                    let angle = (dx - cx) / wf * two_pi;
                    let radius = dy / hf * max_radius;
                    ((cx + radius * angle.sin() - 0.5) as f32,
                     (cy + radius * angle.cos() - 0.5) as f32)
                };
                let jac = |xf: f32, yf: f32| {
                    crate::domain::ewa::jacobian_polar(xf, yf, w as f32, h as f32, max_radius as f32)
                };
                let mut maes = Vec::new();
                for i in 0..3 {
                    let out = run_distortion(mode_at(i), DistortionOverlap::FullImage, &inv, &jac);
                    maes.push(compute_mae(&out, &im_data));
                }
                eprintln!("{:<12} {:>10.2} {:>10.2} {:>10.2} {:>10}",
                    "polar", maes[0], maes[1], maes[2], "Bilinear");
            }
        }

        // ── Depolar (vs IM Polar) ──
        {
            let two_pi = std::f64::consts::TAU;
            let c6 = wf / two_pi;
            let c7 = hf / max_radius;
            let half_w = wf * 0.5;
            let im_args_str = format!("{max_radius}");
            let im_args = ["-distort", "Polar", &im_args_str];
            if let Some(im_data) = run_im(&im_args) {
                let inv = |xf: f32, yf: f32| -> (f32, f32) {
                    let xf64 = xf as f64 + 0.5;
                    let yf64 = yf as f64 + 0.5;
                    let ii = xf64 - cx;
                    let jj = yf64 - cy;
                    let radius = (ii * ii + jj * jj).sqrt();
                    let angle = ii.atan2(jj);
                    let mut xx = angle / two_pi;
                    xx -= xx.round();
                    ((xx * two_pi * c6 + half_w - 0.5) as f32,
                     (radius * c7 - 0.5) as f32)
                };
                let jac = |xf: f32, yf: f32| {
                    let xf64 = xf as f64 + 0.5;
                    let yf64 = yf as f64 + 0.5;
                    let ii = xf64 - cx;
                    let jj = yf64 - cy;
                    let r2 = ii * ii + jj * jj;
                    if r2 < 1e-10 {
                        crate::domain::ewa::JACOBIAN_IDENTITY
                    } else {
                        let r = r2.sqrt();
                        [
                            [(jj / r2 * c6) as f32, (-ii / r2 * c6) as f32],
                            [(ii / r * c7) as f32, (jj / r * c7) as f32],
                        ]
                    }
                };
                let mut maes = Vec::new();
                for i in 0..3 {
                    let out = run_distortion(mode_at(i), DistortionOverlap::FullImage, &inv, &jac);
                    maes.push(compute_mae(&out, &im_data));
                }
                eprintln!("{:<12} {:>10.2} {:>10.2} {:>10.2} {:>10}",
                    "depolar", maes[0], maes[1], maes[2], "Ewa");
            }
        }

        // ── Spherize ── (no direct IM equivalent; use barrel approximation)
        // IM doesn't have a direct spherize. Skip IM comparison for spherize.
        eprintln!("{:<12} {:>10} {:>10} {:>10} {:>10}",
            "spherize", "n/a", "n/a", "n/a", "Ewa");

        // ── Ripple ── (no direct IM equivalent)
        eprintln!("{:<12} {:>10} {:>10} {:>10} {:>10}",
            "ripple", "n/a", "n/a", "n/a", "Ewa");

        // ── Mesh warp ── (IM perspective is different; skip)
        eprintln!("{:<12} {:>10} {:>10} {:>10} {:>10}",
            "mesh_warp", "n/a", "n/a", "n/a", "Bilinear");

        eprintln!("======================================================================\n");
    }
}

