//! Tests for effect filters

use crate::domain::filters::common::*;

#[cfg(test)]
mod noise_tests {
    use super::*;

    #[test]
    fn perlin_deterministic() {
        let a = perlin_noise(64, 64, 42, 0.05, 4);
        let b = perlin_noise(64, 64, 42, 0.05, 4);
        assert_eq!(a, b, "same seed should produce identical output");
    }

    #[test]
    fn simplex_deterministic() {
        let a = simplex_noise(64, 64, 42, 0.05, 4);
        let b = simplex_noise(64, 64, 42, 0.05, 4);
        assert_eq!(a, b, "same seed should produce identical output");
    }

    #[test]
    fn perlin_different_seeds_differ() {
        let a = perlin_noise(64, 64, 1, 0.05, 4);
        let b = perlin_noise(64, 64, 2, 0.05, 4);
        assert_ne!(a, b, "different seeds should produce different output");
    }

    #[test]
    fn simplex_different_seeds_differ() {
        let a = simplex_noise(64, 64, 1, 0.05, 4);
        let b = simplex_noise(64, 64, 2, 0.05, 4);
        assert_ne!(a, b, "different seeds should produce different output");
    }

    #[test]
    fn perlin_output_dimensions() {
        let px = perlin_noise(128, 64, 0, 0.05, 4);
        assert_eq!(px.len(), 128 * 64);
    }

    #[test]
    fn simplex_output_dimensions() {
        let px = simplex_noise(128, 64, 0, 0.05, 4);
        assert_eq!(px.len(), 128 * 64);
    }

    #[test]
    fn perlin_statistical_properties() {
        let px = perlin_noise(256, 256, 42, 0.02, 6);
        let mean = px.iter().map(|&v| v as f64).sum::<f64>() / px.len() as f64;
        let stddev =
            (px.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / px.len() as f64).sqrt();

        eprintln!("Perlin 256x256: mean={mean:.1}, stddev={stddev:.1}");
        // Mean should be roughly centered (~128 ± 30)
        assert!(
            mean > 90.0 && mean < 170.0,
            "Perlin mean={mean:.1} outside expected range"
        );
        // Should have meaningful variation (not all one value)
        assert!(
            stddev > 10.0,
            "Perlin stddev={stddev:.1} too low — not enough variation"
        );
    }

    #[test]
    fn simplex_statistical_properties() {
        let px = simplex_noise(256, 256, 42, 0.02, 6);
        let mean = px.iter().map(|&v| v as f64).sum::<f64>() / px.len() as f64;
        let stddev =
            (px.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / px.len() as f64).sqrt();

        eprintln!("Simplex 256x256: mean={mean:.1}, stddev={stddev:.1}");
        assert!(
            mean > 90.0 && mean < 170.0,
            "Simplex mean={mean:.1} outside expected range"
        );
        assert!(
            stddev > 10.0,
            "Simplex stddev={stddev:.1} too low — not enough variation"
        );
    }

    #[test]
    fn perlin_uses_full_range() {
        let px = perlin_noise(256, 256, 42, 0.01, 8);
        let min = *px.iter().min().unwrap();
        let max = *px.iter().max().unwrap();
        eprintln!("Perlin range: [{min}, {max}]");
        // Should span a reasonable range
        assert!(max - min > 100, "Perlin range too narrow: [{min}, {max}]");
    }

    #[test]
    fn simplex_uses_full_range() {
        let px = simplex_noise(256, 256, 42, 0.01, 8);
        let min = *px.iter().min().unwrap();
        let max = *px.iter().max().unwrap();
        eprintln!("Simplex range: [{min}, {max}]");
        assert!(max - min > 100, "Simplex range too narrow: [{min}, {max}]");
    }

    #[test]
    fn single_octave_is_smooth() {
        // Single octave should produce very smooth output (low high-frequency content)
        let px = perlin_noise(64, 64, 42, 0.05, 1);
        let mut total_diff = 0u64;
        for y in 0..64 {
            for x in 1..64 {
                total_diff +=
                    (px[y * 64 + x] as i16 - px[y * 64 + x - 1] as i16).unsigned_abs() as u64;
            }
        }
        let avg_diff = total_diff as f64 / (64.0 * 63.0);
        eprintln!("Single octave avg adjacent diff: {avg_diff:.2}");
        // Adjacent pixels should differ by small amounts for smooth noise
        assert!(
            avg_diff < 10.0,
            "single octave too rough: avg_diff={avg_diff:.2}"
        );
    }
}

#[cfg(test)]
mod artistic_filter_tests {
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
    fn solarize_below_threshold_unchanged() {
        // All pixels at 100, threshold 128 → below threshold → unchanged
        let pixels = vec![100u8; 32 * 32 * 3];
        let info = rgb_info(32, 32);
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = solarize(r, &mut u, &info, &SolarizeParams { threshold: 128 }).unwrap();
        assert_eq!(result, pixels);
    }

    #[test]
    fn solarize_above_threshold_inverted() {
        // Pixel at 200, threshold 128 → above → 255-200=55
        let pixels = vec![200u8; 3];
        let info = rgb_info(1, 1);
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = solarize(r, &mut u, &info, &SolarizeParams { threshold: 128 }).unwrap();
        assert_eq!(result, vec![55, 55, 55]);
    }

    #[test]
    fn solarize_zero_threshold_inverts_all() {
        // threshold=0 means all non-zero pixels get inverted
        let pixels = vec![128u8; 3];
        let info = rgb_info(1, 1);
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = solarize(r, &mut u, &info, &SolarizeParams { threshold: 0 }).unwrap();
        assert_eq!(result, vec![127, 127, 127]);
    }

    #[test]
    fn emboss_preserves_size() {
        let pixels = vec![128u8; 32 * 32 * 3];
        let info = rgb_info(32, 32);
        let result = emboss_impl(&pixels, &info).unwrap();
        assert_eq!(result.len(), pixels.len());
    }

    #[test]
    fn emboss_flat_produces_midtone() {
        // Uniform image should produce mostly mid-gray after emboss
        let pixels = vec![128u8; 16 * 16 * 3];
        let info = rgb_info(16, 16);
        let result = emboss_impl(&pixels, &info).unwrap();
        let mean: f64 = result.iter().map(|&v| v as f64).sum::<f64>() / result.len() as f64;
        // Emboss of flat field: center weight=1 → ~128
        assert!(
            (mean - 128.0).abs() < 30.0,
            "flat emboss mean should be near 128, got {mean:.0}"
        );
    }

    #[test]
    fn oil_paint_preserves_size() {
        let pixels = vec![128u8; 32 * 32 * 3];
        let info = rgb_info(32, 32);
        let result = oil_paint(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &OilPaintParams { radius: 2 },
        )
        .unwrap();
        assert_eq!(result.len(), pixels.len());
    }

    #[test]
    fn oil_paint_uniform_is_identity() {
        // Uniform image → all pixels in same bin → output = input
        let pixels = vec![128u8; 16 * 16 * 3];
        let info = rgb_info(16, 16);
        let result = oil_paint(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &OilPaintParams { radius: 3 },
        )
        .unwrap();
        assert_eq!(result, pixels);
    }

    #[test]
    fn charcoal_outputs_gray() {
        // Charcoal outputs Gray8 (from Sobel)
        let pixels = vec![128u8; 32 * 32 * 3];
        let info = rgb_info(32, 32);
        let (result, out_info) = charcoal(
            &pixels,
            &info,
            &CharcoalParams {
                radius: 1.0,
                sigma: 0.5,
            },
        )
        .unwrap();
        // Output is Gray8: 32*32 = 1024 bytes (not 3072)
        assert_eq!(result.len(), 32 * 32);
        assert_eq!(out_info.format, PixelFormat::Gray8);
    }

    #[test]
    fn charcoal_flat_is_white() {
        // Flat image → no edges → Sobel = 0 → invert = 255 → white
        let pixels = vec![128u8; 16 * 16 * 3];
        let info = rgb_info(16, 16);
        let (result, out_info) = charcoal(
            &pixels,
            &info,
            &CharcoalParams {
                radius: 0.0,
                sigma: 0.0,
            },
        )
        .unwrap();
        // Output is Gray8
        assert_eq!(result.len(), 16 * 16);
        assert_eq!(out_info.format, PixelFormat::Gray8);
        let mean: f64 = result.iter().map(|&v| v as f64).sum::<f64>() / result.len() as f64;
        assert!(
            mean > 240.0,
            "charcoal of flat image should be near-white, got mean={mean:.0}"
        );
    }
}

#[cfg(test)]
mod add_noise_tests {
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

    // ── Gaussian noise tests ───────────────────────────────────────────────

    #[test]
    fn gaussian_noise_identity_at_zero_amount() {
        let pixels = vec![128u8; 64 * 64 * 3];
        let info = rgb_info(64, 64);
        let config = GaussianNoiseParams {
            amount: 0.0,
            mean: 0.0,
            sigma: 25.0,
            seed: 42,
        };
        let result = gaussian_noise(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &config,
        )
        .unwrap();
        assert_eq!(result, pixels);
    }

    #[test]
    fn gaussian_noise_deterministic_with_same_seed() {
        let pixels: Vec<u8> = (0..64 * 64 * 3).map(|i| (i % 256) as u8).collect();
        let info = rgb_info(64, 64);
        let config = GaussianNoiseParams {
            amount: 50.0,
            mean: 0.0,
            sigma: 25.0,
            seed: 123,
        };
        let r1 = gaussian_noise(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &config,
        )
        .unwrap();
        let r2 = gaussian_noise(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &config,
        )
        .unwrap();
        assert_eq!(r1, r2, "same seed must produce identical output");
    }

    #[test]
    fn gaussian_noise_different_seeds_differ() {
        let pixels = vec![128u8; 32 * 32 * 3];
        let info = rgb_info(32, 32);
        let c1 = GaussianNoiseParams {
            amount: 50.0,
            mean: 0.0,
            sigma: 25.0,
            seed: 1,
        };
        let c2 = GaussianNoiseParams {
            amount: 50.0,
            mean: 0.0,
            sigma: 25.0,
            seed: 2,
        };
        let r1 = gaussian_noise(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &c1,
        )
        .unwrap();
        let r2 = gaussian_noise(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &c2,
        )
        .unwrap();
        assert_ne!(r1, r2, "different seeds should produce different output");
    }

    #[test]
    fn gaussian_noise_statistics() {
        // On a mid-gray image with mean=0, sigma=25, the output mean should be near 128
        let pixels = vec![128u8; 64 * 64];
        let info = gray_info(64, 64);
        let config = GaussianNoiseParams {
            amount: 100.0,
            mean: 0.0,
            sigma: 25.0,
            seed: 42,
        };
        let result = gaussian_noise(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &config,
        )
        .unwrap();
        let mean: f64 = result.iter().map(|&v| v as f64).sum::<f64>() / result.len() as f64;
        assert!(
            (mean - 128.0).abs() < 10.0,
            "gaussian mean should be near 128, got {mean:.1}"
        );
        // Variance should be roughly sigma^2 * (amount/100)^2 = 625 at full strength
        let var: f64 = result
            .iter()
            .map(|&v| (v as f64 - mean).powi(2))
            .sum::<f64>()
            / result.len() as f64;
        assert!(
            var > 100.0 && var < 2000.0,
            "gaussian variance should be in reasonable range, got {var:.0}"
        );
    }

    #[test]
    fn gaussian_noise_preserves_alpha() {
        let pixels = vec![128, 64, 200, 255]; // one RGBA pixel, alpha=255
        let info = ImageInfo {
            width: 1,
            height: 1,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        let config = GaussianNoiseParams {
            amount: 100.0,
            mean: 0.0,
            sigma: 50.0,
            seed: 99,
        };
        let result = gaussian_noise(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &config,
        )
        .unwrap();
        assert_eq!(result[3], 255, "alpha channel must be preserved");
    }

    // ── Salt-and-pepper noise tests ────────────────────────────────────────

    #[test]
    fn salt_pepper_identity_at_zero_density() {
        let pixels = vec![128u8; 64 * 64 * 3];
        let info = rgb_info(64, 64);
        let config = SaltPepperNoiseParams {
            density: 0.0,
            seed: 42,
        };
        let result = salt_pepper_noise(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &config,
        ).unwrap();
        assert_eq!(result, pixels);
    }

    #[test]
    fn salt_pepper_deterministic_with_same_seed() {
        let pixels: Vec<u8> = (0..32 * 32 * 3).map(|i| (i % 256) as u8).collect();
        let info = rgb_info(32, 32);
        let config = SaltPepperNoiseParams {
            density: 0.1,
            seed: 77,
        };
        let r1 = salt_pepper_noise(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &config,
        ).unwrap();
        let r2 = salt_pepper_noise(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &config,
        ).unwrap();
        assert_eq!(r1, r2);
    }

    #[test]
    fn salt_pepper_only_produces_extremes() {
        let pixels = vec![128u8; 64 * 64 * 3];
        let info = rgb_info(64, 64);
        let config = SaltPepperNoiseParams {
            density: 1.0,
            seed: 42,
        };
        let result = salt_pepper_noise(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            &config,
        ).unwrap();
        // Every pixel should be either 0 or 255 (density=1 means all replaced)
        for chunk in result.chunks_exact(3) {
            assert!(
                chunk.iter().all(|&v| v == 0) || chunk.iter().all(|&v| v == 255),
                "salt-pepper at density=1 should only produce 0 or 255, got {:?}",
                chunk,
            );
        }
    }

    // ── Poisson noise tests ────────────────────────────────────────────────

    #[test]
    fn poisson_noise_identity_at_zero_scale() {
        let pixels = vec![128u8; 64 * 64 * 3];
        let info = rgb_info(64, 64);
        let config = PoissonNoiseParams {
            scale: 0.0,
            seed: 42,
        };
        let r = Rect::new(0, 0, info.width, info.height);
        let px = pixels.clone();
        let mut u = |_: Rect| Ok(px.clone());
        let result = poisson_noise(r, &mut u, &info, &config).unwrap();
        assert_eq!(result, pixels);
    }

    #[test]
    fn poisson_noise_deterministic_with_same_seed() {
        let pixels: Vec<u8> = (0..32 * 32 * 3).map(|i| (i % 256) as u8).collect();
        let info = rgb_info(32, 32);
        let config = PoissonNoiseParams {
            scale: 5.0,
            seed: 55,
        };
        let r = Rect::new(0, 0, info.width, info.height);
        let px = pixels.clone();
        let mut u1 = |_: Rect| Ok(px.clone());
        let mut u2 = |_: Rect| Ok(px.clone());
        let r1 = poisson_noise(r, &mut u1, &info, &config).unwrap();
        let r2 = poisson_noise(r, &mut u2, &info, &config).unwrap();
        assert_eq!(r1, r2);
    }

    #[test]
    fn poisson_noise_signal_dependent() {
        // Bright pixels should have more variance than dark pixels
        let w = 64u32;
        let h = 64u32;
        let n = (w * h) as usize;
        let dark: Vec<u8> = vec![20u8; n];
        let bright: Vec<u8> = vec![200u8; n];
        let info = gray_info(w, h);
        let config = PoissonNoiseParams {
            scale: 5.0,
            seed: 42,
        };
        let r = Rect::new(0, 0, info.width, info.height);
        let dk = dark.clone();
        let br = bright.clone();
        let mut u_dark = |_: Rect| Ok(dk.clone());
        let mut u_bright = |_: Rect| Ok(br.clone());
        let dark_result = poisson_noise(r, &mut u_dark, &info, &config).unwrap();
        let bright_result = poisson_noise(r, &mut u_bright, &info, &config).unwrap();
        let dark_var: f64 = {
            let m: f64 = dark_result.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
            dark_result
                .iter()
                .map(|&v| (v as f64 - m).powi(2))
                .sum::<f64>()
                / n as f64
        };
        let bright_var: f64 = {
            let m: f64 = bright_result.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
            bright_result
                .iter()
                .map(|&v| (v as f64 - m).powi(2))
                .sum::<f64>()
                / n as f64
        };
        assert!(
            bright_var > dark_var,
            "Poisson noise should be signal-dependent: bright_var={bright_var:.0} > dark_var={dark_var:.0}"
        );
    }

    // ── Uniform noise tests ────────────────────────────────────────────────

    #[test]
    fn uniform_noise_identity_at_zero_range() {
        let pixels = vec![128u8; 64 * 64 * 3];
        let info = rgb_info(64, 64);
        let config = UniformNoiseParams {
            range: 0.0,
            seed: 42,
        };
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = uniform_noise(r, &mut u, &info, &config).unwrap();
        assert_eq!(result, pixels);
    }

    #[test]
    fn uniform_noise_deterministic_with_same_seed() {
        let pixels: Vec<u8> = (0..32 * 32 * 3).map(|i| (i % 256) as u8).collect();
        let info = rgb_info(32, 32);
        let config = UniformNoiseParams {
            range: 30.0,
            seed: 33,
        };
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u1 = |_: Rect| Ok(pixels.clone());
        let mut u2 = |_: Rect| Ok(pixels.clone());
        let r1 = uniform_noise(r, &mut u1, &info, &config).unwrap();
        let r2 = uniform_noise(r, &mut u2, &info, &config).unwrap();
        assert_eq!(r1, r2);
    }

    #[test]
    fn uniform_noise_bounded() {
        // On a 128 image with range=50, no pixel should exceed [78, 178]
        let pixels = vec![128u8; 64 * 64];
        let info = gray_info(64, 64);
        let config = UniformNoiseParams {
            range: 50.0,
            seed: 42,
        };
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = uniform_noise(r, &mut u, &info, &config).unwrap();
        for &v in &result {
            assert!(
                v >= 78 && v <= 178,
                "uniform noise with range=50 on 128 should stay in [78,178], got {v}"
            );
        }
    }

    #[test]
    fn uniform_noise_statistics() {
        // Mean should remain near original, variance ≈ range^2/3 for uniform [-r,r]
        let pixels = vec![128u8; 64 * 64];
        let info = gray_info(64, 64);
        let config = UniformNoiseParams {
            range: 30.0,
            seed: 42,
        };
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = uniform_noise(r, &mut u, &info, &config).unwrap();
        let mean: f64 = result.iter().map(|&v| v as f64).sum::<f64>() / result.len() as f64;
        assert!(
            (mean - 128.0).abs() < 5.0,
            "uniform noise mean should be near 128, got {mean:.1}"
        );
        let var: f64 = result
            .iter()
            .map(|&v| (v as f64 - mean).powi(2))
            .sum::<f64>()
            / result.len() as f64;
        // Expected variance for uniform [-30, 30] = 30^2/3 = 300
        assert!(
            var > 100.0 && var < 600.0,
            "uniform noise variance should be near 300, got {var:.0}"
        );
    }
}

#[cfg(test)]
mod pixelate_tests {
    use super::*;

    #[test]
    fn default_block_size_is_visible() {
        let params = PixelateParams::default();
        assert!(
            params.block_size >= 2,
            "default block_size should produce a visible mosaic, got {}",
            params.block_size
        );
    }

    #[test]
    fn default_produces_mosaic() {
        // 16x16 gradient image, RGB — default params should visibly pixelate
        let w = 16u32;
        let h = 16u32;
        let ch = 3;
        let mut pixels = vec![0u8; (w * h) as usize * ch];
        for y in 0..h {
            for x in 0..w {
                let off = ((y * w + x) as usize) * ch;
                pixels[off] = (x * 16) as u8;
                pixels[off + 1] = (y * 16) as u8;
                pixels[off + 2] = 128;
            }
        }
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let rect = Rect::new(0, 0, w, h);
        let params = PixelateParams::default();
        let input = pixels.clone();
        let result = pixelate(
            rect,
            &mut |_| Ok(input.clone()),
            &info,
            &params,
        )
        .unwrap();
        // With a visible block_size (>=2), output should differ from input
        assert_ne!(
            result, pixels,
            "pixelate with default params should alter the image"
        );
    }

    #[test]
    fn param_descriptor_has_range() {
        let descriptors = PixelateParams::param_descriptors();
        let bs = descriptors
            .iter()
            .find(|d| d.name == "block_size")
            .expect("block_size descriptor missing");
        assert!(!bs.min.is_empty(), "block_size should have min");
        assert!(!bs.max.is_empty(), "block_size should have max");
        assert!(!bs.step.is_empty(), "block_size should have step");
        assert!(
            bs.default_val != "0",
            "block_size default should not be 0"
        );
    }
}

