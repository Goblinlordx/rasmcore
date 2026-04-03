//! Tests for color filters

use crate::domain::filters::common::*;
#[allow(unused_imports)]
use crate::domain::filter_traits::CpuFilter;

#[cfg(test)]
mod color_manipulation_tests {
    use super::*;
    use crate::domain::types::{ColorSpace, ImageInfo, PixelFormat};

    fn info_rgb8(w: u32, h: u32) -> ImageInfo {
        ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        }
    }

    fn solid_rgb(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        (0..(w * h)).flat_map(|_| [r, g, b]).collect()
    }

    // ── Channel Mixer ──

    #[test]
    fn channel_mixer_identity_preserves_pixels() {
        let pixels = solid_rgb(4, 4, 100, 150, 200);
        let info = info_rgb8(4, 4);
        let result = ChannelMixerParams {
                rr: 1.0,
                rg: 0.0,
                rb: 0.0,
                gr: 0.0,
                gg: 1.0,
                gb: 0.0,
                br: 0.0,
                bg: 0.0,
                bb: 1.0,
            }.compute(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
        )
        .unwrap();
        assert_eq!(result, pixels);
    }

    #[test]
    fn channel_mixer_red_only() {
        let pixels = solid_rgb(2, 2, 100, 150, 200);
        let info = info_rgb8(2, 2);
        // Output red = 1.0*R + 0*G + 0*B, green = 0, blue = 0
        let result = ChannelMixerParams {
                rr: 1.0,
                rg: 0.0,
                rb: 0.0,
                gr: 0.0,
                gg: 0.0,
                gb: 0.0,
                br: 0.0,
                bg: 0.0,
                bb: 0.0,
            }.compute(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
        )
        .unwrap();
        for chunk in result.chunks_exact(3) {
            assert_eq!(chunk[0], 100);
            assert_eq!(chunk[1], 0);
            assert_eq!(chunk[2], 0);
        }
    }

    #[test]
    fn channel_mixer_swap_red_blue() {
        let pixels = solid_rgb(2, 2, 100, 150, 200);
        let info = info_rgb8(2, 2);
        // Swap R and B channels
        let result = ChannelMixerParams {
                rr: 0.0,
                rg: 0.0,
                rb: 1.0,
                gr: 0.0,
                gg: 1.0,
                gb: 0.0,
                br: 1.0,
                bg: 0.0,
                bb: 0.0,
            }.compute(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
        )
        .unwrap();
        for chunk in result.chunks_exact(3) {
            assert_eq!(chunk[0], 200); // was blue
            assert_eq!(chunk[1], 150); // green unchanged
            assert_eq!(chunk[2], 100); // was red
        }
    }

    #[test]
    fn channel_mixer_clamps_overflow() {
        let pixels = solid_rgb(2, 2, 200, 200, 200);
        let info = info_rgb8(2, 2);
        // 2.0 * R would overflow — should clamp to 255
        let result = ChannelMixerParams {
                rr: 2.0,
                rg: 0.0,
                rb: 0.0,
                gr: 0.0,
                gg: 1.0,
                gb: 0.0,
                br: 0.0,
                bg: 0.0,
                bb: 1.0,
            }.compute(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
        )
        .unwrap();
        assert_eq!(result[0], 255);
    }

    // ── Vibrance ──

    #[test]
    fn vibrance_zero_is_identity() {
        let pixels = solid_rgb(4, 4, 100, 150, 200);
        let info = info_rgb8(4, 4);
        let result = VibranceParams { amount: 0.0 }.compute(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
        )
        .unwrap();
        assert_eq!(result, pixels);
    }

    #[test]
    fn vibrance_positive_boosts_muted_more() {
        let info = info_rgb8(1, 2);
        // Pixel 1: low saturation (gray-ish)
        // Pixel 2: high saturation (vivid red)
        let pixels = vec![120, 130, 125, 255, 20, 20];

        let result = VibranceParams { amount: 50.0 }.compute(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
        )
        .unwrap();

        // The muted pixel should change more than the vivid one
        let muted_change = (result[0] as i32 - 120).abs()
            + (result[1] as i32 - 130).abs()
            + (result[2] as i32 - 125).abs();
        let vivid_change = (result[3] as i32 - 255).abs()
            + (result[4] as i32 - 20).abs()
            + (result[5] as i32 - 20).abs();

        assert!(
            muted_change >= vivid_change,
            "muted change ({muted_change}) should be >= vivid change ({vivid_change})"
        );
    }

    #[test]
    fn vibrance_negative_desaturates() {
        // Use a moderately saturated color (not fully saturated)
        let pixels = solid_rgb(2, 2, 200, 100, 80);
        let info = info_rgb8(2, 2);
        let result = VibranceParams { amount: -80.0 }.compute(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
        )
        .unwrap();
        // Should become less saturated: channels should converge toward each other
        let orig_range = 200i32 - 80;
        let new_range = (result[0] as i32 - result[2] as i32).abs();
        assert!(
            new_range < orig_range,
            "negative vibrance should reduce color range: {new_range} should be < {orig_range}"
        );
    }

    // ── Gradient Map ──

    #[test]
    fn gradient_map_bw_produces_grayscale() {
        let info = info_rgb8(2, 2);
        let pixels = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128];
        let result = gradient_map(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            "0.0:000000,1.0:FFFFFF".to_string(),
        )
        .unwrap();
        // Each pixel's RGB should all be equal (grayscale)
        for chunk in result.chunks_exact(3) {
            assert_eq!(chunk[0], chunk[1], "R should equal G for BW gradient");
            assert_eq!(chunk[1], chunk[2], "G should equal B for BW gradient");
        }
    }

    #[test]
    fn gradient_map_solid_black() {
        let pixels = solid_rgb(2, 2, 0, 0, 0);
        let info = info_rgb8(2, 2);
        let result = gradient_map(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            "0.0:FF0000,1.0:0000FF".to_string(),
        )
        .unwrap();
        // Luminance 0 → first stop (red)
        for chunk in result.chunks_exact(3) {
            assert_eq!(chunk[0], 255);
            assert_eq!(chunk[1], 0);
            assert_eq!(chunk[2], 0);
        }
    }

    #[test]
    fn gradient_map_solid_white() {
        let pixels = solid_rgb(2, 2, 255, 255, 255);
        let info = info_rgb8(2, 2);
        let result = gradient_map(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            "0.0:FF0000,1.0:0000FF".to_string(),
        )
        .unwrap();
        // Luminance 1.0 → last stop (blue)
        for chunk in result.chunks_exact(3) {
            assert_eq!(chunk[0], 0);
            assert_eq!(chunk[1], 0);
            assert_eq!(chunk[2], 255);
        }
    }

    #[test]
    fn gradient_map_invalid_stops_returns_error() {
        let pixels = solid_rgb(2, 2, 128, 128, 128);
        let info = info_rgb8(2, 2);
        let result = gradient_map(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            "invalid".to_string(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn gradient_map_preserves_alpha() {
        let info = ImageInfo {
            width: 2,
            height: 1,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        let pixels = vec![128, 128, 128, 200, 0, 0, 0, 100];
        let result = gradient_map(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            "0.0:000000,1.0:FFFFFF".to_string(),
        )
        .unwrap();
        assert_eq!(result[3], 200, "alpha should be preserved");
        assert_eq!(result[7], 100, "alpha should be preserved");
    }

    // ── Sparse Color ──

    #[test]
    fn sparse_color_single_point_fills_uniform() {
        let pixels = solid_rgb(4, 4, 0, 0, 0);
        let info = info_rgb8(4, 4);
        let r = Rect::new(0, 0, 4, 4);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = sparse_color(
            r,
            &mut u,
            &info,
            "2,2:FF0000".to_string(),
            &SparseColorParams {
                points: String::new(),
                power: 2.0,
            },
        )
        .unwrap();
        // All pixels should be red (only one control point)
        for chunk in result.chunks_exact(3) {
            assert_eq!(chunk, [255, 0, 0]);
        }
    }

    #[test]
    fn sparse_color_two_points_gradient() {
        let pixels = solid_rgb(8, 1, 0, 0, 0);
        let info = info_rgb8(8, 1);
        let r = Rect::new(0, 0, 8, 1);
        let mut u = |_: Rect| Ok(pixels.clone());
        // Red at x=0, blue at x=7
        let result = sparse_color(
            r,
            &mut u,
            &info,
            "0,0:FF0000;7,0:0000FF".to_string(),
            &SparseColorParams {
                points: String::new(),
                power: 2.0,
            },
        )
        .unwrap();
        // First pixel should be close to red
        assert!(
            result[0] > 200,
            "first pixel R={} should be >200",
            result[0]
        );
        // Last pixel should be close to blue
        assert!(
            result[7 * 3 + 2] > 200,
            "last pixel B={} should be >200",
            result[7 * 3 + 2]
        );
        // Middle should be a mix
        let mid = 4 * 3;
        assert!(
            result[mid] > 0 && result[mid + 2] > 0,
            "middle should have both R and B"
        );
    }

    #[test]
    fn sparse_color_invalid_points_error() {
        let pixels = solid_rgb(4, 4, 0, 0, 0);
        let info = info_rgb8(4, 4);
        let r = Rect::new(0, 0, 4, 4);
        let mut u = |_: Rect| Ok(pixels.clone());
        assert!(
            sparse_color(
                r,
                &mut u,
                &info,
                "invalid".to_string(),
                &SparseColorParams {
                    points: String::new(),
                    power: 2.0,
                },
            )
            .is_err()
        );
    }

    // ── Modulate ──

    #[test]
    fn modulate_identity() {
        let pixels = solid_rgb(4, 4, 100, 150, 200);
        let info = info_rgb8(4, 4);
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = ModulateParams {
                brightness: 100.0,
                saturation: 100.0,
                hue: 0.0
        }.compute(
            r,
            &mut u,
            &info,
        )
        .unwrap();
        // Identity: (100%, 100%, 0 deg) should preserve pixels
        for (a, b) in result.iter().zip(pixels.iter()) {
            assert!(
                (*a as i32 - *b as i32).abs() <= 1,
                "modulate identity: {a} vs {b}"
            );
        }
    }

    #[test]
    fn modulate_brightness_zero_is_black() {
        let pixels = solid_rgb(2, 2, 100, 150, 200);
        let info = info_rgb8(2, 2);
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = ModulateParams {
                brightness: 0.0,
                saturation: 100.0,
                hue: 0.0
        }.compute(
            r,
            &mut u,
            &info,
        )
        .unwrap();
        for &v in &result {
            assert_eq!(v, 0, "brightness=0 should produce black");
        }
    }

    #[test]
    fn modulate_saturation_zero_is_gray() {
        let pixels = solid_rgb(2, 2, 255, 0, 0);
        let info = info_rgb8(2, 2);
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = ModulateParams {
                brightness: 100.0,
                saturation: 0.0,
                hue: 0.0
        }.compute(
            r,
            &mut u,
            &info,
        )
        .unwrap();
        // Desaturated: all channels should be equal (gray)
        for chunk in result.chunks_exact(3) {
            assert_eq!(chunk[0], chunk[1], "R should equal G when desaturated");
            assert_eq!(chunk[1], chunk[2], "G should equal B when desaturated");
        }
    }

    #[test]
    fn modulate_hue_rotation() {
        let pixels = solid_rgb(2, 2, 255, 0, 0);
        let info = info_rgb8(2, 2);
        // Rotate hue by 120 degrees: red -> green
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = ModulateParams {
                brightness: 100.0,
                saturation: 100.0,
                hue: 120.0
        }.compute(
            r,
            &mut u,
            &info,
        )
        .unwrap();
        // Should be approximately green
        assert!(
            result[1] > result[0],
            "after 120 deg hue shift, G should dominate"
        );
    }

    // ── Photo Filter ──

    #[test]
    fn photo_filter_density_zero_is_identity() {
        let pixels = solid_rgb(4, 4, 100, 150, 200);
        let info = info_rgb8(4, 4);
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = PhotoFilterParams {
                color_r: 255,
                color_g: 200,
                color_b: 0,
                density: 0.0,
                preserve_luminosity: 1
        }.compute(
            r,
            &mut u,
            &info,
        )
        .unwrap();
        assert_eq!(result, pixels);
    }

    #[test]
    fn photo_filter_warm_tint() {
        let pixels = solid_rgb(4, 4, 128, 128, 128);
        let info = info_rgb8(4, 4);
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(pixels.clone());
        // Warm filter (orange) at 50% density
        let result = PhotoFilterParams {
                color_r: 236,
                color_g: 138,
                color_b: 0,
                density: 50.0,
                preserve_luminosity: 0
        }.compute(
            r,
            &mut u,
            &info,
        )
        .unwrap();
        // Red should increase, blue should decrease
        assert!(result[0] > 128, "warm filter should increase red");
        assert!(result[2] < 128, "warm filter should decrease blue");
    }

    #[test]
    fn photo_filter_preserve_luminosity() {
        let pixels = solid_rgb(4, 4, 128, 128, 128);
        let info = info_rgb8(4, 4);
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = PhotoFilterParams {
                color_r: 255,
                color_g: 0,
                color_b: 0,
                density: 50.0,
                preserve_luminosity: 1
        }.compute(
            r,
            &mut u,
            &info,
        )
        .unwrap();
        // With preserve_luminosity, the total brightness should be similar
        let orig_luma = 128u32; // gray pixel
        let new_luma =
            (result[0] as u32 * 2126 + result[1] as u32 * 7152 + result[2] as u32 * 722) / 10000;
        assert!(
            (new_luma as i32 - orig_luma as i32).unsigned_abs() < 5,
            "luminosity should be preserved: orig={orig_luma}, new={new_luma}"
        );
    }

    // ── Spin Blur ──

    #[test]
    fn spin_blur_angle_zero_is_identity() {
        let pixels = solid_rgb(8, 8, 100, 150, 200);
        let info = info_rgb8(8, 8);
        let result = SpinBlurParams {
                center_x: 0.5,
                center_y: 0.5,
                angle: 0.0,
            }.compute(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
        )
        .unwrap();
        assert_eq!(result, pixels);
    }

    #[test]
    fn spin_blur_produces_different_output() {
        let mut pixels = vec![0u8; 32 * 32 * 3];
        // Create a pattern with contrast (not solid)
        for y in 0..32u32 {
            for x in 0..32u32 {
                let idx = (y * 32 + x) as usize * 3;
                pixels[idx] = (x * 8) as u8;
                pixels[idx + 1] = (y * 8) as u8;
                pixels[idx + 2] = 128;
            }
        }
        let info = info_rgb8(32, 32);
        let result = SpinBlurParams {
                center_x: 0.5,
                center_y: 0.5,
                angle: 30.0,
            }.compute(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
        )
        .unwrap();
        assert_ne!(result, pixels, "spin blur should modify pixels");
    }

    #[test]
    fn spin_blur_center_pixel_unchanged() {
        let mut pixels = vec![128u8; 16 * 16 * 3];
        // Put a distinctive pixel at center (8,8)
        let center = (8 * 16 + 8) * 3;
        pixels[center] = 255;
        pixels[center + 1] = 0;
        pixels[center + 2] = 0;
        let info = info_rgb8(16, 16);
        let result = SpinBlurParams {
                center_x: 0.5,
                center_y: 0.5,
                angle: 45.0,
            }.compute(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
        )
        .unwrap();
        // Center pixel should be close to original (radius ≈ 0, no arc blur)
        assert_eq!(result[center], 255, "center should stay red");
    }

    // ── Gradient & Pattern Generators ──

    #[test]
    fn gradient_linear_horizontal() {
        let pixels = gradient_linear(16, 4, 0, 0, 0, 255, 255, 255, 0.0);
        assert_eq!(pixels.len(), 16 * 4 * 3);
        // Left edge should be dark, right edge should be bright
        assert!(pixels[0] < 20, "left should be near black: {}", pixels[0]);
        assert!(
            pixels[15 * 3] > 235,
            "right should be near white: {}",
            pixels[15 * 3]
        );
    }

    #[test]
    fn gradient_linear_vertical() {
        let pixels = gradient_linear(4, 16, 255, 0, 0, 0, 0, 255, 90.0);
        // Top should be red-ish, bottom should be blue-ish
        assert!(pixels[0] > 200, "top R should be high");
        assert!(pixels[(15 * 4) * 3 + 2] > 200, "bottom B should be high");
    }

    #[test]
    fn gradient_radial_center_is_color1() {
        let pixels = gradient_radial(16, 16, 255, 0, 0, 0, 0, 255, 0.5, 0.5);
        // Center pixel (8,8) should be close to color1 (red)
        let center = (8 * 16 + 8) * 3;
        assert!(pixels[center] > 200, "center R={}", pixels[center]);
        assert!(pixels[center + 2] < 50, "center B={}", pixels[center + 2]);
    }

    #[test]
    fn checkerboard_pattern() {
        let pixels = checkerboard(8, 8, 4, 255, 255, 255, 0, 0, 0);
        assert_eq!(pixels.len(), 8 * 8 * 3);
        // (0,0) should be white (cell 0,0 = even)
        assert_eq!(pixels[0], 255);
        // (4,0) should be black (cell 1,0 = odd)
        assert_eq!(pixels[4 * 3], 0);
        // (4,4) should be white (cell 1,1 = even)
        assert_eq!(pixels[(4 * 8 + 4) * 3], 255);
    }

    #[test]
    fn plasma_deterministic() {
        let a = plasma(32, 32, 42, 1.0);
        let b = plasma(32, 32, 42, 1.0);
        assert_eq!(a, b, "same seed should produce same output");
    }

    #[test]
    fn plasma_different_seeds() {
        let a = plasma(32, 32, 1, 1.0);
        let b = plasma(32, 32, 2, 1.0);
        assert_ne!(a, b, "different seeds should produce different output");
    }

    #[test]
    fn plasma_has_color_variation() {
        let pixels = plasma(64, 64, 42, 2.0);
        // Should have some variation (not all one color)
        let unique: std::collections::HashSet<u8> = pixels.iter().copied().collect();
        assert!(
            unique.len() > 10,
            "plasma should have color variation, got {} unique values",
            unique.len()
        );
    }

    // ── Mask Apply ──

    #[test]
    fn mask_apply_white_mask_preserves_opacity() {
        let pixels = solid_rgb(4, 4, 100, 150, 200);
        let info = info_rgb8(4, 4);
        let mask = vec![255u8; 4 * 4]; // All white = fully opaque
        let result = mask_apply(&pixels, &info, &mask, 4, 4, 0).unwrap();
        // Should be RGBA with alpha = 255
        assert_eq!(result.len(), 4 * 4 * 4);
        for chunk in result.chunks_exact(4) {
            assert_eq!(chunk[3], 255, "white mask should give full opacity");
        }
    }

    #[test]
    fn mask_apply_black_mask_makes_transparent() {
        let pixels = solid_rgb(4, 4, 100, 150, 200);
        let info = info_rgb8(4, 4);
        let mask = vec![0u8; 4 * 4]; // All black = fully transparent
        let result = mask_apply(&pixels, &info, &mask, 4, 4, 0).unwrap();
        for chunk in result.chunks_exact(4) {
            assert_eq!(chunk[3], 0, "black mask should give zero opacity");
        }
    }

    #[test]
    fn mask_apply_invert() {
        let pixels = solid_rgb(4, 4, 100, 150, 200);
        let info = info_rgb8(4, 4);
        let mask = vec![200u8; 4 * 4];
        let normal = mask_apply(&pixels, &info, &mask, 4, 4, 0).unwrap();
        let inverted = mask_apply(&pixels, &info, &mask, 4, 4, 1).unwrap();
        assert_eq!(normal[3], 200);
        assert_eq!(inverted[3], 55); // 255 - 200
    }

    #[test]
    fn mask_apply_resize() {
        let pixels = solid_rgb(8, 8, 128, 128, 128);
        let info = info_rgb8(8, 8);
        // 2x2 mask applied to 8x8 image (nearest-neighbor resize)
        let mask = vec![255, 0, 0, 255]; // Gray8: TL=white, TR=black, BL=black, BR=white
        let result = mask_apply(&pixels, &info, &mask, 2, 2, 0).unwrap();
        // Top-left quadrant should be opaque (mask = 255)
        assert_eq!(result[3], 255, "TL should be opaque");
        // Top-right quadrant should be transparent (mask = 0)
        let tr = (0 * 8 + 4) * 4 + 3;
        assert_eq!(result[tr], 0, "TR should be transparent");
    }

    // ── Blend-If ──

    #[test]
    fn blend_if_full_range_is_top_layer() {
        let info = info_rgb8(4, 4);
        let top = solid_rgb(4, 4, 255, 0, 0); // red
        let bottom = solid_rgb(4, 4, 0, 0, 255); // blue
        // Full range (0-255, 0-255) = top layer fully visible
        let result = blend_if(&top, &info, &bottom, 0, 255, 0, 255, 0).unwrap();
        assert_eq!(result[0], 255, "full range should show top (red)");
        assert_eq!(result[2], 0, "full range should not show bottom (blue)");
    }

    #[test]
    fn blend_if_narrow_range_reveals_underlying() {
        let info = info_rgb8(4, 4);
        let top = solid_rgb(4, 4, 200, 200, 200); // bright gray
        let bottom = solid_rgb(4, 4, 50, 50, 50); // dark gray
        // This layer visible only in dark range (0-100): bright top should be hidden
        let result = blend_if(&top, &info, &bottom, 0, 100, 0, 255, 0).unwrap();
        // Top luma ~200, outside [0,100] → should show underlying
        assert!(
            result[0] < 150,
            "bright top should be hidden when range is 0-100: got {}",
            result[0]
        );
    }
}

