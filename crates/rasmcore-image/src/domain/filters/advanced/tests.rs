//! Tests for advanced filters

use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

#[cfg(test)]
mod perspective_tests {
    use super::*;

    fn make_rgb_image(w: u32, h: u32) -> (Vec<u8>, ImageInfo) {
        let pixels: Vec<u8> = (0..(w * h * 3) as usize).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb8,
            color_space: crate::domain::types::ColorSpace::Srgb,
        };
        (pixels, info)
    }

    fn make_rgba_image(w: u32, h: u32) -> (Vec<u8>, ImageInfo) {
        let pixels: Vec<u8> = (0..(w * h * 4) as usize).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgba8,
            color_space: crate::domain::types::ColorSpace::Srgb,
        };
        (pixels, info)
    }

    fn make_gray_image(w: u32, h: u32) -> (Vec<u8>, ImageInfo) {
        let pixels: Vec<u8> = (0..(w * h) as usize).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Gray8,
            color_space: crate::domain::types::ColorSpace::Srgb,
        };
        (pixels, info)
    }

    // ── CvRng ────────────────────────────────────────────────────────────

    #[test]
    fn cv_rng_deterministic_with_same_seed() {
        let mut rng1 = CvRng::new(12345);
        let mut rng2 = CvRng::new(12345);
        for _ in 0..100 {
            assert_eq!(rng1.next_u32(), rng2.next_u32());
        }
    }

    #[test]
    fn cv_rng_different_seeds_differ() {
        let mut rng1 = CvRng::new(1);
        let mut rng2 = CvRng::new(2);
        let mut same = true;
        for _ in 0..10 {
            if rng1.next_u32() != rng2.next_u32() {
                same = false;
                break;
            }
        }
        assert!(!same);
    }

    // ── solve_homography_4pt (OpenCV formulation) ────────────────────────

    #[test]
    fn homography_identity_mapping() {
        let pts = [(0.0f32, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)];
        let h = solve_homography_4pt(&pts, &pts).unwrap();
        for (i, &v) in h.iter().enumerate() {
            let expected = if i == 0 || i == 4 || i == 8 { 1.0 } else { 0.0 };
            assert!(
                (v - expected).abs() < 1e-6,
                "h[{i}] = {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn homography_translation() {
        let src = [(0.0f32, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let dst = [(10.0f32, 20.0), (11.0, 20.0), (11.0, 21.0), (10.0, 21.0)];
        let h = solve_homography_4pt(&src, &dst).unwrap();
        assert!((h[2] - 10.0).abs() < 1e-6, "tx = {}", h[2]);
        assert!((h[5] - 20.0).abs() < 1e-6, "ty = {}", h[5]);
    }

    #[test]
    fn homography_scale() {
        let src = [(0.0f32, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let dst = [(0.0f32, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)];
        let h = solve_homography_4pt(&src, &dst).unwrap();
        assert!((h[0] - 2.0).abs() < 1e-6, "sx = {}", h[0]);
        assert!((h[4] - 2.0).abs() < 1e-6, "sy = {}", h[4]);
    }

    #[test]
    fn homography_c22_is_one() {
        // OpenCV convention: M[2][2] = 1.0 always
        let src = [(0.0f32, 0.0), (100.0, 0.0), (110.0, 80.0), (10.0, 90.0)];
        let dst = [(5.0f32, 5.0), (95.0, 10.0), (90.0, 85.0), (15.0, 95.0)];
        let h = solve_homography_4pt(&src, &dst).unwrap();
        assert!(
            (h[8] - 1.0).abs() < 1e-12,
            "c22 should be exactly 1.0, got {}",
            h[8]
        );
    }

    // ── invert_3x3 ───────────────────────────────────────────────────────

    #[test]
    fn invert_identity() {
        let id = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let inv = invert_3x3(&id).unwrap();
        for i in 0..9 {
            assert!((inv[i] - id[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn invert_singular_returns_none() {
        let singular = [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 0.0, 0.0, 0.0];
        assert!(invert_3x3(&singular).is_none());
    }

    #[test]
    fn invert_roundtrip() {
        let m = [2.0, 1.0, 0.0, 0.0, 3.0, 1.0, 1.0, 0.0, 2.0];
        let inv = invert_3x3(&m).unwrap();
        let mut prod = [0.0f64; 9];
        for r in 0..3 {
            for c in 0..3 {
                for k in 0..3 {
                    prod[r * 3 + c] += m[r * 3 + k] * inv[k * 3 + c];
                }
            }
        }
        for i in 0..9 {
            let expected = if i == 0 || i == 4 || i == 8 { 1.0 } else { 0.0 };
            assert!((prod[i] - expected).abs() < 1e-9, "prod[{i}] = {}", prod[i]);
        }
    }

    // ── line_intersection ────────────────────────────────────────────────

    #[test]
    fn intersection_perpendicular() {
        let l1 = LineSegment {
            x1: 0,
            y1: 5,
            x2: 10,
            y2: 5,
        };
        let l2 = LineSegment {
            x1: 5,
            y1: 0,
            x2: 5,
            y2: 10,
        };
        let (ix, iy) = line_intersection(&l1, &l2).unwrap();
        assert!((ix - 5.0).abs() < 1e-4);
        assert!((iy - 5.0).abs() < 1e-4);
    }

    #[test]
    fn intersection_parallel_returns_none() {
        let l1 = LineSegment {
            x1: 0,
            y1: 0,
            x2: 10,
            y2: 0,
        };
        let l2 = LineSegment {
            x1: 0,
            y1: 5,
            x2: 10,
            y2: 5,
        };
        assert!(line_intersection(&l1, &l2).is_none());
    }

    // ── bilinear weight table ────────────────────────────────────────────

    #[test]
    fn bilinear_tab_weights_sum_to_scale() {
        let tab = build_bilinear_tab();
        for (idx, w4) in tab.iter().enumerate() {
            let sum: i32 = w4.iter().sum();
            assert_eq!(
                sum, INTER_REMAP_COEF_SCALE,
                "tab[{idx}] weights sum to {sum}, expected {INTER_REMAP_COEF_SCALE}"
            );
        }
    }

    #[test]
    fn bilinear_tab_identity_at_origin() {
        let tab = build_bilinear_tab();
        // At (0,0): all weight on top-left
        assert_eq!(tab[0], [INTER_REMAP_COEF_SCALE, 0, 0, 0]);
    }

    // ── perspective_warp (OpenCV fixed-point) ────────────────────────────

    #[test]
    fn warp_identity_preserves_all_pixels() {
        let (px, info) = make_rgb_image(16, 16);
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(px.clone());
        let result = perspective_warp(
            r,
            &mut u,
            &info,
            &identity,
            &PerspectiveWarpParams {
                out_width: 16,
                out_height: 16,
            },
        )
        .unwrap();
        // With fixed-point, identity warp at integer coords should be exact
        assert_eq!(
            result, px,
            "identity warp should preserve all pixels exactly"
        );
    }

    #[test]
    fn warp_preserves_output_dimensions() {
        let (px, info) = make_rgb_image(32, 24);
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(px.clone());
        let result = perspective_warp(
            r,
            &mut u,
            &info,
            &identity,
            &PerspectiveWarpParams {
                out_width: 64,
                out_height: 48,
            },
        )
        .unwrap();
        assert_eq!(result.len(), 64 * 48 * 3);
    }

    #[test]
    fn warp_rgba_preserves_channels() {
        let (px, info) = make_rgba_image(16, 16);
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(px.clone());
        let result = perspective_warp(
            r,
            &mut u,
            &info,
            &identity,
            &PerspectiveWarpParams {
                out_width: 16,
                out_height: 16,
            },
        )
        .unwrap();
        assert_eq!(result.len(), 16 * 16 * 4);
    }

    #[test]
    fn warp_gray_works() {
        let (px, info) = make_gray_image(16, 16);
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(px.clone());
        let result = perspective_warp(
            r,
            &mut u,
            &info,
            &identity,
            &PerspectiveWarpParams {
                out_width: 16,
                out_height: 16,
            },
        )
        .unwrap();
        assert_eq!(result.len(), 16 * 16);
    }

    #[test]
    fn warp_translation_shifts_pixels() {
        let w = 10u32;
        let h = 10u32;
        let mut pixels = vec![0u8; (w * h * 3) as usize];
        let idx = (5 * w as usize + 5) * 3;
        pixels[idx] = 255;
        pixels[idx + 1] = 255;
        pixels[idx + 2] = 255;
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb8,
            color_space: crate::domain::types::ColorSpace::Srgb,
        };

        // Inverse map: output (ox,oy) → input (ox+2, oy+1)
        let mat = [1.0, 0.0, 2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(pixels.clone());
        let result = perspective_warp(
            r,
            &mut u,
            &info,
            &mat,
            &PerspectiveWarpParams {
                out_width: w,
                out_height: h,
            },
        )
        .unwrap();

        // White pixel at input (5,5) → output (3,4)
        let expected_idx = (4 * w as usize + 3) * 3;
        assert_eq!(result[expected_idx], 255);
        assert_eq!(result[expected_idx + 1], 255);
        assert_eq!(result[expected_idx + 2], 255);
    }

    // ── perspective_correct ──────────────────────────────────────────────

    #[test]
    fn correct_zero_strength_is_identity() {
        let (px, info) = make_rgb_image(32, 32);
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(px.clone());
        let result = PerspectiveCorrectParams { strength: 0.0
        }.compute(
            r,
            &mut u,
            &info,
        )
        .unwrap();
        assert_eq!(result, px);
    }

    #[test]
    fn correct_preserves_dimensions() {
        let (px, info) = make_rgb_image(64, 48);
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(px.clone());
        let result = PerspectiveCorrectParams { strength: 1.0
        }.compute(
            r,
            &mut u,
            &info,
        )
        .unwrap();
        assert_eq!(result.len(), px.len());
    }

    #[test]
    fn correct_rgba_preserves_length() {
        let (px, info) = make_rgba_image(32, 32);
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(px.clone());
        let result = PerspectiveCorrectParams { strength: 0.5
        }.compute(
            r,
            &mut u,
            &info,
        )
        .unwrap();
        assert_eq!(result.len(), px.len());
    }

    #[test]
    fn correct_gray_preserves_length() {
        let (px, info) = make_gray_image(32, 32);
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(px.clone());
        let result = PerspectiveCorrectParams { strength: 0.5
        }.compute(
            r,
            &mut u,
            &info,
        )
        .unwrap();
        assert_eq!(result.len(), px.len());
    }

    // ── hough_lines_p (PPHT) ────────────────────────────────────────────

    #[test]
    fn hough_requires_gray8() {
        let (px, info) = make_rgb_image(16, 16);
        let result = hough_lines_p(&px, &info, 1.0, 0.01, 5, 10, 5, 0);
        assert!(result.is_err());
    }

    #[test]
    fn hough_deterministic_with_seed() {
        let w = 64u32;
        let h = 64u32;
        let mut pixels = vec![0u8; (w * h) as usize];
        for x in 5..60 {
            pixels[32 * w as usize + x] = 255;
        }
        for y in 10..55 {
            pixels[y * w as usize + 20] = 255;
        }
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Gray8,
            color_space: crate::domain::types::ColorSpace::Srgb,
        };
        let theta = std::f32::consts::PI / 180.0;

        let lines1 = hough_lines_p(&pixels, &info, 1.0, theta, 15, 20, 5, 42).unwrap();
        let lines2 = hough_lines_p(&pixels, &info, 1.0, theta, 15, 20, 5, 42).unwrap();
        assert_eq!(
            lines1.len(),
            lines2.len(),
            "same seed should give same count"
        );
        for (a, b) in lines1.iter().zip(lines2.iter()) {
            assert_eq!((a.x1, a.y1, a.x2, a.y2), (b.x1, b.y1, b.x2, b.y2));
        }
    }

    #[test]
    fn hough_detects_horizontal_line() {
        let w = 64u32;
        let h = 64u32;
        let mut pixels = vec![0u8; (w * h) as usize];
        for x in 5..60 {
            pixels[32 * w as usize + x] = 255;
        }
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Gray8,
            color_space: crate::domain::types::ColorSpace::Srgb,
        };
        let lines = hough_lines_p(
            &pixels,
            &info,
            1.0,
            std::f32::consts::PI / 180.0,
            20,
            20,
            5,
            0,
        )
        .unwrap();
        assert!(!lines.is_empty(), "should detect horizontal line");
        let has_horizontal = lines.iter().any(|l| {
            let dy = (l.y2 - l.y1).abs();
            let dx = (l.x2 - l.x1).abs();
            dx > 20 && dy < 5
        });
        assert!(has_horizontal, "should find a horizontal line segment");
    }

    #[test]
    fn hough_detects_vertical_line() {
        let w = 64u32;
        let h = 64u32;
        let mut pixels = vec![0u8; (w * h) as usize];
        for y in 5..60 {
            pixels[y * w as usize + 32] = 255;
        }
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Gray8,
            color_space: crate::domain::types::ColorSpace::Srgb,
        };
        let lines = hough_lines_p(
            &pixels,
            &info,
            1.0,
            std::f32::consts::PI / 180.0,
            20,
            20,
            5,
            0,
        )
        .unwrap();
        assert!(!lines.is_empty(), "should detect vertical line");
        let has_vertical = lines.iter().any(|l| {
            let dy = (l.y2 - l.y1).abs();
            let dx = (l.x2 - l.x1).abs();
            dy > 20 && dx < 5
        });
        assert!(has_vertical, "should find a vertical line segment");
    }

    #[test]
    fn hough_vote_decrement_prevents_duplicates() {
        // A single strong line should produce exactly one segment, not many
        let w = 100u32;
        let h = 100u32;
        let mut pixels = vec![0u8; (w * h) as usize];
        for x in 10..90 {
            pixels[50 * w as usize + x] = 255;
        }
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Gray8,
            color_space: crate::domain::types::ColorSpace::Srgb,
        };
        let lines = hough_lines_p(
            &pixels,
            &info,
            1.0,
            std::f32::consts::PI / 180.0,
            30,
            30,
            5,
            0,
        )
        .unwrap();
        // PPHT with vote decrement: one strong line → one segment
        assert!(
            lines.len() <= 2,
            "vote decrement should prevent duplicates, got {} lines",
            lines.len()
        );
    }
}

