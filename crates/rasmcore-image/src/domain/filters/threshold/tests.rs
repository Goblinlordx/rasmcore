//! Tests for threshold filters

use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

#[cfg(test)]
mod threshold_tests {
    use crate::domain::types::ColorSpace;
    use super::*;

    fn gray_info(w: u32, h: u32) -> ImageInfo {
        ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        }
    }

    #[test]
    fn otsu_bimodal() {
        // Bimodal image: half black, half white
        let mut px = vec![0u8; 64 * 64];
        for i in 0..32 * 64 {
            px[i] = 20;
        }
        for i in 32 * 64..64 * 64 {
            px[i] = 220;
        }
        let info = gray_info(64, 64);
        let t = otsu_threshold(&px, &info).unwrap();
        // Otsu should find a threshold between 20 and 220
        assert!(t >= 20 && t <= 220, "otsu={t}, expected between 20-220 (inclusive)");
    }

    #[test]
    fn triangle_unimodal() {
        // Mostly dark image with a few bright pixels
        let mut px = vec![10u8; 64 * 64];
        for i in 0..100 {
            px[i] = 200;
        }
        let info = gray_info(64, 64);
        let t = triangle_threshold(&px, &info).unwrap();
        assert!(t > 0, "triangle={t}, expected > 0");
    }

    #[test]
    fn threshold_binary_basic() {
        let px = vec![50, 100, 150, 200];
        let info = gray_info(2, 2);
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(px.to_vec());
        let out = ThresholdBinaryParams {
                thresh: 120,
                max_value: 255
        }.compute(
            r,
            &mut u,
            &info,
        )
        .unwrap();
        assert_eq!(out, vec![0, 0, 255, 255]);
    }

    #[test]
    fn adaptive_mean_basic() {
        let mut px = vec![128u8; 32 * 32];
        // Make one quadrant brighter
        for y in 0..16 {
            for x in 0..16 {
                px[y * 32 + x] = 200;
            }
        }
        let info = gray_info(32, 32);
        let out = adaptive_threshold(&px, &info, 255, AdaptiveMethod::Mean, 11, 2.0).unwrap();
        assert_eq!(out.len(), 32 * 32);
        // Should produce binary output
        assert!(out.iter().all(|&v| v == 0 || v == 255));
    }

    #[test]
    fn adaptive_gaussian_basic() {
        let px = vec![128u8; 16 * 16];
        let info = gray_info(16, 16);
        let out = adaptive_threshold(&px, &info, 255, AdaptiveMethod::Gaussian, 5, 0.0).unwrap();
        assert_eq!(out.len(), 16 * 16);
        // Uniform image with C=0 → all pixels equal mean → pixel > mean is false → all 0
        // (strict greater-than, matching OpenCV behavior)
        assert!(out.iter().all(|&v| v == 0));
    }

    #[test]
    fn adaptive_rejects_even_block() {
        let px = vec![128u8; 16 * 16];
        let info = gray_info(16, 16);
        assert!(adaptive_threshold(&px, &info, 255, AdaptiveMethod::Mean, 10, 0.0).is_err());
    }
}

