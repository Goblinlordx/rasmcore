//! Tests for morphology filters

use crate::domain::filters::common::*;

#[cfg(test)]
mod morphology_tests {
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
    fn erode_shrinks_bright_region() {
        // 8x8 image: center 4x4 white block on black background
        let mut px = vec![0u8; 64];
        for y in 2..6 {
            for x in 2..6 {
                px[y * 8 + x] = 255;
            }
        }
        let info = gray_info(8, 8);
        let result = erode(&px, &info, 3, MorphShape::Rect).unwrap();
        // Center pixel should still be white
        assert_eq!(result[3 * 8 + 3], 255);
        // Edge of original white block should be eroded
        assert_eq!(result[2 * 8 + 2], 0, "corner should be eroded");
    }

    #[test]
    fn dilate_grows_bright_region() {
        // Single white pixel at center
        let mut px = vec![0u8; 64];
        px[3 * 8 + 3] = 255;
        let info = gray_info(8, 8);
        let result = dilate(&px, &info, 3, MorphShape::Rect).unwrap();
        // 3x3 neighborhood should all be white
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let y = (3 + dy) as usize;
                let x = (3 + dx) as usize;
                assert_eq!(result[y * 8 + x], 255, "({x},{y}) should be dilated");
            }
        }
    }

    #[test]
    fn erode_dilate_identity_on_uniform() {
        let px = vec![128u8; 64];
        let info = gray_info(8, 8);
        let eroded = erode(&px, &info, 3, MorphShape::Rect).unwrap();
        let dilated = dilate(&px, &info, 3, MorphShape::Rect).unwrap();
        assert_eq!(eroded, px);
        assert_eq!(dilated, px);
    }

    #[test]
    fn open_removes_small_bright_noise() {
        // Black image with single white pixel (noise)
        let mut px = vec![0u8; 64];
        px[3 * 8 + 3] = 255;
        let info = gray_info(8, 8);
        let result = morph_open(&px, &info, 3, MorphShape::Rect).unwrap();
        // Opening should remove the single bright pixel
        assert_eq!(
            result[3 * 8 + 3],
            0,
            "single bright pixel removed by opening"
        );
    }

    #[test]
    fn close_fills_small_dark_hole() {
        // White image with single black pixel (hole)
        let mut px = vec![255u8; 64];
        px[3 * 8 + 3] = 0;
        let info = gray_info(8, 8);
        let result = morph_close(&px, &info, 3, MorphShape::Rect).unwrap();
        // Closing should fill the single dark pixel
        assert_eq!(
            result[3 * 8 + 3],
            255,
            "single dark pixel filled by closing"
        );
    }

    #[test]
    fn gradient_highlights_edges() {
        // Step edge: left half black, right half white
        let mut px = vec![0u8; 64];
        for y in 0..8 {
            for x in 4..8 {
                px[y * 8 + x] = 255;
            }
        }
        let info = gray_info(8, 8);
        let result = morph_gradient(&px, &info, 3, MorphShape::Rect).unwrap();
        // Edge at x=3/4 should be highlighted
        assert!(
            result[3 * 8 + 3] > 0 || result[3 * 8 + 4] > 0,
            "edge should be visible"
        );
        // Interior should be zero
        assert_eq!(result[3 * 8 + 0], 0, "interior black should be zero");
        assert_eq!(result[3 * 8 + 7], 0, "interior white should be zero");
    }

    #[test]
    fn cross_structuring_element() {
        let se = make_structuring_element(MorphShape::Cross, 3, 3);
        // Cross: center row and center column
        assert!(!se[0]); // top-left
        assert!(se[1]); // top-center
        assert!(!se[2]); // top-right
        assert!(se[3]); // mid-left
        assert!(se[4]); // center
        assert!(se[5]); // mid-right
        assert!(!se[6]); // bottom-left
        assert!(se[7]); // bottom-center
        assert!(!se[8]); // bottom-right
    }

    #[test]
    fn rgb_morphology() {
        use crate::domain::types::ColorSpace;
        let mut px = vec![0u8; 8 * 8 * 3];
        let idx = (3 * 8 + 3) * 3;
        px[idx] = 255;
        px[idx + 1] = 255;
        px[idx + 2] = 255;
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let result = dilate(&px, &info, 3, MorphShape::Rect).unwrap();
        // Neighbor should be dilated
        let n_idx = (3 * 8 + 4) * 3;
        assert_eq!(result[n_idx], 255);
    }
}

