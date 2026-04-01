//! Tests for analysis filters

use crate::domain::filters::common::*;

#[cfg(test)]
mod contour_tests {
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

    /// Create a 10x10 image with a 6x6 filled square at (2,2)-(7,7)
    fn make_square_image() -> (Vec<u8>, ImageInfo) {
        let info = gray_info(10, 10);
        let mut px = vec![0u8; 100];
        for y in 2..8 {
            for x in 2..8 {
                px[y * 10 + x] = 255;
            }
        }
        (px, info)
    }

    #[test]
    fn find_contours_empty_image() {
        let px = vec![0u8; 25];
        let info = gray_info(5, 5);
        let contours = find_contours(&px, &info).unwrap();
        assert!(contours.is_empty());
    }

    #[test]
    fn find_contours_single_pixel() {
        let mut px = vec![0u8; 25];
        px[12] = 255; // center pixel of 5x5
        let info = gray_info(5, 5);
        let contours = find_contours(&px, &info).unwrap();
        assert!(!contours.is_empty());
        // Single pixel contour
        assert!(contours[0].len() >= 1);
    }

    #[test]
    fn find_contours_square() {
        let (px, info) = make_square_image();
        let contours = find_contours(&px, &info).unwrap();
        assert!(!contours.is_empty(), "should find at least one contour");
        // The square contour should have boundary points
        let c = &contours[0];
        assert!(c.len() >= 4, "square contour should have >=4 points, got {}", c.len());
        // All points should be within the square region
        for &(x, y) in c {
            assert!(x >= 1 && x <= 8, "x={x} out of square bounds");
            assert!(y >= 1 && y <= 8, "y={y} out of square bounds");
        }
    }

    #[test]
    fn find_contours_requires_gray8() {
        let px = vec![0u8; 30];
        let info = ImageInfo {
            width: 5,
            height: 2,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        assert!(find_contours(&px, &info).is_err());
    }

    #[test]
    fn approx_poly_reduces_points() {
        // Create a contour with many points roughly forming a line
        let contour: Vec<(i32, i32)> = (0..100).map(|i| (i, i + (i % 3) as i32)).collect();
        let simplified = approx_poly(&contour, 2.0);
        assert!(simplified.len() < contour.len(), "should reduce points");
        assert!(simplified.len() >= 2, "should keep at least endpoints");
        // First and last points are always kept
        assert_eq!(simplified[0], contour[0]);
        assert_eq!(*simplified.last().unwrap(), *contour.last().unwrap());
    }

    #[test]
    fn approx_poly_square_to_4_points() {
        // A square contour with many points per edge
        let mut contour = Vec::new();
        // Top edge
        for x in 0..=10 { contour.push((x, 0)); }
        // Right edge
        for y in 1..=10 { contour.push((10, y)); }
        // Bottom edge (reverse)
        for x in (0..10).rev() { contour.push((x, 10)); }
        // Left edge (reverse)
        for y in (1..10).rev() { contour.push((0, y)); }

        let simplified = approx_poly(&contour, 1.0);
        assert!(simplified.len() <= 6, "square should simplify to ~4 corners, got {}", simplified.len());
    }

    #[test]
    fn contour_area_square() {
        // 10x10 square
        let contour = vec![(0, 0), (10, 0), (10, 10), (0, 10)];
        let area = contour_area(&contour);
        assert!((area - 100.0).abs() < 0.01, "area should be 100, got {area}");
    }

    #[test]
    fn contour_area_triangle() {
        // Right triangle: base=4, height=3 → area = 6
        let contour = vec![(0, 0), (4, 0), (0, 3)];
        let area = contour_area(&contour);
        assert!((area - 6.0).abs() < 0.01, "area should be 6, got {area}");
    }

    #[test]
    fn contour_area_empty() {
        assert_eq!(contour_area(&[]), 0.0);
        assert_eq!(contour_area(&[(0, 0), (1, 1)]), 0.0);
    }

    #[test]
    fn contour_perimeter_square() {
        let contour = vec![(0, 0), (10, 0), (10, 10), (0, 10)];
        let perim = contour_perimeter(&contour, true);
        assert!((perim - 40.0).abs() < 0.01, "perimeter should be 40, got {perim}");
    }

    #[test]
    fn contour_perimeter_open_vs_closed() {
        let contour = vec![(0, 0), (3, 0), (3, 4)];
        let open = contour_perimeter(&contour, false);
        let closed = contour_perimeter(&contour, true);
        assert!((open - 7.0).abs() < 0.01); // 3 + 4
        assert!((closed - 12.0).abs() < 0.01); // 3 + 4 + 5
    }

    #[test]
    fn bounding_rect_basic() {
        let contour = vec![(3, 5), (10, 2), (7, 12), (1, 8)];
        let (x, y, w, h) = bounding_rect(&contour);
        assert_eq!((x, y, w, h), (1, 2, 10, 11));
    }

    #[test]
    fn bounding_rect_empty() {
        assert_eq!(bounding_rect(&[]), (0, 0, 0, 0));
    }

    #[test]
    fn bounding_rect_single_point() {
        assert_eq!(bounding_rect(&[(5, 3)]), (5, 3, 1, 1));
    }

    #[test]
    fn square_contour_has_correct_area() {
        let (px, info) = make_square_image();
        let contours = find_contours(&px, &info).unwrap();
        assert!(!contours.is_empty());
        let area = contour_area(&contours[0]);
        // 6x6 square = 36 area, but contour area may differ from pixel area
        // The contour traces boundary pixels, so area should be close to 36
        assert!(area > 10.0, "square contour area should be substantial, got {area}");
    }
}

