//! RGB↔YCbCr conversion functions.
//!
//! All functions are parameterized by [`ColorMatrix`] for standard selection.
//! Integer-only arithmetic — no floating point in conversion hot paths.

use crate::matrix::ColorMatrix;
use crate::types::YuvImage;

/// Convert RGB8 pixels to full-resolution YCbCr (4:4:4).
///
/// Returns a [`YuvImage`] where Y, U, V planes are all `width * height` bytes.
pub fn rgb_to_ycbcr(pixels: &[u8], width: u32, height: u32, matrix: &ColorMatrix) -> YuvImage {
    let w = width as usize;
    let h = height as usize;

    let mut y_plane = vec![0u8; w * h];
    let mut u_plane = vec![0u8; w * h];
    let mut v_plane = vec![0u8; w * h];

    for i in 0..w * h {
        let r = pixels[i * 3] as i32;
        let g = pixels[i * 3 + 1] as i32;
        let b = pixels[i * 3 + 2] as i32;

        y_plane[i] = ((matrix.yr * r + matrix.yg * g + matrix.yb * b + 128) >> 8)
            .wrapping_add(matrix.y_offset) as u8;
        u_plane[i] = ((matrix.cbr * r + matrix.cbg * g + matrix.cbb * b + 128) >> 8)
            .wrapping_add(matrix.c_offset) as u8;
        v_plane[i] = ((matrix.crr * r + matrix.crg * g + matrix.crb * b + 128) >> 8)
            .wrapping_add(matrix.c_offset) as u8;
    }

    YuvImage {
        width,
        height,
        y: y_plane,
        u: u_plane,
        v: v_plane,
    }
}

/// Convert RGB8 pixels to YCbCr 4:2:0 (half-resolution chroma).
///
/// Computes full-resolution luma and 2x2-averaged chroma planes.
/// Handles odd dimensions correctly via `div_ceil`.
pub fn rgb_to_ycbcr_420(pixels: &[u8], width: u32, height: u32, matrix: &ColorMatrix) -> YuvImage {
    let w = width as usize;
    let h = height as usize;
    let uv_w = w.div_ceil(2);
    let uv_h = h.div_ceil(2);

    let mut y_plane = vec![0u8; w * h];
    let mut u_plane = vec![0u8; uv_w * uv_h];
    let mut v_plane = vec![0u8; uv_w * uv_h];

    // Full-resolution luma
    for row in 0..h {
        for col in 0..w {
            let i = (row * w + col) * 3;
            let r = pixels[i] as i32;
            let g = pixels[i + 1] as i32;
            let b = pixels[i + 2] as i32;
            y_plane[row * w + col] = ((matrix.yr * r + matrix.yg * g + matrix.yb * b + 128) >> 8)
                .wrapping_add(matrix.y_offset) as u8;
        }
    }

    // Half-resolution chroma (2x2 block averaging)
    for uv_row in 0..uv_h {
        for uv_col in 0..uv_w {
            let mut sum_r = 0i32;
            let mut sum_g = 0i32;
            let mut sum_b = 0i32;
            let mut count = 0i32;

            for dy in 0..2 {
                let row = uv_row * 2 + dy;
                if row >= h {
                    continue;
                }
                for dx in 0..2 {
                    let col = uv_col * 2 + dx;
                    if col >= w {
                        continue;
                    }
                    let i = (row * w + col) * 3;
                    sum_r += pixels[i] as i32;
                    sum_g += pixels[i + 1] as i32;
                    sum_b += pixels[i + 2] as i32;
                    count += 1;
                }
            }

            let r = sum_r / count;
            let g = sum_g / count;
            let b = sum_b / count;

            let idx = uv_row * uv_w + uv_col;
            u_plane[idx] = ((matrix.cbr * r + matrix.cbg * g + matrix.cbb * b + 128) >> 8)
                .wrapping_add(matrix.c_offset) as u8;
            v_plane[idx] = ((matrix.crr * r + matrix.crg * g + matrix.crb * b + 128) >> 8)
                .wrapping_add(matrix.c_offset) as u8;
        }
    }

    YuvImage {
        width,
        height,
        y: y_plane,
        u: u_plane,
        v: v_plane,
    }
}

/// Convert RGBA8 pixels to YCbCr 4:2:0 (alpha discarded).
pub fn rgba_to_ycbcr_420(pixels: &[u8], width: u32, height: u32, matrix: &ColorMatrix) -> YuvImage {
    let w = width as usize;
    let h = height as usize;
    let mut rgb = Vec::with_capacity(w * h * 3);
    for chunk in pixels.chunks_exact(4) {
        rgb.push(chunk[0]);
        rgb.push(chunk[1]);
        rgb.push(chunk[2]);
    }
    rgb_to_ycbcr_420(&rgb, width, height, matrix)
}

/// Convert grayscale (Gray8) pixels to luma-only Y plane.
///
/// For grayscale JPEG, no chroma planes are needed.
/// Applies the luma offset (16 for studio range).
pub fn gray_to_y(pixels: &[u8], width: u32, height: u32, matrix: &ColorMatrix) -> YuvImage {
    let w = width as usize;
    let h = height as usize;

    // For grayscale: Y = (yg_sum * gray + 128) >> 8 + y_offset
    // where yg_sum = yr + yg + yb (should be ~220 for all standards)
    let ysum = matrix.yr + matrix.yg + matrix.yb;
    let y_plane: Vec<u8> = pixels
        .iter()
        .map(|&g| ((ysum * g as i32 + 128) >> 8).wrapping_add(matrix.y_offset) as u8)
        .collect();

    YuvImage {
        width,
        height,
        y: y_plane,
        u: vec![matrix.c_offset as u8; w.div_ceil(2) * h.div_ceil(2)],
        v: vec![matrix.c_offset as u8; w.div_ceil(2) * h.div_ceil(2)],
    }
}

/// Convert YCbCr to RGB8 (inverse conversion).
///
/// Takes separate Y, Cb, Cr values for a single pixel and returns (R, G, B).
/// All planes must be at the same resolution (upsample chroma first if 4:2:0).
#[inline]
pub fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8, matrix: &ColorMatrix) -> (u8, u8, u8) {
    let y_prime = y as i32 - matrix.y_offset;
    let cb_prime = cb as i32 - matrix.c_offset;
    let cr_prime = cr as i32 - matrix.c_offset;

    let r = (matrix.inv_y_scale * y_prime + matrix.inv_cr_r * cr_prime + 128) >> 8;
    let g = (matrix.inv_y_scale * y_prime
        + matrix.inv_cb_g * cb_prime
        + matrix.inv_cr_g * cr_prime
        + 128)
        >> 8;
    let b = (matrix.inv_y_scale * y_prime + matrix.inv_cb_b * cb_prime + 128) >> 8;

    (
        r.clamp(0, 255) as u8,
        g.clamp(0, 255) as u8,
        b.clamp(0, 255) as u8,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── BT.601 reference values ──

    #[test]
    fn bt601_black_pixel() {
        let pixels = [0u8, 0, 0];
        let yuv = rgb_to_ycbcr_420(&pixels, 1, 1, &ColorMatrix::BT601);
        assert_eq!(yuv.y[0], 16, "BT.601 black Y should be 16");
        assert_eq!(yuv.u[0], 128, "BT.601 black Cb should be 128");
        assert_eq!(yuv.v[0], 128, "BT.601 black Cr should be 128");
    }

    #[test]
    fn bt601_white_pixel() {
        let pixels = [255u8, 255, 255];
        let yuv = rgb_to_ycbcr_420(&pixels, 1, 1, &ColorMatrix::BT601);
        assert_eq!(yuv.y[0], 235, "BT.601 white Y should be 235");
        assert_eq!(yuv.u[0], 128, "BT.601 white Cb should be 128");
        assert_eq!(yuv.v[0], 128, "BT.601 white Cr should be 128");
    }

    #[test]
    fn bt601_pure_red() {
        let pixels = [255u8, 0, 0];
        let yuv = rgb_to_ycbcr_420(&pixels, 1, 1, &ColorMatrix::BT601);
        // Red: high Cr, low Y
        assert!(yuv.v[0] > 200, "red Cr should be high, got {}", yuv.v[0]);
    }

    // ── BT.709 reference values ──

    #[test]
    fn bt709_black_pixel() {
        let pixels = [0u8, 0, 0];
        let yuv = rgb_to_ycbcr_420(&pixels, 1, 1, &ColorMatrix::BT709);
        assert_eq!(yuv.y[0], 16, "BT.709 black Y should be 16");
        assert_eq!(yuv.u[0], 128, "BT.709 black Cb should be 128");
        assert_eq!(yuv.v[0], 128, "BT.709 black Cr should be 128");
    }

    #[test]
    fn bt709_white_pixel() {
        let pixels = [255u8, 255, 255];
        let yuv = rgb_to_ycbcr_420(&pixels, 1, 1, &ColorMatrix::BT709);
        // BT.709 white: Y should be near 235 (yr+yg+yb = 47+157+16 = 220, 220*255>>8 + 16 ≈ 235)
        let y = yuv.y[0];
        assert!(
            (233..=237).contains(&y),
            "BT.709 white Y should be ~235, got {y}"
        );
    }

    // ── Roundtrip tests ──

    #[test]
    fn bt601_roundtrip() {
        roundtrip_test(&ColorMatrix::BT601, "BT.601");
    }

    #[test]
    fn bt709_roundtrip() {
        roundtrip_test(&ColorMatrix::BT709, "BT.709");
    }

    #[test]
    fn bt2020_roundtrip() {
        roundtrip_test(&ColorMatrix::BT2020, "BT.2020");
    }

    fn roundtrip_test(matrix: &ColorMatrix, name: &str) {
        // Test with a range of RGB values
        let test_values: Vec<(u8, u8, u8)> = vec![
            (0, 0, 0),
            (255, 255, 255),
            (128, 128, 128),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (100, 150, 200),
            (50, 100, 50),
        ];

        for &(r, g, b) in &test_values {
            let pixels = [r, g, b];
            let yuv = rgb_to_ycbcr(&pixels, 1, 1, matrix);
            let (rr, rg, rb) = ycbcr_to_rgb(yuv.y[0], yuv.u[0], yuv.v[0], matrix);

            let dr = (r as i32 - rr as i32).abs();
            let dg = (g as i32 - rg as i32).abs();
            let db = (b as i32 - rb as i32).abs();

            assert!(
                dr <= 2 && dg <= 2 && db <= 2,
                "{name} roundtrip ({r},{g},{b}) → ({rr},{rg},{rb}), diff=({dr},{dg},{db})"
            );
        }
    }

    // ── Dimension tests ──

    #[test]
    fn dimensions_420() {
        let pixels = vec![128u8; 16 * 16 * 3];
        let yuv = rgb_to_ycbcr_420(&pixels, 16, 16, &ColorMatrix::BT601);
        assert_eq!(yuv.y.len(), 256);
        assert_eq!(yuv.u.len(), 64);
        assert_eq!(yuv.v.len(), 64);
    }

    #[test]
    fn dimensions_444() {
        let pixels = vec![128u8; 8 * 8 * 3];
        let yuv = rgb_to_ycbcr(&pixels, 8, 8, &ColorMatrix::BT601);
        assert_eq!(yuv.y.len(), 64);
        assert_eq!(yuv.u.len(), 64);
        assert_eq!(yuv.v.len(), 64);
    }

    #[test]
    fn odd_dimensions_1x1() {
        let pixels = [100u8, 150, 200];
        let yuv = rgb_to_ycbcr_420(&pixels, 1, 1, &ColorMatrix::BT601);
        assert_eq!(yuv.y.len(), 1);
        assert_eq!(yuv.u.len(), 1);
        assert_eq!(yuv.v.len(), 1);
    }

    #[test]
    fn odd_dimensions_1x2() {
        let pixels = [100u8, 150, 200, 50, 100, 50];
        let yuv = rgb_to_ycbcr_420(&pixels, 1, 2, &ColorMatrix::BT601);
        assert_eq!(yuv.y.len(), 2);
        assert_eq!(yuv.u.len(), 1); // ceil(1/2) * ceil(2/2)
        assert_eq!(yuv.v.len(), 1);
    }

    #[test]
    fn odd_dimensions_3x5() {
        let pixels = vec![128u8; 3 * 5 * 3];
        let yuv = rgb_to_ycbcr_420(&pixels, 3, 5, &ColorMatrix::BT601);
        assert_eq!(yuv.y.len(), 15);
        assert_eq!(yuv.u.len(), 2 * 3); // ceil(3/2) * ceil(5/2)
    }

    // ── RGBA ──

    #[test]
    fn rgba_strips_alpha() {
        let pixels = vec![128u8; 4 * 4 * 4]; // RGBA
        let yuv = rgba_to_ycbcr_420(&pixels, 4, 4, &ColorMatrix::BT601);
        assert_eq!(yuv.y.len(), 16);
        assert_eq!(yuv.u.len(), 4);
    }

    // ── Gray ──

    #[test]
    fn gray_to_y_black() {
        let pixels = [0u8];
        let yuv = gray_to_y(&pixels, 1, 1, &ColorMatrix::BT601);
        assert_eq!(yuv.y[0], 16, "gray black should map to Y=16");
        assert_eq!(yuv.u[0], 128, "gray Cb should be neutral");
    }

    #[test]
    fn gray_to_y_white() {
        let pixels = [255u8];
        let yuv = gray_to_y(&pixels, 1, 1, &ColorMatrix::BT601);
        assert_eq!(yuv.y[0], 235, "gray white should map to Y=235");
    }

    // ── Integer-only verification ──

    #[test]
    fn deterministic_conversion() {
        let pixels: Vec<u8> = (0..48).map(|i| (i * 17 % 256) as u8).collect();
        let yuv1 = rgb_to_ycbcr_420(&pixels, 4, 4, &ColorMatrix::BT601);
        let yuv2 = rgb_to_ycbcr_420(&pixels, 4, 4, &ColorMatrix::BT601);
        assert_eq!(yuv1.y, yuv2.y, "conversion must be deterministic");
        assert_eq!(yuv1.u, yuv2.u);
        assert_eq!(yuv1.v, yuv2.v);
    }
}
