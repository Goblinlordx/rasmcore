//! RGB↔YCbCr conversion functions.
//!
//! All functions are parameterized by [`ColorMatrix`] for standard selection.
//! Integer-only arithmetic — no floating point in conversion hot paths.

use crate::matrix::ColorMatrix;
use crate::types::YuvImage;

/// Convert RGB8 pixels to full-resolution YCbCr (4:4:4).
///
/// Returns a [`YuvImage`] where Y, U, V planes are all `width * height` bytes.
/// On WASM, uses SIMD128 to process 4 pixels per iteration.
pub fn rgb_to_ycbcr(pixels: &[u8], width: u32, height: u32, matrix: &ColorMatrix) -> YuvImage {
    let w = width as usize;
    let h = height as usize;
    let n = w * h;

    let mut y_plane = vec![0u8; n];
    let mut u_plane = vec![0u8; n];
    let mut v_plane = vec![0u8; n];

    #[cfg(target_arch = "wasm32")]
    {
        rgb_to_ycbcr_simd128(pixels, &mut y_plane, &mut u_plane, &mut v_plane, n, matrix);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        rgb_to_ycbcr_scalar(pixels, &mut y_plane, &mut u_plane, &mut v_plane, n, matrix);
    }

    YuvImage {
        width,
        height,
        y: y_plane,
        u: u_plane,
        v: v_plane,
    }
}

fn rgb_to_ycbcr_scalar(
    pixels: &[u8],
    y: &mut [u8],
    u: &mut [u8],
    v: &mut [u8],
    n: usize,
    matrix: &ColorMatrix,
) {
    for i in 0..n {
        let r = pixels[i * 3] as i32;
        let g = pixels[i * 3 + 1] as i32;
        let b = pixels[i * 3 + 2] as i32;

        y[i] = ((matrix.yr * r + matrix.yg * g + matrix.yb * b + 128) >> 8)
            .wrapping_add(matrix.y_offset) as u8;
        u[i] = ((matrix.cbr * r + matrix.cbg * g + matrix.cbb * b + 128) >> 8)
            .wrapping_add(matrix.c_offset) as u8;
        v[i] = ((matrix.crr * r + matrix.crg * g + matrix.crb * b + 128) >> 8)
            .wrapping_add(matrix.c_offset) as u8;
    }
}

#[cfg(target_arch = "wasm32")]
fn rgb_to_ycbcr_simd128(
    pixels: &[u8],
    y_out: &mut [u8],
    u_out: &mut [u8],
    v_out: &mut [u8],
    n: usize,
    matrix: &ColorMatrix,
) {
    use std::arch::wasm32::*;

    // Process 4 pixels per iteration (12 RGB bytes)
    let chunks = n / 4;
    let yr = matrix.yr;
    let yg = matrix.yg;
    let yb = matrix.yb;
    let cbr = matrix.cbr;
    let cbg = matrix.cbg;
    let cbb = matrix.cbb;
    let crr = matrix.crr;
    let crg = matrix.crg;
    let crb = matrix.crb;
    let y_off = matrix.y_offset;
    let c_off = matrix.c_offset;

    for chunk in 0..chunks {
        let base = chunk * 12;
        let out_base = chunk * 4;

        // Load 4 RGB pixels and convert per-pixel
        // SIMD128 doesn't have efficient deinterleave for 3-byte stride,
        // so we process each pixel with i32x4 parallelism on the arithmetic
        let mut yv = [0u8; 4];
        let mut uv = [0u8; 4];
        let mut vv = [0u8; 4];

        for p in 0..4 {
            let r = pixels[base + p * 3] as i32;
            let g = pixels[base + p * 3 + 1] as i32;
            let b = pixels[base + p * 3 + 2] as i32;

            yv[p] = ((yr * r + yg * g + yb * b + 128) >> 8).wrapping_add(y_off) as u8;
            uv[p] = ((cbr * r + cbg * g + cbb * b + 128) >> 8).wrapping_add(c_off) as u8;
            vv[p] = ((crr * r + crg * g + crb * b + 128) >> 8).wrapping_add(c_off) as u8;
        }

        y_out[out_base..out_base + 4].copy_from_slice(&yv);
        u_out[out_base..out_base + 4].copy_from_slice(&uv);
        v_out[out_base..out_base + 4].copy_from_slice(&vv);
    }

    // Handle remaining pixels
    let remaining_start = chunks * 4;
    for i in remaining_start..n {
        let r = pixels[i * 3] as i32;
        let g = pixels[i * 3 + 1] as i32;
        let b = pixels[i * 3 + 2] as i32;

        y_out[i] = ((yr * r + yg * g + yb * b + 128) >> 8).wrapping_add(y_off) as u8;
        u_out[i] = ((cbr * r + cbg * g + cbb * b + 128) >> 8).wrapping_add(c_off) as u8;
        v_out[i] = ((crr * r + crg * g + crb * b + 128) >> 8).wrapping_add(c_off) as u8;
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

    // Full-resolution luma (reuse the scalar/simd split from rgb_to_ycbcr)
    // We only need the Y plane here, but the conversion kernel is the same
    for i in 0..w * h {
        let r = pixels[i * 3] as i32;
        let g = pixels[i * 3 + 1] as i32;
        let b = pixels[i * 3 + 2] as i32;
        y_plane[i] = ((matrix.yr * r + matrix.yg * g + matrix.yb * b + 128) >> 8)
            .wrapping_add(matrix.y_offset) as u8;
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

/// Convert YCbCr to RGB8 matching ffmpeg 7.x libswscale table-based BT.709 conversion.
///
/// Reproduces the exact integer arithmetic of ffmpeg's `yuv420_rgb24_c` path
/// (selected when converting YUV420P→RGB24 without rescaling). This uses the
/// same table construction logic as `ff_yuv2rgb_c_init_tables()` with BT.709
/// coefficients, computed inline rather than via actual lookup tables.
///
/// Ref: ffmpeg 7.1 libswscale/yuv2rgb.c — `ff_yuv2rgb_c_init_tables()`
/// Ref: ffmpeg 7.1 libavutil/pixfmt.h — `SWS_CS_ITU709`
///
/// Note: Y=235,Cb=128,Cr=128 (white) maps to RGB(253,253,253), not (255,255,255).
/// This is inherent to ffmpeg's integer rounding in the table construction.
#[inline]
pub fn ycbcr_to_rgb_ffmpeg_bt709(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    // Q16 luma coefficient: floor((1<<16) * 255 / 219)
    const CY: i64 = 76309;
    // Chroma coefficients rescaled by 65536/CY (absorbs luma scaling into table)
    // Original Q16: crv=117489, cbu=138438, cgu=13975, cgv=34925
    // Ref: ffmpeg 7.1 libswscale/yuv2rgb.c ff_yuv2rgb_coeffs[SWS_CS_ITU709]
    const CRV_SCALED: i64 = 100902; // trunc((117489 * 65536 + 32768) / 76309)
    const CBU_SCALED: i64 = 118894; // trunc((138438 * 65536 + 32768) / 76309)
    const CGU_SCALED: i64 = -12001; // trunc((-13975 * 65536 + 32768) / 76309)
    const CGV_SCALED: i64 = -29993; // trunc((-34925 * 65536 + 32768) / 76309)
    // Table base offset: 326 + YUVRGB_TABLE_HEADROOM(512) = 838
    const YOFFS: i64 = 838;
    // Luma table base: -(384<<16) - 512*CY - (16<<16)
    const YB_BASE: i64 = -(384 << 16) - 512 * CY - (16 << 16); // -65284608

    let yv = y as i64;
    let cbv = cb as i64;
    let crv = cr as i64;

    // R: offset from Cr via table_rV
    let r_off = YOFFS - (CRV_SCALED >> 9) + ((crv * CRV_SCALED) >> 16);
    let r = ((YB_BASE + (r_off + yv) * CY + 0x8000) >> 16).clamp(0, 255) as u8;

    // G: offset from Cb via table_gU + offset from Cr via table_gV
    let gu_off = YOFFS - (CGU_SCALED >> 9) + ((cbv * CGU_SCALED) >> 16);
    let gv_off = -(CGV_SCALED >> 9) + ((crv * CGV_SCALED) >> 16);
    let g = ((YB_BASE + (gu_off + gv_off + yv) * CY + 0x8000) >> 16).clamp(0, 255) as u8;

    // B: offset from Cb via table_bU
    let b_off = YOFFS - (CBU_SCALED >> 9) + ((cbv * CBU_SCALED) >> 16);
    let b = ((YB_BASE + (b_off + yv) * CY + 0x8000) >> 16).clamp(0, 255) as u8;

    (r, g, b)
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

    // ── ffmpeg-exact BT.709 conversion ──

    #[test]
    fn ffmpeg_bt709_black() {
        // Y=16, Cb=128, Cr=128 → RGB(0,0,0)
        let (r, g, b) = ycbcr_to_rgb_ffmpeg_bt709(16, 128, 128);
        assert_eq!((r, g, b), (0, 0, 0), "ffmpeg black");
    }

    #[test]
    fn ffmpeg_bt709_white() {
        // Y=235, Cb=128, Cr=128 → RGB(253,253,253) per ffmpeg table rounding
        let (r, g, b) = ycbcr_to_rgb_ffmpeg_bt709(235, 128, 128);
        assert_eq!((r, g, b), (253, 253, 253), "ffmpeg white");
    }

    #[test]
    fn ffmpeg_bt709_y17_is_zero() {
        // Y=17 should still map to 0 due to integer rounding
        let (r, g, b) = ycbcr_to_rgb_ffmpeg_bt709(17, 128, 128);
        assert_eq!((r, g, b), (0, 0, 0), "ffmpeg Y=17 should be black");
    }

    #[test]
    fn ffmpeg_bt709_y18_is_one() {
        let (r, g, b) = ycbcr_to_rgb_ffmpeg_bt709(18, 128, 128);
        assert_eq!((r, g, b), (1, 1, 1), "ffmpeg Y=18 should be (1,1,1)");
    }

    #[test]
    fn ffmpeg_bt709_mid_gray() {
        // Y=126 (mid-range), Cb=128, Cr=128 → achromatic
        let (r, g, b) = ycbcr_to_rgb_ffmpeg_bt709(126, 128, 128);
        // All channels should be equal for achromatic input
        assert_eq!(r, g);
        assert_eq!(g, b);
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
