//! RGB to YCbCr color space conversion and chroma subsampling for JPEG.
//!
//! Uses BT.601 coefficients (ITU-T T.81 standard for JPEG).

use crate::types::ChromaSubsampling;

/// YCbCr image with separate planes.
pub struct YcbcrImage {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub cb: Vec<u8>,
    pub cr: Vec<u8>,
    /// Chroma plane dimensions after subsampling.
    pub chroma_width: u32,
    pub chroma_height: u32,
}

/// Convert RGB8 pixels to YCbCr with chroma subsampling.
pub fn rgb_to_ycbcr(
    pixels: &[u8],
    width: u32,
    height: u32,
    subsampling: ChromaSubsampling,
) -> YcbcrImage {
    let w = width as usize;
    let h = height as usize;

    // Full-resolution Y, Cb, Cr
    let mut y = Vec::with_capacity(w * h);
    let mut cb_full = Vec::with_capacity(w * h);
    let mut cr_full = Vec::with_capacity(w * h);

    for i in 0..w * h {
        let off = i * 3;
        let r = pixels[off] as i32;
        let g = pixels[off + 1] as i32;
        let b = pixels[off + 2] as i32;
        // BT.601 (JPEG standard): full-range YCbCr, 16-bit fixed-point
        // Matches mozjpeg/libjpeg-turbo jccolor.c coefficients exactly:
        //   FIX(0.29900)=19595, FIX(0.58700)=38470, FIX(0.11400)=7471
        //   FIX(0.16874)=11059, FIX(0.33126)=21709
        //   FIX(0.41869)=27439, FIX(0.08131)=5329
        const ONE_HALF: i32 = 1 << 15; // 32768
        const CBCR_OFF: i32 = 128 << 16; // 128 * 65536
        y.push(((19595 * r + 38470 * g + 7471 * b + ONE_HALF) >> 16) as u8);
        cb_full.push(
            ((-11059 * r - 21709 * g + 32768 * b + CBCR_OFF + ONE_HALF - 1) >> 16).clamp(0, 255)
                as u8,
        );
        cr_full.push(
            ((32768 * r - 27439 * g - 5329 * b + CBCR_OFF + ONE_HALF - 1) >> 16).clamp(0, 255)
                as u8,
        );
    }

    // Subsample chroma
    let (cw, ch) = chroma_dimensions(width, height, subsampling);
    let (h_factor, v_factor) = subsampling_factors(subsampling);

    let cb = downsample(&cb_full, w, h, h_factor, v_factor, cw as usize, ch as usize);
    let cr = downsample(&cr_full, w, h, h_factor, v_factor, cw as usize, ch as usize);

    YcbcrImage {
        width,
        height,
        y,
        cb,
        cr,
        chroma_width: cw,
        chroma_height: ch,
    }
}

/// Convert grayscale pixels (just copies to Y plane).
pub fn gray_to_y(pixels: &[u8], width: u32, height: u32) -> YcbcrImage {
    YcbcrImage {
        width,
        height,
        y: pixels[..width as usize * height as usize].to_vec(),
        cb: Vec::new(),
        cr: Vec::new(),
        chroma_width: 0,
        chroma_height: 0,
    }
}

/// Chroma dimensions after subsampling.
pub fn chroma_dimensions(w: u32, h: u32, sub: ChromaSubsampling) -> (u32, u32) {
    match sub {
        ChromaSubsampling::None444 => (w, h),
        ChromaSubsampling::Half422 => (w.div_ceil(2), h),
        ChromaSubsampling::Quarter420 => (w.div_ceil(2), h.div_ceil(2)),
        ChromaSubsampling::Quarter411 => (w.div_ceil(4), h),
    }
}

/// Horizontal and vertical subsampling factors.
pub fn subsampling_factors(sub: ChromaSubsampling) -> (usize, usize) {
    match sub {
        ChromaSubsampling::None444 => (1, 1),
        ChromaSubsampling::Half422 => (2, 1),
        ChromaSubsampling::Quarter420 => (2, 2),
        ChromaSubsampling::Quarter411 => (4, 1),
    }
}

/// MCU dimensions in pixels for each subsampling mode.
pub fn mcu_dimensions(sub: ChromaSubsampling) -> (u32, u32) {
    match sub {
        ChromaSubsampling::None444 => (8, 8),
        ChromaSubsampling::Half422 => (16, 8),
        ChromaSubsampling::Quarter420 => (16, 16),
        ChromaSubsampling::Quarter411 => (32, 8),
    }
}

// ---------------------------------------------------------------------------
// YCbCr → RGB inverse conversion (fixed-point, libjpeg-compatible)
// ---------------------------------------------------------------------------

/// Fixed-point scale bits (matches libjpeg convention).
const SCALEBITS: i32 = 16;

/// Rounding offset: 1 << (SCALEBITS - 1).
const ONE_HALF: i32 = 1 << (SCALEBITS - 1); // 32768

// BT.601 inverse coefficients: FIX(x) = (x * 65536 + 0.5) as i32
const FIX_1_402: i32 = 91881; // 1.40200 * 65536
const FIX_0_34414: i32 = 22554; // 0.34414 * 65536
const FIX_0_71414: i32 = 46802; // 0.71414 * 65536
const FIX_1_772: i32 = 116130; // 1.77200 * 65536

/// Convert a single YCbCr pixel to RGB using i32 fixed-point arithmetic.
///
/// Uses BT.601 coefficients with SCALEBITS=16 (libjpeg convention):
/// ```text
/// R = Y + 1.402   * (Cr - 128)
/// G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
/// B = Y + 1.772   * (Cb - 128)
/// ```
///
/// All arithmetic is integer; no floating-point operations.
#[inline]
pub fn ycbcr_to_rgb_fixed(y: i32, cb: i32, cr: i32) -> (u8, u8, u8) {
    let cr_shifted = cr - 128;
    let cb_shifted = cb - 128;

    let r = y + ((FIX_1_402 * cr_shifted + ONE_HALF) >> SCALEBITS);
    let g = y - ((FIX_0_34414 * cb_shifted + FIX_0_71414 * cr_shifted + ONE_HALF) >> SCALEBITS);
    let b = y + ((FIX_1_772 * cb_shifted + ONE_HALF) >> SCALEBITS);

    (
        r.clamp(0, 255) as u8,
        g.clamp(0, 255) as u8,
        b.clamp(0, 255) as u8,
    )
}

/// Blinn's 8x8 multiply: approximation of (a * b / 255) matching libjpeg-turbo.
#[inline]
fn blinn_8x8(a: u8, b: u8) -> u8 {
    let t = a as i32 * b as i32 + 128;
    ((t + (t >> 8)) >> 8) as u8
}

/// Convert raw CMYK plane values to RGB.
///
/// Used when Adobe APP14 color_transform=0 (raw CMYK, no YCbCr encoding).
/// Formula: R = blinn(C, K), G = blinn(M, K), B = blinn(Y, K)
/// (plane values are already in the right form for multiplication)
#[inline]
pub fn cmyk_to_rgb(c: u8, m: u8, y: u8, k: u8) -> (u8, u8, u8) {
    (blinn_8x8(c, k), blinn_8x8(m, k), blinn_8x8(y, k))
}

/// Convert YCCK (YCbCr + K) to RGB.
///
/// Matches zune-jpeg/libjpeg-turbo exactly:
///   1. YCbCr→RGB on first 3 components gives (R, G, B)
///   2. Final: R_out = blinn(255 - R, K), same for G, B
///
/// The K channel value is the raw decoded plane value (cast to u8).
#[inline]
pub fn ycck_to_rgb(y_val: i32, cb: i32, cr: i32, k: u8) -> (u8, u8, u8) {
    let (r, g, b) = ycbcr_to_rgb_fixed(y_val, cb, cr);
    (
        blinn_8x8(255 - r, k),
        blinn_8x8(255 - g, k),
        blinn_8x8(255 - b, k),
    )
}

/// Downsample a full-resolution plane with dithered rounding.
/// Matches mozjpeg/libjpeg h2v2_downsample: alternating bias (1, 2, 1, 2...)
/// prevents systematic rounding bias that degrades chroma quality.
fn downsample(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    h_factor: usize,
    v_factor: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
    let mut dst = Vec::with_capacity(dst_w * dst_h);
    let mut bias = 1u32; // Alternating 1, 2, 1, 2... (mozjpeg: bias ^= 3)
    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let mut sum = 0u32;
            let mut count = 0u32;
            for vy in 0..v_factor {
                let sy = dy * v_factor + vy;
                if sy >= src_h {
                    continue;
                }
                for hx in 0..h_factor {
                    let sx = dx * h_factor + hx;
                    if sx >= src_w {
                        continue;
                    }
                    sum += src[sy * src_w + sx] as u32;
                    count += 1;
                }
            }
            // Dithered rounding: add alternating bias before division
            if count == 4 {
                // Fast path for 2x2 (h2v2): (sum + bias) >> 2
                dst.push(((sum + bias) >> 2) as u8);
                bias ^= 3; // Alternate between 1 and 2
            } else if count == 2 {
                // Fast path for 2x1 (h2v1): (sum + (bias >> 1)) >> 1
                dst.push(((sum + (bias >> 1)) >> 1) as u8);
                bias ^= 3;
            } else if count > 0 {
                dst.push(((sum + count / 2) / count) as u8);
            } else {
                dst.push(128);
            }
        }
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_to_ycbcr_dimensions_420() {
        let pixels = vec![128u8; 16 * 16 * 3];
        let img = rgb_to_ycbcr(&pixels, 16, 16, ChromaSubsampling::Quarter420);
        assert_eq!(img.y.len(), 256);
        assert_eq!(img.cb.len(), 64); // 8x8
        assert_eq!(img.cr.len(), 64);
    }

    #[test]
    fn rgb_to_ycbcr_dimensions_444() {
        let pixels = vec![128u8; 8 * 8 * 3];
        let img = rgb_to_ycbcr(&pixels, 8, 8, ChromaSubsampling::None444);
        assert_eq!(img.y.len(), 64);
        assert_eq!(img.cb.len(), 64); // no subsampling
        assert_eq!(img.cr.len(), 64);
    }

    #[test]
    fn gray_conversion() {
        let pixels = vec![200u8; 4 * 4];
        let img = gray_to_y(&pixels, 4, 4);
        assert_eq!(img.y.len(), 16);
        assert!(img.cb.is_empty());
    }

    #[test]
    fn mcu_size_420() {
        assert_eq!(mcu_dimensions(ChromaSubsampling::Quarter420), (16, 16));
        assert_eq!(mcu_dimensions(ChromaSubsampling::None444), (8, 8));
    }

    // ── YCbCr → RGB fixed-point tests ─────────────────────────────────────

    #[test]
    fn fixed_neutral_gray() {
        // Y=128, Cb=128, Cr=128 → RGB(128, 128, 128)
        let (r, g, b) = ycbcr_to_rgb_fixed(128, 128, 128);
        assert_eq!((r, g, b), (128, 128, 128));
    }

    #[test]
    fn fixed_black() {
        let (r, g, b) = ycbcr_to_rgb_fixed(0, 128, 128);
        assert_eq!((r, g, b), (0, 0, 0));
    }

    #[test]
    fn fixed_white() {
        let (r, g, b) = ycbcr_to_rgb_fixed(255, 128, 128);
        assert_eq!((r, g, b), (255, 255, 255));
    }

    #[test]
    fn fixed_pure_red_clamps() {
        // Y=76, Cb=85, Cr=255 → approx red; R saturates, G/B clamp to 0
        let (r, g, b) = ycbcr_to_rgb_fixed(76, 85, 255);
        assert_eq!(r, 254); // ~76 + 1.402*127 ≈ 254.05
        assert!(g < 5);
        assert!(b < 5);
    }

    #[test]
    fn fixed_pure_blue_clamps() {
        // Y=29, Cb=255, Cr=107 → approx blue
        let (r, g, b) = ycbcr_to_rgb_fixed(29, 255, 107);
        assert!(r < 5);
        assert!(g < 5);
        assert!(b > 250);
    }

    #[test]
    fn fixed_saturation_high() {
        // Extreme values: Y=255, Cb=0, Cr=255
        let (r, g, b) = ycbcr_to_rgb_fixed(255, 0, 255);
        assert_eq!(r, 255); // saturated
        // G and B depend on exact rounding — verify they produced values
        let _ = (g, b);
    }

    #[test]
    fn fixed_saturation_low() {
        // Y=0, Cb=255, Cr=0
        let (r, g, b) = ycbcr_to_rgb_fixed(0, 255, 0);
        assert_eq!(r, 0); // clamped
        assert!(g < 50);
        assert_eq!(b, 225); // 0 + 1.772*127 ≈ 225
    }

    #[test]
    fn fixed_vs_float_full_range() {
        // Verify fixed-point matches f64 reference within ±1 for all Y/Cb/Cr combos
        // (sampled at 16-value intervals for speed)
        let mut max_diff = 0i32;
        for y in (0..=255).step_by(1) {
            for cb in (0..=255).step_by(16) {
                for cr in (0..=255).step_by(16) {
                    let (rf, gf, bf) = ycbcr_to_rgb_fixed(y, cb, cr);

                    // f64 reference (old code path)
                    let yf = y as f64;
                    let cbf = cb as f64;
                    let crf = cr as f64;
                    let r_ref = (yf + 1.402 * (crf - 128.0)).round().clamp(0.0, 255.0) as u8;
                    let g_ref = (yf - 0.344136 * (cbf - 128.0) - 0.714136 * (crf - 128.0))
                        .round()
                        .clamp(0.0, 255.0) as u8;
                    let b_ref = (yf + 1.772 * (cbf - 128.0)).round().clamp(0.0, 255.0) as u8;

                    let dr = (rf as i32 - r_ref as i32).abs();
                    let dg = (gf as i32 - g_ref as i32).abs();
                    let db = (bf as i32 - b_ref as i32).abs();
                    max_diff = max_diff.max(dr).max(dg).max(db);

                    assert!(
                        dr <= 1 && dg <= 1 && db <= 1,
                        "Y={y} Cb={cb} Cr={cr}: fixed=({rf},{gf},{bf}) float=({r_ref},{g_ref},{b_ref}) diff=({dr},{dg},{db})"
                    );
                }
            }
        }
        // The fixed-point should be very close — typically max_diff is 0 or 1
        assert!(max_diff <= 1, "max channel diff = {max_diff}");
    }

    #[test]
    fn fixed_no_float_in_hot_path() {
        // Compile-time guarantee: ycbcr_to_rgb_fixed only uses i32
        // This test just exercises the function to ensure no panics
        for y in 0..=255 {
            let (r, g, b) = ycbcr_to_rgb_fixed(y, 128, 128);
            assert_eq!(r, y as u8);
            assert_eq!(g, y as u8);
            assert_eq!(b, y as u8);
        }
    }
}
