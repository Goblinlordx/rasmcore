use super::super::error::ImageError;
use super::super::types::{DecodedImage, ImageInfo, PixelFormat, Rotation};
use super::{bytes_per_pixel, validate_pixel_buffer};

// ─── Block-transpose rotation helpers ──────────────────────────────────────

/// Tile size for cache-friendly block transpose rotation.
const TILE: usize = 8;

/// Rotate 90° CW using 8×8 block transpose (scalar).
///
/// For source pixel (sx, sy) the destination is:
///   dst_row = sx, dst_col = h - 1 - sy  (output dimensions: h_out = w, w_out = h)
#[allow(dead_code)]
fn rotate_90_blocked_scalar(pixels: &[u8], w: usize, h: usize, bpp: usize) -> Vec<u8> {
    let ow = h; // output width
    let mut result = vec![0u8; w * h * bpp];

    for ty in (0..h).step_by(TILE) {
        let tile_h = TILE.min(h - ty);
        for tx in (0..w).step_by(TILE) {
            let tile_w = TILE.min(w - tx);
            for dy in 0..tile_h {
                let sy = ty + dy;
                for dx in 0..tile_w {
                    let sx = tx + dx;
                    let src_off = (sy * w + sx) * bpp;
                    let dst_off = (sx * ow + (h - 1 - sy)) * bpp;
                    result[dst_off..dst_off + bpp].copy_from_slice(&pixels[src_off..src_off + bpp]);
                }
            }
        }
    }
    result
}

/// Rotate 270° CW using 8×8 block transpose (scalar).
///
/// For source pixel (sx, sy) the destination is:
///   dst_row = w - 1 - sx, dst_col = sy  (output dimensions: h_out = w, w_out = h)
#[allow(dead_code)]
fn rotate_270_blocked_scalar(pixels: &[u8], w: usize, h: usize, bpp: usize) -> Vec<u8> {
    let ow = h; // output width
    let mut result = vec![0u8; w * h * bpp];

    for ty in (0..h).step_by(TILE) {
        let tile_h = TILE.min(h - ty);
        for tx in (0..w).step_by(TILE) {
            let tile_w = TILE.min(w - tx);
            for dy in 0..tile_h {
                let sy = ty + dy;
                for dx in 0..tile_w {
                    let sx = tx + dx;
                    let src_off = (sy * w + sx) * bpp;
                    let dst_off = ((w - 1 - sx) * ow + sy) * bpp;
                    result[dst_off..dst_off + bpp].copy_from_slice(&pixels[src_off..src_off + bpp]);
                }
            }
        }
    }
    result
}

/// WASM SIMD128 accelerated 90° CW rotation for RGBA8 (bpp=4).
///
/// Processes full 8×8 tiles using v128 loads and an in-register transpose
/// via the standard unpacklo/unpackhi pattern at 32-bit granularity.
/// Falls back to scalar for edge tiles not divisible by 8.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn rotate_90_simd_rgba(pixels: &[u8], w: usize, h: usize) -> Vec<u8> {
    use core::arch::wasm32::*;

    const BPP: usize = 4;
    let ow = h;
    let mut result = vec![0u8; w * h * BPP];

    for ty in (0..h).step_by(TILE) {
        for tx in (0..w).step_by(TILE) {
            let tile_h = TILE.min(h - ty);
            let tile_w = TILE.min(w - tx);

            // Full 8×8 SIMD path
            if tile_h == TILE && tile_w == TILE {
                // Load 8 rows of 8 RGBA pixels (each row = 2 v128)
                let mut rows_lo: [v128; 8] = [i32x4_splat(0); 8];
                let mut rows_hi: [v128; 8] = [i32x4_splat(0); 8];
                for r in 0..8 {
                    let sy = ty + r;
                    let off = (sy * w + tx) * BPP;
                    // SAFETY: off and off+16 are within pixel buffer bounds
                    // (8×8 tile within w×h image, BPP=4, 32 bytes per row).
                    rows_lo[r] = unsafe { v128_load(pixels.as_ptr().add(off) as *const v128) };
                    rows_hi[r] = unsafe { v128_load(pixels.as_ptr().add(off + 16) as *const v128) };
                }

                // Transpose 8×8 i32 matrix laid out as 8 rows of (lo, hi) v128.
                // Phase 1: interleave adjacent pairs at 32-bit granularity.
                macro_rules! unpack_lo_i32 {
                    ($a:expr, $b:expr) => {
                        i32x4_shuffle::<0, 4, 1, 5>($a, $b)
                    };
                }
                macro_rules! unpack_hi_i32 {
                    ($a:expr, $b:expr) => {
                        i32x4_shuffle::<2, 6, 3, 7>($a, $b)
                    };
                }

                // Step 1: interleave row pairs – lo halves
                let a0 = unpack_lo_i32!(rows_lo[0], rows_lo[1]);
                let a1 = unpack_hi_i32!(rows_lo[0], rows_lo[1]);
                let a2 = unpack_lo_i32!(rows_lo[2], rows_lo[3]);
                let a3 = unpack_hi_i32!(rows_lo[2], rows_lo[3]);
                let a4 = unpack_lo_i32!(rows_lo[4], rows_lo[5]);
                let a5 = unpack_hi_i32!(rows_lo[4], rows_lo[5]);
                let a6 = unpack_lo_i32!(rows_lo[6], rows_lo[7]);
                let a7 = unpack_hi_i32!(rows_lo[6], rows_lo[7]);

                // hi halves
                let b0 = unpack_lo_i32!(rows_hi[0], rows_hi[1]);
                let b1 = unpack_hi_i32!(rows_hi[0], rows_hi[1]);
                let b2 = unpack_lo_i32!(rows_hi[2], rows_hi[3]);
                let b3 = unpack_hi_i32!(rows_hi[2], rows_hi[3]);
                let b4 = unpack_lo_i32!(rows_hi[4], rows_hi[5]);
                let b5 = unpack_hi_i32!(rows_hi[4], rows_hi[5]);
                let b6 = unpack_lo_i32!(rows_hi[6], rows_hi[7]);
                let b7 = unpack_hi_i32!(rows_hi[6], rows_hi[7]);

                // Step 2: interleave quad groups at 64-bit granularity
                macro_rules! unpack_lo_i64 {
                    ($a:expr, $b:expr) => {
                        i64x2_shuffle::<0, 2>($a, $b)
                    };
                }
                macro_rules! unpack_hi_i64 {
                    ($a:expr, $b:expr) => {
                        i64x2_shuffle::<1, 3>($a, $b)
                    };
                }

                // Columns 0-3 from lo halves
                let c0 = unpack_lo_i64!(a0, a2);
                let c1 = unpack_hi_i64!(a0, a2);
                let c2 = unpack_lo_i64!(a1, a3);
                let c3 = unpack_hi_i64!(a1, a3);
                let c0b = unpack_lo_i64!(a4, a6);
                let c1b = unpack_hi_i64!(a4, a6);
                let c2b = unpack_lo_i64!(a5, a7);
                let c3b = unpack_hi_i64!(a5, a7);

                // Columns 4-7 from hi halves
                let c4 = unpack_lo_i64!(b0, b2);
                let c5 = unpack_hi_i64!(b0, b2);
                let c6 = unpack_lo_i64!(b1, b3);
                let c7 = unpack_hi_i64!(b1, b3);
                let c4b = unpack_lo_i64!(b4, b6);
                let c5b = unpack_hi_i64!(b4, b6);
                let c6b = unpack_lo_i64!(b5, b7);
                let c7b = unpack_hi_i64!(b5, b7);

                // For rotate_90: dst_row = sx, dst_col = h-1-sy.
                // Column `col` → output row tx+col. The column holds rows ty..ty+7.
                // dst_col for row r = h-1-(ty+r), reversed order.
                macro_rules! reverse_v128_i32 {
                    ($v:expr) => {
                        i32x4_shuffle::<3, 2, 1, 0>($v, $v)
                    };
                }

                let cols: [(v128, v128); 8] = [
                    (c0, c0b),
                    (c1, c1b),
                    (c2, c2b),
                    (c3, c3b),
                    (c4, c4b),
                    (c5, c5b),
                    (c6, c6b),
                    (c7, c7b),
                ];

                for (col_idx, &(lo, hi)) in cols.iter().enumerate() {
                    let dst_row = tx + col_idx;
                    let dst_col_start = h - 1 - ty - 7;
                    let dst_off = (dst_row * ow + dst_col_start) * BPP;
                    // Write reversed: hi reversed first, then lo reversed
                    let rev_hi = reverse_v128_i32!(hi);
                    let rev_lo = reverse_v128_i32!(lo);
                    // SAFETY: dst_off within result buffer bounds (tile-aligned output).
                    unsafe {
                        v128_store(result.as_mut_ptr().add(dst_off) as *mut v128, rev_hi);
                        v128_store(result.as_mut_ptr().add(dst_off + 16) as *mut v128, rev_lo);
                    }
                }
            } else {
                // Edge tile: scalar fallback
                for dy in 0..tile_h {
                    let sy = ty + dy;
                    for dx in 0..tile_w {
                        let sx = tx + dx;
                        let src_off = (sy * w + sx) * BPP;
                        let dst_off = (sx * ow + (h - 1 - sy)) * BPP;
                        result[dst_off..dst_off + BPP]
                            .copy_from_slice(&pixels[src_off..src_off + BPP]);
                    }
                }
            }
        }
    }
    result
}

/// WASM SIMD128 accelerated 270° CW rotation for RGBA8 (bpp=4).
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn rotate_270_simd_rgba(pixels: &[u8], w: usize, h: usize) -> Vec<u8> {
    use core::arch::wasm32::*;

    const BPP: usize = 4;
    let ow = h;
    let mut result = vec![0u8; w * h * BPP];

    for ty in (0..h).step_by(TILE) {
        for tx in (0..w).step_by(TILE) {
            let tile_h = TILE.min(h - ty);
            let tile_w = TILE.min(w - tx);

            if tile_h == TILE && tile_w == TILE {
                let mut rows_lo: [v128; 8] = [i32x4_splat(0); 8];
                let mut rows_hi: [v128; 8] = [i32x4_splat(0); 8];
                for r in 0..8 {
                    let sy = ty + r;
                    let off = (sy * w + tx) * BPP;
                    // SAFETY: 8×8 tile within bounds, 32 bytes per row.
                    rows_lo[r] = unsafe { v128_load(pixels.as_ptr().add(off) as *const v128) };
                    rows_hi[r] = unsafe { v128_load(pixels.as_ptr().add(off + 16) as *const v128) };
                }

                macro_rules! unpack_lo_i32 {
                    ($a:expr, $b:expr) => {
                        i32x4_shuffle::<0, 4, 1, 5>($a, $b)
                    };
                }
                macro_rules! unpack_hi_i32 {
                    ($a:expr, $b:expr) => {
                        i32x4_shuffle::<2, 6, 3, 7>($a, $b)
                    };
                }
                macro_rules! unpack_lo_i64 {
                    ($a:expr, $b:expr) => {
                        i64x2_shuffle::<0, 2>($a, $b)
                    };
                }
                macro_rules! unpack_hi_i64 {
                    ($a:expr, $b:expr) => {
                        i64x2_shuffle::<1, 3>($a, $b)
                    };
                }

                let a0 = unpack_lo_i32!(rows_lo[0], rows_lo[1]);
                let a1 = unpack_hi_i32!(rows_lo[0], rows_lo[1]);
                let a2 = unpack_lo_i32!(rows_lo[2], rows_lo[3]);
                let a3 = unpack_hi_i32!(rows_lo[2], rows_lo[3]);
                let a4 = unpack_lo_i32!(rows_lo[4], rows_lo[5]);
                let a5 = unpack_hi_i32!(rows_lo[4], rows_lo[5]);
                let a6 = unpack_lo_i32!(rows_lo[6], rows_lo[7]);
                let a7 = unpack_hi_i32!(rows_lo[6], rows_lo[7]);

                let b0 = unpack_lo_i32!(rows_hi[0], rows_hi[1]);
                let b1 = unpack_hi_i32!(rows_hi[0], rows_hi[1]);
                let b2 = unpack_lo_i32!(rows_hi[2], rows_hi[3]);
                let b3 = unpack_hi_i32!(rows_hi[2], rows_hi[3]);
                let b4 = unpack_lo_i32!(rows_hi[4], rows_hi[5]);
                let b5 = unpack_hi_i32!(rows_hi[4], rows_hi[5]);
                let b6 = unpack_lo_i32!(rows_hi[6], rows_hi[7]);
                let b7 = unpack_hi_i32!(rows_hi[6], rows_hi[7]);

                let c0 = unpack_lo_i64!(a0, a2);
                let c1 = unpack_hi_i64!(a0, a2);
                let c2 = unpack_lo_i64!(a1, a3);
                let c3 = unpack_hi_i64!(a1, a3);
                let c0b = unpack_lo_i64!(a4, a6);
                let c1b = unpack_hi_i64!(a4, a6);
                let c2b = unpack_lo_i64!(a5, a7);
                let c3b = unpack_hi_i64!(a5, a7);

                let c4 = unpack_lo_i64!(b0, b2);
                let c5 = unpack_hi_i64!(b0, b2);
                let c6 = unpack_lo_i64!(b1, b3);
                let c7 = unpack_hi_i64!(b1, b3);
                let c4b = unpack_lo_i64!(b4, b6);
                let c5b = unpack_hi_i64!(b4, b6);
                let c6b = unpack_lo_i64!(b5, b7);
                let c7b = unpack_hi_i64!(b5, b7);

                // For rotate_270: dst_row = w-1-sx, dst_col = sy.
                // Column `col` → output row w-1-(tx+col). Row order preserved.
                let cols: [(v128, v128); 8] = [
                    (c0, c0b),
                    (c1, c1b),
                    (c2, c2b),
                    (c3, c3b),
                    (c4, c4b),
                    (c5, c5b),
                    (c6, c6b),
                    (c7, c7b),
                ];

                for (col_idx, &(lo, hi)) in cols.iter().enumerate() {
                    let dst_row = w - 1 - (tx + col_idx);
                    let dst_col_start = ty;
                    let dst_off = (dst_row * ow + dst_col_start) * BPP;
                    // SAFETY: dst_off within output buffer bounds (transposed tile).
                    unsafe {
                        v128_store(result.as_mut_ptr().add(dst_off) as *mut v128, lo);
                        v128_store(result.as_mut_ptr().add(dst_off + 16) as *mut v128, hi);
                    }
                }
            } else {
                for dy in 0..tile_h {
                    let sy = ty + dy;
                    for dx in 0..tile_w {
                        let sx = tx + dx;
                        let src_off = (sy * w + sx) * BPP;
                        let dst_off = ((w - 1 - sx) * ow + sy) * BPP;
                        result[dst_off..dst_off + BPP]
                            .copy_from_slice(&pixels[src_off..src_off + BPP]);
                    }
                }
            }
        }
    }
    result
}

/// Dispatch rotate-90 to SIMD or scalar block transpose.
fn rotate_90_blocked(pixels: &[u8], w: usize, h: usize, bpp: usize) -> Vec<u8> {
    #[cfg(target_arch = "wasm32")]
    {
        if bpp == 4 {
            // SAFETY: WASM SIMD128 is available when targeting wasm32 with simd128.
            return unsafe { rotate_90_simd_rgba(pixels, w, h) };
        }
    }
    rotate_90_blocked_scalar(pixels, w, h, bpp)
}

/// Dispatch rotate-270 to SIMD or scalar block transpose.
fn rotate_270_blocked(pixels: &[u8], w: usize, h: usize, bpp: usize) -> Vec<u8> {
    #[cfg(target_arch = "wasm32")]
    {
        if bpp == 4 {
            return unsafe { rotate_270_simd_rgba(pixels, w, h) };
        }
    }
    rotate_270_blocked_scalar(pixels, w, h, bpp)
}

/// Rotate an image by 90, 180, or 270 degrees using raw buffer ops.
///
/// R90 and R270 use cache-friendly 8×8 block transpose with optional
/// WASM SIMD128 acceleration for RGBA8.
pub fn rotate(
    pixels: &[u8],
    info: &ImageInfo,
    degrees: Rotation,
) -> Result<DecodedImage, ImageError> {
    let bpp = bytes_per_pixel(info.format)?;
    let (w, h) = (info.width as usize, info.height as usize);

    match degrees {
        Rotation::R90 => {
            let result = rotate_90_blocked(pixels, w, h, bpp);
            Ok(DecodedImage {
                pixels: result,
                info: ImageInfo {
                    width: info.height,
                    height: info.width,
                    format: info.format,
                    color_space: info.color_space,
                },
                icc_profile: None,
            })
        }
        Rotation::R180 => {
            // 180°: reverse pixel order
            let mut result = vec![0u8; pixels.len()];
            let total = w * h;
            for i in 0..total {
                let src_off = i * bpp;
                let dst_off = (total - 1 - i) * bpp;
                result[dst_off..dst_off + bpp].copy_from_slice(&pixels[src_off..src_off + bpp]);
            }
            Ok(DecodedImage {
                pixels: result,
                info: info.clone(),
                icc_profile: None,
            })
        }
        Rotation::R270 => {
            let result = rotate_270_blocked(pixels, w, h, bpp);
            Ok(DecodedImage {
                pixels: result,
                info: ImageInfo {
                    width: info.height,
                    height: info.width,
                    format: info.format,
                    color_space: info.color_space,
                },
                icc_profile: None,
            })
        }
    }
}

/// Rotate an image by an arbitrary angle (degrees) with bilinear interpolation.
///
/// The output dimensions are the bounding box of the rotated corners.
/// Uncovered regions are filled with `bg_color` (RGB or RGBA depending on format).
pub fn rotate_arbitrary(
    pixels: &[u8],
    info: &ImageInfo,
    degrees: f64,
    bg_color: &[u8],
) -> Result<DecodedImage, ImageError> {
    let bpp = bytes_per_pixel(info.format)?;
    let (w, h) = (info.width as usize, info.height as usize);
    validate_pixel_buffer(pixels, info, bpp)?;

    let rad = degrees.to_radians();
    let cos_a = rad.cos();
    let sin_a = rad.sin();

    // Compute bounding box of rotated corners
    let cx = w as f64 / 2.0;
    let cy = h as f64 / 2.0;
    let corners = [(-cx, -cy), (cx, -cy), (-cx, cy), (cx, cy)];
    let mut min_x = f64::MAX;
    let mut max_x = f64::MIN;
    let mut min_y = f64::MAX;
    let mut max_y = f64::MIN;
    for &(x, y) in &corners {
        let rx = x * cos_a - y * sin_a;
        let ry = x * sin_a + y * cos_a;
        min_x = min_x.min(rx);
        max_x = max_x.max(rx);
        min_y = min_y.min(ry);
        max_y = max_y.max(ry);
    }

    let out_w = (max_x - min_x).ceil() as usize;
    let out_h = (max_y - min_y).ceil() as usize;
    if out_w == 0 || out_h == 0 {
        return Err(ImageError::InvalidParameters(
            "rotation produced zero-size output".into(),
        ));
    }

    let out_cx = out_w as f64 / 2.0;
    let out_cy = out_h as f64 / 2.0;

    let mut output = vec![0u8; out_w * out_h * bpp];

    // For each output pixel, inverse-map to source and sample
    for oy in 0..out_h {
        for ox in 0..out_w {
            let dx = ox as f64 - out_cx;
            let dy = oy as f64 - out_cy;
            // Inverse rotation: rotate by -angle
            let sx = dx * cos_a + dy * sin_a + cx;
            let sy = -dx * sin_a + dy * cos_a + cy;

            let out_idx = (oy * out_w + ox) * bpp;

            if sx >= 0.0 && sx < (w - 1) as f64 && sy >= 0.0 && sy < (h - 1) as f64 {
                // Bilinear interpolation
                let x0 = sx.floor() as usize;
                let y0 = sy.floor() as usize;
                let x1 = x0 + 1;
                let y1 = y0 + 1;
                let fx = sx - x0 as f64;
                let fy = sy - y0 as f64;

                let is_16 = matches!(
                    info.format,
                    PixelFormat::Rgb16 | PixelFormat::Rgba16 | PixelFormat::Gray16
                );
                if is_16 {
                    // 16-bit: each channel is 2 bytes LE
                    let bytes_per_chan = 2;
                    let channels = bpp / 2;
                    for c in 0..channels {
                        let off = c * bytes_per_chan;
                        let read16 = |y: usize, x: usize| -> f64 {
                            let idx = y * w * bpp + x * bpp + off;
                            u16::from_le_bytes([pixels[idx], pixels[idx + 1]]) as f64
                        };
                        let val = read16(y0, x0) * (1.0 - fx) * (1.0 - fy)
                            + read16(y0, x1) * fx * (1.0 - fy)
                            + read16(y1, x0) * (1.0 - fx) * fy
                            + read16(y1, x1) * fx * fy;
                        let v16 = val.round().clamp(0.0, 65535.0) as u16;
                        let bytes = v16.to_le_bytes();
                        output[out_idx + off] = bytes[0];
                        output[out_idx + off + 1] = bytes[1];
                    }
                } else {
                    // 8-bit: each channel is 1 byte
                    for c in 0..bpp {
                        let p00 = pixels[y0 * w * bpp + x0 * bpp + c] as f64;
                        let p10 = pixels[y0 * w * bpp + x1 * bpp + c] as f64;
                        let p01 = pixels[y1 * w * bpp + x0 * bpp + c] as f64;
                        let p11 = pixels[y1 * w * bpp + x1 * bpp + c] as f64;

                        let val = p00 * (1.0 - fx) * (1.0 - fy)
                            + p10 * fx * (1.0 - fy)
                            + p01 * (1.0 - fx) * fy
                            + p11 * fx * fy;
                        output[out_idx + c] = val.round().clamp(0.0, 255.0) as u8;
                    }
                }
            } else {
                // Background fill
                for c in 0..bpp {
                    output[out_idx + c] = bg_color.get(c).copied().unwrap_or(0);
                }
            }
        }
    }

    Ok(DecodedImage {
        pixels: output,
        info: ImageInfo {
            width: out_w as u32,
            height: out_h as u32,
            format: info.format,
            color_space: info.color_space,
        },
        icc_profile: None,
    })
}
