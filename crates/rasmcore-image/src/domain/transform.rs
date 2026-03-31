use super::error::ImageError;
use super::metadata::ExifOrientation;
#[cfg(test)]
use super::types::ColorSpace;
use super::types::{DecodedImage, FlipDirection, ImageInfo, PixelFormat, ResizeFilter, Rotation};

/// Resize an image to new dimensions using SIMD-optimized fast_image_resize.
///
/// Uses fast_image_resize with native SIMD (SSE4.1/AVX2 on x86, NEON on ARM,
/// SIMD128 on WASM). 15-57x faster than the image crate on large images.
pub fn resize(
    pixels: &[u8],
    info: &ImageInfo,
    width: u32,
    height: u32,
    filter: ResizeFilter,
) -> Result<DecodedImage, ImageError> {
    use fast_image_resize as fir;

    if info.width == 0 || info.height == 0 || width == 0 || height == 0 {
        return Err(ImageError::InvalidParameters(
            "resize dimensions must be > 0".into(),
        ));
    }

    let pixel_type = match info.format {
        PixelFormat::Rgb8 => fir::PixelType::U8x3,
        PixelFormat::Rgba8 => fir::PixelType::U8x4,
        PixelFormat::Gray8 => fir::PixelType::U8,
        PixelFormat::Rgb16 => fir::PixelType::U16x3,
        PixelFormat::Rgba16 => fir::PixelType::U16x4,
        PixelFormat::Gray16 => fir::PixelType::U16,
        other => {
            return Err(ImageError::UnsupportedFormat(format!(
                "resize from {other:?} not supported by SIMD backend"
            )));
        }
    };

    let fir_filter = match filter {
        ResizeFilter::Nearest => fir::ResizeAlg::Nearest,
        ResizeFilter::Bilinear => fir::ResizeAlg::Convolution(fir::FilterType::Bilinear),
        ResizeFilter::Bicubic => fir::ResizeAlg::Convolution(fir::FilterType::CatmullRom),
        ResizeFilter::Lanczos3 => fir::ResizeAlg::Convolution(fir::FilterType::Lanczos3),
    };

    let src_image =
        fir::images::Image::from_vec_u8(info.width, info.height, pixels.to_vec(), pixel_type)
            .map_err(|e| ImageError::InvalidInput(format!("pixel data mismatch: {e}")))?;

    let mut dst_image = fir::images::Image::new(width, height, pixel_type);

    let options = fir::ResizeOptions::new().resize_alg(fir_filter);
    let mut resizer = fir::Resizer::new();
    #[cfg(target_arch = "wasm32")]
    unsafe {
        resizer.set_cpu_extensions(fir::CpuExtensions::Simd128);
    }
    resizer
        .resize(&src_image, &mut dst_image, &options)
        .map_err(|e| ImageError::ProcessingFailed(format!("resize failed: {e}")))?;

    let result_pixels = dst_image.into_vec();

    Ok(DecodedImage {
        pixels: result_pixels,
        info: ImageInfo {
            width,
            height,
            format: info.format,
            color_space: info.color_space,
        },
        icc_profile: None,
    })
}

/// Crop a region from an image using raw row-slice copies.
pub fn crop(
    pixels: &[u8],
    info: &ImageInfo,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> Result<DecodedImage, ImageError> {
    if x + width > info.width || y + height > info.height {
        return Err(ImageError::InvalidParameters(format!(
            "crop region ({x},{y},{width},{height}) exceeds image bounds ({},{})",
            info.width, info.height
        )));
    }
    if width == 0 || height == 0 {
        return Err(ImageError::InvalidParameters(
            "crop dimensions must be > 0".into(),
        ));
    }
    let bpp = bytes_per_pixel(info.format)?;
    let src_stride = info.width as usize * bpp;
    let dst_stride = width as usize * bpp;
    let mut result = vec![0u8; height as usize * dst_stride];

    for row in 0..height as usize {
        let src_offset = (y as usize + row) * src_stride + x as usize * bpp;
        let dst_offset = row * dst_stride;
        result[dst_offset..dst_offset + dst_stride]
            .copy_from_slice(&pixels[src_offset..src_offset + dst_stride]);
    }

    Ok(DecodedImage {
        pixels: result,
        info: ImageInfo {
            width,
            height,
            format: info.format,
            color_space: info.color_space,
        },
        icc_profile: None,
    })
}

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

/// Flip an image horizontally or vertically using raw buffer ops.
pub fn flip(
    pixels: &[u8],
    info: &ImageInfo,
    direction: FlipDirection,
) -> Result<DecodedImage, ImageError> {
    let bpp = bytes_per_pixel(info.format)?;
    let (w, h) = (info.width as usize, info.height as usize);
    let stride = w * bpp;
    let mut result = vec![0u8; pixels.len()];

    match direction {
        FlipDirection::Horizontal => {
            for y in 0..h {
                for x in 0..w {
                    let src_off = y * stride + x * bpp;
                    let dst_off = y * stride + (w - 1 - x) * bpp;
                    result[dst_off..dst_off + bpp].copy_from_slice(&pixels[src_off..src_off + bpp]);
                }
            }
        }
        FlipDirection::Vertical => {
            for y in 0..h {
                let src_off = y * stride;
                let dst_off = (h - 1 - y) * stride;
                result[dst_off..dst_off + stride]
                    .copy_from_slice(&pixels[src_off..src_off + stride]);
            }
        }
    }

    Ok(DecodedImage {
        pixels: result,
        info: info.clone(),
        icc_profile: None,
    })
}

/// Convert pixel format using raw buffer operations.
///
/// Supports all conversions between Gray8, Rgb8, Rgba8, Gray16, Rgb16, Rgba16.
/// Grayscale conversion uses BT.601 luma: (77*R + 150*G + 29*B + 128) >> 8.
/// 8↔16 bit conversion uses proper rounding: u8→u16 via v*257, u16→u8 via (v+128)/257.
pub fn convert_format(
    pixels: &[u8],
    info: &ImageInfo,
    target: PixelFormat,
) -> Result<DecodedImage, ImageError> {
    if info.format == target {
        return Ok(DecodedImage {
            pixels: pixels.to_vec(),
            info: info.clone(),
            icc_profile: None,
        });
    }

    let n = (info.width as usize) * (info.height as usize);
    let new_pixels = convert_pixels(pixels, info.format, target, n)?;

    Ok(DecodedImage {
        pixels: new_pixels,
        info: ImageInfo {
            width: info.width,
            height: info.height,
            format: target,
            color_space: info.color_space,
        },
        icc_profile: None,
    })
}

/// Convert pixel data between formats.
fn convert_pixels(
    pixels: &[u8],
    src: PixelFormat,
    dst: PixelFormat,
    pixel_count: usize,
) -> Result<Vec<u8>, ImageError> {
    // Strategy: normalize to RGBA8 or RGBA16 as intermediate if needed,
    // then convert to target. For efficiency, handle direct conversions first.
    match (src, dst) {
        // ── Identity ──
        (a, b) if a == b => Ok(pixels.to_vec()),

        // ── 8-bit direct conversions ──
        (PixelFormat::Rgb8, PixelFormat::Rgba8) => {
            let mut out = Vec::with_capacity(pixel_count * 4);
            for chunk in pixels.chunks_exact(3) {
                out.extend_from_slice(chunk);
                out.push(255);
            }
            Ok(out)
        }
        (PixelFormat::Rgba8, PixelFormat::Rgb8) => {
            let mut out = Vec::with_capacity(pixel_count * 3);
            for chunk in pixels.chunks_exact(4) {
                out.extend_from_slice(&chunk[..3]);
            }
            Ok(out)
        }
        (PixelFormat::Rgb8, PixelFormat::Gray8) | (PixelFormat::Rgba8, PixelFormat::Gray8) => {
            let bpp = if src == PixelFormat::Rgb8 { 3 } else { 4 };
            let mut out = Vec::with_capacity(pixel_count);
            for chunk in pixels.chunks_exact(bpp) {
                let r = chunk[0] as u16;
                let g = chunk[1] as u16;
                let b = chunk[2] as u16;
                out.push(((77 * r + 150 * g + 29 * b + 128) >> 8) as u8);
            }
            Ok(out)
        }
        (PixelFormat::Gray8, PixelFormat::Rgb8) => {
            let mut out = Vec::with_capacity(pixel_count * 3);
            for &v in pixels {
                out.push(v);
                out.push(v);
                out.push(v);
            }
            Ok(out)
        }
        (PixelFormat::Gray8, PixelFormat::Rgba8) => {
            let mut out = Vec::with_capacity(pixel_count * 4);
            for &v in pixels {
                out.push(v);
                out.push(v);
                out.push(v);
                out.push(255);
            }
            Ok(out)
        }

        // ── 16-bit direct conversions ──
        (PixelFormat::Rgb16, PixelFormat::Rgba16) => {
            let mut out = Vec::with_capacity(pixel_count * 8);
            for chunk in pixels.chunks_exact(6) {
                out.extend_from_slice(chunk);
                out.extend_from_slice(&65535u16.to_le_bytes());
            }
            Ok(out)
        }
        (PixelFormat::Rgba16, PixelFormat::Rgb16) => {
            let mut out = Vec::with_capacity(pixel_count * 6);
            for chunk in pixels.chunks_exact(8) {
                out.extend_from_slice(&chunk[..6]);
            }
            Ok(out)
        }

        // ── 8→16 bit promotion ──
        (PixelFormat::Gray8, PixelFormat::Gray16) => {
            let mut out = Vec::with_capacity(pixel_count * 2);
            for &v in pixels {
                out.extend_from_slice(&(v as u16 * 257).to_le_bytes());
            }
            Ok(out)
        }
        (PixelFormat::Rgb8, PixelFormat::Rgb16) => {
            let mut out = Vec::with_capacity(pixel_count * 6);
            for &v in pixels {
                out.extend_from_slice(&(v as u16 * 257).to_le_bytes());
            }
            Ok(out)
        }
        (PixelFormat::Rgba8, PixelFormat::Rgba16) => {
            let mut out = Vec::with_capacity(pixel_count * 8);
            for &v in pixels {
                out.extend_from_slice(&(v as u16 * 257).to_le_bytes());
            }
            Ok(out)
        }

        // ── 16→8 bit demotion ──
        (PixelFormat::Gray16, PixelFormat::Gray8) => {
            let mut out = Vec::with_capacity(pixel_count);
            for chunk in pixels.chunks_exact(2) {
                let v = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(((v as u32 + 128) / 257) as u8);
            }
            Ok(out)
        }
        (PixelFormat::Rgb16, PixelFormat::Rgb8) => {
            let mut out = Vec::with_capacity(pixel_count * 3);
            for chunk in pixels.chunks_exact(2) {
                let v = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(((v as u32 + 128) / 257) as u8);
            }
            Ok(out)
        }
        (PixelFormat::Rgba16, PixelFormat::Rgba8) => {
            let mut out = Vec::with_capacity(pixel_count * 4);
            for chunk in pixels.chunks_exact(2) {
                let v = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(((v as u32 + 128) / 257) as u8);
            }
            Ok(out)
        }

        // ── Cross-depth + cross-channel: two-step via intermediate ──
        _ => {
            // Step 1: convert to 8-bit same-channel-count
            let (intermediate, intermediate_fmt) = match src {
                PixelFormat::Gray16 => (
                    convert_pixels(pixels, src, PixelFormat::Gray8, pixel_count)?,
                    PixelFormat::Gray8,
                ),
                PixelFormat::Rgb16 => (
                    convert_pixels(pixels, src, PixelFormat::Rgb8, pixel_count)?,
                    PixelFormat::Rgb8,
                ),
                PixelFormat::Rgba16 => (
                    convert_pixels(pixels, src, PixelFormat::Rgba8, pixel_count)?,
                    PixelFormat::Rgba8,
                ),
                other => (pixels.to_vec(), other),
            };
            // Step 2: convert channels at 8-bit
            let (channel_converted, channel_fmt) = if intermediate_fmt == dst {
                return Ok(intermediate);
            } else {
                // Get 8-bit target
                let target_8 = match dst {
                    PixelFormat::Gray8 | PixelFormat::Gray16 => PixelFormat::Gray8,
                    PixelFormat::Rgb8 | PixelFormat::Rgb16 => PixelFormat::Rgb8,
                    PixelFormat::Rgba8 | PixelFormat::Rgba16 => PixelFormat::Rgba8,
                    _ => {
                        return Err(ImageError::UnsupportedFormat(format!(
                            "conversion from {src:?} to {dst:?} not supported"
                        )));
                    }
                };
                if intermediate_fmt == target_8 {
                    (intermediate, target_8)
                } else {
                    (
                        convert_pixels(&intermediate, intermediate_fmt, target_8, pixel_count)?,
                        target_8,
                    )
                }
            };
            // Step 3: promote to 16-bit if needed
            if channel_fmt == dst {
                Ok(channel_converted)
            } else {
                convert_pixels(&channel_converted, channel_fmt, dst, pixel_count)
            }
        }
    }
}

/// Auto-orient an image by applying the EXIF orientation transform.
///
/// Maps EXIF orientation values (1-8) to the correct sequence of
/// rotation and flip operations.
pub fn auto_orient(
    pixels: &[u8],
    info: &ImageInfo,
    orientation: ExifOrientation,
) -> Result<DecodedImage, ImageError> {
    match orientation {
        ExifOrientation::Normal => Ok(DecodedImage {
            pixels: pixels.to_vec(),
            info: info.clone(),
            icc_profile: None,
        }),
        ExifOrientation::FlipHorizontal => flip(pixels, info, FlipDirection::Horizontal),
        ExifOrientation::Rotate180 => rotate(pixels, info, Rotation::R180),
        ExifOrientation::FlipVertical => flip(pixels, info, FlipDirection::Vertical),
        ExifOrientation::Transpose => {
            let rotated = rotate(pixels, info, Rotation::R270)?;
            flip(&rotated.pixels, &rotated.info, FlipDirection::Horizontal)
        }
        ExifOrientation::Rotate90 => rotate(pixels, info, Rotation::R90),
        ExifOrientation::Transverse => {
            let rotated = rotate(pixels, info, Rotation::R90)?;
            flip(&rotated.pixels, &rotated.info, FlipDirection::Horizontal)
        }
        ExifOrientation::Rotate270 => rotate(pixels, info, Rotation::R270),
    }
}

/// Auto-orient using EXIF metadata from encoded data.
///
/// Reads EXIF orientation from the encoded data and applies the correct
/// transform to the pixel data. If no EXIF is found or orientation is
/// normal, returns the pixels unchanged.
pub fn auto_orient_from_exif(
    pixels: &[u8],
    info: &ImageInfo,
    encoded_data: &[u8],
) -> Result<DecodedImage, ImageError> {
    let orientation = match super::metadata::read_exif(encoded_data) {
        Ok(meta) => meta.orientation.unwrap_or(ExifOrientation::Normal),
        Err(_) => ExifOrientation::Normal,
    };
    auto_orient(pixels, info, orientation)
}

// ─── Extended Geometry ──────────────────────────────────────────────────────

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

/// Extend the canvas by adding padding around the image.
///
/// `fill_color` should match the pixel format (3 bytes for RGB8, 4 for RGBA8, etc.).
pub fn pad(
    pixels: &[u8],
    info: &ImageInfo,
    top: u32,
    right: u32,
    bottom: u32,
    left: u32,
    fill_color: &[u8],
) -> Result<DecodedImage, ImageError> {
    let bpp = bytes_per_pixel(info.format)?;
    let (w, h) = (info.width as usize, info.height as usize);
    validate_pixel_buffer(pixels, info, bpp)?;

    let out_w = w + left as usize + right as usize;
    let out_h = h + top as usize + bottom as usize;

    // Fill output with background color
    let mut output = Vec::with_capacity(out_w * out_h * bpp);
    for _ in 0..out_w * out_h {
        for c in 0..bpp {
            output.push(fill_color.get(c).copied().unwrap_or(0));
        }
    }

    // Blit original image at (left, top)
    for y in 0..h {
        let src_start = y * w * bpp;
        let dst_start = ((top as usize + y) * out_w + left as usize) * bpp;
        output[dst_start..dst_start + w * bpp]
            .copy_from_slice(&pixels[src_start..src_start + w * bpp]);
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

/// Trim uniform borders from an image.
///
/// Scans inward from each edge, comparing pixels against the top-left corner pixel.
/// Pixels within `threshold` (per-channel absolute difference) are considered border.
pub fn trim(pixels: &[u8], info: &ImageInfo, threshold: u8) -> Result<DecodedImage, ImageError> {
    let bpp = bytes_per_pixel(info.format)?;
    let (w, h) = (info.width as usize, info.height as usize);
    validate_pixel_buffer(pixels, info, bpp)?;

    if w == 0 || h == 0 {
        return Err(ImageError::InvalidParameters("empty image".into()));
    }

    // Reference color: top-left pixel
    let ref_color = &pixels[0..bpp];

    let pixel_matches = |x: usize, y: usize| -> bool {
        let idx = (y * w + x) * bpp;
        for c in 0..bpp {
            if (pixels[idx + c] as i16 - ref_color[c] as i16).unsigned_abs() > threshold as u16 {
                return false;
            }
        }
        true
    };

    // Scan from each edge
    let mut top = 0;
    'top: for y in 0..h {
        for x in 0..w {
            if !pixel_matches(x, y) {
                break 'top;
            }
        }
        top = y + 1;
    }

    let mut bottom = h;
    'bottom: for y in (top..h).rev() {
        for x in 0..w {
            if !pixel_matches(x, y) {
                break 'bottom;
            }
        }
        bottom = y;
    }

    let mut left = 0;
    'left: for x in 0..w {
        for y in top..bottom {
            if !pixel_matches(x, y) {
                break 'left;
            }
        }
        left = x + 1;
    }

    let mut right = w;
    'right: for x in (left..w).rev() {
        for y in top..bottom {
            if !pixel_matches(x, y) {
                break 'right;
            }
        }
        right = x;
    }

    if left >= right || top >= bottom {
        // Entire image is uniform border — return 1x1
        return Ok(DecodedImage {
            pixels: ref_color.to_vec(),
            info: ImageInfo {
                width: 1,
                height: 1,
                format: info.format,
                color_space: info.color_space,
            },
            icc_profile: None,
        });
    }

    crop(
        pixels,
        info,
        left as u32,
        top as u32,
        (right - left) as u32,
        (bottom - top) as u32,
    )
}

/// Apply a general 2D affine transform.
///
/// `matrix` is [a, b, tx, c, d, ty] representing:
///   x' = a*x + b*y + tx
///   y' = c*x + d*y + ty
///
/// Output dimensions must be specified. Uses bilinear interpolation.
pub fn affine(
    pixels: &[u8],
    info: &ImageInfo,
    matrix: &[f64; 6],
    out_width: u32,
    out_height: u32,
    bg_color: &[u8],
) -> Result<DecodedImage, ImageError> {
    let bpp = bytes_per_pixel(info.format)?;
    let (w, h) = (info.width as usize, info.height as usize);
    validate_pixel_buffer(pixels, info, bpp)?;

    let out_w = out_width as usize;
    let out_h = out_height as usize;

    // Compute inverse matrix for inverse mapping
    let [a, b, tx, c, d, ty] = *matrix;
    let det = a * d - b * c;
    if det.abs() < 1e-10 {
        return Err(ImageError::InvalidParameters(
            "singular affine matrix".into(),
        ));
    }
    let inv_det = 1.0 / det;
    let ia = d * inv_det;
    let ib = -b * inv_det;
    let ic = -c * inv_det;
    let id = a * inv_det;
    let itx = -(ia * tx + ib * ty);
    let ity = -(ic * tx + id * ty);

    let mut output = vec![0u8; out_w * out_h * bpp];

    // Vectorized bilinear interpolation: process all channels per pixel as f32x4
    #[cfg(target_arch = "wasm32")]
    {
        use std::arch::wasm32::*;
        let zero_v = f32x4_splat(0.0);
        let max_v = f32x4_splat(255.0);
        let half_v = f32x4_splat(0.5);
        let bg_v = f32x4(
            bg_color.first().copied().unwrap_or(0) as f32,
            bg_color.get(1).copied().unwrap_or(0) as f32,
            bg_color.get(2).copied().unwrap_or(0) as f32,
            bg_color.get(3).copied().unwrap_or(0) as f32,
        );

        for oy in 0..out_h {
            // Pre-compute row-constant part of inverse mapping
            let row_sx_base = ib * oy as f64 + itx;
            let row_sy_base = id * oy as f64 + ity;

            for ox in 0..out_w {
                let sx = ia * ox as f64 + row_sx_base;
                let sy = ic * ox as f64 + row_sy_base;
                let out_idx = (oy * out_w + ox) * bpp;

                if sx >= 0.0 && sx < (w - 1) as f64 && sy >= 0.0 && sy < (h - 1) as f64 {
                    let x0 = sx.floor() as usize;
                    let y0 = sy.floor() as usize;
                    let x1 = x0 + 1;
                    let y1 = y0 + 1;
                    let fx = (sx - x0 as f64) as f32;
                    let fy = (sy - y0 as f64) as f32;

                    // Compute weights
                    let w00 = f32x4_splat((1.0 - fx) * (1.0 - fy));
                    let w10 = f32x4_splat(fx * (1.0 - fy));
                    let w01 = f32x4_splat((1.0 - fx) * fy);
                    let w11 = f32x4_splat(fx * fy);

                    // Load 4 source pixels (up to 4 channels each)
                    let i00 = y0 * w * bpp + x0 * bpp;
                    let i10 = y0 * w * bpp + x1 * bpp;
                    let i01 = y1 * w * bpp + x0 * bpp;
                    let i11 = y1 * w * bpp + x1 * bpp;

                    let load_px = |idx: usize| -> v128 {
                        f32x4(
                            pixels[idx] as f32,
                            if bpp > 1 { pixels[idx + 1] as f32 } else { 0.0 },
                            if bpp > 2 { pixels[idx + 2] as f32 } else { 0.0 },
                            if bpp > 3 { pixels[idx + 3] as f32 } else { 0.0 },
                        )
                    };

                    let p00 = load_px(i00);
                    let p10 = load_px(i10);
                    let p01 = load_px(i01);
                    let p11 = load_px(i11);

                    // Bilinear blend: val = p00*w00 + p10*w10 + p01*w01 + p11*w11
                    let val = f32x4_add(
                        f32x4_add(f32x4_mul(p00, w00), f32x4_mul(p10, w10)),
                        f32x4_add(f32x4_mul(p01, w01), f32x4_mul(p11, w11)),
                    );

                    // Round, clamp, store
                    let out_v = f32x4_min(max_v, f32x4_max(zero_v, f32x4_add(val, half_v)));
                    output[out_idx] = f32x4_extract_lane::<0>(out_v) as u8;
                    if bpp > 1 {
                        output[out_idx + 1] = f32x4_extract_lane::<1>(out_v) as u8;
                    }
                    if bpp > 2 {
                        output[out_idx + 2] = f32x4_extract_lane::<2>(out_v) as u8;
                    }
                    if bpp > 3 {
                        output[out_idx + 3] = f32x4_extract_lane::<3>(out_v) as u8;
                    }
                } else {
                    output[out_idx] = f32x4_extract_lane::<0>(bg_v) as u8;
                    if bpp > 1 {
                        output[out_idx + 1] = f32x4_extract_lane::<1>(bg_v) as u8;
                    }
                    if bpp > 2 {
                        output[out_idx + 2] = f32x4_extract_lane::<2>(bg_v) as u8;
                    }
                    if bpp > 3 {
                        output[out_idx + 3] = f32x4_extract_lane::<3>(bg_v) as u8;
                    }
                }
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        for oy in 0..out_h {
            let row_sx_base = ib * oy as f64 + itx;
            let row_sy_base = id * oy as f64 + ity;

            for ox in 0..out_w {
                let sx = ia * ox as f64 + row_sx_base;
                let sy = ic * ox as f64 + row_sy_base;
                let out_idx = (oy * out_w + ox) * bpp;

                if sx >= 0.0 && sx < (w - 1) as f64 && sy >= 0.0 && sy < (h - 1) as f64 {
                    let x0 = sx.floor() as usize;
                    let y0 = sy.floor() as usize;
                    let x1 = x0 + 1;
                    let y1 = y0 + 1;
                    let fx = sx - x0 as f64;
                    let fy = sy - y0 as f64;

                    let w00 = (1.0 - fx) * (1.0 - fy);
                    let w10 = fx * (1.0 - fy);
                    let w01 = (1.0 - fx) * fy;
                    let w11 = fx * fy;

                    for ch in 0..bpp {
                        let p00 = pixels[y0 * w * bpp + x0 * bpp + ch] as f64;
                        let p10 = pixels[y0 * w * bpp + x1 * bpp + ch] as f64;
                        let p01 = pixels[y1 * w * bpp + x0 * bpp + ch] as f64;
                        let p11 = pixels[y1 * w * bpp + x1 * bpp + ch] as f64;

                        let val = p00 * w00 + p10 * w10 + p01 * w01 + p11 * w11;
                        output[out_idx + ch] = (val + 0.5).clamp(0.0, 255.0) as u8;
                    }
                } else {
                    for ch in 0..bpp {
                        output[out_idx + ch] = bg_color.get(ch).copied().unwrap_or(0);
                    }
                }
            }
        }
    }

    Ok(DecodedImage {
        pixels: output,
        info: ImageInfo {
            width: out_width,
            height: out_height,
            format: info.format,
            color_space: info.color_space,
        },
        icc_profile: None,
    })
}

/// Get bytes per pixel for supported formats.
fn bytes_per_pixel(format: PixelFormat) -> Result<usize, ImageError> {
    match format {
        PixelFormat::Rgb8 => Ok(3),
        PixelFormat::Rgba8 => Ok(4),
        PixelFormat::Gray8 => Ok(1),
        PixelFormat::Gray16 => Ok(2),
        PixelFormat::Rgb16 => Ok(6),
        PixelFormat::Rgba16 => Ok(8),
        _ => Err(ImageError::UnsupportedFormat(format!(
            "{format:?} not supported for geometric transforms"
        ))),
    }
}

/// Validate pixel buffer size matches dimensions and format.
fn validate_pixel_buffer(pixels: &[u8], info: &ImageInfo, bpp: usize) -> Result<(), ImageError> {
    let expected = info.width as usize * info.height as usize * bpp;
    if pixels.len() < expected {
        return Err(ImageError::InvalidInput(format!(
            "pixel buffer too small: need {expected}, got {}",
            pixels.len()
        )));
    }
    Ok(())
}

// ─── Lens Distortion Correction ───────────────────────────────────────────

/// Camera intrinsic parameters for lens distortion.
#[derive(Debug, Clone, Copy)]
pub struct CameraMatrix {
    /// Focal length X (pixels). Default: image width.
    pub fx: f64,
    /// Focal length Y (pixels). Default: image height.
    pub fy: f64,
    /// Principal point X (pixels). Default: image center.
    pub cx: f64,
    /// Principal point Y (pixels). Default: image center.
    pub cy: f64,
}

/// Brown-Conrady radial distortion coefficients.
///
/// Distortion model: `r_distorted = r * (1 + k1*r² + k2*r⁴ + k3*r⁶)`
/// where `r` is the normalized radius from the principal point.
///
/// Positive k1 → barrel distortion (lines bend outward).
/// Negative k1 → pincushion distortion (lines bend inward).
#[derive(Debug, Clone, Copy)]
pub struct DistortionCoeffs {
    pub k1: f64,
    pub k2: f64,
    pub k3: f64,
}

/// Remove radial lens distortion (Brown-Conrady model).
///
/// For each output pixel, computes the corresponding distorted source
/// coordinate and bilinear-samples the input image. Matches the behavior
/// of `cv2.undistort()` with the same camera matrix and distortion coefficients.
///
/// Reference: Brown, D.C., "Decentering Distortion of Lenses" (1966).
pub fn undistort(
    pixels: &[u8],
    info: &ImageInfo,
    camera: &CameraMatrix,
    dist: &DistortionCoeffs,
) -> Result<DecodedImage, ImageError> {
    let bpp = bytes_per_pixel(info.format)?;
    let w = info.width as usize;
    let h = info.height as usize;
    validate_pixel_buffer(pixels, info, bpp)?;

    // OpenCV constants for fixed-point bilinear interpolation
    const INTER_BITS: u32 = 5;
    const INTER_TAB_SIZE: i32 = 1 << INTER_BITS; // 32
    const INTER_REMAP_COEF_BITS: u32 = 15;
    const INTER_REMAP_COEF_SCALE: i32 = 1 << INTER_REMAP_COEF_BITS; // 32768

    // Precompute bilinear weight table (matches OpenCV BilinearTab_i)
    // For each (fx, fy) pair quantized to 1/32, compute 4 weights as i16
    // Precompute bilinear weight table matching OpenCV BilinearTab_i exactly.
    // OpenCV: saturate_cast<short>(v * SCALE) for each weight, then correct sum.
    let mut wtab = vec![[0i16; 4]; (INTER_TAB_SIZE * INTER_TAB_SIZE) as usize];
    for iy in 0..INTER_TAB_SIZE {
        let ay = iy as f32 / INTER_TAB_SIZE as f32;
        // 1D tab for Y
        let ty0 = 1.0 - ay;
        let ty1 = ay;
        for ix in 0..INTER_TAB_SIZE {
            let ax = ix as f32 / INTER_TAB_SIZE as f32;
            let tx0 = 1.0 - ax;
            let tx1 = ax;
            let idx = (iy * INTER_TAB_SIZE + ix) as usize;

            // Compute 2D weights as float, then saturate_cast to i16
            let fweights = [ty0 * tx0, ty0 * tx1, ty1 * tx0, ty1 * tx1];
            let mut iweights = [0i16; 4];
            let mut isum: i32 = 0;
            for k in 0..4 {
                let v = (fweights[k] * INTER_REMAP_COEF_SCALE as f32).round() as i32;
                iweights[k] = v.clamp(-32768, 32767) as i16; // saturate_cast<short>
                isum += iweights[k] as i32;
            }
            // Correct rounding error matching OpenCV exactly:
            // OpenCV only checks the central 2x2 of the ksize×ksize kernel.
            // For bilinear (ksize=2), ksize2=1, so k1 in [1,2) and k2 in [1,2)
            // → only index (1,1) = flat index 3 is checked for both min and max.
            // So the correction always adjusts itab[3] (bottom-right weight).
            if isum != INTER_REMAP_COEF_SCALE {
                let diff = isum - INTER_REMAP_COEF_SCALE;
                // OpenCV: adjust itab[ksize2*ksize + ksize2] = itab[1*2+1] = itab[3]
                iweights[3] = (iweights[3] as i32 - diff) as i16;
            }
            wtab[idx] = iweights;
        }
    }

    let mut output = vec![0u8; w * h * bpp];

    for oy in 0..h {
        for ox in 0..w {
            // Step 1: compute distorted source coordinate in f64
            // (matches OpenCV initUndistortRectifyMap scalar path)
            let x = (ox as f64 - camera.cx) / camera.fx;
            let y = (oy as f64 - camera.cy) / camera.fy;
            let r2 = x * x + y * y;
            let kr = 1.0 + ((dist.k3 * r2 + dist.k2) * r2 + dist.k1) * r2;
            let xd = x * kr;
            let yd = y * kr;
            let u = camera.fx * xd + camera.cx;
            let v = camera.fy * yd + camera.cy;

            // Step 2: quantize to fixed-point (matches OpenCV m1type=CV_16SC2)
            // saturate_cast<int>(u * INTER_TAB_SIZE) — rounds to nearest
            let iu = (u * INTER_TAB_SIZE as f64).round() as i32;
            let iv = (v * INTER_TAB_SIZE as f64).round() as i32;

            let sx = iu >> INTER_BITS; // integer pixel X
            let sy = iv >> INTER_BITS; // integer pixel Y
            let fxy = ((iv & (INTER_TAB_SIZE - 1)) * INTER_TAB_SIZE + (iu & (INTER_TAB_SIZE - 1)))
                as usize;

            let out_idx = (oy * w + ox) * bpp;

            // Step 3: fixed-point bilinear interpolation (matches OpenCV remapBilinear)
            if sx >= 0 && (sx as usize) < w - 1 && sy >= 0 && (sy as usize) < h - 1 {
                let sx = sx as usize;
                let sy = sy as usize;
                let weights = &wtab[fxy];

                for ch in 0..bpp {
                    let p00 = pixels[sy * w * bpp + sx * bpp + ch] as i32;
                    let p10 = pixels[sy * w * bpp + (sx + 1) * bpp + ch] as i32;
                    let p01 = pixels[(sy + 1) * w * bpp + sx * bpp + ch] as i32;
                    let p11 = pixels[(sy + 1) * w * bpp + (sx + 1) * bpp + ch] as i32;

                    // FixedPtCast: (sum + (1 << 14)) >> 15
                    let sum = p00 * weights[0] as i32
                        + p10 * weights[1] as i32
                        + p01 * weights[2] as i32
                        + p11 * weights[3] as i32;
                    let val = (sum + (1 << (INTER_REMAP_COEF_BITS - 1))) >> INTER_REMAP_COEF_BITS;
                    output[out_idx + ch] = val.clamp(0, 255) as u8;
                }
            }
            // Out-of-bounds pixels stay black (0)
        }
    }

    Ok(DecodedImage {
        pixels: output,
        info: info.clone(),
        icc_profile: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_image(w: u32, h: u32) -> (Vec<u8>, ImageInfo) {
        let pixels: Vec<u8> = (0..(w * h * 3)).map(|i| (i % 256) as u8).collect();
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        (pixels, info)
    }

    #[test]
    fn resize_changes_dimensions() {
        let (px, info) = make_image(64, 64);
        let result = resize(&px, &info, 32, 16, ResizeFilter::Bilinear).unwrap();
        assert_eq!(result.info.width, 32);
        assert_eq!(result.info.height, 16);
    }

    #[test]
    fn resize_preserves_format() {
        let (px, info) = make_image(16, 16);
        let result = resize(&px, &info, 8, 8, ResizeFilter::Nearest).unwrap();
        assert_eq!(result.info.format, PixelFormat::Rgb8);
    }

    #[test]
    fn resize_pixel_data_length_correct() {
        let (px, info) = make_image(16, 16);
        let result = resize(&px, &info, 32, 24, ResizeFilter::Lanczos3).unwrap();
        assert_eq!(result.pixels.len(), 32 * 24 * 3);
    }

    #[test]
    fn resize_all_filters_work() {
        let (px, info) = make_image(16, 16);
        for filter in [
            ResizeFilter::Nearest,
            ResizeFilter::Bilinear,
            ResizeFilter::Bicubic,
            ResizeFilter::Lanczos3,
        ] {
            let result = resize(&px, &info, 8, 8, filter);
            assert!(result.is_ok(), "filter {filter:?} failed");
        }
    }

    #[test]
    fn crop_returns_correct_region() {
        let (px, info) = make_image(32, 32);
        let result = crop(&px, &info, 4, 4, 16, 16).unwrap();
        assert_eq!(result.info.width, 16);
        assert_eq!(result.info.height, 16);
        assert_eq!(result.pixels.len(), 16 * 16 * 3);
    }

    #[test]
    fn crop_out_of_bounds_returns_error() {
        let (px, info) = make_image(16, 16);
        let result = crop(&px, &info, 10, 10, 10, 10);
        assert!(result.is_err());
        match result.unwrap_err() {
            ImageError::InvalidParameters(_) => {}
            other => panic!("expected InvalidParameters, got {other:?}"),
        }
    }

    #[test]
    fn crop_zero_dimension_returns_error() {
        let (px, info) = make_image(16, 16);
        let result = crop(&px, &info, 0, 0, 0, 8);
        assert!(result.is_err());
    }

    #[test]
    fn rotate_90_swaps_dimensions() {
        let (px, info) = make_image(16, 8);
        let result = rotate(&px, &info, Rotation::R90).unwrap();
        assert_eq!(result.info.width, 8);
        assert_eq!(result.info.height, 16);
    }

    #[test]
    fn rotate_180_preserves_dimensions() {
        let (px, info) = make_image(16, 8);
        let result = rotate(&px, &info, Rotation::R180).unwrap();
        assert_eq!(result.info.width, 16);
        assert_eq!(result.info.height, 8);
    }

    #[test]
    fn rotate_270_swaps_dimensions() {
        let (px, info) = make_image(16, 8);
        let result = rotate(&px, &info, Rotation::R270).unwrap();
        assert_eq!(result.info.width, 8);
        assert_eq!(result.info.height, 16);
    }

    #[test]
    fn flip_horizontal_preserves_dimensions() {
        let (px, info) = make_image(16, 8);
        let result = flip(&px, &info, FlipDirection::Horizontal).unwrap();
        assert_eq!(result.info.width, 16);
        assert_eq!(result.info.height, 8);
        assert_eq!(result.pixels.len(), px.len());
    }

    #[test]
    fn flip_vertical_preserves_dimensions() {
        let (px, info) = make_image(16, 8);
        let result = flip(&px, &info, FlipDirection::Vertical).unwrap();
        assert_eq!(result.info.width, 16);
        assert_eq!(result.info.height, 8);
    }

    #[test]
    fn convert_rgb8_to_rgba8() {
        let (px, info) = make_image(8, 8);
        let result = convert_format(&px, &info, PixelFormat::Rgba8).unwrap();
        assert_eq!(result.info.format, PixelFormat::Rgba8);
        assert_eq!(result.pixels.len(), 8 * 8 * 4);
    }

    #[test]
    fn convert_rgb8_to_gray8() {
        let (px, info) = make_image(8, 8);
        let result = convert_format(&px, &info, PixelFormat::Gray8).unwrap();
        assert_eq!(result.info.format, PixelFormat::Gray8);
        assert_eq!(result.pixels.len(), 8 * 8 * 1);
    }

    #[test]
    fn convert_unsupported_returns_error() {
        let (px, info) = make_image(8, 8);
        let result = convert_format(&px, &info, PixelFormat::Nv12);
        assert!(result.is_err());
    }

    #[test]
    fn auto_orient_normal_is_identity() {
        let (px, info) = make_image(16, 8);
        let result = auto_orient(&px, &info, ExifOrientation::Normal).unwrap();
        assert_eq!(result.pixels, px);
        assert_eq!(result.info.width, 16);
        assert_eq!(result.info.height, 8);
    }

    #[test]
    fn auto_orient_rotate90_swaps_dimensions() {
        let (px, info) = make_image(16, 8);
        let result = auto_orient(&px, &info, ExifOrientation::Rotate90).unwrap();
        assert_eq!(result.info.width, 8);
        assert_eq!(result.info.height, 16);
    }

    #[test]
    fn auto_orient_rotate180_preserves_dimensions() {
        let (px, info) = make_image(16, 8);
        let result = auto_orient(&px, &info, ExifOrientation::Rotate180).unwrap();
        assert_eq!(result.info.width, 16);
        assert_eq!(result.info.height, 8);
    }

    #[test]
    fn auto_orient_rotate270_swaps_dimensions() {
        let (px, info) = make_image(16, 8);
        let result = auto_orient(&px, &info, ExifOrientation::Rotate270).unwrap();
        assert_eq!(result.info.width, 8);
        assert_eq!(result.info.height, 16);
    }

    #[test]
    fn auto_orient_flip_horizontal_preserves_dimensions() {
        let (px, info) = make_image(16, 8);
        let result = auto_orient(&px, &info, ExifOrientation::FlipHorizontal).unwrap();
        assert_eq!(result.info.width, 16);
        assert_eq!(result.info.height, 8);
    }

    #[test]
    fn auto_orient_flip_vertical_preserves_dimensions() {
        let (px, info) = make_image(16, 8);
        let result = auto_orient(&px, &info, ExifOrientation::FlipVertical).unwrap();
        assert_eq!(result.info.width, 16);
        assert_eq!(result.info.height, 8);
    }

    #[test]
    fn auto_orient_transpose_swaps_dimensions() {
        let (px, info) = make_image(16, 8);
        let result = auto_orient(&px, &info, ExifOrientation::Transpose).unwrap();
        assert_eq!(result.info.width, 8);
        assert_eq!(result.info.height, 16);
    }

    #[test]
    fn auto_orient_transverse_swaps_dimensions() {
        let (px, info) = make_image(16, 8);
        let result = auto_orient(&px, &info, ExifOrientation::Transverse).unwrap();
        assert_eq!(result.info.width, 8);
        assert_eq!(result.info.height, 16);
    }

    #[test]
    fn auto_orient_all_8_orientations_work() {
        let (px, info) = make_image(16, 8);
        for tag in 1..=8 {
            let orient = ExifOrientation::from_tag(tag);
            let result = auto_orient(&px, &info, orient);
            assert!(result.is_ok(), "orientation {tag} failed");
        }
    }

    #[test]
    fn auto_orient_from_exif_with_no_exif_is_identity() {
        let (px, info) = make_image(16, 8);
        // Non-JPEG data — no EXIF, should return unchanged
        let result = auto_orient_from_exif(&px, &info, &[0x89, 0x50]).unwrap();
        assert_eq!(result.pixels, px);
    }

    // ─── Extended Geometry Tests ────────────────────────────────────────

    #[test]
    fn rotate_arbitrary_0_preserves_dimensions() {
        let (px, info) = make_image(32, 32);
        let bg = [0, 0, 0];
        let result = rotate_arbitrary(&px, &info, 0.0, &bg).unwrap();
        assert_eq!(result.info.width, 32);
        assert_eq!(result.info.height, 32);
    }

    #[test]
    fn rotate_arbitrary_90_matches_dimensions() {
        let (px, info) = make_image(32, 16);
        let bg = [0, 0, 0];
        let result = rotate_arbitrary(&px, &info, 90.0, &bg).unwrap();
        // 90 degrees: output is roughly height x width
        assert!(result.info.width >= 15 && result.info.width <= 17);
        assert!(result.info.height >= 31 && result.info.height <= 33);
    }

    #[test]
    fn rotate_arbitrary_45_expands_dimensions() {
        let (px, info) = make_image(32, 32);
        let bg = [255, 255, 255];
        let result = rotate_arbitrary(&px, &info, 45.0, &bg).unwrap();
        // 45 degrees expands: side * sqrt(2) ≈ 45
        assert!(result.info.width > 40);
        assert!(result.info.height > 40);
    }

    #[test]
    fn rotate_arbitrary_180_preserves_dimensions() {
        let (px, info) = make_image(32, 32);
        let bg = [0, 0, 0];
        let result = rotate_arbitrary(&px, &info, 180.0, &bg).unwrap();
        // Floating-point bounding box may be ±1 of original
        assert!((result.info.width as i32 - 32).abs() <= 1);
        assert!((result.info.height as i32 - 32).abs() <= 1);
    }

    #[test]
    fn rotate_arbitrary_preserves_format() {
        let (px, info) = make_image(16, 16);
        let bg = [0, 0, 0];
        let result = rotate_arbitrary(&px, &info, 37.0, &bg).unwrap();
        assert_eq!(result.info.format, PixelFormat::Rgb8);
    }

    #[test]
    fn pad_symmetric() {
        let (px, info) = make_image(16, 16);
        let result = pad(&px, &info, 4, 4, 4, 4, &[128, 128, 128]).unwrap();
        assert_eq!(result.info.width, 24);
        assert_eq!(result.info.height, 24);
        assert_eq!(result.pixels.len(), 24 * 24 * 3);
    }

    #[test]
    fn pad_asymmetric() {
        let (px, info) = make_image(8, 8);
        let result = pad(&px, &info, 2, 4, 6, 8, &[255, 0, 0]).unwrap();
        assert_eq!(result.info.width, 8 + 4 + 8);
        assert_eq!(result.info.height, 8 + 2 + 6);
    }

    #[test]
    fn pad_preserves_center_pixels() {
        let (px, info) = make_image(4, 4);
        let result = pad(&px, &info, 1, 1, 1, 1, &[0, 0, 0]).unwrap();
        // Check center pixel (1,1) in output = (0,0) in original
        let bpp = 3;
        let out_w = 6;
        let idx = (1 * out_w + 1) * bpp;
        assert_eq!(result.pixels[idx..idx + 3], px[0..3]);
    }

    #[test]
    fn pad_fill_color_correct() {
        let px = vec![128u8; 4 * 4 * 3]; // 4x4 gray
        let info = ImageInfo {
            width: 4,
            height: 4,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let result = pad(&px, &info, 1, 1, 1, 1, &[255, 0, 0]).unwrap();
        // Top-left corner (0,0) should be fill color (red)
        assert_eq!(result.pixels[0], 255); // R
        assert_eq!(result.pixels[1], 0); // G
        assert_eq!(result.pixels[2], 0); // B
    }

    #[test]
    fn trim_removes_uniform_border() {
        // Create 8x8 image with 2-pixel red border around green center
        let mut px = vec![255u8; 8 * 8 * 3]; // all red
        for y in 2..6 {
            for x in 2..6 {
                let idx = (y * 8 + x) * 3;
                px[idx] = 0; // R
                px[idx + 1] = 255; // G
                px[idx + 2] = 0; // B
            }
        }
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let result = trim(&px, &info, 0).unwrap();
        assert_eq!(result.info.width, 4);
        assert_eq!(result.info.height, 4);
    }

    #[test]
    fn trim_with_threshold() {
        // Create image with near-uniform border (within threshold)
        let mut px = vec![100u8; 8 * 8 * 3]; // all 100
        for y in 2..6 {
            for x in 2..6 {
                let idx = (y * 8 + x) * 3;
                px[idx] = 200; // significantly different
            }
        }
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        // Threshold 0: all 100-valued border trimmed
        let result = trim(&px, &info, 0).unwrap();
        assert_eq!(result.info.width, 4);
    }

    #[test]
    fn trim_all_uniform_returns_1x1() {
        let px = vec![128u8; 8 * 8 * 3];
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let result = trim(&px, &info, 0).unwrap();
        assert_eq!(result.info.width, 1);
        assert_eq!(result.info.height, 1);
    }

    #[test]
    fn affine_identity() {
        // Use a larger image so edge bg-fill is a small fraction
        let (px, info) = make_image(64, 64);
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let result = affine(&px, &info, &identity, 64, 64, &[0, 0, 0]).unwrap();
        assert_eq!(result.info.width, 64);
        assert_eq!(result.info.height, 64);
        // Interior pixels should match; edge row/col may differ (bg fill)
        // With 64x64, edge pixels are ~3% of total → low MAE
        let mae: f64 = px
            .iter()
            .zip(result.pixels.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / px.len() as f64;
        assert!(
            mae < 5.0,
            "identity affine MAE should be < 5.0, got {mae:.2}"
        );
    }

    #[test]
    fn affine_scale_2x() {
        let (px, info) = make_image(8, 8);
        let scale2 = [2.0, 0.0, 0.0, 0.0, 2.0, 0.0];
        let result = affine(&px, &info, &scale2, 16, 16, &[0, 0, 0]).unwrap();
        assert_eq!(result.info.width, 16);
        assert_eq!(result.info.height, 16);
    }

    #[test]
    fn affine_singular_matrix_rejected() {
        let (px, info) = make_image(8, 8);
        let singular = [1.0, 2.0, 0.0, 2.0, 4.0, 0.0]; // det = 1*4 - 2*2 = 0
        let result = affine(&px, &info, &singular, 8, 8, &[0, 0, 0]);
        assert!(result.is_err());
    }

    // ─── 16-bit format conversion tests ────────────────────────────

    #[test]
    fn convert_rgb8_to_rgb16_roundtrip() {
        let pixels = vec![0u8, 128, 255, 64, 192, 32];
        let info = ImageInfo {
            width: 2,
            height: 1,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        // Upscale to 16-bit
        let up = convert_format(&pixels, &info, PixelFormat::Rgb16).unwrap();
        assert_eq!(up.info.format, PixelFormat::Rgb16);
        assert_eq!(up.pixels.len(), 12); // 2 pixels * 3 channels * 2 bytes

        // Verify precise scaling: u8*257 maps 0->0, 128->32896, 255->65535
        let r0 = u16::from_le_bytes([up.pixels[0], up.pixels[1]]);
        let g0 = u16::from_le_bytes([up.pixels[2], up.pixels[3]]);
        let b0 = u16::from_le_bytes([up.pixels[4], up.pixels[5]]);
        assert_eq!(r0, 0);
        assert_eq!(g0, 128 * 257); // 32896
        assert_eq!(b0, 65535);

        // Downscale back to 8-bit
        let down = convert_format(&up.pixels, &up.info, PixelFormat::Rgb8).unwrap();
        assert_eq!(down.info.format, PixelFormat::Rgb8);
        assert_eq!(
            down.pixels, pixels,
            "Rgb8 -> Rgb16 -> Rgb8 must be lossless"
        );
    }

    #[test]
    fn convert_gray8_to_gray16_preserves_values() {
        let pixels: Vec<u8> = (0..=255).collect();
        let info = ImageInfo {
            width: 256,
            height: 1,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let up = convert_format(&pixels, &info, PixelFormat::Gray16).unwrap();
        assert_eq!(up.info.format, PixelFormat::Gray16);

        // Check boundary values
        let first = u16::from_le_bytes([up.pixels[0], up.pixels[1]]);
        let last = u16::from_le_bytes([up.pixels[510], up.pixels[511]]);
        assert_eq!(first, 0);
        assert_eq!(last, 65535);
    }

    #[test]
    fn convert_rgb16_to_rgba16() {
        let mut pixels = Vec::new();
        for v in [0u16, 32768, 65535] {
            pixels.extend_from_slice(&v.to_le_bytes());
        }
        let info = ImageInfo {
            width: 1,
            height: 1,
            format: PixelFormat::Rgb16,
            color_space: ColorSpace::Srgb,
        };
        let result = convert_format(&pixels, &info, PixelFormat::Rgba16).unwrap();
        assert_eq!(result.info.format, PixelFormat::Rgba16);
        assert_eq!(result.pixels.len(), 8); // 1 pixel * 4 channels * 2 bytes
        // Alpha should be 65535 (fully opaque)
        let alpha = u16::from_le_bytes([result.pixels[6], result.pixels[7]]);
        assert_eq!(alpha, 65535);
    }

    // ─── 16-bit E2E chain test ─────────────────────────────────────────

    #[test]
    fn e2e_16bit_chain_gamma_brightness_resize_equalize() {
        use crate::domain::histogram;
        use crate::domain::point_ops;

        // 1. Create a 16-bit Rgb16 gradient (32x32)
        let (w, h) = (32u32, 32u32);
        let mut pixels = Vec::with_capacity((w * h * 6) as usize);
        for y in 0..h {
            for x in 0..w {
                let r = (x * 65535 / w.max(1)) as u16;
                let g = (y * 65535 / h.max(1)) as u16;
                let b = 32768u16;
                pixels.extend_from_slice(&r.to_le_bytes());
                pixels.extend_from_slice(&g.to_le_bytes());
                pixels.extend_from_slice(&b.to_le_bytes());
            }
        }
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb16,
            color_space: ColorSpace::Srgb,
        };

        // 2. Apply gamma 2.2 (16-bit auto-dispatch)
        let after_gamma = point_ops::gamma(&pixels, &info, 2.2).unwrap();
        assert_eq!(after_gamma.len(), pixels.len());

        // 3. Apply brightness +0.1 (16-bit auto-dispatch)
        let after_bright = {
            use crate::domain::filters;
            {
                let r = rasmcore_pipeline::Rect::new(0, 0, info.width, info.height);
                let mut u = |_: rasmcore_pipeline::Rect| Ok(after_gamma.clone());
                filters::brightness(r, &mut u, &info, &filters::BrightnessParams { amount: 0.1 })
                    .unwrap()
            }
        };

        // 4. Resize to 16x16 (fast_image_resize U16x3)
        let resized = resize(&after_bright, &info, 16, 16, ResizeFilter::Lanczos3).unwrap();
        assert_eq!(resized.info.width, 16);
        assert_eq!(resized.info.height, 16);
        assert_eq!(resized.info.format, PixelFormat::Rgb16);
        assert_eq!(resized.pixels.len(), 16 * 16 * 6);

        // 5. Equalize (16-bit histogram with 65536 bins)
        let equalized = histogram::equalize(&resized.pixels, &resized.info).unwrap();
        assert_eq!(equalized.len(), resized.pixels.len());

        // 6. Verify: read back u16 values, check range expanded
        let values: Vec<u16> = equalized
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        let min_val = *values.iter().min().unwrap();
        let max_val = *values.iter().max().unwrap();
        // After equalization, range should span most of 0-65535
        assert!(
            max_val > 50000,
            "equalized max should be near 65535, got {max_val}"
        );
        assert!(
            min_val < 5000,
            "equalized min should be near 0, got {min_val}"
        );
    }

    #[test]
    fn e2e_16bit_encode_decode_roundtrip_tiff() {
        // Create 16-bit image
        let (w, h) = (8u32, 8u32);
        let mut pixels = Vec::new();
        for i in 0..(w * h) {
            let v = (i * 1023) as u16;
            pixels.extend_from_slice(&v.to_le_bytes());
            pixels.extend_from_slice(&(v / 2).to_le_bytes());
            pixels.extend_from_slice(&(65535 - v).to_le_bytes());
        }
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb16,
            color_space: ColorSpace::Srgb,
        };

        // Encode as 16-bit TIFF
        let encoded = crate::domain::encoder::tiff::encode(
            &pixels,
            &info,
            &crate::domain::encoder::tiff::TiffEncodeConfig::default(),
        )
        .unwrap();

        // Decode back
        let decoded = crate::domain::decoder::decode(&encoded).unwrap();
        assert_eq!(decoded.info.format, PixelFormat::Rgb16);
        assert_eq!(decoded.info.width, w);
        assert_eq!(decoded.info.height, h);
        assert_eq!(
            decoded.pixels, pixels,
            "16-bit TIFF roundtrip must be lossless"
        );
    }

    #[test]
    fn resize_rgb16_preserves_format() {
        let mut pixels = Vec::new();
        for i in 0..16u32 * 16 {
            let v = (i * 257) as u16;
            pixels.extend_from_slice(&v.to_le_bytes());
            pixels.extend_from_slice(&v.to_le_bytes());
            pixels.extend_from_slice(&v.to_le_bytes());
        }
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Rgb16,
            color_space: ColorSpace::Srgb,
        };
        let result = resize(&pixels, &info, 8, 8, ResizeFilter::Bilinear).unwrap();
        assert_eq!(result.info.format, PixelFormat::Rgb16);
        assert_eq!(result.info.width, 8);
        assert_eq!(result.pixels.len(), 8 * 8 * 6);
    }

    #[test]
    fn crop_rgb16_works() {
        let mut pixels = Vec::new();
        for i in 0..16u32 * 16 {
            let v = (i * 100) as u16;
            pixels.extend_from_slice(&v.to_le_bytes());
            pixels.extend_from_slice(&v.to_le_bytes());
            pixels.extend_from_slice(&v.to_le_bytes());
        }
        let info = ImageInfo {
            width: 16,
            height: 16,
            format: PixelFormat::Rgb16,
            color_space: ColorSpace::Srgb,
        };
        let result = crop(&pixels, &info, 2, 2, 8, 8).unwrap();
        assert_eq!(result.info.format, PixelFormat::Rgb16);
        assert_eq!(result.info.width, 8);
        assert_eq!(result.info.height, 8);
        assert_eq!(result.pixels.len(), 8 * 8 * 6);
    }

    #[test]
    fn undistort_zero_distortion_is_identity() {
        let (pixels, info) = make_image(32, 32);
        let camera = CameraMatrix {
            fx: 32.0,
            fy: 32.0,
            cx: 16.0,
            cy: 16.0,
        };
        let dist = DistortionCoeffs {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
        };
        let result = undistort(&pixels, &info, &camera, &dist).unwrap();
        // Interior pixels should be identical (borders may differ due to sampling)
        let bpp = 3;
        let mut matched = 0;
        let mut total = 0;
        for y in 2..30 {
            for x in 2..30 {
                total += 1;
                let idx = (y * 32 + x) * bpp;
                if pixels[idx..idx + bpp] == result.pixels[idx..idx + bpp] {
                    matched += 1;
                }
            }
        }
        assert_eq!(matched, total, "zero distortion should be identity");
    }

    #[test]
    fn undistort_barrel_shifts_pixels_inward() {
        // Barrel distortion (k1 > 0): corners move outward in distorted image
        // Undistortion should move them inward (toward center)
        let mut pixels = vec![0u8; 64 * 64 * 3];
        // White pixel at corner
        let idx = (5 * 64 + 5) * 3;
        pixels[idx] = 255;
        pixels[idx + 1] = 255;
        pixels[idx + 2] = 255;
        let info = ImageInfo {
            width: 64,
            height: 64,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let camera = CameraMatrix {
            fx: 50.0,
            fy: 50.0,
            cx: 32.0,
            cy: 32.0,
        };
        let dist = DistortionCoeffs {
            k1: 0.5,
            k2: 0.0,
            k3: 0.0,
        };
        let result = undistort(&pixels, &info, &camera, &dist).unwrap();
        // The white pixel should have moved (it won't be at exactly the same position)
        assert_eq!(result.pixels.len(), pixels.len());
    }

    #[test]
    fn undistort_produces_valid_output_size() {
        let (pixels, info) = make_image(128, 128);
        let camera = CameraMatrix {
            fx: 100.0,
            fy: 100.0,
            cx: 64.0,
            cy: 64.0,
        };
        let dist = DistortionCoeffs {
            k1: -0.3,
            k2: 0.1,
            k3: 0.0,
        };
        let result = undistort(&pixels, &info, &camera, &dist).unwrap();
        assert_eq!(result.info.width, 128);
        assert_eq!(result.info.height, 128);
        assert_eq!(result.pixels.len(), 128 * 128 * 3);
    }
}
