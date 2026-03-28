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
        // BT.601 (JPEG standard): full-range YCbCr
        y.push(((77 * r + 150 * g + 29 * b + 128) >> 8) as u8);
        cb_full.push(((-43 * r - 85 * g + 128 * b + 128) >> 8).wrapping_add(128) as u8);
        cr_full.push(((128 * r - 107 * g - 21 * b + 128) >> 8).wrapping_add(128) as u8);
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

/// Downsample a full-resolution plane by averaging blocks.
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
            dst.push(if count > 0 { (sum / count) as u8 } else { 128 });
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
}
