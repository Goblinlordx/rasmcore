//! RGB to YUV420 color space conversion.
//!
//! Converts input pixels to the YUV420 format used by VP8:
//! - Y (luma): full resolution, one value per pixel
//! - U, V (chroma): half resolution in both dimensions
//!
//! Uses BT.601 coefficients matching VP8/libwebp.

/// YUV420 image buffer.
#[derive(Debug)]
pub struct YuvImage {
    pub width: u32,
    pub height: u32,
    /// Luma plane (width × height bytes).
    pub y: Vec<u8>,
    /// Chroma-U plane (uv_w × uv_h bytes).
    pub u: Vec<u8>,
    /// Chroma-V plane (uv_w × uv_h bytes).
    pub v: Vec<u8>,
}

/// Convert RGB8 pixels to YUV420.
///
/// Uses BT.601 coefficients (matching VP8/libwebp):
///   Y =  (66*R + 129*G +  25*B + 128) >> 8 + 16
///   U = (-38*R -  74*G + 112*B + 128) >> 8 + 128
///   V = (112*R -  94*G -  18*B + 128) >> 8 + 128
pub fn rgb_to_yuv420(pixels: &[u8], width: u32, height: u32) -> YuvImage {
    let w = width as usize;
    let h = height as usize;
    let uv_w = w.div_ceil(2);
    let uv_h = h.div_ceil(2);

    let mut y_plane = vec![0u8; w * h];
    let mut u_plane = vec![0u8; uv_w * uv_h];
    let mut v_plane = vec![0u8; uv_w * uv_h];

    // Compute full-resolution Y plane
    for row in 0..h {
        for col in 0..w {
            let i = (row * w + col) * 3;
            let r = pixels[i] as i32;
            let g = pixels[i + 1] as i32;
            let b = pixels[i + 2] as i32;
            y_plane[row * w + col] =
                ((66 * r + 129 * g + 25 * b + 128) >> 8).wrapping_add(16) as u8;
        }
    }

    // Compute half-resolution U, V planes (average 2×2 blocks)
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
            u_plane[idx] = ((-38 * r - 74 * g + 112 * b + 128) >> 8).wrapping_add(128) as u8;
            v_plane[idx] = ((112 * r - 94 * g - 18 * b + 128) >> 8).wrapping_add(128) as u8;
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

/// Convert RGBA8 pixels to YUV420 (alpha channel discarded).
pub fn rgba_to_yuv420(pixels: &[u8], width: u32, height: u32) -> YuvImage {
    // Strip alpha to get RGB, then convert
    let w = width as usize;
    let h = height as usize;
    let mut rgb = Vec::with_capacity(w * h * 3);
    for chunk in pixels.chunks_exact(4) {
        rgb.push(chunk[0]);
        rgb.push(chunk[1]);
        rgb.push(chunk[2]);
    }
    rgb_to_yuv420(&rgb, width, height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_to_yuv420_dimensions() {
        let pixels = vec![128u8; 16 * 16 * 3];
        let yuv = rgb_to_yuv420(&pixels, 16, 16);
        assert_eq!(yuv.y.len(), 16 * 16);
        assert_eq!(yuv.u.len(), 8 * 8);
        assert_eq!(yuv.v.len(), 8 * 8);
    }

    #[test]
    fn rgba_to_yuv420_strips_alpha() {
        let pixels = vec![128u8; 4 * 4 * 4]; // RGBA
        let yuv = rgba_to_yuv420(&pixels, 4, 4);
        assert_eq!(yuv.y.len(), 16);
        assert_eq!(yuv.u.len(), 4);
    }

    #[test]
    fn odd_dimensions() {
        let pixels = vec![128u8; 3 * 5 * 3]; // 3×5, odd
        let yuv = rgb_to_yuv420(&pixels, 3, 5);
        assert_eq!(yuv.y.len(), 15);
        assert_eq!(yuv.u.len(), 2 * 3); // ceil(3/2) * ceil(5/2)
    }

    #[test]
    fn black_pixel_gives_y16() {
        let pixels = vec![0u8; 3]; // 1x1 black
        let yuv = rgb_to_yuv420(&pixels, 1, 1);
        assert_eq!(yuv.y[0], 16); // BT.601 black
        assert_eq!(yuv.u[0], 128); // neutral chroma
        assert_eq!(yuv.v[0], 128);
    }

    #[test]
    fn white_pixel_gives_y235() {
        let pixels = vec![255u8; 3]; // 1x1 white
        let yuv = rgb_to_yuv420(&pixels, 1, 1);
        assert_eq!(yuv.y[0], 235); // BT.601 white (approximately)
    }
}
