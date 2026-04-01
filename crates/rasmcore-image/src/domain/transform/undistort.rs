use super::super::error::ImageError;
use super::super::types::{DecodedImage, ImageInfo};
use super::{bytes_per_pixel, validate_pixel_buffer};

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
