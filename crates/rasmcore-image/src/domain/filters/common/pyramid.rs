//! Pyramid helpers for filters.

#[allow(unused_imports)]
use super::*;


pub const PYR_KERNEL: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];

/// Downsample by 2x using box filter (average of 2x2 blocks).
pub fn downsample_2x(data: &[f32], w: usize, h: usize) -> Vec<f32> {
    let nw = w.div_ceil(2);
    let nh = h.div_ceil(2);
    let mut out = vec![0.0f32; nw * nh];
    for y in 0..nh {
        for x in 0..nw {
            let x0 = x * 2;
            let y0 = y * 2;
            let x1 = (x0 + 1).min(w - 1);
            let y1 = (y0 + 1).min(h - 1);
            out[y * nw + x] =
                (data[y0 * w + x0] + data[y0 * w + x1] + data[y1 * w + x0] + data[y1 * w + x1])
                    / 4.0;
        }
    }
    out
}

/// Build Gaussian pyramid for a single-channel f32 image.
/// Returns levels+1 images: [original, level1, level2, ...].
pub fn gaussian_pyramid_gray(src: &[f32], w: u32, h: u32, levels: usize) -> Vec<Vec<f32>> {
    let mut pyr = Vec::with_capacity(levels + 1);
    pyr.push(src.to_vec());
    let mut cw = w;
    let mut ch = h;
    for _ in 0..levels {
        let (down, nw, nh) = pyr_down_gray(pyr.last().unwrap(), cw, ch);
        cw = nw;
        ch = nh;
        pyr.push(down);
    }
    pyr
}

/// Build Laplacian pyramid for a 3-channel f32 image.
/// Returns levels+1 entries: levels Laplacian layers + 1 low-res residual.
/// Each entry is (pixels, width, height).
pub fn laplacian_pyramid_rgb(src: &[f32], w: u32, h: u32, levels: usize) -> Vec<(Vec<f32>, u32, u32)> {
    // Build Gaussian pyramid first
    let mut gpyr: Vec<(Vec<f32>, u32, u32)> = Vec::with_capacity(levels + 1);
    gpyr.push((src.to_vec(), w, h));
    let mut cw = w;
    let mut ch = h;
    for _ in 0..levels {
        let (down, nw, nh) = pyr_down_rgb(gpyr.last().unwrap().0.as_slice(), cw, ch);
        cw = nw;
        ch = nh;
        gpyr.push((down, nw, nh));
    }

    // Laplacian = Gaussian[i] - pyrUp(Gaussian[i+1])
    let mut lpyr: Vec<(Vec<f32>, u32, u32)> = Vec::with_capacity(levels + 1);
    for i in 0..levels {
        let (ref g_curr, gw, gh) = gpyr[i];
        let (ref g_next, nw, nh) = gpyr[i + 1];
        let upsampled = pyr_up_rgb(g_next, nw, nh, gw, gh);
        let npx = (gw * gh) as usize * 3;
        let mut diff = Vec::with_capacity(npx);
        for j in 0..npx {
            diff.push(g_curr[j] - upsampled[j]);
        }
        lpyr.push((diff, gw, gh));
    }
    // Last level is the low-res residual
    let (ref last, lw, lh) = gpyr[levels];
    lpyr.push((last.clone(), lw, lh));

    lpyr
}

/// Gaussian pyramid downsample: blur + subsample by 2.
///
/// Applies a 5x5 Gaussian kernel then takes every other pixel.
/// Output is (w+1)/2 x (h+1)/2. Matches `cv2.pyrDown`.
pub fn pyr_down(pixels: &[u8], info: &ImageInfo) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    if info.format != PixelFormat::Gray8 {
        return Err(ImageError::UnsupportedFormat(
            "pyr_down requires Gray8".into(),
        ));
    }
    let w = info.width as usize;
    let h = info.height as usize;
    let ow = w.div_ceil(2);
    let oh = h.div_ceil(2);

    // 5x5 Gaussian kernel (1/256 normalization): [1,4,6,4,1] x [1,4,6,4,1]
    let kernel_1d: [i32; 5] = [1, 4, 6, 4, 1];

    // Horizontal pass → temp buffer
    let mut temp = vec![0i32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum: i32 = 0;
            for k in 0..5i32 {
                let sx = reflect101(x as isize + k as isize - 2, w as isize) as usize;
                sum += pixels[y * w + sx] as i32 * kernel_1d[k as usize];
            }
            temp[y * w + x] = sum;
        }
    }

    // Vertical pass + subsample
    let mut output = vec![0u8; ow * oh];
    for oy in 0..oh {
        for ox in 0..ow {
            let x = ox * 2;
            let y = oy * 2;
            let mut sum: i32 = 0;
            for k in 0..5i32 {
                let sy = reflect101(y as isize + k as isize - 2, h as isize) as usize;
                sum += temp[sy * w + x] * kernel_1d[k as usize];
            }
            // Normalize by 256 (16*16)
            output[oy * ow + ox] = ((sum + 128) >> 8).clamp(0, 255) as u8;
        }
    }

    let new_info = ImageInfo {
        width: ow as u32,
        height: oh as u32,
        format: info.format,
        color_space: info.color_space,
    };
    Ok((output, new_info))
}

/// pyrDown for single-channel f32 image.
/// Applies 5×5 Gaussian blur then subsamples by 2 in each dimension.
/// Border handling: BORDER_REFLECT_101 (default OpenCV border for pyrDown).
pub fn pyr_down_gray(src: &[f32], sw: u32, sh: u32) -> (Vec<f32>, u32, u32) {
    let sw = sw as usize;
    let sh = sh as usize;
    let dw = sw.div_ceil(2);
    let dh = sh.div_ceil(2);
    let sws = sw as isize;
    let shs = sh as isize;

    // Horizontal pass → temp (sh × dw)
    let mut tmp = vec![0.0f32; sh * dw];
    for y in 0..sh {
        for dx in 0..dw {
            let sx = (dx * 2) as isize;
            let mut sum = 0.0f32;
            for k in 0..5isize {
                let col = reflect101_safe(sx + k - 2, sws);
                sum += PYR_KERNEL[k as usize] * src[y * sw + col];
            }
            tmp[y * dw + dx] = sum;
        }
    }

    // Vertical pass → dst (dh × dw)
    let mut dst = vec![0.0f32; dh * dw];
    for dy in 0..dh {
        let sy = (dy * 2) as isize;
        for x in 0..dw {
            let mut sum = 0.0f32;
            for k in 0..5isize {
                let row = reflect101_safe(sy + k - 2, shs);
                sum += PYR_KERNEL[k as usize] * tmp[row * dw + x];
            }
            dst[dy * dw + x] = sum;
        }
    }

    (dst, dw as u32, dh as u32)
}

/// pyrDown for 3-channel f32 image (interleaved RGB).
pub fn pyr_down_rgb(src: &[f32], sw: u32, sh: u32) -> (Vec<f32>, u32, u32) {
    let sw = sw as usize;
    let sh = sh as usize;
    let dw = sw.div_ceil(2);
    let dh = sh.div_ceil(2);
    let sws = sw as isize;
    let shs = sh as isize;

    // Horizontal pass
    let mut tmp = vec![0.0f32; sh * dw * 3];
    for y in 0..sh {
        for dx in 0..dw {
            let sx = (dx * 2) as isize;
            let mut sum = [0.0f32; 3];
            for k in 0..5isize {
                let col = reflect101_safe(sx + k - 2, sws);
                let wt = PYR_KERNEL[k as usize];
                sum[0] += wt * src[(y * sw + col) * 3];
                sum[1] += wt * src[(y * sw + col) * 3 + 1];
                sum[2] += wt * src[(y * sw + col) * 3 + 2];
            }
            tmp[(y * dw + dx) * 3] = sum[0];
            tmp[(y * dw + dx) * 3 + 1] = sum[1];
            tmp[(y * dw + dx) * 3 + 2] = sum[2];
        }
    }

    // Vertical pass
    let mut dst = vec![0.0f32; dh * dw * 3];
    for dy in 0..dh {
        let sy = (dy * 2) as isize;
        for x in 0..dw {
            let mut sum = [0.0f32; 3];
            for k in 0..5isize {
                let row = reflect101_safe(sy + k - 2, shs);
                let wt = PYR_KERNEL[k as usize];
                sum[0] += wt * tmp[(row * dw + x) * 3];
                sum[1] += wt * tmp[(row * dw + x) * 3 + 1];
                sum[2] += wt * tmp[(row * dw + x) * 3 + 2];
            }
            dst[(dy * dw + x) * 3] = sum[0];
            dst[(dy * dw + x) * 3 + 1] = sum[1];
            dst[(dy * dw + x) * 3 + 2] = sum[2];
        }
    }

    (dst, dw as u32, dh as u32)
}

/// Gaussian pyramid upsample: upsample by 2 + blur.
///
/// Inserts zeros between pixels, then applies 5x5 Gaussian * 4.
/// Output is w*2 x h*2. Matches `cv2.pyrUp`.
pub fn pyr_up(pixels: &[u8], info: &ImageInfo) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    if info.format != PixelFormat::Gray8 {
        return Err(ImageError::UnsupportedFormat(
            "pyr_up requires Gray8".into(),
        ));
    }
    let w = info.width as usize;
    let h = info.height as usize;
    let ow = w * 2;
    let oh = h * 2;

    // Upsample: insert zeros
    let mut upsampled = vec![0i32; ow * oh];
    for y in 0..h {
        for x in 0..w {
            upsampled[y * 2 * ow + x * 2] = pixels[y * w + x] as i32 * 4; // *4 to compensate for zero-insertion
        }
    }

    // 5x5 Gaussian blur on upsampled
    let kernel_1d: [i32; 5] = [1, 4, 6, 4, 1];

    // Horizontal pass
    let mut temp = vec![0i32; ow * oh];
    for y in 0..oh {
        for x in 0..ow {
            let mut sum: i32 = 0;
            for k in 0..5i32 {
                let sx = reflect101(x as isize + k as isize - 2, ow as isize) as usize;
                sum += upsampled[y * ow + sx] * kernel_1d[k as usize];
            }
            temp[y * ow + x] = sum;
        }
    }

    // Vertical pass
    let mut output = vec![0u8; ow * oh];
    for y in 0..oh {
        for x in 0..ow {
            let mut sum: i32 = 0;
            for k in 0..5i32 {
                let sy = reflect101(y as isize + k as isize - 2, oh as isize) as usize;
                sum += temp[sy * ow + x] * kernel_1d[k as usize];
            }
            output[y * ow + x] = ((sum + 128) >> 8).clamp(0, 255) as u8;
        }
    }

    let new_info = ImageInfo {
        width: ow as u32,
        height: oh as u32,
        format: info.format,
        color_space: info.color_space,
    };
    Ok((output, new_info))
}

/// pyrUp for single-channel f32 — upsample by 2 then apply 5×5 Gaussian × 4.
#[allow(dead_code)] // reserved for pyramid reconstruction path
pub fn pyr_up_gray(src: &[f32], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<f32> {
    let sw = sw as usize;
    let sh = sh as usize;
    let dw = dw as usize;
    let dh = dh as usize;
    let dws = dw as isize;
    let dhs = dh as isize;

    // Insert zeros: place src pixels at even positions, zeros at odd
    let mut upsampled = vec![0.0f32; dh * dw];
    for y in 0..sh {
        for x in 0..sw {
            if y * 2 < dh && x * 2 < dw {
                upsampled[y * 2 * dw + x * 2] = src[y * sw + x] * 4.0;
            }
        }
    }

    // Apply 5×5 Gaussian filter (separable)
    let mut tmp = vec![0.0f32; dh * dw];
    for y in 0..dh {
        for x in 0..dw {
            let mut sum = 0.0f32;
            for k in 0..5isize {
                let col = reflect101_safe(x as isize + k - 2, dws);
                sum += PYR_KERNEL[k as usize] * upsampled[y * dw + col];
            }
            tmp[y * dw + x] = sum;
        }
    }

    let mut dst = vec![0.0f32; dh * dw];
    for y in 0..dh {
        for x in 0..dw {
            let mut sum = 0.0f32;
            for k in 0..5isize {
                let row = reflect101_safe(y as isize + k - 2, dhs);
                sum += PYR_KERNEL[k as usize] * tmp[row * dw + x];
            }
            dst[y * dw + x] = sum;
        }
    }

    dst
}

/// pyrUp for 3-channel f32 (interleaved RGB).
pub fn pyr_up_rgb(src: &[f32], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<f32> {
    let sw = sw as usize;
    let sh = sh as usize;
    let dw = dw as usize;
    let dh = dh as usize;
    let dws = dw as isize;
    let dhs = dh as isize;

    // Insert zeros with 4× scaling at even positions
    let mut upsampled = vec![0.0f32; dh * dw * 3];
    for y in 0..sh {
        for x in 0..sw {
            if y * 2 < dh && x * 2 < dw {
                let di = (y * 2 * dw + x * 2) * 3;
                let si = (y * sw + x) * 3;
                upsampled[di] = src[si] * 4.0;
                upsampled[di + 1] = src[si + 1] * 4.0;
                upsampled[di + 2] = src[si + 2] * 4.0;
            }
        }
    }

    // Horizontal pass
    let mut tmp = vec![0.0f32; dh * dw * 3];
    for y in 0..dh {
        for x in 0..dw {
            let mut sum = [0.0f32; 3];
            for k in 0..5isize {
                let col = reflect101_safe(x as isize + k - 2, dws);
                let wt = PYR_KERNEL[k as usize];
                sum[0] += wt * upsampled[(y * dw + col) * 3];
                sum[1] += wt * upsampled[(y * dw + col) * 3 + 1];
                sum[2] += wt * upsampled[(y * dw + col) * 3 + 2];
            }
            tmp[(y * dw + x) * 3] = sum[0];
            tmp[(y * dw + x) * 3 + 1] = sum[1];
            tmp[(y * dw + x) * 3 + 2] = sum[2];
        }
    }

    // Vertical pass
    let mut dst = vec![0.0f32; dh * dw * 3];
    for y in 0..dh {
        for x in 0..dw {
            let mut sum = [0.0f32; 3];
            for k in 0..5isize {
                let row = reflect101_safe(y as isize + k - 2, dhs);
                let wt = PYR_KERNEL[k as usize];
                sum[0] += wt * tmp[(row * dw + x) * 3];
                sum[1] += wt * tmp[(row * dw + x) * 3 + 1];
                sum[2] += wt * tmp[(row * dw + x) * 3 + 2];
            }
            dst[(y * dw + x) * 3] = sum[0];
            dst[(y * dw + x) * 3 + 1] = sum[1];
            dst[(y * dw + x) * 3 + 2] = sum[2];
        }
    }

    dst
}

/// Process a single channel through the Local Laplacian pyramid.
pub fn pyramid_detail_remap_channel(
    input: &[f32],
    w: usize,
    h: usize,
    levels: usize,
    sigma: f32,
) -> Vec<f32> {
    // Build Gaussian pyramid
    let mut gauss_pyramid = vec![input.to_vec()];
    let mut cw = w;
    let mut ch = h;
    for _ in 1..levels {
        let prev = gauss_pyramid.last().unwrap();
        let (nw, nh) = (cw.div_ceil(2), ch.div_ceil(2));
        let downsampled = downsample_2x(prev, cw, ch);
        gauss_pyramid.push(downsampled);
        cw = nw;
        ch = nh;
    }

    // Build output Laplacian pyramid with remapped detail
    let mut output_laplacian: Vec<Vec<f32>> = Vec::with_capacity(levels);
    cw = w;
    ch = h;

    for level in 0..levels - 1 {
        let (nw, nh) = (cw.div_ceil(2), ch.div_ceil(2));

        // Laplacian = current level - upsampled(next level)
        let upsampled = upsample_2x(&gauss_pyramid[level + 1], nw, nh, cw, ch);
        let mut laplacian = vec![0.0f32; cw * ch];
        #[allow(clippy::needless_range_loop)]
        for i in 0..cw * ch {
            laplacian[i] = gauss_pyramid[level][i] - upsampled[i];
        }

        // Remap detail: attenuate or amplify based on sigma
        // Enhancement: small sigma compresses large gradients, preserves small detail
        // Smoothing: large sigma suppresses small detail
        for laplacian_val in laplacian.iter_mut().take(cw * ch) {
            let d = *laplacian_val;
            // Sigmoidal remapping: f(d) = d * (sigma / (sigma + |d|))
            // sigma < 1: enhances small detail (compresses large)
            // sigma > 1: smooths (suppresses small detail)
            *laplacian_val = d * sigma / (sigma + d.abs());
        }

        output_laplacian.push(laplacian);
        cw = nw;
        ch = nh;
    }

    // Coarsest level is kept as-is (DC component)
    output_laplacian.push(gauss_pyramid[levels - 1].clone());

    // Reconstruct from Laplacian pyramid
    let mut reconstructed = output_laplacian[levels - 1].clone();
    let _ = gauss_pyramid[levels - 1].len(); // dims recalculated below

    // Recompute dimensions for each level
    let mut dims: Vec<(usize, usize)> = Vec::with_capacity(levels);
    let (mut tw, mut th) = (w, h);
    for _ in 0..levels {
        dims.push((tw, th));
        tw = tw.div_ceil(2);
        th = th.div_ceil(2);
    }

    for level in (0..levels - 1).rev() {
        let (target_w, target_h) = dims[level];
        let (src_w, src_h) = dims[level + 1];
        let upsampled = upsample_2x(&reconstructed, src_w, src_h, target_w, target_h);
        reconstructed = vec![0.0f32; target_w * target_h];
        for i in 0..target_w * target_h {
            reconstructed[i] = (upsampled[i] + output_laplacian[level][i]).clamp(0.0, 1.0);
        }
    }

    reconstructed
}

/// Upsample by 2x using bilinear interpolation.
pub fn upsample_2x(data: &[f32], sw: usize, sh: usize, tw: usize, th: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; tw * th];
    for y in 0..th {
        for x in 0..tw {
            let sx = x as f32 / tw as f32 * sw as f32;
            let sy = y as f32 / th as f32 * sh as f32;
            let x0 = (sx as usize).min(sw - 1);
            let y0 = (sy as usize).min(sh - 1);
            let x1 = (x0 + 1).min(sw - 1);
            let y1 = (y0 + 1).min(sh - 1);
            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;
            out[y * tw + x] = data[y0 * sw + x0] * (1.0 - fx) * (1.0 - fy)
                + data[y0 * sw + x1] * fx * (1.0 - fy)
                + data[y1 * sw + x0] * (1.0 - fx) * fy
                + data[y1 * sw + x1] * fx * fy;
        }
    }
    out
}

