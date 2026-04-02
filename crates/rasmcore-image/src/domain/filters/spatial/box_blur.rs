//! Filter: box_blur (category: spatial)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Box blur — separable uniform-weight kernel with O(1) running-sum.
///
/// Each output pixel is the mean of all pixels in a (2r+1)x(2r+1) window.
/// Implemented as two separable passes (horizontal then vertical), each
/// using a running sum: add the entering pixel, subtract the leaving pixel.
/// Cost is O(1) per pixel regardless of radius.
///
/// Reference: matches Photoshop's Box Blur and OpenCV's cv2.blur().

/// Parameters for box blur (uniform-weight kernel).
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct BoxBlurParams {
    /// Blur radius in pixels (kernel width = 2*radius + 1)
    #[param(min = 1, max = 100, step = 1, default = 3)]
    pub radius: u32,
}

impl InputRectProvider for BoxBlurParams {
    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> Rect {
        let overlap = self.radius;
        output.expand_uniform(overlap, bounds_w, bounds_h)
    }
}

#[rasmcore_macros::register_filter(
    name = "box_blur",
    category = "spatial",
    group = "blur",
    variant = "box",
    reference = "Photoshop Box Blur / OpenCV cv2.blur"
)]
pub fn box_blur(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &BoxBlurParams,
) -> Result<Vec<u8>, ImageError> {
    let overlap = config.radius;
    let expanded = request.expand_uniform(overlap, info.width, info.height);
    let pixels = upstream(expanded)?;
    let info = &ImageInfo {
        width: expanded.width,
        height: expanded.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let radius = config.radius;
    validate_format(info.format)?;

    if radius == 0 {
        return Ok(pixels.to_vec());
    }

    if is_16bit(info.format) {
        let result = process_via_8bit(pixels, info, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            box_blur(r, &mut u, i8, config)
        })?;
        return Ok(crop_to_request(&result, expanded, request, info.format));
    }

    let w = info.width as usize;
    let h = info.height as usize;
    let ch = crate::domain::types::bytes_per_pixel(info.format) as usize;
    let r = radius as usize;
    let diam = (2 * r + 1) as u32;

    // Helper: load pixel channels into [u32; 4] (zero-padded for ch < 4)
    #[inline(always)]
    fn load_pixel(src: &[u8], offset: usize, ch: usize) -> [u32; 4] {
        let mut p = [0u32; 4];
        for c in 0..ch.min(4) {
            p[c] = src[offset + c] as u32;
        }
        p
    }

    // Helper: store [u32; 4] as pixel channels, preserving alpha if RGBA
    #[inline(always)]
    fn store_pixel(dst: &mut [u8], offset: usize, vals: [u32; 4], ch: usize, alpha_src: &[u8]) {
        let color_ch = if ch == 4 { 3 } else { ch }; // skip alpha for RGBA
        for c in 0..color_ch {
            dst[offset + c] = vals[c] as u8;
        }
        if ch == 4 {
            dst[offset + 3] = alpha_src[offset + 3]; // preserve alpha
        }
    }

    // Horizontal pass — running sum across all channels simultaneously
    let mut hpass = vec![0u8; pixels.len()];
    let color_ch = if ch == 4 { 3 } else { ch };

    for y in 0..h {
        let row = y * w * ch;
        let mut sums = [0u32; 4];

        // Initialize sums for first pixel (clamped boundary)
        for k in 0..=r {
            let sx = k.min(w - 1);
            let p = load_pixel(pixels, row + sx * ch, color_ch);
            for c in 0..color_ch {
                sums[c] += p[c];
            }
        }
        for _ in 0..r {
            let p = load_pixel(pixels, row, color_ch);
            for c in 0..color_ch {
                sums[c] += p[c];
            }
        }
        // Store first pixel
        let mut mean = [0u32; 4];
        for c in 0..color_ch {
            mean[c] = sums[c] / diam;
        }
        store_pixel(&mut hpass, row, mean, ch, pixels);

        // Slide across row
        for x in 1..w {
            let add_x = (x + r).min(w - 1);
            let add = load_pixel(pixels, row + add_x * ch, color_ch);
            for c in 0..color_ch {
                sums[c] += add[c];
            }

            if x <= r {
                let sub = load_pixel(pixels, row, color_ch);
                for c in 0..color_ch {
                    sums[c] -= sub[c];
                }
            } else {
                let sub_x = x - r - 1;
                let sub = load_pixel(pixels, row + sub_x * ch, color_ch);
                for c in 0..color_ch {
                    sums[c] -= sub[c];
                }
            }

            for c in 0..color_ch {
                mean[c] = sums[c] / diam;
            }
            store_pixel(&mut hpass, row + x * ch, mean, ch, pixels);
        }
    }

    // Vertical pass — running sum down columns, all channels simultaneously
    let mut out = vec![0u8; pixels.len()];

    for x in 0..w {
        let mut sums = [0u32; 4];

        for k in 0..=r {
            let sy = k.min(h - 1);
            let p = load_pixel(&hpass, (sy * w + x) * ch, color_ch);
            for c in 0..color_ch {
                sums[c] += p[c];
            }
        }
        for _ in 0..r {
            let p = load_pixel(&hpass, x * ch, color_ch);
            for c in 0..color_ch {
                sums[c] += p[c];
            }
        }
        let mut mean = [0u32; 4];
        for c in 0..color_ch {
            mean[c] = sums[c] / diam;
        }
        store_pixel(&mut out, x * ch, mean, ch, &hpass);

        for y in 1..h {
            let add_y = (y + r).min(h - 1);
            let add = load_pixel(&hpass, (add_y * w + x) * ch, color_ch);
            for c in 0..color_ch {
                sums[c] += add[c];
            }

            if y <= r {
                let sub = load_pixel(&hpass, x * ch, color_ch);
                for c in 0..color_ch {
                    sums[c] -= sub[c];
                }
            } else {
                let sub_y = y - r - 1;
                let sub = load_pixel(&hpass, (sub_y * w + x) * ch, color_ch);
                for c in 0..color_ch {
                    sums[c] -= sub[c];
                }
            }

            for c in 0..color_ch {
                mean[c] = sums[c] / diam;
            }
            store_pixel(&mut out, (y * w + x) * ch, mean, ch, &hpass);
        }
    }

    Ok(crop_to_request(&out, expanded, request, info.format))
}
