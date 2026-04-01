//! Filter: pixelate (category: effect)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

#[rasmcore_macros::register_filter(
    name = "pixelate",
    category = "effect",
    reference = "block mosaic pixelation"
)]
pub fn pixelate(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &PixelateParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let block_size = config.block_size;

    validate_format(info.format)?;

    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |px, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(px.to_vec());
            pixelate(r, &mut u, i8, config)
        });
    }

    let bs = block_size.max(1) as usize;
    let w = info.width as usize;
    let h = info.height as usize;
    let ch = channels(info.format);
    let mut out = vec![0u8; pixels.len()];

    let mut by = 0;
    while by < h {
        let bh = bs.min(h - by);
        let mut bx = 0;
        while bx < w {
            let bw = bs.min(w - bx);
            let count = bw * bh;

            // Accumulate channel sums
            let mut sums = [0u32; 4]; // max 4 channels
            for row in by..(by + bh) {
                for col in bx..(bx + bw) {
                    let off = (row * w + col) * ch;
                    for c in 0..ch {
                        sums[c] += pixels[off + c] as u32;
                    }
                }
            }

            // Compute averages
            let mut avg = [0u8; 4];
            for c in 0..ch {
                avg[c] = ((sums[c] + count as u32 / 2) / count as u32) as u8;
            }

            // Fill block with average
            for row in by..(by + bh) {
                for col in bx..(bx + bw) {
                    let off = (row * w + col) * ch;
                    out[off..off + ch].copy_from_slice(&avg[..ch]);
                }
            }

            bx += bs;
        }
        by += bs;
    }

    Ok(out)
}
