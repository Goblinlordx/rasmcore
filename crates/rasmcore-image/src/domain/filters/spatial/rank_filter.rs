//! Filter: rank_filter (category: spatial)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

#[rasmcore_macros::register_filter(
    name = "rank_filter",
    category = "spatial",
    reference = "generalized rank/order statistic filter"
)]
pub fn rank_filter(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &RankFilterParams,
) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let radius = config.radius;
    let rank = config.rank;

    if radius == 0 {
        return Ok(pixels.to_vec());
    }
    validate_format(info.format)?;

    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |px, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(px.to_vec());
            rank_filter(r, &mut u, i8, config)
        });
    }

    let w = info.width as usize;
    let h = info.height as usize;
    let ch = channels(info.format);
    let r = radius as i32;
    let diameter = (2 * r + 1) as usize;
    let window_size = diameter * diameter;
    let rank_clamped = rank.clamp(0.0, 1.0);

    // Target position in sorted order: rank 0.0 → index 0, rank 1.0 → last
    let target = ((window_size - 1) as f32 * rank_clamped).round() as usize;

    let mut out = vec![0u8; pixels.len()];

    for c in 0..ch {
        for y in 0..h {
            let mut hist = [0u32; 256];

            // Initialize histogram for first window in row
            for ky in -r..=r {
                let sy = reflect(y as i32 + ky, h);
                for kx in -r..=r {
                    let sx = reflect(kx, w);
                    hist[pixels[(sy * w + sx) * ch + c] as usize] += 1;
                }
            }

            // Find rank for first pixel
            out[y * w * ch + c] = find_rank_in_hist(&hist, target);

            // Slide right
            for x in 1..w {
                // Remove leftmost column
                let old_x = x as i32 - r - 1;
                for ky in -r..=r {
                    let sy = reflect(y as i32 + ky, h);
                    let sx = reflect(old_x, w);
                    hist[pixels[(sy * w + sx) * ch + c] as usize] -= 1;
                }

                // Add rightmost column
                let new_x = x as i32 + r;
                for ky in -r..=r {
                    let sy = reflect(y as i32 + ky, h);
                    let sx = reflect(new_x, w);
                    hist[pixels[(sy * w + sx) * ch + c] as usize] += 1;
                }

                out[(y * w + x) * ch + c] = find_rank_in_hist(&hist, target);
            }
        }
    }
    Ok(out)
}
