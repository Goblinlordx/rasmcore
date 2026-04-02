//! Filter: rank_filter (category: spatial)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;


#[derive(rasmcore_macros::Filter, Clone)]
#[filter(
    name = "rank_filter",
    category = "spatial",
    reference = "generalized rank/order statistic filter"
)]
pub struct RankFilterParams {
    pub radius: u32,
    pub rank: f32,
}

impl CpuFilter for RankFilterParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
        let pixels = upstream(request)?;
        let info = &ImageInfo {
            width: request.width,
            height: request.height,
            ..*info
        };
        let pixels = pixels.as_slice();
        let radius = self.radius;
        let rank = self.rank;

        if radius == 0 {
            return Ok(pixels.to_vec());
        }
        validate_format(info.format)?;

        let w = info.width as usize;
        let h = info.height as usize;
        let ch = channels(info.format);
        let r = radius as i32;
        let diameter = (2 * r + 1) as usize;
        let window_size = diameter * diameter;
        let rank_clamped = rank.clamp(0.0, 1.0);

        // Target position in sorted order: rank 0.0 -> index 0, rank 1.0 -> last
        let target = ((window_size - 1) as f32 * rank_clamped).round() as usize;

        if !is_16bit(info.format) && !is_float(info.format) {
            // Fast u8 path using histogram sliding window
            let mut out = vec![0u8; pixels.len()];

            for c in 0..ch {
                for y in 0..h {
                    let mut hist = [0u32; 256];

                    for ky in -r..=r {
                        let sy = reflect(y as i32 + ky, h);
                        for kx in -r..=r {
                            let sx = reflect(kx, w);
                            hist[pixels[(sy * w + sx) * ch + c] as usize] += 1;
                        }
                    }

                    out[y * w * ch + c] = find_rank_in_hist(&hist, target);

                    for x in 1..w {
                        let old_x = x as i32 - r - 1;
                        for ky in -r..=r {
                            let sy = reflect(y as i32 + ky, h);
                            let sx = reflect(old_x, w);
                            hist[pixels[(sy * w + sx) * ch + c] as usize] -= 1;
                        }

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
            return Ok(out);
        }

        // f32-native path for u16/f16/f32 formats — sorting-based
        let samples = pixels_to_f32_samples(pixels, info.format);
        let mut out_samples = vec![0.0f32; samples.len()];
        let mut window = Vec::with_capacity(window_size);

        for c in 0..ch {
            for y in 0..h {
                for x in 0..w {
                    window.clear();
                    for ky in -r..=r {
                        let sy = reflect(y as i32 + ky, h);
                        for kx in -r..=r {
                            let sx = reflect(x as i32 + kx, w);
                            window.push(samples[(sy * w + sx) * ch + c]);
                        }
                    }
                    window.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    out_samples[(y * w + x) * ch + c] = window[target];
                }
            }
        }
        Ok(f32_samples_to_pixels(&out_samples, info.format))
    }
}
