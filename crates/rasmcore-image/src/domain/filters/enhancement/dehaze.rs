//! Filter: dehaze (category: enhancement)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Dehaze an image using the dark channel prior (He et al. 2009).
///
/// Estimates atmospheric light and transmission from the dark channel (minimum
/// over color channels in a local patch), refines with guided filter, then
/// recovers the scene: `J = (I - A) / max(t, t_min) + A`.
///
/// - `patch_radius`: local patch size for dark channel (typical: 7-15)
/// - `omega`: haze removal strength 0.0-1.0 (typical: 0.95)
/// - `t_min`: minimum transmission to avoid noise amplification (typical: 0.1)
///
/// Parameters for dehaze (dark channel prior).
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "dehaze", category = "enhancement", reference = "He et al. 2009 dark channel prior dehazing")]
pub struct DehazeParams {
    /// Local patch size for dark channel (typical: 7-15)
    #[param(min = 1, max = 30, step = 1, default = 7)]
    pub patch_radius: u32,
    /// Haze removal strength 0.0-1.0
    #[param(min = 0.0, max = 1.0, step = 0.05, default = 0.95)]
    pub omega: f32,
    /// Minimum transmission to avoid noise amplification
    #[param(min = 0.01, max = 0.5, step = 0.01, default = 0.1)]
    pub t_min: f32,
}

impl CpuFilter for DehazeParams {
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
    let patch_radius = self.patch_radius;
    let omega = self.omega;
    let t_min = self.t_min;

    validate_format(info.format)?;
    let (w, h) = (info.width as usize, info.height as usize);
    let channels = match info.format {
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "dehaze requires RGB8 or RGBA8".into(),
            ));
        }
    };
    let n = w * h;
    let r = patch_radius as usize;

    // Step 1: Compute dark channel — min over RGB in local patch
    let mut dark_channel = vec![0.0f32; n];
    for y in 0..h {
        for x in 0..w {
            let mut min_val = f32::MAX;
            let y0 = y.saturating_sub(r);
            let y1 = (y + r + 1).min(h);
            let x0 = x.saturating_sub(r);
            let x1 = (x + r + 1).min(w);
            for py in y0..y1 {
                for px in x0..x1 {
                    let idx = (py * w + px) * channels;
                    let r_val = pixels[idx] as f32 / 255.0;
                    let g_val = pixels[idx + 1] as f32 / 255.0;
                    let b_val = pixels[idx + 2] as f32 / 255.0;
                    let ch_min = r_val.min(g_val).min(b_val);
                    min_val = min_val.min(ch_min);
                }
            }
            dark_channel[y * w + x] = min_val;
        }
    }

    // Step 2: Estimate atmospheric light — brightest 0.1% of dark channel pixels
    let mut dc_indexed: Vec<(usize, f32)> = dark_channel
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    dc_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top_count = (n as f32 * 0.001).max(1.0) as usize;
    let mut atm = [0.0f32; 3];
    let mut max_intensity = 0.0f32;
    for &(idx, _) in dc_indexed.iter().take(top_count) {
        let pi = idx * channels;
        let intensity = pixels[pi] as f32 + pixels[pi + 1] as f32 + pixels[pi + 2] as f32;
        if intensity > max_intensity {
            max_intensity = intensity;
            atm[0] = pixels[pi] as f32 / 255.0;
            atm[1] = pixels[pi + 1] as f32 / 255.0;
            atm[2] = pixels[pi + 2] as f32 / 255.0;
        }
    }

    // Step 3: Estimate transmission — t(x) = 1 - omega * dark_channel(I/A)
    let mut transmission = vec![0.0f32; n];
    for y in 0..h {
        for x in 0..w {
            let mut min_val = f32::MAX;
            let y0 = y.saturating_sub(r);
            let y1 = (y + r + 1).min(h);
            let x0 = x.saturating_sub(r);
            let x1 = (x + r + 1).min(w);
            for py in y0..y1 {
                for px in x0..x1 {
                    let idx = (py * w + px) * channels;
                    let nr = (pixels[idx] as f32 / 255.0) / atm[0].max(0.001);
                    let ng = (pixels[idx + 1] as f32 / 255.0) / atm[1].max(0.001);
                    let nb = (pixels[idx + 2] as f32 / 255.0) / atm[2].max(0.001);
                    min_val = min_val.min(nr.min(ng).min(nb));
                }
            }
            transmission[y * w + x] = (1.0 - omega * min_val).max(t_min);
        }
    }

    // Step 4: Refine transmission with guided filter (use grayscale as guide)
    // Convert transmission to u8, apply guided filter, convert back
    let t_u8: Vec<u8> = transmission
        .iter()
        .map(|&t| (t * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect();
    let gray_info = ImageInfo {
        width: info.width,
        height: info.height,
        format: PixelFormat::Gray8,
        color_space: info.color_space,
    };
    let refined_u8 = guided_filter_impl(
        &t_u8,
        &gray_info,
        &GuidedFilterParams {
            radius: patch_radius.min(15),
            epsilon: 0.001,
        },
    )?;
    let refined: Vec<f32> = refined_u8.iter().map(|&v| v as f32 / 255.0).collect();

    // Step 5: Recover scene — J = (I - A) / max(t, t_min) + A
    let mut result = vec![0u8; pixels.len()];

    #[cfg(target_arch = "wasm32")]
    {
        use std::arch::wasm32::*;
        let inv_255 = f32x4_splat(1.0 / 255.0);
        let scale_255 = f32x4_splat(255.0);
        let half = f32x4_splat(0.5);
        let zero = f32x4_splat(0.0);
        let t_min_v = f32x4_splat(t_min);

        // Process one pixel at a time using f32x4 for RGB channels + padding
        // This vectorizes the 3-channel arithmetic (R, G, B, 0) in one SIMD op
        let atm_v = f32x4(atm[0], atm[1], atm[2], 0.0);

        for i in 0..n {
            let t = refined[i].max(t_min);
            let inv_t = f32x4_splat(1.0 / t);
            let pi = i * channels;

            // Load RGB as f32x4
            let px = f32x4(
                pixels[pi] as f32,
                pixels[pi + 1] as f32,
                pixels[pi + 2] as f32,
                0.0,
            );
            // ic = px / 255.0
            let ic = f32x4_mul(px, inv_255);
            // jc = (ic - atm) / t + atm = (ic - atm) * inv_t + atm
            let diff = f32x4_sub(ic, atm_v);
            let jc = f32x4_add(f32x4_mul(diff, inv_t), atm_v);
            // Convert back: round(jc * 255), clamp [0, 255]
            let out = f32x4_min(
                scale_255,
                f32x4_max(zero, f32x4_add(f32x4_mul(jc, scale_255), half)),
            );

            result[pi] = f32x4_extract_lane::<0>(out) as u8;
            result[pi + 1] = f32x4_extract_lane::<1>(out) as u8;
            result[pi + 2] = f32x4_extract_lane::<2>(out) as u8;
            if channels == 4 {
                result[pi + 3] = pixels[pi + 3];
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let t = refined[i].max(t_min);
            let inv_t = 1.0 / t;
            let pi = i * channels;
            for c in 0..3 {
                let ic = pixels[pi + c] as f32 / 255.0;
                let jc = (ic - atm[c]) * inv_t + atm[c];
                result[pi + c] = (jc * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
            if channels == 4 {
                result[pi + 3] = pixels[pi + 3];
            }
        }
    }

    Ok(result)
}
}

