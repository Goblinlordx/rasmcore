//! Filter: retinex_ssr (category: enhancement)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Single-Scale Retinex (SSR).
///
/// `R(x,y) = log(I(x,y)) - log(G(x,y,sigma) * I(x,y))`
///
/// Enhances local contrast by removing the illumination component estimated
/// via Gaussian blur. Output is normalized to [0, 255].
///
/// - `sigma`: Gaussian scale (typical: 80.0 for general enhancement)
///
/// Reference: Jobson, Rahman, Woodell — "Properties and Performance of a
/// Center/Surround Retinex" (IEEE Trans. Image Processing, 1997)

/// Parameters for single-scale Retinex.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "retinex_ssr", category = "enhancement", group = "retinex", variant = "ssr", reference = "Land 1977 single-scale Retinex")]
pub struct RetinexSsrParams {
    /// Gaussian scale for surround function
    #[param(
        min = 10.0,
        max = 300.0,
        step = 10.0,
        default = 80.0,
        hint = "rc.log_slider"
    )]
    pub sigma: f32,
}

impl CpuFilter for RetinexSsrParams {
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
    let sigma = self.sigma;

    validate_format(info.format)?;
    let channels = match info.format {
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "retinex requires RGB8 or RGBA8".into(),
            ));
        }
    };
    let n = (info.width as usize) * (info.height as usize);

    // Gaussian blur for surround function (OpenCV-compatible for reference alignment)
    let blurred = {
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(pixels.to_vec());
        GaussianBlurCvParams { sigma }.compute(r, &mut u, info)?
    };

    // Compute log(I/blur(I)) per channel using log(a/b) identity, then normalize
    let mut retinex = vec![0.0f32; n * 3];
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;

    #[cfg(target_arch = "wasm32")]
    {
        use std::arch::wasm32::*;
        // Process 4 values at a time with f32x4.
        // We process the retinex buffer as a flat array of n*3 f32 values,
        // computing ln(orig/surround) for each. The u8->f32 conversion and
        // max(1.0) are vectorized.
        let one = f32x4_splat(1.0);
        let total = n * 3;
        let simd_end = total & !3; // round down to multiple of 4

        // Build f32 arrays for orig and surround (contiguous RGB, skip alpha)
        let mut orig_f32 = vec![0.0f32; total];
        let mut surr_f32 = vec![0.0f32; total];
        for i in 0..n {
            let pi = i * channels;
            for c in 0..3 {
                orig_f32[i * 3 + c] = (pixels[pi + c] as f32).max(1.0);
                surr_f32[i * 3 + c] = (blurred[pi + c] as f32).max(1.0);
            }
        }

        // Vectorized ln(orig/surround) — 4 values at a time
        let mut i = 0;
        while i < simd_end {
            // SAFETY: i..i+4 is within bounds (simd_end rounded down to multiple of 4).
            // Pointers from Vec<f32> slices are valid and v128_load handles unaligned.
            let o = unsafe { v128_load(orig_f32[i..].as_ptr() as *const v128) };
            let s = unsafe { v128_load(surr_f32[i..].as_ptr() as *const v128) };
            let ratio = f32x4_div(o, s);
            let mut vals = [0.0f32; 4];
            vals[0] = f32x4_extract_lane::<0>(ratio).ln();
            vals[1] = f32x4_extract_lane::<1>(ratio).ln();
            vals[2] = f32x4_extract_lane::<2>(ratio).ln();
            vals[3] = f32x4_extract_lane::<3>(ratio).ln();
            // SAFETY: same bounds guarantee as above.
            unsafe {
                v128_store(
                    retinex[i..].as_mut_ptr() as *mut v128,
                    f32x4(vals[0], vals[1], vals[2], vals[3]),
                );
            }
            for &v in &vals {
                min_val = min_val.min(v);
                max_val = max_val.max(v);
            }
            i += 4;
        }
        // Remainder
        for j in simd_end..total {
            let r = (orig_f32[j] / surr_f32[j]).ln();
            retinex[j] = r;
            min_val = min_val.min(r);
            max_val = max_val.max(r);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        for i in 0..n {
            let pi = i * channels;
            for c in 0..3 {
                let orig = (pixels[pi + c] as f32).max(1.0);
                let surround = (blurred[pi + c] as f32).max(1.0);
                let r = (orig / surround).ln();
                retinex[i * 3 + c] = r;
                min_val = min_val.min(r);
                max_val = max_val.max(r);
            }
        }
    }

    // Normalize to [0, 255]
    let range = (max_val - min_val).max(1e-6);
    let mut result = vec![0u8; pixels.len()];
    let inv_range_255 = 255.0 / range;

    #[cfg(target_arch = "wasm32")]
    {
        use std::arch::wasm32::*;
        let min_v = f32x4_splat(min_val);
        let scale = f32x4_splat(inv_range_255);
        let zero = f32x4_splat(0.0);
        let max_255 = f32x4_splat(255.0);
        let half = f32x4_splat(0.5);

        let total = n * 3;
        let simd_end = total & !3;
        let mut j = 0;
        let mut ri = 0; // retinex index
        // Process 4 retinex values at a time, write to result (skip alpha for RGBA)
        if channels == 3 {
            // RGB: retinex layout matches pixel layout
            while ri < simd_end {
                // SAFETY: ri..ri+4 within bounds (simd_end rounded down).
                let r = unsafe { v128_load(retinex[ri..].as_ptr() as *const v128) };
                let v = f32x4_mul(f32x4_sub(r, min_v), scale);
                let v = f32x4_max(zero, f32x4_min(max_255, f32x4_add(v, half)));
                result[ri] = f32x4_extract_lane::<0>(v) as u8;
                result[ri + 1] = f32x4_extract_lane::<1>(v) as u8;
                result[ri + 2] = f32x4_extract_lane::<2>(v) as u8;
                result[ri + 3] = f32x4_extract_lane::<3>(v) as u8;
                ri += 4;
            }
            for k in simd_end..total {
                let v = (retinex[k] - min_val) * inv_range_255;
                result[k] = (v + 0.5).clamp(0.0, 255.0) as u8;
            }
        } else {
            // RGBA: write 3 RGB channels, copy alpha
            for i in 0..n {
                let pi = i * 4;
                let ri_base = i * 3;
                for c in 0..3 {
                    let v = (retinex[ri_base + c] - min_val) * inv_range_255;
                    result[pi + c] = (v + 0.5).clamp(0.0, 255.0) as u8;
                }
                result[pi + 3] = pixels[pi + 3];
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        for i in 0..n {
            let pi = i * channels;
            for c in 0..3 {
                let v = (retinex[i * 3 + c] - min_val) * inv_range_255;
                result[pi + c] = (v + 0.5).clamp(0.0, 255.0) as u8;
            }
            if channels == 4 {
                result[pi + 3] = pixels[pi + 3];
            }
        }
    }

    Ok(result)
}
}

