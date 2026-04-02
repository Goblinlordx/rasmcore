//! Filter: smart_sharpen (category: spatial)

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Smart sharpen — edge-preserving unsharp mask using bilateral filter.
///
/// Computes: output = original + amount * (original - bilateral_blurred)
///
/// Unlike standard sharpen (which uses Gaussian blur), smart sharpen
/// uses bilateral filtering for the blur pass. This preserves edges
/// while smoothing flat regions, producing sharpening without halos
/// along strong edges.
///
/// Reference: similar to Photoshop's Smart Sharpen (Remove: Lens Blur mode).

/// Parameters for smart sharpen (edge-preserving unsharp mask).
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct SmartSharpenParams {
    /// Sharpening amount (0-5, 1.0 = standard)
    #[param(min = 0.0, max = 5.0, step = 0.1, default = 1.0)]
    pub amount: f32,
    /// Bilateral filter radius for edge-preserving blur
    #[param(min = 1, max = 20, step = 1, default = 3)]
    pub radius: u32,
    /// Edge threshold — higher values preserve more edges (bilateral sigma_range)
    #[param(min = 1.0, max = 200.0, step = 1.0, default = 50.0)]
    pub threshold: f32,
}

impl InputRectProvider for SmartSharpenParams {
    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> Rect {
        let overlap = self.radius + 4;
        output.expand_uniform(overlap, bounds_w, bounds_h)
    }
}

#[rasmcore_macros::register_filter(
    name = "smart_sharpen",
    category = "spatial",
    group = "sharpen",
    variant = "smart",
    reference = "Photoshop Smart Sharpen (bilateral-based)"
)]
pub fn smart_sharpen(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &SmartSharpenParams,
) -> Result<Vec<u8>, ImageError> {
    let overlap = config.radius + 4; // bilateral + blur
    let expanded = request.expand_uniform(overlap, info.width, info.height);
    let pixels = upstream(expanded)?;
    let info = &ImageInfo {
        width: expanded.width,
        height: expanded.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let amount = config.amount;
    let radius = config.radius;
    let threshold = config.threshold;

    validate_format(info.format)?;

    if amount == 0.0 {
        return Ok(pixels.to_vec());
    }

    if is_16bit(info.format) {
        let result = process_via_8bit(pixels, info, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            smart_sharpen(r, &mut u, i8, config)
        })?;
        return Ok(crop_to_request(&result, expanded, request, info.format));
    }

    // Use bilateral filter for edge-preserving blur
    let bilateral_config = BilateralParams {
        diameter: radius * 2 + 1,
        sigma_color: threshold,
        sigma_space: radius as f32,
    };

    // Bilateral requires Gray8 or Rgb8 — convert if needed
    let (work_pixels, work_info) = if info.format == PixelFormat::Rgba8 {
        // Strip alpha, process RGB, restore alpha
        let rgb: Vec<u8> = pixels.chunks(4).flat_map(|c| &c[..3]).copied().collect();
        let rgb_info = ImageInfo {
            format: PixelFormat::Rgb8,
            ..*info
        };
        (rgb, rgb_info)
    } else {
        (pixels.to_vec(), info.clone())
    };

    let r = Rect::new(0, 0, work_info.width, work_info.height);
    let mut u = |_: Rect| Ok(work_pixels.clone());
    let blurred = bilateral(r, &mut u, &work_info, &bilateral_config)?;

    // Unsharp mask: output = original + amount * (original - blurred)
    let mut result = vec![0u8; work_pixels.len()];

    #[cfg(target_arch = "wasm32")]
    {
        use std::arch::wasm32::*;
        let amount_vec = f32x4_splat(amount);
        let zero = f32x4_splat(0.0);
        let max_val = f32x4_splat(255.0);
        let len = work_pixels.len();
        let chunks = len / 4;

        for i in 0..chunks {
            let base = i * 4;
            let orig = f32x4(
                work_pixels[base] as f32,
                work_pixels[base + 1] as f32,
                work_pixels[base + 2] as f32,
                work_pixels[base + 3] as f32,
            );
            let blur_v = f32x4(
                blurred[base] as f32,
                blurred[base + 1] as f32,
                blurred[base + 2] as f32,
                blurred[base + 3] as f32,
            );
            let diff = f32x4_sub(orig, blur_v);
            let scaled = f32x4_mul(diff, amount_vec);
            let sharp = f32x4_add(orig, scaled);
            let clamped = f32x4_max(zero, f32x4_min(max_val, sharp));

            result[base] = f32x4_extract_lane::<0>(clamped) as u8;
            result[base + 1] = f32x4_extract_lane::<1>(clamped) as u8;
            result[base + 2] = f32x4_extract_lane::<2>(clamped) as u8;
            result[base + 3] = f32x4_extract_lane::<3>(clamped) as u8;
        }
        for i in (chunks * 4)..len {
            let orig = work_pixels[i] as f32;
            let blur_val = blurred[i] as f32;
            result[i] = (orig + amount * (orig - blur_val)).clamp(0.0, 255.0) as u8;
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        for i in 0..work_pixels.len() {
            let orig = work_pixels[i] as f32;
            let blur_val = blurred[i] as f32;
            let sharpened = orig + amount * (orig - blur_val);
            result[i] = sharpened.clamp(0.0, 255.0) as u8;
        }
    }

    // Restore alpha if stripped
    let full_result = if info.format == PixelFormat::Rgba8 {
        let mut rgba = vec![0u8; pixels.len()];
        for i in 0..(pixels.len() / 4) {
            rgba[i * 4] = result[i * 3];
            rgba[i * 4 + 1] = result[i * 3 + 1];
            rgba[i * 4 + 2] = result[i * 3 + 2];
            rgba[i * 4 + 3] = pixels[i * 4 + 3]; // preserve original alpha
        }
        rgba
    } else {
        result
    };
    Ok(crop_to_request(&full_result, expanded, request, info.format))
}
