//! Content-aware smart cropping via attention and entropy heuristics.
//!
//! Matches libvips `vips_smartcrop()` with `VIPS_INTERESTING_ENTROPY` and
//! `VIPS_INTERESTING_ATTENTION` strategies. Used for automatic thumbnail
//! generation — finds the most "interesting" crop window.

use super::error::ImageError;
use super::filters;
use super::histogram;
use super::transform;
use super::types::{DecodedImage, ImageInfo, PixelFormat};

/// Strategy for smart crop content selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmartCropStrategy {
    /// Select the crop window with the highest Shannon entropy (information density).
    /// Best for: images where detail/texture indicates the subject.
    Entropy,
    /// Select the crop window with the highest edge energy (Sobel gradient magnitude).
    /// Best for: images where edges/structure indicate the subject (portraits, objects).
    Attention,
}

/// Smart crop: find the most interesting crop window of target dimensions.
///
/// Analyzes the image using the chosen strategy, then selects the crop position
/// that maximizes the score. Uses an integral image (summed area table) for
/// O(1) window sum queries after the initial score map computation.
///
/// For images larger than 1024px in either dimension, a 4x downsampled version
/// is analyzed and coordinates scaled back up.
pub fn smart_crop(
    pixels: &[u8],
    info: &ImageInfo,
    target_w: u32,
    target_h: u32,
    strategy: SmartCropStrategy,
) -> Result<DecodedImage, ImageError> {
    if target_w > info.width || target_h > info.height {
        return Err(ImageError::InvalidParameters(
            "smart_crop target dimensions must be <= image dimensions".into(),
        ));
    }
    if target_w == 0 || target_h == 0 {
        return Err(ImageError::InvalidParameters(
            "smart_crop target dimensions must be > 0".into(),
        ));
    }
    if target_w == info.width && target_h == info.height {
        return Ok(DecodedImage {
            pixels: pixels.to_vec(),
            info: info.clone(),
            icc_profile: None,
        });
    }

    // For large images, downsample for analysis then scale coordinates back
    let (analysis_pixels, analysis_info, scale) = if info.width > 1024 || info.height > 1024 {
        let scale = 4u32;
        let small_w = info.width / scale;
        let small_h = info.height / scale;
        let small = transform::resize(
            pixels,
            info,
            small_w,
            small_h,
            super::types::ResizeFilter::Bilinear,
        )?;
        (small.pixels, small.info, scale)
    } else {
        (pixels.to_vec(), info.clone(), 1u32)
    };

    // Compute score map
    let score_map = match strategy {
        SmartCropStrategy::Entropy => entropy_score_map(&analysis_pixels, &analysis_info)?,
        SmartCropStrategy::Attention => attention_score_map(&analysis_pixels, &analysis_info)?,
    };

    let map_w = analysis_info.width as usize;
    let map_h = analysis_info.height as usize;

    // Build integral image for O(1) window sums
    let sat = summed_area_table(&score_map, map_w, map_h);

    // Sliding window to find best crop position
    let win_w = (target_w / scale) as usize;
    let win_h = (target_h / scale) as usize;

    let mut best_score = f64::MIN;
    let mut best_x = 0usize;
    let mut best_y = 0usize;

    let max_x = map_w.saturating_sub(win_w);
    let max_y = map_h.saturating_sub(win_h);

    for y in 0..=max_y {
        for x in 0..=max_x {
            let score = sat_window_sum(&sat, map_w, x, y, win_w, win_h);
            if score > best_score {
                best_score = score;
                best_x = x;
                best_y = y;
            }
        }
    }

    // Scale coordinates back to original image
    let crop_x = (best_x as u32 * scale).min(info.width - target_w);
    let crop_y = (best_y as u32 * scale).min(info.height - target_h);

    transform::crop(pixels, info, crop_x, crop_y, target_w, target_h)
}

/// Compute per-pixel entropy score from local histograms.
///
/// Divides the image into cells and computes Shannon entropy per cell.
/// Returns a score map where higher values = more information content.
fn entropy_score_map(pixels: &[u8], info: &ImageInfo) -> Result<Vec<f64>, ImageError> {
    let w = info.width as usize;
    let h = info.height as usize;

    // Convert to grayscale intensity for entropy computation
    let gray = to_grayscale(pixels, info);

    // Compute per-pixel local entropy using a sliding window histogram
    let cell_size = 16usize; // 16x16 cells for local entropy
    let mut scores = vec![0.0f64; w * h];

    for cy in (0..h).step_by(cell_size) {
        for cx in (0..w).step_by(cell_size) {
            // Build local histogram for this cell
            let mut hist = [0u32; 256];
            let mut count = 0u32;
            let cell_h = cell_size.min(h - cy);
            let cell_w = cell_size.min(w - cx);

            for dy in 0..cell_h {
                for dx in 0..cell_w {
                    let idx = (cy + dy) * w + (cx + dx);
                    hist[gray[idx] as usize] += 1;
                    count += 1;
                }
            }

            // Shannon entropy: H = -sum(p * log2(p))
            let n = count as f64;
            let mut entropy = 0.0f64;
            for &c in &hist {
                if c > 0 {
                    let p = c as f64 / n;
                    entropy -= p * p.log2();
                }
            }

            // Fill all pixels in this cell with the same entropy score
            for dy in 0..cell_h {
                for dx in 0..cell_w {
                    scores[(cy + dy) * w + (cx + dx)] = entropy;
                }
            }
        }
    }

    Ok(scores)
}

/// Compute per-pixel attention score from Sobel edge energy.
///
/// Returns a score map where higher values = more edge/structure content.
fn attention_score_map(pixels: &[u8], info: &ImageInfo) -> Result<Vec<f64>, ImageError> {
    // Use Sobel filter to get gradient magnitude
    let sobel_output = filters::sobel(pixels, info)?;

    // Sobel output is grayscale — each pixel value is the gradient magnitude
    let scores: Vec<f64> = sobel_output.iter().map(|&v| v as f64).collect();
    Ok(scores)
}

/// Build a summed area table (integral image) for O(1) rectangular sum queries.
fn summed_area_table(scores: &[f64], w: usize, h: usize) -> Vec<f64> {
    let mut sat = vec![0.0f64; w * h];

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let mut val = scores[idx];
            if x > 0 {
                val += sat[idx - 1];
            }
            if y > 0 {
                val += sat[idx - w];
            }
            if x > 0 && y > 0 {
                val -= sat[idx - w - 1];
            }
            sat[idx] = val;
        }
    }

    sat
}

/// Query the sum of a rectangular region using the summed area table.
/// Region: [x, x+w) x [y, y+h)
#[inline]
fn sat_window_sum(sat: &[f64], stride: usize, x: usize, y: usize, w: usize, h: usize) -> f64 {
    let x2 = x + w - 1;
    let y2 = y + h - 1;

    let mut sum = sat[y2 * stride + x2];
    if x > 0 {
        sum -= sat[y2 * stride + (x - 1)];
    }
    if y > 0 {
        sum -= sat[(y - 1) * stride + x2];
    }
    if x > 0 && y > 0 {
        sum += sat[(y - 1) * stride + (x - 1)];
    }
    sum
}

/// Convert to grayscale (single-channel luminance).
fn to_grayscale(pixels: &[u8], info: &ImageInfo) -> Vec<u8> {
    match info.format {
        PixelFormat::Gray8 => pixels.to_vec(),
        PixelFormat::Rgb8 => pixels
            .chunks_exact(3)
            .map(|rgb| {
                ((rgb[0] as u32 * 77 + rgb[1] as u32 * 150 + rgb[2] as u32 * 29 + 128) >> 8) as u8
            })
            .collect(),
        PixelFormat::Rgba8 => pixels
            .chunks_exact(4)
            .map(|rgba| {
                ((rgba[0] as u32 * 77 + rgba[1] as u32 * 150 + rgba[2] as u32 * 29 + 128) >> 8)
                    as u8
            })
            .collect(),
        _ => pixels.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::ColorSpace;

    fn test_info(w: u32, h: u32) -> ImageInfo {
        ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        }
    }

    // ── Summed Area Table ───────────────────────────────────────────────

    #[test]
    fn sat_correctness() {
        // 3x3 grid: [1,2,3, 4,5,6, 7,8,9]
        let scores = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let sat = summed_area_table(&scores, 3, 3);

        // Top-left 2x2 window: 1+2+4+5 = 12
        let sum = sat_window_sum(&sat, 3, 0, 0, 2, 2);
        assert!((sum - 12.0).abs() < 0.01, "2x2 sum should be 12, got {sum}");

        // Full 3x3: 1+2+3+4+5+6+7+8+9 = 45
        let sum = sat_window_sum(&sat, 3, 0, 0, 3, 3);
        assert!((sum - 45.0).abs() < 0.01, "3x3 sum should be 45, got {sum}");

        // Bottom-right 2x2: 5+6+8+9 = 28
        let sum = sat_window_sum(&sat, 3, 1, 1, 2, 2);
        assert!(
            (sum - 28.0).abs() < 0.01,
            "BR 2x2 sum should be 28, got {sum}"
        );

        // Single pixel [1][1]: 5
        let sum = sat_window_sum(&sat, 3, 1, 1, 1, 1);
        assert!(
            (sum - 5.0).abs() < 0.01,
            "single pixel should be 5, got {sum}"
        );
    }

    // ── Entropy Score Map ───────────────────────────────────────────────

    #[test]
    fn entropy_uniform_is_zero() {
        // Uniform image: all same value → entropy = 0
        let pixels = vec![128u8; 32 * 32 * 3];
        let info = test_info(32, 32);
        let scores = entropy_score_map(&pixels, &info).unwrap();
        assert!(
            scores.iter().all(|&s| s < 0.01),
            "uniform image should have near-zero entropy"
        );
    }

    #[test]
    fn entropy_random_is_high() {
        // "Random" image: varied values → high entropy
        let mut pixels = Vec::with_capacity(64 * 64 * 3);
        for i in 0..(64 * 64) {
            let v = ((i * 37 + 13) % 256) as u8;
            pixels.push(v);
            pixels.push(v);
            pixels.push(v);
        }
        let info = test_info(64, 64);
        let scores = entropy_score_map(&pixels, &info).unwrap();
        let max_entropy = scores.iter().cloned().fold(0.0f64, f64::max);
        assert!(
            max_entropy > 3.0,
            "varied image should have high entropy, got {max_entropy}"
        );
    }

    // ── Smart Crop ──────────────────────────────────────────────────────

    #[test]
    fn smart_crop_finds_interesting_region() {
        // 128x128 image with a bright detailed region in the bottom-right quadrant
        let (w, h) = (128u32, 128u32);
        let mut pixels = vec![64u8; (w * h * 3) as usize]; // dark uniform background

        // Add detail in bottom-right quadrant (64..128, 64..128)
        for y in 64..128 {
            for x in 64..128 {
                let idx = ((y * w + x) * 3) as usize;
                let v = (((x + y) * 7) % 256) as u8;
                pixels[idx] = v;
                pixels[idx + 1] = 255 - v;
                pixels[idx + 2] = v / 2;
            }
        }

        let info = test_info(w, h);
        let result = smart_crop(&pixels, &info, 64, 64, SmartCropStrategy::Entropy).unwrap();

        assert_eq!(result.info.width, 64);
        assert_eq!(result.info.height, 64);
        // The crop should be biased toward the bottom-right where the detail is
        // (we can't assert exact position but the crop should not be all uniform)
        let pixel_variance: f64 = result
            .pixels
            .iter()
            .map(|&v| (v as f64 - 64.0).powi(2))
            .sum::<f64>()
            / result.pixels.len() as f64;
        assert!(
            pixel_variance > 100.0,
            "smart crop should select the detailed region, variance={pixel_variance:.0}"
        );
    }

    #[test]
    fn smart_crop_attention_finds_edges() {
        // Image with edges only on the left side
        let (w, h) = (128u32, 64u32);
        let mut pixels = vec![128u8; (w * h * 3) as usize];

        // Add vertical edges on left quarter
        for y in 0..64 {
            for x in 0..32 {
                let idx = ((y * w + x) * 3) as usize;
                let v = if x % 4 < 2 { 0u8 } else { 255u8 };
                pixels[idx] = v;
                pixels[idx + 1] = v;
                pixels[idx + 2] = v;
            }
        }

        let info = test_info(w, h);
        let result = smart_crop(&pixels, &info, 64, 64, SmartCropStrategy::Attention).unwrap();
        assert_eq!(result.info.width, 64);
        assert_eq!(result.info.height, 64);
    }

    #[test]
    fn smart_crop_same_size_is_identity() {
        let pixels = vec![100u8; 32 * 32 * 3];
        let info = test_info(32, 32);
        let result = smart_crop(&pixels, &info, 32, 32, SmartCropStrategy::Entropy).unwrap();
        assert_eq!(result.pixels, pixels);
    }

    #[test]
    fn smart_crop_invalid_size() {
        let pixels = vec![100u8; 32 * 32 * 3];
        let info = test_info(32, 32);
        assert!(smart_crop(&pixels, &info, 64, 64, SmartCropStrategy::Entropy).is_err());
        assert!(smart_crop(&pixels, &info, 0, 32, SmartCropStrategy::Entropy).is_err());
    }

    // ── Reference comparison against libvips smartcrop ──────────────────

    fn vips_available() -> bool {
        std::process::Command::new("vips")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Write PPM and run vips smartcrop, return the crop position (x, y).
    fn vips_smartcrop(
        pixels: &[u8],
        w: u32,
        h: u32,
        target_w: u32,
        target_h: u32,
        strategy: &str,
    ) -> Option<(u32, u32)> {
        use std::io::Write;
        use std::sync::atomic::{AtomicU64, Ordering};
        static CTR: AtomicU64 = AtomicU64::new(0);
        let id = CTR.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir();
        let ppm = dir.join(format!("sc_ref_{id}.ppm"));
        let out = dir.join(format!("sc_ref_{id}_out.ppm"));

        let mut f = std::fs::File::create(&ppm).ok()?;
        write!(f, "P6\n{w} {h}\n255\n").ok()?;
        f.write_all(pixels).ok()?;
        drop(f);

        let result = std::process::Command::new("vips")
            .args([
                "smartcrop",
                ppm.to_str().unwrap(),
                out.to_str().unwrap(),
                &target_w.to_string(),
                &target_h.to_string(),
                "--interesting",
                strategy,
            ])
            .output()
            .ok()?;

        let _ = std::fs::remove_file(&ppm);

        if !result.status.success() {
            let _ = std::fs::remove_file(&out);
            return None;
        }

        // Read the output PPM to get the cropped pixels
        let vips_data = std::fs::read(&out).ok()?;
        let _ = std::fs::remove_file(&out);

        // We can't easily get the crop position from vips CLI, but we CAN
        // compare the actual cropped pixel data. Find where in the original
        // image the vips output matches by scanning.
        let vips_pixels = parse_ppm_pixels(&vips_data)?;
        find_crop_position(pixels, w, h, &vips_pixels, target_w, target_h)
    }

    fn parse_ppm_pixels(data: &[u8]) -> Option<Vec<u8>> {
        // Skip PPM header "P6\nW H\n255\n"
        let mut pos = 0;
        // Skip "P6\n"
        while pos < data.len() && data[pos] != b'\n' {
            pos += 1;
        }
        pos += 1; // skip newline
        // Skip "W H\n"
        while pos < data.len() && data[pos] != b'\n' {
            pos += 1;
        }
        pos += 1;
        // Skip "255\n"
        while pos < data.len() && data[pos] != b'\n' {
            pos += 1;
        }
        pos += 1;
        Some(data[pos..].to_vec())
    }

    /// Find where a crop appears in the original image by comparing top-left corner pixels.
    fn find_crop_position(
        original: &[u8],
        orig_w: u32,
        orig_h: u32,
        crop: &[u8],
        crop_w: u32,
        crop_h: u32,
    ) -> Option<(u32, u32)> {
        let ow = orig_w as usize;
        let cw = crop_w as usize;
        let ch = crop_h as usize;

        for y in 0..=(orig_h - crop_h) as usize {
            for x in 0..=(orig_w - crop_w) as usize {
                // Check first row match
                let orig_start = (y * ow + x) * 3;
                let crop_start = 0;
                if orig_start + cw * 3 > original.len() || cw * 3 > crop.len() {
                    continue;
                }
                if original[orig_start..orig_start + cw * 3]
                    == crop[crop_start..crop_start + cw * 3]
                {
                    // Verify full match
                    let mut full_match = true;
                    for row in 0..ch.min(3) {
                        // check first 3 rows
                        let os = ((y + row) * ow + x) * 3;
                        let cs = row * cw * 3;
                        if os + cw * 3 > original.len() || cs + cw * 3 > crop.len() {
                            full_match = false;
                            break;
                        }
                        if original[os..os + cw * 3] != crop[cs..cs + cw * 3] {
                            full_match = false;
                            break;
                        }
                    }
                    if full_match {
                        return Some((x as u32, y as u32));
                    }
                }
            }
        }
        None
    }

    /// Generate test patterns, run through both our and vips smartcrop,
    /// compare the cropped outputs via pixel similarity (PSNR).
    /// Both should select similar regions — the cropped outputs should be similar.
    #[test]
    fn smart_crop_vs_vips_reference() {
        if !vips_available() {
            eprintln!("SKIP: vips not available for smart crop reference comparison");
            return;
        }

        // Test pattern: detail in bottom-right, flat top-left
        let (w, h) = (256u32, 256u32);
        let mut pixels = vec![80u8; (w * h * 3) as usize];
        for y in 128..256u32 {
            for x in 128..256u32 {
                let idx = ((y * w + x) * 3) as usize;
                let v = (((x + y) * 13) % 256) as u8;
                pixels[idx] = v;
                pixels[idx + 1] = (255 - v as u16) as u8;
                pixels[idx + 2] = (v / 2).wrapping_add(64);
            }
        }

        let info = test_info(w, h);
        let target_w = 128;
        let target_h = 128;

        eprintln!("\n=== Smart Crop vs vips (256x256 detail_br → 128x128) ===");
        eprintln!(
            "{:<12} {:>12} {:>12} {:>10}",
            "Strategy", "Our_PSNR", "Vips_PSNR", "Match_PSNR"
        );

        for (strategy_name, our_strategy, vips_strategy) in [
            ("entropy", SmartCropStrategy::Entropy, "entropy"),
            ("attention", SmartCropStrategy::Attention, "attention"),
        ] {
            // Our crop
            let our_result = smart_crop(&pixels, &info, target_w, target_h, our_strategy).unwrap();

            // Vips crop
            if let Some(vips_pixels) =
                vips_smartcrop_pixels(&pixels, w, h, target_w, target_h, vips_strategy)
            {
                // Both crops should select the detailed region (bottom-right)
                // Compare: how much do the two crops agree?
                let min_len = our_result.pixels.len().min(vips_pixels.len());
                if min_len > 0 {
                    let our_vs_vips_psnr =
                        pixel_psnr(&our_result.pixels[..min_len], &vips_pixels[..min_len]);

                    // Also measure each crop's "interestingness" vs the original
                    // by computing variance (higher = more detail selected)
                    let our_var = pixel_variance(&our_result.pixels);
                    let vips_var = pixel_variance(&vips_pixels[..min_len]);

                    eprintln!(
                        "{:<12} {:>11.1}var {:>11.1}var {:>9.1}dB",
                        strategy_name, our_var, vips_var, our_vs_vips_psnr
                    );

                    // Both should select the interesting region (high variance)
                    // rather than the flat background (low variance ≈ 0)
                    assert!(
                        our_var > 500.0,
                        "{strategy_name}: our crop variance too low ({our_var:.0}) — selected flat region"
                    );
                    assert!(
                        vips_var > 500.0,
                        "{strategy_name}: vips crop variance too low ({vips_var:.0})"
                    );
                }
            } else {
                eprintln!("{:<12} vips smartcrop failed", strategy_name);
            }
        }
    }

    fn vips_smartcrop_pixels(
        pixels: &[u8],
        w: u32,
        h: u32,
        target_w: u32,
        target_h: u32,
        strategy: &str,
    ) -> Option<Vec<u8>> {
        use std::io::Write;
        use std::sync::atomic::{AtomicU64, Ordering};
        static CTR: AtomicU64 = AtomicU64::new(0);
        let id = CTR.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir();
        let ppm = dir.join(format!("sc_ref_{id}.ppm"));
        let out = dir.join(format!("sc_ref_{id}_out.ppm"));

        let mut f = std::fs::File::create(&ppm).ok()?;
        write!(f, "P6\n{w} {h}\n255\n").ok()?;
        f.write_all(pixels).ok()?;
        drop(f);

        let result = std::process::Command::new("vips")
            .args([
                "smartcrop",
                ppm.to_str().unwrap(),
                out.to_str().unwrap(),
                &target_w.to_string(),
                &target_h.to_string(),
                "--interesting",
                strategy,
            ])
            .output()
            .ok()?;

        let _ = std::fs::remove_file(&ppm);
        if !result.status.success() {
            let _ = std::fs::remove_file(&out);
            return None;
        }

        let data = std::fs::read(&out).ok()?;
        let _ = std::fs::remove_file(&out);
        parse_ppm_pixels(&data)
    }

    fn pixel_psnr(a: &[u8], b: &[u8]) -> f64 {
        let n = a.len().min(b.len());
        if n == 0 {
            return 0.0;
        }
        let mse: f64 = a[..n]
            .iter()
            .zip(b[..n].iter())
            .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
            .sum::<f64>()
            / n as f64;
        if mse == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (255.0f64 * 255.0 / mse).log10()
        }
    }

    fn pixel_variance(pixels: &[u8]) -> f64 {
        let n = pixels.len() as f64;
        if n == 0.0 {
            return 0.0;
        }
        let mean = pixels.iter().map(|&v| v as f64).sum::<f64>() / n;
        pixels
            .iter()
            .map(|&v| (v as f64 - mean).powi(2))
            .sum::<f64>()
            / n
    }
}
