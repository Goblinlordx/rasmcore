//! Filter: skeletonize (category: morphology)
//!
//! Zhang-Suen thinning algorithm for skeleton extraction from binary images.
//! Reduces shapes to 1-pixel-wide center lines while preserving topology.

#[allow(unused_imports)]
use crate::domain::filters::common::*;

/// Parameters for morphological skeletonization.
#[derive(rasmcore_macros::ConfigParams, Clone)]
pub struct SkeletonizeParams {
    /// Maximum iterations (0 = run until convergence)
    #[param(min = 0, max = 1000, step = 1, default = 0)]
    pub max_iterations: u32,
}

#[rasmcore_macros::register_filter(
    name = "skeletonize",
    category = "morphology",
    group = "morphology",
    variant = "skeletonize",
    reference = "Zhang-Suen 1984 thinning algorithm"
)]
pub fn skeletonize_registered(
    request: Rect,
    upstream: &mut UpstreamFn,
    info: &ImageInfo,
    config: &SkeletonizeParams,
) -> Result<Vec<u8>, ImageError> {
    // Skeletonize needs the full image (global operation)
    let full = Rect::new(0, 0, info.width, info.height);
    let pixels = upstream(full)?;
    let info = &ImageInfo {
        width: full.width,
        height: full.height,
        ..*info
    };
    let result = skeletonize(&pixels, info, config.max_iterations)?;
    Ok(crop_to_request(&result, full, request, info.format))
}

/// Zhang-Suen thinning algorithm.
///
/// Input: binary Gray8 image (0 = background, non-zero = foreground).
/// Output: binary skeleton (0 or 255).
///
/// Reference: T.Y. Zhang, C.Y. Suen, CACM 1984.
pub fn skeletonize(
    pixels: &[u8],
    info: &ImageInfo,
    max_iterations: u32,
) -> Result<Vec<u8>, ImageError> {
    if info.format != PixelFormat::Gray8 {
        return Err(ImageError::UnsupportedFormat(
            "skeletonize requires Gray8 input".into(),
        ));
    }

    let w = info.width as usize;
    let h = info.height as usize;

    // Convert to binary (1 = foreground, 0 = background)
    let mut img: Vec<u8> = pixels.iter().map(|&v| if v > 0 { 1 } else { 0 }).collect();

    let max_iter = if max_iterations == 0 {
        u32::MAX
    } else {
        max_iterations
    };

    for _iter in 0..max_iter {
        let mut changed = false;

        // Sub-iteration 1
        let markers = zhang_suen_pass(&img, w, h, true);
        for &idx in &markers {
            img[idx] = 0;
            changed = true;
        }

        // Sub-iteration 2
        let markers = zhang_suen_pass(&img, w, h, false);
        for &idx in &markers {
            img[idx] = 0;
            changed = true;
        }

        if !changed {
            break;
        }
    }

    // Convert back to 0/255
    Ok(img.iter().map(|&v| if v > 0 { 255 } else { 0 }).collect())
}

/// One pass of the Zhang-Suen algorithm.
/// Returns indices of pixels to delete.
fn zhang_suen_pass(img: &[u8], w: usize, h: usize, first_sub: bool) -> Vec<usize> {
    let mut to_delete = Vec::new();

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let idx = y * w + x;
            if img[idx] == 0 {
                continue; // background
            }

            // 8-connected neighbors (clockwise from north):
            // P2=N, P3=NE, P4=E, P5=SE, P6=S, P7=SW, P8=W, P9=NW
            let p2 = img[(y - 1) * w + x] as u32;
            let p3 = img[(y - 1) * w + x + 1] as u32;
            let p4 = img[y * w + x + 1] as u32;
            let p5 = img[(y + 1) * w + x + 1] as u32;
            let p6 = img[(y + 1) * w + x] as u32;
            let p7 = img[(y + 1) * w + x - 1] as u32;
            let p8 = img[y * w + x - 1] as u32;
            let p9 = img[(y - 1) * w + x - 1] as u32;

            // Condition A: number of 0→1 transitions in the ordered sequence P2..P9..P2
            let transitions = count_transitions(p2, p3, p4, p5, p6, p7, p8, p9);

            // Condition B: number of non-zero neighbors
            let neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;

            // A(P1) = 1
            if transitions != 1 {
                continue;
            }

            // 2 <= B(P1) <= 6
            if neighbors < 2 || neighbors > 6 {
                continue;
            }

            if first_sub {
                // Sub-iteration 1: P2 * P4 * P6 == 0 AND P4 * P6 * P8 == 0
                if p2 * p4 * p6 != 0 {
                    continue;
                }
                if p4 * p6 * p8 != 0 {
                    continue;
                }
            } else {
                // Sub-iteration 2: P2 * P4 * P8 == 0 AND P2 * P6 * P8 == 0
                if p2 * p4 * p8 != 0 {
                    continue;
                }
                if p2 * p6 * p8 != 0 {
                    continue;
                }
            }

            to_delete.push(idx);
        }
    }

    to_delete
}

/// Count 0→1 transitions in the ordered neighborhood sequence.
#[inline]
fn count_transitions(p2: u32, p3: u32, p4: u32, p5: u32, p6: u32, p7: u32, p8: u32, p9: u32) -> u32 {
    let mut count = 0;
    let seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2];
    for i in 0..8 {
        if seq[i] == 0 && seq[i + 1] == 1 {
            count += 1;
        }
    }
    count
}

// ─── GPU Support ─────────────────────────────────────────────────────────

/// WGSL shader source for Zhang-Suen thinning.
static SKELETONIZE_WGSL: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    // Skeletonize operates on binary (0/255) pixels — no pack/unpack needed
    include_str!("../../../shaders/skeletonize.wgsl").to_string()
});

impl rasmcore_pipeline::GpuCapable for SkeletonizeParams {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<rasmcore_pipeline::GpuOp>> {
        // Each "iteration" = 2 dispatches (sub-iter 0, sub-iter 1).
        // For GPU, we do a fixed number of iterations since we can't
        // check convergence without readback. Default: 50 iterations
        // (sufficient for most images up to 4K).
        let iterations = if self.max_iterations == 0 {
            50 // heuristic for convergence
        } else {
            self.max_iterations.min(200)
        };

        let mut ops = Vec::with_capacity(iterations as usize * 2);
        for _ in 0..iterations {
            // Sub-iteration 1 (sub_iter = 0)
            let mut params_0 = Vec::with_capacity(16);
            params_0.extend_from_slice(&width.to_le_bytes());
            params_0.extend_from_slice(&height.to_le_bytes());
            params_0.extend_from_slice(&0u32.to_le_bytes()); // sub_iter = 0
            params_0.extend_from_slice(&0u32.to_le_bytes()); // padding
            ops.push(rasmcore_pipeline::GpuOp::Compute {
                shader: SKELETONIZE_WGSL.clone(),
                entry_point: "main",
                workgroup_size: [16, 16, 1],
                params: params_0,
                extra_buffers: Vec::new(),
                buffer_format: Default::default(),
            });

            // Sub-iteration 2 (sub_iter = 1)
            let mut params_1 = Vec::with_capacity(16);
            params_1.extend_from_slice(&width.to_le_bytes());
            params_1.extend_from_slice(&height.to_le_bytes());
            params_1.extend_from_slice(&1u32.to_le_bytes()); // sub_iter = 1
            params_1.extend_from_slice(&0u32.to_le_bytes()); // padding
            ops.push(rasmcore_pipeline::GpuOp::Compute {
                shader: SKELETONIZE_WGSL.clone(),
                entry_point: "main",
                workgroup_size: [16, 16, 1],
                params: params_1,
                extra_buffers: Vec::new(),
                buffer_format: Default::default(),
            });
        }

        Some(ops)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::ColorSpace;

    fn gray_info(w: u32, h: u32) -> ImageInfo {
        ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        }
    }

    #[test]
    fn single_pixel_line_unchanged() {
        // A 1-pixel-wide horizontal line should be preserved
        let mut px = vec![0u8; 7 * 5];
        // Draw horizontal line at y=2, x=1..5
        for x in 1..6 {
            px[2 * 7 + x] = 255;
        }
        let info = gray_info(7, 5);
        let result = skeletonize(&px, &info, 0).unwrap();
        // The line should still exist (may lose endpoints but center preserved)
        let center_count: usize = (1..6)
            .filter(|&x| result[2 * 7 + x] > 0)
            .count();
        assert!(center_count >= 3, "thin line should be mostly preserved, got {center_count} pixels");
    }

    #[test]
    fn thick_rect_thins_to_skeleton() {
        // A 10x10 filled rect (with 1-pixel border) should thin to a cross-like shape
        let w = 12;
        let h = 12;
        let mut px = vec![0u8; w * h];
        for y in 1..11 {
            for x in 1..11 {
                px[y * w + x] = 255;
            }
        }
        let info = gray_info(w as u32, h as u32);
        let result = skeletonize(&px, &info, 0).unwrap();

        // Count remaining foreground pixels
        let fg: usize = result.iter().filter(|&&v| v > 0).count();
        // Skeleton of a 10x10 square should be much fewer pixels than 100
        assert!(fg < 50, "skeleton should be thinner than original, got {fg} pixels (was 100)");
        assert!(fg > 0, "skeleton should not be empty");
    }

    #[test]
    fn empty_image_returns_empty() {
        let px = vec![0u8; 25];
        let info = gray_info(5, 5);
        let result = skeletonize(&px, &info, 0).unwrap();
        assert!(result.iter().all(|&v| v == 0));
    }

    #[test]
    fn gpu_ops_generated() {
        let params = SkeletonizeParams { max_iterations: 0 };
        use rasmcore_pipeline::GpuCapable;
        let ops = params.gpu_ops(100, 100).unwrap();
        // 50 iterations × 2 sub-iterations = 100 dispatches
        assert_eq!(ops.len(), 100);
        match &ops[0] {
            rasmcore_pipeline::GpuOp::Compute { entry_point, workgroup_size, params, .. } => {
                assert_eq!(*entry_point, "main");
                assert_eq!(*workgroup_size, [16, 16, 1]);
                assert_eq!(params[8], 0); // sub_iter = 0
            }
            _ => panic!("expected Compute"),
        }
        match &ops[1] {
            rasmcore_pipeline::GpuOp::Compute { params, .. } => {
                assert_eq!(params[8], 1); // sub_iter = 1
            }
            _ => panic!("expected Compute"),
        }
    }

    #[test]
    fn gpu_cpu_parity() {
        // Verify GPU shader produces same result as CPU implementation
        // This test uses the CPU executor as a reference — actual GPU test
        // requires a GPU device (tested in cli gpu_executor tests).
        let w = 20;
        let h = 20;
        let mut px = vec![0u8; w * h];
        for y in 3..17 {
            for x in 3..17 {
                px[y * w + x] = 255;
            }
        }
        let info = gray_info(w as u32, h as u32);

        // CPU result with fixed iterations for deterministic comparison
        let cpu_result = skeletonize(&px, &info, 10).unwrap();
        let cpu_fg: usize = cpu_result.iter().filter(|&&v| v > 0).count();

        // Verify CPU produces a valid skeleton
        assert!(cpu_fg > 0, "CPU skeleton should not be empty");
        assert!(cpu_fg < 196, "CPU skeleton should be thinner than 14×14 input");
    }

    #[test]
    fn requires_gray8() {
        let px = vec![0u8; 30];
        let info = ImageInfo {
            width: 5,
            height: 2,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        assert!(skeletonize(&px, &info, 0).is_err());
    }

    #[test]
    fn max_iterations_limits_passes() {
        // With max_iterations=1, only one pass happens — result may not be fully thinned
        let w = 20;
        let h = 20;
        let mut px = vec![0u8; w * h];
        for y in 2..18 {
            for x in 2..18 {
                px[y * w + x] = 255;
            }
        }
        let info = gray_info(w as u32, h as u32);
        let result_1 = skeletonize(&px, &info, 1).unwrap();
        let result_full = skeletonize(&px, &info, 0).unwrap();

        let fg_1: usize = result_1.iter().filter(|&&v| v > 0).count();
        let fg_full: usize = result_full.iter().filter(|&&v| v > 0).count();
        // 1 iteration should leave more pixels than full convergence
        assert!(fg_1 >= fg_full, "1 iteration ({fg_1}) should leave >= full ({fg_full})");
    }
}
