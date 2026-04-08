//! Edge detection and threshold filters — Sobel, Scharr, Laplacian, Canny,
//! and various thresholding methods (binary, Otsu, adaptive, triangle).
//!
//! Edge detectors are spatial (need 3x3 neighborhood). Threshold filters are
//! point ops on luminance. All have GPU shaders.

mod sobel;
mod scharr;
mod laplacian;
mod canny;
mod threshold_binary;
mod otsu_threshold;
mod triangle_threshold;
mod adaptive_threshold;

pub use sobel::Sobel;
pub use scharr::Scharr;
pub use laplacian::Laplacian;
pub use canny::Canny;
pub use threshold_binary::ThresholdBinary;
pub use otsu_threshold::OtsuThreshold;
pub use triangle_threshold::TriangleThreshold;
pub use adaptive_threshold::AdaptiveThreshold;

// Re-export kernel constants used by canny
pub(crate) use sobel::{SOBEL_X, SOBEL_Y};

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::Filter;

    fn gradient_image(w: u32, h: u32) -> Vec<f32> {
        let mut px = Vec::with_capacity((w * h * 4) as usize);
        for _y in 0..h {
            for x in 0..w {
                let v = x as f32 / (w - 1) as f32;
                px.extend_from_slice(&[v, v, v, 1.0]);
            }
        }
        px
    }

    #[test]
    fn sobel_detects_vertical_edge() {
        let img = gradient_image(16, 16);
        let out = Sobel { scale: 1.0 }.compute(&img, 16, 16).unwrap();
        let mid = 8 * 4;
        assert!(out[mid] > 0.0, "sobel should detect gradient edge");
    }

    #[test]
    fn threshold_binary_splits() {
        let img = gradient_image(16, 16);
        let out = ThresholdBinary { threshold: 0.5 }.compute(&img, 16, 16).unwrap();
        assert_eq!(out[0], 0.0);
        let last = (15 * 4) as usize;
        assert_eq!(out[last], 1.0);
    }

    #[test]
    fn otsu_auto_threshold() {
        let img = gradient_image(32, 32);
        let out = OtsuThreshold.compute(&img, 32, 32).unwrap();
        for px in out.chunks_exact(4) {
            assert!(px[0] == 0.0 || px[0] == 1.0, "otsu should produce binary: {}", px[0]);
        }
    }

    #[test]
    fn adaptive_threshold_runs() {
        let img = gradient_image(16, 16);
        let f = AdaptiveThreshold { radius: 3, offset: 0.02 };
        let out = f.compute(&img, 16, 16).unwrap();
        assert_eq!(out.len(), img.len());
    }

    #[test]
    fn canny_produces_binary_edges() {
        let img = gradient_image(16, 16);
        let out = Canny { low: 0.05, high: 0.2 }.compute(&img, 16, 16).unwrap();
        for px in out.chunks_exact(4) {
            assert!(
                px[0] == 0.0 || (px[0] - 0.5).abs() < 1e-6 || (px[0] - 1.0).abs() < 1e-6,
                "canny output should be 0/0.5/1, got {}", px[0]
            );
        }
        let edge_count = out.chunks_exact(4).filter(|px| px[0] > 0.0).count();
        assert!(edge_count > 0, "canny should detect some edges in gradient");
    }

    #[test]
    fn filters_registered() {
        let ops = crate::registered_operations();
        let names: Vec<&str> = ops.iter().map(|o| o.name).collect();
        for f in &["sobel", "scharr", "laplacian", "canny", "threshold_binary", "otsu_threshold", "triangle_threshold", "adaptive_threshold"] {
            assert!(names.contains(f), "{f} not registered");
        }
    }

    #[test]
    fn otsu_has_gpu_3_pass() {
        let f = OtsuThreshold;
        let shaders = f.gpu_shader_passes(32, 32);
        assert!(shaders.is_some(), "otsu should have GPU shaders");
        assert_eq!(shaders.unwrap().len(), 3, "otsu should be 3-pass (hist reduce + hist merge + apply)");
    }

    #[test]
    fn triangle_has_gpu_3_pass() {
        let f = TriangleThreshold;
        let shaders = f.gpu_shader_passes(32, 32);
        assert!(shaders.is_some(), "triangle should have GPU shaders");
        assert_eq!(shaders.unwrap().len(), 3, "triangle should be 3-pass");
    }
}
