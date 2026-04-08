//! Analysis and transform filters — feature detection, seam carving,
//! perspective correction, and interpolation.

mod harris_corners;
mod perspective_warp;
mod perspective_correct;
mod sparse_color;
mod seam_carve_width;
mod seam_carve_height;
mod smart_crop;
mod smart_crop_analysis;
mod hough_lines;
mod connected_components;
mod template_match;

pub use harris_corners::HarrisCorners;
pub use perspective_warp::PerspectiveWarp;
pub use perspective_correct::PerspectiveCorrect;
pub use sparse_color::SparseColor;
pub use seam_carve_width::SeamCarveWidth;
pub use seam_carve_height::SeamCarveHeight;
pub use smart_crop::SmartCrop;
pub use smart_crop_analysis::SmartCropAnalysis;
pub use hough_lines::HoughLines;
pub use connected_components::ConnectedComponents;
pub use template_match::TemplateMatch;

const SAMPLE_BILINEAR_WGSL: &str = r#"
fn sample_bilinear_f32(fx: f32, fy: f32) -> vec4<f32> {
  let ix = i32(floor(fx)); let iy = i32(floor(fy));
  let dx = fx - f32(ix); let dy = fy - f32(iy);
  let x0 = clamp(ix, 0, i32(params.width) - 1);
  let x1 = clamp(ix + 1, 0, i32(params.width) - 1);
  let y0 = clamp(iy, 0, i32(params.height) - 1);
  let y1 = clamp(iy + 1, 0, i32(params.height) - 1);
  let p00 = input[u32(x0) + u32(y0) * params.width];
  let p10 = input[u32(x1) + u32(y0) * params.width];
  let p01 = input[u32(x0) + u32(y1) * params.width];
  let p11 = input[u32(x1) + u32(y1) * params.width];
  return mix(mix(p00, p10, vec4<f32>(dx)), mix(p01, p11, vec4<f32>(dx)), vec4<f32>(dy));
}
"#;

pub use smart_crop_analysis::smart_crop_find_rect;


// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::Filter;

    #[test]
    fn all_analysis_transform_filters_registered() {
        let factories = crate::registered_filter_factories();
        for name in &["harris_corners", "perspective_warp", "perspective_correct",
                       "sparse_color", "seam_carve_width", "seam_carve_height",
                       "smart_crop", "hough_lines", "connected_components", "template_match"] {
            assert!(factories.contains(name), "{name} not registered");
        }
    }

    #[test]
    fn perspective_identity() {
        let input = vec![0.5, 0.3, 0.1, 1.0, 0.8, 0.6, 0.4, 1.0];
        let f = PerspectiveWarp {
            h11: 1.0, h12: 0.0, h13: 0.0,
            h21: 0.0, h22: 1.0, h23: 0.0,
            h31: 0.0, h32: 0.0,
        };
        let out = f.compute(&input, 2, 1).unwrap();
        for (a, b) in input.iter().zip(out.iter()) {
            assert!((a - b).abs() < 0.02, "expected {a}, got {b}");
        }
    }
}


#[cfg(test)]
mod smart_crop_analysis_tests {
    use crate::staged::{AnalysisNode, AnalysisResult};
    use super::{SmartCropAnalysis, smart_crop_find_rect};

    fn gradient_image(w: u32, h: u32) -> Vec<f32> {
        let n = (w * h) as usize;
        let mut px = Vec::with_capacity(n * 4);
        for i in 0..n {
            let v = i as f32 / n as f32;
            px.extend_from_slice(&[v, v, v, 1.0]);
        }
        px
    }

    #[test]
    fn smart_crop_analysis_produces_rect() {
        let img = gradient_image(100, 100);
        let info = crate::node::NodeInfo {
            width: 100, height: 100,
            color_space: crate::color_space::ColorSpace::Linear,
        };
        let node = SmartCropAnalysis::new(0, info, 0.5);
        let result = node.analyze(&img, 100, 100).unwrap();
        match result {
            AnalysisResult::Rect(r) => {
                assert!(r.width <= 100 && r.height <= 100);
                assert!(r.width >= 1 && r.height >= 1);
            }
            _ => panic!("expected Rect result"),
        }
    }

    #[test]
    fn smart_crop_find_rect_full_ratio() {
        let img = gradient_image(50, 50);
        let rect = smart_crop_find_rect(&img, 50, 50, 1.0);
        assert_eq!(rect.width, 50);
        assert_eq!(rect.height, 50);
    }

    #[test]
    fn smart_crop_find_rect_valid_bounds() {
        let img = gradient_image(100, 80);
        let rect = smart_crop_find_rect(&img, 100, 80, 0.5);
        assert!(rect.x + rect.width <= 100);
        assert!(rect.y + rect.height <= 80);
    }
}
