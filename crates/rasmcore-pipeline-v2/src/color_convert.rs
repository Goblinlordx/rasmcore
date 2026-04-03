//! Color conversion node and view transform — auto-inserted at color space boundaries.

use crate::color_math::convert_color_space;
use crate::color_space::ColorSpace;
use crate::node::{InputRectEstimate, Node, NodeCapabilities, NodeInfo, PipelineError, Upstream};
use crate::rect::Rect;

/// View transform applied at the output boundary (before encode).
///
/// Converts from the pipeline working space (Linear) to the display color space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewTransform {
    /// sRGB gamma encoding (default for SDR monitors).
    Srgb,
    /// No transform — pass through linear (for EXR/HDR output).
    Linear,
    /// Rec.709 (BT.1886 gamma 2.4).
    Rec709,
    // Future: AcesRrtSrgb, AcesRrtRec709, AcesRrtP3, etc.
}

impl Default for ViewTransform {
    fn default() -> Self {
        ViewTransform::Srgb
    }
}

/// Lightweight color space conversion node.
///
/// Auto-inserted by the graph walker when a downstream node expects a different
/// color space from its upstream. Does per-channel transfer function + optional
/// 3x3 matrix multiplication. Zero spatial dependency (point op).
pub struct ColorConvertNode {
    upstream: u32,
    info: NodeInfo,
    from: ColorSpace,
    to: ColorSpace,
}

impl ColorConvertNode {
    pub fn new(upstream: u32, upstream_info: NodeInfo, from: ColorSpace, to: ColorSpace) -> Self {
        Self {
            upstream,
            info: NodeInfo {
                color_space: to,
                ..upstream_info
            },
            from,
            to,
        }
    }
}

impl Node for ColorConvertNode {
    fn info(&self) -> NodeInfo {
        self.info.clone()
    }

    fn compute(
        &self,
        request: Rect,
        upstream: &mut dyn Upstream,
    ) -> Result<Vec<f32>, PipelineError> {
        let mut pixels = upstream.request(self.upstream, request)?;
        convert_color_space(&mut pixels, self.from, self.to);
        Ok(pixels)
    }

    fn upstream_ids(&self) -> Vec<u32> {
        vec![self.upstream]
    }

    fn capabilities(&self) -> NodeCapabilities {
        NodeCapabilities {
            analytic: false,
            affine: false,
            clut: false,
            gpu: false, // GPU shaders for color conversion added in future
        }
    }

    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> InputRectEstimate {
        // Point op — no spatial expansion
        InputRectEstimate::Exact(output.clamp(bounds_w, bounds_h))
    }

    fn expected_input_color_space(&self) -> ColorSpace {
        self.from
    }
}

/// View transform node — applied at the output boundary.
///
/// Converts from pipeline working space (Linear) to display-ready values.
/// For sRGB: applies linear_to_srgb per channel.
/// For Linear: pass-through (EXR/HDR output).
pub struct ViewTransformNode {
    upstream: u32,
    info: NodeInfo,
    transform: ViewTransform,
}

impl ViewTransformNode {
    pub fn new(upstream: u32, upstream_info: NodeInfo, transform: ViewTransform) -> Self {
        let out_cs = match transform {
            ViewTransform::Srgb => ColorSpace::Srgb,
            ViewTransform::Linear => ColorSpace::Linear,
            ViewTransform::Rec709 => ColorSpace::Rec709,
        };
        Self {
            upstream,
            info: NodeInfo {
                color_space: out_cs,
                ..upstream_info
            },
            transform,
        }
    }
}

impl Node for ViewTransformNode {
    fn info(&self) -> NodeInfo {
        self.info.clone()
    }

    fn compute(
        &self,
        request: Rect,
        upstream: &mut dyn Upstream,
    ) -> Result<Vec<f32>, PipelineError> {
        let mut pixels = upstream.request(self.upstream, request)?;
        match self.transform {
            ViewTransform::Linear => {} // pass-through
            ViewTransform::Srgb => {
                crate::color_math::apply_transfer(&mut pixels, crate::color_math::linear_to_srgb);
            }
            ViewTransform::Rec709 => {
                // BT.1886: gamma 2.4 (simplified — full BT.1886 has black level adjustment)
                for v in pixels.iter_mut() {
                    *v = v.powf(1.0 / 2.4);
                }
            }
        }
        Ok(pixels)
    }

    fn upstream_ids(&self) -> Vec<u32> {
        vec![self.upstream]
    }

    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> InputRectEstimate {
        InputRectEstimate::Exact(output.clamp(bounds_w, bounds_h))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;

    struct LinearSource {
        w: u32,
        h: u32,
        color: [f32; 4],
    }

    impl Node for LinearSource {
        fn info(&self) -> NodeInfo {
            NodeInfo {
                width: self.w,
                height: self.h,
                color_space: ColorSpace::Linear,
            }
        }
        fn compute(
            &self,
            request: Rect,
            _upstream: &mut dyn Upstream,
        ) -> Result<Vec<f32>, PipelineError> {
            let n = request.width as usize * request.height as usize;
            let mut px = Vec::with_capacity(n * 4);
            for _ in 0..n {
                px.extend_from_slice(&self.color);
            }
            Ok(px)
        }
        fn upstream_ids(&self) -> Vec<u32> {
            vec![]
        }
    }

    #[test]
    fn color_convert_linear_to_srgb() {
        let mut g = Graph::new(0);
        let src = g.add_node(Box::new(LinearSource {
            w: 2,
            h: 2,
            color: [0.214, 0.214, 0.214, 1.0], // linear mid-gray
        }));
        let src_info = g.node_info(src).unwrap();
        let conv = g.add_node(Box::new(ColorConvertNode::new(
            src,
            src_info,
            ColorSpace::Linear,
            ColorSpace::Srgb,
        )));

        let pixels = g.request_full(conv).unwrap();
        // Linear 0.214 → sRGB ≈ 0.5
        assert!(
            (pixels[0] - 0.5).abs() < 0.01,
            "expected ~0.5, got {}",
            pixels[0]
        );
    }

    #[test]
    fn color_convert_node_reports_output_colorspace() {
        let mut g = Graph::new(0);
        let src = g.add_node(Box::new(LinearSource {
            w: 2,
            h: 2,
            color: [0.5, 0.5, 0.5, 1.0],
        }));
        let src_info = g.node_info(src).unwrap();
        let conv = g.add_node(Box::new(ColorConvertNode::new(
            src,
            src_info,
            ColorSpace::Linear,
            ColorSpace::AcesCg,
        )));

        let info = g.node_info(conv).unwrap();
        assert_eq!(info.color_space, ColorSpace::AcesCg);
    }

    #[test]
    fn view_transform_srgb() {
        let mut g = Graph::new(0);
        let src = g.add_node(Box::new(LinearSource {
            w: 2,
            h: 2,
            color: [0.214, 0.214, 0.214, 1.0], // linear mid-gray
        }));
        let src_info = g.node_info(src).unwrap();
        let vt = g.add_node(Box::new(ViewTransformNode::new(
            src,
            src_info,
            ViewTransform::Srgb,
        )));

        let pixels = g.request_full(vt).unwrap();
        assert!((pixels[0] - 0.5).abs() < 0.01);
        let info = g.node_info(vt).unwrap();
        assert_eq!(info.color_space, ColorSpace::Srgb);
    }

    #[test]
    fn view_transform_linear_passthrough() {
        let mut g = Graph::new(0);
        let src = g.add_node(Box::new(LinearSource {
            w: 2,
            h: 2,
            color: [5.0, 0.5, 0.1, 1.0], // HDR
        }));
        let src_info = g.node_info(src).unwrap();
        let vt = g.add_node(Box::new(ViewTransformNode::new(
            src,
            src_info,
            ViewTransform::Linear,
        )));

        let pixels = g.request_full(vt).unwrap();
        assert!((pixels[0] - 5.0).abs() < 1e-6); // pass-through, no clamp
    }

    #[test]
    fn color_convert_chain_roundtrip() {
        let mut g = Graph::new(0);
        let src = g.add_node(Box::new(LinearSource {
            w: 2,
            h: 2,
            color: [0.5, 0.3, 0.1, 1.0],
        }));
        let info1 = g.node_info(src).unwrap();
        let to_acescg = g.add_node(Box::new(ColorConvertNode::new(
            src,
            info1,
            ColorSpace::Linear,
            ColorSpace::AcesCg,
        )));
        let info2 = g.node_info(to_acescg).unwrap();
        let back = g.add_node(Box::new(ColorConvertNode::new(
            to_acescg,
            info2,
            ColorSpace::AcesCg,
            ColorSpace::Linear,
        )));

        let pixels = g.request_full(back).unwrap();
        assert!((pixels[0] - 0.5).abs() < 0.001);
        assert!((pixels[1] - 0.3).abs() < 0.001);
        assert!((pixels[2] - 0.1).abs() < 0.001);
    }
}
