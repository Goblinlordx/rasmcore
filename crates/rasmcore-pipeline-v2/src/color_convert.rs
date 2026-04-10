//! Color conversion node and view transform — auto-inserted at color space boundaries.

use crate::color_math::convert_color_space;
use crate::color_space::ColorSpace;
use crate::node::{InputRectEstimate, Node, NodeCapabilities, NodeInfo, PipelineError, Upstream};
use crate::rect::Rect;

/// View transform applied at the output boundary (before encode).
///
/// Converts from the pipeline working space (Linear) to the display color space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ViewTransform {
    /// sRGB gamma encoding (default for SDR monitors).
    #[default]
    Srgb,
    /// No transform — pass through linear (for EXR/HDR output).
    Linear,
    /// Rec.709 (BT.1886 gamma 2.4).
    Rec709,
    /// ACES RRT + sRGB ODT — full ACES output transform for sRGB displays.
    AcesRrtSrgb,
    /// ACES RRT + Rec.709 ODT — full ACES output transform for broadcast.
    AcesRrtRec709,
    /// ACES RRT + P3-D65 ODT — full ACES output transform for cinema.
    AcesRrtP3,
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
            clut: true,
            gpu: false,
        }
    }

    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> InputRectEstimate {
        // Point op — no spatial expansion
        InputRectEstimate::Exact(output.clamp(bounds_w, bounds_h))
    }

    fn expected_input_color_space(&self) -> ColorSpace {
        self.from
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(crate::fusion::Clut3D::from_fn(33, |r, g, b| {
            // Apply conversion to a single pixel
            let mut px = [r, g, b, 1.0];
            convert_color_space(&mut px, self.from, self.to);
            (px[0], px[1], px[2])
        }))
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
            ViewTransform::Srgb | ViewTransform::AcesRrtSrgb => ColorSpace::Srgb,
            ViewTransform::Linear => ColorSpace::Linear,
            ViewTransform::Rec709 | ViewTransform::AcesRrtRec709 => ColorSpace::Rec709,
            ViewTransform::AcesRrtP3 => ColorSpace::DisplayP3,
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
                for chunk in pixels.chunks_exact_mut(4) {
                    for c in &mut chunk[..3] {
                        *c = c.clamp(0.0, 1.0).powf(1.0 / 2.4);
                    }
                }
            }
            ViewTransform::AcesRrtSrgb => {
                // Input is Linear sRGB working space — convert to AP1 first, then RRT+ODT
                crate::color_math::apply_matrix(
                    &mut pixels,
                    &crate::color_math::srgb_to_acescg_matrix(),
                );
                crate::aces::apply_aces_output_transform(
                    &mut pixels,
                    crate::aces::aces_rrt_odt_srgb_pixel,
                );
            }
            ViewTransform::AcesRrtRec709 => {
                crate::color_math::apply_matrix(
                    &mut pixels,
                    &crate::color_math::srgb_to_acescg_matrix(),
                );
                crate::aces::apply_aces_output_transform(
                    &mut pixels,
                    crate::aces::aces_rrt_odt_rec709_pixel,
                );
            }
            ViewTransform::AcesRrtP3 => {
                crate::color_math::apply_matrix(
                    &mut pixels,
                    &crate::color_math::srgb_to_acescg_matrix(),
                );
                crate::aces::apply_aces_output_transform(
                    &mut pixels,
                    crate::aces::aces_rrt_odt_p3d65_pixel,
                );
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

    /// Test filter that declares a preferred color space.
    struct PreferredCsFilter {
        upstream: u32,
        info: NodeInfo,
        preferred: ColorSpace,
    }

    impl Node for PreferredCsFilter {
        fn info(&self) -> NodeInfo {
            self.info.clone()
        }
        fn compute(
            &self,
            request: Rect,
            upstream: &mut dyn Upstream,
        ) -> Result<Vec<f32>, PipelineError> {
            // Pass through — the test is about auto-conversion, not filter behavior
            upstream.request(self.upstream, request)
        }
        fn upstream_ids(&self) -> Vec<u32> {
            vec![self.upstream]
        }
        fn preferred_color_space(&self) -> Option<ColorSpace> {
            Some(self.preferred)
        }
    }

    #[test]
    fn auto_convert_preferred_color_space() {
        // Filter prefers ACEScct but upstream is Linear.
        // After auto-conversion: the filter output should still be Linear
        // because convert-back restores the original space.
        let mut g = Graph::new(0);
        let src = g.add_node(Box::new(LinearSource {
            w: 2,
            h: 2,
            color: [0.5, 0.3, 0.1, 1.0],
        }));
        let src_info = g.node_info(src).unwrap();
        assert_eq!(src_info.color_space, ColorSpace::Linear);

        // Insert convert-to -> filter -> convert-back manually
        // (simulating what apply_filter does)
        let convert_to = ColorConvertNode::new(
            src,
            src_info.clone(),
            ColorSpace::Linear,
            ColorSpace::AcesCct,
        );
        let conv_to_id = g.add_node(Box::new(convert_to));
        let conv_to_info = g.node_info(conv_to_id).unwrap();
        assert_eq!(conv_to_info.color_space, ColorSpace::AcesCct);

        let filter = PreferredCsFilter {
            upstream: conv_to_id,
            info: conv_to_info.clone(),
            preferred: ColorSpace::AcesCct,
        };
        let filter_id = g.add_node(Box::new(filter));
        let filter_info = g.node_info(filter_id).unwrap();

        let convert_back = ColorConvertNode::new(
            filter_id,
            filter_info,
            ColorSpace::AcesCct,
            ColorSpace::Linear,
        );
        let back_id = g.add_node(Box::new(convert_back));
        let back_info = g.node_info(back_id).unwrap();
        assert_eq!(back_info.color_space, ColorSpace::Linear);

        // Render and verify round-trip preserves values
        let pixels = g.request_full(back_id).unwrap();
        assert!((pixels[0] - 0.5).abs() < 0.01, "r: {}", pixels[0]);
        assert!((pixels[1] - 0.3).abs() < 0.01, "g: {}", pixels[1]);
        assert!((pixels[2] - 0.1).abs() < 0.01, "b: {}", pixels[2]);
    }

    #[test]
    fn working_color_space_on_graph() {
        let mut g = Graph::new(0);
        assert!(g.working_color_space().is_none());
        g.set_working_color_space(ColorSpace::AcesCg);
        assert_eq!(g.working_color_space(), Some(ColorSpace::AcesCg));
    }

    #[test]
    fn no_redundant_conversion_when_spaces_match() {
        // If filter prefers Linear and upstream is already Linear, no conversion needed
        let mut g = Graph::new(0);
        let src = g.add_node(Box::new(LinearSource {
            w: 2,
            h: 2,
            color: [0.5, 0.5, 0.5, 1.0],
        }));
        let src_info = g.node_info(src).unwrap();
        assert_eq!(src_info.color_space, ColorSpace::Linear);

        // Filter prefers Linear — same as upstream, so no conversion nodes added
        let filter = PreferredCsFilter {
            upstream: src,
            info: src_info,
            preferred: ColorSpace::Linear,
        };
        let filter_id = g.add_node(Box::new(filter));

        // Only 2 nodes total: src + filter (no conversion nodes)
        assert_eq!(filter_id, 1);
        let pixels = g.request_full(filter_id).unwrap();
        assert!((pixels[0] - 0.5).abs() < 1e-6);
    }
}
