//! Look Modification Transform (LMT) — unified color transform type.
//!
//! An LMT represents a creative color transform in the ACES pipeline.
//! All forms — analytical expressions, 3D CLUTs, ASC CDL, or chains of
//! these — are represented as variants of the [`Lmt`] enum.
//!
//! Predefined filters (exposure, contrast, gamma, etc.) are just convenience
//! constructors that produce `Lmt` values. User-loaded `.cube` or `.clf`
//! files enter through the same type. The fusion optimizer composes them
//! uniformly.

use crate::fusion::Clut3D;
use crate::node::{
    Node, NodeCapabilities, NodeInfo, PipelineError,
    Upstream,
};
use crate::ops::PointOpExpr;
use crate::rect::Rect;

// ─── Lmt Enum ────────────────────────────────────────────────────────────────

/// A Look Modification Transform — any per-pixel color operation.
#[derive(Debug, Clone)]
pub enum Lmt {
    /// Per-channel analytical expressions [R, G, B] (fusable symbolically).
    /// Each channel has its own expression tree, composed independently.
    /// For uniform operations (same formula all channels), all 3 are identical.
    /// Example: exposure → [Mul(Input, 2.0), Mul(Input, 2.0), Mul(Input, 2.0)]
    /// Example: CDL → [pow(s[0]*x+o[0], p[0]), pow(s[1]*x+o[1], p[1]), pow(s[2]*x+o[2], p[2])]
    Analytical([PointOpExpr; 3]),

    /// 3D Color Lookup Table (tetrahedral interpolation).
    /// Loaded from .cube files or baked from analytical chains.
    Clut3D(Clut3D),

    /// Chain of LMTs applied sequentially.
    /// Produced by CLF containers or user-assembled pipelines.
    Chain(Vec<Lmt>),
}

/// Convenience: create a uniform Analytical LMT (same expression all channels).
pub fn analytical_uniform(expr: PointOpExpr) -> Lmt {
    Lmt::Analytical([expr.clone(), expr.clone(), expr])
}

/// Build per-channel CDL expressions: pow(slope * x + offset, power) per channel.
pub fn analytical_cdl(slope: [f32; 3], offset: [f32; 3], power: [f32; 3]) -> Lmt {
    let build = |s: f32, o: f32, p: f32| -> PointOpExpr {
        PointOpExpr::Pow(
            Box::new(PointOpExpr::Add(
                Box::new(PointOpExpr::Mul(
                    Box::new(PointOpExpr::Input),
                    Box::new(PointOpExpr::Constant(s)),
                )),
                Box::new(PointOpExpr::Constant(o)),
            )),
            Box::new(PointOpExpr::Constant(p)),
        )
    };
    Lmt::Analytical([
        build(slope[0], offset[0], power[0]),
        build(slope[1], offset[1], power[1]),
        build(slope[2], offset[2], power[2]),
    ])
}

impl Lmt {
    /// Extract per-channel analytical expressions if this is an Analytical LMT.
    /// Returns None for Clut3D and Chain.
    pub fn to_analytical(&self) -> Option<[PointOpExpr; 3]> {
        match self {
            Lmt::Analytical(exprs) => Some(exprs.clone()),
            _ => None,
        }
    }

    /// Check if this LMT is a CLUT.
    pub fn as_clut(&self) -> Option<&Clut3D> {
        match self {
            Lmt::Clut3D(clut) => Some(clut),
            _ => None,
        }
    }

    /// Apply this LMT to f32 RGBA pixel data.
    pub fn apply(&self, pixels: &[f32]) -> Vec<f32> {
        match self {
            Lmt::Analytical([expr_r, expr_g, expr_b]) => {
                let mut out = pixels.to_vec();
                for pixel in out.chunks_exact_mut(4) {
                    pixel[0] = expr_r.evaluate(pixel[0] as f64) as f32;
                    pixel[1] = expr_g.evaluate(pixel[1] as f64) as f32;
                    pixel[2] = expr_b.evaluate(pixel[2] as f64) as f32;
                    // alpha unchanged
                }
                out
            }
            Lmt::Clut3D(clut) => clut.apply(pixels),
            Lmt::Chain(stages) => {
                let mut data = pixels.to_vec();
                for stage in stages {
                    data = stage.apply(&data);
                }
                data
            }
        }
    }
}

// ─── LmtNode ─────────────────────────────────────────────────────────────────

/// Pipeline node that applies an LMT.
///
/// Wraps any `Lmt` value as a graph node. Exposes `analytic_expression()`
/// and `fusion_clut()` for compatibility with the existing fusion optimizer.
pub struct LmtNode {
    upstream: u32,
    info: NodeInfo,
    lmt: Lmt,
}

impl LmtNode {
    pub fn new(upstream: u32, info: NodeInfo, lmt: Lmt) -> Self {
        Self { upstream, info, lmt }
    }

    /// Access the underlying LMT.
    pub fn lmt(&self) -> &Lmt {
        &self.lmt
    }
}

impl Node for LmtNode {
    fn info(&self) -> NodeInfo {
        self.info.clone()
    }

    fn compute(
        &self,
        request: Rect,
        upstream: &mut dyn Upstream,
    ) -> Result<Vec<f32>, PipelineError> {
        let input = upstream.request(self.upstream, request)?;
        Ok(self.lmt.apply(&input))
    }

    fn upstream_ids(&self) -> Vec<u32> {
        vec![self.upstream]
    }

    fn capabilities(&self) -> NodeCapabilities {
        NodeCapabilities {
            gpu: false, // GPU dispatch handled by fusion (fused WGSL kernel)
            analytic: self.lmt.to_analytical().is_some(),
            affine: false,
            clut: self.lmt.as_clut().is_some(),
        }
    }

    fn analytic_expression_per_channel(&self) -> Option<[PointOpExpr; 3]> {
        self.lmt.to_analytical()
    }

    fn fusion_clut(&self) -> Option<Clut3D> {
        self.lmt.as_clut().cloned()
    }

    fn as_lmt(&self) -> Option<&Lmt> {
        Some(&self.lmt)
    }
}

// ─── .cube Parser ────────────────────────────────────────────────────────────

/// Parse an Adobe/Resolve .cube LUT file (1D or 3D) into an `Lmt::Clut3D`.
///
/// Supports the standard .cube format:
/// - `TITLE "..."` (optional, ignored)
/// - `DOMAIN_MIN r g b` / `DOMAIN_MAX r g b` (optional)
/// - `LUT_3D_SIZE N` — 3D LUT with N^3 entries
/// - `LUT_1D_SIZE N` — 1D LUT with N per-channel entries (converted to 3D)
///
/// Delegates to `filters::grading::parse_cube_lut` for the actual parsing,
/// then wraps the result in `Lmt::Clut3D`.
pub fn parse_cube(text: &str) -> Result<Lmt, PipelineError> {
    let clut = crate::filters::grading::parse_cube_lut(text)?;
    Ok(Lmt::Clut3D(clut))
}

// ─── Basic CLF Parser ────────────────────────────────────────────────────────

/// Parse a minimal CLF (Common LUT Format) XML file into an `Lmt::Chain`.
///
/// This handles the most common CLF structure: a ProcessList containing
/// Matrix, LUT1D, LUT3D, and Range operators. Full CLF spec coverage
/// is out of scope — this covers the 80% case (1D shaper + 3D LUT).
pub fn parse_clf(_xml: &str) -> Result<Lmt, PipelineError> {
    // Minimal CLF parsing — extract LUT3D elements from the XML.
    // Full implementation deferred to a dedicated CLF track.
    Err(PipelineError::InvalidParams(
        "CLF parsing not yet implemented — use .cube format".into(),
    ))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color_space::ColorSpace;
    use crate::graph::Graph;

    fn test_info(w: u32, h: u32) -> NodeInfo {
        NodeInfo { width: w, height: h, color_space: ColorSpace::Linear }
    }

    // ── Source node for tests ────────────────────────────────────────────

    struct SolidSource {
        w: u32,
        h: u32,
        color: [f32; 4],
    }

    impl Node for SolidSource {
        fn info(&self) -> NodeInfo {
            test_info(self.w, self.h)
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

    // ── Lmt::Analytical tests ────────────────────────────────────────────

    #[test]
    fn analytical_lmt_matches_pointopexpr() {
        let expr = PointOpExpr::Mul(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(2.0)),
        );
        let lmt = analytical_uniform(expr.clone());
        let pixels = vec![0.25, 0.5, 0.125, 1.0];
        let result = lmt.apply(&pixels);
        assert!((result[0] - 0.5).abs() < 1e-6);
        assert!((result[1] - 1.0).abs() < 1e-6);
        assert!((result[2] - 0.25).abs() < 1e-6);
        assert_eq!(result[3], 1.0); // alpha unchanged
    }

    #[test]
    fn analytical_lmt_to_analytical_roundtrip() {
        let expr = PointOpExpr::Add(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(0.1)),
        );
        let lmt = analytical_uniform(expr);
        assert!(lmt.to_analytical().is_some());
    }

    // ── Lmt::Clut3D tests ───────────────────────────────────────────────

    #[test]
    fn identity_clut_preserves_pixels() {
        let clut = Clut3D::identity(17);
        let lmt = Lmt::Clut3D(clut);
        let pixels = vec![0.3, 0.6, 0.9, 1.0, 0.0, 0.5, 1.0, 0.8];
        let result = lmt.apply(&pixels);
        for i in 0..3 {
            assert!((result[i] - pixels[i]).abs() < 0.01, "identity CLUT failed at channel {i}");
        }
        assert_eq!(result[3], 1.0);
    }

    #[test]
    fn clut_lmt_exposes_fusion_clut() {
        let clut = Clut3D::identity(9);
        let lmt = Lmt::Clut3D(clut);
        assert!(lmt.as_clut().is_some());
        assert!(lmt.to_analytical().is_none());
    }

    // ── CDL via Analytical tests ──────────────────────────────────────

    #[test]
    fn cdl_identity_preserves_pixels() {
        let lmt = analytical_cdl([1.0; 3], [0.0; 3], [1.0; 3]);
        let pixels = vec![0.5, 0.3, 0.7, 1.0];
        let result = lmt.apply(&pixels);
        assert!((result[0] - 0.5).abs() < 1e-6);
        assert!((result[1] - 0.3).abs() < 1e-6);
        assert!((result[2] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn cdl_slope_doubles() {
        let lmt = analytical_cdl([2.0; 3], [0.0; 3], [1.0; 3]);
        let pixels = vec![0.25, 0.5, 0.125, 1.0];
        let result = lmt.apply(&pixels);
        assert!((result[0] - 0.5).abs() < 1e-6);
        assert!((result[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cdl_converts_to_analytical() {
        let lmt = analytical_cdl([1.5; 3], [0.1; 3], [1.0; 3]);
        let exprs = lmt.to_analytical().unwrap();
        // Should be pow((1.5 * x + 0.1), 1.0) = 1.5 * x + 0.1
        let v = exprs[0].evaluate(0.5);
        assert!((v - 0.85).abs() < 1e-6); // 1.5 * 0.5 + 0.1
    }

    #[test]
    fn cdl_per_channel_different_slopes() {
        let lmt = analytical_cdl([2.0, 1.0, 0.5], [0.0; 3], [1.0; 3]);
        let pixels = vec![0.5, 0.5, 0.5, 1.0];
        let result = lmt.apply(&pixels);
        assert!((result[0] - 1.0).abs() < 1e-6, "R: 0.5 * 2.0 = 1.0");
        assert!((result[1] - 0.5).abs() < 1e-6, "G: 0.5 * 1.0 = 0.5");
        assert!((result[2] - 0.25).abs() < 1e-6, "B: 0.5 * 0.5 = 0.25");
    }

    // ── Lmt::Chain tests ─────────────────────────────────────────────────

    #[test]
    fn chain_applies_sequentially() {
        let chain = Lmt::Chain(vec![
            analytical_uniform(PointOpExpr::Add(
                Box::new(PointOpExpr::Input),
                Box::new(PointOpExpr::Constant(0.1)),
            )),
            analytical_uniform(PointOpExpr::Mul(
                Box::new(PointOpExpr::Input),
                Box::new(PointOpExpr::Constant(2.0)),
            )),
        ]);
        let pixels = vec![0.5, 0.5, 0.5, 1.0];
        let result = chain.apply(&pixels);
        // (0.5 + 0.1) * 2.0 = 1.2
        assert!((result[0] - 1.2).abs() < 1e-5);
    }

    // ── LmtNode in Graph tests ───────────────────────────────────────────

    #[test]
    fn lmt_node_renders_in_graph() {
        let mut g = Graph::new(0);
        let src = g.add_node(Box::new(SolidSource {
            w: 2, h: 2, color: [0.5, 0.3, 0.1, 1.0],
        }));
        let info = g.node_info(src).unwrap();
        let lmt = analytical_uniform(PointOpExpr::Mul(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(2.0)),
        ));
        let node = g.add_node(Box::new(LmtNode::new(src, info, lmt)));
        let pixels = g.request_full(node).unwrap();
        assert!((pixels[0] - 1.0).abs() < 1e-6); // 0.5 * 2
        assert!((pixels[1] - 0.6).abs() < 1e-6); // 0.3 * 2
    }

    #[test]
    fn lmt_node_exposes_analytic_for_fusion() {
        let lmt = analytical_uniform(PointOpExpr::Add(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(0.1)),
        ));
        let node = LmtNode::new(0, test_info(1, 1), lmt);
        assert!(node.analytic_expression_per_channel().is_some());
        assert!(node.capabilities().analytic);
    }

    #[test]
    fn lmt_node_chain_with_filter_nodes() {
        use crate::filter_node::FilterNode;
        use crate::ops::Filter;

        // Simple brightness filter for comparison
        #[derive(Clone)]
        struct Brightness { amount: f32 }
        impl Filter for Brightness {
            fn compute(&self, input: &[f32], _w: u32, _h: u32) -> Result<Vec<f32>, PipelineError> {
                Ok(input.iter().enumerate().map(|(i, &v)| {
                    if i % 4 == 3 { v } else { v + self.amount }
                }).collect())
            }
        }

        let mut g = Graph::new(0);
        let src = g.add_node(Box::new(SolidSource {
            w: 2, h: 2, color: [0.5, 0.5, 0.5, 1.0],
        }));
        let info = g.node_info(src).unwrap();

        // LmtNode: multiply by 2
        let lmt_node = g.add_node(Box::new(LmtNode::new(
            src, info.clone(),
            analytical_uniform(PointOpExpr::Mul(
                Box::new(PointOpExpr::Input),
                Box::new(PointOpExpr::Constant(2.0)),
            )),
        )));

        // FilterNode: add 0.1
        let filter_node = g.add_node(Box::new(FilterNode::new(
            lmt_node, info,
            Brightness { amount: 0.1 },
        )));

        let pixels = g.request_full(filter_node).unwrap();
        // 0.5 * 2.0 + 0.1 = 1.1
        assert!((pixels[0] - 1.1).abs() < 1e-5);
    }

    // ── .cube Parser tests ───────────────────────────────────────────────

    #[test]
    fn parse_cube_identity() {
        let cube = "\
TITLE \"Identity\"
LUT_3D_SIZE 2

0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
1.0 1.0 0.0
0.0 0.0 1.0
1.0 0.0 1.0
0.0 1.0 1.0
1.0 1.0 1.0
";
        let lmt = parse_cube(cube).unwrap();
        assert!(matches!(lmt, Lmt::Clut3D(_)));
        // Apply to test pixel
        let pixels = vec![0.5, 0.5, 0.5, 1.0];
        let result = lmt.apply(&pixels);
        for c in 0..3 {
            assert!((result[c] - 0.5).abs() < 0.05, "identity cube failed at ch {c}: {}", result[c]);
        }
    }

    #[test]
    fn parse_cube_missing_size_errors() {
        let cube = "0.0 0.0 0.0\n1.0 1.0 1.0\n";
        assert!(parse_cube(cube).is_err());
    }

    #[test]
    fn parse_cube_wrong_entry_count_errors() {
        let cube = "LUT_3D_SIZE 2\n0.0 0.0 0.0\n";
        assert!(parse_cube(cube).is_err());
    }

    // ── Fusion tests — LmtNode participates in existing fusion optimizer ─

    #[test]
    fn analytical_lmt_chain_fuses_in_graph() {
        // Two LmtNode(Analytical) should fuse into one FusedPointOpNode
        let mut g = Graph::new(0);
        let src = g.add_node(Box::new(SolidSource {
            w: 2, h: 2, color: [0.5, 0.5, 0.5, 1.0],
        }));
        let info = g.node_info(src).unwrap();

        // LMT 1: multiply by 2 (exposure +1 EV)
        let n1 = g.add_node(Box::new(LmtNode::new(
            src, info.clone(),
            analytical_uniform(PointOpExpr::Mul(
                Box::new(PointOpExpr::Input),
                Box::new(PointOpExpr::Constant(2.0)),
            )),
        )));

        // LMT 2: multiply by 0.5 (exposure -1 EV)
        let n2 = g.add_node(Box::new(LmtNode::new(
            n1, info,
            analytical_uniform(PointOpExpr::Mul(
                Box::new(PointOpExpr::Input),
                Box::new(PointOpExpr::Constant(0.5)),
            )),
        )));

        // Should fuse: 2.0 * 0.5 = 1.0 (identity, but FusedPointOpNode still created)
        let result = g.request_full(n2).unwrap();
        // 0.5 * 2.0 * 0.5 = 0.5 (round-trip)
        assert!((result[0] - 0.5).abs() < 1e-5, "fused LMT chain should produce 0.5, got {}", result[0]);
    }

    #[test]
    fn cdl_lmt_fuses_with_analytical() {
        // CDL converts to analytical, so it should fuse with adjacent analytical LMTs
        let mut g = Graph::new(0);
        let src = g.add_node(Box::new(SolidSource {
            w: 1, h: 1, color: [0.5, 0.5, 0.5, 1.0],
        }));
        let info = g.node_info(src).unwrap();

        // CDL: slope=2, offset=0, power=1 → just doubles (like exposure +1 EV)
        let n1 = g.add_node(Box::new(LmtNode::new(
            src, info.clone(),
            analytical_cdl([2.0; 3], [0.0; 3], [1.0; 3]),
        )));

        // Analytical: add 0.1
        let n2 = g.add_node(Box::new(LmtNode::new(
            n1, info,
            analytical_uniform(PointOpExpr::Add(
                Box::new(PointOpExpr::Input),
                Box::new(PointOpExpr::Constant(0.1)),
            )),
        )));

        let result = g.request_full(n2).unwrap();
        // 0.5 * 2.0 + 0.1 = 1.1
        assert!((result[0] - 1.1).abs() < 1e-4, "CDL+analytical fusion should produce 1.1, got {}", result[0]);
    }

    #[test]
    fn mixed_lmt_and_filter_nodes_fuse() {
        use crate::filter_node::FilterNode;
        use crate::ops::Filter;

        #[derive(Clone)]
        struct AddOffset { amount: f32 }
        impl Filter for AddOffset {
            fn compute(&self, input: &[f32], _w: u32, _h: u32) -> Result<Vec<f32>, PipelineError> {
                Ok(input.iter().enumerate().map(|(i, &v)| {
                    if i % 4 == 3 { v } else { v + self.amount }
                }).collect())
            }
            fn analytic_expression_per_channel(&self) -> Option<[PointOpExpr; 3]> {
                let expr = PointOpExpr::Add(
                    Box::new(PointOpExpr::Input),
                    Box::new(PointOpExpr::Constant(self.amount)),
                );
                Some([expr.clone(), expr.clone(), expr])
            }
        }

        let mut g = Graph::new(0);
        let src = g.add_node(Box::new(SolidSource {
            w: 1, h: 1, color: [0.5, 0.5, 0.5, 1.0],
        }));
        let info = g.node_info(src).unwrap();

        // FilterNode: add 0.1
        let f1 = g.add_node(Box::new(FilterNode::new(
            src, info.clone(), AddOffset { amount: 0.1 },
        )));

        // LmtNode: multiply by 2
        let l1 = g.add_node(Box::new(LmtNode::new(
            f1, info.clone(),
            analytical_uniform(PointOpExpr::Mul(
                Box::new(PointOpExpr::Input),
                Box::new(PointOpExpr::Constant(2.0)),
            )),
        )));

        // FilterNode: add 0.05
        let f2 = g.add_node(Box::new(FilterNode::new(
            l1, info, AddOffset { amount: 0.05 },
        )));

        let result = g.request_full(f2).unwrap();
        // (0.5 + 0.1) * 2.0 + 0.05 = 1.25
        assert!((result[0] - 1.25).abs() < 1e-4, "mixed chain should produce 1.25, got {}", result[0]);
    }

    #[test]
    fn lmt_node_as_lmt_returns_value() {
        use crate::node::Node;
        let lmt = analytical_uniform(PointOpExpr::Input);
        let node = LmtNode::new(0, test_info(1, 1), lmt);
        assert!(node.as_lmt().is_some());
        assert!(matches!(node.as_lmt().unwrap(), Lmt::Analytical(_)));
    }
}
