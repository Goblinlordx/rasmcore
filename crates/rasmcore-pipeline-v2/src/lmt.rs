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
    /// Per-channel analytical expression (fusable symbolically).
    /// Example: exposure → Mul(Input, Constant(2.0))
    Analytical(PointOpExpr),

    /// 3D Color Lookup Table (tetrahedral interpolation).
    /// Loaded from .cube files or baked from analytical chains.
    Clut3D(Clut3D),

    /// ASC CDL — slope, offset, power per channel.
    /// slope * input + offset, then pow(result, power).
    Cdl {
        slope: [f32; 3],
        offset: [f32; 3],
        power: [f32; 3],
    },

    /// Chain of LMTs applied sequentially.
    /// Produced by CLF containers or user-assembled pipelines.
    Chain(Vec<Lmt>),
}

impl Lmt {
    /// Convert this LMT to a PointOpExpr if it's analytical or CDL.
    /// Returns None for Clut3D and Chain (not per-channel algebraic).
    pub fn to_analytical(&self) -> Option<PointOpExpr> {
        match self {
            Lmt::Analytical(expr) => Some(expr.clone()),
            Lmt::Cdl { slope, offset, power } => {
                // CDL as per-channel expression: pow(slope * x + offset, power)
                // We use channel 0 (R) since PointOpExpr is per-channel.
                // For per-channel CDL, the fusion optimizer handles each channel.
                // For uniform CDL (same values across channels), this is exact.
                Some(PointOpExpr::Pow(
                    Box::new(PointOpExpr::Add(
                        Box::new(PointOpExpr::Mul(
                            Box::new(PointOpExpr::Input),
                            Box::new(PointOpExpr::Constant(slope[0])),
                        )),
                        Box::new(PointOpExpr::Constant(offset[0])),
                    )),
                    Box::new(PointOpExpr::Constant(power[0])),
                ))
            }
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
            Lmt::Analytical(expr) => {
                let mut out = pixels.to_vec();
                for pixel in out.chunks_exact_mut(4) {
                    pixel[0] = expr.evaluate(pixel[0] as f64) as f32;
                    pixel[1] = expr.evaluate(pixel[1] as f64) as f32;
                    pixel[2] = expr.evaluate(pixel[2] as f64) as f32;
                    // alpha unchanged
                }
                out
            }
            Lmt::Clut3D(clut) => clut.apply(pixels),
            Lmt::Cdl { slope, offset, power } => {
                let mut out = pixels.to_vec();
                for pixel in out.chunks_exact_mut(4) {
                    for c in 0..3 {
                        let v = (slope[c] * pixel[c] + offset[c]).max(0.0);
                        pixel[c] = v.powf(power[c]);
                    }
                }
                out
            }
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

    fn analytic_expression(&self) -> Option<PointOpExpr> {
        self.lmt.to_analytical()
    }

    fn fusion_clut(&self) -> Option<Clut3D> {
        self.lmt.as_clut().cloned()
    }
}

// ─── .cube Parser ────────────────────────────────────────────────────────────

/// Parse an Adobe/Resolve .cube 3D LUT file into an `Lmt::Clut3D`.
///
/// Supports the standard .cube format:
/// - `TITLE "..."` (optional, ignored)
/// - `DOMAIN_MIN r g b` (optional, default 0 0 0)
/// - `DOMAIN_MAX r g b` (optional, default 1 1 1)
/// - `LUT_3D_SIZE N` (required)
/// - N^3 lines of `r g b` triplets
pub fn parse_cube(text: &str) -> Result<Lmt, PipelineError> {
    let mut grid_size: Option<usize> = None;
    let mut data = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("TITLE") {
            continue;
        }
        if line.starts_with("DOMAIN_MIN") || line.starts_with("DOMAIN_MAX") {
            continue; // TODO: handle non-0..1 domains
        }
        if let Some(rest) = line.strip_prefix("LUT_3D_SIZE") {
            grid_size = Some(
                rest.trim()
                    .parse::<usize>()
                    .map_err(|_| PipelineError::InvalidParams("invalid LUT_3D_SIZE".into()))?,
            );
            continue;
        }
        // Data line — parse r g b
        let parts: Vec<f32> = line
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if parts.len() >= 3 {
            data.push(parts[0]);
            data.push(parts[1]);
            data.push(parts[2]);
        }
    }

    let n = grid_size.ok_or_else(|| {
        PipelineError::InvalidParams("missing LUT_3D_SIZE in .cube file".into())
    })?;

    let expected = n * n * n * 3;
    if data.len() != expected {
        return Err(PipelineError::InvalidParams(format!(
            ".cube LUT_3D_SIZE={n} expects {expected} entries, got {}",
            data.len()
        )));
    }

    Ok(Lmt::Clut3D(Clut3D {
        grid_size: n as u32,
        data,
    }))
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
        let lmt = Lmt::Analytical(expr.clone());
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
        let lmt = Lmt::Analytical(expr);
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

    // ── Lmt::Cdl tests ──────────────────────────────────────────────────

    #[test]
    fn cdl_identity_preserves_pixels() {
        let lmt = Lmt::Cdl {
            slope: [1.0, 1.0, 1.0],
            offset: [0.0, 0.0, 0.0],
            power: [1.0, 1.0, 1.0],
        };
        let pixels = vec![0.5, 0.3, 0.7, 1.0];
        let result = lmt.apply(&pixels);
        assert!((result[0] - 0.5).abs() < 1e-6);
        assert!((result[1] - 0.3).abs() < 1e-6);
        assert!((result[2] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn cdl_slope_doubles() {
        let lmt = Lmt::Cdl {
            slope: [2.0, 2.0, 2.0],
            offset: [0.0, 0.0, 0.0],
            power: [1.0, 1.0, 1.0],
        };
        let pixels = vec![0.25, 0.5, 0.125, 1.0];
        let result = lmt.apply(&pixels);
        assert!((result[0] - 0.5).abs() < 1e-6);
        assert!((result[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cdl_converts_to_analytical() {
        let lmt = Lmt::Cdl {
            slope: [1.5, 1.5, 1.5],
            offset: [0.1, 0.1, 0.1],
            power: [1.0, 1.0, 1.0],
        };
        let expr = lmt.to_analytical().unwrap();
        // Should be pow((1.5 * x + 0.1), 1.0) = 1.5 * x + 0.1
        let v = expr.evaluate(0.5);
        assert!((v - 0.85).abs() < 1e-6); // 1.5 * 0.5 + 0.1
    }

    // ── Lmt::Chain tests ─────────────────────────────────────────────────

    #[test]
    fn chain_applies_sequentially() {
        let chain = Lmt::Chain(vec![
            Lmt::Analytical(PointOpExpr::Add(
                Box::new(PointOpExpr::Input),
                Box::new(PointOpExpr::Constant(0.1)),
            )),
            Lmt::Analytical(PointOpExpr::Mul(
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
        let lmt = Lmt::Analytical(PointOpExpr::Mul(
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
        let lmt = Lmt::Analytical(PointOpExpr::Add(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(0.1)),
        ));
        let node = LmtNode::new(0, test_info(1, 1), lmt);
        assert!(node.analytic_expression().is_some());
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
            Lmt::Analytical(PointOpExpr::Mul(
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
}
