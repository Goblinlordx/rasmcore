//! FilterNode — bridges the Filter/GpuFilter/SpatialFilter traits to the
//! pipeline graph's Node trait.
//!
//! This is the glue between "I wrote a filter" and "the pipeline can execute it."
//! The derive(Operation) macro generates this wrapper automatically, but it can
//! also be constructed manually.

use crate::node::{
    GpuShader, InputRectEstimate, Node, NodeCapabilities, NodeInfo, PipelineError, TileHint,
    Upstream,
};
use crate::ops::{Filter, GpuFilter};
use crate::rect::Rect;

/// IO_F32 fragment — declares f32 input/output bindings and load_pixel/store_pixel.
///
/// Auto-composed with every GPU shader body. Shader bodies use only
/// `load_pixel(idx)` and `store_pixel(idx, color)` — they never declare
/// their own bindings.
pub const IO_F32: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;

fn load_pixel(idx: u32) -> vec4<f32> {
  return input[idx];
}

fn store_pixel(idx: u32, color: vec4<f32>) {
  output[idx] = color;
}
"#;

/// Compose a complete WGSL shader from a filter's shader body.
///
/// Prepends the io_f32 bindings. The filter body should declare only
/// `@group(0) @binding(2) var<uniform> params: Params` and the entry point.
pub fn compose_shader(body: &str) -> String {
    let mut src = String::with_capacity(IO_F32.len() + body.len() + 1);
    src.push_str(IO_F32);
    src.push('\n');
    src.push_str(body);
    src
}

/// Wraps a `Filter` (+ optional `GpuFilter`, `SpatialFilter`) into a pipeline `Node`.
///
/// This is the standard way to connect a filter to the V2 graph.
/// The pipeline calls `Node::compute()` which delegates to `Filter::compute()`.
/// GPU dispatch checks `GpuFilter::shader_body()` and auto-composes with io_f32.
/// Unified filter node — wraps any `Filter` into the pipeline graph.
///
/// Capabilities are detected at runtime from the filter:
/// - `Filter::GPU_SHADER_BODY` → GPU dispatch
/// - `Filter::tile_overlap()` → spatial tile expansion
/// - `Filter::analytic_expression()` → fusion optimizer
///
/// One constructor. No variants. The node adapts to whatever the filter provides.
pub struct FilterNode<F: Filter> {
    upstream: u32,
    info: NodeInfo,
    filter: F,
}

impl<F: Filter> FilterNode<F> {
    /// Create a FilterNode. Capabilities detected from the filter at runtime.
    pub fn new(upstream: u32, info: NodeInfo, filter: F) -> Self {
        Self { upstream, info, filter }
    }

    /// Backward compat — alias for `new()`.
    pub fn point_op(upstream: u32, info: NodeInfo, filter: F) -> Self {
        Self::new(upstream, info, filter)
    }

    /// Backward compat — overlap is now detected from `filter.tile_overlap()`.
    pub fn spatial(upstream: u32, info: NodeInfo, filter: F, _tile_overlap: u32) -> Self {
        Self::new(upstream, info, filter)
    }

    /// Get a reference to the underlying filter.
    pub fn filter(&self) -> &F {
        &self.filter
    }
}

impl<F: Filter + 'static> Node for FilterNode<F> {
    fn info(&self) -> NodeInfo {
        self.info.clone()
    }

    fn compute(
        &self,
        request: Rect,
        upstream: &mut dyn Upstream,
    ) -> Result<Vec<f32>, PipelineError> {
        let input_rect = match self.input_rect(request, self.info.width, self.info.height) {
            InputRectEstimate::Exact(r) => r,
            InputRectEstimate::UpperBound(r) => r,
            InputRectEstimate::FullImage => Rect::new(0, 0, self.info.width, self.info.height),
        };

        let input = upstream.request(self.upstream, input_rect)?;
        let output = self.filter.compute(&input, input_rect.width, input_rect.height)?;

        if input_rect == request {
            Ok(output)
        } else {
            Ok(crop_f32(&output, input_rect, request))
        }
    }

    fn gpu_shader(&self, width: u32, height: u32) -> Option<GpuShader> {
        let body = self.filter.gpu_shader_body()?;
        let params = self.filter.gpu_params(width, height)?;
        Some(GpuShader {
            body: compose_shader(body),
            entry_point: self.filter.gpu_entry_point(),
            workgroup_size: self.filter.gpu_workgroup_size(),
            params,
            extra_buffers: self.filter.gpu_extra_buffers(),
            reduction_buffers: vec![],
        })
    }

    fn gpu_shaders(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        self.filter.gpu_shader_passes(width, height)
    }

    fn upstream_ids(&self) -> Vec<u32> {
        vec![self.upstream]
    }

    fn capabilities(&self) -> NodeCapabilities {
        NodeCapabilities {
            gpu: self.filter.gpu_shader_body().is_some(),
            analytic: false, // detected via analytic_expression() at runtime
            affine: false,
            clut: false,
        }
    }

    fn analytic_expression(&self) -> Option<crate::ops::PointOpExpr> {
        self.filter.analytic_expression()
    }

    fn tile_hint(&self) -> Option<TileHint> {
        let r = self.filter.tile_overlap();
        if r > 0 {
            Some(TileHint {
                min_efficient_tile: r * 8,
                tile_overlap: r,
            })
        } else {
            None
        }
    }

    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> InputRectEstimate {
        let r = self.filter.tile_overlap();
        if r > 0 {
            InputRectEstimate::Exact(output.expand_uniform(r, bounds_w, bounds_h))
        } else {
            InputRectEstimate::Exact(output.clamp(bounds_w, bounds_h))
        }
    }
}

/// Legacy GPU filter node — use `FilterNode` instead.
/// FilterNode now detects GPU capability from `Filter::GPU_SHADER_BODY`.
#[deprecated(note = "Use FilterNode::new() — GPU detected from Filter::GPU_SHADER_BODY")]
#[allow(deprecated)]
pub struct GpuFilterNode<F: Filter + GpuFilter> {
    upstream: u32,
    info: NodeInfo,
    filter: F,
    tile_overlap: u32,
    capabilities: NodeCapabilities,
}

#[allow(deprecated)]
impl<F: Filter + GpuFilter> GpuFilterNode<F> {
    pub fn point_op(upstream: u32, info: NodeInfo, filter: F) -> Self {
        Self {
            upstream,
            info,
            filter,
            tile_overlap: 0,
            capabilities: NodeCapabilities {
                gpu: true,
                ..NodeCapabilities::default()
            },
        }
    }

    pub fn spatial(upstream: u32, info: NodeInfo, filter: F, tile_overlap: u32) -> Self {
        Self {
            upstream,
            info,
            filter,
            tile_overlap,
            capabilities: NodeCapabilities {
                gpu: true,
                ..NodeCapabilities::default()
            },
        }
    }

    pub fn with_capabilities(mut self, caps: NodeCapabilities) -> Self {
        self.capabilities = caps;
        self.capabilities.gpu = true; // always GPU capable
        self
    }
}

#[allow(deprecated)]
impl<F: Filter + GpuFilter + 'static> Node for GpuFilterNode<F> {
    fn info(&self) -> NodeInfo {
        self.info.clone()
    }

    fn compute(
        &self,
        request: Rect,
        upstream: &mut dyn Upstream,
    ) -> Result<Vec<f32>, PipelineError> {
        let input_rect = match self.input_rect(request, self.info.width, self.info.height) {
            InputRectEstimate::Exact(r) => r,
            InputRectEstimate::UpperBound(r) => r,
            InputRectEstimate::FullImage => Rect::new(0, 0, self.info.width, self.info.height),
        };

        let input = upstream.request(self.upstream, input_rect)?;
        let output = self.filter.compute(&input, input_rect.width, input_rect.height)?;

        if input_rect == request {
            Ok(output)
        } else {
            Ok(crop_f32(&output, input_rect, request))
        }
    }

    fn gpu_shader(&self, width: u32, height: u32) -> Option<GpuShader> {
        let body = compose_shader(self.filter.shader_body());
        Some(GpuShader {
            body,
            entry_point: self.filter.entry_point(),
            workgroup_size: self.filter.workgroup_size(),
            params: self.filter.params(width, height),
            extra_buffers: self.filter.extra_buffers(),
            reduction_buffers: vec![],
        })
    }

    fn upstream_ids(&self) -> Vec<u32> {
        vec![self.upstream]
    }

    fn capabilities(&self) -> NodeCapabilities {
        self.capabilities
    }

    fn tile_hint(&self) -> Option<TileHint> {
        if self.tile_overlap > 0 {
            Some(TileHint {
                min_efficient_tile: self.tile_overlap * 8,
                tile_overlap: self.tile_overlap,
            })
        } else {
            None
        }
    }

    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> InputRectEstimate {
        if self.tile_overlap > 0 {
            InputRectEstimate::Exact(
                output.expand_uniform(self.tile_overlap, bounds_w, bounds_h),
            )
        } else {
            InputRectEstimate::Exact(output.clamp(bounds_w, bounds_h))
        }
    }
}

/// Crop f32 pixel data from a source rect to a destination rect.
fn crop_f32(data: &[f32], src_rect: Rect, dst_rect: Rect) -> Vec<f32> {
    let sw = src_rect.width as usize;
    let dw = dst_rect.width as usize;
    let dh = dst_rect.height as usize;
    let dx = (dst_rect.x - src_rect.x) as usize;
    let dy = (dst_rect.y - src_rect.y) as usize;

    let mut out = Vec::with_capacity(dw * dh * 4);
    for row in 0..dh {
        let src_off = ((dy + row) * sw + dx) * 4;
        out.extend_from_slice(&data[src_off..src_off + dw * 4]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color_space::ColorSpace;
    use crate::graph::Graph;
    use crate::ops::{AnalyticOp, PointOpExpr};

    // ─── Reference filter: Brightness (point op + analytic) ──────────────────

    struct Brightness {
        offset: f32,
    }

    impl Filter for Brightness {
        fn compute(&self, input: &[f32], _w: u32, _h: u32) -> Result<Vec<f32>, PipelineError> {
            Ok(input.iter().map(|&v| v + self.offset).collect())
        }
    }

    impl AnalyticOp for Brightness {
        fn expression(&self) -> PointOpExpr {
            PointOpExpr::Add(
                Box::new(PointOpExpr::Input),
                Box::new(PointOpExpr::Constant(self.offset)),
            )
        }
    }

    // ─── Reference filter: Invert (point op + analytic) ──────────────────────

    struct Invert;

    impl Filter for Invert {
        fn compute(&self, input: &[f32], _w: u32, _h: u32) -> Result<Vec<f32>, PipelineError> {
            Ok(input.iter().map(|&v| 1.0 - v).collect())
        }
    }

    impl AnalyticOp for Invert {
        fn expression(&self) -> PointOpExpr {
            PointOpExpr::Sub(
                Box::new(PointOpExpr::Constant(1.0)),
                Box::new(PointOpExpr::Input),
            )
        }
    }

    // ─── Reference filter: BoxBlur (spatial, needs overlap) ──────────────────

    struct BoxBlur {
        radius: u32,
    }

    impl Filter for BoxBlur {
        fn compute(&self, input: &[f32], w: u32, h: u32) -> Result<Vec<f32>, PipelineError> {
            let r = self.radius as i32;
            let w = w as usize;
            let h = h as usize;
            let mut output = vec![0.0f32; w * h * 4];
            for y in 0..h {
                for x in 0..w {
                    let mut sum = [0.0f32; 4];
                    let mut count = 0.0f32;
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let sx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                            let sy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                            let idx = (sy * w + sx) * 4;
                            for c in 0..4 {
                                sum[c] += input[idx + c];
                            }
                            count += 1.0;
                        }
                    }
                    let out_idx = (y * w + x) * 4;
                    for c in 0..4 {
                        output[out_idx + c] = sum[c] / count;
                    }
                }
            }
            Ok(output)
        }

        fn tile_overlap(&self) -> u32 {
            self.radius
        }
    }

    // ─── Source node for tests ────────────────────────────────────────────────

    struct SolidSource {
        w: u32,
        h: u32,
        color: [f32; 4],
    }

    impl Node for SolidSource {
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

    // ─── Tests ───────────────────────────────────────────────────────────────

    #[test]
    fn brightness_filter_node() {
        let mut g = Graph::new(0);
        let src = g.add_node(Box::new(SolidSource {
            w: 4,
            h: 4,
            color: [0.5, 0.3, 0.1, 1.0],
        }));
        let info = g.node_info(src).unwrap();
        let bright = FilterNode::point_op(src, info, Brightness { offset: 0.2 });
        let node_id = g.add_node(Box::new(bright));

        let pixels = g.request_full(node_id).unwrap();
        assert!((pixels[0] - 0.7).abs() < 1e-6); // 0.5 + 0.2
        assert!((pixels[1] - 0.5).abs() < 1e-6); // 0.3 + 0.2
    }

    #[test]
    fn invert_filter_node() {
        let mut g = Graph::new(0);
        let src = g.add_node(Box::new(SolidSource {
            w: 2,
            h: 2,
            color: [0.3, 0.7, 0.0, 1.0],
        }));
        let info = g.node_info(src).unwrap();
        let inv = FilterNode::point_op(src, info, Invert);
        let node_id = g.add_node(Box::new(inv));

        let pixels = g.request_full(node_id).unwrap();
        assert!((pixels[0] - 0.7).abs() < 1e-6); // 1.0 - 0.3
        assert!((pixels[1] - 0.3).abs() < 1e-6); // 1.0 - 0.7
    }

    #[test]
    fn chain_brightness_invert() {
        let mut g = Graph::new(0);
        let src = g.add_node(Box::new(SolidSource {
            w: 2,
            h: 2,
            color: [0.5, 0.5, 0.5, 1.0],
        }));
        let info = g.node_info(src).unwrap();
        let bright = g.add_node(Box::new(FilterNode::point_op(
            src,
            info.clone(),
            Brightness { offset: 0.1 },
        )));
        let inv = g.add_node(Box::new(FilterNode::point_op(bright, info, Invert)));

        let pixels = g.request_full(inv).unwrap();
        // 1.0 - (0.5 + 0.1) = 0.4
        assert!((pixels[0] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn hdr_values_survive_chain() {
        let mut g = Graph::new(0);
        let src = g.add_node(Box::new(SolidSource {
            w: 2,
            h: 2,
            color: [5.0, -0.5, 100.0, 1.0], // HDR: >1.0 and <0.0
        }));
        let info = g.node_info(src).unwrap();
        let bright = g.add_node(Box::new(FilterNode::point_op(
            src,
            info,
            Brightness { offset: 1.0 },
        )));

        let pixels = g.request_full(bright).unwrap();
        assert!((pixels[0] - 6.0).abs() < 1e-6);   // 5.0 + 1.0 (no clamp!)
        assert!((pixels[1] - 0.5).abs() < 1e-6);    // -0.5 + 1.0
        assert!((pixels[2] - 101.0).abs() < 1e-6);  // 100.0 + 1.0
    }

    #[test]
    fn spatial_filter_with_overlap() {
        let mut g = Graph::new(0);
        let src = g.add_node(Box::new(SolidSource {
            w: 16,
            h: 16,
            color: [1.0, 1.0, 1.0, 1.0],
        }));
        let info = g.node_info(src).unwrap();

        let blur_node = FilterNode::spatial(src, info, BoxBlur { radius: 2 }, 2);
        assert!(blur_node.tile_hint().is_some());
        assert_eq!(blur_node.tile_hint().unwrap().tile_overlap, 2);

        let node_id = g.add_node(Box::new(blur_node));
        let pixels = g.request_full(node_id).unwrap();
        // Solid white blurred = still white
        assert!((pixels[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn compose_shader_includes_io_f32() {
        let body = r#"
struct Params { width: u32, height: u32, offset: f32, _pad: u32 }
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params.width;
    let color = load_pixel(idx);
    store_pixel(idx, color + vec4<f32>(params.offset));
}
"#;
        let composed = compose_shader(body);
        assert!(composed.contains("fn load_pixel("));
        assert!(composed.contains("fn store_pixel("));
        assert!(composed.contains("array<vec4<f32>>"));
        assert!(composed.contains("params.offset"));
    }

    #[test]
    fn analytic_op_expression_from_filter() {
        let bright = Brightness { offset: 0.3 };
        let expr = bright.expression();
        assert!((expr.evaluate(0.5) - 0.8).abs() < 1e-6);

        let inv = Invert;
        let inv_expr = inv.expression();
        assert!((inv_expr.evaluate(0.3) - 0.7).abs() < 1e-6);

        // Compose: invert(brightness(v))
        let composed = PointOpExpr::compose(&inv_expr, &expr);
        // 1.0 - (0.5 + 0.3) = 0.2
        assert!((composed.evaluate(0.5) - 0.2).abs() < 1e-6);
    }
}
