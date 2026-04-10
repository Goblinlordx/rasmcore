//! CompositorNode — bridges the Compositor trait to the pipeline graph's Node trait.
//!
//! Wraps a dual-input `Compositor` into a pipeline `Node` with two upstream
//! connections. The pipeline calls `Node::compute()` which fetches both upstream
//! buffers and delegates to `Compositor::compute()`.

use crate::node::{
    GpuShader, InputRectEstimate, Node, NodeCapabilities, NodeInfo, PipelineError, Upstream,
};
use crate::ops::Compositor;
use crate::rect::Rect;

/// IO_F32_DUAL fragment — declares dual-input f32 bindings for compositor GPU shaders.
///
/// Provides `load_pixel_a(idx)`, `load_pixel_b(idx)`, and `store_pixel(idx, color)`.
/// Auto-composed with every compositor GPU shader body.
pub const IO_F32_DUAL: &str = r#"
@group(0) @binding(0) var<storage, read> input_a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> input_b: array<vec4<f32>>;

fn load_pixel_a(idx: u32) -> vec4<f32> {
  return input_a[idx];
}

fn load_pixel_b(idx: u32) -> vec4<f32> {
  return input_b[idx];
}

fn store_pixel(idx: u32, color: vec4<f32>) {
  output[idx] = color;
}
"#;

/// Compose a complete WGSL shader from a compositor's shader body.
///
/// Prepends the io_f32_dual bindings. The compositor body should declare only
/// `@group(0) @binding(2) var<uniform> params: Params` and the entry point.
pub fn compose_dual_shader(body: &str) -> String {
    let mut src = String::with_capacity(IO_F32_DUAL.len() + body.len() + 1);
    src.push_str(IO_F32_DUAL);
    src.push('\n');
    src.push_str(body);
    src
}

/// Wraps a `Compositor` into a pipeline `Node` with two upstream connections.
///
/// `upstream_a` is the foreground (top layer), `upstream_b` is the background (bottom layer).
/// Both must have the same dimensions — the node errors at compute time if they differ.
pub struct CompositorNode<C: Compositor> {
    upstream_a: u32,
    upstream_b: u32,
    info: NodeInfo,
    compositor: C,
}

impl<C: Compositor> CompositorNode<C> {
    /// Create a CompositorNode with two upstream connections.
    pub fn new(upstream_a: u32, upstream_b: u32, info: NodeInfo, compositor: C) -> Self {
        Self {
            upstream_a,
            upstream_b,
            info,
            compositor,
        }
    }

    /// Get a reference to the underlying compositor.
    pub fn compositor(&self) -> &C {
        &self.compositor
    }
}

impl<C: Compositor + 'static> Node for CompositorNode<C> {
    fn info(&self) -> NodeInfo {
        self.info.clone()
    }

    fn compute(
        &self,
        request: Rect,
        upstream: &mut dyn Upstream,
    ) -> Result<Vec<f32>, PipelineError> {
        let clamped = request.clamp(self.info.width, self.info.height);
        let fg = upstream.request(self.upstream_a, clamped)?;
        let bg = upstream.request(self.upstream_b, clamped)?;

        let info = NodeInfo {
            width: clamped.width,
            height: clamped.height,
            color_space: self.info.color_space,
        };
        self.compositor.compute_with_info(&fg, &bg, &info)
    }

    fn gpu_shader(&self, width: u32, height: u32) -> Option<GpuShader> {
        let body = self.compositor.gpu_shader_body()?;
        let info = NodeInfo {
            width,
            height,
            color_space: self.info.color_space,
        };
        let params = self.compositor.gpu_params_with_info(&info)?;
        Some(GpuShader {
            body: compose_dual_shader(body),
            entry_point: self.compositor.gpu_entry_point(),
            workgroup_size: self.compositor.gpu_workgroup_size(),
            params,
            extra_buffers: self.compositor.gpu_extra_buffers(),
            reduction_buffers: vec![],
            convergence_check: None,
            loop_dispatch: None,
            setup: None,
        })
    }

    fn upstream_ids(&self) -> Vec<u32> {
        vec![self.upstream_a, self.upstream_b]
    }

    fn set_upstream(&mut self, new_upstream: u32) -> bool {
        // Rewire primary (foreground) upstream
        self.upstream_a = new_upstream;
        true
    }

    fn capabilities(&self) -> NodeCapabilities {
        NodeCapabilities {
            gpu: self.compositor.gpu_shader_body().is_some(),
            analytic: false,
            affine: false,
            clut: false,
        }
    }

    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> InputRectEstimate {
        InputRectEstimate::Exact(output.clamp(bounds_w, bounds_h))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color_space::ColorSpace;
    use crate::graph::Graph;

    // ─── Test compositor: weighted average of fg and bg ──────────────────

    struct WeightedMix {
        weight: f32,
    }

    impl Compositor for WeightedMix {
        fn compute(
            &self,
            fg: &[f32],
            bg: &[f32],
            _w: u32,
            _h: u32,
        ) -> Result<Vec<f32>, PipelineError> {
            let mut out = Vec::with_capacity(fg.len());
            for (a, b) in fg.iter().zip(bg.iter()) {
                out.push(a * self.weight + b * (1.0 - self.weight));
            }
            Ok(out)
        }
    }

    // ─── Source nodes for tests ─────────────────────────────────────────

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

    // ─── Tests ──────────────────────────────────────────────────────────

    #[test]
    fn compositor_node_two_sources() {
        let mut g = Graph::new(0);
        let src_a = g.add_node(Box::new(SolidSource {
            w: 4,
            h: 4,
            color: [1.0, 0.0, 0.0, 1.0], // red
        }));
        let src_b = g.add_node(Box::new(SolidSource {
            w: 4,
            h: 4,
            color: [0.0, 0.0, 1.0, 1.0], // blue
        }));
        let info = g.node_info(src_a).unwrap();
        let comp = CompositorNode::new(src_a, src_b, info, WeightedMix { weight: 0.5 });
        let node_id = g.add_node(Box::new(comp));

        let pixels = g.request_full(node_id).unwrap();
        // 50% red + 50% blue = (0.5, 0.0, 0.5, 1.0)
        assert!((pixels[0] - 0.5).abs() < 1e-6, "R: {}", pixels[0]);
        assert!(pixels[1].abs() < 1e-6, "G: {}", pixels[1]);
        assert!((pixels[2] - 0.5).abs() < 1e-6, "B: {}", pixels[2]);
        assert!((pixels[3] - 1.0).abs() < 1e-6, "A: {}", pixels[3]);
    }

    #[test]
    fn compositor_weight_zero_is_bg() {
        let mut g = Graph::new(0);
        let src_a = g.add_node(Box::new(SolidSource {
            w: 2,
            h: 2,
            color: [1.0, 1.0, 1.0, 1.0],
        }));
        let src_b = g.add_node(Box::new(SolidSource {
            w: 2,
            h: 2,
            color: [0.3, 0.5, 0.7, 1.0],
        }));
        let info = g.node_info(src_a).unwrap();
        let comp = CompositorNode::new(src_a, src_b, info, WeightedMix { weight: 0.0 });
        let node_id = g.add_node(Box::new(comp));

        let pixels = g.request_full(node_id).unwrap();
        assert!((pixels[0] - 0.3).abs() < 1e-6);
        assert!((pixels[1] - 0.5).abs() < 1e-6);
        assert!((pixels[2] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn compositor_weight_one_is_fg() {
        let mut g = Graph::new(0);
        let src_a = g.add_node(Box::new(SolidSource {
            w: 2,
            h: 2,
            color: [0.3, 0.5, 0.7, 1.0],
        }));
        let src_b = g.add_node(Box::new(SolidSource {
            w: 2,
            h: 2,
            color: [1.0, 1.0, 1.0, 1.0],
        }));
        let info = g.node_info(src_a).unwrap();
        let comp = CompositorNode::new(src_a, src_b, info, WeightedMix { weight: 1.0 });
        let node_id = g.add_node(Box::new(comp));

        let pixels = g.request_full(node_id).unwrap();
        assert!((pixels[0] - 0.3).abs() < 1e-6);
        assert!((pixels[1] - 0.5).abs() < 1e-6);
        assert!((pixels[2] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn compositor_upstream_ids() {
        let comp = CompositorNode::new(
            5,
            10,
            NodeInfo {
                width: 1,
                height: 1,
                color_space: ColorSpace::Linear,
            },
            WeightedMix { weight: 0.5 },
        );
        assert_eq!(comp.upstream_ids(), vec![5, 10]);
    }

    #[test]
    fn compose_dual_shader_includes_bindings() {
        let body = r#"
struct Params { width: u32, height: u32 }
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params.width;
    let a = load_pixel_a(idx);
    let b = load_pixel_b(idx);
    store_pixel(idx, mix(a, b, 0.5));
}
"#;
        let composed = compose_dual_shader(body);
        assert!(composed.contains("fn load_pixel_a("));
        assert!(composed.contains("fn load_pixel_b("));
        assert!(composed.contains("fn store_pixel("));
        assert!(composed.contains("input_a: array<vec4<f32>>"));
        assert!(composed.contains("input_b: array<vec4<f32>>"));
    }
}
