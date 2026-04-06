//! rasmcore V2 Native Rust SDK — fluent pipeline API with GPU acceleration.
//!
//! # Example
//!
//! ```ignore
//! use rasmcore_sdk_v2::Pipeline;
//!
//! let output = Pipeline::open(&png_bytes)?
//!     .brightness(0.15)?
//!     .gaussian_blur(3.0)?
//!     .contrast(0.4)?
//!     .write("png", None)?;
//! ```
//!
//! ## GPU Acceleration
//!
//! The SDK auto-initializes a native wgpu GPU executor on first pipeline
//! creation. Filters with GPU shaders run on GPU automatically; CPU fallback
//! is transparent.
//!
//! ## Codegen
//!
//! Typed filter methods are auto-generated at build time from the V2 filter
//! registry. Run `cargo build` to regenerate after adding new filters.

// Force-link filter registrations (including scope filters)
#[allow(unused_imports)]
use rasmcore_pipeline_v2::filters as _v2_filters;
#[allow(unused_imports)]
use rasmcore_codecs_v2 as _v2_codecs;

use std::cell::RefCell;
use std::rc::Rc;

use rasmcore_pipeline_v2 as v2;
use v2::{Graph, NodeInfo, ColorSpace, ParamMap, PipelineError};
use v2::node::{Node, Upstream};
use v2::rect::Rect;

// ─── Source Node ────────────────────────────────────────────────────────────

struct SourceNode {
    pixels: Vec<f32>,
    info: NodeInfo,
}

impl Node for SourceNode {
    fn info(&self) -> NodeInfo { self.info.clone() }
    fn compute(&self, _r: Rect, _u: &mut dyn Upstream) -> Result<Vec<f32>, PipelineError> {
        Ok(self.pixels.clone())
    }
    fn upstream_ids(&self) -> Vec<u32> { vec![] }
}

// ─── Pipeline ──────────────────────────────────────────────────────────────

/// Fluent pipeline builder — wraps a V2 Graph with auto GPU dispatch.
///
/// Each filter method returns a new Pipeline pointing to the new node.
/// The underlying Graph is shared (Rc<RefCell<>>).
pub struct Pipeline {
    graph: Rc<RefCell<Graph>>,
    node: u32,
}

impl Pipeline {
    /// Open an image from encoded bytes (auto-detect format).
    pub fn open(data: &[u8]) -> Result<Self, PipelineError> {
        Self::open_with_hint(data, None)
    }

    /// Open an image from encoded bytes with a format hint.
    pub fn open_with_hint(data: &[u8], hint: Option<&str>) -> Result<Self, PipelineError> {
        let decoded = if let Some(h) = hint {
            rasmcore_codecs_v2::decode_with_hint(data, h)
        } else {
            rasmcore_codecs_v2::decode(data)
        }.map_err(|e| PipelineError::ComputeError(format!("decode: {e}")))?;

        let mut graph = Graph::new(16 * 1024 * 1024);

        // Auto-initialize GPU executor
        if let Ok(executor) = rasmcore_gpu_native::WgpuExecutorV2::try_new() {
            graph.set_gpu_executor(Rc::new(executor));
        }

        let source = SourceNode {
            pixels: decoded.pixels,
            info: NodeInfo {
                width: decoded.info.width,
                height: decoded.info.height,
                color_space: decoded.info.color_space,
            },
        };

        let graph = Rc::new(RefCell::new(graph));
        let node = graph.borrow_mut().add_node(Box::new(source));
        Ok(Pipeline { graph, node })
    }

    /// Create a pipeline from raw f32 RGBA pixels.
    pub fn from_pixels(pixels: Vec<f32>, width: u32, height: u32) -> Self {
        let mut graph = Graph::new(16 * 1024 * 1024);

        if let Ok(executor) = rasmcore_gpu_native::WgpuExecutorV2::try_new() {
            graph.set_gpu_executor(Rc::new(executor));
        }

        let source = SourceNode {
            pixels,
            info: NodeInfo { width, height, color_space: ColorSpace::Linear },
        };

        let graph = Rc::new(RefCell::new(graph));
        let node = graph.borrow_mut().add_node(Box::new(source));
        Pipeline { graph, node }
    }

    /// Get the current node's output dimensions and color space.
    pub fn info(&self) -> NodeInfo {
        self.graph.borrow().node_info(self.node).unwrap_or(NodeInfo {
            width: 0, height: 0, color_space: ColorSpace::Linear,
        })
    }

    /// Apply a named filter with the given parameters.
    ///
    /// This is the generic dispatch method — typed methods (brightness, blur, etc.)
    /// delegate to this.
    pub fn apply(&self, name: &str, params: &ParamMap) -> Result<Pipeline, PipelineError> {
        let info = self.graph.borrow().node_info(self.node)?;
        let node = v2::create_filter_node(name, self.node, info, params)
            .ok_or_else(|| PipelineError::InvalidParams(format!("unknown filter: {name}")))?;
        let id = self.graph.borrow_mut().add_node(node);
        Ok(Pipeline { graph: self.graph.clone(), node: id })
    }

    /// Render the pipeline and return raw f32 RGBA pixels.
    pub fn render(&self) -> Result<Vec<f32>, PipelineError> {
        self.graph.borrow_mut().request_full(self.node)
    }

    /// Render and encode to the specified format.
    pub fn write(&self, format: &str, quality: Option<u8>) -> Result<Vec<u8>, PipelineError> {
        let pixels = self.render()?;
        let info = self.info();
        let mut params = v2::ParamMap::new();
        if let Some(q) = quality {
            params.ints.insert("quality".into(), q as i64);
        }
        if let Some(result) = v2::encode_via_registry(format, &pixels, info.width, info.height, &params) {
            result
        } else {
            rasmcore_codecs_v2::encode(&pixels, info.width, info.height, format, quality)
                .map_err(|e| PipelineError::ComputeError(format!("encode: {e}")))
        }
    }

    /// Mark this node as a named ref for branching.
    pub fn set_ref(&self, name: &str) -> &Self {
        self.graph.borrow_mut().set_ref(name, self.node);
        self
    }

    /// Fork from a named ref — returns a new Pipeline at that node.
    pub fn branch(&self, name: &str) -> Result<Pipeline, PipelineError> {
        let node = self.graph.borrow().get_ref(name)
            .ok_or_else(|| PipelineError::InvalidParams(format!("unknown ref: {name}")))?;
        Ok(Pipeline { graph: self.graph.clone(), node })
    }

    /// Get the underlying node ID (for advanced use).
    pub fn node_id(&self) -> u32 { self.node }
}

// ─── Generated typed filter methods ────────────────────────────────────────
//
// These are convenience methods that construct ParamMap and delegate to apply().
// Generated from the filter registry — one method per registered filter.
//
// The codegen approach: a build-time binary iterates registered_filter_registrations()
// and emits Rust source. For now, the most commonly used filters are hand-written
// here as a bootstrap. The full codegen binary is a follow-up.

impl Pipeline {
    // ── Adjustment ──

    pub fn brightness(&self, amount: f32) -> Result<Pipeline, PipelineError> {
        let mut p = ParamMap::new(); p.floats.insert("amount".into(), amount);
        self.apply("brightness", &p)
    }

    pub fn contrast(&self, amount: f32) -> Result<Pipeline, PipelineError> {
        let mut p = ParamMap::new(); p.floats.insert("amount".into(), amount);
        self.apply("contrast", &p)
    }

    pub fn gamma(&self, gamma: f32) -> Result<Pipeline, PipelineError> {
        let mut p = ParamMap::new(); p.floats.insert("gamma".into(), gamma);
        self.apply("gamma", &p)
    }

    pub fn exposure(&self, ev: f32) -> Result<Pipeline, PipelineError> {
        let mut p = ParamMap::new(); p.floats.insert("ev".into(), ev);
        self.apply("exposure", &p)
    }

    pub fn invert(&self) -> Result<Pipeline, PipelineError> {
        self.apply("invert", &ParamMap::new())
    }

    // ── Spatial ──

    pub fn gaussian_blur(&self, radius: f32) -> Result<Pipeline, PipelineError> {
        let mut p = ParamMap::new(); p.floats.insert("radius".into(), radius);
        self.apply("gaussian_blur", &p)
    }

    pub fn sharpen(&self, amount: f32) -> Result<Pipeline, PipelineError> {
        let mut p = ParamMap::new(); p.floats.insert("amount".into(), amount);
        self.apply("sharpen", &p)
    }

    pub fn box_blur(&self, radius: u32) -> Result<Pipeline, PipelineError> {
        let mut p = ParamMap::new(); p.ints.insert("radius".into(), radius as i64);
        self.apply("box_blur", &p)
    }

    // ── Color ──

    pub fn hue_rotate(&self, degrees: f32) -> Result<Pipeline, PipelineError> {
        let mut p = ParamMap::new(); p.floats.insert("degrees".into(), degrees);
        self.apply("hue_rotate", &p)
    }

    pub fn saturate(&self, amount: f32) -> Result<Pipeline, PipelineError> {
        let mut p = ParamMap::new(); p.floats.insert("amount".into(), amount);
        self.apply("saturate", &p)
    }

    pub fn sepia(&self, amount: f32) -> Result<Pipeline, PipelineError> {
        let mut p = ParamMap::new(); p.floats.insert("amount".into(), amount);
        self.apply("sepia", &p)
    }

    // ── Edge detection ──

    pub fn sobel(&self, scale: f32) -> Result<Pipeline, PipelineError> {
        let mut p = ParamMap::new(); p.floats.insert("scale".into(), scale);
        self.apply("sobel", &p)
    }

    pub fn canny(&self, low: f32, high: f32) -> Result<Pipeline, PipelineError> {
        let mut p = ParamMap::new();
        p.floats.insert("low".into(), low);
        p.floats.insert("high".into(), high);
        self.apply("canny", &p)
    }

    // ── Scopes ──

    pub fn scope_histogram(&self) -> Result<Pipeline, PipelineError> {
        self.apply("scope_histogram", &ParamMap::new())
    }

    pub fn scope_waveform(&self) -> Result<Pipeline, PipelineError> {
        self.apply("scope_waveform", &ParamMap::new())
    }

    pub fn scope_vectorscope(&self) -> Result<Pipeline, PipelineError> {
        self.apply("scope_vectorscope", &ParamMap::new())
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_pixels(w: u32, h: u32) -> Vec<f32> {
        let n = (w * h) as usize;
        let mut px = Vec::with_capacity(n * 4);
        for i in 0..n {
            let v = i as f32 / n as f32;
            px.extend_from_slice(&[v, v, v, 1.0]);
        }
        px
    }

    #[test]
    fn pipeline_from_pixels() {
        let px = test_pixels(4, 4);
        let pipe = Pipeline::from_pixels(px.clone(), 4, 4);
        let info = pipe.info();
        assert_eq!(info.width, 4);
        assert_eq!(info.height, 4);
        let rendered = pipe.render().unwrap();
        assert_eq!(rendered, px);
    }

    #[test]
    fn pipeline_brightness() {
        let px = test_pixels(4, 4);
        let pipe = Pipeline::from_pixels(px, 4, 4)
            .brightness(0.1).unwrap();
        let out = pipe.render().unwrap();
        // First pixel should be brighter
        assert!(out[0] > 0.0, "brightness should increase value");
    }

    #[test]
    fn pipeline_chain() {
        let px = test_pixels(4, 4);
        let pipe = Pipeline::from_pixels(px, 4, 4)
            .brightness(0.1).unwrap()
            .contrast(0.3).unwrap()
            .invert().unwrap();
        let out = pipe.render().unwrap();
        assert_eq!(out.len(), 4 * 4 * 4);
    }

    #[test]
    fn pipeline_ref_and_branch() {
        let px = test_pixels(4, 4);
        let pipe = Pipeline::from_pixels(px, 4, 4)
            .brightness(0.1).unwrap();
        pipe.set_ref("graded");
        let scope = pipe.branch("graded").unwrap()
            .scope_histogram().unwrap();
        let scope_info = scope.info();
        assert_eq!(scope_info.width, 512);
        assert_eq!(scope_info.height, 512);
    }

    #[test]
    fn pipeline_apply_generic() {
        let px = test_pixels(4, 4);
        let pipe = Pipeline::from_pixels(px, 4, 4);
        let mut params = ParamMap::new();
        params.floats.insert("amount".into(), 0.2);
        let pipe2 = pipe.apply("brightness", &params).unwrap();
        let out = pipe2.render().unwrap();
        assert_eq!(out.len(), 4 * 4 * 4);
    }

    #[test]
    fn pipeline_unknown_filter_errors() {
        let px = test_pixels(4, 4);
        let pipe = Pipeline::from_pixels(px, 4, 4);
        let result = pipe.apply("nonexistent_filter", &ParamMap::new());
        assert!(result.is_err());
    }
}
