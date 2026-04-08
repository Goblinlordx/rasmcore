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
use v2::gpu_shaders::pixel_source;
use v2::hash::source_hash_pixels;

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

// ─── Host-Decoded Pixel Source Node ────────────────────────────────────────

struct SourceNodePixels {
    raw_bytes: Vec<u8>,
    format: HostPixelFormat,
    info: NodeInfo,
}

impl Node for SourceNodePixels {
    fn info(&self) -> NodeInfo { self.info.clone() }
    fn compute(&self, _r: Rect, _u: &mut dyn Upstream) -> Result<Vec<f32>, PipelineError> {
        match self.format {
            HostPixelFormat::Rgba8 => Ok(v2::color_math::srgb_rgba8_to_f32_linear(&self.raw_bytes)),
            HostPixelFormat::Rgba16 => {
                let mut out = Vec::with_capacity((self.info.width * self.info.height * 4) as usize);
                for chunk in self.raw_bytes.chunks_exact(8) {
                    out.push(u16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 65535.0);
                    out.push(u16::from_le_bytes([chunk[2], chunk[3]]) as f32 / 65535.0);
                    out.push(u16::from_le_bytes([chunk[4], chunk[5]]) as f32 / 65535.0);
                    out.push(u16::from_le_bytes([chunk[6], chunk[7]]) as f32 / 65535.0);
                }
                Ok(out)
            }
            HostPixelFormat::RgbaF32 => {
                let mut out = Vec::with_capacity(self.raw_bytes.len() / 4);
                for chunk in self.raw_bytes.chunks_exact(4) {
                    out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                Ok(out)
            }
        }
    }
    fn gpu_shader(&self, width: u32, height: u32) -> Option<v2::GpuShader> {
        pixel_source::conversion_shader(self.format, &self.raw_bytes, width, height)
    }
    fn upstream_ids(&self) -> Vec<u32> { vec![] }
}

// Re-export HostPixelFormat for SDK consumers
pub use v2::gpu_shaders::pixel_source::HostPixelFormat;

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
        let mut node = graph.borrow_mut().add_node(Box::new(source));

        // Auto-IDT: convert source color space to working space when color-managed
        let source_cs = decoded.info.color_space;
        if graph.borrow().is_color_managed() {
            let working = graph.borrow().working_color_space()
                .unwrap_or(ColorSpace::AcesCg);
            if source_cs != working && source_cs != ColorSpace::Unknown {
                let idt_name = match source_cs {
                    ColorSpace::Srgb => "idt-srgb",
                    ColorSpace::Rec709 | ColorSpace::Linear => "idt-rec709",
                    ColorSpace::Rec2020 => "idt-rec2020",
                    ColorSpace::DisplayP3 => "idt-p3",
                    _ => "idt-srgb", // safe default for unrecognized spaces
                };
                if let Ok(transform) = v2::color_transform::load_preset(idt_name) {
                    let idt_info = NodeInfo {
                        width: decoded.info.width,
                        height: decoded.info.height,
                        color_space: working,
                    };
                    let ct_node = v2::color_transform::ColorTransformNode::new(
                        node, idt_info, transform,
                    );
                    node = graph.borrow_mut().add_node(Box::new(ct_node));
                }
            }
        }

        Ok(Pipeline { graph, node })
    }

    /// Create a pipeline from host-decoded raw pixel bytes.
    ///
    /// Accepts u8 sRGB, u16 linear, or f32 linear pixels. For non-f32 formats,
    /// a GPU conversion shader is prepended to the pipeline chain.
    pub fn open_pixels(data: Vec<u8>, width: u32, height: u32, format: HostPixelFormat) -> Self {
        let mut graph = Graph::new(16 * 1024 * 1024);

        if let Ok(executor) = rasmcore_gpu_native::WgpuExecutorV2::try_new() {
            graph.set_gpu_executor(Rc::new(executor));
        }

        let format_tag = match format {
            HostPixelFormat::Rgba8 => "source-rgba8",
            HostPixelFormat::Rgba16 => "source-rgba16",
            HostPixelFormat::RgbaF32 => "source-f32",
        };
        let src_hash = source_hash_pixels(format_tag, &data, width, height);

        let color_space = match format {
            HostPixelFormat::Rgba8 => ColorSpace::Srgb,
            HostPixelFormat::Rgba16 | HostPixelFormat::RgbaF32 => ColorSpace::Linear,
        };

        let node: Box<dyn Node> = if format == HostPixelFormat::RgbaF32 {
            let mut pixels = Vec::with_capacity((width * height * 4) as usize);
            for chunk in data.chunks_exact(4) {
                pixels.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Box::new(SourceNode {
                pixels,
                info: NodeInfo { width, height, color_space },
            })
        } else {
            Box::new(SourceNodePixels {
                raw_bytes: data,
                format,
                info: NodeInfo { width, height, color_space },
            })
        };

        let graph = Rc::new(RefCell::new(graph));
        let id = graph.borrow_mut().add_node_with_hash(node, src_hash);
        Pipeline { graph, node: id }
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

    /// Render the pipeline and return working-space f32 RGBA pixels.
    /// No OT is applied — useful for compositing or further processing.
    pub fn render(&self) -> Result<Vec<f32>, PipelineError> {
        self.graph.borrow_mut().request_full(self.node)
    }

    /// Render with the display OT applied, returning display-referred f32 RGBA.
    /// Uses the configured display transform (default: "ot-srgb").
    /// Returns working-space pixels if color management is disabled.
    pub fn render_display(&self) -> Result<Vec<f32>, PipelineError> {
        if !self.graph.borrow().is_color_managed() {
            return self.render();
        }
        let ot_name = self.graph.borrow().display_transform().to_string();
        if let Ok(transform) = v2::color_transform::load_preset(&ot_name) {
            let info = self.info();
            let ot_node = v2::color_transform::ColorTransformNode::new(
                self.node, info, transform,
            );
            let ot_id = self.graph.borrow_mut().add_node(Box::new(ot_node));
            self.graph.borrow_mut().request_full(ot_id)
        } else {
            self.render()
        }
    }

    /// Render and encode to the specified format.
    /// For display-referred formats (png, jpeg, webp, gif, bmp), auto-applies
    /// the display OT when color-managed. For scene-referred formats (exr, tiff,
    /// hdr), outputs working-space pixels directly.
    pub fn write(&self, format: &str, quality: Option<u8>) -> Result<Vec<u8>, PipelineError> {
        let scene_referred = v2::registry::is_scene_referred_format(format);
        let pixels = if self.graph.borrow().is_color_managed() && !scene_referred {
            self.render_display()?
        } else {
            self.render()?
        };
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

    /// Set the display output transform preset for auto-OT insertion.
    /// Default: "ot-srgb". See docs/guides/color-management.md for presets.
    pub fn set_display_transform(&self, preset: &str) -> &Self {
        self.graph.borrow_mut().set_display_transform(preset);
        self
    }

    /// Enable or disable automatic IDT/OT insertion.
    pub fn set_color_managed(&self, enabled: bool) -> &Self {
        self.graph.borrow_mut().set_color_managed(enabled);
        self
    }
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

    #[test]
    fn open_pixels_rgba8_cpu_conversion() {
        // Known sRGB u8 value: 128 (0.502) -> sRGB EOTF -> ~0.2158 linear
        let u8_pixels: Vec<u8> = vec![128, 128, 128, 255, 0, 0, 0, 255];
        let pipe = Pipeline::open_pixels(u8_pixels, 2, 1, HostPixelFormat::Rgba8);
        let info = pipe.info();
        assert_eq!(info.width, 2);
        assert_eq!(info.height, 1);
        let rendered = pipe.render().unwrap();
        assert_eq!(rendered.len(), 2 * 1 * 4);
        // sRGB 128/255 = 0.502 -> linear ~0.2158
        let expected = v2::color_math::srgb_to_linear(128.0 / 255.0);
        assert!((rendered[0] - expected).abs() < 1e-5, "r: {} vs {}", rendered[0], expected);
        assert!((rendered[1] - expected).abs() < 1e-5, "g: {} vs {}", rendered[1], expected);
        assert!((rendered[2] - expected).abs() < 1e-5, "b: {} vs {}", rendered[2], expected);
        assert!((rendered[3] - 1.0).abs() < 1e-5, "a should be 1.0");
        // Second pixel: sRGB 0 -> linear 0
        assert!((rendered[4]).abs() < 1e-5, "black r should be 0");
    }

    #[test]
    fn open_pixels_rgba16_normalization() {
        // u16 value 32768 = 0.5 (already linear)
        let mut data = Vec::new();
        data.extend_from_slice(&32768u16.to_le_bytes()); // R
        data.extend_from_slice(&32768u16.to_le_bytes()); // G
        data.extend_from_slice(&32768u16.to_le_bytes()); // B
        data.extend_from_slice(&65535u16.to_le_bytes()); // A
        let pipe = Pipeline::open_pixels(data, 1, 1, HostPixelFormat::Rgba16);
        let rendered = pipe.render().unwrap();
        assert_eq!(rendered.len(), 4);
        let expected = 32768.0 / 65535.0;
        assert!((rendered[0] - expected).abs() < 1e-5, "r: {} vs {}", rendered[0], expected);
        assert!((rendered[3] - 1.0).abs() < 1e-5, "a should be 1.0");
    }

    #[test]
    fn open_pixels_f32_passthrough() {
        // f32 pixels passed through unchanged
        let f32_vals: Vec<f32> = vec![0.5, 0.3, 0.1, 1.0];
        let bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let pipe = Pipeline::open_pixels(bytes, 1, 1, HostPixelFormat::RgbaF32);
        let rendered = pipe.render().unwrap();
        assert_eq!(rendered.len(), 4);
        assert!((rendered[0] - 0.5).abs() < 1e-6);
        assert!((rendered[1] - 0.3).abs() < 1e-6);
        assert!((rendered[2] - 0.1).abs() < 1e-6);
        assert!((rendered[3] - 1.0).abs() < 1e-6);
    }
}
