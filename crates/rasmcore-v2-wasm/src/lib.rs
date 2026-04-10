//! V2 WASM Component — f32-native pipeline, zero V1.
//!
//! Implements the V2 WIT pipeline interface using exclusively:
//! - rasmcore-pipeline-v2 (Graph, Filter, create_filter_node, fusion)
//! - rasmcore-codecs-v2 (decode, encode)
//!
//! No V1 NodeGraph, no V1 ImageNode, no u8 LUTs, no PixelFormat dispatch.

// Force linker to include modules with inventory registrations.
// Without these, the linker drops unused modules and their inventory::submit! entries.
#[allow(unused_imports)]
use rasmcore_pipeline_v2::filters as _v2_filters;

// Scope filters use manual inventory::submit! (not the V2Filter macro) because they
// need a custom ScopeNode. The WASM linker strips these statics unless referenced.
#[used]
static _SCOPE_LINK: fn() = rasmcore_pipeline_v2::filters::scope::ensure_linked;
#[allow(unused_imports)]
use rasmcore_codecs_v2 as _v2_codecs;

#[cfg(target_arch = "wasm32")]
mod bindings;

#[cfg(target_arch = "wasm32")]
use bindings::exports::rasmcore::v2_image::pipeline_v2 as wit;

#[cfg(target_arch = "wasm32")]
use bindings::rasmcore::core::errors::RasmcoreError;

use std::cell::RefCell;
use std::rc::Rc;

use rasmcore_pipeline_v2::{
    self as v2, Graph, LayerCache, NodeInfo, ParamMap, PipelineError,
    create_filter_node,
    hash::{content_hash, source_hash, source_hash_pixels},
};
use rasmcore_pipeline_v2::ColorSpace;
use rasmcore_pipeline_v2::gpu_shaders::pixel_source::{self, HostPixelFormat};

// ─── Output Color Space ──────────────────────────────────────────────────────

/// Determine the preferred output color space for a format.
///
/// Each image format has a natural color space. SDR display formats (PNG, JPEG,
/// WebP) expect sRGB gamma-encoded pixels. HDR/scene-referred formats (EXR, HDR,
/// FITS) expect linear. Fallback is sRGB.
fn output_cs_for_format(format: &str) -> ColorSpace {
    match format {
        "exr" | "hdr" | "fits" => ColorSpace::Linear,
        _ => ColorSpace::Srgb, // PNG, JPEG, WebP, BMP, GIF, TIFF, AVIF, etc.
    }
}

/// Apply the output transform: convert working-space pixels to the format's
/// preferred color space. No-op if already in the target space.
fn apply_output_transform(pixels: &mut [f32], info: &NodeInfo, format: &str) {
    let target_cs = output_cs_for_format(format);
    let source_cs = info.color_space;
    if source_cs != target_cs {
        v2::color_math::convert_color_space(pixels, source_cs, target_cs);
    }
}

// ─── Instance-Level Resource Registry ────────────────────────────────────────

thread_local! {
    static RESOURCE_FONTS: RefCell<std::collections::HashMap<u32, std::rc::Rc<v2::font::Font>>> = RefCell::new(std::collections::HashMap::new());
    static RESOURCE_LUTS: RefCell<std::collections::HashMap<u32, v2::lmt::Lmt>> = RefCell::new(std::collections::HashMap::new());
    static RESOURCE_TRANSFORMS: RefCell<std::collections::HashMap<u32, v2::color_transform::ColorTransform>> = RefCell::new(std::collections::HashMap::new());
    static NEXT_RESOURCE_ID: std::cell::Cell<u32> = const { std::cell::Cell::new(1) };
}

fn instance_register_font(data: &[u8]) -> Result<u32, PipelineError> {
    let font = v2::font::Font::from_bytes(data)
        .map_err(|e| PipelineError::ComputeError(e))?;
    let id = NEXT_RESOURCE_ID.with(|c| { let id = c.get(); c.set(id + 1); id });
    RESOURCE_FONTS.with(|m| m.borrow_mut().insert(id, std::rc::Rc::new(font)));
    Ok(id)
}

fn instance_register_lut(data: &[u8]) -> Result<u32, PipelineError> {
    let text = std::str::from_utf8(data).map_err(|_|
        PipelineError::InvalidParams("LUT data is not valid UTF-8".into())
    )?;
    let lmt = if text.contains("LUT_3D_SIZE") || text.contains("LUT_1D_SIZE") {
        v2::parse_cube(text)?
    } else if text.trim_start().starts_with("<?xml") || text.contains("<ProcessList") || text.contains("<CLF") {
        v2::lmt::parse_clf(text)?
    } else {
        return Err(PipelineError::InvalidParams("unsupported LUT format".into()));
    };
    let id = NEXT_RESOURCE_ID.with(|c| { let id = c.get(); c.set(id + 1); id });
    RESOURCE_LUTS.with(|m| m.borrow_mut().insert(id, lmt));
    Ok(id)
}

#[allow(dead_code)]
fn instance_get_font(id: u32) -> Option<std::rc::Rc<v2::font::Font>> {
    RESOURCE_FONTS.with(|m| m.borrow().get(&id).cloned())
}

#[allow(dead_code)]
fn instance_get_lut(id: u32) -> Option<v2::lmt::Lmt> {
    RESOURCE_LUTS.with(|m| m.borrow().get(&id).cloned())
}

fn instance_register_transform(data: &[u8], format_hint: Option<&str>) -> Result<u32, PipelineError> {
    let transform = v2::color_transform::parse_transform(data, format_hint)?;
    let id = NEXT_RESOURCE_ID.with(|c| { let id = c.get(); c.set(id + 1); id });
    RESOURCE_TRANSFORMS.with(|m| m.borrow_mut().insert(id, transform));
    Ok(id)
}

fn instance_register_transform_preset(name: &str) -> Result<u32, PipelineError> {
    let transform = v2::color_transform::load_preset(name)?;
    let id = NEXT_RESOURCE_ID.with(|c| { let id = c.get(); c.set(id + 1); id });
    RESOURCE_TRANSFORMS.with(|m| m.borrow_mut().insert(id, transform));
    Ok(id)
}

#[allow(dead_code)]
fn instance_get_transform(id: u32) -> Option<v2::color_transform::ColorTransform> {
    RESOURCE_TRANSFORMS.with(|m| m.borrow().get(&id).cloned())
}

// ─── Error conversion ───────────────────────────────────────────────────────

#[cfg(target_arch = "wasm32")]
fn to_wit_error(e: PipelineError) -> RasmcoreError {
    match e {
        PipelineError::NodeNotFound(id) => {
            RasmcoreError::InvalidInput(format!("invalid node id: {id}"))
        }
        PipelineError::ComputeError(msg) => RasmcoreError::CodecError(msg),
        PipelineError::InvalidParams(msg) => RasmcoreError::InvalidInput(msg),
        PipelineError::GpuError(msg) => RasmcoreError::CodecError(format!("GPU: {msg}")),
        PipelineError::BufferMismatch { expected, actual } => {
            RasmcoreError::InvalidInput(format!(
                "buffer size mismatch: expected {expected}, got {actual}"
            ))
        }
    }
}

/// Convert a Rust DynamicsCurve to WIT flat f32 list (pairs of x,y).
/// Empty curve → None (identity). Non-empty → Some(vec![x0,y0, x1,y1, ...]).
#[cfg(target_arch = "wasm32")]
fn curve_to_wit(curve: &v2::brush::DynamicsCurve) -> Option<Vec<f32>> {
    if curve.points.is_empty() {
        None
    } else {
        Some(curve.points.iter().flat_map(|&(x, y)| [x, y]).collect())
    }
}

// ─── V2 Source Node ─────────────────────────────────────────────────────────

/// Source node that holds decoded f32 pixel data.
struct SourceNode {
    pixels: Vec<f32>,
    info: NodeInfo,
}

impl v2::Node for SourceNode {
    fn info(&self) -> NodeInfo {
        self.info.clone()
    }

    fn compute(
        &self,
        _request: v2::Rect,
        _upstream: &mut dyn v2::Upstream,
    ) -> Result<Vec<f32>, PipelineError> {
        Ok(self.pixels.clone())
    }

    fn upstream_ids(&self) -> Vec<u32> {
        vec![]
    }
}

// ─── Host-Decoded Pixel Source Node ─────────────────────────────────────────

/// Source node for host-decoded raw pixel bytes.
///
/// Stores raw bytes in the host format (u8 sRGB, u16 linear, or f32 linear).
/// For non-f32 formats, provides a GPU shader that converts to f32 linear as
/// the first node in the GPU chain. CPU fallback converts in-place.
struct SourceNodePixels {
    raw_bytes: Vec<u8>,
    format: HostPixelFormat,
    info: NodeInfo,
}

impl v2::Node for SourceNodePixels {
    fn info(&self) -> NodeInfo {
        self.info.clone()
    }

    fn compute(
        &self,
        _request: v2::Rect,
        _upstream: &mut dyn v2::Upstream,
    ) -> Result<Vec<f32>, PipelineError> {
        match self.format {
            HostPixelFormat::Rgba8 => {
                Ok(v2::color_math::srgb_rgba8_to_f32_linear(&self.raw_bytes))
            }
            HostPixelFormat::Rgba16 => {
                let pixel_count = (self.info.width * self.info.height) as usize;
                let mut out = Vec::with_capacity(pixel_count * 4);
                for chunk in self.raw_bytes.chunks_exact(8) {
                    let r = u16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 65535.0;
                    let g = u16::from_le_bytes([chunk[2], chunk[3]]) as f32 / 65535.0;
                    let b = u16::from_le_bytes([chunk[4], chunk[5]]) as f32 / 65535.0;
                    let a = u16::from_le_bytes([chunk[6], chunk[7]]) as f32 / 65535.0;
                    out.extend_from_slice(&[r, g, b, a]);
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

    fn upstream_ids(&self) -> Vec<u32> {
        vec![]
    }
}

// ─── Source Resource ────────────────────────────────────────────────────────

/// Decoded image source — created once, reused across pipeline chains.
/// Holds decoded pixel data and a shared buffer pool for zero-alloc
/// rendering. Passing the same source to read_source() skips re-decoding.
pub struct SourceResource {
    pixels: Vec<f32>,
    info: NodeInfo,
    /// Shared buffer pool — reusable pixel buffers for this image's dimensions.
    /// Persists across pipeline lifetimes, eliminating allocation churn.
    buffer_pool: Rc<RefCell<v2::BufferPool>>,
    /// Raw host-decoded bytes (non-empty for from_pixels sources).
    #[allow(dead_code)]
    raw_pixels: Option<(Vec<u8>, HostPixelFormat)>,
    /// Content hash for this source (precomputed).
    #[allow(dead_code)]
    src_hash: v2::hash::ContentHash,
}

impl SourceResource {
    pub fn new(data: &[u8], format_hint: Option<&str>) -> Result<Self, PipelineError> {
        let decoded = if let Some(hint) = format_hint {
            if let Some(result) = v2::decode_with_hint_via_registry(data, hint) {
                let d = result?;
                (d.pixels, d.width, d.height, d.color_space)
            } else {
                let d = rasmcore_codecs_v2::decode_with_hint(data, hint)
                    .map_err(|e| PipelineError::ComputeError(format!("decode: {e}")))?;
                (d.pixels, d.info.width, d.info.height, d.info.color_space)
            }
        } else if let Some(result) = v2::decode_via_registry(data) {
            let d = result?;
            (d.pixels, d.width, d.height, d.color_space)
        } else {
            let d = rasmcore_codecs_v2::decode(data)
                .map_err(|e| PipelineError::ComputeError(format!("decode: {e}")))?;
            (d.pixels, d.info.width, d.info.height, d.info.color_space)
        };

        let src_hash = source_hash(data);
        Ok(Self {
            pixels: decoded.0,
            info: NodeInfo {
                width: decoded.1,
                height: decoded.2,
                color_space: decoded.3,
            },
            buffer_pool: Rc::new(RefCell::new(v2::BufferPool::new())),
            raw_pixels: None,
            src_hash,
        })
    }

    /// Create a source from host-decoded raw pixel bytes.
    pub fn from_pixels(data: Vec<u8>, width: u32, height: u32, format: HostPixelFormat) -> Self {
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

        // Eagerly convert to f32 for the read_source() path.
        // The read_pixels() path uses SourceNodePixels directly for GPU conversion.
        let pixels = match format {
            HostPixelFormat::Rgba8 => {
                v2::color_math::srgb_rgba8_to_f32_linear(&data)
            }
            HostPixelFormat::Rgba16 => {
                let mut out = Vec::with_capacity((width * height * 4) as usize);
                for chunk in data.chunks_exact(8) {
                    let r = u16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 65535.0;
                    let g = u16::from_le_bytes([chunk[2], chunk[3]]) as f32 / 65535.0;
                    let b = u16::from_le_bytes([chunk[4], chunk[5]]) as f32 / 65535.0;
                    let a = u16::from_le_bytes([chunk[6], chunk[7]]) as f32 / 65535.0;
                    out.extend_from_slice(&[r, g, b, a]);
                }
                out
            }
            HostPixelFormat::RgbaF32 => {
                let mut out = Vec::with_capacity((width * height * 4) as usize);
                for chunk in data.chunks_exact(4) {
                    out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                out
            }
        };

        Self {
            pixels,
            info: NodeInfo { width, height, color_space },
            buffer_pool: Rc::new(RefCell::new(v2::BufferPool::new())),
            raw_pixels: Some((data, format)),
            src_hash,
        }
    }

    pub fn info(&self) -> NodeInfo {
        self.info.clone()
    }
}

// ─── Pipeline Resource ──────────────────────────────────────────────────────

/// V2 pipeline resource — wraps a V2 Graph exclusively.
pub struct PipelineResource {
    graph: RefCell<Graph>,
    /// Shared layer cache (if set). Injected from outside, persists across pipelines.
    layer_cache: RefCell<Option<Rc<RefCell<LayerCache>>>>,
    /// Proxy scale factor for spatial param auto-scaling.
    /// Default 1.0 = full resolution. Values < 1.0 indicate proxy resolution;
    /// spatial params (hint = rc.pixels) are multiplied by this factor.
    proxy_scale: std::cell::Cell<f32>,
    fonts: RefCell<std::collections::HashMap<u32, std::rc::Rc<v2::font::Font>>>,
    luts: RefCell<std::collections::HashMap<u32, v2::lmt::Lmt>>,
    next_resource_id: std::cell::Cell<u32>,
}

impl PipelineResource {
    pub fn new() -> Self {
        Self {
            graph: RefCell::new(Graph::new(16 * 1024 * 1024)),
            layer_cache: RefCell::new(None),
            proxy_scale: std::cell::Cell::new(1.0),
            fonts: RefCell::new(std::collections::HashMap::new()),
            luts: RefCell::new(std::collections::HashMap::new()),
            next_resource_id: std::cell::Cell::new(1),
        }
    }

    /// Set the proxy scale factor for spatial param auto-scaling.
    /// Spatial params (hint = rc.pixels) are multiplied by this factor.
    /// Set the graph-level working color space.
    pub fn set_working_color_space(&self, cs: ColorSpace) {
        self.graph.borrow_mut().set_working_color_space(cs);
    }

    pub fn register_font(&self, font: std::rc::Rc<v2::font::Font>) -> u32 {
        let id = self.next_resource_id.get();
        self.next_resource_id.set(id + 1);
        self.fonts.borrow_mut().insert(id, font);
        id
    }

    pub fn register_lut(&self, data: &[u8]) -> Result<u32, PipelineError> {
        let text = std::str::from_utf8(data).map_err(|_|
            PipelineError::InvalidParams("LUT data is not valid UTF-8".into())
        )?;
        let lmt = if text.contains("LUT_3D_SIZE") || text.contains("LUT_1D_SIZE") {
            v2::parse_cube(text)?
        } else if text.trim_start().starts_with("<?xml") || text.contains("<ProcessList") || text.contains("<CLF") {
            v2::lmt::parse_clf(text)?
        } else {
            return Err(PipelineError::InvalidParams("unsupported LUT format".into()));
        };
        let id = self.next_resource_id.get();
        self.next_resource_id.set(id + 1);
        self.luts.borrow_mut().insert(id, lmt);
        Ok(id)
    }

    pub fn get_font(&self, id: u32) -> Option<std::rc::Rc<v2::font::Font>> {
        self.fonts.borrow().get(&id).cloned()
    }

    pub fn get_lut(&self, id: u32) -> Option<v2::lmt::Lmt> {
        self.luts.borrow().get(&id).cloned()
    }

    pub fn set_proxy_scale(&self, scale: f32) {
        self.proxy_scale.set(scale.max(0.01));
    }

    /// Set the shared layer cache for cross-pipeline result reuse.
    pub fn set_layer_cache(&self, cache: Rc<RefCell<LayerCache>>) {
        self.graph.borrow_mut().set_layer_cache(cache.clone());
        *self.layer_cache.borrow_mut() = Some(cache);
    }

    pub fn read(&self, data: &[u8], format_hint: Option<&str>) -> Result<u32, PipelineError> {
        // Compute source content hash from raw input bytes
        let src_hash = source_hash(data);

        // Try V2 registry: hint-based first, then auto-detect, then old fallback
        let decoded = if let Some(hint) = format_hint {
            if let Some(result) = v2::decode_with_hint_via_registry(data, hint) {
                let d = result?;
                (d.pixels, d.width, d.height, d.color_space)
            } else {
                let d = rasmcore_codecs_v2::decode_with_hint(data, hint)
                    .map_err(|e| PipelineError::ComputeError(format!("decode: {e}")))?;
                (d.pixels, d.info.width, d.info.height, d.info.color_space)
            }
        } else if let Some(result) = v2::decode_via_registry(data) {
            let d = result?;
            (d.pixels, d.width, d.height, d.color_space)
        } else {
            let d = rasmcore_codecs_v2::decode(data)
                .map_err(|e| PipelineError::ComputeError(format!("decode: {e}")))?;
            (d.pixels, d.info.width, d.info.height, d.info.color_space)
        };

        let source = SourceNode {
            pixels: decoded.0,
            info: NodeInfo {
                width: decoded.1,
                height: decoded.2,
                color_space: decoded.3,
            },
        };

        let id = self.graph.borrow_mut().add_node_with_hash(Box::new(source), src_hash);
        Ok(id)
    }

    /// Add a source node from host-decoded raw pixel bytes.
    ///
    /// For u8/u16 formats, creates a SourceNodePixels that produces a GPU conversion
    /// shader as the first node in the chain. For f32, creates a regular SourceNode.
    pub fn read_pixels(&self, data: Vec<u8>, width: u32, height: u32, format: HostPixelFormat) -> u32 {
        let format_tag = match format {
            HostPixelFormat::Rgba8 => "source-rgba8",
            HostPixelFormat::Rgba16 => "source-rgba16",
            HostPixelFormat::RgbaF32 => "source-f32",
        };
        let src_hash = source_hash_pixels(format_tag, &data, width, height);

        match format {
            HostPixelFormat::RgbaF32 => {
                // f32: reinterpret bytes as f32 and create a regular SourceNode
                let mut pixels = Vec::with_capacity((width * height * 4) as usize);
                for chunk in data.chunks_exact(4) {
                    pixels.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                let node = SourceNode {
                    pixels,
                    info: NodeInfo { width, height, color_space: ColorSpace::Linear },
                };
                self.graph.borrow_mut().add_node_with_hash(Box::new(node), src_hash)
            }
            _ => {
                // u8/u16: create SourceNodePixels that carries raw bytes + GPU shader
                let color_space = match format {
                    HostPixelFormat::Rgba8 => ColorSpace::Srgb,
                    _ => ColorSpace::Linear,
                };
                let node = SourceNodePixels {
                    raw_bytes: data,
                    format,
                    info: NodeInfo { width, height, color_space },
                };
                self.graph.borrow_mut().add_node_with_hash(Box::new(node), src_hash)
            }
        }
    }

    /// Add a source node from a pre-decoded SourceResource.
    /// No decoding — pixels are copied from the cached source using a pooled buffer.
    pub fn read_source(&self, source: &SourceResource) -> u32 {
        // Use pooled buffer for the source pixel copy (avoids fresh allocation)
        let mut buf = source.buffer_pool.borrow_mut().acquire(source.pixels.len());
        buf.copy_from_slice(&source.pixels);

        let node = SourceNode {
            pixels: buf,
            info: source.info.clone(),
        };

        // Inject the source's buffer pool into the graph for intermediate node reuse
        self.graph.borrow_mut().set_buffer_pool(source.buffer_pool.clone());
        self.graph.borrow_mut().add_node(Box::new(node))
    }

    pub fn node_info(&self, node_id: u32) -> Result<NodeInfo, PipelineError> {
        self.graph.borrow().node_info(node_id)
    }

    pub fn apply_filter(
        &self,
        source: u32,
        name: &str,
        params: &ParamMap,
    ) -> Result<u32, PipelineError> {
        let info = self.graph.borrow().node_info(source)?;

        // Apply proxy scale to spatial params (hint = "rc.pixels")
        let scale = self.proxy_scale.get();
        let mut scaled_params = if (scale - 1.0).abs() > f32::EPSILON {
            scale_spatial_params(name, params, scale)
        } else {
            params.clone()
        };

        // Compute content hash from ORIGINAL params (not scaled) so that
        // the same user params + same scale produce a consistent hash
        let upstream_hash = self.graph.borrow().content_hash(source);
        let mut hash_input = params.to_hash_bytes();
        hash_input.extend_from_slice(&scale.to_le_bytes());
        let filter_hash = content_hash(&upstream_hash, name, &hash_input);

        // Resolve all resource refs in params before factory call.
        for (key, typed_ref) in &scaled_params.refs.clone() {
            match typed_ref {
                v2::TypedRef::Font(id) => {
                    if let Some(font) = self.get_font(*id) {
                        scaled_params.resolved_fonts.insert(key.clone(), font);
                    }
                }
                v2::TypedRef::Lut(id) => {
                    if let Some(lut) = self.get_lut(*id) {
                        scaled_params.resolved_luts.insert(key.clone(), lut);
                    }
                }
                v2::TypedRef::Node(_) => {}
            }
        }

        // Create the filter node
        let node = create_filter_node(name, source, info.clone(), &scaled_params).ok_or_else(|| {
            PipelineError::InvalidParams(format!("unknown filter: {name}"))
        })?;

        // If the filter declares a preferred color space different from its
        // upstream, wrap it with CSC nodes: upstream_cs → preferred → filter → preferred → upstream_cs.
        // This must happen here (not in the fusion optimizer) because the caller
        // holds the returned node ID — the optimizer can't change that.
        let upstream_cs = info.color_space;
        if let Some(preferred) = node.preferred_color_space() {
            if preferred != upstream_cs && upstream_cs != ColorSpace::Unknown {
                let csc_to = v2::ColorConvertNode::new(source, info.clone(), upstream_cs, preferred);
                let csc_to_id = self.graph.borrow_mut().add_node(Box::new(csc_to));

                let converted_info = NodeInfo { color_space: preferred, ..info };
                let filter_node = create_filter_node(name, csc_to_id, converted_info.clone(), &scaled_params)
                    .expect("filter creation already validated");
                let filter_id = self.graph.borrow_mut().add_node_with_hash(filter_node, filter_hash);

                let filter_out_info = self.graph.borrow().node_info(filter_id)?;
                let csc_back = v2::ColorConvertNode::new(filter_id, filter_out_info, preferred, upstream_cs);
                let csc_back_id = self.graph.borrow_mut().add_node(Box::new(csc_back));

                return Ok(csc_back_id);
            }
        }

        let id = self.graph.borrow_mut().add_node_with_hash(node, filter_hash);
        Ok(id)
    }

    pub fn use_lmt(&self, source: u32, data: &[u8]) -> Result<u32, PipelineError> {
        let text = std::str::from_utf8(data).map_err(|_| {
            PipelineError::InvalidParams("LMT data is not valid UTF-8".into())
        })?;

        // Auto-detect format from content
        let lmt = if text.contains("LUT_3D_SIZE") || text.contains("LUT_1D_SIZE") {
            rasmcore_pipeline_v2::parse_cube(text)?
        } else if text.trim_start().starts_with("<?xml")
            || text.contains("<ProcessList")
            || text.contains("<CLF")
        {
            rasmcore_pipeline_v2::lmt::parse_clf(text)?
        } else {
            return Err(PipelineError::InvalidParams(
                "unsupported LMT format — expected .cube or .clf content".into(),
            ));
        };

        let info = self.graph.borrow().node_info(source)?;
        let upstream_hash = self.graph.borrow().content_hash(source);
        // Hash the raw LMT data for content-addressed caching
        let lmt_hash = content_hash(&upstream_hash, "lmt", data);
        let node = Box::new(rasmcore_pipeline_v2::LmtNode::new(source, info, lmt));
        let id = self.graph.borrow_mut().add_node_with_hash(node, lmt_hash);
        Ok(id)
    }

    /// Apply a registered color transform (IDT, LMT, or Output Transform).
    pub fn apply_color_transform(&self, source: u32, transform_id: u32) -> Result<u32, PipelineError> {
        let info = self.graph.borrow().node_info(source)?;
        let transform = instance_get_transform(transform_id)
            .ok_or_else(|| PipelineError::InvalidParams(format!("transform ID {transform_id} not found")))?;
        let node = Box::new(v2::color_transform::ColorTransformNode::new(source, info, transform));
        let id = self.graph.borrow_mut().add_node(node);
        Ok(id)
    }

    /// Draw text using an externally-provided Font resource.
    pub fn draw_text(
        &self,
        source: u32,
        font: std::rc::Rc<v2::font::Font>,
        params: &ParamMap,
    ) -> Result<u32, PipelineError> {
        let info = self.graph.borrow().node_info(source)?;
        let text = params.strings.get("text").cloned().unwrap_or_default();
        let x = params.floats.get("x").copied().unwrap_or(0.0);
        let y = params.floats.get("y").copied().unwrap_or(0.0);
        let size = params.floats.get("size").copied().unwrap_or(24.0);
        let cr = params.floats.get("color_r").copied().unwrap_or(1.0);
        let cg = params.floats.get("color_g").copied().unwrap_or(1.0);
        let cb = params.floats.get("color_b").copied().unwrap_or(1.0);
        let ca = params.floats.get("color_a").copied().unwrap_or(1.0);

        let upstream_hash = self.graph.borrow().content_hash(source);
        let filter_hash = content_hash(&upstream_hash, "draw_text", &params.to_hash_bytes());

        let node = v2::filters::draw::DrawTextNode::new(
            source, info, font, text, x, y, size, [cr, cg, cb, ca],
        );
        let id = self.graph.borrow_mut().add_node_with_hash(Box::new(node), filter_hash);
        Ok(id)
    }

    pub fn apply_stroke(
        &self,
        source: u32,
        stroke: &StrokeInputResource,
        color: [f32; 3],
        _blend_mode: &str,
    ) -> Result<u32, PipelineError> {
        let info = self.graph.borrow().node_info(source)?;

        // Content hash includes stroke point data + brush params for caching
        let upstream_hash = self.graph.borrow().content_hash(source);
        // Hash stroke point data as bytes for content-addressed caching
        let pts = stroke.points.borrow();
        let point_bytes: Vec<u8> = pts.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let stroke_hash = content_hash(&upstream_hash, "apply_stroke", &point_bytes);

        let n_points = pts.len() / STROKE_POINT_STRIDE;

        // Placeholder: pass-through node (actual brush engine added in brush-engine-cpu track)
        let node = Box::new(PassthroughStrokeNode {
            upstream: source,
            info,
            point_count: n_points,
            color,
        });
        let id = self.graph.borrow_mut().add_node_with_hash(node, stroke_hash);
        Ok(id)
    }

    pub fn write(
        &self,
        node_id: u32,
        format: &str,
        quality: Option<u8>,
    ) -> Result<Vec<u8>, PipelineError> {
        let mut pixels = self.graph.borrow_mut().request_full(node_id)?;
        let info = self.graph.borrow().node_info(node_id)?;

        // Apply output transform: convert from working space (Linear) to
        // the format's preferred color space (sRGB for web formats, Linear for EXR/HDR).
        apply_output_transform(&mut pixels, &info, format);

        // Try V2 registry first, fall back to old codecs-v2 encode
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

    /// Render pixels with output transform applied (sRGB by default).
    ///
    /// Returns f32 pixels in sRGB color space for display. Callers that
    /// need raw linear working-space pixels should use `render_linear()`.
    pub fn render(&self, node_id: u32) -> Result<Vec<f32>, PipelineError> {
        let mut pixels = self.graph.borrow_mut().request_full(node_id)?;
        let info = self.graph.borrow().node_info(node_id)?;

        // Default output transform: Linear → sRGB for display
        apply_output_transform(&mut pixels, &info, "png");
        Ok(pixels)
    }

    /// Render raw f32 pixels in the pipeline's working color space (Linear).
    ///
    /// No output transform applied. Use for GPU display paths, EXR workflows,
    /// or any consumer that handles color space conversion externally.
    pub fn render_linear(&self, node_id: u32) -> Result<Vec<f32>, PipelineError> {
        self.graph.borrow_mut().request_full(node_id)
    }

    /// Get layer cache statistics (if a cache is set).
    pub fn layer_cache_stats(&self) -> Option<v2::CacheStats> {
        self.layer_cache.borrow().as_ref().map(|lc| lc.borrow().stats())
    }

    /// Finalize layer cache — reset references and clean up unreferenced entries.
    /// Call after each render cycle to evict stale entries from previous images/chains.
    pub fn finalize_layer_cache(&self) {
        self.graph.borrow().reset_layer_cache_references();
        self.graph.borrow().cleanup_layer_cache();
    }

    /// Set a named ref pointing to a node in the graph.
    pub fn set_ref(&self, name: &str, node_id: u32) {
        self.graph.borrow_mut().set_ref(name, node_id);
    }

    /// Get the node ID for a named ref, or None if not set.
    pub fn get_ref(&self, name: &str) -> Option<u32> {
        self.graph.borrow().get_ref(name)
    }
}

// ─── WIT Bindings ���──────────────────────────────────────────────────────────

// ─── Layer Cache Resource ────────────────────────────────────────────────────

/// WASM-exported layer cache resource.
///
/// Wraps the domain LayerCache in Rc<RefCell<>> so it can be shared
/// across pipeline instances.
pub struct LayerCacheResource {
    #[allow(dead_code)] // read in wasm32 WIT adapter impl
    inner: Rc<RefCell<LayerCache>>,
}

impl LayerCacheResource {
    /// Create with capacity in megabytes.
    pub fn new_with_capacity(capacity_mb: u32) -> Self {
        let budget = capacity_mb as usize * 1024 * 1024;
        Self {
            inner: Rc::new(RefCell::new(LayerCache::new(budget))),
        }
    }
}

/// WASM-exported font resource.




/// Placeholder stroke node — passes through source unchanged.
///
/// This will be replaced by the actual brush engine in the brush-engine-cpu track.
/// For now it validates the wiring works end-to-end.
struct PassthroughStrokeNode {
    upstream: u32,
    info: NodeInfo,
    #[allow(dead_code)]
    point_count: usize,
    #[allow(dead_code)]
    color: [f32; 3],
}

impl v2::Node for PassthroughStrokeNode {
    fn info(&self) -> NodeInfo {
        self.info.clone()
    }

    fn compute(
        &self,
        request: v2::rect::Rect,
        upstream: &mut dyn v2::node::Upstream,
    ) -> Result<Vec<f32>, PipelineError> {
        // Passthrough — actual brush rendering will be added in brush-engine-cpu
        upstream.request(self.upstream, request)
    }

    fn upstream_ids(&self) -> Vec<u32> {
        vec![self.upstream]
    }
}

/// WASM-exported stroke input resource.
pub struct StrokeInputResource {
    points: RefCell<Vec<f32>>,
    #[allow(dead_code)]
    brush_params: StrokeBrushParams,
    #[allow(dead_code)]
    ended: std::cell::Cell<bool>,
}

/// Internal brush params (mirrors WIT brush-params).
#[derive(Clone)]
pub struct StrokeBrushParams {
    pub diameter: f32,
    pub spacing: f32,
    pub hardness: f32,
    pub flow: f32,
    pub opacity: f32,
    pub angle: f32,
    pub roundness: f32,
    pub scatter: f32,
    pub smoothing: f32,
}

/// Number of f32 fields per stroke point.
const STROKE_POINT_STRIDE: usize = 8;

/// WASM-exported undo stack resource.
pub struct UndoStackResource {
    inner: RefCell<v2::undo::UndoStack>,
}

#[cfg(target_arch = "wasm32")]
struct Component;

#[cfg(target_arch = "wasm32")]
bindings::export!(Component with_types_in bindings);

#[cfg(target_arch = "wasm32")]
impl wit::Guest for Component {
    type ImagePipelineV2 = PipelineResource;
    type LayerCache = LayerCacheResource;
    type Source = SourceResource;
    type StrokeInput = StrokeInputResource;
    type UndoStack = UndoStackResource;

    fn register_font(data: Vec<u8>) -> Result<u32, RasmcoreError> {
        instance_register_font(&data).map_err(to_wit_error)
    }

    fn register_lut(data: Vec<u8>) -> Result<u32, RasmcoreError> {
        instance_register_lut(&data).map_err(to_wit_error)
    }

    fn register_transform(data: Vec<u8>, format_hint: Option<String>) -> Result<u32, RasmcoreError> {
        instance_register_transform(&data, format_hint.as_deref()).map_err(to_wit_error)
    }

    fn get_transform(name: String) -> Result<u32, RasmcoreError> {
        instance_register_transform_preset(&name).map_err(to_wit_error)
    }

    fn list_transform_presets() -> Vec<wit::TransformPresetInfo> {
        v2::color_transform::preset_list().into_iter().map(|p| wit::TransformPresetInfo {
            name: p.name.to_string(),
            display_name: p.display_name.to_string(),
            kind: format!("{:?}", p.kind).to_lowercase(),
            source_space: p.source_space.to_string(),
            target_space: p.target_space.to_string(),
            vendor: p.vendor.to_string(),
            description: p.description.to_string(),
        }).collect()
    }
}

#[cfg(target_arch = "wasm32")]
impl wit::GuestUndoStack for UndoStackResource {
    fn new(memory_budget_mb: u32) -> Self {
        let budget = memory_budget_mb as usize * 1024 * 1024;
        Self { inner: RefCell::new(v2::undo::UndoStack::new(budget)) }
    }

    fn push_stroke(&self, pixels: Vec<f32>, width: u32, height: u32, x: u32, y: u32, w: u32, h: u32) {
        let bounds = v2::Rect::new(x, y, w, h);
        self.inner.borrow_mut().push_stroke(&pixels, width, height, bounds);
    }

    fn record_post(&self, pixels: Vec<f32>, width: u32, height: u32) {
        self.inner.borrow_mut().record_post_state(&pixels, width, height);
    }

    fn undo(&self, pixels: Vec<f32>, width: u32, height: u32) -> Option<Vec<f32>> {
        let mut px = pixels;
        self.inner.borrow_mut().undo(&mut px, width, height).map(|_| px)
    }

    fn redo(&self, pixels: Vec<f32>, width: u32, height: u32) -> Option<Vec<f32>> {
        let mut px = pixels;
        self.inner.borrow_mut().redo(&mut px, width, height).map(|_| px)
    }

    fn clear(&self) {
        self.inner.borrow_mut().clear();
    }

    fn set_memory_budget(&self, mb: u32) {
        self.inner.borrow_mut().set_memory_budget(mb as usize * 1024 * 1024);
    }

    fn undo_count(&self) -> u32 {
        self.inner.borrow().undo_count() as u32
    }

    fn redo_count(&self) -> u32 {
        self.inner.borrow().redo_count() as u32
    }
}

#[cfg(target_arch = "wasm32")]
impl wit::GuestSource for SourceResource {
    fn new(data: Vec<u8>, format_hint: Option<String>) -> Self {
        SourceResource::new(&data, format_hint.as_deref())
            .unwrap_or_else(|e| {
                // Return a 1x1 transparent pixel on decode failure
                // (WIT constructors cannot return Result)
                eprintln!("Source decode failed: {e}");
                SourceResource {
                    pixels: vec![0.0, 0.0, 0.0, 0.0],
                    info: NodeInfo {
                        width: 1,
                        height: 1,
                        color_space: ColorSpace::Srgb,
                    },
                    buffer_pool: Rc::new(RefCell::new(v2::BufferPool::new())),
                    raw_pixels: None,
                    src_hash: v2::hash::ZERO_HASH,
                }
            })
    }

    fn from_pixels(data: Vec<u8>, width: u32, height: u32, format: wit::HostPixelFormat) -> wit::Source {
        let fmt = match format {
            wit::HostPixelFormat::Rgba8 => HostPixelFormat::Rgba8,
            wit::HostPixelFormat::Rgba16 => HostPixelFormat::Rgba16,
            wit::HostPixelFormat::RgbaF32 => HostPixelFormat::RgbaF32,
        };
        wit::Source::new(SourceResource::from_pixels(data, width, height, fmt))
    }

    fn info(&self) -> wit::NodeInfo {
        let info = SourceResource::info(self);
        wit::NodeInfo {
            width: info.width,
            height: info.height,
            color_space: match info.color_space {
                ColorSpace::Linear => wit::ColorSpace::Linear,
                ColorSpace::Srgb => wit::ColorSpace::Srgb,
                ColorSpace::AcesCg => wit::ColorSpace::AcesCg,
                ColorSpace::AcesCct => wit::ColorSpace::AcesCct,
                ColorSpace::AcesCc => wit::ColorSpace::AcesCc,
                ColorSpace::Aces2065_1 => wit::ColorSpace::Aces2065,
                ColorSpace::DisplayP3 => wit::ColorSpace::DisplayP3,
                ColorSpace::Rec709 => wit::ColorSpace::Rec709,
                ColorSpace::Rec2020 => wit::ColorSpace::Rec2020,
                _ => wit::ColorSpace::Unknown,
            },
        }
    }

    fn metadata(&self) -> Vec<wit::MetadataEntry> {
        // SourceResource doesn't carry ImageMetadata yet (pipeline flow
        // track will add it). Return empty for now.
        vec![]
    }
}


#[cfg(target_arch = "wasm32")]
impl wit::GuestStrokeInput for StrokeInputResource {
    fn new(brush: wit::BrushParams) -> Self {
        Self {
            points: RefCell::new(Vec::new()),
            brush_params: StrokeBrushParams {
                diameter: brush.diameter,
                spacing: brush.spacing,
                hardness: brush.hardness,
                flow: brush.flow,
                opacity: brush.opacity,
                angle: brush.angle,
                roundness: brush.roundness,
                scatter: brush.scatter,
                smoothing: brush.smoothing,
            },
            ended: std::cell::Cell::new(false),
        }
    }

    fn add_point(&self, point: wit::StrokePoint) {
        let mut pts = self.points.borrow_mut();
        pts.push(point.x);
        pts.push(point.y);
        pts.push(point.pressure);
        pts.push(point.tilt_x);
        pts.push(point.tilt_y);
        pts.push(point.rotation);
        pts.push(point.velocity);
        pts.push(point.timestamp);
    }

    fn end_stroke(&self) {
        self.ended.set(true);
    }

    fn point_count(&self) -> u32 {
        (self.points.borrow().len() / STROKE_POINT_STRIDE) as u32
    }

    fn point_data(&self) -> Vec<f32> {
        self.points.borrow().clone()
    }
}

#[cfg(target_arch = "wasm32")]
impl wit::GuestLayerCache for LayerCacheResource {
    fn new(capacity_mb: u32) -> Self {
        LayerCacheResource::new_with_capacity(capacity_mb)
    }

    fn stats(&self) -> wit::CacheStats {
        let s = self.inner.borrow().stats();
        wit::CacheStats {
            entries: s.entries,
            hits: s.hits,
            misses: s.misses,
            size_bytes: s.size_bytes,
        }
    }

    fn clear(&self) {
        self.inner.borrow_mut().clear();
    }

    fn set_cache_quality(&self, quality: wit::CacheQuality) {
        let q = match quality {
            wit::CacheQuality::Full => v2::CacheQuality::Full,
            wit::CacheQuality::Q16 => v2::CacheQuality::Q16,
            wit::CacheQuality::Q8 => v2::CacheQuality::Q8,
        };
        self.inner.borrow_mut().set_cache_quality(q);
    }
}

#[cfg(target_arch = "wasm32")]
impl wit::GuestImagePipelineV2 for PipelineResource {
    fn new() -> Self {
        PipelineResource::new()
    }

    fn list_operations(&self) -> Vec<wit::OperationInfo> {
        v2::registered_operations()
            .into_iter()
            .map(|op| wit::OperationInfo {
                name: op.name.to_string(),
                display_name: op.display_name.to_string(),
                category: op.category.to_string(),
                kind: match op.kind {
                    v2::OperationKind::Filter => wit::OperationKind::Filter,
                    v2::OperationKind::Encoder => wit::OperationKind::Encoder,
                    v2::OperationKind::Decoder => wit::OperationKind::Decoder,
                    v2::OperationKind::Transform => wit::OperationKind::Transform,
                    v2::OperationKind::ColorConversion => wit::OperationKind::ColorConversion,
                },
                gpu_capable: op.capabilities.gpu,
                params: op
                    .params
                    .iter()
                    .map(|p| wit::ParamDescriptor {
                        name: p.name.to_string(),
                        value_type: match p.value_type {
                            v2::ParamType::F32 => wit::ParamType::F32Val,
                            v2::ParamType::F64 => wit::ParamType::F64Val,
                            v2::ParamType::U32 => wit::ParamType::U32Val,
                            v2::ParamType::I32 => wit::ParamType::I32Val,
                            v2::ParamType::Bool => wit::ParamType::BoolVal,
                            v2::ParamType::String => wit::ParamType::StringVal,
                            v2::ParamType::Rect => wit::ParamType::RectVal,
                            v2::ParamType::NodeRef | v2::ParamType::FontRef | v2::ParamType::LutRef => wit::ParamType::U32Val,
                        },
                        min: p.min,
                        max: p.max,
                        step: p.step,
                        default_val: p.default,
                        hint: p.hint.map(|s| s.to_string()),
                    })
                    .collect(),
            })
            .collect()
    }

    fn find_operation(&self, name: String) -> Option<wit::OperationInfo> {
        // Reuse list_operations and filter — simple enough for POC
        self.list_operations().into_iter().find(|op| op.name == name)
    }

    fn read(&self, data: Vec<u8>, config: Option<wit::ReadConfig>) -> Result<u32, RasmcoreError> {
        let hint = config.as_ref().and_then(|c| c.format_hint.as_deref());
        PipelineResource::read(self, &data, hint).map_err(to_wit_error)
    }

    fn read_source(&self, source: wit::SourceBorrow<'_>) -> Result<u32, RasmcoreError> {
        let source_res = source.get::<SourceResource>();
        Ok(PipelineResource::read_source(self, source_res))
    }

    fn read_pixels(
        &self,
        data: Vec<u8>,
        width: u32,
        height: u32,
        format: wit::HostPixelFormat,
    ) -> Result<u32, RasmcoreError> {
        let fmt = match format {
            wit::HostPixelFormat::Rgba8 => HostPixelFormat::Rgba8,
            wit::HostPixelFormat::Rgba16 => HostPixelFormat::Rgba16,
            wit::HostPixelFormat::RgbaF32 => HostPixelFormat::RgbaF32,
        };
        Ok(PipelineResource::read_pixels(self, data, width, height, fmt))
    }

    fn node_info(&self, node: u32) -> Result<wit::NodeInfo, RasmcoreError> {
        let info = PipelineResource::node_info(self, node).map_err(to_wit_error)?;
        Ok(wit::NodeInfo {
            width: info.width,
            height: info.height,
            color_space: match info.color_space {
                ColorSpace::Linear => wit::ColorSpace::Linear,
                ColorSpace::Srgb => wit::ColorSpace::Srgb,
                ColorSpace::AcesCg => wit::ColorSpace::AcesCg,
                ColorSpace::AcesCct => wit::ColorSpace::AcesCct,
                ColorSpace::AcesCc => wit::ColorSpace::AcesCc,
                ColorSpace::Aces2065_1 => wit::ColorSpace::Aces2065,
                ColorSpace::DisplayP3 => wit::ColorSpace::DisplayP3,
                ColorSpace::Rec709 => wit::ColorSpace::Rec709,
                _ => wit::ColorSpace::Unknown,
            },
        })
    }

    fn node_metadata(&self, node: u32) -> Result<Vec<wit::MetadataEntry>, RasmcoreError> {
        // For now, return empty metadata — source nodes will carry metadata
        // once the graph propagation is wired (SourceNode stores ImageMetadata).
        let _ = PipelineResource::node_info(self, node).map_err(to_wit_error)?;
        Ok(vec![])
    }

    fn apply_filter(
        &self,
        source: u32,
        name: String,
        params: Vec<u8>,
    ) -> Result<u32, RasmcoreError> {
        // Deserialize params from buffer — simple key=value pairs
        // For now, use a minimal binary format: [name_len:u8, name_bytes, value:f32] repeated
        let param_map = deserialize_params(&params);
        self.apply_filter(source, &name, &param_map)
            .map_err(to_wit_error)
    }

    fn apply_transform(
        &self,
        _source: u32,
        _name: String,
        _params: Vec<u8>,
    ) -> Result<u32, RasmcoreError> {
        Err(RasmcoreError::NotImplemented)
    }

    fn convert_color_space(
        &self,
        _source: u32,
        _target: wit::ColorSpace,
    ) -> Result<u32, RasmcoreError> {
        Err(RasmcoreError::NotImplemented)
    }

    fn apply_view_transform(
        &self,
        _source: u32,
        _transform: wit::ViewTransform,
    ) -> Result<u32, RasmcoreError> {
        Err(RasmcoreError::NotImplemented)
    }

    fn use_lmt(&self, source: u32, data: Vec<u8>) -> Result<u32, RasmcoreError> {
        self.use_lmt(source, &data).map_err(to_wit_error)
    }

    // ─── Color Transforms (IDT / LMT / Output Transform) ──────────────

    fn apply_idt(&self, source: u32, transform_id: u32) -> Result<u32, RasmcoreError> {
        self.apply_color_transform(source, transform_id).map_err(to_wit_error)
    }

    fn apply_lmt(&self, source: u32, transform_id: u32) -> Result<u32, RasmcoreError> {
        self.apply_color_transform(source, transform_id).map_err(to_wit_error)
    }

    fn apply_output_transform(&self, source: u32, transform_id: u32) -> Result<u32, RasmcoreError> {
        self.apply_color_transform(source, transform_id).map_err(to_wit_error)
    }

    fn list_brush_presets(&self) -> Vec<wit::BrushPresetInfo> {
        v2::brush::registered_presets().into_iter().map(|p| wit::BrushPresetInfo {
            name: p.name.to_string(),
            display_name: p.display_name.to_string(),
            category: p.category.to_string(),
            description: p.description.to_string(),
        }).collect()
    }

    fn get_brush_preset(&self, name: String) -> Option<wit::BrushParams> {
        v2::brush::find_preset(&name).map(|p| wit::BrushParams {
            diameter: p.params.diameter,
            spacing: p.params.spacing,
            hardness: p.params.hardness,
            flow: p.params.flow,
            opacity: p.params.opacity,
            angle: p.params.angle,
            roundness: p.params.roundness,
            scatter: p.params.scatter,
            smoothing: p.params.smoothing,
            pressure_size_curve: curve_to_wit(&p.params.dynamics.pressure_size),
            pressure_opacity_curve: curve_to_wit(&p.params.dynamics.pressure_opacity),
            velocity_size_curve: curve_to_wit(&p.params.dynamics.velocity_size),
            tip_texture: None,
            tip_width: 0,
            tip_height: 0,
        })
    }

    fn apply_stroke(
        &self,
        source: u32,
        stroke: wit::StrokeInputBorrow<'_>,
        color_r: f32,
        color_g: f32,
        color_b: f32,
        blend_mode: String,
    ) -> Result<u32, RasmcoreError> {
        let stroke_res = stroke.get::<StrokeInputResource>();
        self.apply_stroke(source, stroke_res, [color_r, color_g, color_b], &blend_mode)
            .map_err(to_wit_error)
    }

    fn set_demand_strategy(&self, _strategy: wit::DemandStrategy) {}

    fn set_gpu_available(&self, _available: bool) {}

    fn set_layer_cache(&self, cache: wit::LayerCacheBorrow<'_>) {
        let cache_resource = cache.get::<LayerCacheResource>();
        self.set_layer_cache(cache_resource.inner.clone());
    }

    fn set_working_color_space(&self, cs: wit::ColorSpace) {
        let domain_cs = match cs {
            wit::ColorSpace::Linear => ColorSpace::Linear,
            wit::ColorSpace::Srgb => ColorSpace::Srgb,
            wit::ColorSpace::AcesCg => ColorSpace::AcesCg,
            wit::ColorSpace::AcesCct => ColorSpace::AcesCct,
            wit::ColorSpace::AcesCc => ColorSpace::AcesCc,
            wit::ColorSpace::Aces2065 => ColorSpace::Aces2065_1,
            wit::ColorSpace::DisplayP3 => ColorSpace::DisplayP3,
            wit::ColorSpace::Rec709 => ColorSpace::Rec709,
            wit::ColorSpace::Rec2020 => ColorSpace::Rec2020,
            wit::ColorSpace::Unknown => ColorSpace::Unknown,
        };
        self.set_working_color_space(domain_cs);
    }

    fn set_proxy_scale(&self, scale: f32) {
        self.set_proxy_scale(scale);
    }

    fn write(
        &self,
        node: u32,
        format: String,
        quality: Option<u8>,
    ) -> Result<Vec<u8>, RasmcoreError> {
        PipelineResource::write(self, node, &format, quality).map_err(to_wit_error)
    }

    fn render(&self, node: u32) -> Result<Vec<f32>, RasmcoreError> {
        self.render(node).map_err(to_wit_error)
    }

    fn render_linear(&self, node: u32) -> Result<Vec<f32>, RasmcoreError> {
        self.render_linear(node).map_err(to_wit_error)
    }

    fn render_gpu_plan(&self, node: u32) -> Result<Option<wit::GpuPlan>, RasmcoreError> {
        let plan = self
            .graph
            .borrow_mut()
            .gpu_plan(node)
            .map_err(to_wit_error)?;

        Ok(plan.map(|p| wit::GpuPlan {
            shaders: p
                .shaders
                .into_iter()
                .map(|s| wit::GpuShader {
                    source: s.body,
                    entry_point: s.entry_point.to_string(),
                    workgroup_x: s.workgroup_size[0],
                    workgroup_y: s.workgroup_size[1],
                    workgroup_z: s.workgroup_size[2],
                    params: s.params,
                    extra_buffers: s.extra_buffers,
                })
                .collect(),
            input_pixels: p.input_pixels,
            width: p.width,
            height: p.height,
        }))
    }

    fn render_multi_gpu_plan(
        &self,
        targets: Vec<(String, u32)>,
    ) -> Result<wit::MultiGpuPlan, RasmcoreError> {
        let plan = self
            .graph
            .borrow_mut()
            .render_multi_gpu_plan(&targets)
            .map_err(to_wit_error)?;

        let stages = plan
            .stages
            .into_iter()
            .map(|s| {
                let shaders = s
                    .shaders
                    .into_iter()
                    .map(|sh| wit::GpuShader {
                        source: sh.body,
                        entry_point: sh.entry_point.to_string(),
                        workgroup_x: sh.workgroup_size[0],
                        workgroup_y: sh.workgroup_size[1],
                        workgroup_z: sh.workgroup_size[2],
                        params: sh.params,
                        extra_buffers: sh.extra_buffers,
                    })
                    .collect();
                let input = match s.input {
                    v2::StageInput::Pixels(px) => wit::StageInput::Pixels(px),
                    v2::StageInput::PriorStage(name) => wit::StageInput::PriorStage(name),
                };
                wit::GpuStage {
                    target_name: s.target_name,
                    shaders,
                    input,
                    width: s.width,
                    height: s.height,
                }
            })
            .collect();

        Ok(wit::MultiGpuPlan { stages })
    }

    fn inject_gpu_result(&self, node: u32, pixels: Vec<f32>) {
        self.graph.borrow_mut().inject_gpu_result(node, pixels);
    }

    fn finalize_layer_cache(&self) {
        PipelineResource::finalize_layer_cache(self);
    }

    fn set_ref(&self, name: String, node: u32) {
        self.set_ref(&name, node);
    }

    fn get_ref(&self, name: String) -> Option<u32> {
        PipelineResource::get_ref(self, &name)
    }

    fn set_tracing(&self, enabled: bool) {
        self.graph.borrow_mut().set_tracing(enabled);
    }

    fn take_trace(&self) -> Vec<wit::TraceEvent> {
        let trace = self.graph.borrow_mut().take_trace();
        trace
            .events
            .into_iter()
            .map(|e| wit::TraceEvent {
                kind: match e.kind {
                    v2::TraceEventKind::Fusion => wit::TraceEventKind::Fusion,
                    v2::TraceEventKind::ShaderCompile => wit::TraceEventKind::ShaderCompile,
                    v2::TraceEventKind::GpuDispatch => wit::TraceEventKind::GpuDispatch,
                    v2::TraceEventKind::CpuFallback => wit::TraceEventKind::CpuFallback,
                    v2::TraceEventKind::Encode => wit::TraceEventKind::Encode,
                },
                name: e.name,
                duration_us: e.duration_us,
                detail: e.detail,
            })
            .collect()
    }

    // ─── ML Operations ─────────────────────────────────────────────────

    fn apply_ml(
        &self,
        source: u32,
        model_name: String,
        params: Vec<u8>,
    ) -> Result<u32, RasmcoreError> {
        use rasmcore_pipeline_v2::ml_node::*;

        let info = self.graph.borrow().node_info(source).map_err(to_wit_error)?;

        // TODO: Query host ml-capabilities to get model info for tiling/output.
        // For now, create a basic MlNode with defaults.
        // The actual ml-execute callback will be wired in the ml-sdk-host track.
        let node = MlNode::new(
            source,
            info,
            model_name,
            "1.0".to_string(),
            params,
            MlInputSpec::Dynamic,
            MlOutputKind::Image,
            1, // no upscale by default
            TensorLayout::Nchw,
            TensorDtype::Float32,
        );
        let id = self.graph.borrow_mut().add_node(Box::new(node));
        Ok(id)
    }

    fn list_ml_models(&self) -> Vec<wit::MlModelInfo> {
        // TODO: Call host ml-capabilities() when available and translate
        // to MlModelInfo records. For now returns empty.
        Vec::new()
    }
}

// ─── Param deserialization ──────────────────────────────────────────────────

/// Scale spatial params (hint = "rc.pixels") by the given proxy scale factor.
///
/// Returns a new ParamMap with pixel-hint float params multiplied by `scale`.
/// Non-pixel params are passed through unchanged.
fn scale_spatial_params(filter_name: &str, params: &ParamMap, scale: f32) -> ParamMap {
    let mut scaled = params.clone();
    if let Some(descriptors) = v2::param_descriptors(filter_name) {
        for desc in descriptors {
            if desc.hint == Some("rc.pixels") {
                if let Some(val) = scaled.floats.get_mut(desc.name) {
                    *val *= scale;
                }
            }
        }
    }
    scaled
}

/// Deserialize a simple binary param buffer into a ParamMap.
///
/// Format: repeated [name_len:u8, name_bytes, type:u8, value_bytes]
///   type 0 = f32 (4 bytes), type 1 = u32 (4 bytes), type 2 = bool (1 byte)
fn deserialize_params(buf: &[u8]) -> ParamMap {
    let mut map = ParamMap::new();
    let mut i = 0;
    while i < buf.len() {
        if i >= buf.len() {
            break;
        }
        let name_len = buf[i] as usize;
        i += 1;
        if i + name_len > buf.len() {
            break;
        }
        let name = String::from_utf8_lossy(&buf[i..i + name_len]).to_string();
        i += name_len;
        if i >= buf.len() {
            break;
        }
        let value_type = buf[i];
        i += 1;
        match value_type {
            0 => {
                // f32
                if i + 4 > buf.len() {
                    break;
                }
                let v = f32::from_le_bytes([buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]);
                map.floats.insert(name, v);
                i += 4;
            }
            1 => {
                // u32 (stored as i64 in ParamMap)
                if i + 4 > buf.len() {
                    break;
                }
                let v = u32::from_le_bytes([buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]);
                map.ints.insert(name, v as i64);
                i += 4;
            }
            2 => {
                // bool
                if i + 1 > buf.len() {
                    break;
                }
                map.bools.insert(name, buf[i] != 0);
                i += 1;
            }
            3 => { // NodeRef
                if i + 4 > buf.len() { break; }
                let v = u32::from_le_bytes([buf[i], buf[i+1], buf[i+2], buf[i+3]]);
                map.refs.insert(name, v2::TypedRef::Node(v));
                i += 4;
            }
            4 => { // FontRef
                if i + 4 > buf.len() { break; }
                let v = u32::from_le_bytes([buf[i], buf[i+1], buf[i+2], buf[i+3]]);
                map.refs.insert(name, v2::TypedRef::Font(v));
                i += 4;
            }
            5 => { // LutRef
                if i + 4 > buf.len() { break; }
                let v = u32::from_le_bytes([buf[i], buf[i+1], buf[i+2], buf[i+3]]);
                map.refs.insert(name, v2::TypedRef::Lut(v));
                i += 4;
            }
            _ => break,
        }
    }
    map
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_resource_exists() {
        let pipe = PipelineResource::new();
        // Just verify construction works
        drop(pipe);
    }

    #[test]
    fn deserialize_brightness_params() {
        // Encode: name="amount", type=f32, value=0.5
        let mut buf = Vec::new();
        buf.push(6); // name len
        buf.extend_from_slice(b"amount");
        buf.push(0); // f32 type
        buf.extend_from_slice(&0.5f32.to_le_bytes());

        let params = deserialize_params(&buf);
        assert!((params.get_f32("amount") - 0.5).abs() < 1e-6);
    }

    #[test]
    fn apply_brightness_filter() {
        let pipe = PipelineResource::new();

        // Create a solid white 2x2 source manually
        let source = SourceNode {
            pixels: vec![0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 1.0,
                         0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 1.0],
            info: NodeInfo {
                width: 2,
                height: 2,
                color_space: ColorSpace::Linear,
            },
        };
        let src_id = pipe.graph.borrow_mut().add_node(Box::new(source));

        // Apply brightness +0.25
        let mut params = ParamMap::new();
        params.floats.insert("amount".into(), 0.25);
        let bright_id = pipe.apply_filter(src_id, "brightness", &params).unwrap();

        // Apply brightness -0.25 (should cancel out)
        let mut params2 = ParamMap::new();
        params2.floats.insert("amount".into(), -0.25);
        let back_id = pipe.apply_filter(bright_id, "brightness", &params2).unwrap();

        // Render — should be back to 0.5 (fusion composes +0.25 and -0.25 = +0.0)
        let output = pipe.render(back_id).unwrap();
        assert!(
            (output[0] - 0.5).abs() < 1e-5,
            "expected 0.5, got {} — fusion should cancel +0.25 and -0.25",
            output[0]
        );
    }

    #[test]
    fn apply_brightness_extreme_roundtrip() {
        let pipe = PipelineResource::new();

        let source = SourceNode {
            pixels: vec![0.5, 0.3, 0.7, 1.0],
            info: NodeInfo {
                width: 1,
                height: 1,
                color_space: ColorSpace::Linear,
            },
        };
        let src_id = pipe.graph.borrow_mut().add_node(Box::new(source));

        // +0.5, +0.5, -0.5, -0.5 = should be identity
        let mut current = src_id;
        for amount in [0.5, 0.5, -0.5, -0.5] {
            let mut p = ParamMap::new();
            p.floats.insert("amount".into(), amount);
            current = pipe.apply_filter(current, "brightness", &p).unwrap();
        }

        let output = pipe.render(current).unwrap();
        assert!(
            (output[0] - 0.5).abs() < 1e-5,
            "expected 0.5, got {} — 4x brightness should cancel to identity",
            output[0]
        );
        assert!(
            (output[1] - 0.3).abs() < 1e-5,
            "expected 0.3, got {}",
            output[1]
        );
        assert!(
            (output[2] - 0.7).abs() < 1e-5,
            "expected 0.7, got {}",
            output[2]
        );
    }

    #[test]
    fn layer_cache_integration() {
        let lc = Rc::new(RefCell::new(LayerCache::new(64 * 1024 * 1024)));

        // Pipeline 1: source → brightness(+0.25)
        let pipe1 = PipelineResource::new();
        pipe1.set_layer_cache(lc.clone());
        let source1 = SourceNode {
            pixels: vec![0.5, 0.3, 0.1, 1.0],
            info: NodeInfo { width: 1, height: 1, color_space: ColorSpace::Linear },
        };
        // Use add_node_with_hash to simulate read() with known hash
        let src_hash = v2::source_hash(b"test_image");
        let src_id1 = pipe1.graph.borrow_mut().add_node_with_hash(Box::new(source1), src_hash);

        let mut params = ParamMap::new();
        params.floats.insert("amount".into(), 0.25);
        let bright_id1 = pipe1.apply_filter(src_id1, "brightness", &params).unwrap();
        let output1 = pipe1.render(bright_id1).unwrap();

        // Check stats: all misses, entries stored
        let stats1 = lc.borrow().stats();
        assert!(stats1.entries > 0, "layer cache should have entries after first run");

        // Pipeline 2: same source + same filter → should get cache hits
        let pipe2 = PipelineResource::new();
        pipe2.set_layer_cache(lc.clone());
        let source2 = SourceNode {
            pixels: vec![0.5, 0.3, 0.1, 1.0],
            info: NodeInfo { width: 1, height: 1, color_space: ColorSpace::Linear },
        };
        let src_id2 = pipe2.graph.borrow_mut().add_node_with_hash(Box::new(source2), src_hash);

        let mut params2 = ParamMap::new();
        params2.floats.insert("amount".into(), 0.25);
        let bright_id2 = pipe2.apply_filter(src_id2, "brightness", &params2).unwrap();
        let output2 = pipe2.render(bright_id2).unwrap();

        // Same output
        assert_eq!(output1, output2, "cached result should match computed result");

        // Should have hits
        let stats2 = lc.borrow().stats();
        assert!(stats2.hits > 0, "layer cache should have hits on second run");
    }

    #[test]
    fn layer_cache_param_change_invalidates_downstream() {
        let lc = Rc::new(RefCell::new(LayerCache::new(64 * 1024 * 1024)));

        // Pipeline 1: source → brightness(+0.25) → brightness(+0.1)
        let pipe1 = PipelineResource::new();
        pipe1.set_layer_cache(lc.clone());
        let source1 = SourceNode {
            pixels: vec![0.5, 0.3, 0.1, 1.0],
            info: NodeInfo { width: 1, height: 1, color_space: ColorSpace::Linear },
        };
        let src_hash = v2::source_hash(b"test_image_2");
        let src_id = pipe1.graph.borrow_mut().add_node_with_hash(Box::new(source1), src_hash);

        let mut p1 = ParamMap::new();
        p1.floats.insert("amount".into(), 0.25);
        let b1 = pipe1.apply_filter(src_id, "brightness", &p1).unwrap();

        let mut p2 = ParamMap::new();
        p2.floats.insert("amount".into(), 0.1);
        let b2 = pipe1.apply_filter(b1, "brightness", &p2).unwrap();
        let _out1 = pipe1.render(b2).unwrap();

        let stats_after_first = lc.borrow().stats();
        let hits_after_first = stats_after_first.hits;

        // Pipeline 2: same source → brightness(+0.25) → brightness(+0.2) — last param changed
        let pipe2 = PipelineResource::new();
        pipe2.set_layer_cache(lc.clone());
        let source2 = SourceNode {
            pixels: vec![0.5, 0.3, 0.1, 1.0],
            info: NodeInfo { width: 1, height: 1, color_space: ColorSpace::Linear },
        };
        let src_id2 = pipe2.graph.borrow_mut().add_node_with_hash(Box::new(source2), src_hash);

        let mut p1b = ParamMap::new();
        p1b.floats.insert("amount".into(), 0.25);
        let b1b = pipe2.apply_filter(src_id2, "brightness", &p1b).unwrap();

        let mut p2b = ParamMap::new();
        p2b.floats.insert("amount".into(), 0.2); // Changed!
        let b2b = pipe2.apply_filter(b1b, "brightness", &p2b).unwrap();
        let _out2 = pipe2.render(b2b).unwrap();

        let stats_after_second = lc.borrow().stats();
        // The first brightness(+0.25) should be a cache hit (same upstream + same params)
        // The second brightness(+0.2) is a miss (different params from +0.1)
        assert!(
            stats_after_second.hits > hits_after_first,
            "should have at least one cache hit from unchanged upstream filter"
        );
    }

    #[test]
    fn stroke_input_round_trip() {
        let stroke = StrokeInputResource {
            points: RefCell::new(Vec::new()),
            brush_params: StrokeBrushParams {
                diameter: 20.0,
                spacing: 0.15,
                hardness: 0.8,
                flow: 1.0,
                opacity: 1.0,
                angle: 0.0,
                roundness: 1.0,
                scatter: 0.0,
                smoothing: 0.0,
            },
            ended: std::cell::Cell::new(false),
        };

        // Add 3 points
        {
            let mut pts = stroke.points.borrow_mut();
            // Point 1: (100, 200) pressure=0.5
            pts.extend_from_slice(&[100.0, 200.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]);
            // Point 2: (110, 205) pressure=0.7
            pts.extend_from_slice(&[110.0, 205.0, 0.7, 0.0, 0.0, 0.0, 1.0, 16.0]);
            // Point 3: (120, 210) pressure=0.3
            pts.extend_from_slice(&[120.0, 210.0, 0.3, 0.0, 0.0, 0.0, 1.0, 32.0]);
        }

        // Verify point count
        assert_eq!(stroke.points.borrow().len() / STROKE_POINT_STRIDE, 3);

        // Verify data round-trip
        let data = stroke.points.borrow().clone();
        assert_eq!(data.len(), 24); // 3 points * 8 floats
        assert_eq!(data[0], 100.0); // first point x
        assert_eq!(data[8], 110.0); // second point x
        assert_eq!(data[18], 0.3);  // third point pressure

        // Verify apply_stroke with passthrough
        let pipe = PipelineResource::new();
        let source = SourceNode {
            pixels: vec![0.5, 0.3, 0.1, 1.0],
            info: NodeInfo { width: 1, height: 1, color_space: ColorSpace::Linear },
        };
        let src_id = pipe.graph.borrow_mut().add_node(Box::new(source));
        let result_id = pipe.apply_stroke(src_id, &stroke, [1.0, 0.0, 0.0], "normal").unwrap();
        let output = pipe.render(result_id).unwrap();
        // Passthrough: output should match source
        assert_eq!(output, vec![0.5, 0.3, 0.1, 1.0]);
    }

}
