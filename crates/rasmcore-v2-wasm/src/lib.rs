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
#[allow(unused_imports)]
use rasmcore_codecs_v2 as _v2_codecs;

#[cfg(target_arch = "wasm32")]
mod bindings;

#[cfg(target_arch = "wasm32")]
use bindings::exports::rasmcore::v2_image::pipeline_v2 as wit;

#[cfg(target_arch = "wasm32")]
use bindings::rasmcore::core::errors::RasmcoreError;

use std::cell::RefCell;

use rasmcore_pipeline_v2::{
    self as v2, ColorSpace, Graph, NodeInfo, ParamMap, PipelineError,
    create_filter_node,
};

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
        _ => RasmcoreError::InvalidInput(format!("{e}")),
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

// ─── Pipeline Resource ──────────────────────────────────────────────────────

/// V2 pipeline resource — wraps a V2 Graph exclusively.
pub struct PipelineResource {
    graph: RefCell<Graph>,
}

impl PipelineResource {
    pub fn new() -> Self {
        Self {
            graph: RefCell::new(Graph::new(16 * 1024 * 1024)),
        }
    }

    pub fn read(&self, data: &[u8], format_hint: Option<&str>) -> Result<u32, PipelineError> {
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

        let id = self.graph.borrow_mut().add_node(Box::new(source));
        Ok(id)
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

        let node = create_filter_node(name, source, info, params).ok_or_else(|| {
            PipelineError::InvalidParams(format!("unknown filter: {name}"))
        })?;

        let id = self.graph.borrow_mut().add_node(node);
        Ok(id)
    }

    pub fn write(
        &self,
        node_id: u32,
        format: &str,
        quality: Option<u8>,
    ) -> Result<Vec<u8>, PipelineError> {
        let pixels = self.graph.borrow_mut().request_full(node_id)?;
        let info = self.graph.borrow().node_info(node_id)?;

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

    pub fn render(&self, node_id: u32) -> Result<Vec<f32>, PipelineError> {
        self.graph.borrow_mut().request_full(node_id)
    }
}

// ─── WIT Bindings ���──────────────────────────────────────────────────────────

#[cfg(target_arch = "wasm32")]
struct Component;

#[cfg(target_arch = "wasm32")]
bindings::export!(Component with_types_in bindings);

#[cfg(target_arch = "wasm32")]
impl wit::Guest for Component {
    type ImagePipelineV2 = PipelineResource;
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

    fn set_demand_strategy(&self, _strategy: wit::DemandStrategy) {}

    fn set_gpu_available(&self, _available: bool) {}

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
}

// ─── Param deserialization ──────────────────────────────────────────────────

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
}
