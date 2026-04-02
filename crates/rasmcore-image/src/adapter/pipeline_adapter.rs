//! WIT adapter for the pipeline resource.

use std::cell::RefCell;

use crate::bindings::exports::rasmcore::image::pipeline::{
    self, CacheStats, GuestImagePipeline, GuestLayerCache, LayerCacheBorrow, NodeId,
};
use crate::bindings::rasmcore::core::{errors::RasmcoreError, types};

use crate::domain;
use crate::domain::pipeline::graph::NodeGraph;
#[allow(unused_imports)]
use crate::domain::pipeline::nodes::{
    color, composite, filters, frame_source, precision, sink, source, transform,
};

use super::{to_domain_frame_selection, to_wit_error, to_wit_image_info};

// Import generated macros that provide pipeline filter + write + transform methods
include!(concat!(env!("OUT_DIR"), "/generated_pipeline_adapter.rs"));
include!(concat!(
    env!("OUT_DIR"),
    "/generated_pipeline_write_adapter.rs"
));
include!(concat!(
    env!("OUT_DIR"),
    "/generated_pipeline_transform_adapter.rs"
));

fn to_domain_png_filter_type_pipeline(
    f: Option<pipeline::PngFilterType>,
) -> crate::domain::encoder::png::PngFilterType {
    match f {
        None => crate::domain::encoder::png::PngFilterType::Adaptive,
        Some(pipeline::PngFilterType::NoFilter) => {
            crate::domain::encoder::png::PngFilterType::NoFilter
        }
        Some(pipeline::PngFilterType::Sub) => crate::domain::encoder::png::PngFilterType::Sub,
        Some(pipeline::PngFilterType::Up) => crate::domain::encoder::png::PngFilterType::Up,
        Some(pipeline::PngFilterType::Avg) => crate::domain::encoder::png::PngFilterType::Avg,
        Some(pipeline::PngFilterType::Paeth) => crate::domain::encoder::png::PngFilterType::Paeth,
        Some(pipeline::PngFilterType::Adaptive) => {
            crate::domain::encoder::png::PngFilterType::Adaptive
        }
    }
}

fn to_domain_tiff_compression_pipeline(
    c: Option<pipeline::TiffCompression>,
) -> crate::domain::encoder::tiff::TiffCompression {
    match c {
        None => crate::domain::encoder::tiff::TiffCompression::Lzw,
        Some(pipeline::TiffCompression::None) => {
            crate::domain::encoder::tiff::TiffCompression::None
        }
        Some(pipeline::TiffCompression::Lzw) => crate::domain::encoder::tiff::TiffCompression::Lzw,
        Some(pipeline::TiffCompression::Deflate) => {
            crate::domain::encoder::tiff::TiffCompression::Deflate
        }
        Some(pipeline::TiffCompression::Packbits) => {
            crate::domain::encoder::tiff::TiffCompression::PackBits
        }
    }
}

/// Pipeline resource implementation wrapping the domain NodeGraph.
/// Tracks metadata operations (keep/set/strip/load) for resolution at encode time.
#[derive(Default)]
struct MetadataOps {
    /// Source data bytes (for reading metadata from headers).
    source_data: Option<Vec<u8>>,
    /// Whether to carry source metadata through (opt-in via keepMetadata).
    keep: bool,
    /// Individual field overrides: (container, field, value).
    sets: Vec<(String, String, String)>,
    /// Field removals: (container, field). Applied after keep.
    strips: Vec<(String, String)>,
    /// Bulk load from JSON-shaped object: (container, field, value).
    loads: Vec<(String, String, String)>,
}

impl MetadataOps {
    /// Resolve all metadata operations into a final MetadataSet for encoding.
    ///
    /// Resolution order:
    /// 1. If keep=true, start with source metadata; otherwise start empty
    /// 2. Apply load entries (bulk set)
    /// 3. Apply individual set entries (override)
    /// 4. Apply strip entries (remove)
    fn resolve(&self) -> crate::domain::metadata::set::MetadataSet {
        use crate::domain::metadata;
        use crate::domain::metadata::set::MetadataSet;

        let mut ms = if self.keep {
            // Parse source metadata
            self.source_data
                .as_ref()
                .and_then(|data| metadata::read_metadata(data).ok())
                .unwrap_or_else(MetadataSet::new)
        } else {
            MetadataSet::new()
        };

        // Apply loads and sets by rebuilding the relevant container.
        // For now, loads and sets modify the ExifMetadata/XmpMetadata/IptcMetadata
        // structs and re-serialize. This is the simplest correct approach.

        // Collect all field operations (loads first, then sets to override)
        let mut exif_ops: Vec<(&str, &str)> = Vec::new();
        let mut xmp_ops: Vec<(&str, &str)> = Vec::new();
        let mut iptc_ops: Vec<(&str, &str)> = Vec::new();

        for (container, field, value) in self.loads.iter().chain(self.sets.iter()) {
            match container.as_str() {
                "exif" => exif_ops.push((field, value)),
                "xmp" => xmp_ops.push((field, value)),
                "iptc" => iptc_ops.push((field, value)),
                _ => {}
            }
        }

        // Apply EXIF operations
        if !exif_ops.is_empty() {
            let mut exif = ms
                .exif
                .as_ref()
                .and_then(|bytes| {
                    // Parse existing EXIF
                    let mut jpeg = vec![0xFF, 0xD8, 0xFF, 0xE1];
                    let len = (bytes.len() + 2) as u16;
                    jpeg.extend_from_slice(&len.to_be_bytes());
                    jpeg.extend_from_slice(bytes);
                    jpeg.extend_from_slice(&[0xFF, 0xD9]);
                    metadata::read_exif(&jpeg).ok()
                })
                .unwrap_or_default();

            for (field, value) in &exif_ops {
                match *field {
                    "Artist" | "Copyright" | "Software" | "ImageDescription" => {
                        // These go into the Software/Make/Model string fields
                        match *field {
                            "Software" => exif.software = Some(value.to_string()),
                            _ => {} // Other string EXIF fields require raw TIFF IFD editing
                        }
                    }
                    "Make" => exif.camera_make = Some(value.to_string()),
                    "Model" => exif.camera_model = Some(value.to_string()),
                    "DateTime" => exif.date_time = Some(value.to_string()),
                    _ => {}
                }
            }

            if let Ok(bytes) = metadata::write_exif(&exif) {
                ms.exif = Some(bytes);
            }
        }

        // Apply XMP operations
        if !xmp_ops.is_empty() {
            use crate::domain::metadata::xmp as metadata_xmp;
            let mut xmp = ms
                .xmp
                .as_ref()
                .and_then(|bytes| metadata_xmp::parse_xmp(bytes).ok())
                .unwrap_or_default();

            for (field, value) in &xmp_ops {
                match *field {
                    "Title" => xmp.title = Some(value.to_string()),
                    "Description" => xmp.description = Some(value.to_string()),
                    "Creator" => xmp.creator = Some(value.to_string()),
                    "Rights" => xmp.rights = Some(value.to_string()),
                    "CreateDate" => xmp.create_date = Some(value.to_string()),
                    "ModifyDate" => xmp.modify_date = Some(value.to_string()),
                    "CreatorTool" => xmp.creator_tool = Some(value.to_string()),
                    _ => {}
                }
            }

            if let Ok(bytes) = metadata_xmp::serialize_xmp(&xmp) {
                ms.xmp = Some(bytes);
            }
        }

        // Apply IPTC operations
        if !iptc_ops.is_empty() {
            use crate::domain::metadata::iptc as metadata_iptc;
            let mut iptc = ms
                .iptc
                .as_ref()
                .and_then(|bytes| metadata_iptc::parse_iptc(bytes).ok())
                .unwrap_or_default();

            for (field, value) in &iptc_ops {
                match *field {
                    "Title" | "Headline" => iptc.title = Some(value.to_string()),
                    "Caption" => iptc.caption = Some(value.to_string()),
                    "Byline" => iptc.byline = Some(value.to_string()),
                    "Copyright" => iptc.copyright = Some(value.to_string()),
                    "Category" => iptc.category = Some(value.to_string()),
                    "Keywords" => {
                        iptc.keywords = value.split(',').map(|s| s.trim().to_string()).collect();
                    }
                    _ => {}
                }
            }

            if let Ok(bytes) = metadata_iptc::serialize_iptc(&iptc) {
                ms.iptc = Some(bytes);
            }
        }

        // Apply strip operations
        for (container, field) in &self.strips {
            match container.as_str() {
                "exif" => {
                    // For strip, clear the entire EXIF container if stripping individual fields
                    // is too complex (raw TIFF IFD editing). Simple approach: if any EXIF field
                    // is stripped, remove the whole container if it's the only field.
                    // TODO: granular field removal requires TIFF IFD editing
                    if ms.exif.is_some() {
                        // For GPS fields, strip entire EXIF for now (privacy-safe)
                        if field.starts_with("GPS") {
                            ms.exif = None;
                        }
                    }
                }
                "xmp" => {
                    if ms.xmp.is_some() {
                        ms.xmp = None; // Strip entire XMP for now
                    }
                }
                "iptc" => {
                    if ms.iptc.is_some() {
                        ms.iptc = None;
                    }
                }
                "icc" => {
                    ms.icc_profile = None;
                }
                _ => {}
            }
        }

        ms
    }
}

// ─── Layer Cache Resource ────────────────────────────────────────────────

pub struct LayerCacheResource {
    inner: std::rc::Rc<RefCell<rasmcore_pipeline::LayerCache>>,
}

impl GuestLayerCache for LayerCacheResource {
    fn new(memory_budget_mb: u32) -> Self {
        let budget = memory_budget_mb as usize * 1024 * 1024;
        Self {
            inner: std::rc::Rc::new(RefCell::new(rasmcore_pipeline::LayerCache::new(budget))),
        }
    }

    fn stats(&self) -> CacheStats {
        let s = self.inner.borrow().stats();
        CacheStats {
            entries: s.entries as u64,
            hits: s.hits,
            misses: s.misses,
            size_bytes: s.size_bytes,
        }
    }

    fn clear(&self) {
        self.inner.borrow_mut().clear();
    }

    fn set_cache_quality(&self, quality: pipeline::CacheQuality) {
        let q = match quality {
            pipeline::CacheQuality::Full => rasmcore_pipeline::CacheQuality::Full,
            pipeline::CacheQuality::Q16 => rasmcore_pipeline::CacheQuality::Q16,
            pipeline::CacheQuality::Q8 => rasmcore_pipeline::CacheQuality::Q8,
        };
        self.inner.borrow_mut().set_cache_quality(q);
    }
}

// ─── Pipeline Resource ──────────────────────────────────────────────────

pub struct PipelineResource {
    graph: RefCell<NodeGraph>,
    /// Metadata operations, keyed by source node ID.
    metadata_ops: RefCell<MetadataOps>,
    /// Rc handle to frame source for sequence execution.
    frame_source: RefCell<Option<std::rc::Rc<frame_source::FrameSourceNode>>>,
    /// Optional layer cache for cross-pipeline result reuse.
    layer_cache: RefCell<Option<std::rc::Rc<RefCell<rasmcore_pipeline::LayerCache>>>>,
    /// Metadata filter for write operations (default: drop all).
    metadata_filter: RefCell<rasmcore_pipeline::MetadataFilter>,
    /// Script plugin registry (loaded via load-scripts WIT method).
    script_registry: RefCell<Option<domain::script_plugin::ScriptRegistry>>,
    /// Pipeline precision mode (Standard or HighPrecision).
    precision: std::cell::Cell<domain::pixel_sample::PipelinePrecision>,
    /// Proxy scale factor for automatic spatial parameter scaling.
    /// Default 1.0 = full resolution (no scaling). Values < 1.0 indicate
    /// working at reduced resolution; spatial params (hint = rc.pixels)
    /// are multiplied by this factor before reaching domain code.
    proxy_scale: std::cell::Cell<f32>,
}

impl PipelineResource {
    /// Finalize layer cache after any write operation, then clean up graph state.
    /// The layer cache itself is preserved — only transient graph nodes are cleared.
    fn finalize_cache(&self) {
        self.graph.borrow_mut().finalize_layer_cache();
        self.graph.borrow_mut().cleanup();
    }
}

impl GuestImagePipeline for PipelineResource {
    fn new() -> Self {
        Self {
            graph: RefCell::new(NodeGraph::new(16 * 1024 * 1024)), // 16MB cache budget
            metadata_ops: RefCell::new(MetadataOps::default()),
            frame_source: RefCell::new(None),
            layer_cache: RefCell::new(None),
            metadata_filter: RefCell::new(rasmcore_pipeline::MetadataFilter::DropAll),
            script_registry: RefCell::new(None),
            precision: std::cell::Cell::new(domain::pixel_sample::PipelinePrecision::Standard),
            proxy_scale: std::cell::Cell::new(1.0),
        }
    }

    fn set_precision(&self, precision: pipeline::PipelinePrecision) {
        let domain_precision = match precision {
            pipeline::PipelinePrecision::Standard => {
                domain::pixel_sample::PipelinePrecision::Standard
            }
            pipeline::PipelinePrecision::HalfPrecision => {
                domain::pixel_sample::PipelinePrecision::HalfPrecision
            }
            pipeline::PipelinePrecision::HighPrecision => {
                domain::pixel_sample::PipelinePrecision::HighPrecision
            }
        };
        self.precision.set(domain_precision);
    }

    fn set_proxy_scale(&self, scale: f32) {
        self.proxy_scale.set(scale.max(0.01)); // clamp to prevent zero/negative
    }

    fn set_layer_cache(&self, cache: LayerCacheBorrow<'_>) {
        let cache_resource = cache.get::<LayerCacheResource>();
        let lc = cache_resource.inner.clone();
        *self.layer_cache.borrow_mut() = Some(lc.clone());
        // Recreate graph with layer cache attached
        let mut graph = self.graph.borrow_mut();
        let new_graph = NodeGraph::with_layer_cache(16 * 1024 * 1024, lc);
        *graph = new_graph;
    }

    fn read(&self, data: Vec<u8>) -> Result<NodeId, RasmcoreError> {
        // Parse metadata immediately from container headers (before pixel decode)
        let mut meta = rasmcore_pipeline::Metadata::new();

        // Detect format
        if let Some(fmt) = domain::decoder::detect_format(&data) {
            meta.set(
                "format",
                rasmcore_pipeline::MetadataValue::String(fmt.clone()),
            );

            // Extract ICC profile
            let icc = match fmt.as_str() {
                "jpeg" | "jpg" => domain::color::extract_icc_from_jpeg(&data),
                "png" => domain::color::extract_icc_from_png(&data),
                _ => None,
            };
            if let Some(profile) = icc {
                meta.set(
                    "icc_profile",
                    rasmcore_pipeline::MetadataValue::Bytes(profile),
                );
            }
        }

        // Extract EXIF metadata (non-fatal — corrupt EXIF shouldn't block decode)
        if let Ok(exif) = domain::metadata::read_exif(&data) {
            if let Some(orient) = exif.orientation {
                meta.set(
                    "exif.Orientation",
                    rasmcore_pipeline::MetadataValue::Int(orient as i64),
                );
            }
            if let Some(ref make) = exif.camera_make {
                meta.set(
                    "exif.Make",
                    rasmcore_pipeline::MetadataValue::String(make.clone()),
                );
            }
            if let Some(ref model) = exif.camera_model {
                meta.set(
                    "exif.Model",
                    rasmcore_pipeline::MetadataValue::String(model.clone()),
                );
            }
            if let Some(ref software) = exif.software {
                meta.set(
                    "exif.Software",
                    rasmcore_pipeline::MetadataValue::String(software.clone()),
                );
            }
            if let Some(ref date) = exif.date_time {
                meta.set(
                    "exif.DateTime",
                    rasmcore_pipeline::MetadataValue::String(date.clone()),
                );
            }
            if let Some(w) = exif.width {
                meta.set("width", rasmcore_pipeline::MetadataValue::Int(w as i64));
            }
            if let Some(h) = exif.height {
                meta.set("height", rasmcore_pipeline::MetadataValue::Int(h as i64));
            }
        }

        // Still stash source data for legacy metadata operations
        self.metadata_ops.borrow_mut().source_data = Some(data.clone());

        // Create source node with content hash and metadata
        let source_hash = rasmcore_pipeline::compute_source_hash(&data);
        let node = source::SourceNode::new(data).map_err(to_wit_error)?;
        let src_info = <source::SourceNode as domain::pipeline::graph::ImageNode>::info(&node);
        let source_id =
            self.graph
                .borrow_mut()
                .add_source_node(Box::new(node), source_hash, meta, "source");

        // f32 pipeline: promote to Rgba32f immediately after decode.
        // All downstream nodes see Rgba32f — no format branching needed.
        if src_info.format != domain::types::PixelFormat::Rgba32f {
            let promote = precision::PromoteNode::new(source_id, src_info);
            let promote_hash = rasmcore_pipeline::compute_hash(
                &self.graph.borrow().node_hash(source_id),
                "promote_f32",
                &[],
            );
            let id = self.graph.borrow_mut().add_node_described(
                Box::new(promote),
                promote_hash,
                source_id,
                domain::pipeline::graph::NodeKind::Filter,
                "promote_f32",
            );
            Ok(id)
        } else {
            Ok(source_id)
        }
    }

    fn read_frames(
        &self,
        data: Vec<u8>,
        selection: pipeline::FrameSelection,
    ) -> Result<NodeId, RasmcoreError> {
        let domain_sel = to_domain_frame_selection(selection);
        let node = frame_source::FrameSourceNode::new(data, domain_sel).map_err(to_wit_error)?;
        let (id, rc) = self.graph.borrow_mut().add_frame_source(node);
        // Store the Rc handle for later use by execute_sequence
        *self.frame_source.borrow_mut() = Some(rc);
        Ok(id)
    }

    fn node_info(&self, node: NodeId) -> Result<types::ImageInfo, RasmcoreError> {
        let info = self.graph.borrow().node_info(node).map_err(to_wit_error)?;
        Ok(to_wit_image_info(&info))
    }

    // Auto-generated transform methods (resize, crop, rotate, flip, auto_orient, icc_to_srgb)
    generated_pipeline_transform_methods!();

    fn convert_format(
        &self,
        _source: NodeId,
        _target: types::PixelFormat,
    ) -> Result<NodeId, RasmcoreError> {
        Err(RasmcoreError::NotImplemented)
    }

    // Auto-generated pipeline filter methods (all registered filters)
    generated_pipeline_filter_methods!();

    // Auto-generated write methods for all registered encoders
    generated_pipeline_write_methods!();

    fn write(
        &self,
        source: NodeId,
        format: String,
        quality: Option<u8>,
        metadata: Option<pipeline::MetadataSet>,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let domain_meta = metadata.as_ref().map(super::to_domain_metadata_set);
        let result = sink::write(
            &mut self.graph.borrow_mut(),
            source,
            &format,
            quality,
            domain_meta.as_ref(),
        )
        .map_err(to_wit_error);
        self.finalize_cache();
        result
    }

    // ─── Multi-Frame ───

    fn execute_sequence(&self, _source: NodeId, output: NodeId) -> Result<u32, RasmcoreError> {
        let frame_src = self.frame_source.borrow();
        let rc = frame_src.as_ref().ok_or_else(|| {
            RasmcoreError::InvalidInput(
                "no frame source: call read_frames() before execute_sequence()".into(),
            )
        })?;
        let seq = self
            .graph
            .borrow_mut()
            .execute_sequence(rc, output)
            .map_err(to_wit_error)?;
        let count = seq.len() as u32;
        // Store the sequence for later retrieval by multi-frame encode (track 2)
        // For now, just return the count — the FrameSequence is consumed.
        Ok(count)
    }

    // ─── Metadata ───

    fn metadata_dump(&self) -> Result<String, RasmcoreError> {
        let ops = self.metadata_ops.borrow();
        let ms = ops.resolve();
        match ops.source_data.as_ref() {
            Some(data) => Ok(domain::metadata::query::metadata_dump_json_from_bytes(
                data, &ms,
            )),
            None => Ok(domain::metadata::query::metadata_dump_json(&ms)),
        }
    }

    fn metadata_read(&self, path: String) -> Result<Option<String>, RasmcoreError> {
        let ops = self.metadata_ops.borrow();
        let ms = ops.resolve();
        match ops.source_data.as_ref() {
            Some(data) => domain::metadata::query::metadata_read_from_bytes(data, &ms, &path)
                .map_err(to_wit_error),
            None => domain::metadata::query::metadata_read(&ms, &path).map_err(to_wit_error),
        }
    }

    fn keep_metadata(&self) -> Result<NodeId, RasmcoreError> {
        self.metadata_ops.borrow_mut().keep = true;
        *self.metadata_filter.borrow_mut() = rasmcore_pipeline::MetadataFilter::KeepAll;
        Ok(0)
    }

    fn node_metadata_dump(&self, node: NodeId) -> Result<String, RasmcoreError> {
        let graph = self.graph.borrow();
        let meta = graph.node_metadata(node);
        Ok(meta.to_json())
    }

    fn node_metadata_read(
        &self,
        node: NodeId,
        key: String,
    ) -> Result<Option<String>, RasmcoreError> {
        let graph = self.graph.borrow();
        let meta = graph.node_metadata(node);
        Ok(meta.get(&key).map(|v| match v {
            rasmcore_pipeline::MetadataValue::String(s) => s.clone(),
            rasmcore_pipeline::MetadataValue::Int(i) => i.to_string(),
            rasmcore_pipeline::MetadataValue::Float(f) => f.to_string(),
            rasmcore_pipeline::MetadataValue::Bool(b) => b.to_string(),
            rasmcore_pipeline::MetadataValue::Bytes(b) => format!("<{} bytes>", b.len()),
        }))
    }

    fn include_metadata(&self, patterns: Vec<String>) {
        *self.metadata_filter.borrow_mut() = rasmcore_pipeline::MetadataFilter::Include(patterns);
    }

    fn exclude_metadata(&self, patterns: Vec<String>) {
        *self.metadata_filter.borrow_mut() = rasmcore_pipeline::MetadataFilter::Exclude(patterns);
    }

    fn set_metadata(&self, path: String, value: String) -> Result<NodeId, RasmcoreError> {
        let parts: Vec<&str> = path.splitn(2, '.').collect();
        if parts.len() != 2 {
            return Err(RasmcoreError::InvalidInput(format!(
                "Invalid metadata path \"{path}\". Expected format: container.Field"
            )));
        }
        self.metadata_ops.borrow_mut().sets.push((
            parts[0].to_string(),
            parts[1].to_string(),
            value,
        ));
        Ok(0)
    }

    fn load_metadata(&self, json: String) -> Result<NodeId, RasmcoreError> {
        // Parse JSON object: {"container": {"field": "value", ...}, ...}
        // and push each entry into loads.
        let mut ops = self.metadata_ops.borrow_mut();
        // Simple JSON parsing: iterate the resolved entries.
        // For now, just store the raw JSON and let resolve() handle it.
        // Actually, parse with the simple parser we have.
        let entries = parse_metadata_json(&json)
            .map_err(|e| RasmcoreError::InvalidInput(format!("Invalid metadata JSON: {e}")))?;
        for (container, field, value) in entries {
            ops.loads.push((container, field, value));
        }
        Ok(0)
    }

    fn strip_metadata(&self, path: String) -> Result<NodeId, RasmcoreError> {
        let parts: Vec<&str> = path.splitn(2, '.').collect();
        if parts.len() != 2 {
            return Err(RasmcoreError::InvalidInput(format!(
                "Invalid metadata path \"{path}\". Expected format: container.Field"
            )));
        }
        self.metadata_ops
            .borrow_mut()
            .strips
            .push((parts[0].to_string(), parts[1].to_string()));
        Ok(0)
    }

    // ─── Script Plugins ───

    fn load_scripts(&self, sources: Vec<String>) -> Result<(), RasmcoreError> {
        let registry = domain::script_plugin::ScriptRegistry::new(&sources)
            .map_err(|e| to_wit_error(domain::error::ImageError::ScriptError(e)))?;
        *self.script_registry.borrow_mut() = Some(registry);
        Ok(())
    }

    fn apply_script_filter(
        &self,
        name: String,
        source_node: NodeId,
        config: Vec<(String, String)>,
    ) -> Result<NodeId, RasmcoreError> {
        let registry_ref = self.script_registry.borrow();
        let registry = registry_ref.as_ref().ok_or_else(|| {
            to_wit_error(domain::error::ImageError::ScriptError(
                "no scripts loaded — call load-scripts first".to_string(),
            ))
        })?;

        let info = {
            let graph = self.graph.borrow();
            graph.node_info(source_node).map_err(|e| to_wit_error(e))?
        };

        let params: std::collections::HashMap<String, String> = config.into_iter().collect();
        let node = domain::script_plugin::dispatch_script_filter(
            registry,
            &name,
            source_node,
            info,
            &params,
        )
        .map_err(|e| to_wit_error(domain::error::ImageError::ScriptError(e)))?;

        let id = self.graph.borrow_mut().add_node(node);
        Ok(id)
    }

    fn list_script_filters(&self) -> Vec<String> {
        self.script_registry
            .borrow()
            .as_ref()
            .map(|r| r.list().into_iter().map(|s| s.to_string()).collect())
            .unwrap_or_default()
    }

    // ─── Graph Description ───

    fn graph_describe(&self) -> String {
        self.graph.borrow().description().to_json()
    }

    fn graph_serialize(&self) -> Vec<u8> {
        self.graph.borrow().description().serialize()
    }

    fn graph_node_count(&self) -> u32 {
        self.graph.borrow().description().len() as u32
    }

    fn write_from_description(
        &self,
        description: Vec<u8>,
        source_data: Vec<u8>,
        terminal_node: NodeId,
        format: String,
        quality: Option<u8>,
    ) -> Result<Vec<u8>, RasmcoreError> {
        use crate::domain::pipeline::graph::{GraphDescription, execute_from_description};

        let desc = GraphDescription::deserialize(&description).map_err(to_wit_error)?;
        execute_from_description(&desc, &source_data, terminal_node, &format, quality)
            .map_err(to_wit_error)
    }
}

/// Dispatch a chain operation by name to the appropriate pipeline method.
///
/// Operations with no config use an empty params list.
/// Operations with config map params positionally to config fields (f32).
/// The filter manifest documents parameter order for each operation.
pub fn dispatch_chain_op(
    pipe: &PipelineResource,
    source: NodeId,
    name: &str,
    params: &[f32],
) -> Result<NodeId, RasmcoreError> {
    use crate::bindings::exports::rasmcore::image::pipeline::{self as p, GuestImagePipeline as P};

    // Helper to get param at index or return default
    let f = |i: usize, default: f32| -> f32 { params.get(i).copied().unwrap_or(default) };

    match name {
        // ─── Transforms ───
        "flip" => {
            // params[0]: 0.0 = horizontal (default), 1.0 = vertical
            let dir = if f(0, 0.0) > 0.5 {
                p::FlipDirection::Vertical
            } else {
                p::FlipDirection::Horizontal
            };
            P::flip(pipe, source, p::FlipConfig { direction: dir })
        }
        "rotate" => P::rotate(
            pipe,
            source,
            p::RotateConfig {
                rotation: match f(0, 90.0) as u32 {
                    180 => p::Rotation::R180,
                    270 => p::Rotation::R270,
                    _ => p::Rotation::R90,
                },
            },
        ),
        "resize" => P::resize(
            pipe,
            source,
            p::ResizeConfig {
                width: f(0, 0.0) as u32,
                height: f(1, 0.0) as u32,
                filter: p::ResizeFilter::Lanczos3,
            },
        ),
        "crop" => P::crop(
            pipe,
            source,
            p::CropConfig {
                x: f(0, 0.0) as u32,
                y: f(1, 0.0) as u32,
                width: f(2, 0.0) as u32,
                height: f(3, 0.0) as u32,
            },
        ),

        // ─── No-config filters ───
        "invert" => P::invert(pipe, source),
        "equalize" => P::equalize(pipe, source),
        "normalize" => P::normalize(pipe, source),
        "auto_level" | "auto-level" => P::auto_level(pipe, source),
        "emboss" => P::emboss(pipe, source),
        "tonemap_reinhard" | "tonemap-reinhard" => P::tonemap_reinhard(pipe, source),
        "white_balance_gray_world" | "white-balance-gray-world" => {
            P::white_balance_gray_world(pipe, source)
        }
        "depolar" => P::depolar(pipe, source),
        "polar" => P::polar(pipe, source),
        "premultiply" => P::premultiply(pipe, source),
        "unpremultiply" => P::unpremultiply(pipe, source),
        "triangle_threshold" | "triangle-threshold" => P::triangle_threshold(pipe, source),
        "otsu_threshold" | "otsu-threshold" => P::otsu_threshold(pipe, source),
        "evaluate_abs" | "evaluate-abs" => P::evaluate_abs(pipe, source),
        "average_blur" | "average-blur" => P::average_blur(pipe, source),

        // ─── Parameterized filters ───
        "blur" => P::blur(pipe, source, p::BlurConfig { radius: f(0, 3.0) }),
        "brightness" => P::brightness(pipe, source, p::BrightnessConfig { amount: f(0, 0.0) }),
        "contrast" => P::contrast(pipe, source, p::ContrastConfig { amount: f(0, 0.0) }),
        "gamma" => P::gamma(
            pipe,
            source,
            p::GammaConfig {
                gamma_value: f(0, 1.0),
            },
        ),
        "exposure" => P::exposure(
            pipe,
            source,
            p::ExposureConfig {
                ev: f(0, 0.0),
                offset: f(1, 0.0),
                gamma_correction: f(2, 1.0),
            },
        ),
        "sepia" => P::sepia(
            pipe,
            source,
            p::SepiaConfig {
                intensity: f(0, 1.0),
            },
        ),
        "saturate" => P::saturate(pipe, source, p::SaturateConfig { factor: f(0, 1.0) }),
        "hue_rotate" | "hue-rotate" => {
            P::hue_rotate(pipe, source, p::HueRotateConfig { degrees: f(0, 0.0) })
        }
        "vibrance" => P::vibrance(pipe, source, p::VibranceConfig { amount: f(0, 0.0) }),
        "sharpen" => P::sharpen(pipe, source, p::SharpenConfig { amount: f(0, 1.0) }),
        "solarize" => P::solarize(
            pipe,
            source,
            p::SolarizeConfig {
                threshold: f(0, 128.0) as u8,
            },
        ),
        "posterize" => P::posterize(
            pipe,
            source,
            p::PosterizeConfig {
                levels: f(0, 4.0) as u8,
            },
        ),
        "pixelate" => P::pixelate(
            pipe,
            source,
            p::PixelateConfig {
                block_size: f(0, 10.0) as u32,
            },
        ),
        "oil_paint" | "oil-paint" => P::oil_paint(
            pipe,
            source,
            p::OilPaintConfig {
                radius: f(0, 4.0) as u32,
            },
        ),
        "median" => P::median(
            pipe,
            source,
            p::MedianConfig {
                radius: f(0, 3.0) as u32,
            },
        ),
        "box_blur" | "box-blur" => P::box_blur(
            pipe,
            source,
            p::BoxBlurConfig {
                radius: f(0, 3.0) as u32,
            },
        ),
        "motion_blur" | "motion-blur" => P::motion_blur(
            pipe,
            source,
            p::MotionBlurConfig {
                length: f(0, 10.0) as u32,
                angle_degrees: f(1, 0.0),
            },
        ),
        "spin_blur" | "spin-blur" => P::spin_blur(
            pipe,
            source,
            p::SpinBlurConfig {
                center_x: f(0, 0.5),
                center_y: f(1, 0.5),
                angle: f(2, 10.0),
            },
        ),
        "spherize" => P::spherize(pipe, source, p::SpherizeConfig { amount: f(0, 1.0) }),
        "swirl" => P::swirl(
            pipe,
            source,
            p::SwirlConfig {
                angle: f(0, 1.0),
                radius: f(1, 0.5),
            },
        ),
        "ripple" => P::ripple(
            pipe,
            source,
            p::RippleConfig {
                amplitude: f(0, 10.0),
                wavelength: f(1, 50.0),
                center_x: f(2, 0.5),
                center_y: f(3, 0.5),
            },
        ),
        "wave" => P::wave(
            pipe,
            source,
            p::WaveConfig {
                amplitude: f(0, 10.0),
                wavelength: f(1, 50.0),
                vertical: f(2, 0.0),
            },
        ),
        "bilateral" => P::bilateral(
            pipe,
            source,
            p::BilateralConfig {
                diameter: f(0, 9.0) as u32,
                sigma_color: f(1, 75.0),
                sigma_space: f(2, 75.0),
            },
        ),
        "gaussian_noise" | "gaussian-noise" => P::gaussian_noise(
            pipe,
            source,
            p::GaussianNoiseConfig {
                amount: f(0, 25.0),
                mean: f(1, 0.0),
                sigma: f(2, 25.0),
                seed: params.get(3).map_or(0, |&v| v as u64),
            },
        ),
        "film_grain" | "film-grain" => P::film_grain(
            pipe,
            source,
            p::FilmGrainConfig {
                amount: f(0, 25.0),
                size: f(1, 1.0),
                seed: params.get(2).map_or(0, |&v| v as u32),
            },
        ),
        "threshold_binary" | "threshold-binary" => P::threshold_binary(
            pipe,
            source,
            p::ThresholdBinaryConfig {
                thresh: f(0, 128.0) as u8,
                max_value: f(1, 255.0) as u8,
            },
        ),
        "halftone" => P::halftone(
            pipe,
            source,
            p::HalftoneConfig {
                dot_size: f(0, 4.0),
                angle_offset: f(1, 45.0),
            },
        ),

        // Unknown op — return helpful error
        _ => Err(RasmcoreError::InvalidInput(format!(
            "unknown chain operation: \"{name}\". Use get-filter-manifest() for available operations."
        ))),
    }
}

/// Parse a simple JSON object of the form `{"container": {"field": "value", ...}, ...}`
/// into a flat list of (container, field, value) triples.
fn parse_metadata_json(json: &str) -> Result<Vec<(String, String, String)>, String> {
    let json = json.trim();
    if !json.starts_with('{') || !json.ends_with('}') {
        return Err("expected JSON object".into());
    }
    let mut entries = Vec::new();
    // Use the domain metadata_query's dump_json format as reference:
    // {"exif":{"Artist":"...", ...}, "xmp":{...}}
    // Minimal parser: split on container objects.
    let inner = &json[1..json.len() - 1];
    let mut chars = inner.chars().peekable();
    loop {
        skip_ws(&mut chars);
        if chars.peek().is_none() {
            break;
        }
        let container = parse_json_string(&mut chars)?;
        skip_ws(&mut chars);
        expect_char(&mut chars, ':')?;
        skip_ws(&mut chars);
        expect_char(&mut chars, '{')?;
        loop {
            skip_ws(&mut chars);
            if chars.peek() == Some(&'}') {
                chars.next();
                break;
            }
            let field = parse_json_string(&mut chars)?;
            skip_ws(&mut chars);
            expect_char(&mut chars, ':')?;
            skip_ws(&mut chars);
            let value = parse_json_string(&mut chars)?;
            entries.push((container.clone(), field, value));
            skip_ws(&mut chars);
            if chars.peek() == Some(&',') {
                chars.next();
            }
        }
        skip_ws(&mut chars);
        if chars.peek() == Some(&',') {
            chars.next();
        }
    }
    Ok(entries)
}

fn skip_ws(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) {
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
        } else {
            break;
        }
    }
}

fn expect_char(
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
    expected: char,
) -> Result<(), String> {
    match chars.next() {
        Some(c) if c == expected => Ok(()),
        Some(c) => Err(format!("expected '{expected}', got '{c}'")),
        None => Err(format!("expected '{expected}', got EOF")),
    }
}

fn parse_json_string(
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
) -> Result<String, String> {
    match chars.next() {
        Some('"') => {}
        Some(c) => return Err(format!("expected '\"', got '{c}'")),
        None => return Err("expected string, got EOF".into()),
    }
    let mut s = String::new();
    loop {
        match chars.next() {
            Some('"') => return Ok(s),
            Some('\\') => match chars.next() {
                Some(c @ ('"' | '\\' | '/')) => s.push(c),
                Some('n') => s.push('\n'),
                Some('t') => s.push('\t'),
                Some('r') => s.push('\r'),
                Some(c) => {
                    s.push('\\');
                    s.push(c);
                }
                None => return Err("unexpected EOF in escape".into()),
            },
            Some(c) => s.push(c),
            None => return Err("unterminated string".into()),
        }
    }
}
