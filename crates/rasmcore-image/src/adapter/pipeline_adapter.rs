//! WIT adapter for the pipeline resource.

use std::cell::RefCell;

use crate::bindings::exports::rasmcore::image::pipeline::{
    self, BlendMode as WitBlendMode, ExifOrientation, FlipDirection, GuestImagePipeline, NodeId,
    ResizeFilter, Rotation,
};
use crate::bindings::rasmcore::core::{errors::RasmcoreError, types};

use crate::domain;
use crate::domain::pipeline::graph::NodeGraph;
use crate::domain::pipeline::nodes::{color, composite, filters, frame_source, sink, source, transform};

use super::{to_domain_frame_selection, to_wit_error, to_wit_image_info};

fn to_domain_png_filter_pipeline(
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
    fn resolve(&self) -> crate::domain::metadata_set::MetadataSet {
        use crate::domain::metadata;
        use crate::domain::metadata_set::MetadataSet;

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
            use crate::domain::metadata_xmp;
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
            use crate::domain::metadata_iptc;
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

pub struct PipelineResource {
    graph: RefCell<NodeGraph>,
    /// Metadata operations, keyed by source node ID.
    metadata_ops: RefCell<MetadataOps>,
    /// Rc handle to frame source for sequence execution.
    frame_source: RefCell<Option<std::rc::Rc<frame_source::FrameSourceNode>>>,
}

impl GuestImagePipeline for PipelineResource {
    fn new() -> Self {
        Self {
            graph: RefCell::new(NodeGraph::new(16 * 1024 * 1024)), // 16MB cache budget
            metadata_ops: RefCell::new(MetadataOps::default()),
            frame_source: RefCell::new(None),
        }
    }

    fn read(&self, data: Vec<u8>) -> Result<NodeId, RasmcoreError> {
        // Capture source data for metadata operations
        self.metadata_ops.borrow_mut().source_data = Some(data.clone());
        let node = source::SourceNode::new(data).map_err(to_wit_error)?;
        let id = self.graph.borrow_mut().add_node(Box::new(node));
        Ok(id)
    }

    fn read_frames(
        &self,
        data: Vec<u8>,
        selection: pipeline::FrameSelection,
    ) -> Result<NodeId, RasmcoreError> {
        let domain_sel = to_domain_frame_selection(selection);
        let node =
            frame_source::FrameSourceNode::new(data, domain_sel).map_err(to_wit_error)?;
        let (id, rc) = self.graph.borrow_mut().add_frame_source(node);
        // Store the Rc handle for later use by execute_sequence
        *self.frame_source.borrow_mut() = Some(rc);
        Ok(id)
    }

    fn node_info(&self, node: NodeId) -> Result<types::ImageInfo, RasmcoreError> {
        let info = self.graph.borrow().node_info(node).map_err(to_wit_error)?;
        Ok(to_wit_image_info(&info))
    }

    fn resize(
        &self,
        source: NodeId,
        width: u32,
        height: u32,
        filter: ResizeFilter,
    ) -> Result<NodeId, RasmcoreError> {
        let src_info = self
            .graph
            .borrow()
            .node_info(source)
            .map_err(to_wit_error)?;
        let domain_filter = match filter {
            ResizeFilter::Nearest => domain::types::ResizeFilter::Nearest,
            ResizeFilter::Bilinear => domain::types::ResizeFilter::Bilinear,
            ResizeFilter::Bicubic => domain::types::ResizeFilter::Bicubic,
            ResizeFilter::Lanczos3 => domain::types::ResizeFilter::Lanczos3,
        };
        let node = transform::ResizeNode::new(source, src_info, width, height, domain_filter);
        Ok(self.graph.borrow_mut().add_node(Box::new(node)))
    }

    fn crop(
        &self,
        source: NodeId,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> Result<NodeId, RasmcoreError> {
        let src_info = self
            .graph
            .borrow()
            .node_info(source)
            .map_err(to_wit_error)?;
        let node = transform::CropNode::new(source, src_info, x, y, width, height);
        Ok(self.graph.borrow_mut().add_node(Box::new(node)))
    }

    fn rotate(&self, source: NodeId, angle: Rotation) -> Result<NodeId, RasmcoreError> {
        let src_info = self
            .graph
            .borrow()
            .node_info(source)
            .map_err(to_wit_error)?;
        let domain_rot = match angle {
            Rotation::R90 => domain::types::Rotation::R90,
            Rotation::R180 => domain::types::Rotation::R180,
            Rotation::R270 => domain::types::Rotation::R270,
        };
        let node = transform::RotateNode::new(source, src_info, domain_rot);
        Ok(self.graph.borrow_mut().add_node(Box::new(node)))
    }

    fn flip(&self, source: NodeId, direction: FlipDirection) -> Result<NodeId, RasmcoreError> {
        let src_info = self
            .graph
            .borrow()
            .node_info(source)
            .map_err(to_wit_error)?;
        let domain_dir = match direction {
            FlipDirection::Horizontal => domain::types::FlipDirection::Horizontal,
            FlipDirection::Vertical => domain::types::FlipDirection::Vertical,
        };
        let node = transform::FlipNode::new(source, src_info, domain_dir);
        Ok(self.graph.borrow_mut().add_node(Box::new(node)))
    }

    fn convert_format(
        &self,
        _source: NodeId,
        _target: types::PixelFormat,
    ) -> Result<NodeId, RasmcoreError> {
        Err(RasmcoreError::NotImplemented)
    }

    fn icc_to_srgb(&self, source: NodeId, icc_profile: Vec<u8>) -> Result<NodeId, RasmcoreError> {
        let src_info = self
            .graph
            .borrow()
            .node_info(source)
            .map_err(to_wit_error)?;
        let node =
            color::IccToSrgbNode::new(source, src_info, icc_profile).map_err(to_wit_error)?;
        Ok(self.graph.borrow_mut().add_node(Box::new(node)))
    }

    fn auto_orient(
        &self,
        source: NodeId,
        orientation: ExifOrientation,
    ) -> Result<NodeId, RasmcoreError> {
        let src_info = self
            .graph
            .borrow()
            .node_info(source)
            .map_err(to_wit_error)?;
        let domain_orient = match orientation {
            ExifOrientation::Normal => domain::metadata::ExifOrientation::Normal,
            ExifOrientation::FlipHorizontal => domain::metadata::ExifOrientation::FlipHorizontal,
            ExifOrientation::Rotate180 => domain::metadata::ExifOrientation::Rotate180,
            ExifOrientation::FlipVertical => domain::metadata::ExifOrientation::FlipVertical,
            ExifOrientation::Transpose => domain::metadata::ExifOrientation::Transpose,
            ExifOrientation::Rotate90 => domain::metadata::ExifOrientation::Rotate90,
            ExifOrientation::Transverse => domain::metadata::ExifOrientation::Transverse,
            ExifOrientation::Rotate270 => domain::metadata::ExifOrientation::Rotate270,
        };
        let node = transform::AutoOrientNode::new(source, src_info, domain_orient);
        Ok(self.graph.borrow_mut().add_node(Box::new(node)))
    }


        // Auto-generated pipeline filter methods (all registered filters)
        include!(concat!(env!("OUT_DIR"), "/generated_pipeline_adapter.rs"));


    fn composite(
        &self,
        fg: NodeId,
        bg: NodeId,
        x: i32,
        y: i32,
        mode: Option<WitBlendMode>,
    ) -> Result<NodeId, RasmcoreError> {
        let graph = self.graph.borrow();
        let fg_info = graph.node_info(fg).map_err(to_wit_error)?;
        let bg_info = graph.node_info(bg).map_err(to_wit_error)?;
        drop(graph);
        let domain_mode = mode.map(|m| match m {
            WitBlendMode::Multiply => domain::filters::BlendMode::Multiply,
            WitBlendMode::Screen => domain::filters::BlendMode::Screen,
            WitBlendMode::Overlay => domain::filters::BlendMode::Overlay,
            WitBlendMode::Darken => domain::filters::BlendMode::Darken,
            WitBlendMode::Lighten => domain::filters::BlendMode::Lighten,
            WitBlendMode::SoftLight => domain::filters::BlendMode::SoftLight,
            WitBlendMode::HardLight => domain::filters::BlendMode::HardLight,
            WitBlendMode::Difference => domain::filters::BlendMode::Difference,
            WitBlendMode::Exclusion => domain::filters::BlendMode::Exclusion,
            WitBlendMode::ColorDodge => domain::filters::BlendMode::ColorDodge,
            WitBlendMode::ColorBurn => domain::filters::BlendMode::ColorBurn,
            WitBlendMode::VividLight => domain::filters::BlendMode::VividLight,
            WitBlendMode::LinearDodge => domain::filters::BlendMode::LinearDodge,
            WitBlendMode::LinearBurn => domain::filters::BlendMode::LinearBurn,
            WitBlendMode::LinearLight => domain::filters::BlendMode::LinearLight,
            WitBlendMode::PinLight => domain::filters::BlendMode::PinLight,
            WitBlendMode::HardMix => domain::filters::BlendMode::HardMix,
            WitBlendMode::Subtract => domain::filters::BlendMode::Subtract,
            WitBlendMode::Divide => domain::filters::BlendMode::Divide,
        });
        let node = composite::CompositeNode::new(fg, bg, fg_info, bg_info, x, y, domain_mode);
        Ok(self.graph.borrow_mut().add_node(Box::new(node)))
    }

    fn write_jpeg(
        &self,
        source: NodeId,
        config: pipeline::JpegWriteConfig,
        metadata: Option<pipeline::MetadataSet>,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let cfg = domain::encoder::jpeg::JpegEncodeConfig {
            quality: config.quality.unwrap_or(85),
            progressive: config.progressive.unwrap_or(false),
            turbo: false,
        };
        let domain_meta = metadata.as_ref().map(super::to_domain_metadata_set);
        sink::write_jpeg(
            &mut self.graph.borrow_mut(),
            source,
            &cfg,
            domain_meta.as_ref(),
        )
        .map_err(to_wit_error)
    }

    fn write_png(
        &self,
        source: NodeId,
        config: pipeline::PngWriteConfig,
        metadata: Option<pipeline::MetadataSet>,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let cfg = domain::encoder::png::PngEncodeConfig {
            compression_level: config.compression_level.unwrap_or(6),
            filter_type: to_domain_png_filter_pipeline(config.filter_type),
        };
        let domain_meta = metadata.as_ref().map(super::to_domain_metadata_set);
        sink::write_png(
            &mut self.graph.borrow_mut(),
            source,
            &cfg,
            domain_meta.as_ref(),
        )
        .map_err(to_wit_error)
    }

    fn write_webp(
        &self,
        source: NodeId,
        config: pipeline::WebpWriteConfig,
        metadata: Option<pipeline::MetadataSet>,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let cfg = domain::encoder::webp::WebpEncodeConfig {
            quality: config.quality.unwrap_or(75),
            lossless: config.lossless.unwrap_or(false),
        };
        let domain_meta = metadata.as_ref().map(super::to_domain_metadata_set);
        sink::write_webp(
            &mut self.graph.borrow_mut(),
            source,
            &cfg,
            domain_meta.as_ref(),
        )
        .map_err(to_wit_error)
    }

    fn write_bmp(
        &self,
        source: NodeId,
        _config: pipeline::BmpWriteConfig,
    ) -> Result<Vec<u8>, RasmcoreError> {
        sink::write_bmp(&mut self.graph.borrow_mut(), source).map_err(to_wit_error)
    }

    fn write_ico(
        &self,
        source: NodeId,
        _config: pipeline::IcoWriteConfig,
    ) -> Result<Vec<u8>, RasmcoreError> {
        sink::write_ico(&mut self.graph.borrow_mut(), source).map_err(to_wit_error)
    }

    fn write_qoi(
        &self,
        source: NodeId,
        _config: pipeline::QoiWriteConfig,
    ) -> Result<Vec<u8>, RasmcoreError> {
        sink::write_qoi(&mut self.graph.borrow_mut(), source).map_err(to_wit_error)
    }

    fn write_gif(
        &self,
        source: NodeId,
        config: pipeline::GifWriteConfig,
        metadata: Option<pipeline::MetadataSet>,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let cfg = domain::encoder::gif::GifEncodeConfig {
            repeat: config.repeat.unwrap_or(0),
        };
        let domain_meta = metadata.as_ref().map(super::to_domain_metadata_set);
        sink::write_gif(
            &mut self.graph.borrow_mut(),
            source,
            &cfg,
            domain_meta.as_ref(),
        )
        .map_err(to_wit_error)
    }

    fn write_avif(
        &self,
        source: NodeId,
        config: pipeline::AvifWriteConfig,
        metadata: Option<pipeline::MetadataSet>,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let cfg = domain::encoder::avif::AvifEncodeConfig {
            quality: config.quality.unwrap_or(75),
            speed: config.speed.unwrap_or(6),
        };
        let domain_meta = metadata.as_ref().map(super::to_domain_metadata_set);
        sink::write_avif(
            &mut self.graph.borrow_mut(),
            source,
            &cfg,
            domain_meta.as_ref(),
        )
        .map_err(to_wit_error)
    }

    fn write_tiff(
        &self,
        source: NodeId,
        config: pipeline::TiffWriteConfig,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let cfg = domain::encoder::tiff::TiffEncodeConfig {
            compression: to_domain_tiff_compression_pipeline(config.compression),
        };
        sink::write_tiff(&mut self.graph.borrow_mut(), source, &cfg).map_err(to_wit_error)
    }

    fn write(
        &self,
        source: NodeId,
        format: String,
        quality: Option<u8>,
        metadata: Option<pipeline::MetadataSet>,
    ) -> Result<Vec<u8>, RasmcoreError> {
        let domain_meta = metadata.as_ref().map(super::to_domain_metadata_set);
        sink::write(
            &mut self.graph.borrow_mut(),
            source,
            &format,
            quality,
            domain_meta.as_ref(),
        )
        .map_err(to_wit_error)
    }

    // ─── Multi-Frame ───

    fn execute_sequence(
        &self,
        _source: NodeId,
        output: NodeId,
    ) -> Result<u32, RasmcoreError> {
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
            Some(data) => Ok(domain::metadata_query::metadata_dump_json_from_bytes(
                data, &ms,
            )),
            None => Ok(domain::metadata_query::metadata_dump_json(&ms)),
        }
    }

    fn metadata_read(&self, path: String) -> Result<Option<String>, RasmcoreError> {
        let ops = self.metadata_ops.borrow();
        let ms = ops.resolve();
        match ops.source_data.as_ref() {
            Some(data) => domain::metadata_query::metadata_read_from_bytes(data, &ms, &path)
                .map_err(to_wit_error),
            None => domain::metadata_query::metadata_read(&ms, &path).map_err(to_wit_error),
        }
    }

    fn keep_metadata(&self) -> Result<NodeId, RasmcoreError> {
        self.metadata_ops.borrow_mut().keep = true;
        Ok(0)
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
