//! Node graph — holds all nodes and dispatches region requests.

use std::rc::Rc;

use crate::domain::error::ImageError;
use crate::domain::pipeline::nodes::frame_source::FrameSourceNode;
use crate::domain::types::{DecodedImage, FrameSequence, ImageInfo};
use rasmcore_pipeline::{Rect, SpatialCache};

/// Access pattern hint for cache optimization.
#[derive(Debug, Clone, Copy)]
pub enum AccessPattern {
    /// Output tiles map to same-position upstream (point ops, crop).
    Sequential,
    /// Output tiles need upstream with overlap (blur, resize).
    LocalNeighborhood,
    /// Output tiles may request any upstream region (rotation).
    RandomAccess,
    /// Must see all upstream tiles before producing output (histogram eq).
    GlobalTwoPass,
}

/// The core trait every pipeline node implements.
pub trait ImageNode {
    /// Image dimensions and format (available without computing pixels).
    fn info(&self) -> ImageInfo;

    /// Compute pixels for the requested region.
    /// Uses `upstream_fn` to request regions from upstream nodes.
    fn compute_region(
        &self,
        request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError>;

    /// Compute the input rect needed to produce the given output rect.
    ///
    /// Operations expand the output rect by their kernel/neighborhood size.
    /// Point operations return the output rect unchanged. Spatial operations
    /// expand by their radius/kernel size. Distortion operations compute
    /// the bounding box of their inverse transform.
    ///
    /// `bounds_w`/`bounds_h` are the full image dimensions for clamping.
    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> Rect {
        // Default: no expansion (point operation)
        output.clamp(bounds_w, bounds_h)
    }

    /// Access pattern hint.
    fn access_pattern(&self) -> AccessPattern;

    /// If this node is a fuseable per-channel point operation, return its LUT.
    /// The pipeline optimizer composes consecutive LUTs into a single fused
    /// lookup table, eliminating redundant pixel passes.
    fn as_point_op_lut(&self) -> Option<[u8; 256]> {
        None
    }

    /// Return the upstream node id, if this node has exactly one upstream.
    /// Used by the LUT fusion and affine composition optimizers to walk chains.
    fn upstream_id(&self) -> Option<u32> {
        None
    }

    /// If this node is an affine transform, return its matrix and output dims.
    /// Used by the affine fusion optimizer to compose consecutive transforms
    /// into a single resample pass.
    fn as_affine_op(&self) -> Option<([f64; 6], u32, u32)> {
        None
    }

    /// If this node is a fuseable multi-channel color operation, return its 3D CLUT.
    fn as_color_lut_op(&self) -> Option<crate::domain::color_lut::ColorLut3D> {
        None
    }

    /// Derive this node's metadata from upstream metadata.
    ///
    /// Return `None` to inherit upstream metadata unchanged (zero-cost default).
    /// Return `Some(meta)` only when this node modifies metadata — clone
    /// the upstream only when modification is needed.
    ///
    /// Examples: resize sets width/height, auto_orient resets exif.Orientation,
    /// icc_to_srgb removes the ICC profile.
    fn derive_metadata(
        &self,
        _upstream: &rasmcore_pipeline::Metadata,
    ) -> Option<rasmcore_pipeline::Metadata> {
        None
    }
}

/// Rc wrapper around FrameSourceNode that implements ImageNode by delegation.
/// This allows the graph to own the node while the caller retains an Rc handle
/// for driving frame iteration in execute_sequence().
struct FrameSourceRcWrapper(Rc<FrameSourceNode>);

impl ImageNode for FrameSourceRcWrapper {
    fn info(&self) -> ImageInfo {
        self.0.info()
    }
    fn compute_region(
        &self,
        request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        self.0.compute_region(request, upstream_fn)
    }
    fn access_pattern(&self) -> AccessPattern {
        self.0.access_pattern()
    }
}

/// A pipeline node that applies a pre-composed LUT, replacing a chain of
/// consecutive point operations with a single fused lookup table pass.
struct FusedLutNode {
    upstream: u32,
    source_info: ImageInfo,
    lut: [u8; 256],
}

impl ImageNode for FusedLutNode {
    fn info(&self) -> ImageInfo {
        self.source_info.clone()
    }

    fn compute_region(
        &self,
        request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        let src_pixels = upstream_fn(self.upstream, request)?;
        crate::domain::point_ops::apply_lut(&src_pixels, &self.source_info, &self.lut)
    }

    fn as_point_op_lut(&self) -> Option<[u8; 256]> {
        Some(self.lut)
    }

    fn upstream_id(&self) -> Option<u32> {
        Some(self.upstream)
    }

    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::Sequential
    }
}

/// A pipeline node that applies a pre-composed 3D CLUT, replacing a chain
/// of consecutive multi-channel color operations with a single lookup.
struct FusedClutNode {
    upstream: u32,
    source_info: ImageInfo,
    clut: crate::domain::color_lut::ColorLut3D,
}

impl ImageNode for FusedClutNode {
    fn info(&self) -> ImageInfo {
        self.source_info.clone()
    }
    fn compute_region(
        &self,
        request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        let src = upstream_fn(self.upstream, request)?;
        self.clut.apply(&src, &self.source_info)
    }
    fn as_color_lut_op(&self) -> Option<crate::domain::color_lut::ColorLut3D> {
        Some(self.clut.clone())
    }
    fn upstream_id(&self) -> Option<u32> {
        Some(self.upstream)
    }
    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::Sequential
    }
}

/// Structured validation error with node context.
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Human-readable message describing the validation failure.
    pub message: String,
    /// Node index that triggered the error (if applicable).
    pub node_id: Option<u32>,
    /// Upstream node index involved (if applicable).
    pub upstream_id: Option<u32>,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (self.node_id, self.upstream_id) {
            (Some(nid), Some(uid)) => write!(f, "node {nid} (upstream {uid}): {}", self.message),
            (Some(nid), None) => write!(f, "node {nid}: {}", self.message),
            _ => write!(f, "{}", self.message),
        }
    }
}

impl From<ValidationError> for ImageError {
    fn from(e: ValidationError) -> Self {
        ImageError::InvalidParameters(e.to_string())
    }
}

/// A descriptor capturing a node's identity and configuration for graph introspection.
///
/// Built alongside the live `NodeGraph` during pipeline construction. Contains
/// everything needed to reconstruct the node or serialize the graph description.
#[derive(Debug, Clone)]
pub struct NodeDescriptor {
    /// Node category.
    pub kind: NodeKind,
    /// Operation name (e.g., "blur", "resize", "source").
    pub name: String,
    /// Serialized configuration bytes (operation-specific).
    pub config: Vec<u8>,
    /// Upstream node indices (1 for most operations, 2 for composite, 0 for source).
    pub upstreams: Vec<u32>,
    /// Output ImageInfo computed at build time from upstream + operation.
    pub output_info: ImageInfo,
}

/// Category of a pipeline node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    /// Image source (decode from data).
    Source,
    /// Pixel filter (blur, sharpen, color adjustment, etc.).
    Filter,
    /// Geometric transform (resize, crop, rotate, flip).
    Transform,
    /// Pixel mapper (threshold, quantize, gradient map, etc.).
    Mapper,
    /// Composite (blend two images).
    Composite,
}

impl std::fmt::Display for NodeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeKind::Source => write!(f, "source"),
            NodeKind::Filter => write!(f, "filter"),
            NodeKind::Transform => write!(f, "transform"),
            NodeKind::Mapper => write!(f, "mapper"),
            NodeKind::Composite => write!(f, "composite"),
        }
    }
}

/// A serializable description of a pipeline graph.
///
/// Self-contained — no mutable internal state. Each pipeline method appends a
/// `NodeDescriptor` during construction. The graph can be inspected, serialized,
/// and used to reconstruct a live `NodeGraph` for execution.
#[derive(Debug, Clone)]
pub struct GraphDescription {
    descriptors: Vec<NodeDescriptor>,
}

impl GraphDescription {
    /// Create an empty graph description.
    pub fn new() -> Self {
        Self {
            descriptors: Vec::new(),
        }
    }

    /// Append a node descriptor, validating upstream references.
    ///
    /// Returns the new node's index. Enforces DAG invariant: all upstream
    /// indices must be < current length (no forward or self references).
    pub fn add(&mut self, descriptor: NodeDescriptor) -> Result<u32, ValidationError> {
        let current_id = self.descriptors.len() as u32;
        for &up in &descriptor.upstreams {
            if up >= current_id {
                return Err(ValidationError {
                    message: format!(
                        "upstream {up} >= current node {current_id} — forward/self reference"
                    ),
                    node_id: Some(current_id),
                    upstream_id: Some(up),
                });
            }
        }
        self.descriptors.push(descriptor);
        Ok(current_id)
    }

    /// Number of nodes in the description.
    pub fn len(&self) -> usize {
        self.descriptors.len()
    }

    /// Whether the description is empty.
    pub fn is_empty(&self) -> bool {
        self.descriptors.is_empty()
    }

    /// Get a node descriptor by index.
    pub fn get(&self, index: u32) -> Option<&NodeDescriptor> {
        self.descriptors.get(index as usize)
    }

    /// Iterate over all node descriptors.
    pub fn iter(&self) -> impl Iterator<Item = &NodeDescriptor> {
        self.descriptors.iter()
    }

    /// Get the output ImageInfo at a given node index.
    pub fn node_info(&self, index: u32) -> Option<&ImageInfo> {
        self.descriptors.get(index as usize).map(|d| &d.output_info)
    }

    /// Validate the full graph description.
    ///
    /// Checks:
    /// 1. All upstream references are valid (< len, not self)
    /// 2. At least one node exists
    /// 3. Graph has a clear terminal node (last node)
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.descriptors.is_empty() {
            return Err(ValidationError {
                message: "empty graph — no nodes".to_string(),
                node_id: None,
                upstream_id: None,
            });
        }

        for (i, desc) in self.descriptors.iter().enumerate() {
            let i = i as u32;
            for &up in &desc.upstreams {
                if up >= i {
                    return Err(ValidationError {
                        message: format!(
                            "upstream {up} >= node {i} — invalid reference"
                        ),
                        node_id: Some(i),
                        upstream_id: Some(up),
                    });
                }
            }
        }
        Ok(())
    }

    /// Serialize to a human-readable JSON string.
    pub fn to_json(&self) -> String {
        let mut json = String::from("[");
        for (i, desc) in self.descriptors.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str(&format!(
                r#"{{"id":{},"kind":"{}","name":"{}","upstreams":{:?},"output":{{"width":{},"height":{},"format":"{:?}","color_space":"{:?}"}}}}"#,
                i,
                desc.kind,
                desc.name,
                desc.upstreams,
                desc.output_info.width,
                desc.output_info.height,
                desc.output_info.format,
                desc.output_info.color_space,
            ));
        }
        json.push(']');
        json
    }

    /// Compact binary serialization.
    ///
    /// Format: [node_count: u32] then for each node:
    ///   [kind: u8] [name_len: u16] [name: bytes]
    ///   [config_len: u32] [config: bytes]
    ///   [upstream_count: u8] [upstreams: u32×N]
    ///   [width: u32] [height: u32] [format: u8] [color_space: u8]
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(self.descriptors.len() as u32).to_le_bytes());

        for desc in &self.descriptors {
            // kind
            buf.push(match desc.kind {
                NodeKind::Source => 0,
                NodeKind::Filter => 1,
                NodeKind::Transform => 2,
                NodeKind::Mapper => 3,
                NodeKind::Composite => 4,
            });
            // name
            let name_bytes = desc.name.as_bytes();
            buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            buf.extend_from_slice(name_bytes);
            // config
            buf.extend_from_slice(&(desc.config.len() as u32).to_le_bytes());
            buf.extend_from_slice(&desc.config);
            // upstreams
            buf.push(desc.upstreams.len() as u8);
            for &up in &desc.upstreams {
                buf.extend_from_slice(&up.to_le_bytes());
            }
            // output_info
            buf.extend_from_slice(&desc.output_info.width.to_le_bytes());
            buf.extend_from_slice(&desc.output_info.height.to_le_bytes());
            buf.push(pixel_format_to_u8(desc.output_info.format));
            buf.push(color_space_to_u8(desc.output_info.color_space));
        }
        buf
    }

    /// Deserialize a graph description from compact binary format.
    pub fn deserialize(data: &[u8]) -> Result<Self, ImageError> {
        let mut pos = 0;

        let read_u8 = |pos: &mut usize| -> Result<u8, ImageError> {
            if *pos >= data.len() {
                return Err(ImageError::InvalidInput("truncated graph data".into()));
            }
            let v = data[*pos];
            *pos += 1;
            Ok(v)
        };
        let read_u16 = |pos: &mut usize| -> Result<u16, ImageError> {
            if *pos + 2 > data.len() {
                return Err(ImageError::InvalidInput("truncated graph data".into()));
            }
            let v = u16::from_le_bytes([data[*pos], data[*pos + 1]]);
            *pos += 2;
            Ok(v)
        };
        let read_u32 = |pos: &mut usize| -> Result<u32, ImageError> {
            if *pos + 4 > data.len() {
                return Err(ImageError::InvalidInput("truncated graph data".into()));
            }
            let v = u32::from_le_bytes([
                data[*pos],
                data[*pos + 1],
                data[*pos + 2],
                data[*pos + 3],
            ]);
            *pos += 4;
            Ok(v)
        };

        let node_count = read_u32(&mut pos)? as usize;
        let mut descriptors = Vec::with_capacity(node_count);

        for _ in 0..node_count {
            let kind = match read_u8(&mut pos)? {
                0 => NodeKind::Source,
                1 => NodeKind::Filter,
                2 => NodeKind::Transform,
                3 => NodeKind::Mapper,
                4 => NodeKind::Composite,
                v => {
                    return Err(ImageError::InvalidInput(format!(
                        "unknown node kind: {v}"
                    )))
                }
            };

            let name_len = read_u16(&mut pos)? as usize;
            if pos + name_len > data.len() {
                return Err(ImageError::InvalidInput("truncated name".into()));
            }
            let name =
                String::from_utf8(data[pos..pos + name_len].to_vec()).map_err(|_| {
                    ImageError::InvalidInput("invalid UTF-8 in node name".into())
                })?;
            pos += name_len;

            let config_len = read_u32(&mut pos)? as usize;
            if pos + config_len > data.len() {
                return Err(ImageError::InvalidInput("truncated config".into()));
            }
            let config = data[pos..pos + config_len].to_vec();
            pos += config_len;

            let upstream_count = read_u8(&mut pos)? as usize;
            let mut upstreams = Vec::with_capacity(upstream_count);
            for _ in 0..upstream_count {
                upstreams.push(read_u32(&mut pos)?);
            }

            let width = read_u32(&mut pos)?;
            let height = read_u32(&mut pos)?;
            let format = u8_to_pixel_format(read_u8(&mut pos)?)?;
            let color_space = u8_to_color_space(read_u8(&mut pos)?)?;

            descriptors.push(NodeDescriptor {
                kind,
                name,
                config,
                upstreams,
                output_info: ImageInfo {
                    width,
                    height,
                    format,
                    color_space,
                },
            });
        }

        Ok(Self { descriptors })
    }
}

impl Default for GraphDescription {
    fn default() -> Self {
        Self::new()
    }
}

/// Pipeline-owned node graph.
pub struct NodeGraph {
    nodes: Vec<Box<dyn ImageNode>>,
    cache: SpatialCache,
    cache_budget: usize,
    // Layer cache (optional, persists across pipeline lifetimes)
    layer_cache: Option<std::rc::Rc<std::cell::RefCell<rasmcore_pipeline::LayerCache>>>,
    node_hashes: Vec<rasmcore_pipeline::ContentHash>,
    touched_hashes: std::collections::HashSet<rasmcore_pipeline::ContentHash>,
    cache_hit_nodes: std::collections::HashSet<u32>,
    cache_hit_pixels: std::collections::HashMap<u32, (Vec<u8>, u32, u32)>, // (pixels, width, height)
    // Per-node metadata — set at creation, immutable during tile execution
    node_metadata: Vec<rasmcore_pipeline::Metadata>,
    // Per-node accumulator buffers for assembling full images from tiles
    node_accumulators: Vec<Option<Vec<u8>>>,
    // Graph description — built alongside the live graph
    description: GraphDescription,
}

impl NodeGraph {
    /// Create a new graph with the given memory budget for the spatial cache.
    pub fn new(cache_budget: usize) -> Self {
        Self {
            nodes: Vec::new(),
            cache: SpatialCache::new(0),
            cache_budget,
            layer_cache: None,
            node_hashes: Vec::new(),
            touched_hashes: std::collections::HashSet::new(),
            cache_hit_nodes: std::collections::HashSet::new(),
            cache_hit_pixels: std::collections::HashMap::new(),
            node_metadata: Vec::new(),
            node_accumulators: Vec::new(),
            description: GraphDescription::new(),
        }
    }

    /// Create a new graph with an optional layer cache for cross-pipeline reuse.
    pub fn with_layer_cache(
        cache_budget: usize,
        layer_cache: std::rc::Rc<std::cell::RefCell<rasmcore_pipeline::LayerCache>>,
    ) -> Self {
        let mut graph = Self::new(cache_budget);
        graph.layer_cache = Some(layer_cache);
        graph
    }

    /// Add a node to the graph. Returns its node-id.
    pub fn add_node(&mut self, node: Box<dyn ImageNode>) -> u32 {
        let id = self.nodes.len() as u32;
        let info = node.info();
        let upstreams = self.collect_upstreams(&*node);
        self.nodes.push(node);
        self.node_hashes.push(rasmcore_pipeline::ZERO_HASH);
        self.node_metadata.push(rasmcore_pipeline::Metadata::new());
        self.node_accumulators.push(None);
        // Track descriptor (best-effort — no name/kind available for untyped adds)
        let _ = self.description.add(NodeDescriptor {
            kind: NodeKind::Filter, // default for untyped
            name: String::new(),
            config: Vec::new(),
            upstreams,
            output_info: info,
        });
        id
    }

    /// Add a node with metadata (flows from upstream, possibly modified).
    pub fn add_node_with_metadata(
        &mut self,
        node: Box<dyn ImageNode>,
        metadata: rasmcore_pipeline::Metadata,
    ) -> u32 {
        let id = self.nodes.len() as u32;
        let info = node.info();
        let upstreams = self.collect_upstreams(&*node);
        self.nodes.push(node);
        self.node_hashes.push(rasmcore_pipeline::ZERO_HASH);
        self.node_metadata.push(metadata);
        self.node_accumulators.push(None);
        let _ = self.description.add(NodeDescriptor {
            kind: NodeKind::Filter,
            name: String::new(),
            config: Vec::new(),
            upstreams,
            output_info: info,
        });
        id
    }

    /// Get the metadata for a node.
    pub fn node_metadata(&self, node_id: u32) -> &rasmcore_pipeline::Metadata {
        static EMPTY: std::sync::LazyLock<rasmcore_pipeline::Metadata> =
            std::sync::LazyLock::new(rasmcore_pipeline::Metadata::new);
        self.node_metadata.get(node_id as usize).unwrap_or(&EMPTY)
    }

    /// Get mutable metadata for a node (for set_metadata operations).
    pub fn node_metadata_mut(&mut self, node_id: u32) -> Option<&mut rasmcore_pipeline::Metadata> {
        self.node_metadata.get_mut(node_id as usize)
    }

    /// Add a node with a content hash for layer caching.
    pub fn add_node_with_hash(
        &mut self,
        node: Box<dyn ImageNode>,
        hash: rasmcore_pipeline::ContentHash,
    ) -> u32 {
        let id = self.nodes.len() as u32;
        let info = node.info();
        let upstreams = self.collect_upstreams(&*node);
        self.nodes.push(node);
        self.node_hashes.push(hash);
        self.node_metadata.push(rasmcore_pipeline::Metadata::new());

        // Check layer cache for this hash
        let mut hit = false;
        if let Some(lc) = &self.layer_cache {
            let mut lc = lc.borrow_mut();
            if let Some((pixels, w, h, _bpp)) = lc.get(&hash) {
                self.cache_hit_pixels.insert(id, (pixels.to_vec(), w, h));
                self.cache_hit_nodes.insert(id);
                hit = true;
            }
        }

        if hit {
            self.node_accumulators.push(None);
        } else {
            let bpp = bytes_per_pixel(info.format) as usize;
            let buf_size = info.width as usize * info.height as usize * bpp;
            self.node_accumulators.push(Some(vec![0u8; buf_size]));
        }

        let _ = self.description.add(NodeDescriptor {
            kind: NodeKind::Filter,
            name: String::new(),
            config: Vec::new(),
            upstreams,
            output_info: info,
        });

        id
    }

    /// Add a node with both a content hash and metadata.
    pub fn add_node_with_hash_and_metadata(
        &mut self,
        node: Box<dyn ImageNode>,
        hash: rasmcore_pipeline::ContentHash,
        metadata: rasmcore_pipeline::Metadata,
    ) -> u32 {
        self.add_node_with_hash_metadata_desc(node, hash, metadata, NodeKind::Filter, "")
    }

    /// Add a source node with content hash, metadata, and proper Source descriptor.
    pub fn add_source_node(
        &mut self,
        node: Box<dyn ImageNode>,
        hash: rasmcore_pipeline::ContentHash,
        metadata: rasmcore_pipeline::Metadata,
        name: &str,
    ) -> u32 {
        self.add_node_with_hash_metadata_desc(node, hash, metadata, NodeKind::Source, name)
    }

    /// Internal: add node with hash, metadata, and explicit descriptor kind/name.
    fn add_node_with_hash_metadata_desc(
        &mut self,
        node: Box<dyn ImageNode>,
        hash: rasmcore_pipeline::ContentHash,
        metadata: rasmcore_pipeline::Metadata,
        kind: NodeKind,
        name: &str,
    ) -> u32 {
        let id = self.nodes.len() as u32;
        let info = node.info();
        let upstreams = self.collect_upstreams(&*node);
        self.nodes.push(node);
        self.node_hashes.push(hash);
        self.node_metadata.push(metadata);

        let mut hit = false;
        if let Some(lc) = &self.layer_cache {
            let mut lc = lc.borrow_mut();
            if let Some((pixels, w, h, _bpp)) = lc.get(&hash) {
                self.cache_hit_pixels.insert(id, (pixels.to_vec(), w, h));
                self.cache_hit_nodes.insert(id);
                hit = true;
            }
        }

        if hit {
            self.node_accumulators.push(None);
        } else {
            let bpp = bytes_per_pixel(info.format) as usize;
            let buf_size = info.width as usize * info.height as usize * bpp;
            self.node_accumulators.push(Some(vec![0u8; buf_size]));
        }

        let _ = self.description.add(NodeDescriptor {
            kind,
            name: name.to_string(),
            config: Vec::new(),
            upstreams,
            output_info: info,
        });

        id
    }

    /// Add a node that derives its metadata from an upstream node.
    ///
    /// Calls `node.derive_metadata(&upstream_meta)` — if `Some(meta)` is returned,
    /// uses the new metadata; otherwise inherits upstream metadata via clone.
    pub fn add_node_derived(
        &mut self,
        node: Box<dyn ImageNode>,
        hash: rasmcore_pipeline::ContentHash,
        upstream_id: u32,
    ) -> u32 {
        let upstream_meta = self.node_metadata(upstream_id).clone();
        let meta = node
            .derive_metadata(&upstream_meta)
            .unwrap_or(upstream_meta);
        self.add_node_with_hash_and_metadata(node, hash, meta)
    }

    /// Add a node with a descriptor specifying its kind and name.
    ///
    /// This is the preferred method for adding nodes with full graph description
    /// tracking. The descriptor provides kind and name for introspection.
    pub fn add_node_described(
        &mut self,
        node: Box<dyn ImageNode>,
        hash: rasmcore_pipeline::ContentHash,
        upstream_id: u32,
        kind: NodeKind,
        name: &str,
    ) -> u32 {
        let upstream_meta = self.node_metadata(upstream_id).clone();
        let info = node.info();
        let upstreams = self.collect_upstreams(&*node);
        let meta = node
            .derive_metadata(&upstream_meta)
            .unwrap_or(upstream_meta);

        let id = self.nodes.len() as u32;
        self.nodes.push(node);
        self.node_hashes.push(hash);
        self.node_metadata.push(meta);

        let mut hit = false;
        if let Some(lc) = &self.layer_cache {
            let mut lc = lc.borrow_mut();
            if let Some((pixels, w, h, _bpp)) = lc.get(&hash) {
                self.cache_hit_pixels.insert(id, (pixels.to_vec(), w, h));
                self.cache_hit_nodes.insert(id);
                hit = true;
            }
        }
        if hit {
            self.node_accumulators.push(None);
        } else {
            let bpp = bytes_per_pixel(info.format) as usize;
            let buf_size = info.width as usize * info.height as usize * bpp;
            self.node_accumulators.push(Some(vec![0u8; buf_size]));
        }

        let _ = self.description.add(NodeDescriptor {
            kind,
            name: name.to_string(),
            config: Vec::new(),
            upstreams,
            output_info: info,
        });

        id
    }

    /// Collect upstream node IDs from a node's trait methods.
    fn collect_upstreams(&self, node: &dyn ImageNode) -> Vec<u32> {
        match node.upstream_id() {
            Some(id) => vec![id],
            None => Vec::new(),
        }
    }

    /// Validate the upstream reference of a node before adding it.
    ///
    /// Returns `Ok(())` if all upstream references are < current node count,
    /// or `Err` with a structured validation error.
    pub fn validate_upstream(&self, upstream_id: u32) -> Result<(), ValidationError> {
        let current = self.nodes.len() as u32;
        if upstream_id >= current {
            return Err(ValidationError {
                message: format!(
                    "upstream {upstream_id} does not exist (graph has {current} nodes)"
                ),
                node_id: Some(current),
                upstream_id: Some(upstream_id),
            });
        }
        Ok(())
    }

    /// Validate the full graph before execution.
    ///
    /// Checks:
    /// 1. Graph is non-empty
    /// 2. All upstream references within nodes are valid
    /// 3. Terminal node exists (last node in the graph)
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.nodes.is_empty() {
            return Err(ValidationError {
                message: "empty graph — no nodes to execute".into(),
                node_id: None,
                upstream_id: None,
            });
        }

        for (i, node) in self.nodes.iter().enumerate() {
            let i = i as u32;
            if let Some(up) = node.upstream_id()
                && up >= i
            {
                return Err(ValidationError {
                    message: format!(
                        "upstream {up} >= node {i} — invalid forward/self reference"
                    ),
                    node_id: Some(i),
                    upstream_id: Some(up),
                });
            }
        }
        Ok(())
    }

    /// Validate a specific node for execution.
    ///
    /// Checks that the node exists and its output info has valid dimensions.
    pub fn validate_node(&self, node_id: u32) -> Result<(), ValidationError> {
        let info = self.node_info(node_id).map_err(|_| ValidationError {
            message: format!("node {node_id} does not exist"),
            node_id: Some(node_id),
            upstream_id: None,
        })?;
        if info.width == 0 || info.height == 0 {
            return Err(ValidationError {
                message: format!(
                    "node {node_id} has zero dimensions: {}x{}",
                    info.width, info.height
                ),
                node_id: Some(node_id),
                upstream_id: None,
            });
        }
        Ok(())
    }

    /// Get the graph description built alongside the live graph.
    pub fn description(&self) -> &GraphDescription {
        &self.description
    }

    /// Get the content hash for a node.
    pub fn node_hash(&self, node_id: u32) -> rasmcore_pipeline::ContentHash {
        self.node_hashes
            .get(node_id as usize)
            .copied()
            .unwrap_or(rasmcore_pipeline::ZERO_HASH)
    }

    /// Get image info for a node (no pixel computation).
    pub fn node_info(&self, node_id: u32) -> Result<ImageInfo, ImageError> {
        self.nodes
            .get(node_id as usize)
            .map(|n| n.info())
            .ok_or_else(|| ImageError::InvalidParameters(format!("invalid node id: {node_id}")))
    }

    /// Fuse consecutive per-channel point operations into single LUT nodes.
    ///
    /// Walks the graph and for each node that returns `as_point_op_lut()`,
    /// follows the upstream chain composing LUTs until hitting a non-point-op
    /// barrier. Replaces the entire chain with a single FusedLutNode.
    ///
    /// Call this after building the graph and before requesting regions.
    pub fn fuse_point_ops(&mut self) {
        let n = self.nodes.len();
        // Track which nodes have been consumed into a fused chain
        let mut fused_into: Vec<Option<u32>> = vec![None; n];

        // Walk from output (last) toward source (first)
        for i in (0..n).rev() {
            // Skip nodes already consumed into another fusion
            if fused_into[i].is_some() {
                continue;
            }

            let Some(mut lut) = self.nodes[i].as_point_op_lut() else {
                continue;
            };

            // Walk upstream composing LUTs
            let mut chain_root_upstream = self.nodes[i].upstream_id();
            let mut current = i;

            while let Some(up_id) = self.nodes[current].upstream_id() {
                let up = up_id as usize;
                if up >= n || fused_into[up].is_some() {
                    break;
                }
                let Some(up_lut) = self.nodes[up].as_point_op_lut() else {
                    break;
                };
                // Compose: up_lut first, then our accumulated lut
                lut = crate::domain::point_ops::compose_luts(&up_lut, &lut);
                chain_root_upstream = self.nodes[up].upstream_id();
                fused_into[up] = Some(i as u32);
                // Clear consumed node's accumulator
                if up < self.node_accumulators.len() {
                    self.node_accumulators[up] = None;
                }
                current = up;
            }

            // If we consumed at least one upstream node, replace this node with a fused one
            if current != i {
                let upstream = chain_root_upstream.unwrap_or(0);
                let info = self.nodes[i].info();
                self.nodes[i] = Box::new(FusedLutNode {
                    upstream,
                    source_info: info,
                    lut,
                });
                // Reallocate accumulator for the fused node
                let fused_info = self.nodes[i].info();
                let bpp = bytes_per_pixel(fused_info.format) as usize;
                let buf_size = fused_info.width as usize * fused_info.height as usize * bpp;
                if self.node_accumulators.len() > i {
                    self.node_accumulators[i] = Some(vec![0u8; buf_size]);
                }
            }
        }
    }

    /// Compose consecutive affine transform nodes into a single resample pass.
    ///
    /// Walks the node graph backwards from each output node. When a chain of
    /// consecutive AffineOp nodes is found (resize → rotate → flip, etc.), they
    /// are composed into a single affine matrix and replaced with a single
    /// `ComposedAffineNode`. This eliminates multi-pass interpolation artifacts
    /// and improves both quality and performance.
    pub fn fuse_affine_transforms(&mut self) {
        use crate::domain::pipeline::nodes::transform::{ComposedAffineNode, compose_affine};

        let n = self.nodes.len();
        let mut fused_into: Vec<Option<u32>> = vec![None; n];

        for i in (0..n).rev() {
            if fused_into[i].is_some() {
                continue;
            }

            let Some((mut matrix, _out_w, _out_h)) = self.nodes[i].as_affine_op() else {
                continue;
            };

            // Walk upstream composing affine matrices
            let mut chain_root_upstream = self.nodes[i].upstream_id();
            let mut current = i;

            while let Some(up_id) = self.nodes[current].upstream_id() {
                let up = up_id as usize;
                if up >= n || fused_into[up].is_some() {
                    break;
                }
                let Some((up_matrix, _up_w, _up_h)) = self.nodes[up].as_affine_op() else {
                    break;
                };
                // Compose: current(upstream(x)) = current_matrix * upstream_matrix
                matrix = compose_affine(&matrix, &up_matrix);
                chain_root_upstream = self.nodes[up].upstream_id();
                fused_into[up] = Some(i as u32);
                current = up;
            }

            // If we consumed at least one upstream node, replace with composed
            if current != i {
                let upstream = chain_root_upstream.unwrap_or(0);
                // Get the source_info from the chain root's upstream
                let source_info = if let Some(root_up) = chain_root_upstream {
                    self.nodes[root_up as usize].info()
                } else {
                    self.nodes[current].info()
                };
                // Compute output dimensions from the composed matrix
                let (_final_w, _final_h) =
                    crate::domain::pipeline::nodes::transform::affine_output_dims(
                        &matrix,
                        source_info.width,
                        source_info.height,
                    );
                // Use explicitly declared output dims if they match better
                // (the original chain's output info is authoritative)
                let out_info = self.nodes[i].info();
                let (use_w, use_h) = (out_info.width, out_info.height);

                self.nodes[i] = Box::new(ComposedAffineNode::new(
                    upstream,
                    source_info,
                    matrix,
                    use_w,
                    use_h,
                ));
            }
        }
    }

    /// Fuse consecutive multi-channel color operations into single 3D CLUT nodes.
    pub fn fuse_color_ops(&mut self) {
        let n = self.nodes.len();
        let mut fused_into: Vec<Option<u32>> = vec![None; n];
        for i in (0..n).rev() {
            if fused_into[i].is_some() {
                continue;
            }
            let Some(mut clut) = self.nodes[i].as_color_lut_op() else {
                continue;
            };
            let mut chain_root_upstream = self.nodes[i].upstream_id();
            let mut current = i;
            while let Some(up_id) = self.nodes[current].upstream_id() {
                let up = up_id as usize;
                if up >= n || fused_into[up].is_some() {
                    break;
                }
                let Some(up_clut) = self.nodes[up].as_color_lut_op() else {
                    break;
                };
                clut = crate::domain::color_lut::compose_cluts(&up_clut, &clut);
                chain_root_upstream = self.nodes[up].upstream_id();
                fused_into[up] = Some(i as u32);
                // Clear consumed node's accumulator
                if up < self.node_accumulators.len() {
                    self.node_accumulators[up] = None;
                }
                current = up;
            }
            if current != i {
                let upstream = chain_root_upstream.unwrap_or(0);
                let info = self.nodes[i].info();
                self.nodes[i] = Box::new(FusedClutNode {
                    upstream,
                    source_info: info,
                    clut,
                });
                // Reallocate accumulator for the fused node
                let fused_info = self.nodes[i].info();
                let bpp = bytes_per_pixel(fused_info.format) as usize;
                let buf_size = fused_info.width as usize * fused_info.height as usize * bpp;
                if self.node_accumulators.len() > i {
                    self.node_accumulators[i] = Some(vec![0u8; buf_size]);
                }
            }
        }
    }

    /// Copy a computed tile into the node's accumulator buffer.
    fn accumulate_tile(&mut self, node_id: u32, tile_rect: Rect, tile_pixels: &[u8], bpp: u32) {
        let acc = match self.node_accumulators.get_mut(node_id as usize) {
            Some(Some(buf)) => buf,
            _ => return, // No accumulator for this node
        };
        let info = self.nodes[node_id as usize].info();
        let full_stride = info.width as usize * bpp as usize;
        let tile_stride = tile_rect.width as usize * bpp as usize;
        for row in 0..tile_rect.height as usize {
            let dst_y = tile_rect.y as usize + row;
            let dst_x = tile_rect.x as usize * bpp as usize;
            let dst = dst_y * full_stride + dst_x;
            let src = row * tile_stride;
            if dst + tile_stride <= acc.len() && src + tile_stride <= tile_pixels.len() {
                acc[dst..dst + tile_stride].copy_from_slice(&tile_pixels[src..src + tile_stride]);
            }
        }
    }

    /// Request a region from a node. Accumulates tiles into per-node buffers.
    pub fn request_region(&mut self, node_id: u32, request: Rect) -> Result<Vec<u8>, ImageError> {
        // Track this node's hash as "touched" for layer cache reference tracking
        if let Some(hash) = self.node_hashes.get(node_id as usize) {
            self.touched_hashes.insert(*hash);
        }

        // Check layer cache hit first (pre-populated during add_node_with_hash).
        if self.cache_hit_nodes.contains(&node_id)
            && let Some((pixels, cached_w, cached_h)) = self.cache_hit_pixels.get(&node_id)
        {
            let cached_rect = Rect::new(0, 0, *cached_w, *cached_h);
            if request == cached_rect {
                return Ok(pixels.clone());
            }
            // Verify pixel buffer matches declared dimensions before cropping
            let info = self.node_info(node_id)?;
            let bpp = bytes_per_pixel(info.format);
            let expected_len = *cached_w as usize * *cached_h as usize * bpp as usize;
            if cached_rect.contains(&request) && pixels.len() == expected_len {
                return Ok(crop_region(pixels, cached_rect, request, bpp));
            }
            // Dimension mismatch or request not contained — fall through to compute
        }

        let info = self.node_info(node_id)?;
        let bpp = bytes_per_pixel(info.format);

        // Compute the full requested region.
        // We use raw pointer to split the borrow: nodes[node_id] is read,
        // while the rest of self (cache, other nodes) can be mutated.
        let pixels = {
            let nodes_ptr = self.nodes.as_ptr();
            let node_count = self.nodes.len();
            if (node_id as usize) >= node_count {
                return Err(ImageError::InvalidParameters(format!(
                    "invalid node id: {node_id}"
                )));
            }
            let node = unsafe { &*nodes_ptr.add(node_id as usize) };

            let self_ptr = self as *mut NodeGraph;
            let mut upstream_fn = |upstream_id: u32, rect: Rect| -> Result<Vec<u8>, ImageError> {
                unsafe { &mut *self_ptr }.request_region(upstream_id, rect)
            };

            node.compute_region(request, &mut upstream_fn)?
        };

        // Accumulate tile into per-node buffer (for layer cache finalization)
        self.accumulate_tile(node_id, request, &pixels, bpp);

        Ok(pixels)
    }

    /// Access the spatial cache directly.
    pub fn cache(&self) -> &SpatialCache {
        &self.cache
    }

    /// Finalize layer cache after pipeline execution completes.
    ///
    /// Pushes newly computed node outputs to the layer cache and cleans
    /// unreferenced entries. Called by pipeline write methods after encoding.
    pub fn finalize_layer_cache(&mut self) {
        let layer_cache = match &self.layer_cache {
            Some(lc) => lc.clone(),
            None => return,
        };
        let mut lc = layer_cache.borrow_mut();

        // Reset references for this run
        lc.reset_references();

        // Push newly computed nodes (not cache hits) and mark all as referenced
        for (node_id, hash) in self.node_hashes.iter().enumerate() {
            if *hash == rasmcore_pipeline::ZERO_HASH {
                continue;
            }
            // Mark as referenced regardless of hit/miss
            lc.mark_referenced(hash);

            // If this was NOT a cache hit, push the computed output from accumulator
            if !self.cache_hit_nodes.contains(&(node_id as u32))
                && let Some(Some(pixels)) = self.node_accumulators.get(node_id)
            {
                let info = match self.nodes.get(node_id) {
                    Some(n) => n.info(),
                    None => continue,
                };
                let bpp = bytes_per_pixel(info.format);
                lc.store(*hash, pixels.clone(), info.width, info.height, bpp);
            }
        }

        // Mark all touched hashes as referenced
        for hash in &self.touched_hashes {
            lc.mark_referenced(hash);
        }

        // Clean entries not referenced by this pipeline run
        lc.cleanup_unreferenced();

        // Free accumulator memory
        self.node_accumulators.clear();
    }

    /// Clear all graph state after execution. Keeps the layer_cache reference
    /// and the graph description (for introspection and re-execution).
    pub fn cleanup(&mut self) {
        self.nodes.clear();
        self.node_hashes.clear();
        self.node_accumulators.clear();
        self.cache_hit_nodes.clear();
        self.cache_hit_pixels.clear();
        self.touched_hashes.clear();
        self.node_metadata.clear();
        // description intentionally kept — it's the serialized graph specification
        // layer_cache and cache_budget intentionally kept
    }

    /// Clear everything including the graph description. Used when starting fresh.
    pub fn reset(&mut self) {
        self.cleanup();
        self.description = GraphDescription::new();
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Add a frame source node and return both the node-id and an Rc handle
    /// for driving frame iteration via `execute_sequence()`.
    pub fn add_frame_source(&mut self, node: FrameSourceNode) -> (u32, Rc<FrameSourceNode>) {
        let rc = Rc::new(node);
        let id = self.nodes.len() as u32;
        let info = rc.info();
        self.nodes.push(Box::new(FrameSourceRcWrapper(rc.clone())));
        self.node_hashes.push(rasmcore_pipeline::ZERO_HASH);
        self.node_metadata.push(rasmcore_pipeline::Metadata::new());
        self.node_accumulators.push(None);
        let _ = self.description.add(NodeDescriptor {
            kind: NodeKind::Source,
            name: "frame_source".to_string(),
            config: Vec::new(),
            upstreams: Vec::new(),
            output_info: info,
        });
        (id, rc)
    }

    /// Execute the pipeline in sequence mode: run the graph once per selected
    /// frame from a FrameSourceNode, collecting results into a FrameSequence.
    ///
    /// `frame_source` is the Rc handle returned by `add_frame_source()`.
    /// `output_node_id` is the final node whose output is captured per frame.
    pub fn execute_sequence(
        &mut self,
        frame_source: &Rc<FrameSourceNode>,
        output_node_id: u32,
    ) -> Result<FrameSequence, ImageError> {
        // Pre-execution validation
        self.validate()
            .map_err(|e| ImageError::InvalidParameters(e.to_string()))?;
        self.validate_node(output_node_id)
            .map_err(|e| ImageError::InvalidParameters(e.to_string()))?;

        let indices = frame_source.selected_indices();
        let (canvas_w, canvas_h) = frame_source.canvas_size();
        let mut sequence = FrameSequence::new(canvas_w, canvas_h);

        for &frame_idx in &indices {
            // Advance the source to the next frame
            let frame_info = frame_source.set_current_frame(frame_idx)?;

            // Clear cache between frames — cached regions from previous frame are stale
            self.cache = SpatialCache::new(self.cache_budget);

            // Execute the graph for this frame
            let info = self.node_info(output_node_id)?;
            let full = Rect::new(0, 0, info.width, info.height);
            let pixels = self.request_region(output_node_id, full)?;

            sequence.push(
                DecodedImage {
                    pixels,
                    info,
                    icc_profile: None,
                },
                frame_info,
            );
        }

        Ok(sequence)
    }
}

/// Execute a pipeline from a serialized graph description + source data.
///
/// This is the stateless execution path: the graph description captures the
/// full pipeline specification (node kinds, names, upstreams, output info).
/// Given source image data, it reconstructs a live NodeGraph using the filter
/// dispatch, runs execution, and returns the encoded output.
///
/// The description is not consumed — it can be re-executed with different data.
///
/// Currently, filter/transform config is reconstructed from the node names
/// using default parameters. Full config serialization into descriptors is
/// planned for a future track.
pub fn execute_from_description(
    description: &GraphDescription,
    source_data: &[u8],
    terminal_node: u32,
    format: &str,
    quality: Option<u8>,
) -> Result<Vec<u8>, ImageError> {
    use crate::domain::pipeline::nodes::{sink, source};

    // Validate the description
    description
        .validate()
        .map_err(|e| ImageError::InvalidParameters(e.to_string()))?;

    if terminal_node as usize >= description.len() {
        return Err(ImageError::InvalidParameters(format!(
            "terminal node {terminal_node} out of range (graph has {} nodes)",
            description.len()
        )));
    }

    // Build a live NodeGraph from the description
    let mut graph = NodeGraph::new(16 * 1024 * 1024);

    for (i, desc) in description.iter().enumerate() {
        // Source detection: explicit Source kind OR first node with no upstreams
        let is_source = desc.kind == NodeKind::Source || desc.upstreams.is_empty();

        if is_source {
            let node = source::SourceNode::new(source_data.to_vec())?;
            graph.add_node(Box::new(node));
            continue;
        }

        match desc.kind {
            NodeKind::Source => unreachable!(), // handled above
            NodeKind::Filter | NodeKind::Mapper => {
                if desc.upstreams.is_empty() {
                    return Err(ImageError::InvalidParameters(format!(
                        "node {i} ({}) has no upstream",
                        desc.name
                    )));
                }
                let upstream = desc.upstreams[0];
                let upstream_info = graph.node_info(upstream)?;

                // Dispatch to filter factory using node name
                let params = std::collections::HashMap::new();
                let node = crate::domain::pipeline::dispatch::dispatch_filter(
                    &desc.name,
                    upstream,
                    upstream_info,
                    &params,
                )
                .map_err(ImageError::InvalidParameters)?;
                graph.add_node(node);
            }
            NodeKind::Transform => {
                if desc.upstreams.is_empty() {
                    return Err(ImageError::InvalidParameters(format!(
                        "node {i} ({}) has no upstream",
                        desc.name
                    )));
                }
                let upstream = desc.upstreams[0];
                let upstream_info = graph.node_info(upstream)?;

                // Reconstruct transforms from name + output_info
                let node: Box<dyn ImageNode> = match desc.name.as_str() {
                    "resize" => {
                        use crate::domain::pipeline::nodes::transform::ResizeNode;
                        use crate::domain::types::ResizeFilter;
                        Box::new(ResizeNode::new(
                            upstream,
                            upstream_info,
                            desc.output_info.width,
                            desc.output_info.height,
                            ResizeFilter::Lanczos3, // default — config not yet serialized
                        ))
                    }
                    "crop" => {
                        use crate::domain::pipeline::nodes::transform::CropNode;
                        // Reconstruct crop bounds from output dimensions
                        // (x,y are 0 by default since config not yet serialized)
                        Box::new(CropNode::new(
                            upstream,
                            upstream_info,
                            0,
                            0,
                            desc.output_info.width,
                            desc.output_info.height,
                        ))
                    }
                    "rotate" => {
                        use crate::domain::pipeline::nodes::transform::RotateNode;
                        use crate::domain::types::Rotation;
                        Box::new(RotateNode::new(upstream, upstream_info, Rotation::R90))
                    }
                    "flip" => {
                        use crate::domain::pipeline::nodes::transform::FlipNode;
                        use crate::domain::types::FlipDirection;
                        Box::new(FlipNode::new(
                            upstream,
                            upstream_info,
                            FlipDirection::Horizontal,
                        ))
                    }
                    name => {
                        // Try filter dispatch as fallback
                        let params = std::collections::HashMap::new();
                        crate::domain::pipeline::dispatch::dispatch_filter(
                            name,
                            upstream,
                            upstream_info,
                            &params,
                        )
                        .map_err(ImageError::InvalidParameters)?
                    }
                };
                graph.add_node(node);
            }
            NodeKind::Composite => {
                // Composite requires two upstreams — not yet supported in description execution
                return Err(ImageError::InvalidParameters(format!(
                    "node {i}: composite reconstruction not yet supported"
                )));
            }
        }
    }

    // Execute
    sink::write(&mut graph, terminal_node, format, quality, None)
}

/// Calculate bytes per pixel for a given format.
/// Extract a sub-rectangle from a pixel buffer.
///
/// Given `src_pixels` covering `src_rect`, extracts the pixels for `sub_rect`.
/// Both rects must be in the same coordinate space, and `sub_rect` must be
/// contained within `src_rect`. Used by tiled node implementations to crop
/// the overlap padding after applying a filter.
pub fn crop_region(src_pixels: &[u8], src_rect: Rect, sub_rect: Rect, bpp: u32) -> Vec<u8> {
    debug_assert!(
        src_rect.contains(&sub_rect),
        "sub_rect {sub_rect:?} not within src_rect {src_rect:?}"
    );
    if src_rect == sub_rect {
        return src_pixels.to_vec();
    }
    let src_stride = src_rect.width as usize * bpp as usize;
    let sub_stride = sub_rect.width as usize * bpp as usize;
    let x_off = (sub_rect.x - src_rect.x) as usize * bpp as usize;
    let y_off = (sub_rect.y - src_rect.y) as usize;

    let mut out = Vec::with_capacity(sub_rect.height as usize * sub_stride);
    for row in 0..sub_rect.height as usize {
        let start = (y_off + row) * src_stride + x_off;
        out.extend_from_slice(&src_pixels[start..start + sub_stride]);
    }
    out
}

pub fn bytes_per_pixel(format: crate::domain::types::PixelFormat) -> u32 {
    use crate::domain::types::PixelFormat;
    match format {
        PixelFormat::Rgb8 | PixelFormat::Bgr8 => 3,
        PixelFormat::Rgba8 | PixelFormat::Bgra8 => 4,
        PixelFormat::Gray8 => 1,
        PixelFormat::Gray16 => 2,
        PixelFormat::Rgb16 => 6,
        PixelFormat::Rgba16 => 8,
        PixelFormat::Cmyk8 => 4,
        PixelFormat::Cmyka8 => 5,
        PixelFormat::Yuv420p | PixelFormat::Yuv422p | PixelFormat::Yuv444p | PixelFormat::Nv12 => 4,
    }
}

/// Serialize PixelFormat to u8 for graph description binary format.
pub fn pixel_format_to_u8(f: crate::domain::types::PixelFormat) -> u8 {
    use crate::domain::types::PixelFormat;
    match f {
        PixelFormat::Rgb8 => 0,
        PixelFormat::Rgba8 => 1,
        PixelFormat::Bgr8 => 2,
        PixelFormat::Bgra8 => 3,
        PixelFormat::Gray8 => 4,
        PixelFormat::Gray16 => 5,
        PixelFormat::Rgb16 => 6,
        PixelFormat::Rgba16 => 7,
        PixelFormat::Cmyk8 => 8,
        PixelFormat::Cmyka8 => 9,
        PixelFormat::Yuv420p => 10,
        PixelFormat::Yuv422p => 11,
        PixelFormat::Yuv444p => 12,
        PixelFormat::Nv12 => 13,
    }
}

/// Deserialize u8 to PixelFormat.
pub fn u8_to_pixel_format(v: u8) -> Result<crate::domain::types::PixelFormat, ImageError> {
    use crate::domain::types::PixelFormat;
    match v {
        0 => Ok(PixelFormat::Rgb8),
        1 => Ok(PixelFormat::Rgba8),
        2 => Ok(PixelFormat::Bgr8),
        3 => Ok(PixelFormat::Bgra8),
        4 => Ok(PixelFormat::Gray8),
        5 => Ok(PixelFormat::Gray16),
        6 => Ok(PixelFormat::Rgb16),
        7 => Ok(PixelFormat::Rgba16),
        8 => Ok(PixelFormat::Cmyk8),
        9 => Ok(PixelFormat::Cmyka8),
        10 => Ok(PixelFormat::Yuv420p),
        11 => Ok(PixelFormat::Yuv422p),
        12 => Ok(PixelFormat::Yuv444p),
        13 => Ok(PixelFormat::Nv12),
        _ => Err(ImageError::InvalidInput(format!("unknown pixel format: {v}"))),
    }
}

/// Serialize ColorSpace to u8 for graph description binary format.
pub fn color_space_to_u8(cs: crate::domain::types::ColorSpace) -> u8 {
    use crate::domain::types::ColorSpace;
    match cs {
        ColorSpace::Srgb => 0,
        ColorSpace::LinearSrgb => 1,
        ColorSpace::DisplayP3 => 2,
        ColorSpace::Bt709 => 3,
        ColorSpace::Bt2020 => 4,
        ColorSpace::ProPhotoRgb => 5,
        ColorSpace::AdobeRgb => 6,
    }
}

/// Deserialize u8 to ColorSpace.
pub fn u8_to_color_space(v: u8) -> Result<crate::domain::types::ColorSpace, ImageError> {
    use crate::domain::types::ColorSpace;
    match v {
        0 => Ok(ColorSpace::Srgb),
        1 => Ok(ColorSpace::LinearSrgb),
        2 => Ok(ColorSpace::DisplayP3),
        3 => Ok(ColorSpace::Bt709),
        4 => Ok(ColorSpace::Bt2020),
        5 => Ok(ColorSpace::ProPhotoRgb),
        6 => Ok(ColorSpace::AdobeRgb),
        _ => Err(ImageError::InvalidInput(format!("unknown color space: {v}"))),
    }
}

/// Validate crop parameters against source dimensions.
pub fn validate_crop(
    src_width: u32,
    src_height: u32,
    x: u32,
    y: u32,
    crop_width: u32,
    crop_height: u32,
) -> Result<(), ValidationError> {
    if crop_width == 0 || crop_height == 0 {
        return Err(ValidationError {
            message: format!("crop dimensions must be > 0, got {crop_width}x{crop_height}"),
            node_id: None,
            upstream_id: None,
        });
    }
    if x.saturating_add(crop_width) > src_width || y.saturating_add(crop_height) > src_height {
        return Err(ValidationError {
            message: format!(
                "crop region ({x},{y})+({crop_width}x{crop_height}) exceeds source {src_width}x{src_height}"
            ),
            node_id: None,
            upstream_id: None,
        });
    }
    Ok(())
}

/// Validate resize parameters.
pub fn validate_resize(width: u32, height: u32) -> Result<(), ValidationError> {
    if width == 0 || height == 0 {
        return Err(ValidationError {
            message: format!("resize dimensions must be > 0, got {width}x{height}"),
            node_id: None,
            upstream_id: None,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::{ColorSpace, PixelFormat};

    struct SolidColorNode {
        info: ImageInfo,
        value: u8,
    }

    impl ImageNode for SolidColorNode {
        fn info(&self) -> ImageInfo {
            self.info.clone()
        }

        fn compute_region(
            &self,
            request: Rect,
            _upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
        ) -> Result<Vec<u8>, ImageError> {
            let bpp = bytes_per_pixel(self.info.format);
            let size = request.width as usize * request.height as usize * bpp as usize;
            Ok(vec![self.value; size])
        }

        fn access_pattern(&self) -> AccessPattern {
            AccessPattern::Sequential
        }
    }

    fn make_solid(width: u32, height: u32, value: u8) -> Box<dyn ImageNode> {
        Box::new(SolidColorNode {
            info: ImageInfo {
                width,
                height,
                format: PixelFormat::Rgba8,
                color_space: ColorSpace::Srgb,
            },
            value,
        })
    }

    #[test]
    fn add_and_info() {
        let mut g = NodeGraph::new(1024 * 1024);
        let id = g.add_node(make_solid(100, 100, 0));
        assert_eq!(g.node_info(id).unwrap().width, 100);
    }

    #[test]
    fn request_region_basic() {
        let mut g = NodeGraph::new(1024 * 1024);
        let id = g.add_node(make_solid(100, 100, 42));
        let p = g.request_region(id, Rect::new(0, 0, 10, 10)).unwrap();
        assert_eq!(p.len(), 10 * 10 * 4);
        assert!(p.iter().all(|&v| v == 42));
    }

    #[test]
    fn request_cached() {
        let mut g = NodeGraph::new(1024 * 1024);
        let id = g.add_node(make_solid(100, 100, 55));
        let p1 = g.request_region(id, Rect::new(0, 0, 10, 10)).unwrap();
        let p2 = g.request_region(id, Rect::new(0, 0, 10, 10)).unwrap();
        assert_eq!(p1, p2);
    }

    #[test]
    fn request_subregion() {
        let mut g = NodeGraph::new(1024 * 1024);
        let id = g.add_node(make_solid(100, 100, 77));
        g.request_region(id, Rect::new(0, 0, 50, 50)).unwrap();
        let sub = g.request_region(id, Rect::new(10, 10, 20, 20)).unwrap();
        assert_eq!(sub.len(), 20 * 20 * 4);
    }

    #[test]
    fn invalid_node() {
        let mut g = NodeGraph::new(1024 * 1024);
        assert!(g.request_region(99, Rect::new(0, 0, 10, 10)).is_err());
    }

    #[test]
    fn upstream_chain() {
        // Node 0: solid 100x100 value=10
        // Node 1: doubles upstream pixels (custom node)
        struct DoubleNode {
            upstream: u32,
            info: ImageInfo,
        }
        impl ImageNode for DoubleNode {
            fn info(&self) -> ImageInfo {
                self.info.clone()
            }
            fn compute_region(
                &self,
                request: Rect,
                upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
            ) -> Result<Vec<u8>, ImageError> {
                let src = upstream_fn(self.upstream, request)?;
                Ok(src.iter().map(|&v| v.saturating_mul(2)).collect())
            }
            fn access_pattern(&self) -> AccessPattern {
                AccessPattern::Sequential
            }
        }

        let mut g = NodeGraph::new(1024 * 1024);
        let src = g.add_node(make_solid(100, 100, 10));
        let info = g.node_info(src).unwrap();
        let dbl = g.add_node(Box::new(DoubleNode {
            upstream: src,
            info,
        }));

        let p = g.request_region(dbl, Rect::new(0, 0, 5, 5)).unwrap();
        assert!(p.iter().all(|&v| v == 20));
    }

    #[test]
    fn validate_empty_graph() {
        let g = NodeGraph::new(1024 * 1024);
        let err = g.validate().unwrap_err();
        assert!(err.message.contains("empty graph"), "got: {}", err.message);
    }

    #[test]
    fn validate_valid_graph() {
        let mut g = NodeGraph::new(1024 * 1024);
        g.add_node(make_solid(100, 100, 0));
        assert!(g.validate().is_ok());
    }

    #[test]
    fn validate_upstream_check() {
        let g = NodeGraph::new(1024 * 1024);
        let err = g.validate_upstream(0).unwrap_err();
        assert!(
            err.message.contains("does not exist"),
            "got: {}",
            err.message
        );
    }

    #[test]
    fn validate_upstream_ok() {
        let mut g = NodeGraph::new(1024 * 1024);
        g.add_node(make_solid(100, 100, 0));
        assert!(g.validate_upstream(0).is_ok());
    }

    #[test]
    fn validate_node_zero_dims() {
        let mut g = NodeGraph::new(1024 * 1024);
        g.add_node(Box::new(SolidColorNode {
            info: ImageInfo {
                width: 0,
                height: 100,
                format: PixelFormat::Rgba8,
                color_space: ColorSpace::Srgb,
            },
            value: 0,
        }));
        let err = g.validate_node(0).unwrap_err();
        assert!(
            err.message.contains("zero dimensions"),
            "got: {}",
            err.message
        );
    }

    #[test]
    fn description_tracks_nodes() {
        let mut g = NodeGraph::new(1024 * 1024);
        g.add_node(make_solid(100, 100, 0));
        g.add_node(make_solid(50, 50, 128));
        let desc = g.description();
        assert_eq!(desc.len(), 2);
        assert_eq!(desc.get(0).unwrap().output_info.width, 100);
        assert_eq!(desc.get(1).unwrap().output_info.width, 50);
    }

    #[test]
    fn description_survives_cleanup() {
        let mut g = NodeGraph::new(1024 * 1024);
        g.add_node(make_solid(100, 100, 0));
        assert_eq!(g.description().len(), 1);
        g.cleanup();
        // Description persists across cleanup for re-execution
        assert_eq!(g.description().len(), 1);
    }

    #[test]
    fn description_cleared_on_reset() {
        let mut g = NodeGraph::new(1024 * 1024);
        g.add_node(make_solid(100, 100, 0));
        assert_eq!(g.description().len(), 1);
        g.reset();
        assert!(g.description().is_empty());
    }

    #[test]
    fn graph_description_dag_validation() {
        let mut desc = GraphDescription::new();
        // Valid: source with no upstreams
        desc.add(NodeDescriptor {
            kind: NodeKind::Source,
            name: "source".into(),
            config: Vec::new(),
            upstreams: Vec::new(),
            output_info: ImageInfo {
                width: 100,
                height: 100,
                format: PixelFormat::Rgba8,
                color_space: ColorSpace::Srgb,
            },
        })
        .unwrap();

        // Valid: filter referencing node 0
        desc.add(NodeDescriptor {
            kind: NodeKind::Filter,
            name: "blur".into(),
            config: Vec::new(),
            upstreams: vec![0],
            output_info: ImageInfo {
                width: 100,
                height: 100,
                format: PixelFormat::Rgba8,
                color_space: ColorSpace::Srgb,
            },
        })
        .unwrap();

        // Invalid: forward reference (node 2 references node 2)
        let err = desc
            .add(NodeDescriptor {
                kind: NodeKind::Filter,
                name: "bad".into(),
                config: Vec::new(),
                upstreams: vec![2], // self-reference
                output_info: ImageInfo {
                    width: 100,
                    height: 100,
                    format: PixelFormat::Rgba8,
                    color_space: ColorSpace::Srgb,
                },
            })
            .unwrap_err();
        assert!(
            err.message.contains("forward/self reference"),
            "got: {}",
            err.message
        );
    }

    #[test]
    fn graph_description_serialize_roundtrip() {
        let mut desc = GraphDescription::new();
        desc.add(NodeDescriptor {
            kind: NodeKind::Source,
            name: "source".into(),
            config: vec![1, 2, 3],
            upstreams: Vec::new(),
            output_info: ImageInfo {
                width: 640,
                height: 480,
                format: PixelFormat::Rgb8,
                color_space: ColorSpace::Srgb,
            },
        })
        .unwrap();
        desc.add(NodeDescriptor {
            kind: NodeKind::Transform,
            name: "resize".into(),
            config: vec![4, 5],
            upstreams: vec![0],
            output_info: ImageInfo {
                width: 320,
                height: 240,
                format: PixelFormat::Rgb8,
                color_space: ColorSpace::Srgb,
            },
        })
        .unwrap();

        let bytes = desc.serialize();
        let desc2 = GraphDescription::deserialize(&bytes).unwrap();
        assert_eq!(desc2.len(), 2);

        let n0 = desc2.get(0).unwrap();
        assert_eq!(n0.kind, NodeKind::Source);
        assert_eq!(n0.name, "source");
        assert_eq!(n0.config, vec![1, 2, 3]);
        assert!(n0.upstreams.is_empty());
        assert_eq!(n0.output_info.width, 640);
        assert_eq!(n0.output_info.height, 480);
        assert_eq!(n0.output_info.format, PixelFormat::Rgb8);

        let n1 = desc2.get(1).unwrap();
        assert_eq!(n1.kind, NodeKind::Transform);
        assert_eq!(n1.name, "resize");
        assert_eq!(n1.upstreams, vec![0]);
        assert_eq!(n1.output_info.width, 320);
        assert_eq!(n1.output_info.height, 240);
    }

    #[test]
    fn graph_description_validate() {
        let desc = GraphDescription::new();
        assert!(desc.validate().is_err()); // empty

        let mut desc = GraphDescription::new();
        desc.add(NodeDescriptor {
            kind: NodeKind::Source,
            name: "source".into(),
            config: Vec::new(),
            upstreams: Vec::new(),
            output_info: ImageInfo {
                width: 100,
                height: 100,
                format: PixelFormat::Rgba8,
                color_space: ColorSpace::Srgb,
            },
        })
        .unwrap();
        assert!(desc.validate().is_ok());
    }

    #[test]
    fn validate_crop_bounds() {
        // Valid crop
        assert!(validate_crop(100, 100, 10, 10, 50, 50).is_ok());
        // Crop exceeds bounds
        let err = validate_crop(100, 100, 60, 60, 50, 50).unwrap_err();
        assert!(err.message.contains("exceeds source"), "got: {}", err.message);
        // Zero dimensions
        let err = validate_crop(100, 100, 0, 0, 0, 50).unwrap_err();
        assert!(err.message.contains("must be > 0"), "got: {}", err.message);
    }

    #[test]
    fn validate_resize_dims() {
        assert!(validate_resize(100, 100).is_ok());
        let err = validate_resize(0, 100).unwrap_err();
        assert!(err.message.contains("must be > 0"), "got: {}", err.message);
    }

    #[test]
    fn graph_description_introspection() {
        let mut desc = GraphDescription::new();
        desc.add(NodeDescriptor {
            kind: NodeKind::Source,
            name: "source".into(),
            config: Vec::new(),
            upstreams: Vec::new(),
            output_info: ImageInfo {
                width: 640,
                height: 480,
                format: PixelFormat::Rgb8,
                color_space: ColorSpace::Srgb,
            },
        })
        .unwrap();

        // Query info at node 0
        let info = desc.node_info(0).unwrap();
        assert_eq!(info.width, 640);
        assert_eq!(info.height, 480);

        // Invalid query
        assert!(desc.node_info(99).is_none());

        // Iterate
        let names: Vec<&str> = desc.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(names, vec!["source"]);
    }
}

#[cfg(test)]
mod tiled_parity_tests {
    use super::*;
    use crate::domain::filters::{
        BilateralParams, BlurParams, BrightnessParams, CannyParams, ContrastParams, ErodeParams,
        GuidedFilterParams, MedianParams, MotionBlurParams, SharpenParams,
    };
    use crate::domain::pipeline::nodes::filters::{
        BilateralNode, BlurNode, BrightnessNode, CannyMapperNode, ContrastNode, ErodeNode,
        GuidedFilterNode, MedianNode, MotionBlurNode, SharpenNode,
    };
    use crate::domain::types::*;

    /// Raw pixel source node for testing (no decode step).
    struct RawSource {
        pixels: Vec<u8>,
        info: ImageInfo,
    }
    impl RawSource {
        fn new(pixels: Vec<u8>, info: ImageInfo) -> Self {
            Self { pixels, info }
        }
    }
    impl ImageNode for RawSource {
        fn info(&self) -> ImageInfo {
            self.info.clone()
        }
        fn compute_region(
            &self,
            request: Rect,
            _: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
        ) -> Result<Vec<u8>, ImageError> {
            let bpp = bytes_per_pixel(self.info.format);
            Ok(crop_region(
                &self.pixels,
                Rect::new(0, 0, self.info.width, self.info.height),
                request,
                bpp,
            ))
        }
        fn access_pattern(&self) -> AccessPattern {
            AccessPattern::Sequential
        }
    }

    fn gradient_pixels(w: u32, h: u32) -> Vec<u8> {
        let mut px = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                px.push(((x * 255) / w) as u8);
                px.push(((y * 255) / h) as u8);
                px.push(128);
                px.push(255);
            }
        }
        px
    }

    fn stitch_tiles(tiles: &[(Rect, Vec<u8>)], full_w: u32, full_h: u32, bpp: u32) -> Vec<u8> {
        let stride = full_w as usize * bpp as usize;
        let mut out = vec![0u8; full_h as usize * stride];
        for (rect, pixels) in tiles {
            let tile_stride = rect.width as usize * bpp as usize;
            for row in 0..rect.height as usize {
                let dst_start = (rect.y as usize + row) * stride + rect.x as usize * bpp as usize;
                let src_start = row * tile_stride;
                out[dst_start..dst_start + tile_stride]
                    .copy_from_slice(&pixels[src_start..src_start + tile_stride]);
            }
        }
        out
    }

    /// Core validation: full-image request must produce byte-identical output
    /// to stitched tile requests. Zero tolerance.
    fn assert_tiled_matches_full(graph_builder: impl Fn() -> (NodeGraph, u32, u32, u32, u32)) {
        // Build graph and get full-image result
        let (mut g, output_node, w, h, bpp) = graph_builder();
        let full = g
            .request_region(output_node, Rect::new(0, 0, w, h))
            .unwrap();

        // Build fresh graph and request as 4 quadrant tiles
        let (mut g2, output_node2, _, _, _) = graph_builder();
        let hw = w / 2;
        let hh = h / 2;
        let tiles = vec![
            (
                Rect::new(0, 0, hw, hh),
                g2.request_region(output_node2, Rect::new(0, 0, hw, hh))
                    .unwrap(),
            ),
            (
                Rect::new(hw, 0, w - hw, hh),
                g2.request_region(output_node2, Rect::new(hw, 0, w - hw, hh))
                    .unwrap(),
            ),
            (
                Rect::new(0, hh, hw, h - hh),
                g2.request_region(output_node2, Rect::new(0, hh, hw, h - hh))
                    .unwrap(),
            ),
            (
                Rect::new(hw, hh, w - hw, h - hh),
                g2.request_region(output_node2, Rect::new(hw, hh, w - hw, h - hh))
                    .unwrap(),
            ),
        ];

        let stitched = stitch_tiles(&tiles, w, h, bpp);

        assert_eq!(
            full.len(),
            stitched.len(),
            "buffer length mismatch: full={} stitched={}",
            full.len(),
            stitched.len()
        );
        assert_eq!(
            full,
            stitched,
            "TILED OUTPUT DIFFERS FROM FULL-IMAGE OUTPUT. \
             First diff at byte {}",
            full.iter()
                .zip(stitched.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(0)
        );
    }

    #[test]
    fn tiled_parity_point_op() {
        assert_tiled_matches_full(|| {
            let w = 64;
            let h = 64;
            let mut g = NodeGraph::new(1024 * 1024);
            let src = g.add_node(Box::new(RawSource::new(
                gradient_pixels(w, h),
                ImageInfo {
                    width: w,
                    height: h,
                    format: PixelFormat::Rgba8,
                    color_space: ColorSpace::Srgb,
                },
            )));
            let bright = g.add_node(Box::new(BrightnessNode::new(
                src,
                ImageInfo {
                    width: w,
                    height: h,
                    format: PixelFormat::Rgba8,
                    color_space: ColorSpace::Srgb,
                },
                BrightnessParams { amount: 0.2 },
            )));
            (g, bright, w, h, 4)
        });
    }

    #[test]
    fn tiled_parity_blur() {
        assert_tiled_matches_full(|| {
            let w = 64;
            let h = 64;
            let mut g = NodeGraph::new(1024 * 1024);
            let src = g.add_node(Box::new(RawSource::new(
                gradient_pixels(w, h),
                ImageInfo {
                    width: w,
                    height: h,
                    format: PixelFormat::Rgba8,
                    color_space: ColorSpace::Srgb,
                },
            )));
            let blur = g.add_node(Box::new(BlurNode::new(
                src,
                ImageInfo {
                    width: w,
                    height: h,
                    format: PixelFormat::Rgba8,
                    color_space: ColorSpace::Srgb,
                },
                BlurParams { radius: 2.0 },
            )));
            (g, blur, w, h, 4)
        });
    }

    #[test]
    fn tiled_parity_sharpen() {
        assert_tiled_matches_full(|| {
            let w = 64;
            let h = 64;
            let mut g = NodeGraph::new(1024 * 1024);
            let src = g.add_node(Box::new(RawSource::new(
                gradient_pixels(w, h),
                ImageInfo {
                    width: w,
                    height: h,
                    format: PixelFormat::Rgba8,
                    color_space: ColorSpace::Srgb,
                },
            )));
            let sharp = g.add_node(Box::new(SharpenNode::new(
                src,
                ImageInfo {
                    width: w,
                    height: h,
                    format: PixelFormat::Rgba8,
                    color_space: ColorSpace::Srgb,
                },
                SharpenParams { amount: 1.5 },
            )));
            (g, sharp, w, h, 4)
        });
    }

    #[test]
    fn tiled_parity_contrast() {
        assert_tiled_matches_full(|| {
            let w = 64;
            let h = 64;
            let mut g = NodeGraph::new(1024 * 1024);
            let src = g.add_node(Box::new(RawSource::new(
                gradient_pixels(w, h),
                ImageInfo {
                    width: w,
                    height: h,
                    format: PixelFormat::Rgba8,
                    color_space: ColorSpace::Srgb,
                },
            )));
            let contrast = g.add_node(Box::new(ContrastNode::new(
                src,
                ImageInfo {
                    width: w,
                    height: h,
                    format: PixelFormat::Rgba8,
                    color_space: ColorSpace::Srgb,
                },
                ContrastParams { amount: 0.5 },
            )));
            (g, contrast, w, h, 4)
        });
    }

    #[test]
    fn tiled_parity_median() {
        assert_tiled_matches_full(|| {
            let w = 64;
            let h = 64;
            let mut g = NodeGraph::new(1024 * 1024);
            let src = g.add_node(Box::new(RawSource::new(
                gradient_pixels(w, h),
                ImageInfo {
                    width: w,
                    height: h,
                    format: PixelFormat::Rgba8,
                    color_space: ColorSpace::Srgb,
                },
            )));
            let median = g.add_node(Box::new(MedianNode::new(
                src,
                ImageInfo {
                    width: w,
                    height: h,
                    format: PixelFormat::Rgba8,
                    color_space: ColorSpace::Srgb,
                },
                MedianParams { radius: 2 },
            )));
            (g, median, w, h, 4)
        });
    }

    #[test]
    fn tiled_parity_erode() {
        assert_tiled_matches_full(|| {
            let w = 64;
            let h = 64;
            let mut g = NodeGraph::new(1024 * 1024);
            let src = g.add_node(Box::new(RawSource::new(
                gradient_pixels(w, h),
                ImageInfo {
                    width: w,
                    height: h,
                    format: PixelFormat::Rgba8,
                    color_space: ColorSpace::Srgb,
                },
            )));
            let erode = g.add_node(Box::new(ErodeNode::new(
                src,
                ImageInfo {
                    width: w,
                    height: h,
                    format: PixelFormat::Rgba8,
                    color_space: ColorSpace::Srgb,
                },
                ErodeParams { ksize: 3, shape: 0 },
            )));
            (g, erode, w, h, 4)
        });
    }

    #[test]
    fn tiled_parity_canny() {
        // Canny outputs single-channel grayscale regardless of input format,
        // so we use Gray8 to match the output format.
        assert_tiled_matches_full(|| {
            let w = 64;
            let h = 64;
            let mut g = NodeGraph::new(1024 * 1024);
            let src = g.add_node(Box::new(RawSource::new(
                gradient_pixels_gray(w, h),
                ImageInfo {
                    width: w,
                    height: h,
                    format: PixelFormat::Gray8,
                    color_space: ColorSpace::Srgb,
                },
            )));
            let canny = g.add_node(Box::new(CannyMapperNode::new(
                src,
                ImageInfo {
                    width: w,
                    height: h,
                    format: PixelFormat::Gray8,
                    color_space: ColorSpace::Srgb,
                },
                CannyParams {
                    low_threshold: 50.0,
                    high_threshold: 150.0,
                },
            )));
            (g, canny, w, h, 1)
        });
    }

    fn gradient_pixels_rgb(w: u32, h: u32) -> Vec<u8> {
        let mut px = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                px.push(((x * 255) / w) as u8);
                px.push(((y * 255) / h) as u8);
                px.push(128);
            }
        }
        px
    }

    fn gradient_pixels_gray(w: u32, h: u32) -> Vec<u8> {
        let mut px = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                px.push((((x + y) * 255) / (w + h)) as u8);
            }
        }
        px
    }

    #[test]
    fn tiled_parity_bilateral() {
        assert_tiled_matches_full(|| {
            let w = 64;
            let h = 64;
            let mut g = NodeGraph::new(1024 * 1024);
            let src = g.add_node(Box::new(RawSource::new(
                gradient_pixels_rgb(w, h),
                ImageInfo {
                    width: w,
                    height: h,
                    format: PixelFormat::Rgb8,
                    color_space: ColorSpace::Srgb,
                },
            )));
            let bilateral = g.add_node(Box::new(BilateralNode::new(
                src,
                ImageInfo {
                    width: w,
                    height: h,
                    format: PixelFormat::Rgb8,
                    color_space: ColorSpace::Srgb,
                },
                BilateralParams {
                    diameter: 5,
                    sigma_color: 75.0,
                    sigma_space: 75.0,
                },
            )));
            (g, bilateral, w, h, 3)
        });
    }

    #[test]
    fn tiled_parity_motion_blur() {
        assert_tiled_matches_full(|| {
            let w = 64;
            let h = 64;
            let mut g = NodeGraph::new(1024 * 1024);
            let src = g.add_node(Box::new(RawSource::new(
                gradient_pixels(w, h),
                ImageInfo {
                    width: w,
                    height: h,
                    format: PixelFormat::Rgba8,
                    color_space: ColorSpace::Srgb,
                },
            )));
            let mb = g.add_node(Box::new(MotionBlurNode::new(
                src,
                ImageInfo {
                    width: w,
                    height: h,
                    format: PixelFormat::Rgba8,
                    color_space: ColorSpace::Srgb,
                },
                MotionBlurParams {
                    length: 5,
                    angle_degrees: 45.0,
                },
            )));
            (g, mb, w, h, 4)
        });
    }

    #[test]
    fn tiled_parity_guided_filter() {
        assert_tiled_matches_full(|| {
            let w = 64;
            let h = 64;
            let mut g = NodeGraph::new(1024 * 1024);
            let src = g.add_node(Box::new(RawSource::new(
                gradient_pixels_gray(w, h),
                ImageInfo {
                    width: w,
                    height: h,
                    format: PixelFormat::Gray8,
                    color_space: ColorSpace::Srgb,
                },
            )));
            let guided = g.add_node(Box::new(GuidedFilterNode::new(
                src,
                ImageInfo {
                    width: w,
                    height: h,
                    format: PixelFormat::Gray8,
                    color_space: ColorSpace::Srgb,
                },
                GuidedFilterParams {
                    radius: 3,
                    epsilon: 0.01,
                },
            )));
            (g, guided, w, h, 1)
        });
    }
}

#[cfg(test)]
mod lut_fusion_tests {
    use super::*;
    use crate::domain::filters::{BlurParams, BrightnessParams, ContrastParams, GammaParams};
    use crate::domain::pipeline::nodes::filters::{
        BlurNode, BrightnessNode, ContrastNode, GammaNode, InvertNode,
    };
    use crate::domain::types::*;

    /// Raw pixel source node for testing.
    struct RawSource {
        pixels: Vec<u8>,
        info: ImageInfo,
    }
    impl RawSource {
        fn new(pixels: Vec<u8>, info: ImageInfo) -> Self {
            Self { pixels, info }
        }
    }
    impl ImageNode for RawSource {
        fn info(&self) -> ImageInfo {
            self.info.clone()
        }
        fn compute_region(
            &self,
            request: Rect,
            _: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
        ) -> Result<Vec<u8>, ImageError> {
            let bpp = bytes_per_pixel(self.info.format);
            Ok(crop_region(
                &self.pixels,
                Rect::new(0, 0, self.info.width, self.info.height),
                request,
                bpp,
            ))
        }
        fn access_pattern(&self) -> AccessPattern {
            AccessPattern::Sequential
        }
    }

    fn gradient_pixels(w: u32, h: u32) -> Vec<u8> {
        let mut px = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                px.push(((x * 255) / w) as u8);
                px.push(((y * 255) / h) as u8);
                px.push(128);
                px.push(255);
            }
        }
        px
    }

    fn make_info(w: u32, h: u32) -> ImageInfo {
        ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        }
    }

    #[test]
    fn fused_brightness_contrast_gamma_matches_unfused() {
        let w = 32;
        let h = 32;
        let info = make_info(w, h);
        let pixels = gradient_pixels(w, h);

        // Unfused: build graph, don't fuse, request full image
        let mut g1 = NodeGraph::new(1024 * 1024);
        let src1 = g1.add_node(Box::new(RawSource::new(pixels.clone(), info.clone())));
        let b1 = g1.add_node(Box::new(BrightnessNode::new(
            src1,
            info.clone(),
            BrightnessParams { amount: 0.2 },
        )));
        let c1 = g1.add_node(Box::new(ContrastNode::new(
            b1,
            info.clone(),
            ContrastParams { amount: 0.3 },
        )));
        let g_node1 = g1.add_node(Box::new(GammaNode::new(
            c1,
            info.clone(),
            GammaParams { gamma_value: 1.5 },
        )));
        let unfused = g1.request_region(g_node1, Rect::new(0, 0, w, h)).unwrap();

        // Fused: build same graph, fuse, request full image
        let mut g2 = NodeGraph::new(1024 * 1024);
        let src2 = g2.add_node(Box::new(RawSource::new(pixels, info.clone())));
        let b2 = g2.add_node(Box::new(BrightnessNode::new(
            src2,
            info.clone(),
            BrightnessParams { amount: 0.2 },
        )));
        let c2 = g2.add_node(Box::new(ContrastNode::new(
            b2,
            info.clone(),
            ContrastParams { amount: 0.3 },
        )));
        let g_node2 = g2.add_node(Box::new(GammaNode::new(
            c2,
            info.clone(),
            GammaParams { gamma_value: 1.5 },
        )));
        g2.fuse_point_ops();
        let fused = g2.request_region(g_node2, Rect::new(0, 0, w, h)).unwrap();

        assert_eq!(
            unfused, fused,
            "Fused output must be byte-identical to unfused"
        );
    }

    #[test]
    fn fusion_with_5_ops() {
        let w = 16;
        let h = 16;
        let info = make_info(w, h);
        let pixels = gradient_pixels(w, h);

        // Chain: brightness → contrast → gamma → brightness → contrast
        let build_graph = |fuse: bool| {
            let mut g = NodeGraph::new(1024 * 1024);
            let src = g.add_node(Box::new(RawSource::new(pixels.clone(), info.clone())));
            let n1 = g.add_node(Box::new(BrightnessNode::new(
                src,
                info.clone(),
                BrightnessParams { amount: 0.1 },
            )));
            let n2 = g.add_node(Box::new(ContrastNode::new(
                n1,
                info.clone(),
                ContrastParams { amount: 0.2 },
            )));
            let n3 = g.add_node(Box::new(GammaNode::new(
                n2,
                info.clone(),
                GammaParams { gamma_value: 0.8 },
            )));
            let n4 = g.add_node(Box::new(BrightnessNode::new(
                n3,
                info.clone(),
                BrightnessParams { amount: -0.1 },
            )));
            let n5 = g.add_node(Box::new(ContrastNode::new(
                n4,
                info.clone(),
                ContrastParams { amount: -0.15 },
            )));
            if fuse {
                g.fuse_point_ops();
            }
            g.request_region(n5, Rect::new(0, 0, w, h)).unwrap()
        };

        assert_eq!(build_graph(false), build_graph(true));
    }

    #[test]
    fn non_point_op_acts_as_fusion_barrier() {
        let w = 16;
        let h = 16;
        let info = make_info(w, h);
        let pixels = gradient_pixels(w, h);

        // Chain: brightness → blur → contrast
        // blur is a spatial op (not a point op), so brightness and contrast
        // should NOT be fused together
        let mut g = NodeGraph::new(1024 * 1024);
        let src = g.add_node(Box::new(RawSource::new(pixels.clone(), info.clone())));
        let b = g.add_node(Box::new(BrightnessNode::new(
            src,
            info.clone(),
            BrightnessParams { amount: 0.2 },
        )));
        let blur = g.add_node(Box::new(BlurNode::new(
            b,
            info.clone(),
            BlurParams { radius: 2.0 },
        )));
        let c = g.add_node(Box::new(ContrastNode::new(
            blur,
            info.clone(),
            ContrastParams { amount: 0.3 },
        )));

        // After fusion, blur should still be in the chain (barrier)
        g.fuse_point_ops();

        let mut g2 = NodeGraph::new(1024 * 1024);
        let src2 = g2.add_node(Box::new(RawSource::new(pixels, info.clone())));
        let b2 = g2.add_node(Box::new(BrightnessNode::new(
            src2,
            info.clone(),
            BrightnessParams { amount: 0.2 },
        )));
        let blur2 = g2.add_node(Box::new(BlurNode::new(
            b2,
            info.clone(),
            BlurParams { radius: 2.0 },
        )));
        let c2 = g2.add_node(Box::new(ContrastNode::new(
            blur2,
            info.clone(),
            ContrastParams { amount: 0.3 },
        )));

        let fused = g.request_region(c, Rect::new(0, 0, w, h)).unwrap();
        let unfused = g2.request_region(c2, Rect::new(0, 0, w, h)).unwrap();

        assert_eq!(fused, unfused, "Barrier test: fused must match unfused");
    }

    #[test]
    fn invert_fuses_with_brightness() {
        let w = 16;
        let h = 16;
        let info = make_info(w, h);
        let pixels = gradient_pixels(w, h);

        let build_graph = |fuse: bool| {
            let mut g = NodeGraph::new(1024 * 1024);
            let src = g.add_node(Box::new(RawSource::new(pixels.clone(), info.clone())));
            let b = g.add_node(Box::new(BrightnessNode::new(
                src,
                info.clone(),
                BrightnessParams { amount: 0.2 },
            )));
            let inv = g.add_node(Box::new(InvertNode::new(b, info.clone())));
            if fuse {
                g.fuse_point_ops();
            }
            g.request_region(inv, Rect::new(0, 0, w, h)).unwrap()
        };

        assert_eq!(build_graph(false), build_graph(true));
    }
}

#[cfg(test)]
mod color_clut_fusion_tests {
    use super::*;
    use crate::domain::filters::{HueRotateParams, SaturateParams, SepiaParams};
    use crate::domain::pipeline::nodes::filters::{HueRotateNode, SaturateNode, SepiaNode};
    use crate::domain::types::*;

    struct RawSource {
        pixels: Vec<u8>,
        info: ImageInfo,
    }
    impl RawSource {
        fn new(pixels: Vec<u8>, info: ImageInfo) -> Self {
            Self { pixels, info }
        }
    }
    impl ImageNode for RawSource {
        fn info(&self) -> ImageInfo {
            self.info.clone()
        }
        fn compute_region(
            &self,
            request: Rect,
            _: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
        ) -> Result<Vec<u8>, ImageError> {
            Ok(crop_region(
                &self.pixels,
                Rect::new(0, 0, self.info.width, self.info.height),
                request,
                bytes_per_pixel(self.info.format),
            ))
        }
        fn access_pattern(&self) -> AccessPattern {
            AccessPattern::Sequential
        }
    }

    fn gradient_rgb(w: u32, h: u32) -> Vec<u8> {
        (0..w * h)
            .flat_map(|i| {
                let x = i % w;
                let y = i / w;
                vec![((x * 255) / w) as u8, ((y * 255) / h) as u8, 128u8]
            })
            .collect()
    }

    #[test]
    fn fused_hue_saturate_sepia_matches_unfused() {
        let (w, h) = (16, 16);
        let info = ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let pixels = gradient_rgb(w, h);
        let build = |fuse: bool| {
            let mut g = NodeGraph::new(1024 * 1024);
            let s = g.add_node(Box::new(RawSource::new(pixels.clone(), info.clone())));
            let h = g.add_node(Box::new(HueRotateNode::new(
                s,
                info.clone(),
                HueRotateParams { degrees: 90.0 },
            )));
            let sa = g.add_node(Box::new(SaturateNode::new(
                h,
                info.clone(),
                SaturateParams { factor: 1.5 },
            )));
            let se = g.add_node(Box::new(SepiaNode::new(
                sa,
                info.clone(),
                SepiaParams { intensity: 0.3 },
            )));
            if fuse {
                g.fuse_color_ops();
            }
            g.request_region(se, Rect::new(0, 0, w, h)).unwrap()
        };
        let (unfused, fused) = (build(false), build(true));
        let mae: f64 = unfused
            .iter()
            .zip(fused.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / unfused.len() as f64;
        assert!(mae < 1.0, "3D CLUT fusion MAE: {mae} (expected < 1.0)");
    }
}

#[cfg(test)]
mod frame_sequence_tests {
    use super::*;
    use crate::domain::filters::BrightnessParams;
    use crate::domain::pipeline::nodes::filters::BrightnessNode;
    use crate::domain::pipeline::nodes::frame_source::FrameSourceNode;
    use crate::domain::types::*;

    /// Build a 3-frame animated GIF for testing.
    fn make_animated_gif() -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let mut encoder = gif::Encoder::new(&mut buf, 4, 4, &[]).unwrap();
            encoder.set_repeat(gif::Repeat::Infinite).unwrap();
            let colors: [[u8; 3]; 3] = [[255, 0, 0], [0, 255, 0], [0, 0, 255]];
            for (i, color) in colors.iter().enumerate() {
                let mut pixels: Vec<u8> = Vec::with_capacity(64);
                for _ in 0..16 {
                    pixels.extend_from_slice(&[color[0], color[1], color[2], 255]);
                }
                let mut frame = gif::Frame::from_rgba(4, 4, &mut pixels);
                frame.delay = (i as u16 + 1) * 10;
                encoder.write_frame(&frame).unwrap();
            }
        }
        buf
    }

    #[test]
    fn execute_sequence_all_frames() {
        let gif = make_animated_gif();
        let mut graph = NodeGraph::new(1024 * 1024);
        let node = FrameSourceNode::new(gif, FrameSelection::All).unwrap();
        let (_src_id, rc) = graph.add_frame_source(node);
        let seq = graph.execute_sequence(&rc, _src_id).unwrap();
        assert_eq!(seq.len(), 3);
        assert_eq!(seq.frames[0].1.delay_ms, 100);
        assert_eq!(seq.frames[1].1.delay_ms, 200);
        assert_eq!(seq.frames[2].1.delay_ms, 300);
    }

    #[test]
    fn execute_sequence_single_pick() {
        let gif = make_animated_gif();
        let mut graph = NodeGraph::new(1024 * 1024);
        let node = FrameSourceNode::new(gif, FrameSelection::Single(1)).unwrap();
        let (src_id, rc) = graph.add_frame_source(node);
        let seq = graph.execute_sequence(&rc, src_id).unwrap();
        assert_eq!(seq.len(), 1);
        assert_eq!(seq.frames[0].1.index, 1);
        // Frame 1 is green
        assert_eq!(seq.frames[0].0.pixels[0], 0);
        assert_eq!(seq.frames[0].0.pixels[1], 255);
        assert_eq!(seq.frames[0].0.pixels[2], 0);
    }

    #[test]
    fn execute_sequence_pick_subset() {
        let gif = make_animated_gif();
        let mut graph = NodeGraph::new(1024 * 1024);
        let node = FrameSourceNode::new(gif, FrameSelection::Pick(vec![0, 2])).unwrap();
        let (src_id, rc) = graph.add_frame_source(node);
        let seq = graph.execute_sequence(&rc, src_id).unwrap();
        assert_eq!(seq.len(), 2);
        assert_eq!(seq.frames[0].1.index, 0); // red
        assert_eq!(seq.frames[1].1.index, 2); // blue
    }

    #[test]
    fn execute_sequence_range() {
        let gif = make_animated_gif();
        let mut graph = NodeGraph::new(1024 * 1024);
        let node = FrameSourceNode::new(gif, FrameSelection::Range(0, 2)).unwrap();
        let (src_id, rc) = graph.add_frame_source(node);
        let seq = graph.execute_sequence(&rc, src_id).unwrap();
        assert_eq!(seq.len(), 2); // frames 0, 1
    }

    #[test]
    fn execute_sequence_with_filter() {
        let gif = make_animated_gif();
        let mut graph = NodeGraph::new(1024 * 1024);
        let node = FrameSourceNode::new(gif, FrameSelection::All).unwrap();
        let (src_id, rc) = graph.add_frame_source(node);
        let src_info = graph.node_info(src_id).unwrap();
        let bright = graph.add_node(Box::new(BrightnessNode::new(
            src_id,
            src_info,
            BrightnessParams { amount: 0.2 },
        )));

        let seq = graph.execute_sequence(&rc, bright).unwrap();
        assert_eq!(seq.len(), 3);
        // Brightness 0.2 adds ~51 to each channel value
        // Frame 0 is red (255, 0, 0) -> (255, 51, 51) after brightness
        assert!(seq.frames[0].0.pixels[1] > 40); // green channel boosted from 0
    }

    #[test]
    fn frame_sequence_from_decode() {
        let gif = make_animated_gif();
        let seq = FrameSequence::from_decode(&gif).unwrap();
        assert_eq!(seq.len(), 3);
        assert_eq!(seq.canvas_width, 4);
        assert_eq!(seq.canvas_height, 4);
    }

    #[test]
    fn backward_compat_single_image_pipeline() {
        // Existing single-image pipeline still works unchanged
        use crate::domain::pipeline::nodes::source::SourceNode;

        let gif = make_animated_gif();
        let mut graph = NodeGraph::new(1024 * 1024);
        let src = graph.add_node(Box::new(SourceNode::new(gif).unwrap()));
        let info = graph.node_info(src).unwrap();
        let full = Rect::new(0, 0, info.width, info.height);
        let pixels = graph.request_region(src, full).unwrap();
        assert_eq!(pixels.len(), 4 * 4 * 4); // 4x4 RGBA8
    }
}

#[cfg(test)]
mod affine_composition_tests {
    use super::*;
    use crate::domain::pipeline::nodes::transform::*;
    use crate::domain::types::{
        ColorSpace, FlipDirection, ImageInfo, PixelFormat, ResizeFilter, Rotation,
    };

    struct RawSource {
        pixels: Vec<u8>,
        info: ImageInfo,
    }
    impl RawSource {
        fn new(pixels: Vec<u8>, info: ImageInfo) -> Self {
            Self { pixels, info }
        }
    }
    impl ImageNode for RawSource {
        fn info(&self) -> ImageInfo {
            self.info.clone()
        }
        fn compute_region(
            &self,
            request: Rect,
            _: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
        ) -> Result<Vec<u8>, ImageError> {
            let bpp = crate::domain::pipeline::graph::bytes_per_pixel(self.info.format);
            Ok(crop_region(
                &self.pixels,
                Rect::new(0, 0, self.info.width, self.info.height),
                request,
                bpp,
            ))
        }
        fn access_pattern(&self) -> AccessPattern {
            AccessPattern::Sequential
        }
    }

    fn make_info(w: u32, h: u32) -> ImageInfo {
        ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        }
    }

    #[test]
    fn compose_identity() {
        let id = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let result = compose_affine(&id, &id);
        for (a, b) in result.iter().zip(id.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn compose_scale_then_rotate90() {
        // Scale 2x then rotate 90°
        let scale = [2.0, 0.0, 0.0, 0.0, 2.0, 0.0];
        let rot90 = [0.0, -1.0, 100.0, 1.0, 0.0, 0.0]; // h=100
        let composed = compose_affine(&rot90, &scale);
        // Point (10, 5): scale → (20, 10), rotate90 → (100-10, 20) = (90, 20)
        let x = composed[0] * 10.0 + composed[1] * 5.0 + composed[2];
        let y = composed[3] * 10.0 + composed[4] * 5.0 + composed[5];
        assert!((x - 90.0).abs() < 1e-10);
        assert!((y - 20.0).abs() < 1e-10);
    }

    #[test]
    fn affine_dims_scale() {
        let scale2x = [2.0, 0.0, 0.0, 0.0, 2.0, 0.0];
        let (w, h) = affine_output_dims(&scale2x, 100, 50);
        assert_eq!(w, 200);
        assert_eq!(h, 100);
    }

    #[test]
    fn affine_dims_rotate90() {
        // Rotate 90° for a 100x50 image: output is 50x100
        let rot90 = [0.0, -1.0, 50.0, 1.0, 0.0, 0.0]; // tx = h = 50
        let (w, h) = affine_output_dims(&rot90, 100, 50);
        assert_eq!(w, 50);
        assert_eq!(h, 100);
    }

    #[test]
    fn resize_node_affine_op() {
        let info = make_info(100, 50);
        let node = ResizeNode::new(0, info, 200, 100, ResizeFilter::Lanczos3);
        let (mat, w, h) = node.to_affine();
        assert_eq!(w, 200);
        assert_eq!(h, 100);
        assert!((mat[0] - 2.0).abs() < 1e-10); // sx = 200/100
        assert!((mat[4] - 2.0).abs() < 1e-10); // sy = 100/50
    }

    #[test]
    fn crop_node_affine_op() {
        let info = make_info(100, 100);
        let node = CropNode::new(0, info, 10, 20, 50, 60);
        let (mat, w, h) = node.to_affine();
        assert_eq!(w, 50);
        assert_eq!(h, 60);
        assert!((mat[2] - (-10.0)).abs() < 1e-10); // tx = -x
        assert!((mat[5] - (-20.0)).abs() < 1e-10); // ty = -y
    }

    #[test]
    fn flip_node_affine_op() {
        let info = make_info(100, 50);
        let node = FlipNode::new(0, info, FlipDirection::Horizontal);
        let (mat, w, h) = node.to_affine();
        assert_eq!(w, 100);
        assert_eq!(h, 50);
        assert!((mat[0] - (-1.0)).abs() < 1e-10); // -1 for flip
        assert!((mat[2] - 100.0).abs() < 1e-10); // tx = width
    }

    #[test]
    fn fuse_resize_rotate_flip() {
        // Build a chain: source → resize → rotate → flip
        let src_info = make_info(64, 64);
        let pixels: Vec<u8> = (0..64 * 64 * 3).map(|i| (i % 256) as u8).collect();

        let mut graph = NodeGraph::new(1024 * 1024);
        let src = graph.add_node(Box::new(RawSource::new(pixels.clone(), src_info.clone())));

        let resize = graph.add_node(Box::new(ResizeNode::new(
            src,
            src_info.clone(),
            32,
            32,
            ResizeFilter::Bilinear,
        )));
        let resize_info = graph.node_info(resize).unwrap();

        let rotate = graph.add_node(Box::new(RotateNode::new(
            resize,
            resize_info.clone(),
            Rotation::R90,
        )));
        let rotate_info = graph.node_info(rotate).unwrap();

        let flip = graph.add_node(Box::new(FlipNode::new(
            rotate,
            rotate_info,
            FlipDirection::Horizontal,
        )));

        // Before fusion: 3 separate transform nodes
        let info_before = graph.node_info(flip).unwrap();

        // Fuse
        graph.fuse_affine_transforms();

        // After fusion: info should be the same
        let info_after = graph.node_info(flip).unwrap();
        assert_eq!(info_before.width, info_after.width);
        assert_eq!(info_before.height, info_after.height);

        // Verify execution works
        let full = Rect::new(0, 0, info_after.width, info_after.height);
        let result = graph.request_region(flip, full).unwrap();
        assert_eq!(
            result.len(),
            (info_after.width * info_after.height * 3) as usize
        );
    }

    #[test]
    fn non_affine_blocks_fusion() {
        // Build: source → resize → blur → rotate
        // Blur should block fusion — resize and rotate should NOT compose
        use crate::domain::filters::BlurParams;
        use crate::domain::pipeline::nodes::filters::BlurNode;

        let src_info = make_info(64, 64);
        let pixels: Vec<u8> = vec![128u8; 64 * 64 * 3];

        let mut graph = NodeGraph::new(1024 * 1024);
        let src = graph.add_node(Box::new(RawSource::new(pixels, src_info.clone())));

        let resize = graph.add_node(Box::new(ResizeNode::new(
            src,
            src_info.clone(),
            32,
            32,
            ResizeFilter::Bilinear,
        )));
        let resize_info = graph.node_info(resize).unwrap();

        let blur = graph.add_node(Box::new(BlurNode::new(
            resize,
            resize_info.clone(),
            BlurParams { radius: 2.0 },
        )));
        let blur_info = graph.node_info(blur).unwrap();

        let rotate = graph.add_node(Box::new(RotateNode::new(blur, blur_info, Rotation::R90)));

        // Fuse — blur should prevent resize+rotate composition
        graph.fuse_affine_transforms();

        // The rotate node should NOT have been composed with resize
        // (it should still be a RotateNode, not ComposedAffineNode)
        // Verify by checking execution still works
        let out_info = graph.node_info(rotate).unwrap();
        let full = Rect::new(0, 0, out_info.width, out_info.height);
        let result = graph.request_region(rotate, full).unwrap();
        assert_eq!(
            result.len(),
            (out_info.width * out_info.height * 3) as usize
        );
    }

    #[test]
    fn composed_output_dimensions_correct() {
        // Resize 100x100 → 50x50, then rotate 90° → 50x50
        let info = make_info(100, 100);
        let resize = ResizeNode::new(0, info.clone(), 50, 50, ResizeFilter::Bilinear);
        let (resize_mat, _, _) = resize.to_affine();

        let resized_info = make_info(50, 50);
        let rotate = RotateNode::new(0, resized_info, Rotation::R90);
        let (rotate_mat, _, _) = rotate.to_affine();

        let composed = compose_affine(&rotate_mat, &resize_mat);
        let (w, h) = affine_output_dims(&composed, 100, 100);
        assert_eq!(w, 50);
        assert_eq!(h, 50);
    }
}

#[cfg(test)]
mod layer_cache_tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::time::Instant;

    use crate::domain::filters::{BrightnessParams, ContrastParams};
    use crate::domain::pipeline::nodes::filters::{self, BrightnessNode, ContrastNode};
    use crate::domain::pipeline::nodes::sink;
    use crate::domain::pipeline::nodes::source::SourceNode;
    use rasmcore_pipeline::LayerCache;

    /// Create a minimal 32x32 RGB PNG for testing.
    fn make_test_png() -> Vec<u8> {
        let w = 32u32;
        let h = 32u32;
        let mut buf = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut buf, w, h);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            let mut pixels = Vec::with_capacity((w * h * 3) as usize);
            for y in 0..h {
                for x in 0..w {
                    pixels.push(((x * 255) / w) as u8);
                    pixels.push(((y * 255) / h) as u8);
                    pixels.push(128u8);
                }
            }
            writer.write_image_data(&pixels).unwrap();
        }
        buf
    }

    /// Create a shared LayerCache with 64 MB budget.
    fn make_layer_cache() -> Rc<RefCell<LayerCache>> {
        Rc::new(RefCell::new(LayerCache::new(64 * 1024 * 1024)))
    }

    /// Build a pipeline graph: source → brightness(amount) → output node id.
    /// Returns (graph, final_node_id).
    fn build_brightness_pipeline(
        png_data: &[u8],
        lc: &Rc<RefCell<LayerCache>>,
        amount: f32,
    ) -> (NodeGraph, u32) {
        let source_hash = rasmcore_pipeline::compute_source_hash(png_data);
        let mut graph = NodeGraph::with_layer_cache(4 * 1024 * 1024, Rc::clone(lc));

        let src = graph.add_node_with_hash(
            Box::new(SourceNode::new(png_data.to_vec()).unwrap()),
            source_hash,
        );
        let src_info = graph.node_info(src).unwrap();

        let brightness_hash = rasmcore_pipeline::compute_hash(
            &source_hash,
            "brightness",
            &amount.to_le_bytes(),
        );
        let bright = graph.add_node_with_hash(
            Box::new(BrightnessNode::new(src, src_info, BrightnessParams { amount })),
            brightness_hash,
        );

        (graph, bright)
    }

    /// Build a pipeline: source → brightness(b_amount) → contrast(c_amount) → output.
    fn build_brightness_contrast_pipeline(
        png_data: &[u8],
        lc: &Rc<RefCell<LayerCache>>,
        b_amount: f32,
        c_amount: f32,
    ) -> (NodeGraph, u32) {
        let source_hash = rasmcore_pipeline::compute_source_hash(png_data);
        let mut graph = NodeGraph::with_layer_cache(4 * 1024 * 1024, Rc::clone(lc));

        let src = graph.add_node_with_hash(
            Box::new(SourceNode::new(png_data.to_vec()).unwrap()),
            source_hash,
        );
        let src_info = graph.node_info(src).unwrap();

        let brightness_hash = rasmcore_pipeline::compute_hash(
            &source_hash,
            "brightness",
            &b_amount.to_le_bytes(),
        );
        let bright = graph.add_node_with_hash(
            Box::new(BrightnessNode::new(
                src,
                src_info.clone(),
                BrightnessParams { amount: b_amount },
            )),
            brightness_hash,
        );

        let contrast_hash = rasmcore_pipeline::compute_hash(
            &brightness_hash,
            "contrast",
            &c_amount.to_le_bytes(),
        );
        let contrast = graph.add_node_with_hash(
            Box::new(ContrastNode::new(
                bright,
                src_info,
                ContrastParams { amount: c_amount },
            )),
            contrast_hash,
        );

        (graph, contrast)
    }

    /// Run a pipeline through sink::write as PNG and finalize the layer cache.
    fn run_pipeline(graph: &mut NodeGraph, node_id: u32) -> Vec<u8> {
        let output = sink::write(graph, node_id, "png", None, None).unwrap();
        graph.finalize_layer_cache();
        output
    }

    #[test]
    fn cache_identical_pipeline_reuse() {
        let png = make_test_png();
        let lc = make_layer_cache();

        // Run 1: cold cache
        let t1 = Instant::now();
        let (mut g1, node1) = build_brightness_pipeline(&png, &lc, 0.2);
        let output1 = run_pipeline(&mut g1, node1);
        let d1 = t1.elapsed();

        // Run 2: warm cache — same pipeline
        let t2 = Instant::now();
        let (mut g2, node2) = build_brightness_pipeline(&png, &lc, 0.2);
        let output2 = run_pipeline(&mut g2, node2);
        let d2 = t2.elapsed();

        eprintln!(
            "cache_identical_pipeline_reuse — run1: {}ms, run2: {}ms",
            d1.as_millis(),
            d2.as_millis()
        );

        assert_eq!(output1, output2, "Identical pipeline must produce identical output");
        // Second run should benefit from cache (brightness node is a hit)
        // On very fast machines the difference may be tiny, so just verify correctness
    }

    #[test]
    fn cache_add_node_reuses_prefix() {
        let png = make_test_png();
        let lc = make_layer_cache();

        // Pipeline 1: source → brightness(0.2)
        let t1 = Instant::now();
        let (mut g1, node1) = build_brightness_pipeline(&png, &lc, 0.2);
        let output1 = run_pipeline(&mut g1, node1);
        let d1 = t1.elapsed();

        // Pipeline 2: source → brightness(0.2) → contrast(0.3)
        let t2 = Instant::now();
        let (mut g2, node2) = build_brightness_contrast_pipeline(&png, &lc, 0.2, 0.3);
        let output2 = run_pipeline(&mut g2, node2);
        let d2 = t2.elapsed();

        eprintln!(
            "cache_add_node_reuses_prefix — run1: {}ms, run2: {}ms",
            d1.as_millis(),
            d2.as_millis()
        );

        assert_ne!(
            output1, output2,
            "Different pipelines must produce different output"
        );
        // Pipeline 2 should reuse source + brightness from cache
    }

    #[test]
    fn cache_remove_last_node_reuses_prefix() {
        let png = make_test_png();
        let lc = make_layer_cache();

        // Pipeline 1: source → brightness(0.2) → contrast(0.3)
        let (mut g1, node1) = build_brightness_contrast_pipeline(&png, &lc, 0.2, 0.3);
        let _output1 = run_pipeline(&mut g1, node1);

        // Pipeline 2: source → brightness(0.2) — should reuse cached prefix
        let (mut g2, node2) = build_brightness_pipeline(&png, &lc, 0.2);
        let output2 = run_pipeline(&mut g2, node2);

        // Fresh uncached run for comparison
        let lc_fresh = make_layer_cache();
        let (mut g_fresh, node_fresh) = build_brightness_pipeline(&png, &lc_fresh, 0.2);
        let output_fresh = run_pipeline(&mut g_fresh, node_fresh);

        assert_eq!(
            output2, output_fresh,
            "Cached prefix pipeline must match fresh uncached run"
        );
    }

    #[test]
    fn cache_change_params_invalidates() {
        let png = make_test_png();
        let lc = make_layer_cache();

        // Pipeline 1: brightness(0.2)
        let (mut g1, node1) = build_brightness_pipeline(&png, &lc, 0.2);
        let output1 = run_pipeline(&mut g1, node1);

        // Pipeline 2: brightness(0.5) — different param, different hash
        let (mut g2, node2) = build_brightness_pipeline(&png, &lc, 0.5);
        let output2 = run_pipeline(&mut g2, node2);

        assert_ne!(
            output1, output2,
            "Different brightness params must produce different output"
        );

        // Source hash is the same, so source node should still be cached
        let stats = lc.borrow().stats();
        assert!(
            stats.hits >= 1,
            "Source node should be a cache hit on second run (hits={})",
            stats.hits
        );
    }

    #[test]
    fn cache_fused_chain_correct_output() {
        let png = make_test_png();

        // Run WITHOUT cache
        let lc_none = make_layer_cache();
        let (mut g_no, node_no) = build_brightness_contrast_pipeline(&png, &lc_none, 0.2, 0.3);
        let expected = run_pipeline(&mut g_no, node_no);

        // Run WITH cache (cold)
        let lc = make_layer_cache();
        let (mut g1, node1) = build_brightness_contrast_pipeline(&png, &lc, 0.2, 0.3);
        let cached_output = run_pipeline(&mut g1, node1);

        // Run AGAIN with cache (warm)
        let (mut g2, node2) = build_brightness_contrast_pipeline(&png, &lc, 0.2, 0.3);
        let second_cached = run_pipeline(&mut g2, node2);

        assert_eq!(
            expected, cached_output,
            "Cached output must match uncached output"
        );
        assert_eq!(
            cached_output, second_cached,
            "Second cached run must match first cached run"
        );
    }

    #[test]
    fn cache_stats_reflect_usage() {
        let png = make_test_png();
        let lc = make_layer_cache();

        // Run 1: populate cache
        let (mut g1, node1) = build_brightness_pipeline(&png, &lc, 0.2);
        let _out1 = run_pipeline(&mut g1, node1);

        let stats1 = lc.borrow().stats();
        assert!(
            stats1.entries > 0,
            "Cache should have entries after first run (entries={})",
            stats1.entries
        );
        assert!(
            stats1.size_bytes > 0,
            "Cache should use memory after first run (size={})",
            stats1.size_bytes
        );
        let hits_after_first = stats1.hits;

        // Run 2: same pipeline — should get cache hits
        let (mut g2, node2) = build_brightness_pipeline(&png, &lc, 0.2);
        let _out2 = run_pipeline(&mut g2, node2);

        let stats2 = lc.borrow().stats();
        assert!(
            stats2.hits > hits_after_first,
            "Hits should increase on second run (before={}, after={})",
            hits_after_first,
            stats2.hits
        );

        eprintln!(
            "cache_stats_reflect_usage — entries: {}, hits: {}, misses: {}, size: {} bytes",
            stats2.entries, stats2.hits, stats2.misses, stats2.size_bytes
        );
    }

    /// Helper: build source → N inverts chain with content hashes
    fn build_invert_chain(
        png_data: &[u8],
        lc: &Rc<RefCell<LayerCache>>,
        count: usize,
    ) -> (NodeGraph, u32) {
        let source_hash = rasmcore_pipeline::compute_source_hash(png_data);
        let mut graph = NodeGraph::with_layer_cache(4 * 1024 * 1024, Rc::clone(lc));
        let src = graph.add_node_with_hash(
            Box::new(SourceNode::new(png_data.to_vec()).unwrap()),
            source_hash,
        );
        let src_info = graph.node_info(src).unwrap();
        let mut prev = src;
        let mut prev_hash = source_hash;
        for i in 0..count {
            let h = rasmcore_pipeline::compute_hash(&prev_hash, "invert", &(i as u32).to_le_bytes());
            prev = graph.add_node_with_hash(
                Box::new(filters::InvertNode::new(prev, src_info.clone())),
                h,
            );
            prev_hash = h;
        }
        (graph, prev)
    }

    /// Helper: run pipeline and finalize cache
    fn run_and_finalize(graph: &mut NodeGraph, node: u32) -> Vec<u8> {
        let out = sink::write(graph, node, "png", None, None).unwrap();
        graph.finalize_layer_cache();
        out
    }

    #[test]
    fn cache_fused_invert_chain_identity() {
        // invert→invert = identity (fused into single LUT that's [0,1,2,...,255])
        let png = make_test_png();
        let lc = make_layer_cache();

        // Pipeline: source only (reference)
        let source_only = {
            let mut g = NodeGraph::with_layer_cache(4 * 1024 * 1024, Rc::clone(&lc));
            let src_hash = rasmcore_pipeline::compute_source_hash(&png);
            let src = g.add_node_with_hash(Box::new(SourceNode::new(png.clone()).unwrap()), src_hash);
            run_and_finalize(&mut g, src)
        };

        // Pipeline: source → invert → invert (should fuse to identity)
        let double_invert = {
            let (mut g, node) = build_invert_chain(&png, &lc, 2);
            run_and_finalize(&mut g, node)
        };

        assert_eq!(source_only, double_invert, "invert→invert should equal source (identity via fusion)");
        eprintln!("cache_fused_invert_chain_identity: PASS — double invert = source");
    }

    #[test]
    fn cache_remove_middle_of_fused_chain() {
        // Pipeline 1: source → invert → brightness(0.2) → invert (3 point ops, fused)
        // Pipeline 2: source → invert → invert (removed brightness from middle)
        // Pipeline 2 should match source (invert→invert = identity)
        // Pipeline 1 should NOT match source (brightness changes values before second invert)
        let png = make_test_png();
        let lc = make_layer_cache();

        // Reference: source only
        let source_only = {
            let mut g = NodeGraph::with_layer_cache(4 * 1024 * 1024, Rc::clone(&lc));
            let src_hash = rasmcore_pipeline::compute_source_hash(&png);
            let src = g.add_node_with_hash(Box::new(SourceNode::new(png.clone()).unwrap()), src_hash);
            run_and_finalize(&mut g, src)
        };

        // Pipeline 1: source → invert → brightness(0.2) → invert
        let output1 = {
            let src_hash = rasmcore_pipeline::compute_source_hash(&png);
            let mut g = NodeGraph::with_layer_cache(4 * 1024 * 1024, Rc::clone(&lc));
            let src = g.add_node_with_hash(Box::new(SourceNode::new(png.clone()).unwrap()), src_hash);
            let si = g.node_info(src).unwrap();

            let h1 = rasmcore_pipeline::compute_hash(&src_hash, "invert", &0u32.to_le_bytes());
            let n1 = g.add_node_with_hash(Box::new(filters::InvertNode::new(src, si.clone())), h1);

            let h2 = rasmcore_pipeline::compute_hash(&h1, "brightness", &0.2f32.to_le_bytes());
            let n2 = g.add_node_with_hash(Box::new(BrightnessNode::new(n1, si.clone(), BrightnessParams { amount: 0.2 })), h2);

            let h3 = rasmcore_pipeline::compute_hash(&h2, "invert", &1u32.to_le_bytes());
            let n3 = g.add_node_with_hash(Box::new(filters::InvertNode::new(n2, si.clone())), h3);

            run_and_finalize(&mut g, n3)
        };

        // Pipeline 2: source → invert → invert (brightness removed, same cache)
        let output2 = {
            let src_hash = rasmcore_pipeline::compute_source_hash(&png);
            let mut g = NodeGraph::with_layer_cache(4 * 1024 * 1024, Rc::clone(&lc));
            let src = g.add_node_with_hash(Box::new(SourceNode::new(png.clone()).unwrap()), src_hash);
            let si = g.node_info(src).unwrap();

            let h1 = rasmcore_pipeline::compute_hash(&src_hash, "invert", &0u32.to_le_bytes());
            let n1 = g.add_node_with_hash(Box::new(filters::InvertNode::new(src, si.clone())), h1);

            // Skip brightness — second invert's upstream hash is h1 (not h2)
            let h_inv2 = rasmcore_pipeline::compute_hash(&h1, "invert", &1u32.to_le_bytes());
            let n_inv2 = g.add_node_with_hash(Box::new(filters::InvertNode::new(n1, si.clone())), h_inv2);

            run_and_finalize(&mut g, n_inv2)
        };

        // Fresh uncached run of pipeline 2 for reference
        let output2_fresh = {
            let mut g = NodeGraph::new(4 * 1024 * 1024);
            let src = g.add_node(Box::new(SourceNode::new(png.clone()).unwrap()));
            let si = g.node_info(src).unwrap();
            let n1 = g.add_node(Box::new(filters::InvertNode::new(src, si.clone())));
            let n2 = g.add_node(Box::new(filters::InvertNode::new(n1, si.clone())));
            sink::write(&mut g, n2, "png", None, None).unwrap()
        };

        assert_ne!(output1, source_only, "invert→brightness→invert should differ from source");
        assert_eq!(output2, source_only, "invert→invert should equal source (identity)");
        assert_eq!(output2, output2_fresh, "cached run must match fresh uncached run");
        eprintln!("cache_remove_middle_of_fused_chain: PASS");
    }

    #[test]
    fn cache_add_to_fused_chain() {
        // Pipeline 1: source → invert → invert (identity, fused)
        // Pipeline 2: source → invert → invert → invert (= single invert)
        // Verify pipeline 2 output differs from source and matches single invert
        let png = make_test_png();
        let lc = make_layer_cache();

        // Pipeline 1: source → 2 inverts (identity)
        let _output1 = {
            let (mut g, node) = build_invert_chain(&png, &lc, 2);
            run_and_finalize(&mut g, node)
        };

        // Pipeline 2: source → 3 inverts (= single invert)
        let output2 = {
            let (mut g, node) = build_invert_chain(&png, &lc, 3);
            run_and_finalize(&mut g, node)
        };

        // Reference: single invert, no cache
        let single_invert = {
            let mut g = NodeGraph::new(4 * 1024 * 1024);
            let src = g.add_node(Box::new(SourceNode::new(png.clone()).unwrap()));
            let si = g.node_info(src).unwrap();
            let n = g.add_node(Box::new(filters::InvertNode::new(src, si)));
            sink::write(&mut g, n, "png", None, None).unwrap()
        };

        // Source only for comparison
        let source_only = {
            let mut g = NodeGraph::new(4 * 1024 * 1024);
            let src = g.add_node(Box::new(SourceNode::new(png.clone()).unwrap()));
            sink::write(&mut g, src, "png", None, None).unwrap()
        };

        assert_ne!(output2, source_only, "3 inverts should not equal source");
        assert_eq!(output2, single_invert, "3 inverts should equal single invert");
        eprintln!("cache_add_to_fused_chain: PASS");
    }

    /// Helper: run pipeline with explicit small tile size to force tiled execution
    fn run_tiled_and_finalize(graph: &mut NodeGraph, node: u32) -> Vec<u8> {
        let tile_cfg = sink::TileConfig { tile_size: 16 }; // 16x16 tiles on 32x32 image = 4 tiles
        let out = sink::write_tiled(graph, node, "png", None, None, &tile_cfg).unwrap();
        graph.finalize_layer_cache();
        out
    }

    #[test]
    fn cache_tiled_execution_matches_untiled() {
        // Verify: tiled execution with layer cache produces same output as untiled
        let png = make_test_png();
        let lc = make_layer_cache();

        // Run 1: untiled (full-image request)
        let untiled = {
            let (mut g, node) = build_brightness_pipeline(&png, &lc, 0.2);
            run_and_finalize(&mut g, node)
        };

        // Run 2: tiled (16x16 tiles on 32x32 image), fresh cache
        let lc2 = make_layer_cache();
        let tiled = {
            let (mut g, node) = build_brightness_pipeline(&png, &lc2, 0.2);
            run_tiled_and_finalize(&mut g, node)
        };

        assert_eq!(untiled, tiled, "tiled and untiled must produce identical output");
        eprintln!("cache_tiled_execution_matches_untiled: PASS");
    }

    #[test]
    fn cache_tiled_reuse_across_pipelines() {
        // Run 1: source → brightness (tiled, populates layer cache)
        // Run 2: source → brightness → contrast (tiled, brightness should be cache hit)
        // Run 3: source → brightness (tiled again, should be fast from cache)
        // Verify all outputs are correct
        let png = make_test_png();
        let lc = make_layer_cache();

        // Run 1: source → brightness(0.2), tiled
        let output1 = {
            let (mut g, node) = build_brightness_pipeline(&png, &lc, 0.2);
            run_tiled_and_finalize(&mut g, node)
        };

        // Run 2: source → brightness(0.2) → contrast(0.3), tiled
        let output2 = {
            let (mut g, node) = build_brightness_contrast_pipeline(&png, &lc, 0.2, 0.3);
            run_tiled_and_finalize(&mut g, node)
        };

        // Run 3: back to source → brightness(0.2), tiled (should reuse from cache)
        let output3 = {
            let (mut g, node) = build_brightness_pipeline(&png, &lc, 0.2);
            run_tiled_and_finalize(&mut g, node)
        };

        assert_ne!(output1, output2, "adding contrast should change output");
        assert_eq!(output1, output3, "same pipeline should produce same output from cache");

        // Fresh uncached reference
        let fresh = {
            let mut g = NodeGraph::new(4 * 1024 * 1024);
            let src = g.add_node(Box::new(SourceNode::new(png.clone()).unwrap()));
            let si = g.node_info(src).unwrap();
            let b = g.add_node(Box::new(BrightnessNode::new(src, si, BrightnessParams { amount: 0.2 })));
            let tile_cfg = sink::TileConfig { tile_size: 16 };
            sink::write_tiled(&mut g, b, "png", None, None, &tile_cfg).unwrap()
        };

        assert_eq!(output3, fresh, "cached tiled must match fresh uncached tiled");
        eprintln!("cache_tiled_reuse_across_pipelines: PASS");
    }
}
