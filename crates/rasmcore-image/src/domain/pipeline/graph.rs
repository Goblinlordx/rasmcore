//! Node graph — holds all nodes and dispatches region requests.

use std::rc::Rc;

use crate::domain::error::ImageError;
use crate::domain::pipeline::nodes::frame_source::FrameSourceNode;
use crate::domain::types::{DecodedImage, FrameSequence, ImageInfo};
use rasmcore_pipeline::{Overlap, Rect, SpatialCache};

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

    /// Overlap this node needs from upstream for each output region.
    fn overlap(&self) -> Overlap;

    /// Access pattern hint.
    fn access_pattern(&self) -> AccessPattern;
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
    fn overlap(&self) -> Overlap {
        self.0.overlap()
    }
    fn access_pattern(&self) -> AccessPattern {
        self.0.access_pattern()
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
    cache_hit_pixels: std::collections::HashMap<u32, Vec<u8>>,
    // Per-node metadata — set at creation, immutable during tile execution
    node_metadata: Vec<rasmcore_pipeline::Metadata>,
}

impl NodeGraph {
    /// Create a new graph with the given memory budget for the spatial cache.
    pub fn new(cache_budget: usize) -> Self {
        Self {
            nodes: Vec::new(),
            cache: SpatialCache::new(cache_budget),
            cache_budget,
            layer_cache: None,
            node_hashes: Vec::new(),
            touched_hashes: std::collections::HashSet::new(),
            cache_hit_nodes: std::collections::HashSet::new(),
            cache_hit_pixels: std::collections::HashMap::new(),
            node_metadata: Vec::new(),
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
        self.nodes.push(node);
        self.node_hashes.push(rasmcore_pipeline::ZERO_HASH);
        self.node_metadata.push(rasmcore_pipeline::Metadata::new());
        id
    }

    /// Add a node with metadata (flows from upstream, possibly modified).
    pub fn add_node_with_metadata(
        &mut self,
        node: Box<dyn ImageNode>,
        metadata: rasmcore_pipeline::Metadata,
    ) -> u32 {
        let id = self.nodes.len() as u32;
        self.nodes.push(node);
        self.node_hashes.push(rasmcore_pipeline::ZERO_HASH);
        self.node_metadata.push(metadata);
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
        self.nodes.push(node);
        self.node_hashes.push(hash);
        self.node_metadata.push(rasmcore_pipeline::Metadata::new());

        // Check layer cache for this hash
        if let Some(lc) = &self.layer_cache {
            let mut lc = lc.borrow_mut();
            if let Some((pixels, _w, _h, _bpp)) = lc.get(&hash) {
                self.cache_hit_pixels.insert(id, pixels.to_vec());
                self.cache_hit_nodes.insert(id);
            }
        }

        id
    }

    /// Add a node with both a content hash and metadata.
    pub fn add_node_with_hash_and_metadata(
        &mut self,
        node: Box<dyn ImageNode>,
        hash: rasmcore_pipeline::ContentHash,
        metadata: rasmcore_pipeline::Metadata,
    ) -> u32 {
        let id = self.nodes.len() as u32;
        self.nodes.push(node);
        self.node_hashes.push(hash);
        self.node_metadata.push(metadata);

        if let Some(lc) = &self.layer_cache {
            let mut lc = lc.borrow_mut();
            if let Some((pixels, _w, _h, _bpp)) = lc.get(&hash) {
                self.cache_hit_pixels.insert(id, pixels.to_vec());
                self.cache_hit_nodes.insert(id);
            }
        }

        id
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

    /// Request a region from a node. Uses the spatial cache for reuse.
    pub fn request_region(&mut self, node_id: u32, request: Rect) -> Result<Vec<u8>, ImageError> {
        // Track this node's hash as "touched" for layer cache reference tracking
        if let Some(hash) = self.node_hashes.get(node_id as usize) {
            self.touched_hashes.insert(*hash);
        }

        // Check layer cache hit first (pre-populated during add_node_with_hash)
        if self.cache_hit_nodes.contains(&node_id)
            && let Some(pixels) = self.cache_hit_pixels.get(&node_id)
        {
            return Ok(pixels.clone());
        }

        let info = self.node_info(node_id)?;
        let bpp = bytes_per_pixel(info.format);

        // Check spatial cache
        let query = self.cache.query(node_id, request);
        if query.fully_cached {
            let handle = query.hits[0];
            let cached_rect = self.cache.rect(handle);
            if cached_rect == request {
                return Ok(self.cache.read(handle).to_vec());
            }
            return Ok(self.cache.extract_subregion(handle, request, bpp));
        }

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

        // Store in cache
        let handle = self.cache.store(node_id, request, pixels, bpp);
        Ok(self.cache.read(handle).to_vec())
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

            // If this was NOT a cache hit, push the computed output
            if !self.cache_hit_nodes.contains(&(node_id as u32)) {
                let info = match self.nodes.get(node_id) {
                    Some(n) => n.info(),
                    None => continue,
                };
                let bpp = bytes_per_pixel(info.format);
                let full_rect = Rect::new(0, 0, info.width, info.height);

                // Try to get from spatial cache
                let query = self.cache.query(node_id as u32, full_rect);
                if query.fully_cached {
                    let handle = query.hits[0];
                    let pixels = self.cache.read(handle).to_vec();
                    lc.store(*hash, pixels, info.width, info.height, bpp);
                }
            }
        }

        // Mark all touched hashes as referenced
        for hash in &self.touched_hashes {
            lc.mark_referenced(hash);
        }

        // Clean entries not referenced by this pipeline run
        lc.cleanup_unreferenced();
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
        self.nodes.push(Box::new(FrameSourceRcWrapper(rc.clone())));
        self.node_hashes.push(rasmcore_pipeline::ZERO_HASH);
        self.node_metadata.push(rasmcore_pipeline::Metadata::new());
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

        fn overlap(&self) -> Overlap {
            Overlap::zero()
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
            fn overlap(&self) -> Overlap {
                Overlap::zero()
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
}

#[cfg(test)]
mod tiled_parity_tests {
    use super::*;
    use crate::domain::pipeline::nodes::filters::{
        BlurNode, BrightnessNode, ContrastNode, SharpenNode,
    };
    use crate::domain::filters::{BlurParams, BrightnessParams, ContrastParams, SharpenParams};
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
        fn overlap(&self) -> rasmcore_pipeline::Overlap {
            rasmcore_pipeline::Overlap::zero()
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
}

#[cfg(test)]
mod frame_sequence_tests {
    use super::*;
    use crate::domain::pipeline::nodes::filters::BrightnessNode;
    use crate::domain::filters::BrightnessParams;
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
        let bright = graph.add_node(Box::new(BrightnessNode::new(src_id, src_info, BrightnessParams { amount: 0.2 })));

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
