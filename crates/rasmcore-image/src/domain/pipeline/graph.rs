//! Node graph — holds all nodes and dispatches region requests.

use super::cache::SpatialCache;
use super::rect::{Overlap, Rect};
use crate::domain::error::ImageError;
use crate::domain::types::ImageInfo;

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

/// Pipeline-owned node graph.
pub struct NodeGraph {
    nodes: Vec<Box<dyn ImageNode>>,
    cache: SpatialCache,
}

impl NodeGraph {
    /// Create a new graph with the given memory budget for the spatial cache.
    pub fn new(cache_budget: usize) -> Self {
        Self {
            nodes: Vec::new(),
            cache: SpatialCache::new(cache_budget),
        }
    }

    /// Add a node to the graph. Returns its node-id.
    pub fn add_node(&mut self, node: Box<dyn ImageNode>) -> u32 {
        let id = self.nodes.len() as u32;
        self.nodes.push(node);
        id
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
        let info = self.node_info(node_id)?;
        let bpp = bytes_per_pixel(info.format);

        // Check cache first
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

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

/// Calculate bytes per pixel for a given format.
pub fn bytes_per_pixel(format: crate::domain::types::PixelFormat) -> u32 {
    use crate::domain::types::PixelFormat;
    match format {
        PixelFormat::Rgb8 | PixelFormat::Bgr8 => 3,
        PixelFormat::Rgba8 | PixelFormat::Bgra8 => 4,
        PixelFormat::Gray8 => 1,
        PixelFormat::Gray16 => 2,
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
