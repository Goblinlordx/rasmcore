//! Graph engine — demand-driven tile execution with GPU-primary dispatch.
//!
//! The graph manages node topology and executes tile requests by pulling
//! data through the graph from sinks to sources. GPU dispatch is checked
//! first for every node; CPU `compute()` is the fallback.
//!
//! All data is `Vec<f32>` — 4 channels (RGBA) per pixel. No format dispatch.

use std::rc::Rc;

use crate::cache::SpatialCache;
use crate::demand::DemandStrategy;
use crate::gpu::GpuExecutor;
use crate::node::{Node, NodeInfo, PipelineError, Upstream};
use crate::rect::Rect;

/// The V2 pipeline graph.
///
/// Manages node topology, caching, GPU dispatch, and demand-driven execution.
/// Knows nothing about specific operations — just dispatches through the
/// `Node` trait.
pub struct Graph {
    pub(crate) nodes: Vec<Box<dyn Node>>,
    cache: SpatialCache,
    gpu_executor: Option<Rc<dyn GpuExecutor>>,
    demand_strategy: DemandStrategy,
    pub(crate) aces_strict: bool,
}

impl Graph {
    /// Create a new graph with the given cache budget (bytes).
    pub fn new(cache_budget: usize) -> Self {
        Self {
            nodes: Vec::new(),
            cache: SpatialCache::new(cache_budget),
            gpu_executor: None,
            demand_strategy: DemandStrategy::default(),
            aces_strict: false,
        }
    }

    /// Add a node to the graph. Returns the node ID.
    pub fn add_node(&mut self, node: Box<dyn Node>) -> u32 {
        let id = self.nodes.len() as u32;
        self.nodes.push(node);
        id
    }

    /// Set the GPU executor for GPU-primary dispatch.
    pub fn set_gpu_executor(&mut self, executor: Rc<dyn GpuExecutor>) {
        self.gpu_executor = Some(executor);
    }

    /// Set the demand strategy (tile sizing).
    pub fn set_demand_strategy(&mut self, strategy: DemandStrategy) {
        self.demand_strategy = strategy;
    }

    /// Get node info for a given node ID.
    pub fn node_info(&self, node_id: u32) -> Result<NodeInfo, PipelineError> {
        self.nodes
            .get(node_id as usize)
            .map(|n| n.info())
            .ok_or(PipelineError::NodeNotFound(node_id))
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> u32 {
        self.nodes.len() as u32
    }

    /// Request f32 pixel data for a region from a node.
    ///
    /// This is the core execution entry point. It:
    /// 1. Checks the spatial cache
    /// 2. Tries GPU dispatch (primary path)
    /// 3. Falls back to CPU compute
    ///
    /// Returns `Vec<f32>` with `request.width * request.height * 4` elements.
    /// Collect a chain of consecutive GPU-capable nodes ending at `node_id`.
    ///
    /// Walks upstream from `node_id`, collecting GPU shaders as long as each
    /// node is GPU-capable with a single upstream. Stops at the first non-GPU
    /// node or a node with multiple/no upstreams. Returns (source_node_id, shaders).
    fn collect_gpu_chain(
        &self,
        node_id: u32,
        width: u32,
        height: u32,
    ) -> Option<(u32, Vec<crate::node::GpuShader>)> {
        let mut chain = Vec::new();
        let mut current = node_id;

        loop {
            let node = self.nodes.get(current as usize)?;
            let shaders = node.gpu_shaders(width, height);
            let upstream_ids = node.upstream_ids();

            match (shaders, upstream_ids.len()) {
                (Some(s), 1) if !s.is_empty() => {
                    // Prepend shaders (we're walking backwards)
                    chain.splice(0..0, s);
                    current = upstream_ids[0];
                }
                _ => {
                    // Not GPU-capable or has multiple/no upstreams — stop here
                    break;
                }
            }
        }

        if chain.is_empty() {
            None
        } else {
            Some((current, chain))
        }
    }

    pub fn request_region(
        &mut self,
        node_id: u32,
        request: Rect,
    ) -> Result<Vec<f32>, PipelineError> {
        // 1. Cache check
        if let Some(cached) = self.cache.query(node_id, request) {
            return Ok(cached);
        }

        let node = self
            .nodes
            .get(node_id as usize)
            .ok_or(PipelineError::NodeNotFound(node_id))?;

        let info = node.info();

        // 2. GPU-primary dispatch — batch consecutive GPU nodes
        if let Some(executor) = &self.gpu_executor {
            if let Some((source_id, gpu_chain)) = self.collect_gpu_chain(node_id, info.width, info.height) {
                // Get input from the source node (first non-GPU node in the chain)
                let executor = executor.clone();
                let input = self.request_region(source_id, request)?;
                match executor.execute(
                    &gpu_chain,
                    &input,
                    request.width,
                    request.height,
                ) {
                    Ok(gpu_pixels) => {
                        self.cache.store(
                            node_id,
                            request,
                            gpu_pixels.clone(),
                            crate::hash::ZERO_HASH,
                        );
                        return Ok(gpu_pixels);
                    }
                    Err(_) => {
                        // GPU failed — fall through to CPU
                    }
                }
            }
        }

        // 3. CPU fallback
        // Use raw pointer to split the borrow (node.compute needs &mut self for upstream)
        let self_ptr: *mut Graph = self;
        let node_ptr: *const dyn Node = &**self
            .nodes
            .get(node_id as usize)
            .ok_or(PipelineError::NodeNotFound(node_id))?;

        let mut upstream_adapter = GraphUpstream { graph: self_ptr };
        let pixels = unsafe { &*node_ptr }.compute(request, &mut upstream_adapter)?;

        // Validate output size
        let expected = request.width as usize * request.height as usize * 4;
        if pixels.len() != expected {
            return Err(PipelineError::BufferMismatch {
                expected,
                actual: pixels.len(),
            });
        }

        // Cache
        self.cache
            .store(node_id, request, pixels.clone(), crate::hash::ZERO_HASH);
        Ok(pixels)
    }

    /// Request the full output of a node (convenience).
    pub fn request_full(&mut self, node_id: u32) -> Result<Vec<f32>, PipelineError> {
        // Run fusion optimizer before execution (idempotent — safe to call multiple times)
        crate::fusion::optimize(self);
        let info = self.node_info(node_id)?;
        self.request_region(node_id, Rect::new(0, 0, info.width, info.height))
    }

    /// Request full output as tiled execution (bounds peak memory).
    pub fn request_tiled(
        &mut self,
        node_id: u32,
        tile_size: u32,
    ) -> Result<Vec<f32>, PipelineError> {
        let info = self.node_info(node_id)?;
        let w = info.width;
        let h = info.height;

        if tile_size == 0 || (w <= tile_size && h <= tile_size) {
            return self.request_full(node_id);
        }

        let mut out = vec![0.0f32; w as usize * h as usize * 4];
        let stride = w as usize * 4;

        let mut y = 0u32;
        while y < h {
            let th = tile_size.min(h - y);
            let mut x = 0u32;
            while x < w {
                let tw = tile_size.min(w - x);
                let tile = self.request_region(node_id, Rect::new(x, y, tw, th))?;
                let tile_stride = tw as usize * 4;
                for row in 0..th as usize {
                    let dst = (y as usize + row) * stride + x as usize * 4;
                    let src = row * tile_stride;
                    out[dst..dst + tile_stride]
                        .copy_from_slice(&tile[src..src + tile_stride]);
                }
                x += tile_size;
            }
            y += tile_size;
        }

        Ok(out)
    }

    /// Get the demand strategy.
    pub fn demand_strategy(&self) -> &DemandStrategy {
        &self.demand_strategy
    }

    /// Clear the spatial cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    // ─── Optimizer helpers ──────────────────────────────────────────────

    /// Get a reference to a node by ID (for optimizer inspection).
    pub fn get_node(&self, node_id: u32) -> &dyn Node {
        &*self.nodes[node_id as usize]
    }

    /// Replace a node in the graph (for optimizer fusion).
    ///
    /// The cache for this node is invalidated.
    pub fn replace_node(&mut self, node_id: u32, node: Box<dyn Node>) {
        self.nodes[node_id as usize] = node;
        self.cache.invalidate(node_id);
    }

    /// Get the analytic expression for a node (if it has one).
    ///
    /// Returns `Err` if the node doesn't support analytic expressions.
    pub fn get_node_expression(
        &self,
        node_id: u32,
    ) -> Result<crate::ops::PointOpExpr, PipelineError> {
        let node = self
            .nodes
            .get(node_id as usize)
            .ok_or(PipelineError::NodeNotFound(node_id))?;

        // Try to downcast to AnalyticOp — this requires the node to expose
        // its expression. For now, use the node's capabilities check.
        // The actual expression extraction will be via a trait method added
        // to Node in a follow-up.
        // Placeholder: return Input (identity) — real implementation needs
        // Node trait extension.
        if node.capabilities().analytic {
            // TODO: Add fn expression(&self) -> Option<PointOpExpr> to Node trait
            Ok(crate::ops::PointOpExpr::Input)
        } else {
            Err(PipelineError::ComputeError(format!(
                "node {} does not support analytic expressions",
                node_id
            )))
        }
    }
}

/// Adapter that bridges Graph::request_region to the Upstream trait.
struct GraphUpstream {
    graph: *mut Graph,
}

impl Upstream for GraphUpstream {
    fn request(&mut self, upstream_id: u32, rect: Rect) -> Result<Vec<f32>, PipelineError> {
        // SAFETY: The graph is only mutated through request_region, which is
        // not reentrant on the same node. Different nodes can safely recurse.
        unsafe { &mut *self.graph }.request_region(upstream_id, rect)
    }

    fn info(&self, upstream_id: u32) -> Result<NodeInfo, PipelineError> {
        unsafe { &*self.graph }.node_info(upstream_id)
    }
}

/// Crop f32 pixel data from a source rect to a destination rect.
fn crop_f32(data: &[f32], src_rect: Rect, dst_rect: Rect) -> Vec<f32> {
    let sw = src_rect.width as usize;
    let dw = dst_rect.width as usize;
    let dh = dst_rect.height as usize;
    let dx = (dst_rect.x - src_rect.x) as usize;
    let dy = (dst_rect.y - src_rect.y) as usize;

    let mut out = Vec::with_capacity(dw * dh * 4);
    for row in 0..dh {
        let src_off = ((dy + row) * sw + dx) * 4;
        out.extend_from_slice(&data[src_off..src_off + dw * 4]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color_space::ColorSpace;
    use crate::node::NodeInfo;

    /// A trivial source node that produces a solid color.
    struct SolidColorNode {
        width: u32,
        height: u32,
        color: [f32; 4],
    }

    impl Node for SolidColorNode {
        fn info(&self) -> NodeInfo {
            NodeInfo {
                width: self.width,
                height: self.height,
                color_space: ColorSpace::Linear,
            }
        }

        fn compute(
            &self,
            request: Rect,
            _upstream: &mut dyn Upstream,
        ) -> Result<Vec<f32>, PipelineError> {
            let n = request.width as usize * request.height as usize;
            let mut pixels = Vec::with_capacity(n * 4);
            for _ in 0..n {
                pixels.extend_from_slice(&self.color);
            }
            Ok(pixels)
        }

        fn upstream_ids(&self) -> Vec<u32> {
            vec![] // source node, no upstream
        }
    }

    /// A trivial filter that scales brightness by a factor.
    struct ScaleNode {
        upstream: u32,
        factor: f32,
        info: NodeInfo,
    }

    impl Node for ScaleNode {
        fn info(&self) -> NodeInfo {
            self.info.clone()
        }

        fn compute(
            &self,
            request: Rect,
            upstream: &mut dyn Upstream,
        ) -> Result<Vec<f32>, PipelineError> {
            let input = upstream.request(self.upstream, request)?;
            Ok(input.iter().map(|&v| v * self.factor).collect())
        }

        fn upstream_ids(&self) -> Vec<u32> {
            vec![self.upstream]
        }
    }

    #[test]
    fn graph_single_source() {
        let mut g = Graph::new(0);
        let src = g.add_node(Box::new(SolidColorNode {
            width: 4,
            height: 4,
            color: [0.5, 0.3, 0.1, 1.0],
        }));

        let pixels = g.request_full(src).unwrap();
        assert_eq!(pixels.len(), 4 * 4 * 4); // 4x4 * RGBA
        assert!((pixels[0] - 0.5).abs() < 1e-6); // R
        assert!((pixels[1] - 0.3).abs() < 1e-6); // G
        assert!((pixels[2] - 0.1).abs() < 1e-6); // B
        assert!((pixels[3] - 1.0).abs() < 1e-6); // A
    }

    #[test]
    fn graph_source_and_filter() {
        let mut g = Graph::new(0);
        let src = g.add_node(Box::new(SolidColorNode {
            width: 4,
            height: 4,
            color: [0.5, 0.3, 0.1, 1.0],
        }));
        let info = g.node_info(src).unwrap();
        let scale = g.add_node(Box::new(ScaleNode {
            upstream: src,
            factor: 2.0,
            info,
        }));

        let pixels = g.request_full(scale).unwrap();
        assert!((pixels[0] - 1.0).abs() < 1e-6); // 0.5 * 2.0 = 1.0
        assert!((pixels[1] - 0.6).abs() < 1e-6); // 0.3 * 2.0 = 0.6
    }

    #[test]
    fn graph_tiled_matches_full() {
        let mut g = Graph::new(0);
        let src = g.add_node(Box::new(SolidColorNode {
            width: 32,
            height: 32,
            color: [0.25, 0.5, 0.75, 1.0],
        }));

        let full = g.request_full(src).unwrap();
        g.clear_cache();
        let tiled = g.request_tiled(src, 16).unwrap();

        assert_eq!(full.len(), tiled.len());
        assert_eq!(full, tiled);
    }

    #[test]
    fn graph_hdr_values_preserved() {
        let mut g = Graph::new(0);
        let src = g.add_node(Box::new(SolidColorNode {
            width: 2,
            height: 2,
            color: [5.0, 10.0, -0.5, 1.0], // HDR: values > 1.0 and < 0.0
        }));

        let pixels = g.request_full(src).unwrap();
        assert!((pixels[0] - 5.0).abs() < 1e-6);
        assert!((pixels[1] - 10.0).abs() < 1e-6);
        assert!((pixels[2] - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn graph_node_count() {
        let mut g = Graph::new(0);
        assert_eq!(g.node_count(), 0);
        g.add_node(Box::new(SolidColorNode {
            width: 1,
            height: 1,
            color: [0.0; 4],
        }));
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn graph_buffer_validation() {
        // A broken node that returns wrong-sized data
        struct BrokenNode;
        impl Node for BrokenNode {
            fn info(&self) -> NodeInfo {
                NodeInfo {
                    width: 4,
                    height: 4,
                    color_space: ColorSpace::Linear,
                }
            }
            fn compute(
                &self,
                _request: Rect,
                _upstream: &mut dyn Upstream,
            ) -> Result<Vec<f32>, PipelineError> {
                Ok(vec![0.0; 10]) // Wrong size
            }
            fn upstream_ids(&self) -> Vec<u32> {
                vec![]
            }
        }

        let mut g = Graph::new(0);
        let n = g.add_node(Box::new(BrokenNode));
        let result = g.request_full(n);
        assert!(matches!(result, Err(PipelineError::BufferMismatch { .. })));
    }
}
