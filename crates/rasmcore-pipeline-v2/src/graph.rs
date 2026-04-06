//! Graph engine — demand-driven tile execution with GPU-primary dispatch.
//!
//! The graph manages node topology and executes tile requests by pulling
//! data through the graph from sinks to sources. GPU dispatch is checked
//! first for every node; CPU `compute()` is the fallback.
//!
//! All data is `Vec<f32>` — 4 channels (RGBA) per pixel. No format dispatch.

use std::cell::RefCell;
use std::rc::Rc;

use crate::cache::SpatialCache;
use crate::demand::DemandStrategy;
use crate::gpu::GpuExecutor;
use crate::hash::{ContentHash, ZERO_HASH};
use crate::layer_cache::LayerCache;
use crate::node::{Node, NodeInfo, PipelineError, Upstream};
use crate::rect::Rect;
use crate::trace::{PipelineTrace, TraceEventKind, TraceTimer};

/// Reusable pixel buffer pool — avoids Vec allocation churn in the CPU render path.
///
/// Buffers are pre-allocated at image dimensions and recycled across pipeline
/// lifetimes. Associated with a Source so all pipelines processing the same
/// image share the same buffers.
pub struct BufferPool {
    buffers: Vec<Vec<f32>>,
    /// Maximum idle buffers to keep. Excess are dropped on release.
    max_idle: usize,
}

impl BufferPool {
    pub fn new() -> Self {
        Self { buffers: Vec::new(), max_idle: 4 }
    }

    /// Create a pool with a custom max idle count.
    pub fn with_max_idle(max_idle: usize) -> Self {
        Self { buffers: Vec::new(), max_idle }
    }

    /// Take a buffer from the pool, or allocate a new one at the given size.
    /// The returned buffer has exactly `size` elements (resized if needed).
    pub fn acquire(&mut self, size: usize) -> Vec<f32> {
        if let Some(mut buf) = self.buffers.pop() {
            buf.resize(size, 0.0);
            buf
        } else {
            vec![0.0; size]
        }
    }

    /// Return a buffer to the pool for reuse.
    /// Drops the buffer if pool is at max_idle, or if buffer capacity
    /// is more than 2x its length (oversized from a previous larger image).
    pub fn release(&mut self, buf: Vec<f32>) {
        if self.buffers.len() >= self.max_idle {
            return; // drop — pool is full
        }
        if buf.capacity() > buf.len().max(1) * 2 {
            return; // drop — oversized, not worth keeping
        }
        self.buffers.push(buf);
    }

    /// Number of idle buffers in the pool.
    pub fn idle_count(&self) -> usize {
        self.buffers.len()
    }
}

/// GPU execution plan — returned by `Graph::gpu_plan()` for host-side dispatch.
pub struct GpuPlan {
    /// GPU shaders to execute in sequence (ping-pong).
    pub shaders: Vec<crate::node::GpuShader>,
    /// Input f32 pixel data from the source node (upstream of GPU chain).
    pub input_pixels: Vec<f32>,
    /// Width of the input/output image.
    pub width: u32,
    /// Height of the input/output image.
    pub height: u32,
}

// ─── Multi-output GPU plan ─────────────────────────────────────────────────

/// Input source for a GPU stage in a multi-output plan.
#[derive(Debug, Clone)]
pub enum StageInput {
    /// Upload f32 pixels from CPU (source node or CPU-computed result).
    Pixels(Vec<f32>),
    /// Read from a prior stage's GPU output buffer (by target name).
    PriorStage(String),
}

/// A single stage in a multi-output GPU execution plan.
#[derive(Debug)]
pub struct GpuStage {
    /// Display target name this stage serves (e.g., "viewport", "histogram").
    pub target_name: String,
    /// GPU shaders to execute (empty = passthrough, just blit the input).
    pub shaders: Vec<crate::node::GpuShader>,
    /// Where the input data comes from.
    pub input: StageInput,
    /// Output dimensions.
    pub width: u32,
    pub height: u32,
}

/// Multi-output GPU plan — ordered list of stages to execute in one submit.
#[derive(Debug)]
pub struct MultiGpuPlan {
    /// Stages in execution order. Prior-stage refs are always resolved
    /// by an earlier stage in the list.
    pub stages: Vec<GpuStage>,
}

/// The V2 pipeline graph.
///
/// Manages node topology, caching, GPU dispatch, and demand-driven execution.
/// Knows nothing about specific operations — just dispatches through the
/// `Node` trait.
///
/// ## Two-level caching
///
/// - **SpatialCache** (per-pipeline): tile-level reuse within a single execution.
/// - **LayerCache** (cross-pipeline): content-addressed reuse across executions.
///   Injected externally, persists across pipeline lifetimes. Keyed by blake3
///   hash chains that encode the full computation lineage.
pub struct Graph {
    pub(crate) nodes: Vec<Box<dyn Node>>,
    cache: SpatialCache,
    /// Content hash per node — encodes full computation lineage.
    content_hashes: Vec<ContentHash>,
    /// Optional cross-pipeline layer cache (injected, shared).
    layer_cache: Option<Rc<RefCell<LayerCache>>>,
    gpu_executor: Option<Rc<dyn GpuExecutor>>,
    demand_strategy: DemandStrategy,
    pub(crate) aces_strict: bool,
    /// Whether fusion optimization has already run on this graph.
    pub(crate) optimized: bool,
    /// Opt-in tracing — collects timing events when enabled.
    tracing: bool,
    pub(crate) trace: PipelineTrace,
    /// Shared reusable pixel buffer pool — avoids allocation churn in the CPU render path.
    /// Lives outside the graph (e.g. on Source or module instance) and is injected.
    buffer_pool: Option<Rc<RefCell<BufferPool>>>,
    /// Named branch points in the DAG — ref name → node_id.
    /// Used by the multi-output pipeline to fork computation from a named point.
    refs: std::collections::HashMap<String, u32>,
}

impl Graph {
    /// Create a new graph with the given cache budget (bytes).
    /// Uses the thread-local spatial cache pool to avoid allocation churn.
    pub fn new(cache_budget: usize) -> Self {
        // Try to get a pooled cache; fall back to fresh allocation
        let cache = if cache_budget == 16 * 1024 * 1024 {
            // Default budget — use pool
            crate::cache::acquire_pooled_cache()
        } else {
            SpatialCache::new(cache_budget)
        };
        Self {
            nodes: Vec::new(),
            cache,
            content_hashes: Vec::new(),
            layer_cache: None,
            gpu_executor: None,
            demand_strategy: DemandStrategy::default(),
            aces_strict: false,
            optimized: false,
            tracing: false,
            trace: PipelineTrace::new(),
            buffer_pool: None,
            refs: std::collections::HashMap::new(),
        }
    }

    /// Create a graph with a pre-allocated spatial cache (e.g. from a pool).
    pub fn with_cache(cache: SpatialCache) -> Self {
        Self {
            nodes: Vec::new(),
            cache,
            content_hashes: Vec::new(),
            layer_cache: None,
            gpu_executor: None,
            demand_strategy: DemandStrategy::default(),
            aces_strict: false,
            optimized: false,
            tracing: false,
            trace: PipelineTrace::new(),
            buffer_pool: None,
            refs: std::collections::HashMap::new(),
        }
    }

    /// Set a shared buffer pool for pixel buffer reuse across pipeline lifetimes.
    /// Associate with a Source so all pipelines for the same image share buffers.
    pub fn set_buffer_pool(&mut self, pool: Rc<RefCell<BufferPool>>) {
        self.buffer_pool = Some(pool);
    }

    /// Acquire a pixel buffer from the pool (or allocate if no pool set).
    #[allow(dead_code)]
    fn acquire_buffer(&self, size: usize) -> Vec<f32> {
        if let Some(pool) = &self.buffer_pool {
            pool.borrow_mut().acquire(size)
        } else {
            vec![0.0; size]
        }
    }

    /// Return a pixel buffer to the pool for reuse.
    #[allow(dead_code)]
    fn release_buffer(&self, buf: Vec<f32>) {
        if let Some(pool) = &self.buffer_pool {
            pool.borrow_mut().release(buf);
        }
        // If no pool, buf is just dropped (normal allocation behavior)
    }

    /// Add a node to the graph. Returns the node ID.
    ///
    /// The node gets ZERO_HASH as its content hash. Use `add_node_with_hash`
    /// to provide a content hash for layer cache integration.
    ///
    /// # Panics
    /// Panics if any upstream ID references a node that doesn't exist yet
    /// (forward reference), which would create a cycle in the DAG.
    pub fn add_node(&mut self, node: Box<dyn Node>) -> u32 {
        let id = self.nodes.len() as u32;
        self.assert_no_forward_refs(&node, id);
        self.nodes.push(node);
        self.content_hashes.push(ZERO_HASH);
        self.optimized = false; // new node invalidates fusion
        id
    }

    /// Add a node with a pre-computed content hash.
    ///
    /// The content hash encodes the full computation lineage:
    /// `hash(upstream_hash || op_name || param_bytes)`.
    /// Used by the pipeline resource to enable layer cache lookups.
    ///
    /// # Panics
    /// Panics if any upstream ID references a node that doesn't exist yet.
    pub fn add_node_with_hash(&mut self, node: Box<dyn Node>, hash: ContentHash) -> u32 {
        let id = self.nodes.len() as u32;
        self.assert_no_forward_refs(&node, id);
        self.nodes.push(node);
        self.content_hashes.push(hash);
        self.optimized = false;
        id
    }

    /// Verify that all upstream IDs reference existing (earlier) nodes.
    /// The graph is append-only — nodes can only depend on lower IDs.
    /// This is sufficient to guarantee acyclicity.
    fn assert_no_forward_refs(&self, node: &Box<dyn Node>, new_id: u32) {
        for up in node.upstream_ids() {
            assert!(
                up < new_id,
                "cycle detected: node {new_id} references upstream {up} which is >= {new_id}"
            );
        }
    }

    /// Get the content hash for a node.
    pub fn content_hash(&self, node_id: u32) -> ContentHash {
        self.content_hashes.get(node_id as usize).copied().unwrap_or(ZERO_HASH)
    }

    /// Set the cross-pipeline layer cache (injected, shared).
    pub fn set_layer_cache(&mut self, cache: Rc<RefCell<LayerCache>>) {
        self.layer_cache = Some(cache);
    }

    /// Reset layer cache references (call at start of a new pipeline run).
    pub fn reset_layer_cache_references(&self) {
        if let Some(lc) = &self.layer_cache {
            lc.borrow_mut().reset_references();
        }
    }

    /// Clean up unreferenced layer cache entries (call after pipeline completion).
    pub fn cleanup_layer_cache(&self) {
        if let Some(lc) = &self.layer_cache {
            lc.borrow_mut().cleanup_unreferenced();
        }
    }

    /// Set the GPU executor for GPU-primary dispatch.
    pub fn set_gpu_executor(&mut self, executor: Rc<dyn GpuExecutor>) {
        self.gpu_executor = Some(executor);
    }

    /// Set the demand strategy (tile sizing).
    pub fn set_demand_strategy(&mut self, strategy: DemandStrategy) {
        self.demand_strategy = strategy;
    }

    /// Enable or disable pipeline tracing.
    pub fn set_tracing(&mut self, enabled: bool) {
        self.tracing = enabled;
        if !enabled {
            self.trace = PipelineTrace::new();
        }
    }

    /// Take the collected trace data, replacing it with an empty trace.
    pub fn take_trace(&mut self) -> PipelineTrace {
        std::mem::take(&mut self.trace)
    }

    /// Whether tracing is currently enabled.
    pub fn is_tracing(&self) -> bool {
        self.tracing
    }

    // ─── Named refs (DAG branch points) ──────────────────────────────

    /// Set a named ref pointing to a node ID.
    ///
    /// Refs are named branch points in the DAG. Downstream consumers can
    /// fork from a ref to build multiple output branches (e.g., viewport
    /// + scopes) without re-executing the upstream chain.
    pub fn set_ref(&mut self, name: &str, node_id: u32) {
        self.refs.insert(name.to_string(), node_id);
    }

    /// Get the node ID for a named ref, or None if not set.
    pub fn get_ref(&self, name: &str) -> Option<u32> {
        self.refs.get(name).copied()
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

    /// Extract the GPU execution plan for a node without executing it.
    ///
    /// Runs fusion, then collects the GPU shader chain. Returns the shaders,
    /// the source node's pixel data, and dimensions. Returns None if no GPU
    /// chain exists.
    ///
    /// Used by hosts (browser JS) that handle GPU dispatch externally.
    pub fn gpu_plan(
        &mut self,
        node_id: u32,
    ) -> Result<Option<GpuPlan>, PipelineError> {
        // Run fusion first
        if !self.optimized {
            crate::fusion::optimize(self);
            self.optimized = true;
        }

        let info = self.node_info(node_id)?;
        let chain = self.collect_gpu_chain(node_id, info.width, info.height);

        match chain {
            Some((source_id, shaders)) => {
                // GPU chain found — source pixels + shaders (may be empty for passthrough)
                let request = Rect::new(0, 0, info.width, info.height);
                let input = self.request_region(source_id, request)?;
                Ok(Some(GpuPlan {
                    shaders,
                    input_pixels: input,
                    width: info.width,
                    height: info.height,
                }))
            }
            None => {
                // No GPU chain at all — return a passthrough plan with the node's
                // pixels and an empty shader list. The host uploads and blits directly.
                let request = Rect::new(0, 0, info.width, info.height);
                let input = self.request_region(node_id, request)?;
                Ok(Some(GpuPlan {
                    shaders: vec![],
                    input_pixels: input,
                    width: info.width,
                    height: info.height,
                }))
            }
        }
    }

    /// Extract a multi-output GPU execution plan for multiple display targets.
    ///
    /// Each target is a `(name, node_id)` pair. The plan detects shared upstream
    /// chains: if multiple targets branch from the same node (via refs), the shared
    /// chain is computed once and downstream stages read from that buffer.
    ///
    /// Stages are topologically ordered: upstream stages appear first.
    pub fn render_multi_gpu_plan(
        &mut self,
        targets: &[(String, u32)],
    ) -> Result<MultiGpuPlan, PipelineError> {
        if !self.optimized {
            crate::fusion::optimize(self);
            self.optimized = true;
        }

        if targets.is_empty() {
            return Ok(MultiGpuPlan { stages: vec![] });
        }

        // For each target, extract (source_node_id, shaders, output_info)
        struct TargetChain {
            name: String,
            node_id: u32,
            source_id: u32,
            shaders: Vec<crate::node::GpuShader>,
            width: u32,
            height: u32,
        }

        let mut chains = Vec::new();
        for (name, node_id) in targets {
            let info = self.node_info(*node_id)?;
            let chain = self.collect_gpu_chain(*node_id, info.width, info.height);
            match chain {
                Some((source_id, shaders)) => {
                    chains.push(TargetChain {
                        name: name.clone(),
                        node_id: *node_id,
                        source_id,
                        shaders,
                        width: info.width,
                        height: info.height,
                    });
                }
                None => {
                    // No GPU chain — passthrough (pixels only, empty shaders)
                    chains.push(TargetChain {
                        name: name.clone(),
                        node_id: *node_id,
                        source_id: *node_id,
                        shaders: vec![],
                        width: info.width,
                        height: info.height,
                    });
                }
            }
        }

        // Detect shared source nodes. If multiple targets share the same source_id,
        // the first one provides the pixels and subsequent ones use PriorStage.
        //
        // Additionally, if a target's source_id matches another target's node_id
        // (i.e., it branches from a ref that is itself a display target), it uses
        // PriorStage referencing that target.
        let mut stages = Vec::new();
        // Track which source_ids have been computed (name of the stage that computed them)
        let mut computed_sources: std::collections::HashMap<u32, String> = std::collections::HashMap::new();
        // Also track which node_ids are display targets (for ref-based branching)
        let mut target_nodes: std::collections::HashMap<u32, String> = std::collections::HashMap::new();
        for c in &chains {
            target_nodes.insert(c.node_id, c.name.clone());
        }

        // Sort: stages whose source_id is another target's node_id come after that target
        // Simple approach: emit stages with Pixels input first, then PriorStage
        let mut deferred = Vec::new();

        for c in &chains {
            // Check if this target's source was already computed by a prior stage
            if let Some(prior_name) = computed_sources.get(&c.source_id) {
                stages.push(GpuStage {
                    target_name: c.name.clone(),
                    shaders: c.shaders.clone(),
                    input: StageInput::PriorStage(prior_name.clone()),
                    width: c.width,
                    height: c.height,
                });
            } else if let Some(prior_name) = target_nodes.get(&c.source_id) {
                // Source is another target's output — defer until that target is computed
                if prior_name != &c.name {
                    deferred.push(c);
                    continue;
                }
                // Self-reference — treat as pixels
                let request = Rect::new(0, 0, c.width, c.height);
                let pixels = self.request_region(c.source_id, request)?;
                computed_sources.insert(c.source_id, c.name.clone());
                stages.push(GpuStage {
                    target_name: c.name.clone(),
                    shaders: c.shaders.clone(),
                    input: StageInput::Pixels(pixels),
                    width: c.width,
                    height: c.height,
                });
            } else {
                // First time seeing this source — fetch pixels
                let src_info = self.node_info(c.source_id)?;
                let request = Rect::new(0, 0, src_info.width, src_info.height);
                let pixels = self.request_region(c.source_id, request)?;
                computed_sources.insert(c.source_id, c.name.clone());
                stages.push(GpuStage {
                    target_name: c.name.clone(),
                    shaders: c.shaders.clone(),
                    input: StageInput::Pixels(pixels),
                    width: c.width,
                    height: c.height,
                });
            }
        }

        // Emit deferred stages (targets that branch from another target's output)
        for c in deferred {
            let prior_name = target_nodes.get(&c.source_id)
                .or_else(|| computed_sources.get(&c.source_id))
                .cloned()
                .unwrap_or_default();
            stages.push(GpuStage {
                target_name: c.name.clone(),
                shaders: c.shaders.clone(),
                input: StageInput::PriorStage(prior_name),
                width: c.width,
                height: c.height,
            });
        }

        Ok(MultiGpuPlan { stages })
    }

    /// Pre-compile all GPU shaders reachable from `node_id`.
    ///
    /// Walks the graph, collects all unique WGSL shader sources from GPU-capable
    /// nodes, and passes them to the executor's `prepare()` method for compilation
    /// and caching. Subsequent `request_full()` calls will hit O(1) cache lookups.
    ///
    /// No-op if no GPU executor is set.
    pub fn prepare_gpu(&mut self, node_id: u32) -> Result<(), PipelineError> {
        // Run fusion first to ensure FusedPointOpNodes exist
        if !self.optimized {
            crate::fusion::optimize(self);
            self.optimized = true;
        }

        let executor = match &self.gpu_executor {
            Some(e) => e.clone(),
            None => return Ok(()),
        };

        let info = self.node_info(node_id)?;
        let chain = self.collect_gpu_chain(node_id, info.width, info.height);

        if let Some((_source_id, shaders)) = chain {
            let sources: Vec<String> = shaders
                .iter()
                .map(|s| s.body.clone())
                .collect();
            executor.prepare(&sources);
        }

        Ok(())
    }

    /// Inject externally-computed GPU result into the cache.
    ///
    /// After the host executes the GPU plan, call this to cache the result.
    /// Subsequent render()/write() calls for this node will use the cached data.
    pub fn inject_gpu_result(&mut self, node_id: u32, pixels: Vec<f32>) {
        let info = match self.node_info(node_id) {
            Ok(i) => i,
            Err(_) => return,
        };
        let rect = Rect::new(0, 0, info.width, info.height);
        self.cache
            .store(node_id, rect, pixels, crate::hash::ZERO_HASH);
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
        // 0. Layer cache check (content-addressed, cross-pipeline)
        let node_hash = self.content_hash(node_id);
        if node_hash != ZERO_HASH
            && let Some(lc) = &self.layer_cache
            && let Some((pixels, _w, _h)) = lc.borrow_mut().get(&node_hash)
        {
            // Layer cache stores full-node output. If request is a sub-region, crop.
            let info = self.nodes.get(node_id as usize)
                .ok_or(PipelineError::NodeNotFound(node_id))?
                .info();
            let full_rect = Rect::new(0, 0, info.width, info.height);
            if request == full_rect {
                return Ok(pixels);
            }
            return Ok(crop_f32(&pixels, full_rect, request));
        }

        // 1. Spatial cache check (tile-level, per-pipeline)
        if let Some(cached) = self.cache.query(node_id, request) {
            return Ok(cached);
        }

        let node = self
            .nodes
            .get(node_id as usize)
            .ok_or(PipelineError::NodeNotFound(node_id))?;

        let info = node.info();

        // 2. GPU-primary dispatch — batch consecutive GPU nodes
        if let Some(executor) = &self.gpu_executor
            && let Some((source_id, gpu_chain)) = self.collect_gpu_chain(node_id, info.width, info.height)
        {
            let timer = if self.tracing {
                Some(TraceTimer::new(TraceEventKind::GpuDispatch, format!("node_{node_id}"))
                    .with_detail(format!("{} shaders, {}x{}", gpu_chain.len(), request.width, request.height)))
            } else {
                None
            };

            let executor = executor.clone();
            let input = self.request_region(source_id, request)?;
            match executor.execute(&gpu_chain, &input, request.width, request.height) {
                Ok(gpu_pixels) => {
                    if let Some(t) = timer {
                        self.trace.push(t.finish());
                    }
                    self.cache.store(node_id, request, gpu_pixels.clone(), node_hash);
                    return Ok(gpu_pixels);
                }
                Err(_) => {
                    // GPU failed — fall through to CPU
                }
            }
        }

        // 3. CPU fallback
        let cpu_timer = if self.tracing {
            Some(TraceTimer::new(TraceEventKind::CpuFallback, format!("node_{node_id}"))
                .with_detail(format!("{}x{}", request.width, request.height)))
        } else {
            None
        };

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

        if let Some(t) = cpu_timer {
            self.trace.push(t.finish());
        }

        // Spatial cache — store a clone, but recycle the allocation from the pool
        // if available to avoid a fresh 4MB allocation
        let cache_pixels = if let Some(pool) = &self.buffer_pool {
            let mut buf = pool.borrow_mut().acquire(pixels.len());
            buf.copy_from_slice(&pixels);
            buf
        } else {
            pixels.clone()
        };
        self.cache
            .store(node_id, request, cache_pixels, node_hash);

        // Layer cache — store full-node results for cross-pipeline reuse.
        // Only store when the request covers the full node output.
        if node_hash != ZERO_HASH
            && let Some(lc) = &self.layer_cache
        {
            let full_rect = Rect::new(0, 0, info.width, info.height);
            if request == full_rect && !lc.borrow().contains(&node_hash) {
                lc.borrow_mut().store(node_hash, &pixels, info.width, info.height);
            }
        }

        Ok(pixels)
    }

    /// Request the full output of a node (convenience).
    ///
    /// Layer cache storage happens inside `request_region()` when the request
    /// covers the full node output and the node has a content hash.
    pub fn request_full(&mut self, node_id: u32) -> Result<Vec<f32>, PipelineError> {
        // Run fusion optimizer before execution — skip if already optimized
        if !self.optimized {
            let timer = if self.tracing {
                Some(TraceTimer::new(TraceEventKind::Fusion, "optimize")
                    .with_detail(format!("{} nodes", self.nodes.len())))
            } else {
                None
            };

            crate::fusion::optimize(self);
            self.optimized = true;

            if let Some(t) = timer {
                self.trace.push(t.finish());
            }
        }
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

impl Drop for Graph {
    fn drop(&mut self) {
        // Return spatial cache to the pool for reuse.
        // Swap with a zero-budget dummy to take ownership of the allocated cache.
        let cache = std::mem::replace(&mut self.cache, SpatialCache::new(0));
        if cache.usage_bytes() > 0 || cache.budget_bytes() > 0 {
            crate::cache::release_pooled_cache(cache);
        }
    }
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

    #[test]
    fn tracing_captures_cpu_fallback_events() {
        let mut g = Graph::new(0);
        g.set_tracing(true);

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

        let _pixels = g.request_full(scale).unwrap();
        let trace = g.take_trace();

        // Should have a fusion event
        assert!(
            trace.by_kind(crate::trace::TraceEventKind::Fusion).len() == 1,
            "expected 1 fusion event, got {}",
            trace.by_kind(crate::trace::TraceEventKind::Fusion).len()
        );

        // Should have CPU fallback events (no GPU executor set)
        let cpu_events = trace.by_kind(crate::trace::TraceEventKind::CpuFallback);
        assert!(
            cpu_events.len() >= 1,
            "expected at least 1 CPU fallback event, got {}",
            cpu_events.len()
        );
    }

    #[test]
    fn optimize_guard_skips_redundant_fusion() {
        let mut g = Graph::new(0);
        g.set_tracing(true);

        let src = g.add_node(Box::new(SolidColorNode {
            width: 4,
            height: 4,
            color: [0.5, 0.3, 0.1, 1.0],
        }));

        // First request_full — should run fusion
        let _p1 = g.request_full(src).unwrap();
        let t1 = g.take_trace();
        assert_eq!(t1.by_kind(crate::trace::TraceEventKind::Fusion).len(), 1);

        // Second request_full — should skip fusion (optimized flag set)
        g.clear_cache();
        let _p2 = g.request_full(src).unwrap();
        let t2 = g.take_trace();
        assert_eq!(
            t2.by_kind(crate::trace::TraceEventKind::Fusion).len(),
            0,
            "fusion should be skipped on second request_full"
        );

        // Adding a node resets the flag
        let info = g.node_info(src).unwrap();
        let _scale = g.add_node(Box::new(ScaleNode {
            upstream: src,
            factor: 2.0,
            info,
        }));
        assert!(!g.optimized, "adding a node should reset optimized flag");
    }

    #[test]
    fn tracing_disabled_collects_nothing() {
        let mut g = Graph::new(0);
        // tracing is disabled by default

        let src = g.add_node(Box::new(SolidColorNode {
            width: 4,
            height: 4,
            color: [0.5, 0.3, 0.1, 1.0],
        }));

        let _pixels = g.request_full(src).unwrap();
        let trace = g.take_trace();
        assert!(trace.events.is_empty(), "no events when tracing is disabled");
    }

    #[test]
    fn gpu_plan_returns_none_for_non_gpu_nodes() {
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

        // ScaleNode doesn't have gpu_shaders() — plan should be None
        let plan = g.gpu_plan(scale).unwrap();
        assert!(plan.is_none(), "non-GPU nodes should return None gpu_plan");
    }

    #[test]
    fn inject_gpu_result_caches_for_render() {
        let mut g = Graph::new(16 * 1024 * 1024); // enable cache
        let src = g.add_node(Box::new(SolidColorNode {
            width: 2,
            height: 2,
            color: [0.5, 0.3, 0.1, 1.0],
        }));

        // Inject custom pixels as if GPU computed them
        let injected = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                            0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        g.inject_gpu_result(src, injected.clone());

        // Subsequent render should return the injected data (from cache)
        let result = g.request_full(src).unwrap();
        assert_eq!(result, injected, "inject_gpu_result should be cached");
    }

    // ─── Layer Cache Tests ──────────────────────────────────────────────

    /// Helper: build a 5-filter chain with content hashes.
    /// Returns (graph, node_ids) where node_ids[0] = source, node_ids[1..5] = filters.
    fn build_5_filter_chain(
        lc: Rc<RefCell<crate::layer_cache::LayerCache>>,
        factors: [f32; 5],
    ) -> (Graph, Vec<u32>) {
        use crate::hash::{content_hash, source_hash};

        let mut g = Graph::new(0);
        g.set_layer_cache(lc);

        let src_hash = source_hash(b"test image");
        let src = g.add_node_with_hash(
            Box::new(SolidColorNode { width: 2, height: 2, color: [0.5, 0.3, 0.1, 1.0] }),
            src_hash,
        );

        let mut ids = vec![src];
        let mut prev_hash = src_hash;
        let mut prev_id = src;

        for (i, &factor) in factors.iter().enumerate() {
            let info = g.node_info(prev_id).unwrap();
            let h = content_hash(&prev_hash, &format!("scale_{i}"), &factor.to_le_bytes());
            let id = g.add_node_with_hash(
                Box::new(ScaleNode { upstream: prev_id, factor, info }),
                h,
            );
            ids.push(id);
            prev_hash = h;
            prev_id = id;
        }

        (g, ids)
    }

    #[test]
    fn layer_cache_identity_all_hits_on_second_run() {
        let lc = Rc::new(RefCell::new(crate::layer_cache::LayerCache::new(64 * 1024 * 1024)));

        // First run: all misses, all results stored
        let (mut g1, ids1) = build_5_filter_chain(lc.clone(), [1.0, 2.0, 0.5, 1.5, 0.8]);
        let last1 = *ids1.last().unwrap();
        let out1 = g1.request_full(last1).unwrap();

        let stats1 = lc.borrow().stats();
        assert_eq!(stats1.entries, 6, "all 6 nodes (source + 5 filters) should be stored");
        assert_eq!(stats1.hits, 0, "no hits on first run");
        assert_eq!(stats1.misses, 6, "6 misses on first run (one per node)");

        // Second run: same chain, same params
        // Layer cache short-circuits at the last node (instant hit on the final result)
        let (mut g2, ids2) = build_5_filter_chain(lc.clone(), [1.0, 2.0, 0.5, 1.5, 0.8]);
        let last2 = *ids2.last().unwrap();
        let out2 = g2.request_full(last2).unwrap();

        assert_eq!(out1, out2, "cached result should match computed result");

        let stats2 = lc.borrow().stats();
        // The last node hits immediately — no intermediate nodes are checked
        assert_eq!(stats2.hits, 1, "1 hit (last node short-circuits entire chain)");
        assert_eq!(stats2.misses, 6, "no new misses on identical second run");
    }

    #[test]
    fn layer_cache_5_filter_chain_last_param_changed() {
        let lc = Rc::new(RefCell::new(crate::layer_cache::LayerCache::new(64 * 1024 * 1024)));

        // First run: source → scale(1.0) → scale(2.0) → scale(0.5) → scale(1.5) → scale(0.8)
        let (mut g1, ids1) = build_5_filter_chain(lc.clone(), [1.0, 2.0, 0.5, 1.5, 0.8]);
        let last1 = *ids1.last().unwrap();
        let _out1 = g1.request_full(last1).unwrap();

        let stats_after_first = lc.borrow().stats();

        // Second run: change only last param (0.8 → 0.9)
        // Execution trace:
        //   request_region(5) → LC miss (different hash)
        //   node5.compute → request(4) → LC hit! (node4 hash unchanged)
        //   Only node5 recomputes from cached node4.
        let (mut g2, ids2) = build_5_filter_chain(lc.clone(), [1.0, 2.0, 0.5, 1.5, 0.9]);
        let last2 = *ids2.last().unwrap();
        let _out2 = g2.request_full(last2).unwrap();

        let stats_after_second = lc.borrow().stats();
        let new_hits = stats_after_second.hits - stats_after_first.hits;
        let new_misses = stats_after_second.misses - stats_after_first.misses;

        assert_eq!(new_hits, 1, "1 hit (node4 — closest unchanged ancestor)");
        assert_eq!(new_misses, 1, "1 miss (node5 — changed param)");
    }

    #[test]
    fn layer_cache_middle_param_change_invalidates_downstream() {
        let lc = Rc::new(RefCell::new(crate::layer_cache::LayerCache::new(64 * 1024 * 1024)));

        // First run
        let (mut g1, ids1) = build_5_filter_chain(lc.clone(), [1.0, 2.0, 0.5, 1.5, 0.8]);
        let _out1 = g1.request_full(*ids1.last().unwrap()).unwrap();
        let stats1 = lc.borrow().stats();

        // Second run: change filter 1 (index 1), params: [1.0, 3.0, 0.5, 1.5, 0.8]
        // Execution trace:
        //   request(5) → miss, request(4) → miss, request(3) → miss,
        //   request(2) → miss (hash changed), request(1) → hit! (unchanged)
        //   Nodes 2-5 recompute from cached node1.
        let (mut g2, ids2) = build_5_filter_chain(lc.clone(), [1.0, 3.0, 0.5, 1.5, 0.8]);
        let _out2 = g2.request_full(*ids2.last().unwrap()).unwrap();
        let stats2 = lc.borrow().stats();

        let new_hits = stats2.hits - stats1.hits;
        let new_misses = stats2.misses - stats1.misses;

        assert_eq!(new_hits, 1, "1 hit (node1 — closest unchanged ancestor)");
        assert_eq!(new_misses, 4, "4 misses (nodes 2-5 — changed + downstream)");
    }
}
