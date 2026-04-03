//! Demand regulation — controls how tile requests propagate through the graph.
//!
//! Three-layer cascade (CSS-like):
//! 1. Pipeline-level DemandStrategy (global baseline)
//! 2. DemandRegulator node (subgraph override)
//! 3. Node TileHint (advisory, informational)

/// Pipeline-level demand strategy. Sets the baseline tile size for the graph.
///
/// Zero-config default is `Fixed { tile_size: 512 }` — identical to V1 behavior.
#[derive(Debug, Clone)]
pub enum DemandStrategy {
    /// Fixed tile size for all nodes.
    Fixed { tile_size: u32 },

    /// Adapt tile size to fit within a total memory budget.
    /// Computes: `tile_size = sqrt(budget / (16 * pipeline_depth))`
    /// where 16 = bytes per f32 RGBA pixel.
    AdaptiveMemory { budget_bytes: usize },

    /// Separate CPU and GPU tile sizing. GPU tiles sized to VRAM budget.
    AdaptiveVram {
        vram_budget_bytes: usize,
        cpu_tile_size: u32,
    },
}

impl Default for DemandStrategy {
    fn default() -> Self {
        DemandStrategy::Fixed { tile_size: 512 }
    }
}

impl DemandStrategy {
    /// Resolve the effective tile size for a given pixel depth.
    ///
    /// `bpp` is always 16 (4 channels * 4 bytes) in V2 f32 pipeline.
    /// `depth` is the maximum chain length (nodes between source and sink).
    pub fn resolve_tile_size(&self, depth: u32) -> u32 {
        let bpp = 16u64; // f32 RGBA = 16 bytes per pixel
        match self {
            DemandStrategy::Fixed { tile_size } => *tile_size,
            DemandStrategy::AdaptiveMemory { budget_bytes } => {
                // Each tile in the chain needs w*h*16 bytes.
                // Total: depth * tile_size^2 * 16 <= budget
                let budget = *budget_bytes as u64;
                let max_tile_pixels = budget / (bpp * depth.max(1) as u64);
                let tile = (max_tile_pixels as f64).sqrt() as u32;
                tile.clamp(64, 8192) // floor at 64, ceiling at 8K
            }
            DemandStrategy::AdaptiveVram {
                vram_budget_bytes,
                cpu_tile_size,
            } => {
                // For GPU: tile sized to VRAM budget (2 ping-pong buffers)
                let budget = *vram_budget_bytes as u64;
                let max_pixels = budget / (bpp * 2); // 2 buffers
                let gpu_tile = (max_pixels as f64).sqrt() as u32;
                gpu_tile.clamp(64, 8192).min(*cpu_tile_size)
            }
        }
    }
}

/// Subgraph demand override — inserted as a node in the graph.
///
/// Controls demand propagation for all nodes upstream of this point.
/// The graph walker checks for regulators when determining tile size.
#[derive(Debug, Clone)]
pub enum DemandHint {
    /// Fixed tile size for this subgraph.
    TileSize(u32),

    /// Adapt tile size to fit within this memory budget.
    MemoryBudget { budget_bytes: usize },

    /// Use node-reported preferred tile sizes (TileHint trait).
    Auto,
}
