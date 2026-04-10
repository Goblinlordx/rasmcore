//! Cross-node analysis buffer protocol — declarations, references, and negotiation.
//!
//! Analysis nodes produce small GPU buffers (histograms, channel stats, min/max)
//! that downstream render nodes consume. The buffer negotiation protocol assigns
//! globally unique IDs so that merged GPU shader chains can share reduction
//! buffers across node boundaries.
//!
//! # Protocol
//!
//! 1. Analysis nodes declare outputs via `analysis_outputs()` on the Node trait
//! 2. Render nodes declare inputs via `analysis_inputs()` on the Node trait
//! 3. The graph walker calls `negotiate_analysis_buffers()` to assign resolved IDs
//! 4. Nodes receive resolved IDs via `gpu_shaders_with_context()`
//! 5. The merged shader chain executes in one GPU submit — zero CPU readback

use std::collections::HashMap;

use crate::node::PipelineError;

/// What kind of analysis data a buffer holds.
///
/// This is advisory — the executor treats all reduction buffers as raw bytes.
/// The kind helps with debugging, documentation, and future type-checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnalysisBufferKind {
    /// 768 u32 bins: [0..255]=R, [256..511]=G, [512..767]=B.
    Histogram256,
    /// 2 × vec4<f32>: [0]=(min_r,g,b,_), [1]=(max_r,g,b,_).
    ChannelMinMax,
    /// vec4<f32>: (sum_r, sum_g, sum_b, count).
    ChannelSum,
    /// Application-specific buffer with caller-defined layout.
    Generic,
}

impl AnalysisBufferKind {
    /// Typical buffer size in bytes for this kind.
    pub fn typical_size(self) -> usize {
        match self {
            AnalysisBufferKind::Histogram256 => 768 * 4, // 768 u32s
            AnalysisBufferKind::ChannelMinMax => 32,     // 2 × vec4<f32>
            AnalysisBufferKind::ChannelSum => 16,        // 1 × vec4<f32>
            AnalysisBufferKind::Generic => 0,            // caller must specify
        }
    }
}

/// Declaration of an analysis buffer that a node produces.
///
/// Returned by `Node::analysis_outputs()`. The logical_id is node-local —
/// the negotiation step assigns a globally unique resolved ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnalysisBufferDecl {
    /// Node-local logical ID for this buffer. Matches the ID used in
    /// the node's internal `ReductionBuffer` declarations.
    pub logical_id: u32,
    /// What kind of data this buffer holds.
    pub kind: AnalysisBufferKind,
    /// Buffer size in bytes. If 0, uses `kind.typical_size()`.
    pub size_bytes: usize,
}

impl AnalysisBufferDecl {
    /// Create a new buffer declaration.
    pub fn new(logical_id: u32, kind: AnalysisBufferKind) -> Self {
        Self {
            logical_id,
            kind,
            size_bytes: kind.typical_size(),
        }
    }

    /// Create with explicit size (for Generic or non-standard sizes).
    pub fn with_size(logical_id: u32, kind: AnalysisBufferKind, size_bytes: usize) -> Self {
        Self {
            logical_id,
            kind,
            size_bytes,
        }
    }

    /// Effective size — explicit size if set, otherwise typical for the kind.
    pub fn effective_size(&self) -> usize {
        if self.size_bytes > 0 {
            self.size_bytes
        } else {
            self.kind.typical_size()
        }
    }
}

/// Reference to an analysis buffer that a node consumes.
///
/// Returned by `Node::analysis_inputs()`. The logical_id must match
/// a declared `AnalysisBufferDecl::logical_id` from an upstream analysis node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnalysisBufferRef {
    /// Logical ID of the upstream analysis buffer to consume.
    /// Must match an `AnalysisBufferDecl::logical_id` from an upstream node.
    pub logical_id: u32,
}

impl AnalysisBufferRef {
    /// Create a new buffer reference.
    pub fn new(logical_id: u32) -> Self {
        Self { logical_id }
    }
}

/// Resolved buffer ID mapping for a merged GPU shader chain.
///
/// Created by `negotiate_analysis_buffers()`. Passed to
/// `Node::gpu_shaders_with_context()` so nodes can use globally unique
/// buffer IDs in their shader declarations.
#[derive(Debug, Clone, Default)]
pub struct AnalysisBufferContext {
    /// Maps (node_index_in_chain, logical_id) → resolved_id.
    /// Node index is the position in the merged chain's node list.
    resolved: HashMap<(usize, u32), u32>,
    /// Next available resolved ID.
    next_id: u32,
}

/// Resolved ID mapping for a single node within the chain.
///
/// A lightweight view into `AnalysisBufferContext` for one specific node.
/// This is what `gpu_shaders_with_context()` receives.
#[derive(Debug, Clone, Default)]
pub struct NodeBufferMapping {
    /// Maps logical_id → resolved_id for this node's buffers.
    pub mapping: HashMap<u32, u32>,
}

impl NodeBufferMapping {
    /// Look up the resolved ID for a logical buffer ID.
    /// Returns the logical ID unchanged if no mapping exists (backward compatible).
    pub fn resolve(&self, logical_id: u32) -> u32 {
        self.mapping.get(&logical_id).copied().unwrap_or(logical_id)
    }

    /// True if this mapping has any entries (node participates in cross-node analysis).
    pub fn is_empty(&self) -> bool {
        self.mapping.is_empty()
    }
}

/// Base offset for resolved cross-node buffer IDs.
/// Starts at 1000 to avoid collision with node-internal buffer IDs (typically 0-based).
const RESOLVED_ID_BASE: u32 = 1000;

impl AnalysisBufferContext {
    /// Create a new empty context.
    pub fn new() -> Self {
        Self {
            resolved: HashMap::new(),
            next_id: RESOLVED_ID_BASE,
        }
    }

    /// Register a producer (analysis node) and assign resolved IDs for its buffers.
    pub fn register_producer(&mut self, node_index: usize, decls: &[AnalysisBufferDecl]) {
        for decl in decls {
            let resolved_id = self.next_id;
            self.next_id += 1;
            self.resolved
                .insert((node_index, decl.logical_id), resolved_id);
        }
    }

    /// Register a consumer (render node) by linking its logical IDs to the
    /// nearest upstream producer's resolved IDs.
    ///
    /// `producer_index` is the node index of the upstream analysis node.
    pub fn register_consumer(
        &mut self,
        consumer_index: usize,
        refs: &[AnalysisBufferRef],
        producer_index: usize,
    ) {
        for r in refs {
            if let Some(&resolved_id) = self.resolved.get(&(producer_index, r.logical_id)) {
                self.resolved
                    .insert((consumer_index, r.logical_id), resolved_id);
            }
        }
    }

    /// Get the resolved buffer mapping for a specific node in the chain.
    pub fn node_mapping(&self, node_index: usize) -> NodeBufferMapping {
        let mut mapping = HashMap::new();
        for (&(idx, logical_id), &resolved_id) in &self.resolved {
            if idx == node_index {
                mapping.insert(logical_id, resolved_id);
            }
        }
        NodeBufferMapping { mapping }
    }

    /// True if this context has any resolved mappings.
    pub fn is_empty(&self) -> bool {
        self.resolved.is_empty()
    }

    /// Number of unique resolved buffer IDs assigned.
    pub fn resolved_count(&self) -> usize {
        let unique: std::collections::HashSet<u32> = self.resolved.values().copied().collect();
        unique.len()
    }
}

/// Metadata about a node in a chain, for negotiation purposes.
#[derive(Debug)]
pub struct ChainNodeInfo<'a> {
    /// Index in the chain (0 = most upstream).
    pub index: usize,
    /// Analysis buffers this node produces.
    pub outputs: &'a [AnalysisBufferDecl],
    /// Analysis buffers this node consumes.
    pub inputs: &'a [AnalysisBufferRef],
}

/// Negotiate analysis buffer IDs for a chain of nodes.
///
/// Walks the chain (upstream to downstream), assigns globally unique IDs
/// to each producer's buffers, and links consumers to the nearest upstream
/// producer with a matching logical ID.
///
/// Returns an error if a consumer references a logical ID that no upstream
/// producer declared.
pub fn negotiate_analysis_buffers(
    chain: &[ChainNodeInfo<'_>],
) -> Result<AnalysisBufferContext, PipelineError> {
    let mut ctx = AnalysisBufferContext::new();

    // Track the most recent producer for each logical_id.
    // When a consumer references logical_id X, it links to the last producer of X.
    let mut latest_producer: HashMap<u32, usize> = HashMap::new();

    for node_info in chain {
        // Register outputs (producer)
        if !node_info.outputs.is_empty() {
            ctx.register_producer(node_info.index, node_info.outputs);
            for decl in node_info.outputs {
                latest_producer.insert(decl.logical_id, node_info.index);
            }
        }

        // Register inputs (consumer) — link to nearest upstream producer
        if !node_info.inputs.is_empty() {
            for r in node_info.inputs {
                let producer_idx = latest_producer.get(&r.logical_id).ok_or_else(|| {
                    PipelineError::InvalidParams(format!(
                        "analysis buffer consumer at chain index {} references logical_id {} \
                         but no upstream producer declared it",
                        node_info.index, r.logical_id
                    ))
                })?;
                ctx.register_consumer(node_info.index, &[r.clone()], *producer_idx);
            }
        }
    }

    Ok(ctx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_kind_typical_sizes() {
        assert_eq!(AnalysisBufferKind::Histogram256.typical_size(), 768 * 4);
        assert_eq!(AnalysisBufferKind::ChannelMinMax.typical_size(), 32);
        assert_eq!(AnalysisBufferKind::ChannelSum.typical_size(), 16);
        assert_eq!(AnalysisBufferKind::Generic.typical_size(), 0);
    }

    #[test]
    fn decl_effective_size() {
        let d = AnalysisBufferDecl::new(0, AnalysisBufferKind::Histogram256);
        assert_eq!(d.effective_size(), 768 * 4);

        let d2 = AnalysisBufferDecl::with_size(0, AnalysisBufferKind::Generic, 128);
        assert_eq!(d2.effective_size(), 128);
    }

    #[test]
    fn node_buffer_mapping_resolve() {
        let mut m = NodeBufferMapping::default();
        // No mapping — returns logical_id unchanged
        assert_eq!(m.resolve(0), 0);
        assert!(m.is_empty());

        m.mapping.insert(0, 1000);
        assert_eq!(m.resolve(0), 1000);
        assert_eq!(m.resolve(1), 1); // unmapped → pass-through
        assert!(!m.is_empty());
    }

    #[test]
    fn context_register_producer() {
        let mut ctx = AnalysisBufferContext::new();
        ctx.register_producer(
            0,
            &[AnalysisBufferDecl::new(0, AnalysisBufferKind::Histogram256)],
        );

        let mapping = ctx.node_mapping(0);
        assert_eq!(mapping.resolve(0), RESOLVED_ID_BASE);
        assert_eq!(ctx.resolved_count(), 1);
    }

    #[test]
    fn context_producer_consumer_link() {
        let mut ctx = AnalysisBufferContext::new();
        // Node 0 produces buffer logical_id=0
        ctx.register_producer(
            0,
            &[AnalysisBufferDecl::new(
                0,
                AnalysisBufferKind::ChannelMinMax,
            )],
        );
        // Node 2 consumes buffer logical_id=0
        ctx.register_consumer(2, &[AnalysisBufferRef::new(0)], 0);

        let producer_map = ctx.node_mapping(0);
        let consumer_map = ctx.node_mapping(2);

        // Both should resolve to the same resolved ID
        assert_eq!(producer_map.resolve(0), consumer_map.resolve(0));
        assert_eq!(producer_map.resolve(0), RESOLVED_ID_BASE);
    }

    #[test]
    fn negotiate_single_pair() {
        let outputs = [AnalysisBufferDecl::new(0, AnalysisBufferKind::Histogram256)];
        let inputs = [AnalysisBufferRef::new(0)];

        let chain = [
            ChainNodeInfo {
                index: 0,
                outputs: &outputs,
                inputs: &[],
            },
            ChainNodeInfo {
                index: 1,
                outputs: &[],
                inputs: &[],
            }, // passthrough
            ChainNodeInfo {
                index: 2,
                outputs: &[],
                inputs: &inputs,
            },
        ];

        let ctx = negotiate_analysis_buffers(&chain).unwrap();
        let prod = ctx.node_mapping(0);
        let cons = ctx.node_mapping(2);

        assert_eq!(prod.resolve(0), cons.resolve(0));
        assert_eq!(ctx.resolved_count(), 1);
    }

    #[test]
    fn negotiate_two_independent_producers() {
        let outputs_a = [AnalysisBufferDecl::new(0, AnalysisBufferKind::Histogram256)];
        let outputs_b = [AnalysisBufferDecl::new(
            0,
            AnalysisBufferKind::ChannelMinMax,
        )];
        let inputs_a = [AnalysisBufferRef::new(0)];
        let inputs_b = [AnalysisBufferRef::new(0)];

        // Chain: producer_A → consumer_A → producer_B → consumer_B
        // Both use logical_id=0 but should get different resolved IDs.
        let chain = [
            ChainNodeInfo {
                index: 0,
                outputs: &outputs_a,
                inputs: &[],
            },
            ChainNodeInfo {
                index: 1,
                outputs: &[],
                inputs: &inputs_a,
            },
            ChainNodeInfo {
                index: 2,
                outputs: &outputs_b,
                inputs: &[],
            },
            ChainNodeInfo {
                index: 3,
                outputs: &[],
                inputs: &inputs_b,
            },
        ];

        let ctx = negotiate_analysis_buffers(&chain).unwrap();

        let prod_a = ctx.node_mapping(0);
        let cons_a = ctx.node_mapping(1);
        let prod_b = ctx.node_mapping(2);
        let cons_b = ctx.node_mapping(3);

        // A's pair shares one resolved ID
        assert_eq!(prod_a.resolve(0), cons_a.resolve(0));
        // B's pair shares a different resolved ID
        assert_eq!(prod_b.resolve(0), cons_b.resolve(0));
        // The two pairs have different resolved IDs
        assert_ne!(prod_a.resolve(0), prod_b.resolve(0));
        assert_eq!(ctx.resolved_count(), 2);
    }

    #[test]
    fn negotiate_unmatched_consumer_errors() {
        let inputs = [AnalysisBufferRef::new(5)]; // no producer for id=5

        let chain = [
            ChainNodeInfo {
                index: 0,
                outputs: &[],
                inputs: &[],
            },
            ChainNodeInfo {
                index: 1,
                outputs: &[],
                inputs: &inputs,
            },
        ];

        let result = negotiate_analysis_buffers(&chain);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("logical_id 5"),
            "error should mention the ID: {err}"
        );
    }

    #[test]
    fn negotiate_sequential_analysis_render_analysis_render() {
        // analysis1 → render1 → analysis2 → render2
        // analysis1 produces histogram (id=0), render1 consumes it
        // analysis2 produces minmax (id=0), render2 consumes it
        let hist_out = [AnalysisBufferDecl::new(0, AnalysisBufferKind::Histogram256)];
        let hist_in = [AnalysisBufferRef::new(0)];
        let mm_out = [AnalysisBufferDecl::new(
            0,
            AnalysisBufferKind::ChannelMinMax,
        )];
        let mm_in = [AnalysisBufferRef::new(0)];

        let chain = [
            ChainNodeInfo {
                index: 0,
                outputs: &hist_out,
                inputs: &[],
            },
            ChainNodeInfo {
                index: 1,
                outputs: &[],
                inputs: &hist_in,
            },
            ChainNodeInfo {
                index: 2,
                outputs: &mm_out,
                inputs: &[],
            },
            ChainNodeInfo {
                index: 3,
                outputs: &[],
                inputs: &mm_in,
            },
        ];

        let ctx = negotiate_analysis_buffers(&chain).unwrap();

        // First pair
        let a1 = ctx.node_mapping(0);
        let r1 = ctx.node_mapping(1);
        assert_eq!(a1.resolve(0), r1.resolve(0));

        // Second pair
        let a2 = ctx.node_mapping(2);
        let r2 = ctx.node_mapping(3);
        assert_eq!(a2.resolve(0), r2.resolve(0));

        // Different resolved IDs between pairs
        assert_ne!(a1.resolve(0), a2.resolve(0));
    }

    #[test]
    fn context_empty_by_default() {
        let ctx = AnalysisBufferContext::new();
        assert!(ctx.is_empty());
        assert_eq!(ctx.resolved_count(), 0);
    }

    #[test]
    fn negotiate_empty_chain() {
        let chain: &[ChainNodeInfo<'_>] = &[];
        let ctx = negotiate_analysis_buffers(chain).unwrap();
        assert!(ctx.is_empty());
    }

    #[test]
    fn negotiate_no_analysis_nodes() {
        // All nodes are plain render nodes — no analysis outputs or inputs
        let chain = [
            ChainNodeInfo {
                index: 0,
                outputs: &[],
                inputs: &[],
            },
            ChainNodeInfo {
                index: 1,
                outputs: &[],
                inputs: &[],
            },
            ChainNodeInfo {
                index: 2,
                outputs: &[],
                inputs: &[],
            },
        ];

        let ctx = negotiate_analysis_buffers(&chain).unwrap();
        assert!(ctx.is_empty());
    }
}
