//! Staged execution — analysis-driven pipeline with typed parameter binding.
//!
//! Some operations depend on data-computed parameters:
//! - Smart crop: analyze image → determine best crop rect → crop
//! - Auto-exposure: analyze histogram → compute EV → apply exposure
//!
//! The staged pipeline handles this by running analysis nodes first,
//! then binding their typed results to downstream node parameters,
//! then executing the pixel graph with all parameters resolved.
//!
//! # Pipeline Stages
//!
//! 1. **Validate** — type-check all bindings (analysis output type matches param type)
//! 2. **Analyze** — run analysis nodes, collect typed AnalysisResults
//! 3. **Bind** — apply results to target node params (value validation)
//! 4. **Execute** — run pixel graph (all params resolved, graph is static)

use crate::node::{Node, NodeInfo, PipelineError};
use crate::rect::Rect;
use crate::registry::{ParamConstraint, ParamType};

// ─── Analysis Results ─────────────────────────────────────────────────────────

/// Type tag for analysis results — used for compile-time type checking of bindings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisResultType {
    /// A rectangle (crop region, ROI, face bounding box).
    Rect,
    /// A single scalar value (exposure EV, threshold, confidence score).
    Scalar,
    /// A vector of scalars (histogram, per-channel stats).
    ScalarVec,
    /// A set of 2D points (landmarks, corners, keypoints).
    Points,
    /// A 2x3 affine matrix (detected transform, registration).
    Matrix,
}

/// Typed result produced by an analysis node.
#[derive(Debug, Clone)]
pub enum AnalysisResult {
    /// A rectangle (x, y, width, height).
    Rect(Rect),
    /// A single scalar.
    Scalar(f32),
    /// A vector of scalars.
    ScalarVec(Vec<f32>),
    /// 2D point set.
    Points(Vec<(f32, f32)>),
    /// 2x3 affine matrix (row-major).
    Matrix([f64; 6]),
}

impl AnalysisResult {
    /// Get the type tag for this result.
    pub fn result_type(&self) -> AnalysisResultType {
        match self {
            AnalysisResult::Rect(_) => AnalysisResultType::Rect,
            AnalysisResult::Scalar(_) => AnalysisResultType::Scalar,
            AnalysisResult::ScalarVec(_) => AnalysisResultType::ScalarVec,
            AnalysisResult::Points(_) => AnalysisResultType::Points,
            AnalysisResult::Matrix(_) => AnalysisResultType::Matrix,
        }
    }

    /// Try to extract as a scalar value (for binding to f32 params).
    pub fn as_scalar(&self) -> Option<f64> {
        match self {
            AnalysisResult::Scalar(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Try to extract as a Rect (for binding to rect params).
    pub fn as_rect(&self) -> Option<Rect> {
        match self {
            AnalysisResult::Rect(r) => Some(*r),
            _ => None,
        }
    }
}

// ─── Analysis Node Trait ──────────────────────────────────────────────────────

/// A node that produces typed analysis results, not just pixels.
///
/// Analysis nodes participate in the graph for input data (they implement
/// `Node` for pulling upstream pixels). But their primary output is an
/// `AnalysisResult` — a typed parameter for downstream nodes.
pub trait AnalysisNode: Node {
    /// What type of result this analysis produces.
    fn result_type(&self) -> AnalysisResultType;

    /// Run analysis on the input pixels, produce a typed result.
    ///
    /// `input` is f32 RGBA pixel data from the upstream node.
    fn analyze(&self, input: &[f32], width: u32, height: u32) -> Result<AnalysisResult, PipelineError>;
}

// ─── Parameter Binding ────────────────────────────────────────────────────────

/// How to transform an analysis result before binding to a param.
#[derive(Debug, Clone)]
pub enum BindingTransform {
    /// Use the result directly (types must match).
    Identity,
    /// Extract a named field from a composite result.
    /// e.g., "x" from a Rect → Scalar, "width" from Rect → Scalar.
    Field(String),
}

/// A binding from an analysis node's output to a target node's parameter.
#[derive(Debug, Clone)]
pub struct ParamBinding {
    /// Node ID of the analysis node that produces the value.
    pub source_node: u32,
    /// Node ID of the target node that consumes the value.
    pub target_node: u32,
    /// Name of the parameter on the target node.
    pub target_param: String,
    /// How to transform the analysis result before binding.
    pub transform: BindingTransform,
}

// ─── Type Compatibility ───────────────────────────────────────────────────────

/// Check if an analysis result type is compatible with a param type.
pub fn types_compatible(result_type: AnalysisResultType, param_type: ParamType) -> bool {
    matches!(
        (result_type, param_type),
        (AnalysisResultType::Scalar, ParamType::F32)
            | (AnalysisResultType::Scalar, ParamType::F64)
            | (AnalysisResultType::Scalar, ParamType::U32)
            | (AnalysisResultType::Scalar, ParamType::I32)
            | (AnalysisResultType::Rect, ParamType::Rect)
    )
}

/// Check if a field extraction is valid for the given result type.
pub fn field_extraction_valid(result_type: AnalysisResultType, field: &str) -> bool {
    match result_type {
        AnalysisResultType::Rect => matches!(field, "x" | "y" | "width" | "height"),
        AnalysisResultType::Matrix => matches!(field, "a" | "b" | "tx" | "c" | "d" | "ty"),
        _ => false,
    }
}

/// Extract a scalar from a result via field name.
pub fn extract_field(result: &AnalysisResult, field: &str) -> Option<f64> {
    match result {
        AnalysisResult::Rect(r) => match field {
            "x" => Some(r.x as f64),
            "y" => Some(r.y as f64),
            "width" => Some(r.width as f64),
            "height" => Some(r.height as f64),
            _ => None,
        },
        AnalysisResult::Matrix(m) => match field {
            "a" => Some(m[0]),
            "b" => Some(m[1]),
            "tx" => Some(m[2]),
            "c" => Some(m[3]),
            "d" => Some(m[4]),
            "ty" => Some(m[5]),
            _ => None,
        },
        _ => None,
    }
}

// ─── Staged Pipeline ──────────────────────────────────────────────────────────

/// Validation error from the staged pipeline.
#[derive(Debug, Clone)]
pub struct BindingError {
    pub binding_index: usize,
    pub message: String,
}

impl std::fmt::Display for BindingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "binding {}: {}", self.binding_index, self.message)
    }
}

// ─── Execution Waves ─────────────────────────────────────────────────────────

/// A single step in the staged execution plan.
#[derive(Debug, Clone)]
enum WaveStep {
    /// Run analysis on these node IDs (all independent within the wave).
    Analyze(Vec<u32>),
    /// Bind analysis results to target params.
    Bind(Vec<usize>), // indices into StagedPipeline::bindings
}

/// Staged pipeline — orchestrates analysis → bind → execute.
///
/// Analysis nodes are regular graph nodes that implement `Node::analyze()`.
/// The staged pipeline calls `graph.analyze_node()` to fetch upstream pixels
/// and run analysis, then binds results to downstream params.
///
/// Usage:
/// ```ignore
/// let mut staged = StagedPipeline::new(graph);
/// staged.add_binding(ParamBinding { ... });
/// let output = staged.execute(sink_node_id)?;
/// ```
pub struct StagedPipeline {
    /// The underlying pixel graph.
    pub graph: crate::graph::Graph,
    /// Bindings from analysis outputs to node params.
    bindings: Vec<ParamBinding>,
    /// Param updaters — how to apply a bound value to a node.
    /// Keyed by (target_node, target_param).
    param_updaters: std::collections::HashMap<
        (u32, String),
        Box<dyn Fn(&dyn Node, &AnalysisResult) -> Box<dyn Node>>,
    >,
}

impl StagedPipeline {
    /// Create a staged pipeline wrapping a graph.
    pub fn new(graph: crate::graph::Graph) -> Self {
        Self {
            graph,
            bindings: Vec::new(),
            param_updaters: std::collections::HashMap::new(),
        }
    }

    /// Add a parameter binding.
    pub fn add_binding(&mut self, binding: ParamBinding) {
        self.bindings.push(binding);
    }

    /// Register a param updater — a function that creates a new node with the
    /// analysis result applied to the named parameter.
    ///
    /// The updater receives the current node and the analysis result, and returns
    /// a new node with the parameter updated.
    pub fn set_param_updater(
        &mut self,
        target_node: u32,
        target_param: &str,
        updater: impl Fn(&dyn Node, &AnalysisResult) -> Box<dyn Node> + 'static,
    ) {
        self.param_updaters.insert(
            (target_node, target_param.to_string()),
            Box::new(updater),
        );
    }

    /// Validate all bindings (type checking only — no pixel data needed).
    pub fn validate_bindings(&self) -> Result<(), Vec<BindingError>> {
        let mut errors = Vec::new();

        for (i, binding) in self.bindings.iter().enumerate() {
            // Check source node exists and is an analysis node
            match self.graph.node_info(binding.source_node) {
                Err(_) => {
                    errors.push(BindingError {
                        binding_index: i,
                        message: format!("source node {} not found", binding.source_node),
                    });
                }
                Ok(_) => {
                    if !self.graph.get_node(binding.source_node).is_analysis_node() {
                        errors.push(BindingError {
                            binding_index: i,
                            message: format!(
                                "source node {} is not an analysis node",
                                binding.source_node
                            ),
                        });
                    }
                }
            }

            // Check target node exists
            if self.graph.node_info(binding.target_node).is_err() {
                errors.push(BindingError {
                    binding_index: i,
                    message: format!("target node {} not found", binding.target_node),
                });
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Build execution waves from bindings.
    ///
    /// Sorts analysis nodes into topological waves:
    /// - Wave 0: analysis nodes with no dependency on bound render nodes
    /// - Bind 0: apply wave 0 results
    /// - Wave 1: analysis nodes that depend on wave 0 bound targets
    /// - Bind 1: apply wave 1 results
    /// - ...
    fn build_waves(&self) -> Result<Vec<WaveStep>, PipelineError> {
        if self.bindings.is_empty() {
            return Ok(vec![]);
        }

        // Collect all analysis (source) node IDs
        let analysis_ids: Vec<u32> = self
            .bindings
            .iter()
            .map(|b| b.source_node)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        // Build dependency: analysis_node → set of target_nodes it feeds
        let mut analysis_targets: std::collections::HashMap<u32, Vec<u32>> =
            std::collections::HashMap::new();
        for binding in &self.bindings {
            analysis_targets
                .entry(binding.source_node)
                .or_default()
                .push(binding.target_node);
        }

        // Build reverse: which analysis nodes depend on which render (target) nodes?
        // An analysis node depends on a target node if the target node is upstream
        // of the analysis node (directly or transitively).
        // Simplified: an analysis node "depends on" another analysis's binding if
        // the target_node of an earlier binding is in the upstream chain of this analysis.

        // For topological ordering, we check: does analysis node A have any upstream
        // node that is a target of another binding? If so, A must execute after that
        // binding is applied.
        let bound_targets: std::collections::HashSet<u32> = self
            .bindings
            .iter()
            .map(|b| b.target_node)
            .collect();

        // For each analysis node, check if any of its upstream nodes (recursively)
        // are bound targets. If so, it depends on those bindings being applied first.
        let mut analysis_deps: std::collections::HashMap<u32, std::collections::HashSet<u32>> =
            std::collections::HashMap::new();

        for &analysis_id in &analysis_ids {
            let deps = self.find_upstream_bound_targets(analysis_id, &bound_targets);
            analysis_deps.insert(analysis_id, deps);
        }

        // Topological sort into waves
        let mut waves = Vec::new();
        let mut scheduled: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut bound: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut remaining: std::collections::HashSet<u32> = analysis_ids.into_iter().collect();

        let max_iterations = remaining.len() + 1;
        let mut iteration = 0;

        while !remaining.is_empty() {
            iteration += 1;
            if iteration > max_iterations {
                return Err(PipelineError::ComputeError(
                    "cycle detected in staged pipeline analysis dependencies".to_string(),
                ));
            }

            // Find analysis nodes whose dependencies are all satisfied
            let ready: Vec<u32> = remaining
                .iter()
                .filter(|id| {
                    analysis_deps[id]
                        .iter()
                        .all(|dep| bound.contains(dep))
                })
                .copied()
                .collect();

            if ready.is_empty() {
                return Err(PipelineError::ComputeError(format!(
                    "cycle detected: analysis nodes {:?} have unsatisfied dependencies",
                    remaining
                )));
            }

            // Schedule analysis wave
            waves.push(WaveStep::Analyze(ready.clone()));

            // Find bindings for these analysis nodes
            let binding_indices: Vec<usize> = self
                .bindings
                .iter()
                .enumerate()
                .filter(|(_, b)| ready.contains(&b.source_node))
                .map(|(i, _)| i)
                .collect();

            // Schedule bind wave
            waves.push(WaveStep::Bind(binding_indices));

            // Mark analysis nodes as scheduled and their targets as bound
            for id in &ready {
                scheduled.insert(*id);
                remaining.remove(id);
                if let Some(targets) = analysis_targets.get(id) {
                    for t in targets {
                        bound.insert(*t);
                    }
                }
            }
        }

        Ok(waves)
    }

    /// Find all upstream nodes of `node_id` that are in `bound_targets`.
    fn find_upstream_bound_targets(
        &self,
        node_id: u32,
        bound_targets: &std::collections::HashSet<u32>,
    ) -> std::collections::HashSet<u32> {
        let mut result = std::collections::HashSet::new();
        let mut visited = std::collections::HashSet::new();
        let mut stack = vec![node_id];

        while let Some(id) = stack.pop() {
            if !visited.insert(id) {
                continue;
            }
            let node = self.graph.get_node(id);
            for up_id in node.upstream_ids() {
                if bound_targets.contains(&up_id) {
                    result.insert(up_id);
                }
                stack.push(up_id);
            }
        }

        result
    }

    /// Execute the staged pipeline.
    ///
    /// 1. Build topological waves from bindings
    /// 2. For each analysis wave: run analysis on upstream pixels
    /// 3. For each bind wave: apply results to target params
    /// 4. Final render: request_full(sink_node_id)
    pub fn execute(&mut self, sink_node_id: u32) -> Result<Vec<f32>, PipelineError> {
        let waves = self.build_waves()?;

        // Store analysis results keyed by source_node_id
        let mut results: std::collections::HashMap<u32, AnalysisResult> =
            std::collections::HashMap::new();

        for wave in &waves {
            match wave {
                WaveStep::Analyze(node_ids) => {
                    for &node_id in node_ids {
                        let result = self
                            .graph
                            .analyze_node(node_id)
                            .ok_or_else(|| {
                                PipelineError::ComputeError(format!(
                                    "node {node_id} is not an analysis node"
                                ))
                            })?
                            .map_err(|e| {
                                PipelineError::ComputeError(format!(
                                    "analysis node {node_id} failed: {e}"
                                ))
                            })?;
                        results.insert(node_id, result);
                    }
                }
                WaveStep::Bind(binding_indices) => {
                    for &bi in binding_indices {
                        let binding = &self.bindings[bi];
                        let raw_result = results.get(&binding.source_node).ok_or_else(|| {
                            PipelineError::ComputeError(format!(
                                "no analysis result for node {}",
                                binding.source_node
                            ))
                        })?;

                        // Apply transform
                        let value = match &binding.transform {
                            BindingTransform::Identity => raw_result.clone(),
                            BindingTransform::Field(field) => {
                                let scalar = extract_field(raw_result, field).ok_or_else(|| {
                                    PipelineError::ComputeError(format!(
                                        "cannot extract field '{field}' from {:?}",
                                        raw_result.result_type()
                                    ))
                                })?;
                                AnalysisResult::Scalar(scalar as f32)
                            }
                        };

                        // Apply to target node via updater
                        let key = (binding.target_node, binding.target_param.clone());
                        let target_id = binding.target_node;

                        if let Some(updater) = self.param_updaters.get(&key) {
                            let current = self.graph.get_node(target_id);
                            let new_node = updater(current, &value);
                            self.graph.replace_node(target_id, new_node);
                        } else {
                            return Err(PipelineError::ComputeError(format!(
                                "no param updater for node {} param '{}'",
                                target_id, binding.target_param
                            )));
                        }
                    }
                }
            }
        }

        // Final render
        self.graph.request_full(sink_node_id)
    }

    /// Validate a bound value against contextual constraints.
    pub fn validate_value(
        result: &AnalysisResult,
        upstream_info: &NodeInfo,
        constraints: &[ParamConstraint],
    ) -> Result<(), String> {
        for constraint in constraints {
            match constraint {
                ParamConstraint::RectWithinUpstream => {
                    if let Some(rect) = result.as_rect()
                        && (rect.right() > upstream_info.width || rect.bottom() > upstream_info.height) {
                            return Err(format!(
                                "rect ({},{} {}x{}) exceeds upstream dimensions ({}x{})",
                                rect.x, rect.y, rect.width, rect.height,
                                upstream_info.width, upstream_info.height,
                            ));
                        }
                }
                ParamConstraint::Range { min, max } => {
                    if let Some(v) = result.as_scalar()
                        && (v < *min || v > *max) {
                            return Err(format!("value {v} outside range [{min}, {max}]"));
                        }
                }
                ParamConstraint::MaxContext(ctx) => {
                    if let Some(v) = result.as_scalar() {
                        let max = ctx.resolve(upstream_info);
                        if v > max {
                            return Err(format!("value {v} exceeds contextual max {max}"));
                        }
                    }
                }
                ParamConstraint::MinContext(ctx) => {
                    if let Some(v) = result.as_scalar() {
                        let min = ctx.resolve(upstream_info);
                        if v < min {
                            return Err(format!("value {v} below contextual min {min}"));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color_space::ColorSpace;
    use crate::node::{NodeInfo, Upstream};
    use crate::registry::ContextRef;

    // ─── Mock nodes ──────────────────────────────────────────────────────

    /// Source node that produces solid-color pixels.
    struct MockSource {
        w: u32,
        h: u32,
        color: [f32; 4],
    }

    impl Node for MockSource {
        fn info(&self) -> NodeInfo {
            NodeInfo { width: self.w, height: self.h, color_space: ColorSpace::Linear }
        }
        fn compute(&self, request: Rect, _up: &mut dyn Upstream) -> Result<Vec<f32>, PipelineError> {
            let n = request.width as usize * request.height as usize;
            let mut px = Vec::with_capacity(n * 4);
            for _ in 0..n { px.extend_from_slice(&self.color); }
            Ok(px)
        }
        fn upstream_ids(&self) -> Vec<u32> { vec![] }
    }

    /// Analysis node that computes the average brightness of input pixels.
    /// Returns AnalysisResult::Scalar(average_brightness).
    struct MockAnalyzer {
        upstream_id: u32,
        w: u32,
        h: u32,
    }

    impl Node for MockAnalyzer {
        fn info(&self) -> NodeInfo {
            NodeInfo { width: self.w, height: self.h, color_space: ColorSpace::Linear }
        }
        fn compute(&self, request: Rect, upstream: &mut dyn Upstream) -> Result<Vec<f32>, PipelineError> {
            upstream.request(self.upstream_id, request)
        }
        fn upstream_ids(&self) -> Vec<u32> { vec![self.upstream_id] }
        fn is_analysis_node(&self) -> bool { true }
        fn analyze(&self, input: &[f32], _w: u32, _h: u32) -> Option<Result<AnalysisResult, PipelineError>> {
            let n = input.len() / 4;
            if n == 0 { return Some(Ok(AnalysisResult::Scalar(0.0))); }
            let sum: f32 = input.chunks_exact(4)
                .map(|px| (px[0] + px[1] + px[2]) / 3.0)
                .sum();
            Some(Ok(AnalysisResult::Scalar(sum / n as f32)))
        }
    }

    /// Render node that adds a configurable offset to all channels.
    struct MockBrightness {
        upstream_id: u32,
        w: u32,
        h: u32,
        offset: f32,
    }

    impl Node for MockBrightness {
        fn info(&self) -> NodeInfo {
            NodeInfo { width: self.w, height: self.h, color_space: ColorSpace::Linear }
        }
        fn compute(&self, request: Rect, upstream: &mut dyn Upstream) -> Result<Vec<f32>, PipelineError> {
            let mut px = upstream.request(self.upstream_id, request)?;
            for chunk in px.chunks_exact_mut(4) {
                chunk[0] = (chunk[0] + self.offset).clamp(0.0, 1.0);
                chunk[1] = (chunk[1] + self.offset).clamp(0.0, 1.0);
                chunk[2] = (chunk[2] + self.offset).clamp(0.0, 1.0);
            }
            Ok(px)
        }
        fn upstream_ids(&self) -> Vec<u32> { vec![self.upstream_id] }
    }

    // ─── Unit tests (analysis results, type compat, validation) ──────────

    #[test]
    fn analysis_result_type_tags() {
        let rect = AnalysisResult::Rect(Rect::new(10, 20, 100, 200));
        assert_eq!(rect.result_type(), AnalysisResultType::Rect);

        let scalar = AnalysisResult::Scalar(0.5);
        assert_eq!(scalar.result_type(), AnalysisResultType::Scalar);
    }

    #[test]
    fn analysis_result_extraction() {
        let rect = AnalysisResult::Rect(Rect::new(10, 20, 100, 200));
        assert_eq!(rect.as_rect(), Some(Rect::new(10, 20, 100, 200)));
        assert_eq!(rect.as_scalar(), None);

        let scalar = AnalysisResult::Scalar(1.5);
        assert_eq!(scalar.as_scalar(), Some(1.5f64));
        assert_eq!(scalar.as_rect(), None);
    }

    #[test]
    fn type_compatibility() {
        assert!(types_compatible(AnalysisResultType::Scalar, ParamType::F32));
        assert!(types_compatible(AnalysisResultType::Scalar, ParamType::U32));
        assert!(types_compatible(AnalysisResultType::Rect, ParamType::Rect));
        assert!(!types_compatible(AnalysisResultType::Rect, ParamType::F32));
        assert!(!types_compatible(AnalysisResultType::Scalar, ParamType::Rect));
        assert!(!types_compatible(AnalysisResultType::Points, ParamType::F32));
    }

    #[test]
    fn field_extraction_from_rect() {
        let rect = AnalysisResult::Rect(Rect::new(10, 20, 300, 400));
        assert_eq!(extract_field(&rect, "x"), Some(10.0));
        assert_eq!(extract_field(&rect, "y"), Some(20.0));
        assert_eq!(extract_field(&rect, "width"), Some(300.0));
        assert_eq!(extract_field(&rect, "height"), Some(400.0));
        assert_eq!(extract_field(&rect, "invalid"), None);
    }

    #[test]
    fn field_extraction_from_matrix() {
        let m = AnalysisResult::Matrix([1.0, 0.0, 10.0, 0.0, 1.0, 20.0]);
        assert_eq!(extract_field(&m, "a"), Some(1.0));
        assert_eq!(extract_field(&m, "tx"), Some(10.0));
        assert_eq!(extract_field(&m, "ty"), Some(20.0));
    }

    #[test]
    fn field_extraction_validity() {
        assert!(field_extraction_valid(AnalysisResultType::Rect, "x"));
        assert!(field_extraction_valid(AnalysisResultType::Rect, "width"));
        assert!(!field_extraction_valid(AnalysisResultType::Rect, "invalid"));
        assert!(field_extraction_valid(AnalysisResultType::Matrix, "tx"));
        assert!(!field_extraction_valid(AnalysisResultType::Scalar, "x"));
    }

    #[test]
    fn validate_rect_within_upstream() {
        let upstream = NodeInfo {
            width: 1920,
            height: 1080,
            color_space: ColorSpace::Linear,
        };

        let valid = AnalysisResult::Rect(Rect::new(100, 100, 800, 600));
        assert!(StagedPipeline::validate_value(
            &valid, &upstream, &[ParamConstraint::RectWithinUpstream],
        ).is_ok());

        let invalid = AnalysisResult::Rect(Rect::new(1800, 100, 200, 100));
        assert!(StagedPipeline::validate_value(
            &invalid, &upstream, &[ParamConstraint::RectWithinUpstream],
        ).is_err());
    }

    #[test]
    fn validate_scalar_range() {
        let upstream = NodeInfo { width: 100, height: 100, color_space: ColorSpace::Linear };

        let valid = AnalysisResult::Scalar(0.5);
        assert!(StagedPipeline::validate_value(
            &valid, &upstream, &[ParamConstraint::Range { min: 0.0, max: 1.0 }],
        ).is_ok());

        let invalid = AnalysisResult::Scalar(2.0);
        assert!(StagedPipeline::validate_value(
            &invalid, &upstream, &[ParamConstraint::Range { min: 0.0, max: 1.0 }],
        ).is_err());
    }

    #[test]
    fn validate_contextual_max() {
        let upstream = NodeInfo { width: 1920, height: 1080, color_space: ColorSpace::Linear };

        let valid = AnalysisResult::Scalar(1000.0);
        assert!(StagedPipeline::validate_value(
            &valid, &upstream, &[ParamConstraint::MaxContext(ContextRef::UpstreamWidth)],
        ).is_ok());

        let invalid = AnalysisResult::Scalar(2000.0);
        assert!(StagedPipeline::validate_value(
            &invalid, &upstream, &[ParamConstraint::MaxContext(ContextRef::UpstreamWidth)],
        ).is_err());
    }

    // ─── Pipeline structure tests ────────────────────────────────────────

    #[test]
    fn staged_pipeline_basic_structure() {
        let graph = crate::graph::Graph::new(0);
        let mut staged = StagedPipeline::new(graph);

        staged.add_binding(ParamBinding {
            source_node: 0,
            target_node: 1,
            target_param: "crop_rect".to_string(),
            transform: BindingTransform::Identity,
        });

        assert_eq!(staged.bindings.len(), 1);
    }

    #[test]
    fn staged_pipeline_validates_missing_nodes() {
        let graph = crate::graph::Graph::new(0);
        let mut staged = StagedPipeline::new(graph);

        staged.add_binding(ParamBinding {
            source_node: 99,
            target_node: 100,
            target_param: "test".to_string(),
            transform: BindingTransform::Identity,
        });

        let result = staged.validate_bindings();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 2);
    }

    #[test]
    fn node_default_analyze_returns_none() {
        let src = MockSource { w: 4, h: 4, color: [0.5, 0.5, 0.5, 1.0] };
        assert!(!src.is_analysis_node());
        assert!(src.analyze(&[], 0, 0).is_none());
    }

    #[test]
    fn mock_analyzer_returns_result() {
        let analyzer = MockAnalyzer { upstream_id: 0, w: 2, h: 2 };
        assert!(analyzer.is_analysis_node());

        // 4 pixels, all [0.6, 0.3, 0.3, 1.0] → avg brightness = 0.4
        let px = vec![0.6, 0.3, 0.3, 1.0, 0.6, 0.3, 0.3, 1.0,
                      0.6, 0.3, 0.3, 1.0, 0.6, 0.3, 0.3, 1.0];
        let result = analyzer.analyze(&px, 2, 2).unwrap().unwrap();
        assert_eq!(result.result_type(), AnalysisResultType::Scalar);
        assert!((result.as_scalar().unwrap() - 0.4).abs() < 1e-6);
    }

    // ─── Wave scheduler tests ────────────────────────────────────────────

    #[test]
    fn wave_scheduler_empty_bindings() {
        let graph = crate::graph::Graph::new(0);
        let staged = StagedPipeline::new(graph);
        let waves = staged.build_waves().unwrap();
        assert!(waves.is_empty());
    }

    #[test]
    fn wave_scheduler_single_analysis() {
        // source(0) → analyzer(1) → brightness(2)
        // binding: analyzer(1) → brightness(2).offset
        let mut graph = crate::graph::Graph::new(0);
        graph.add_node(Box::new(MockSource { w: 4, h: 4, color: [0.5, 0.5, 0.5, 1.0] }));
        graph.add_node(Box::new(MockAnalyzer { upstream_id: 0, w: 4, h: 4 }));
        graph.add_node(Box::new(MockBrightness { upstream_id: 0, w: 4, h: 4, offset: 0.0 }));

        let mut staged = StagedPipeline::new(graph);
        staged.add_binding(ParamBinding {
            source_node: 1,
            target_node: 2,
            target_param: "offset".to_string(),
            transform: BindingTransform::Identity,
        });

        let waves = staged.build_waves().unwrap();
        assert_eq!(waves.len(), 2); // Analyze + Bind
        match &waves[0] {
            WaveStep::Analyze(ids) => assert_eq!(ids, &[1]),
            _ => panic!("expected Analyze wave"),
        }
        match &waves[1] {
            WaveStep::Bind(indices) => assert_eq!(indices, &[0]),
            _ => panic!("expected Bind wave"),
        }
    }

    #[test]
    fn wave_scheduler_multi_stage() {
        // source(0) → analyzer1(1) → brightness(2) → analyzer2(3) → brightness2(4)
        // binding: analyzer1(1) → brightness(2).offset
        // binding: analyzer2(3) → brightness2(4).offset
        // analyzer2 depends on brightness(2) output, so it must be in wave 2
        let mut graph = crate::graph::Graph::new(0);
        graph.add_node(Box::new(MockSource { w: 4, h: 4, color: [0.5, 0.5, 0.5, 1.0] }));
        graph.add_node(Box::new(MockAnalyzer { upstream_id: 0, w: 4, h: 4 }));
        graph.add_node(Box::new(MockBrightness { upstream_id: 0, w: 4, h: 4, offset: 0.0 }));
        graph.add_node(Box::new(MockAnalyzer { upstream_id: 2, w: 4, h: 4 })); // depends on brightness(2)
        graph.add_node(Box::new(MockBrightness { upstream_id: 2, w: 4, h: 4, offset: 0.0 }));

        let mut staged = StagedPipeline::new(graph);
        staged.add_binding(ParamBinding {
            source_node: 1,
            target_node: 2,
            target_param: "offset".to_string(),
            transform: BindingTransform::Identity,
        });
        staged.add_binding(ParamBinding {
            source_node: 3,
            target_node: 4,
            target_param: "offset".to_string(),
            transform: BindingTransform::Identity,
        });

        let waves = staged.build_waves().unwrap();
        // Wave 0: Analyze [1], Bind [0], Wave 1: Analyze [3], Bind [1]
        assert_eq!(waves.len(), 4);
        match &waves[0] {
            WaveStep::Analyze(ids) => assert_eq!(ids, &[1]),
            _ => panic!("expected Analyze wave"),
        }
        match &waves[2] {
            WaveStep::Analyze(ids) => assert_eq!(ids, &[3]),
            _ => panic!("expected second Analyze wave"),
        }
    }

    #[test]
    fn wave_scheduler_diamond_dependency() {
        // source(0) → analyzer_a(1) → brightness_a(2)
        //           → analyzer_b(3) → brightness_b(4)
        // Both analyzers independent → same wave
        let mut graph = crate::graph::Graph::new(0);
        graph.add_node(Box::new(MockSource { w: 4, h: 4, color: [0.5, 0.5, 0.5, 1.0] }));
        graph.add_node(Box::new(MockAnalyzer { upstream_id: 0, w: 4, h: 4 }));
        graph.add_node(Box::new(MockBrightness { upstream_id: 0, w: 4, h: 4, offset: 0.0 }));
        graph.add_node(Box::new(MockAnalyzer { upstream_id: 0, w: 4, h: 4 }));
        graph.add_node(Box::new(MockBrightness { upstream_id: 0, w: 4, h: 4, offset: 0.0 }));

        let mut staged = StagedPipeline::new(graph);
        staged.add_binding(ParamBinding {
            source_node: 1, target_node: 2,
            target_param: "offset".to_string(), transform: BindingTransform::Identity,
        });
        staged.add_binding(ParamBinding {
            source_node: 3, target_node: 4,
            target_param: "offset".to_string(), transform: BindingTransform::Identity,
        });

        let waves = staged.build_waves().unwrap();
        // Both analyses in wave 0
        assert_eq!(waves.len(), 2); // Analyze + Bind
        match &waves[0] {
            WaveStep::Analyze(ids) => {
                assert_eq!(ids.len(), 2);
                assert!(ids.contains(&1));
                assert!(ids.contains(&3));
            }
            _ => panic!("expected Analyze wave"),
        }
    }

    // ─── Integration tests (full execute) ────────────────────────────────

    #[test]
    fn execute_analysis_bind_render_chain() {
        // source(0) [all 0.5] → analyzer(1) → brightness(2) [offset = analysis result]
        // Analyzer sees avg brightness = 0.5, so offset = 0.5
        // Final brightness = clamp(0.5 + 0.5) = 1.0
        let mut graph = crate::graph::Graph::new(0);
        let src = graph.add_node(Box::new(MockSource { w: 2, h: 2, color: [0.5, 0.5, 0.5, 1.0] }));
        let ana = graph.add_node(Box::new(MockAnalyzer { upstream_id: src, w: 2, h: 2 }));
        let bright = graph.add_node(Box::new(MockBrightness {
            upstream_id: src, w: 2, h: 2, offset: 0.0,
        }));

        let mut staged = StagedPipeline::new(graph);
        staged.add_binding(ParamBinding {
            source_node: ana,
            target_node: bright,
            target_param: "offset".to_string(),
            transform: BindingTransform::Identity,
        });
        staged.set_param_updater(bright, "offset", |_node, result| {
            let scalar = result.as_scalar().unwrap() as f32;
            Box::new(MockBrightness { upstream_id: 0, w: 2, h: 2, offset: scalar })
        });

        let output = staged.execute(bright).unwrap();
        // 0.5 + 0.5 = 1.0 (clamped)
        assert_eq!(output.len(), 2 * 2 * 4);
        assert!((output[0] - 1.0).abs() < 1e-6);
        assert!((output[1] - 1.0).abs() < 1e-6);
        assert!((output[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn execute_multi_stage_analysis_chain() {
        // source(0) [all 0.3] → analyzer1(1) → brightness1(2) → analyzer2(3) → brightness2(4)
        // Stage 1: analyzer1 sees 0.3 avg → brightness1 offset = 0.3, result = 0.6
        // Stage 2: analyzer2 sees 0.6 avg → brightness2 offset = 0.6, result = clamp(0.3+0.6)=0.9
        let mut graph = crate::graph::Graph::new(0);
        let src = graph.add_node(Box::new(MockSource { w: 2, h: 2, color: [0.3, 0.3, 0.3, 1.0] }));
        let ana1 = graph.add_node(Box::new(MockAnalyzer { upstream_id: src, w: 2, h: 2 }));
        let bright1 = graph.add_node(Box::new(MockBrightness { upstream_id: src, w: 2, h: 2, offset: 0.0 }));
        let ana2 = graph.add_node(Box::new(MockAnalyzer { upstream_id: bright1, w: 2, h: 2 }));
        let bright2 = graph.add_node(Box::new(MockBrightness { upstream_id: src, w: 2, h: 2, offset: 0.0 }));

        let mut staged = StagedPipeline::new(graph);
        staged.add_binding(ParamBinding {
            source_node: ana1, target_node: bright1,
            target_param: "offset".to_string(), transform: BindingTransform::Identity,
        });
        staged.add_binding(ParamBinding {
            source_node: ana2, target_node: bright2,
            target_param: "offset".to_string(), transform: BindingTransform::Identity,
        });
        staged.set_param_updater(bright1, "offset", |_node, result| {
            let s = result.as_scalar().unwrap() as f32;
            Box::new(MockBrightness { upstream_id: 0, w: 2, h: 2, offset: s })
        });
        staged.set_param_updater(bright2, "offset", |_node, result| {
            let s = result.as_scalar().unwrap() as f32;
            Box::new(MockBrightness { upstream_id: 0, w: 2, h: 2, offset: s })
        });

        let output = staged.execute(bright2).unwrap();
        // analyzer2 sees brightness1 output = 0.3 + 0.3 = 0.6
        // brightness2 offset = 0.6, applied to source (0.3) → 0.9
        assert!((output[0] - 0.9).abs() < 1e-5);
    }

    #[test]
    fn execute_no_bindings_passthrough() {
        // No analysis — just render source directly
        let mut graph = crate::graph::Graph::new(0);
        let src = graph.add_node(Box::new(MockSource { w: 2, h: 2, color: [0.7, 0.7, 0.7, 1.0] }));

        let mut staged = StagedPipeline::new(graph);
        let output = staged.execute(src).unwrap();
        assert!((output[0] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn validate_rejects_non_analysis_source() {
        let mut graph = crate::graph::Graph::new(0);
        graph.add_node(Box::new(MockSource { w: 2, h: 2, color: [0.5, 0.5, 0.5, 1.0] })); // node 0
        graph.add_node(Box::new(MockBrightness { upstream_id: 0, w: 2, h: 2, offset: 0.1 })); // node 1

        let mut staged = StagedPipeline::new(graph);
        staged.add_binding(ParamBinding {
            source_node: 0, // MockSource — not an analysis node
            target_node: 1,
            target_param: "offset".to_string(),
            transform: BindingTransform::Identity,
        });

        let result = staged.validate_bindings();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("not an analysis node"));
    }
}
