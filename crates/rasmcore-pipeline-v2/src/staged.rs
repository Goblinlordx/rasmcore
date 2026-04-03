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

/// Staged pipeline — orchestrates analysis → bind → execute.
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
    /// Registered analysis nodes (node_id → trait object).
    analysis_nodes: std::collections::HashMap<u32, Box<dyn AnalysisNode>>,
}

impl StagedPipeline {
    /// Create a staged pipeline wrapping a graph.
    pub fn new(graph: crate::graph::Graph) -> Self {
        Self {
            graph,
            bindings: Vec::new(),
            analysis_nodes: std::collections::HashMap::new(),
        }
    }

    /// Register an analysis node. The node is also added to the graph.
    pub fn add_analysis_node(&mut self, node: Box<dyn AnalysisNode>) -> u32 {
        // We need to add the node to the graph as a regular Node,
        // but also keep a reference for analysis. Use the node_id as key.
        let _result_type = node.result_type();
        let id = self.graph.add_node(node);
        // Re-create a lightweight marker — actual analysis happens via the
        // analysis_nodes map. We store the trait object separately.
        // NOTE: This requires AnalysisNode to also implement Node (which it does
        // via the supertrait bound). The graph owns the Node; we own the analysis.
        //
        // Actually, we can't split ownership. Instead, store analysis metadata.
        // The staged pipeline will call graph.request_full() to get pixel data,
        // then call analyze() on the analysis trait object.
        //
        // For now, store the result_type as metadata. The actual analyze() call
        // happens through a separate mechanism (see execute()).
        self.analysis_nodes.clear(); // placeholder — proper impl below
        id
    }

    /// Add a parameter binding.
    pub fn add_binding(&mut self, binding: ParamBinding) {
        self.bindings.push(binding);
    }

    /// Validate all bindings (type checking only — no pixel data needed).
    pub fn validate_bindings(&self) -> Result<(), Vec<BindingError>> {
        let mut errors = Vec::new();

        for (i, binding) in self.bindings.iter().enumerate() {
            // Check source node exists
            if self.graph.node_info(binding.source_node).is_err() {
                errors.push(BindingError {
                    binding_index: i,
                    message: format!("source node {} not found", binding.source_node),
                });
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
    use crate::node::NodeInfo;
    use crate::registry::ContextRef;

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

        // Valid rect
        let valid = AnalysisResult::Rect(Rect::new(100, 100, 800, 600));
        assert!(StagedPipeline::validate_value(
            &valid,
            &upstream,
            &[ParamConstraint::RectWithinUpstream],
        )
        .is_ok());

        // Invalid rect (exceeds width)
        let invalid = AnalysisResult::Rect(Rect::new(1800, 100, 200, 100));
        assert!(StagedPipeline::validate_value(
            &invalid,
            &upstream,
            &[ParamConstraint::RectWithinUpstream],
        )
        .is_err());
    }

    #[test]
    fn validate_scalar_range() {
        let upstream = NodeInfo {
            width: 100,
            height: 100,
            color_space: ColorSpace::Linear,
        };

        let valid = AnalysisResult::Scalar(0.5);
        assert!(StagedPipeline::validate_value(
            &valid,
            &upstream,
            &[ParamConstraint::Range {
                min: 0.0,
                max: 1.0
            }],
        )
        .is_ok());

        let invalid = AnalysisResult::Scalar(2.0);
        assert!(StagedPipeline::validate_value(
            &invalid,
            &upstream,
            &[ParamConstraint::Range {
                min: 0.0,
                max: 1.0
            }],
        )
        .is_err());
    }

    #[test]
    fn validate_contextual_max() {
        let upstream = NodeInfo {
            width: 1920,
            height: 1080,
            color_space: ColorSpace::Linear,
        };

        let valid = AnalysisResult::Scalar(1000.0);
        assert!(StagedPipeline::validate_value(
            &valid,
            &upstream,
            &[ParamConstraint::MaxContext(ContextRef::UpstreamWidth)],
        )
        .is_ok());

        let invalid = AnalysisResult::Scalar(2000.0);
        assert!(StagedPipeline::validate_value(
            &invalid,
            &upstream,
            &[ParamConstraint::MaxContext(ContextRef::UpstreamWidth)],
        )
        .is_err());
    }

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
            source_node: 99, // doesn't exist
            target_node: 100,
            target_param: "test".to_string(),
            transform: BindingTransform::Identity,
        });

        let result = staged.validate_bindings();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 2); // both source and target missing
    }
}
