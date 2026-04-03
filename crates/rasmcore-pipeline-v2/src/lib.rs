//! V2 Pipeline Engine — f32-native, GPU-primary, zero built-in knowledge.
//!
//! The pipeline engine manages graph topology, demand-driven tile execution,
//! GPU dispatch, and caching. It knows NOTHING about specific filters, codecs,
//! color spaces, or operations. Everything is a `Node` trait object registered
//! externally.
//!
//! # Core Principles
//!
//! 1. **f32-native**: All pixel data is `&[f32]` (RGBA, 4 channels). No format
//!    dispatch, no PixelFormat enum, no u8/u16 code paths.
//! 2. **GPU-primary**: GPU dispatch is checked first. CPU is the fallback.
//! 3. **Zero built-in knowledge**: The engine doesn't know what a "blur" or
//!    "JPEG" is. Nodes register capabilities; the engine dispatches.
//! 4. **Demand-driven**: Tiles are pulled from sinks through the graph. Each
//!    node computes only what's requested.

pub mod rect;
pub mod node;
pub mod color_space;
pub mod gpu;
pub mod graph;
pub mod demand;
pub mod cache;
pub mod hash;
pub mod registry;
pub mod ops;
pub mod filter_node;
pub mod color_math;
pub mod color_convert;

// Re-export core types at crate root
pub use rect::{Rect, Overlap};
pub use node::{Node, NodeInfo, NodeCapabilities, GpuShader, TileHint, PipelineError, Upstream, InputRectEstimate};
pub use color_space::ColorSpace;
pub use gpu::{GpuError, GpuExecutor};
pub use graph::Graph;
pub use demand::{DemandStrategy, DemandHint};
pub use cache::SpatialCache;
pub use hash::content_hash;

// Re-export registration system
pub use registry::{
    OperationRegistration, OperationKind, OperationCapabilities,
    ParamDescriptor, ParamType, ParamConstraint, ContextRef,
    registered_operations, find_operation, operations_by_category,
    operations_by_kind, param_descriptors,
};

// Re-export operation traits
pub use ops::{
    Filter, GpuFilter, Decoder, Encoder, Transform,
    AnalyticOp, PointOpExpr, DecodedImage,
};

// Re-export node wrappers
pub use filter_node::{FilterNode, GpuFilterNode, compose_shader, IO_F32};

// Re-export color pipeline
pub use color_math::{
    srgb_to_linear, linear_to_srgb,
    linear_to_acescct, acescct_to_linear,
    linear_to_acescc, acescc_to_linear,
    convert_color_space, apply_transfer, apply_matrix, mat3_mul,
};
pub use color_convert::{ColorConvertNode, ViewTransformNode, ViewTransform};
