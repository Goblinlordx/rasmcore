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

// Allow the V2Filter derive macro to reference `rasmcore_pipeline_v2::` from within this crate
extern crate self as rasmcore_pipeline_v2;

pub mod rect;
pub mod node;
pub mod color_space;
pub mod gpu;
pub mod graph;
pub mod demand;
pub mod cache;
pub mod hash;
pub mod layer_cache;
pub mod registry;
pub mod ops;
#[macro_use]
pub mod gpu_macros;
pub mod filter_node;
pub mod ml_node;
pub mod color_math;
pub mod color_convert;
pub mod aces;
pub mod fusion;
pub mod trace;
pub mod staged;
pub mod analysis_buffer;
pub mod gpu_shaders;
pub mod aces_audit;
pub mod noise;
pub mod lmt;
pub mod brush;
pub mod font;
pub mod undo;
pub mod filters;

// Re-export core types at crate root
pub use rect::{Rect, Overlap, tiles, extract_tile, place_tile};
pub use node::{Node, NodeInfo, NodeCapabilities, GpuShader, TileHint, PipelineError, Upstream, InputRectEstimate, AcesCompliance};
pub use aces_audit::{AcesViolation, AcesAuditResult};
pub use color_space::ColorSpace;
pub use gpu::{GpuError, GpuExecutor};
pub use graph::{Graph, GpuPlan, MultiGpuPlan, GpuStage, StageInput, BufferPool};
pub use demand::{DemandStrategy, DemandHint};
pub use cache::{SpatialCache, SpatialCachePool};
pub use hash::{content_hash, source_hash};
pub use layer_cache::{LayerCache, CacheQuality, CacheStats};

// Re-export registration system
pub use registry::{
    OperationRegistration, OperationKind, OperationCapabilities,
    ParamDescriptor, ParamType, ParamConstraint, ContextRef,
    registered_operations, find_operation, operations_by_category,
    operations_by_kind, param_descriptors,
    ParamMap, TypedRef, FilterFactory, FilterFactoryRegistration,
    create_filter_node, registered_filter_factories,
    EncoderFactory, EncoderFactoryRegistration,
    encode_via_registry, registered_encoders,
    DecoderFactory, DecoderFactoryRegistration, DecodedImageV2,
    decode_via_registry, decode_with_hint_via_registry, registered_decoders,
};

// Re-export operation traits
#[allow(deprecated)]
pub use ops::{
    Filter, GpuFilter, Decoder, Encoder, Transform,
    AnalyticOp, PointOpExpr, DecodedImage,
};

// Re-export node wrappers
#[allow(deprecated)]
pub use filter_node::{FilterNode, GpuFilterNode, compose_shader, IO_F32};

// Re-export LMT types
pub use lmt::{Lmt, LmtNode, parse_cube, analytical_uniform, analytical_cdl};

// Re-export color pipeline
pub use color_math::{
    srgb_to_linear, linear_to_srgb,
    linear_to_acescct, acescct_to_linear,
    linear_to_acescc, acescc_to_linear,
    convert_color_space, apply_transfer, apply_matrix, mat3_mul,
};
pub use color_convert::{ColorConvertNode, ViewTransformNode, ViewTransform};
pub use trace::{PipelineTrace, TraceEvent, TraceEventKind};

// Re-export analysis buffer protocol
pub use analysis_buffer::{
    AnalysisBufferKind, AnalysisBufferDecl, AnalysisBufferRef,
    AnalysisBufferContext, NodeBufferMapping,
    negotiate_analysis_buffers, ChainNodeInfo,
};
