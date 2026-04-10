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

pub mod cache;
pub mod color_space;
pub mod demand;
pub mod gpu;
pub mod graph;
pub mod hash;
pub mod layer_cache;
pub mod node;
pub mod ops;
pub mod rect;
pub mod registry;
#[macro_use]
pub mod gpu_macros;
pub mod aces;
pub mod aces2;
pub mod aces_audit;
pub mod analysis_buffer;
pub mod brush;
pub mod camera_idt;
pub mod cdl;
pub mod color_convert;
pub mod color_math;
pub mod color_transform;
pub mod filter_node;
pub mod filters;
pub mod font;
pub mod fusion;
pub mod gpu_shaders;
pub mod image_metadata;
pub mod lmt;
pub mod ml_node;
pub mod noise;
pub mod staged;
pub mod trace;
pub mod undo;

// Re-export core types at crate root
pub use aces_audit::{AcesAuditResult, AcesViolation};
pub use cache::{SpatialCache, SpatialCachePool};
pub use color_space::ColorSpace;
pub use demand::{DemandHint, DemandStrategy};
pub use gpu::{GpuError, GpuExecutor};
pub use graph::{BufferPool, GpuPlan, GpuStage, Graph, MultiGpuPlan, StageInput};
pub use hash::{content_hash, source_hash};
pub use layer_cache::{CacheQuality, CacheStats, LayerCache};
pub use node::{
    AcesCompliance, GpuSetup, GpuShader, InputRectEstimate, Node, NodeCapabilities, NodeInfo,
    PipelineError, TileHint, Upstream,
};
pub use rect::{Overlap, Rect, extract_tile, place_tile, tiles};

// Re-export registration system
pub use registry::{
    ContextRef, DecodedImageV2, DecoderFactory, DecoderFactoryRegistration, EncoderFactory,
    EncoderFactoryRegistration, FilterFactory, FilterFactoryRegistration, OperationCapabilities,
    OperationKind, OperationRegistration, ParamConstraint, ParamDescriptor, ParamMap, ParamType,
    TypedRef, create_filter_node, decode_via_registry, decode_with_hint_via_registry,
    encode_via_registry, find_operation, is_scene_referred_format, operations_by_category,
    operations_by_kind, param_descriptors, registered_decoders, registered_encoders,
    registered_filter_factories, registered_operations,
};

// Re-export operation traits
#[allow(deprecated)]
pub use ops::{
    AnalyticOp, DecodedImage, Decoder, Encoder, Filter, GpuFilter, PointOpExpr, Transform,
};

// Re-export node wrappers
#[allow(deprecated)]
pub use filter_node::{FilterNode, GpuFilterNode, IO_F32, compose_shader};

// Re-export LMT types
pub use lmt::{Lmt, LmtNode, analytical_cdl, analytical_uniform, parse_cube};

// Re-export color pipeline
pub use color_convert::{ColorConvertNode, ViewTransform, ViewTransformNode};
pub use color_math::{
    acescc_to_linear, acescct_to_linear, apply_matrix, apply_transfer, convert_color_space,
    linear_srgb_to_oklab, linear_to_acescc, linear_to_acescct, linear_to_srgb, mat3_mul,
    oklab_to_linear_srgb, oklab_to_oklch, oklch_to_oklab, srgb_to_linear,
};
pub use trace::{PipelineTrace, TraceEvent, TraceEventKind};

// Re-export analysis buffer protocol
pub use analysis_buffer::{
    AnalysisBufferContext, AnalysisBufferDecl, AnalysisBufferKind, AnalysisBufferRef,
    ChainNodeInfo, NodeBufferMapping, negotiate_analysis_buffers,
};
