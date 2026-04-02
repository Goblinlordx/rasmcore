//! Shared pipeline engine primitives for rasmcore components.
//!
//! Provides the spatial cache, rectangle geometry, and overlap types used by
//! both image and video processing pipelines. Domain-specific node traits
//! (ImageNode, VideoNode) are defined in their respective crates.

pub mod cache;
pub mod gpu;
pub mod layer_cache;
pub mod metadata;
pub mod ml;
pub mod rect;

pub use cache::{CacheQuery, RegionHandle, RegionKey, SpatialCache};
pub use layer_cache::{
    CacheQuality, CacheStats, ContentHash, LayerCache, ZERO_HASH, compute_hash,
    compute_source_hash,
};
pub use metadata::{Metadata, MetadataFilter, MetadataValue, glob_match};
pub use rect::{Overlap, Rect};
pub use gpu::{BufferFormat, GpuCapable, GpuConfig, GpuError, GpuExecutor, GpuOp};
pub use ml::{MlCapable, MlCapabilityInfo, MlError, MlExecutor, MlOp, ModelRef, TensorDesc, TensorDtype};
