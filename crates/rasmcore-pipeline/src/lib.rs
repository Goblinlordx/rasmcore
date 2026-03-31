//! Shared pipeline engine primitives for rasmcore components.
//!
//! Provides the spatial cache, rectangle geometry, and overlap types used by
//! both image and video processing pipelines. Domain-specific node traits
//! (ImageNode, VideoNode) are defined in their respective crates.

pub mod cache;
pub mod layer_cache;
pub mod metadata;
pub mod rect;

pub use cache::{CacheQuery, RegionHandle, RegionKey, SpatialCache};
pub use layer_cache::{
    compute_hash, compute_source_hash, CacheStats, ContentHash, LayerCache, ZERO_HASH,
};
pub use metadata::{glob_match, Metadata, MetadataFilter, MetadataValue};
pub use rect::{Overlap, Rect};
