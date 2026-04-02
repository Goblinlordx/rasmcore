//! Pipeline node implementations.
//!
//! Each node wraps an existing domain operation and implements the ImageNode trait.

pub mod color;
pub mod composite;
pub mod filters;
pub mod frame_source;
pub mod gpu_impls;
pub mod sink;
pub mod source;
pub mod subgraph;
pub mod transform;
