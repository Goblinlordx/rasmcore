//! Filter nodes — auto-generated from #[register_filter] annotations.
//!
//! Each registered filter gets a typed node struct with named fields,
//! a constructor, and an ImageNode implementation that handles tiled
//! execution with overlap expansion and cropping.

include!(concat!(env!("OUT_DIR"), "/generated_pipeline_nodes.rs"));
