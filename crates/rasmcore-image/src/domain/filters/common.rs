//! Shared helpers re-exported for use by individual filter files.
//! 
//! Functions are defined in the parent filters/mod.rs and made pub there.
//! Individual filter files import via `use crate::domain::filters::common::*;`
//! which is equivalent to `use super::super::*;` from within a category.
//!
//! This module exists to provide a stable import path as filters are
//! incrementally moved from mod.rs to their own files.

// Re-export everything from the parent module
pub use super::*;
