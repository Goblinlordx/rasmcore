//! GpuCapable implementations for auto-generated filter pipeline nodes.
//!
//! All spatial and distortion filter GPU impls have been migrated to GpuFilter
//! trait impls on their respective config structs (derive(Filter) pattern).
//! The codegen automatically generates GpuCapable bridge impls for derive-style
//! filters that implement GpuFilter.
//!
//! This file is retained for any future non-derive GPU filter impls.
