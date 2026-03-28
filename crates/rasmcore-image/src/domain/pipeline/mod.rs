//! Demand-driven tile pipeline engine.
//!
//! The pipeline is a graph of image nodes, each producing pixel regions on demand.
//! `write()` drives execution by pulling regions backward through the graph.
//! A spatial cache with ref-counted borrowing ensures overlap regions are
//! computed once and reused.
//!
//! Design reference: .agent/kf/_reports/pipeline-architecture.md

pub mod cache;
pub mod graph;
pub mod nodes;
pub mod rect;
mod tests;
