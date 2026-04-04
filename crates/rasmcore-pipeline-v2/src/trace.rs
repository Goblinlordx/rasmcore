//! Pipeline tracing — opt-in diagnostics for fusion, GPU dispatch, and execution.
//!
//! When tracing is enabled on a `Graph`, each stage records a `TraceEvent`
//! with name, duration, and optional detail. After execution, call
//! `graph.take_trace()` to retrieve and clear the collected events.

use std::time::Instant;

/// A single pipeline trace event.
#[derive(Debug, Clone)]
pub struct TraceEvent {
    /// Category of the event.
    pub kind: TraceEventKind,
    /// Human-readable label (e.g., filter name, shader ID).
    pub name: String,
    /// Duration in microseconds.
    pub duration_us: u64,
    /// Optional detail (e.g., shader source hash, node count).
    pub detail: Option<String>,
}

/// Category of trace events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceEventKind {
    /// Graph fusion optimization pass.
    Fusion,
    /// GPU shader compilation.
    ShaderCompile,
    /// GPU compute dispatch (successful).
    GpuDispatch,
    /// CPU fallback execution.
    CpuFallback,
    /// Output encoding.
    Encode,
}

impl std::fmt::Display for TraceEventKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TraceEventKind::Fusion => write!(f, "fusion"),
            TraceEventKind::ShaderCompile => write!(f, "shader_compile"),
            TraceEventKind::GpuDispatch => write!(f, "gpu_dispatch"),
            TraceEventKind::CpuFallback => write!(f, "cpu_fallback"),
            TraceEventKind::Encode => write!(f, "encode"),
        }
    }
}

/// Collected pipeline trace data.
#[derive(Debug, Clone, Default)]
pub struct PipelineTrace {
    pub events: Vec<TraceEvent>,
}

impl PipelineTrace {
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Record an event.
    pub fn push(&mut self, event: TraceEvent) {
        self.events.push(event);
    }

    /// Total duration across all events (microseconds).
    pub fn total_us(&self) -> u64 {
        self.events.iter().map(|e| e.duration_us).sum()
    }

    /// Events filtered by kind.
    pub fn by_kind(&self, kind: TraceEventKind) -> Vec<&TraceEvent> {
        self.events.iter().filter(|e| e.kind == kind).collect()
    }
}

/// RAII timer that records a TraceEvent on drop.
///
/// Used internally by graph/fusion code when tracing is enabled.
pub(crate) struct TraceTimer {
    kind: TraceEventKind,
    name: String,
    detail: Option<String>,
    start: Instant,
}

impl TraceTimer {
    pub fn new(kind: TraceEventKind, name: impl Into<String>) -> Self {
        Self {
            kind,
            name: name.into(),
            detail: None,
            start: Instant::now(),
        }
    }

    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }

    /// Finish timing and return the event.
    pub fn finish(self) -> TraceEvent {
        let elapsed = self.start.elapsed();
        TraceEvent {
            kind: self.kind,
            name: self.name,
            duration_us: elapsed.as_micros() as u64,
            detail: self.detail,
        }
    }
}
