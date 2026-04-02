//! Sub-graph (composite) nodes — pipeline-as-node with input/output refs.
//!
//! A `SubGraphNode` wraps an internal `NodeGraph` and presents it as a single
//! `ImageNode` to the outer graph. Input pixels flow through a shared buffer
//! from the outer upstream into the internal `InputRefNode`.
//!
//! This enables:
//! - Pipeline composition (chain multiple pipelines as one node)
//! - Analysis sinks (consume pixels, output params, feed into processing)
//! - Presets as composites (serializable graph definitions)
//! - Nested composites (recursive sub-graphs)

use std::cell::RefCell;
use std::rc::Rc;

use crate::domain::error::ImageError;
use crate::domain::pipeline::graph::{AccessPattern, ImageNode, NodeGraph};
use crate::domain::types::*;
use rasmcore_pipeline::Rect;

/// Shared pixel buffer between SubGraphNode and its InputRefNode.
/// SubGraphNode writes upstream pixels here; InputRefNode reads from it.
pub type SharedBuffer = Rc<RefCell<Vec<u8>>>;

// ─── InputRefNode ────────────────────────────────────────────────────────────

/// Placeholder source node inside a sub-graph that receives pixels from the
/// outer graph via a shared buffer.
pub struct InputRefNode {
    info: ImageInfo,
    buffer: SharedBuffer,
}

impl InputRefNode {
    /// Create a new input ref node with a shared buffer.
    /// The caller retains a clone of the `SharedBuffer` to inject pixels.
    pub fn new(info: ImageInfo, buffer: SharedBuffer) -> Self {
        Self { info, buffer }
    }
}

impl ImageNode for InputRefNode {
    fn info(&self) -> ImageInfo {
        self.info.clone()
    }

    fn compute_region(
        &self,
        request: Rect,
        _upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        let buf = self.buffer.borrow();
        if buf.is_empty() {
            return Err(ImageError::InvalidInput(
                "InputRefNode: no pixels injected".into(),
            ));
        }
        let full = Rect::new(0, 0, self.info.width, self.info.height);
        if request == full {
            return Ok(buf.clone());
        }
        // Crop from the full buffer
        let bpp = bytes_per_pixel(self.info.format) as usize;
        let stride = self.info.width as usize * bpp;
        let mut out = Vec::with_capacity(request.width as usize * request.height as usize * bpp);
        for row in request.y..(request.y + request.height) {
            let start = row as usize * stride + request.x as usize * bpp;
            let end = start + request.width as usize * bpp;
            if end <= buf.len() {
                out.extend_from_slice(&buf[start..end]);
            }
        }
        Ok(out)
    }

    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::Sequential
    }
}

// ─── OutputRefNode ───────────────────────────────────────────────────────────

/// Pass-through node marking the output of an internal sub-graph.
/// Simply forwards pixels from its upstream unchanged.
pub struct OutputRefNode {
    upstream: u32,
    info: ImageInfo,
}

impl OutputRefNode {
    pub fn new(upstream: u32, info: ImageInfo) -> Self {
        Self { upstream, info }
    }
}

impl ImageNode for OutputRefNode {
    fn info(&self) -> ImageInfo {
        self.info.clone()
    }

    fn compute_region(
        &self,
        request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        upstream_fn(self.upstream, request)
    }

    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> Rect {
        output.clamp(bounds_w, bounds_h)
    }

    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::Sequential
    }

    fn upstream_id(&self) -> Option<u32> {
        Some(self.upstream)
    }
}

// ─── SubGraphNode ────────────────────────────────────────────────────────────

/// A node whose implementation is an internal pipeline graph.
///
/// The outer graph treats this as an opaque `ImageNode`. Internally, it:
/// 1. Fetches upstream pixels from the outer graph
/// 2. Writes them to the shared buffer (read by InputRefNode)
/// 3. Executes the internal graph from the output ref node
/// 4. Returns the result
pub struct SubGraphNode {
    /// Node ID in the outer graph providing input pixels.
    outer_upstream: u32,
    /// The internal pipeline graph (RefCell for interior mutability —
    /// request_region takes &mut, but ImageNode::compute_region takes &self).
    internal_graph: RefCell<NodeGraph>,
    /// Shared buffer: SubGraphNode writes, InputRefNode reads.
    shared_input: SharedBuffer,
    /// Node ID of the output ref in the internal graph.
    output_ref_id: u32,
    /// Output image info (from the output ref node).
    output_info: ImageInfo,
    /// Input image info (for requesting upstream pixels).
    input_info: ImageInfo,
}

impl SubGraphNode {
    /// Create a sub-graph node.
    ///
    /// - `outer_upstream`: node ID in the outer graph to pull input from
    /// - `internal_graph`: the internal pipeline (ownership transferred)
    /// - `shared_input`: the shared buffer wired to the InputRefNode
    /// - `output_ref_id`: node ID of the OutputRefNode in the internal graph
    pub fn new(
        outer_upstream: u32,
        internal_graph: NodeGraph,
        shared_input: SharedBuffer,
        output_ref_id: u32,
    ) -> Result<Self, ImageError> {
        let output_info = internal_graph.node_info(output_ref_id)?;
        // Input info: derived from the InputRefNode (node 0 by convention)
        let input_info = internal_graph.node_info(0)?;
        Ok(Self {
            outer_upstream,
            internal_graph: RefCell::new(internal_graph),
            shared_input,
            output_ref_id,
            output_info,
            input_info,
        })
    }
}

impl ImageNode for SubGraphNode {
    fn info(&self) -> ImageInfo {
        self.output_info.clone()
    }

    fn compute_region(
        &self,
        request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        // 1. Fetch full input from the outer graph's upstream
        let input_rect = Rect::new(0, 0, self.input_info.width, self.input_info.height);
        let input_pixels = upstream_fn(self.outer_upstream, input_rect)?;

        // 2. Write into the shared buffer (InputRefNode will read from here)
        *self.shared_input.borrow_mut() = input_pixels;

        // 3. Execute the internal graph from the output ref node
        self.internal_graph
            .borrow_mut()
            .request_region(self.output_ref_id, request)
    }

    fn input_rect(&self, _output: Rect, bounds_w: u32, bounds_h: u32) -> Rect {
        // Sub-graph needs the full input to execute its internal pipeline
        Rect::new(0, 0, bounds_w, bounds_h).clamp(bounds_w, bounds_h)
    }

    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::Sequential
    }

    fn upstream_id(&self) -> Option<u32> {
        Some(self.outer_upstream)
    }
}

// ─── Builder ─────────────────────────────────────────────────────────────────

/// Builder for constructing sub-graph node definitions.
///
/// Usage:
/// ```ignore
/// let (sub_graph_node, _output_id) = SubGraphBuilder::new(input_info)
///     .add_filter(|input_id, graph| {
///         // Add nodes to the internal graph, return the terminal node ID
///         let filter_id = graph.add_node(Box::new(some_filter_node));
///         filter_id
///     })
///     .build(outer_upstream_id)?;
/// ```
pub struct SubGraphBuilder {
    input_info: ImageInfo,
    /// Closures that build the internal graph. Each receives (input_node_id, &mut NodeGraph)
    /// and returns the ID of its terminal node.
    build_steps: Vec<Box<dyn FnOnce(u32, &mut NodeGraph) -> u32>>,
}

impl SubGraphBuilder {
    pub fn new(input_info: ImageInfo) -> Self {
        Self {
            input_info,
            build_steps: Vec::new(),
        }
    }

    /// Add a build step that adds nodes to the internal graph.
    /// The closure receives the current terminal node ID and the graph,
    /// and must return the new terminal node ID.
    pub fn add_step(
        mut self,
        step: impl FnOnce(u32, &mut NodeGraph) -> u32 + 'static,
    ) -> Self {
        self.build_steps.push(Box::new(step));
        self
    }

    /// Build the sub-graph node.
    /// Returns `(SubGraphNode, output_info)`.
    pub fn build(self, outer_upstream: u32) -> Result<(SubGraphNode, ImageInfo), ImageError> {
        let shared_buf: SharedBuffer = Rc::new(RefCell::new(Vec::new()));

        let mut graph = NodeGraph::new(0); // No cache budget for sub-graphs
        let input_node = InputRefNode::new(self.input_info.clone(), Rc::clone(&shared_buf));
        let input_id = graph.add_node(Box::new(input_node));

        // Execute build steps, threading the terminal node ID through
        let mut terminal_id = input_id;
        for step in self.build_steps {
            terminal_id = step(terminal_id, &mut graph);
        }

        // Add output ref node
        let output_info = graph.node_info(terminal_id)?;
        let output_id = graph.add_node(Box::new(OutputRefNode::new(terminal_id, output_info.clone())));

        let node = SubGraphNode::new(outer_upstream, graph, shared_buf, output_id)?;
        Ok((node, output_info))
    }
}

// ─── Analysis Sink ───────────────────────────────────────────────────────────

/// Result of an analysis sink operation.
#[derive(Debug, Clone)]
pub enum AnalysisResult {
    /// Auto-levels: black point, white point, gamma
    Levels {
        black: f32,
        white: f32,
        gamma: f32,
    },
    /// Auto-crop: detected content bounds
    CropRect {
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    },
    /// Histogram data
    Histogram {
        r: Vec<u32>,
        g: Vec<u32>,
        b: Vec<u32>,
    },
}

/// Trait for analysis sinks that consume pixels and produce structured results.
pub trait AnalysisSink {
    /// Analyze the given pixel buffer and return a structured result.
    fn analyze(&self, pixels: &[u8], info: &ImageInfo) -> Result<AnalysisResult, ImageError>;
}

/// Auto-levels analysis: computes black/white points from histogram.
pub struct AutoLevelsAnalysis {
    /// Percentage of pixels to clip at each end (0.0-1.0). Default 0.01 (1%).
    pub clip_percent: f32,
}

impl Default for AutoLevelsAnalysis {
    fn default() -> Self {
        Self {
            clip_percent: 0.01,
        }
    }
}

impl AnalysisSink for AutoLevelsAnalysis {
    fn analyze(&self, pixels: &[u8], info: &ImageInfo) -> Result<AnalysisResult, ImageError> {
        let bpp = bytes_per_pixel(info.format) as usize;
        let n_pixels = (info.width as usize) * (info.height as usize);

        // Build luminance histogram
        let mut histogram = [0u32; 256];
        for i in 0..n_pixels {
            let offset = i * bpp;
            let luma = match info.format {
                PixelFormat::Gray8 => pixels[offset],
                PixelFormat::Rgb8 | PixelFormat::Rgba8 => {
                    let r = pixels[offset] as f32;
                    let g = pixels[offset + 1] as f32;
                    let b = pixels[offset + 2] as f32;
                    (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 255.0) as u8
                }
                _ => pixels[offset],
            };
            histogram[luma as usize] += 1;
        }

        let clip_count = (n_pixels as f32 * self.clip_percent) as u32;

        // Find black point (clip_percent from the left)
        let mut accum = 0u32;
        let mut black = 0u8;
        for (i, &count) in histogram.iter().enumerate() {
            accum += count;
            if accum >= clip_count {
                black = i as u8;
                break;
            }
        }

        // Find white point (clip_percent from the right)
        accum = 0;
        let mut white = 255u8;
        for (i, &count) in histogram.iter().enumerate().rev() {
            accum += count;
            if accum >= clip_count {
                white = i as u8;
                break;
            }
        }

        // Compute gamma (midtone balance)
        // Mean luminance in the [black, white] range, normalized to 0-1
        let range = (white as f32 - black as f32).max(1.0);
        let mut sum = 0f64;
        let mut count = 0u64;
        for i in (black as usize)..=(white as usize) {
            sum += (i as f64 - black as f64) * histogram[i] as f64;
            count += histogram[i] as u64;
        }
        let mean_norm = if count > 0 {
            (sum / count as f64 / range as f64) as f32
        } else {
            0.5
        };
        // Gamma: if mean is below 0.5, image is dark → gamma < 1 brightens
        // Approximate: gamma = log(0.5) / log(mean_norm)
        let gamma = if mean_norm > 0.001 && mean_norm < 0.999 {
            (0.5f32.ln() / mean_norm.ln()).clamp(0.1, 10.0)
        } else {
            1.0
        };

        Ok(AnalysisResult::Levels {
            black: black as f32 / 255.0,
            white: white as f32 / 255.0,
            gamma,
        })
    }
}
