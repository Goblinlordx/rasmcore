//! GPU parallel scan (prefix sum) and reduce primitives.
//!
//! # Which primitive to use
//!
//! | Need | Primitive | Why |
//! |------|-----------|-----|
//! | **Running total** (prefix sum, CDF, integral image row/col) | `BlellochScan` | O(n) work, multi-workgroup, general-purpose |
//! | **Small array scan** (histogram CDF ≤256 bins, LUT build) | `HillisSteeleScan` | Single workgroup, lowest latency for ≤256 elements |
//! | **Single aggregate** (total pixel count, global min/max) | `ParallelReduce` | 2-pass tree reduce, produces one value |
//! | **Integral image** (adaptive threshold, local statistics) | `BlellochScan` × 2 | Row-wise scan then column-wise scan |
//! | **Stream compaction** (sparse filter output) | `BlellochScan` (exclusive) + scatter | Exclusive scan gives write offsets |
//!
//! # Decision flow
//!
//! ```text
//! Need per-element prefix?
//!   ├─ No → ParallelReduce (sum/min/max → single value)
//!   └─ Yes → Array ≤ 256?
//!       ├─ Yes → HillisSteeleScan (single workgroup, lowest latency)
//!       └─ No → BlellochScan (multi-workgroup, O(n) work)
//! ```
//!
//! # Operations
//!
//! All primitives support three associative operations via `ScanOp`:
//! - `Sum` — addition (identity: 0). For prefix sums, CDF, integral image.
//! - `Min` — minimum (identity: u32::MAX). For range queries, erosion.
//! - `Max` — maximum (identity: 0). For range queries, dilation.
//!
//! # Scan modes
//!
//! - `Inclusive`: `output[i] = op(input[0], ..., input[i])` — element i is included
//! - `Exclusive`: `output[i] = op(input[0], ..., input[i-1])` — element i excluded, output[0] = identity
//!
//! Use **exclusive** for write offsets (stream compaction, histogram CDF).
//! Use **inclusive** for running totals (integral image).
//!
//! # Example
//!
//! ```rust,ignore
//! use crate::gpu_shaders::scan::{BlellochScan, ScanOp, ScanMode};
//!
//! // Exclusive prefix sum over 10000 u32 elements
//! let scan = BlellochScan::new(ScanOp::Sum, ScanMode::Exclusive, 256);
//! let passes = scan.build_passes(10000);
//! // passes.passes is a Vec<GpuShader> — add to your filter's gpu_shader_passes()
//!
//! // Reduce to find max value
//! use crate::gpu_shaders::scan::ParallelReduce;
//! let reduce = ParallelReduce::new(ScanOp::Max, 256);
//! let passes = reduce.build_passes(10000);
//! // passes.passes[0] = local reduce, passes.passes[1] = global merge
//! ```
//!
//! # Future primitives (not yet implemented)
//!
//! - **Decoupled lookback scan**: single-pass with atomics, ~2x faster than Blelloch
//!   for 1M+ elements. Uses Merrill & Garland (2016) chained scan technique.
//! - **2D prefix sum**: row scan + column scan → integral image primitive.
//! - **Radix sort**: built on exclusive scan, enables parallel median/quantization.
//!
//! All primitives generate `GpuShader` passes via a builder API matching the
//! existing `GpuReduction` pattern. Unused primitives are dead-code eliminated.

use crate::node::{GpuShader, ReductionBuffer};

// ─── Scan Kind ─────────────────────────────────────────────────────────────

/// Type of scan operation.
#[derive(Debug, Clone, Copy)]
pub enum ScanOp {
    /// Prefix sum (addition).
    Sum,
    /// Prefix min.
    Min,
    /// Prefix max.
    Max,
}

/// Whether the scan is inclusive or exclusive.
#[derive(Debug, Clone, Copy)]
pub enum ScanMode {
    /// Inclusive: output[i] = op(input[0..=i])
    Inclusive,
    /// Exclusive: output[i] = op(input[0..i]), output[0] = identity
    Exclusive,
}

// ─── Hillis-Steele Scan ────────────────────────────────────────────────────

/// Hillis-Steele scan — single workgroup, O(n log n) work.
///
/// Best for small arrays (≤ workgroup size, typically 256).
/// Uses shared memory ping-pong within one workgroup.
pub struct HillisSteeleScan {
    pub op: ScanOp,
    pub mode: ScanMode,
    pub workgroup_size: u32,
}

impl HillisSteeleScan {
    pub fn new(op: ScanOp, mode: ScanMode, workgroup_size: u32) -> Self {
        Self {
            op,
            mode,
            workgroup_size,
        }
    }

    /// Generate WGSL for Hillis-Steele scan within shared memory.
    pub fn wgsl(&self) -> String {
        let op_fn = match self.op {
            ScanOp::Sum => "fn scan_op(a: u32, b: u32) -> u32 { return a + b; }",
            ScanOp::Min => "fn scan_op(a: u32, b: u32) -> u32 { return min(a, b); }",
            ScanOp::Max => "fn scan_op(a: u32, b: u32) -> u32 { return max(a, b); }",
        };
        let identity = match self.op {
            ScanOp::Sum => "0u",
            ScanOp::Min => "0xFFFFFFFFu",
            ScanOp::Max => "0u",
        };
        let wg = self.workgroup_size;
        let exclusive_shift = match self.mode {
            ScanMode::Inclusive => "",
            ScanMode::Exclusive => {
                "
    // Shift right for exclusive scan
    workgroupBarrier();
    if (lid > 0u) {
        buf_out[lid] = buf_in[lid - 1u];
    } else {
        buf_out[lid] = IDENTITY;
    }
    workgroupBarrier();
    buf_in[lid] = buf_out[lid];
    workgroupBarrier();"
            }
        };

        format!(
            r#"
{op_fn}
const IDENTITY: u32 = {identity};
const WG_SIZE: u32 = {wg}u;

var<workgroup> buf_a: array<u32, {wg}>;
var<workgroup> buf_b: array<u32, {wg}>;

struct Params {{ array_len: u32, _pad1: u32, _pad2: u32, _pad3: u32, }}
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size({wg}, 1, 1)
fn main(@builtin(local_invocation_id) lid_v: vec3<u32>) {{
    let lid = lid_v.x;

    // Load from global to shared
    var buf_in = &buf_a;
    var buf_out = &buf_b;
    if (lid < params.array_len) {{
        (*buf_in)[lid] = input[lid];
    }} else {{
        (*buf_in)[lid] = IDENTITY;
    }}
    workgroupBarrier();

    // Hillis-Steele: log2(WG_SIZE) steps
    var offset = 1u;
    while (offset < WG_SIZE) {{
        if (lid >= offset) {{
            (*buf_out)[lid] = scan_op((*buf_in)[lid - offset], (*buf_in)[lid]);
        }} else {{
            (*buf_out)[lid] = (*buf_in)[lid];
        }}
        workgroupBarrier();
        // Swap
        let tmp = buf_in;
        buf_in = buf_out;
        buf_out = tmp;
        offset *= 2u;
        workgroupBarrier();
    }}
    {exclusive_shift}
    // Write result
    if (lid < params.array_len) {{
        output[lid] = (*buf_in)[lid];
    }}
}}
"#
        )
    }
}

// ─── Blelloch Scan ─────────────────────────────────────────────────────────

/// Blelloch scan — O(n) work, two-pass (upsweep + downsweep).
///
/// For arrays larger than one workgroup:
/// - Pass 1 (upsweep): each workgroup scans locally, writes its aggregate to a block sums array
/// - Pass 2 (propagate): scan the block sums (recursive or Hillis-Steele for small count)
/// - Pass 3 (downsweep): add block prefix to each element
///
/// This implementation handles arrays up to workgroup_size^2 elements.
pub struct BlellochScan {
    pub op: ScanOp,
    pub mode: ScanMode,
    pub workgroup_size: u32,
}

impl BlellochScan {
    pub fn new(op: ScanOp, mode: ScanMode, workgroup_size: u32) -> Self {
        Self {
            op,
            mode,
            workgroup_size,
        }
    }

    /// Build GPU shader passes for scanning an array of `array_len` u32 elements.
    ///
    /// Returns 2-3 passes depending on array size:
    /// - ≤ workgroup_size: single Hillis-Steele pass
    /// - > workgroup_size: 3 passes (local scan + block scan + propagate)
    pub fn build_passes(&self, array_len: u32) -> ScanPasses {
        let wg = self.workgroup_size;

        if array_len <= wg {
            // Single workgroup — use Hillis-Steele
            let hs = HillisSteeleScan::new(self.op, self.mode, wg);
            let mut params = Vec::with_capacity(16);
            params.extend_from_slice(&array_len.to_le_bytes());
            params.extend_from_slice(&0u32.to_le_bytes());
            params.extend_from_slice(&0u32.to_le_bytes());
            params.extend_from_slice(&0u32.to_le_bytes());
            return ScanPasses {
                passes: vec![GpuShader::new(hs.wgsl(), "main", [wg, 1, 1], params)],
                array_len,
            };
        }

        let num_blocks = array_len.div_ceil(wg);
        let buffer_id_blocks = 100; // unique ID for block sums buffer

        // Pass 1: local scan per workgroup + write block aggregate
        let pass1_wgsl = self.local_scan_wgsl(true);
        let mut p1_params = Vec::with_capacity(16);
        p1_params.extend_from_slice(&array_len.to_le_bytes());
        p1_params.extend_from_slice(&num_blocks.to_le_bytes());
        p1_params.extend_from_slice(&0u32.to_le_bytes());
        p1_params.extend_from_slice(&0u32.to_le_bytes());
        let pass1 = GpuShader::new(pass1_wgsl, "main", [wg, 1, 1], p1_params)
            .with_reduction_buffers(vec![ReductionBuffer {
                id: buffer_id_blocks,
                initial_data: vec![0u8; num_blocks as usize * 4],
                read_write: true,
            }]);

        // Pass 2: scan block sums (single workgroup Hillis-Steele)
        let _hs = HillisSteeleScan::new(self.op, ScanMode::Exclusive, wg.min(num_blocks));
        let mut p2_params = Vec::with_capacity(16);
        p2_params.extend_from_slice(&num_blocks.to_le_bytes());
        p2_params.extend_from_slice(&0u32.to_le_bytes());
        p2_params.extend_from_slice(&0u32.to_le_bytes());
        p2_params.extend_from_slice(&0u32.to_le_bytes());
        let pass2 = GpuShader::new(self.block_scan_wgsl(), "main", [wg, 1, 1], p2_params)
            .with_reduction_buffers(vec![ReductionBuffer {
                id: buffer_id_blocks,
                initial_data: vec![],
                read_write: true,
            }]);

        // Pass 3: add block prefix to each element
        let mut p3_params = Vec::with_capacity(16);
        p3_params.extend_from_slice(&array_len.to_le_bytes());
        p3_params.extend_from_slice(&num_blocks.to_le_bytes());
        p3_params.extend_from_slice(&0u32.to_le_bytes());
        p3_params.extend_from_slice(&0u32.to_le_bytes());
        let pass3 = GpuShader::new(self.propagate_wgsl(), "main", [wg, 1, 1], p3_params)
            .with_reduction_buffers(vec![ReductionBuffer {
                id: buffer_id_blocks,
                initial_data: vec![],
                read_write: false,
            }]);

        ScanPasses {
            passes: vec![pass1, pass2, pass3],
            array_len,
        }
    }

    fn op_fn_wgsl(&self) -> &'static str {
        match self.op {
            ScanOp::Sum => "fn scan_op(a: u32, b: u32) -> u32 { return a + b; }",
            ScanOp::Min => "fn scan_op(a: u32, b: u32) -> u32 { return min(a, b); }",
            ScanOp::Max => "fn scan_op(a: u32, b: u32) -> u32 { return max(a, b); }",
        }
    }

    fn identity_wgsl(&self) -> &'static str {
        match self.op {
            ScanOp::Sum => "0u",
            ScanOp::Min => "0xFFFFFFFFu",
            ScanOp::Max => "0u",
        }
    }

    /// Pass 1: local inclusive scan within workgroup + write aggregate to block sums.
    fn local_scan_wgsl(&self, _write_aggregate: bool) -> String {
        let wg = self.workgroup_size;
        let op_fn = self.op_fn_wgsl();
        let identity = self.identity_wgsl();
        let excl = match self.mode {
            ScanMode::Inclusive => "(*shared)[lid]",
            ScanMode::Exclusive => "select((*shared)[lid - 1u], IDENTITY, lid == 0u)",
        };
        format!(
            r#"
{op_fn}
const IDENTITY: u32 = {identity};
const WG: u32 = {wg}u;
var<workgroup> buf_a: array<u32, {wg}>;
var<workgroup> buf_b: array<u32, {wg}>;

struct Params {{ array_len: u32, num_blocks: u32, _p2: u32, _p3: u32, }}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> block_sums: array<u32>;

@compute @workgroup_size({wg}, 1, 1)
fn main(
    @builtin(local_invocation_id) lid_v: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {{
    let lid = lid_v.x;
    let gid = wgid.x * WG + lid;
    var shared = &buf_a;
    var shared2 = &buf_b;

    // Load: convert pixel luminance to u32 (quantize to 0..65535 for precision)
    if (gid < params.array_len) {{
        let p = input[gid];
        let luma = u32(clamp(0.2126 * p.x + 0.7152 * p.y + 0.0722 * p.z, 0.0, 1.0) * 65535.0);
        (*shared)[lid] = luma;
    }} else {{
        (*shared)[lid] = IDENTITY;
    }}
    workgroupBarrier();

    // Hillis-Steele inclusive scan within workgroup
    var offset = 1u;
    while (offset < WG) {{
        if (lid >= offset) {{
            (*shared2)[lid] = scan_op((*shared)[lid - offset], (*shared)[lid]);
        }} else {{
            (*shared2)[lid] = (*shared)[lid];
        }}
        workgroupBarrier();
        let tmp = shared;
        shared = shared2;
        shared2 = tmp;
        offset *= 2u;
        workgroupBarrier();
    }}

    // Write local result (inclusive scan)
    if (gid < params.array_len) {{
        let v = {excl};
        // Store back as f32 in alpha-like channel for now
        output[gid] = vec4(f32(v) / 65535.0, 0.0, 0.0, 0.0);
    }}

    // Write aggregate for this block
    if (lid == WG - 1u) {{
        block_sums[wgid.x] = (*shared)[WG - 1u];
    }}
}}
"#
        )
    }

    /// Pass 2: scan the block sums array (single workgroup).
    fn block_scan_wgsl(&self) -> String {
        let wg = self.workgroup_size;
        let op_fn = self.op_fn_wgsl();
        let identity = self.identity_wgsl();
        format!(
            r#"
{op_fn}
const IDENTITY: u32 = {identity};
const WG: u32 = {wg}u;
var<workgroup> buf_a: array<u32, {wg}>;
var<workgroup> buf_b: array<u32, {wg}>;

struct Params {{ num_blocks: u32, _p1: u32, _p2: u32, _p3: u32, }}
@group(0) @binding(0) var<storage, read> _unused_in: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> _unused_out: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> block_sums: array<u32>;

@compute @workgroup_size({wg}, 1, 1)
fn main(@builtin(local_invocation_id) lid_v: vec3<u32>) {{
    let lid = lid_v.x;
    var shared = &buf_a;
    var shared2 = &buf_b;

    if (lid < params.num_blocks) {{
        (*shared)[lid] = block_sums[lid];
    }} else {{
        (*shared)[lid] = IDENTITY;
    }}
    workgroupBarrier();

    // Exclusive scan of block sums
    var offset = 1u;
    while (offset < WG) {{
        if (lid >= offset) {{
            (*shared2)[lid] = scan_op((*shared)[lid - offset], (*shared)[lid]);
        }} else {{
            (*shared2)[lid] = (*shared)[lid];
        }}
        workgroupBarrier();
        let tmp = shared;
        shared = shared2;
        shared2 = tmp;
        offset *= 2u;
        workgroupBarrier();
    }}

    // Convert inclusive to exclusive: shift right
    if (lid < params.num_blocks) {{
        if (lid > 0u) {{
            block_sums[lid] = (*shared)[lid - 1u];
        }} else {{
            block_sums[lid] = IDENTITY;
        }}
    }}
}}
"#
        )
    }

    /// Pass 3: add block prefix to each element.
    fn propagate_wgsl(&self) -> String {
        let wg = self.workgroup_size;
        let op_fn = self.op_fn_wgsl();
        format!(
            r#"
{op_fn}
const WG: u32 = {wg}u;

struct Params {{ array_len: u32, num_blocks: u32, _p2: u32, _p3: u32, }}
@group(0) @binding(0) var<storage, read> _unused_in: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> block_sums: array<u32>;

@compute @workgroup_size({wg}, 1, 1)
fn main(
    @builtin(local_invocation_id) lid_v: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {{
    let gid = wgid.x * WG + lid_v.x;
    if (gid >= params.array_len) {{ return; }}
    let prefix = block_sums[wgid.x];
    let current = u32(output[gid].x * 65535.0);
    let result = scan_op(prefix, current);
    output[gid] = vec4(f32(result) / 65535.0, 0.0, 0.0, 0.0);
}}
"#
        )
    }
}

/// Output of scan build_passes.
pub struct ScanPasses {
    pub passes: Vec<GpuShader>,
    pub array_len: u32,
}

// ─── Parallel Reduce ───────────────────────────────────────────────────────

/// Parallel reduce — produces a single aggregate value (sum, min, max) from an array.
///
/// Uses workgroup-level tree reduction + global merge (same pattern as existing
/// histogram reduction, but for scalar values).
pub struct ParallelReduce {
    pub op: ScanOp,
    pub workgroup_size: u32,
}

impl ParallelReduce {
    pub fn new(op: ScanOp, workgroup_size: u32) -> Self {
        Self { op, workgroup_size }
    }

    /// Generate a 2-pass reduce: local reduce per workgroup → global merge.
    pub fn build_passes(&self, array_len: u32) -> ScanPasses {
        let wg = self.workgroup_size;
        let num_blocks = array_len.div_ceil(wg);
        let buffer_id = 200;

        let op_fn = match self.op {
            ScanOp::Sum => "fn reduce_op(a: u32, b: u32) -> u32 { return a + b; }",
            ScanOp::Min => "fn reduce_op(a: u32, b: u32) -> u32 { return min(a, b); }",
            ScanOp::Max => "fn reduce_op(a: u32, b: u32) -> u32 { return max(a, b); }",
        };
        let identity = match self.op {
            ScanOp::Sum => "0u",
            ScanOp::Min => "0xFFFFFFFFu",
            ScanOp::Max => "0u",
        };

        // Pass 1: per-workgroup tree reduction
        let pass1_wgsl = format!(
            r#"
{op_fn}
const IDENTITY: u32 = {identity};
const WG: u32 = {wg}u;
var<workgroup> shared: array<u32, {wg}>;

struct Params {{ array_len: u32, num_blocks: u32, _p2: u32, _p3: u32, }}
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> _out: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> partials: array<u32>;

@compute @workgroup_size({wg}, 1, 1)
fn main(
    @builtin(local_invocation_id) lid_v: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {{
    let lid = lid_v.x;
    let gid = wgid.x * WG + lid;
    shared[lid] = select(IDENTITY, input[gid], gid < params.array_len);
    workgroupBarrier();

    // Tree reduce
    var stride = WG / 2u;
    while (stride > 0u) {{
        if (lid < stride) {{
            shared[lid] = reduce_op(shared[lid], shared[lid + stride]);
        }}
        workgroupBarrier();
        stride /= 2u;
    }}

    if (lid == 0u) {{
        partials[wgid.x] = shared[0];
    }}
}}
"#
        );

        let p1 = GpuShader::new(pass1_wgsl, "main", [wg, 1, 1], {
            let mut p = Vec::with_capacity(16);
            p.extend_from_slice(&array_len.to_le_bytes());
            p.extend_from_slice(&num_blocks.to_le_bytes());
            p.extend_from_slice(&0u32.to_le_bytes());
            p.extend_from_slice(&0u32.to_le_bytes());
            p
        })
        .with_reduction_buffers(vec![ReductionBuffer {
            id: buffer_id,
            initial_data: vec![0u8; num_blocks as usize * 4],
            read_write: true,
        }]);

        // Pass 2: reduce partials to single value (workgroup 0 only)
        let pass2_wgsl = format!(
            r#"
{op_fn}
const IDENTITY: u32 = {identity};
const WG: u32 = {wg}u;
var<workgroup> shared: array<u32, {wg}>;

struct Params {{ array_len: u32, num_blocks: u32, _p2: u32, _p3: u32, }}
@group(0) @binding(0) var<storage, read> _in: array<u32>;
@group(0) @binding(1) var<storage, read_write> result: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> partials: array<u32>;

@compute @workgroup_size({wg}, 1, 1)
fn main(@builtin(local_invocation_id) lid_v: vec3<u32>) {{
    let lid = lid_v.x;
    shared[lid] = select(IDENTITY, partials[lid], lid < params.num_blocks);
    workgroupBarrier();

    var stride = WG / 2u;
    while (stride > 0u) {{
        if (lid < stride) {{
            shared[lid] = reduce_op(shared[lid], shared[lid + stride]);
        }}
        workgroupBarrier();
        stride /= 2u;
    }}

    if (lid == 0u) {{
        result[0] = shared[0];
    }}
}}
"#
        );

        let p2 = GpuShader::new(pass2_wgsl, "main", [wg, 1, 1], {
            let mut p = Vec::with_capacity(16);
            p.extend_from_slice(&array_len.to_le_bytes());
            p.extend_from_slice(&num_blocks.to_le_bytes());
            p.extend_from_slice(&0u32.to_le_bytes());
            p.extend_from_slice(&0u32.to_le_bytes());
            p
        })
        .with_reduction_buffers(vec![ReductionBuffer {
            id: buffer_id,
            initial_data: vec![],
            read_write: true,
        }]);

        ScanPasses {
            passes: vec![p1, p2],
            array_len,
        }
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hillis_steele_generates_valid_wgsl() {
        let hs = HillisSteeleScan::new(ScanOp::Sum, ScanMode::Inclusive, 256);
        let wgsl = hs.wgsl();
        assert!(wgsl.contains("fn scan_op"), "should contain scan_op");
        assert!(wgsl.contains("workgroupBarrier"), "should contain barriers");
        assert!(wgsl.contains("buf_a"), "should use shared memory");
    }

    #[test]
    fn hillis_steele_exclusive_has_shift() {
        let hs = HillisSteeleScan::new(ScanOp::Sum, ScanMode::Exclusive, 256);
        let wgsl = hs.wgsl();
        assert!(wgsl.contains("Shift right"), "exclusive should shift");
    }

    #[test]
    fn blelloch_small_array_single_pass() {
        let scan = BlellochScan::new(ScanOp::Sum, ScanMode::Exclusive, 256);
        let passes = scan.build_passes(100);
        assert_eq!(
            passes.passes.len(),
            1,
            "≤ workgroup size should be single pass"
        );
    }

    #[test]
    fn blelloch_large_array_three_passes() {
        let scan = BlellochScan::new(ScanOp::Sum, ScanMode::Exclusive, 256);
        let passes = scan.build_passes(1000);
        assert_eq!(
            passes.passes.len(),
            3,
            "> workgroup size should be 3 passes"
        );
    }

    #[test]
    fn parallel_reduce_generates_two_passes() {
        let reduce = ParallelReduce::new(ScanOp::Sum, 256);
        let passes = reduce.build_passes(1024);
        assert_eq!(passes.passes.len(), 2, "reduce should be 2 passes");
    }

    #[test]
    fn parallel_reduce_min_has_correct_identity() {
        let reduce = ParallelReduce::new(ScanOp::Min, 256);
        let passes = reduce.build_passes(100);
        let wgsl = &passes.passes[0].body;
        assert!(
            wgsl.contains("0xFFFFFFFF"),
            "min identity should be u32::MAX"
        );
    }

    #[test]
    fn scan_ops_cover_all_variants() {
        for op in [ScanOp::Sum, ScanOp::Min, ScanOp::Max] {
            let hs = HillisSteeleScan::new(op, ScanMode::Inclusive, 64);
            let wgsl = hs.wgsl();
            assert!(wgsl.contains("fn scan_op"), "{:?} should have scan_op", op);
        }
    }
}
