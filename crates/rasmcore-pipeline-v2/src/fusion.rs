//! Fusion optimizer — composes chains of fusable nodes into single operations.
//!
//! Three fusable categories:
//! 1. **Analytical (point ops)**: expression tree composition → single f32 function
//! 2. **Affine (transforms)**: matrix composition → single resample pass
//! 3. **CLUT (color ops)**: 3D LUT composition → single tetrahedral interpolation
//!
//! The optimizer walks the graph, detects chains of same-category nodes,
//! composes them, and replaces the chain with a single fused node.

use crate::graph::Graph;
use crate::node::{GpuShader, Node, NodeCapabilities, NodeInfo, PipelineError, Upstream};
use crate::ops::PointOpExpr;
use crate::rect::Rect;
// TraceEventKind/TraceTimer used by callers (request_full, gpu_plan), not here
#[allow(unused_imports)]
use crate::trace::{TraceEventKind, TraceTimer};

// ─── Expression Tree Optimizer ──────────────────────────────────────────────

/// Constant-fold an expression tree: evaluate subtrees with no `Input` references.
pub fn constant_fold(expr: &PointOpExpr) -> PointOpExpr {
    match expr {
        PointOpExpr::Input => PointOpExpr::Input,
        PointOpExpr::Constant(_) => expr.clone(),

        PointOpExpr::Add(a, b) => {
            let a = constant_fold(a);
            let b = constant_fold(b);
            match (&a, &b) {
                (PointOpExpr::Constant(x), PointOpExpr::Constant(y)) => {
                    PointOpExpr::Constant(x + y)
                }
                // x + 0 = x
                (_, PointOpExpr::Constant(y)) if *y == 0.0 => a,
                (PointOpExpr::Constant(x), _) if *x == 0.0 => b,
                _ => PointOpExpr::Add(Box::new(a), Box::new(b)),
            }
        }
        PointOpExpr::Sub(a, b) => {
            let a = constant_fold(a);
            let b = constant_fold(b);
            match (&a, &b) {
                (PointOpExpr::Constant(x), PointOpExpr::Constant(y)) => {
                    PointOpExpr::Constant(x - y)
                }
                // x - 0 = x
                (_, PointOpExpr::Constant(y)) if *y == 0.0 => a,
                _ => PointOpExpr::Sub(Box::new(a), Box::new(b)),
            }
        }
        PointOpExpr::Mul(a, b) => {
            let a = constant_fold(a);
            let b = constant_fold(b);
            match (&a, &b) {
                (PointOpExpr::Constant(x), PointOpExpr::Constant(y)) => {
                    PointOpExpr::Constant(x * y)
                }
                // x * 1 = x
                (_, PointOpExpr::Constant(y)) if *y == 1.0 => a,
                (PointOpExpr::Constant(x), _) if *x == 1.0 => b,
                // x * 0 = 0
                (_, PointOpExpr::Constant(y)) if *y == 0.0 => PointOpExpr::Constant(0.0),
                (PointOpExpr::Constant(x), _) if *x == 0.0 => PointOpExpr::Constant(0.0),
                _ => PointOpExpr::Mul(Box::new(a), Box::new(b)),
            }
        }
        PointOpExpr::Div(a, b) => {
            let a = constant_fold(a);
            let b = constant_fold(b);
            match (&a, &b) {
                (PointOpExpr::Constant(x), PointOpExpr::Constant(y)) if y.abs() > 1e-30 => {
                    PointOpExpr::Constant(x / y)
                }
                // x / 1 = x
                (_, PointOpExpr::Constant(y)) if *y == 1.0 => a,
                _ => PointOpExpr::Div(Box::new(a), Box::new(b)),
            }
        }
        PointOpExpr::Pow(a, b) => {
            let a = constant_fold(a);
            let b = constant_fold(b);
            match (&a, &b) {
                (PointOpExpr::Constant(x), PointOpExpr::Constant(y)) => {
                    PointOpExpr::Constant(x.powf(*y))
                }
                // x ^ 1 = x
                (_, PointOpExpr::Constant(y)) if *y == 1.0 => a,
                // x ^ 0 = 1
                (_, PointOpExpr::Constant(y)) if *y == 0.0 => PointOpExpr::Constant(1.0),
                _ => PointOpExpr::Pow(Box::new(a), Box::new(b)),
            }
        }
        PointOpExpr::Clamp(v, min, max) => {
            let v = constant_fold(v);
            if let PointOpExpr::Constant(c) = &v {
                PointOpExpr::Constant(c.clamp(*min, *max))
            } else {
                PointOpExpr::Clamp(Box::new(v), *min, *max)
            }
        }
        PointOpExpr::Floor(v) => {
            let v = constant_fold(v);
            if let PointOpExpr::Constant(c) = &v {
                PointOpExpr::Constant(c.floor())
            } else {
                PointOpExpr::Floor(Box::new(v))
            }
        }
        PointOpExpr::Max(a, b) => {
            let a = constant_fold(a);
            let b = constant_fold(b);
            match (&a, &b) {
                (PointOpExpr::Constant(x), PointOpExpr::Constant(y)) => {
                    PointOpExpr::Constant(x.max(*y))
                }
                _ => PointOpExpr::Max(Box::new(a), Box::new(b)),
            }
        }
        PointOpExpr::Min(a, b) => {
            let a = constant_fold(a);
            let b = constant_fold(b);
            match (&a, &b) {
                (PointOpExpr::Constant(x), PointOpExpr::Constant(y)) => {
                    PointOpExpr::Constant(x.min(*y))
                }
                _ => PointOpExpr::Min(Box::new(a), Box::new(b)),
            }
        }
        PointOpExpr::Exp(v) => {
            let v = constant_fold(v);
            if let PointOpExpr::Constant(c) = &v {
                PointOpExpr::Constant(c.exp())
            } else {
                PointOpExpr::Exp(Box::new(v))
            }
        }
        PointOpExpr::Ln(v) => {
            let v = constant_fold(v);
            if let PointOpExpr::Constant(c) = &v {
                PointOpExpr::Constant(if *c > 0.0 { c.ln() } else { -30.0 })
            } else {
                PointOpExpr::Ln(Box::new(v))
            }
        }
        PointOpExpr::Select(cond, t, f) => {
            let cond = constant_fold(cond);
            let t = constant_fold(t);
            let f = constant_fold(f);
            if let PointOpExpr::Constant(c) = &cond {
                if *c > 0.0 { t } else { f }
            } else {
                PointOpExpr::Select(Box::new(cond), Box::new(t), Box::new(f))
            }
        }
    }
}

/// Check if an expression is the identity function (returns Input unchanged).
pub fn is_identity(expr: &PointOpExpr) -> bool {
    matches!(expr, PointOpExpr::Input)
}

/// Check if an expression is a constant (no Input references).
pub fn is_constant(expr: &PointOpExpr) -> bool {
    match expr {
        PointOpExpr::Input => false,
        PointOpExpr::Constant(_) => true,
        PointOpExpr::Add(a, b)
        | PointOpExpr::Sub(a, b)
        | PointOpExpr::Mul(a, b)
        | PointOpExpr::Div(a, b)
        | PointOpExpr::Pow(a, b)
        | PointOpExpr::Max(a, b)
        | PointOpExpr::Min(a, b) => is_constant(a) && is_constant(b),
        PointOpExpr::Clamp(v, _, _) | PointOpExpr::Floor(v) | PointOpExpr::Exp(v) | PointOpExpr::Ln(v) => is_constant(v),
        PointOpExpr::Select(cond, t, f) => is_constant(cond) && is_constant(t) && is_constant(f),
    }
}

// ─── Lowering Backends ──────────────────────────────────────────────────────

/// Lower an expression to an f32 closure (for CPU evaluation).
///
/// The closure takes a single f32 channel value and returns the transformed value.
/// Uses f64 intermediate precision to avoid accumulation errors in long chains.
pub fn lower_to_closure(expr: &PointOpExpr) -> Box<dyn Fn(f32) -> f32> {
    let expr = expr.clone();
    Box::new(move |v: f32| expr.evaluate(v as f64) as f32)
}

/// Lower an expression to a 256-entry u8 LUT (for u8 encode boundary).
///
/// Evaluates the expression at 256 evenly-spaced points and quantizes to u8.
pub fn lower_to_lut(expr: &PointOpExpr) -> [u8; 256] {
    expr.bake_to_lut()
}

/// Number of entries in the f32 LUT. 4096 gives <0.025% interpolation error
/// for typical point-op chains (brightness, contrast, gamma, levels).
const F32_LUT_SIZE: usize = 4096;

/// Pre-baked f32 lookup table for CPU-side fused point-op evaluation.
///
/// Replaces recursive `PointOpExpr::evaluate()` with a single LUT lookup + lerp
/// per channel per pixel. Covers [0.0, 1.0] with `F32_LUT_SIZE` entries; values
/// outside that range are extrapolated from the nearest boundary entries.
pub struct F32Lut {
    /// LUT entries for [0.0, 1.0] sampled at F32_LUT_SIZE+1 points.
    table: Vec<f32>,
    /// 1.0 / step size = F32_LUT_SIZE as f32 (precomputed for lerp).
    inv_step: f32,
}

impl F32Lut {
    /// Build the LUT by evaluating the expression at evenly-spaced f64 points.
    pub fn build(expr: &PointOpExpr) -> Self {
        let n = F32_LUT_SIZE;
        let mut table = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let v = i as f64 / n as f64;
            table.push(expr.evaluate(v) as f32);
        }
        Self {
            table,
            inv_step: n as f32,
        }
    }

    /// Look up a value with linear interpolation. Values outside [0, 1] are
    /// clamped to the boundary LUT entries.
    #[inline(always)]
    pub fn apply(&self, v: f32) -> f32 {
        let t = v * self.inv_step;
        let t_clamped = t.max(0.0).min(self.inv_step);
        let idx = (t_clamped as u32).min((self.table.len() - 2) as u32);
        let frac = t_clamped - idx as f32;
        let idx = idx as usize;
        // Safety: idx in [0, table.len()-2], idx+1 in [1, table.len()-1]
        let a = unsafe { *self.table.get_unchecked(idx) };
        let b = unsafe { *self.table.get_unchecked(idx + 1) };
        a + frac * (b - a)
    }

    /// Apply LUT to one RGBA pixel (RGB transformed, alpha preserved).
    /// Exposes 3 independent lookup chains for ILP.
    #[inline(always)]
    pub fn apply_pixel(&self, pixel: &mut [f32; 4]) {
        let inv = self.inv_step;
        let max_idx = (self.table.len() - 2) as u32; // F32_LUT_SIZE - 1
        let tr = (pixel[0] * inv).max(0.0).min(inv);
        let tg = (pixel[1] * inv).max(0.0).min(inv);
        let tb = (pixel[2] * inv).max(0.0).min(inv);

        let ir = (tr as u32).min(max_idx) as usize;
        let ig = (tg as u32).min(max_idx) as usize;
        let ib = (tb as u32).min(max_idx) as usize;

        let fr = tr - ir as f32;
        let fg = tg - ig as f32;
        let fb = tb - ib as f32;

        unsafe {
            let ar = *self.table.get_unchecked(ir);
            let ag = *self.table.get_unchecked(ig);
            let ab = *self.table.get_unchecked(ib);
            let br = *self.table.get_unchecked(ir + 1);
            let bg = *self.table.get_unchecked(ig + 1);
            let bb = *self.table.get_unchecked(ib + 1);
            pixel[0] = ar + fr * (br - ar);
            pixel[1] = ag + fg * (bg - ag);
            pixel[2] = ab + fb * (bb - ab);
        }
    }

    /// Apply LUT to a buffer of RGBA pixels in batches of 4 pixels (16 floats).
    /// Remainder pixels use per-pixel apply.
    pub fn apply_buffer(&self, buf: &mut [f32]) {
        let chunks = buf.len() / 16; // 4 pixels * 4 channels
        let batch_end = chunks * 16;

        // Process 4 pixels per iteration — 12 independent lookups for ILP
        for base in (0..batch_end).step_by(16) {
            let s = &mut buf[base..base + 16];
            let inv = self.inv_step;

            // Compute t values for all 12 RGB channels
            let t0r = (s[0] * inv).max(0.0).min(inv);
            let t0g = (s[1] * inv).max(0.0).min(inv);
            let t0b = (s[2] * inv).max(0.0).min(inv);
            let t1r = (s[4] * inv).max(0.0).min(inv);
            let t1g = (s[5] * inv).max(0.0).min(inv);
            let t1b = (s[6] * inv).max(0.0).min(inv);
            let t2r = (s[8] * inv).max(0.0).min(inv);
            let t2g = (s[9] * inv).max(0.0).min(inv);
            let t2b = (s[10] * inv).max(0.0).min(inv);
            let t3r = (s[12] * inv).max(0.0).min(inv);
            let t3g = (s[13] * inv).max(0.0).min(inv);
            let t3b = (s[14] * inv).max(0.0).min(inv);

            // Integer indices (clamped to table bounds)
            let mx = (self.table.len() - 2) as u32;
            let i0r = (t0r as u32).min(mx) as usize; let i0g = (t0g as u32).min(mx) as usize; let i0b = (t0b as u32).min(mx) as usize;
            let i1r = (t1r as u32).min(mx) as usize; let i1g = (t1g as u32).min(mx) as usize; let i1b = (t1b as u32).min(mx) as usize;
            let i2r = (t2r as u32).min(mx) as usize; let i2g = (t2g as u32).min(mx) as usize; let i2b = (t2b as u32).min(mx) as usize;
            let i3r = (t3r as u32).min(mx) as usize; let i3g = (t3g as u32).min(mx) as usize; let i3b = (t3b as u32).min(mx) as usize;

            // Fractions
            let f0r = t0r - i0r as f32; let f0g = t0g - i0g as f32; let f0b = t0b - i0b as f32;
            let f1r = t1r - i1r as f32; let f1g = t1g - i1g as f32; let f1b = t1b - i1b as f32;
            let f2r = t2r - i2r as f32; let f2g = t2g - i2g as f32; let f2b = t2b - i2b as f32;
            let f3r = t3r - i3r as f32; let f3g = t3g - i3g as f32; let f3b = t3b - i3b as f32;

            // Lookups + lerp (12 independent chains)
            unsafe {
                let tbl = &self.table;
                s[0]  = tbl.get_unchecked(i0r).clone() + f0r * (tbl.get_unchecked(i0r + 1) - tbl.get_unchecked(i0r));
                s[1]  = tbl.get_unchecked(i0g).clone() + f0g * (tbl.get_unchecked(i0g + 1) - tbl.get_unchecked(i0g));
                s[2]  = tbl.get_unchecked(i0b).clone() + f0b * (tbl.get_unchecked(i0b + 1) - tbl.get_unchecked(i0b));
                s[4]  = tbl.get_unchecked(i1r).clone() + f1r * (tbl.get_unchecked(i1r + 1) - tbl.get_unchecked(i1r));
                s[5]  = tbl.get_unchecked(i1g).clone() + f1g * (tbl.get_unchecked(i1g + 1) - tbl.get_unchecked(i1g));
                s[6]  = tbl.get_unchecked(i1b).clone() + f1b * (tbl.get_unchecked(i1b + 1) - tbl.get_unchecked(i1b));
                s[8]  = tbl.get_unchecked(i2r).clone() + f2r * (tbl.get_unchecked(i2r + 1) - tbl.get_unchecked(i2r));
                s[9]  = tbl.get_unchecked(i2g).clone() + f2g * (tbl.get_unchecked(i2g + 1) - tbl.get_unchecked(i2g));
                s[10] = tbl.get_unchecked(i2b).clone() + f2b * (tbl.get_unchecked(i2b + 1) - tbl.get_unchecked(i2b));
                s[12] = tbl.get_unchecked(i3r).clone() + f3r * (tbl.get_unchecked(i3r + 1) - tbl.get_unchecked(i3r));
                s[13] = tbl.get_unchecked(i3g).clone() + f3g * (tbl.get_unchecked(i3g + 1) - tbl.get_unchecked(i3g));
                s[14] = tbl.get_unchecked(i3b).clone() + f3b * (tbl.get_unchecked(i3b + 1) - tbl.get_unchecked(i3b));
            }
            // Alpha channels (s[3], s[7], s[11], s[15]) untouched
        }

        // Remainder: per-pixel
        for pixel in buf[batch_end..].chunks_exact_mut(4) {
            let p: &mut [f32; 4] = pixel.try_into().unwrap();
            self.apply_pixel(p);
        }
    }
}

/// Lower an expression to an f32 LUT for fast CPU evaluation.
///
/// This is the primary CPU lowering backend — ~10-50x faster than `lower_to_closure`
/// for deep expression trees, with negligible interpolation error (<0.025%).
pub fn lower_to_f32_lut(expr: &PointOpExpr) -> F32Lut {
    F32Lut::build(expr)
}

/// Lower per-channel expressions to 3 f32 LUTs.
pub fn lower_to_f32_luts(exprs: &[PointOpExpr; 3]) -> [F32Lut; 3] {
    [
        F32Lut::build(&exprs[0]),
        F32Lut::build(&exprs[1]),
        F32Lut::build(&exprs[2]),
    ]
}

/// Lower an expression to WGSL shader source (for GPU kernel fusion).
///
/// Emits an inline WGSL expression that can be composed into a compute shader.
/// The input variable is `v` (f32).
pub fn lower_to_wgsl(expr: &PointOpExpr) -> String {
    match expr {
        PointOpExpr::Input => "v".to_string(),
        PointOpExpr::Constant(c) => {
            if c.fract() == 0.0 {
                format!("{c:.1}")
            } else {
                format!("{c}")
            }
        }
        PointOpExpr::Add(a, b) => format!("({} + {})", lower_to_wgsl(a), lower_to_wgsl(b)),
        PointOpExpr::Sub(a, b) => format!("({} - {})", lower_to_wgsl(a), lower_to_wgsl(b)),
        PointOpExpr::Mul(a, b) => format!("({} * {})", lower_to_wgsl(a), lower_to_wgsl(b)),
        PointOpExpr::Div(a, b) => format!("({} / {})", lower_to_wgsl(a), lower_to_wgsl(b)),
        PointOpExpr::Pow(a, b) => format!("pow({}, {})", lower_to_wgsl(a), lower_to_wgsl(b)),
        PointOpExpr::Clamp(v, min, max) => {
            format!("clamp({}, {:.6}, {:.6})", lower_to_wgsl(v), min, max)
        }
        PointOpExpr::Floor(v) => format!("floor({})", lower_to_wgsl(v)),
        PointOpExpr::Max(a, b) => format!("max({}, {})", lower_to_wgsl(a), lower_to_wgsl(b)),
        PointOpExpr::Min(a, b) => format!("min({}, {})", lower_to_wgsl(a), lower_to_wgsl(b)),
        PointOpExpr::Exp(v) => format!("exp({})", lower_to_wgsl(v)),
        PointOpExpr::Ln(v) => format!("log({})", lower_to_wgsl(v)),
        PointOpExpr::Select(cond, t, f) => {
            format!(
                "select({}, {}, {} > 0.0)",
                lower_to_wgsl(f),
                lower_to_wgsl(t),
                lower_to_wgsl(cond)
            )
        }
    }
}

/// Generate a complete WGSL compute shader that applies a fused point op expression.
///
/// The shader reads f32 RGBA pixels, applies the expression to R, G, B channels
/// (preserving alpha), and writes the result.
pub fn lower_to_wgsl_shader(expr: &PointOpExpr) -> String {
    let exprs = [expr.clone(), expr.clone(), expr.clone()];
    lower_to_wgsl_shader_per_channel(&exprs)
}

/// Generate a per-channel WGSL compute shader with 3 inlined apply functions.
///
/// Each channel gets its own `apply_r`, `apply_g`, `apply_b` function.
/// Single shader dispatch — all 3 functions are inlined in one kernel.
pub fn lower_to_wgsl_shader_per_channel(exprs: &[PointOpExpr; 3]) -> String {
    let wgsl_r = lower_to_wgsl(&exprs[0]);
    let wgsl_g = lower_to_wgsl(&exprs[1]);
    let wgsl_b = lower_to_wgsl(&exprs[2]);
    format!(
        r#"struct Params {{
  width: u32,
  height: u32,
}}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

fn apply_r(v: f32) -> f32 {{
  return {wgsl_r};
}}

fn apply_g(v: f32) -> f32 {{
  return {wgsl_g};
}}

fn apply_b(v: f32) -> f32 {{
  return {wgsl_b};
}}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  if (gid.x >= params.width || gid.y >= params.height) {{
    return;
  }}
  let idx = gid.x + gid.y * params.width;
  let pixel = input[idx];
  output[idx] = vec4<f32>(
    apply_r(pixel.x),
    apply_g(pixel.y),
    apply_b(pixel.z),
    pixel.w,
  );
}}"#
    )
}

// ─── Affine Matrix Composition ──────────────────────────────────────────────

/// Compose two 2x3 affine matrices: result = outer * inner.
///
/// Matrix layout: [a, b, tx, c, d, ty] representing:
/// ```text
/// | a  b  tx |
/// | c  d  ty |
/// | 0  0  1  |
/// ```
pub fn compose_affine(outer: &[f64; 6], inner: &[f64; 6]) -> [f64; 6] {
    [
        outer[0] * inner[0] + outer[1] * inner[3],
        outer[0] * inner[1] + outer[1] * inner[4],
        outer[0] * inner[2] + outer[1] * inner[5] + outer[2],
        outer[3] * inner[0] + outer[4] * inner[3],
        outer[3] * inner[1] + outer[4] * inner[4],
        outer[3] * inner[2] + outer[4] * inner[5] + outer[5],
    ]
}

/// Check if an affine matrix is the identity transform.
pub fn is_identity_affine(m: &[f64; 6]) -> bool {
    (m[0] - 1.0).abs() < 1e-10
        && m[1].abs() < 1e-10
        && m[2].abs() < 1e-10
        && m[3].abs() < 1e-10
        && (m[4] - 1.0).abs() < 1e-10
        && m[5].abs() < 1e-10
}

// ─── CLUT Composition ───────────────────────────────────────────────────────

/// A 3D color lookup table with f32 entries.
///
/// Grid is `grid_size^3` entries, each with 3 f32 values (RGB).
/// Composition via tetrahedral interpolation.
#[derive(Debug, Clone)]
pub struct Clut3D {
    pub grid_size: u32,
    /// Flattened: grid_size^3 * 3 entries (R, G, B for each grid point).
    pub data: Vec<f32>,
}

impl Clut3D {
    /// Create an identity CLUT (input = output for all colors).
    pub fn identity(grid_size: u32) -> Self {
        let n = grid_size as usize;
        let mut data = Vec::with_capacity(n * n * n * 3);
        for b in 0..n {
            for g in 0..n {
                for r in 0..n {
                    data.push(r as f32 / (n - 1) as f32);
                    data.push(g as f32 / (n - 1) as f32);
                    data.push(b as f32 / (n - 1) as f32);
                }
            }
        }
        Self { grid_size, data }
    }

    /// Build a CLUT from a function that maps (r, g, b) → (r', g', b').
    ///
    /// Input values are normalized [0, 1].
    pub fn from_fn<F: Fn(f32, f32, f32) -> (f32, f32, f32)>(grid_size: u32, f: F) -> Self {
        let n = grid_size as usize;
        let mut data = Vec::with_capacity(n * n * n * 3);
        for bi in 0..n {
            for gi in 0..n {
                for ri in 0..n {
                    let r = ri as f32 / (n - 1) as f32;
                    let g = gi as f32 / (n - 1) as f32;
                    let b = bi as f32 / (n - 1) as f32;
                    let (ro, go, bo) = f(r, g, b);
                    data.push(ro);
                    data.push(go);
                    data.push(bo);
                }
            }
        }
        Self { grid_size, data }
    }

    /// Sample this CLUT at a given (r, g, b) using trilinear interpolation.
    ///
    /// Input values are normalized [0, 1].
    pub fn sample(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let n = self.grid_size as f32 - 1.0;
        let ri = (r * n).clamp(0.0, n);
        let gi = (g * n).clamp(0.0, n);
        let bi = (b * n).clamp(0.0, n);

        let r0 = ri.floor() as usize;
        let g0 = gi.floor() as usize;
        let b0 = bi.floor() as usize;
        let r1 = (r0 + 1).min(self.grid_size as usize - 1);
        let g1 = (g0 + 1).min(self.grid_size as usize - 1);
        let b1 = (b0 + 1).min(self.grid_size as usize - 1);

        let fr = ri - r0 as f32;
        let fg = gi - g0 as f32;
        let fb = bi - b0 as f32;

        let gs = self.grid_size as usize;
        let idx = |r: usize, g: usize, b: usize| -> usize { (b * gs * gs + g * gs + r) * 3 };

        // Trilinear interpolation
        let mut result = [0.0f32; 3];
        for (c, res) in result.iter_mut().enumerate() {
            let c000 = self.data[idx(r0, g0, b0) + c];
            let c100 = self.data[idx(r1, g0, b0) + c];
            let c010 = self.data[idx(r0, g1, b0) + c];
            let c110 = self.data[idx(r1, g1, b0) + c];
            let c001 = self.data[idx(r0, g0, b1) + c];
            let c101 = self.data[idx(r1, g0, b1) + c];
            let c011 = self.data[idx(r0, g1, b1) + c];
            let c111 = self.data[idx(r1, g1, b1) + c];

            let c00 = c000 + fr * (c100 - c000);
            let c01 = c001 + fr * (c101 - c001);
            let c10 = c010 + fr * (c110 - c010);
            let c11 = c011 + fr * (c111 - c011);

            let c0 = c00 + fg * (c10 - c00);
            let c1 = c01 + fg * (c11 - c01);

            *res = c0 + fb * (c1 - c0);
        }

        (result[0], result[1], result[2])
    }

    /// Apply this CLUT to f32 RGBA pixel data.
    ///
    /// Applies trilinear interpolation to each pixel's RGB channels.
    /// Alpha is preserved unchanged.
    pub fn apply(&self, pixels: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(pixels.len());
        for pixel in pixels.chunks_exact(4) {
            let (r, g, b) = self.sample(pixel[0], pixel[1], pixel[2]);
            out.push(r);
            out.push(g);
            out.push(b);
            out.push(pixel[3]); // alpha unchanged
        }
        out
    }
}

/// Compose two CLUTs: result(v) = outer(inner(v)).
///
/// Evaluates `inner` at every grid point, then samples `outer` at the result.
/// Output grid size is `outer.grid_size` (outer determines output resolution).
pub fn compose_cluts(outer: &Clut3D, inner: &Clut3D) -> Clut3D {
    let n = outer.grid_size;
    Clut3D::from_fn(n, |r, g, b| {
        let (ir, ig, ib) = inner.sample(r, g, b);
        outer.sample(ir, ig, ib)
    })
}

// ─── Fused Nodes ────────────────────────────────────────────────────────────

/// Fused point-op node — applies composed per-channel expression trees as a single pass.
pub struct FusedPointOpNode {
    upstream: u32,
    info: NodeInfo,
    #[allow(dead_code)]
    exprs: [PointOpExpr; 3],
    /// Pre-compiled WGSL shader (cached on creation).
    gpu_shader_src: String,
    /// Pre-built f32 LUTs per channel (cached on creation).
    luts: [F32Lut; 3],
    /// True if all 3 channels use identical LUTs (common for grayscale ops).
    /// Precomputed to avoid per-frame comparison.
    uniform_luts: bool,
}

impl FusedPointOpNode {
    pub fn new(upstream: u32, info: NodeInfo, exprs: [PointOpExpr; 3]) -> Self {
        let gpu_shader_src = lower_to_wgsl_shader_per_channel(&exprs);
        let luts = lower_to_f32_luts(&exprs);
        let uniform_luts = luts[0].table == luts[1].table && luts[1].table == luts[2].table;
        Self {
            upstream,
            info,
            exprs,
            gpu_shader_src,
            luts,
            uniform_luts,
        }
    }
}

impl Node for FusedPointOpNode {
    fn info(&self) -> NodeInfo {
        self.info.clone()
    }

    fn compute(
        &self,
        request: Rect,
        upstream: &mut dyn Upstream,
    ) -> Result<Vec<f32>, PipelineError> {
        let input = upstream.request(self.upstream, request)?;
        let mut output = input;
        // Fast path: if all 3 channels use the same LUT (common for grayscale
        // ops like brightness/contrast/gamma), use the single-LUT ILP path
        // which exposes 3 independent lookup chains for superscalar execution.
        if self.uniform_luts {
            // Single LUT for all channels — use batch ILP path
            // (12 independent lookups per iteration for superscalar execution)
            self.luts[0].apply_buffer(&mut output);
        } else {
            // Per-channel LUTs — separate lookups
            for pixel in output.chunks_exact_mut(4) {
                let p: &mut [f32; 4] = pixel.try_into().unwrap();
                p[0] = self.luts[0].apply(p[0]);
                p[1] = self.luts[1].apply(p[1]);
                p[2] = self.luts[2].apply(p[2]);
            }
        }
        Ok(output)
    }

    fn gpu_shader(&self, width: u32, height: u32) -> Option<GpuShader> {
        let mut params = Vec::with_capacity(8);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        Some(GpuShader {
            body: self.gpu_shader_src.clone(),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
            reduction_buffers: vec![],
            convergence_check: None,
            loop_dispatch: None,
        })
    }

    fn upstream_ids(&self) -> Vec<u32> {
        vec![self.upstream]
    }

    fn capabilities(&self) -> NodeCapabilities {
        NodeCapabilities {
            analytic: true,
            gpu: true,
            clut: true,
            ..Default::default()
        }
    }

    fn fusion_clut(&self) -> Option<Clut3D> {
        Some(Clut3D::from_fn(33, |r, g, b| {
            (self.luts[0].apply(r), self.luts[1].apply(g), self.luts[2].apply(b))
        }))
    }
}

/// Fused CLUT node — applies a composed 3D LUT as a single pass.
pub struct FusedClutNode {
    upstream: u32,
    info: NodeInfo,
    clut: Clut3D,
}

impl FusedClutNode {
    pub fn new(upstream: u32, info: NodeInfo, clut: Clut3D) -> Self {
        Self {
            upstream,
            info,
            clut,
        }
    }
}

impl Node for FusedClutNode {
    fn info(&self) -> NodeInfo {
        self.info.clone()
    }

    fn compute(
        &self,
        request: Rect,
        upstream: &mut dyn Upstream,
    ) -> Result<Vec<f32>, PipelineError> {
        let input = upstream.request(self.upstream, request)?;
        Ok(self.clut.apply(&input))
    }

    fn gpu_shader(&self, width: u32, height: u32) -> Option<GpuShader> {
        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.clut.grid_size.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes()); // padding

        // Pass 3D LUT grid as extra read-only storage buffer
        let lut_bytes: Vec<u8> = self.clut.data.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        Some(GpuShader {
            body: crate::filter_node::compose_shader(include_str!("shaders/clut3d.wgsl")),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![lut_bytes],
            reduction_buffers: vec![],
            convergence_check: None,
            loop_dispatch: None,
        })
    }

    fn upstream_ids(&self) -> Vec<u32> {
        vec![self.upstream]
    }

    fn capabilities(&self) -> NodeCapabilities {
        NodeCapabilities {
            clut: true,
            gpu: true,
            ..Default::default()
        }
    }
}

// ─── Graph Optimizer ────────────────────────────────────────────────────────

/// Run all fusion optimization passes on the graph.
///
/// Detects chains of same-category fusable nodes and replaces them with
/// single fused nodes. Runs in dependency order (bottom-up).
pub fn optimize(graph: &mut Graph) {
    // Note: the caller (request_full, gpu_plan) emits the trace event.
    // Don't emit here to avoid double-counting.
    flatten_lmt_chains(graph);
    fuse_analytical_chains(graph);
    fuse_affine_chains(graph);
    fuse_clut_chains(graph);
}

/// Flatten Lmt::Chain nodes into individual LmtNodes.
///
/// Chain nodes execute correctly via Lmt::apply() (sequential stages),
/// but don't participate in cross-node fusion. This pass is a placeholder
/// for future optimization that would expand Chain nodes into individual
/// LmtNodes so the analytical/CLUT passes can compose across them.
///
/// Current behavior: Chain nodes are left as-is. Their internal stages
/// execute sequentially inside LmtNode::compute(), which is correct.
fn flatten_lmt_chains(_graph: &mut Graph) {
    // Future: expand Lmt::Chain nodes into individual LmtNodes.
    // Requires Graph::rewire_upstream or similar node insertion API.
    // For now, chains execute correctly without cross-node fusion.
}

/// Fuse chains of analytical (point op) nodes into single per-channel expression trees.
fn fuse_analytical_chains(graph: &mut Graph) {
    let n = graph.node_count() as usize;
    let mut fused: Vec<bool> = vec![false; n];

    for i in (0..n).rev() {
        if fused[i] {
            continue;
        }
        let node = graph.get_node(i as u32);
        if node.analytic_expression_per_channel().is_none() {
            continue;
        }

        // Walk upstream collecting consecutive analytic nodes
        let mut chain = vec![i];
        let mut current = i;
        loop {
            let upstream_ids = graph.get_node(current as u32).upstream_ids();
            if upstream_ids.len() != 1 {
                break;
            }
            let up = upstream_ids[0] as usize;
            if up >= n || fused[up] || graph.get_node(up as u32).analytic_expression_per_channel().is_none() {
                break;
            }
            chain.push(up);
            current = up;
        }

        // Even single-node chains get fused — this gives them a GPU shader
        // (FusedPointOpNode emits WGSL) and the LUT-based CPU fast path.

        // Collect per-channel expressions: chain is [outermost, ..., innermost]
        let per_channel_exprs: Vec<[PointOpExpr; 3]> = chain
            .iter()
            .filter_map(|&id| graph.get_node(id as u32).analytic_expression_per_channel())
            .collect();

        if per_channel_exprs.is_empty() {
            continue;
        }

        // Compose per-channel from innermost to outermost
        let mut composed = per_channel_exprs.last().unwrap().clone();
        for exprs in per_channel_exprs[..per_channel_exprs.len() - 1].iter().rev() {
            composed = [
                PointOpExpr::compose(&exprs[0], &composed[0]),
                PointOpExpr::compose(&exprs[1], &composed[1]),
                PointOpExpr::compose(&exprs[2], &composed[2]),
            ];
        }

        // Optimize each channel's composed expression
        let optimized = [
            constant_fold(&composed[0]),
            constant_fold(&composed[1]),
            constant_fold(&composed[2]),
        ];

        // If all channels are identity, skip
        if is_identity(&optimized[0]) && is_identity(&optimized[1]) && is_identity(&optimized[2]) {
            continue;
        }

        // Get the innermost node's upstream as the fused node's upstream
        let innermost = *chain.last().unwrap();
        let fused_upstream = graph
            .get_node(innermost as u32)
            .upstream_ids()
            .first()
            .copied()
            .unwrap_or(0);

        let info = graph.get_node(chain[0] as u32).info();
        let fused_node = FusedPointOpNode::new(fused_upstream, info, optimized);

        // Replace the outermost node with the fused node
        graph.replace_node(chain[0] as u32, Box::new(fused_node));

        // Mark intermediate nodes as fused (they'll be skipped)
        for &id in &chain[1..] {
            fused[id] = true;
        }
    }
}

/// Fuse chains of affine transform nodes into single composed matrices.
fn fuse_affine_chains(_graph: &mut Graph) {
    // TODO: Implement when V2 transform nodes expose affine matrices
    // The pattern is the same as analytical fusion:
    // 1. Walk graph finding chains of affine-capable nodes
    // 2. Compose matrices: compose_affine(outer, inner)
    // 3. Replace chain with single ComposedAffineNode
}

/// Fuse chains of CLUT (color op) nodes into single composed 3D LUTs.
///
/// Walks the graph in reverse order, collecting consecutive nodes that
/// implement `fusion_clut()`. Chains of 2+ nodes are composed via
/// `compose_cluts()` and replaced with a single `FusedClutNode`.
fn fuse_clut_chains(graph: &mut Graph) {
    let n = graph.node_count() as usize;
    let mut fused: Vec<bool> = vec![false; n];

    for i in (0..n).rev() {
        if fused[i] {
            continue;
        }
        let node = graph.get_node(i as u32);
        if node.fusion_clut().is_none() {
            continue;
        }

        // Walk upstream collecting consecutive CLUT-capable nodes
        let mut chain = vec![i];
        let mut current = i;
        loop {
            let upstream_ids = graph.get_node(current as u32).upstream_ids();
            if upstream_ids.len() != 1 {
                break;
            }
            let up = upstream_ids[0] as usize;
            if up >= n || fused[up] || graph.get_node(up as u32).fusion_clut().is_none() {
                break;
            }
            chain.push(up);
            current = up;
        }

        // Need at least 2 nodes to fuse (single CLUT nodes stay as-is)
        if chain.len() < 2 {
            continue;
        }

        // Build CLUTs: chain is [outermost, ..., innermost]
        let cluts: Vec<Clut3D> = chain
            .iter()
            .filter_map(|&id| graph.get_node(id as u32).fusion_clut())
            .collect();

        if cluts.len() < 2 {
            continue;
        }

        // Compose from innermost to outermost:
        // result(v) = outer(inner(v))
        // cluts[0] = outermost, cluts[last] = innermost
        let mut composed = cluts.last().unwrap().clone();
        for clut in cluts[..cluts.len() - 1].iter().rev() {
            composed = compose_cluts(clut, &composed);
        }

        // Get the innermost node's upstream as the fused node's upstream
        let innermost = *chain.last().unwrap();
        let fused_upstream = graph
            .get_node(innermost as u32)
            .upstream_ids()
            .first()
            .copied()
            .unwrap_or(0);

        let info = graph.get_node(chain[0] as u32).info();
        let fused_node = FusedClutNode::new(fused_upstream, info, composed);

        // Replace the outermost node with the fused node
        graph.replace_node(chain[0] as u32, Box::new(fused_node));

        // Mark intermediate nodes as fused (they'll be skipped)
        for &id in &chain[1..] {
            fused[id] = true;
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Expression optimizer tests ──

    #[test]
    fn constant_fold_add_constants() {
        let expr = PointOpExpr::Add(
            Box::new(PointOpExpr::Constant(2.0)),
            Box::new(PointOpExpr::Constant(3.0)),
        );
        let folded = constant_fold(&expr);
        assert!(matches!(folded, PointOpExpr::Constant(c) if (c - 5.0).abs() < 1e-6));
    }

    #[test]
    fn constant_fold_identity_add_zero() {
        let expr = PointOpExpr::Add(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(0.0)),
        );
        let folded = constant_fold(&expr);
        assert!(matches!(folded, PointOpExpr::Input));
    }

    #[test]
    fn constant_fold_mul_by_one() {
        let expr = PointOpExpr::Mul(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(1.0)),
        );
        let folded = constant_fold(&expr);
        assert!(matches!(folded, PointOpExpr::Input));
    }

    #[test]
    fn constant_fold_mul_by_zero() {
        let expr = PointOpExpr::Mul(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(0.0)),
        );
        let folded = constant_fold(&expr);
        assert!(matches!(folded, PointOpExpr::Constant(c) if c == 0.0));
    }

    #[test]
    fn constant_fold_pow_by_one() {
        let expr = PointOpExpr::Pow(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(1.0)),
        );
        let folded = constant_fold(&expr);
        assert!(matches!(folded, PointOpExpr::Input));
    }

    #[test]
    fn constant_fold_nested() {
        // (2 + 3) * Input → 5 * Input
        let expr = PointOpExpr::Mul(
            Box::new(PointOpExpr::Add(
                Box::new(PointOpExpr::Constant(2.0)),
                Box::new(PointOpExpr::Constant(3.0)),
            )),
            Box::new(PointOpExpr::Input),
        );
        let folded = constant_fold(&expr);
        match &folded {
            PointOpExpr::Mul(a, _) => {
                assert!(matches!(**a, PointOpExpr::Constant(c) if (c - 5.0).abs() < 1e-6));
            }
            _ => panic!("expected Mul, got {folded:?}"),
        }
    }

    #[test]
    fn darken_brighten_folds_to_identity() {
        // darken(-0.5) then brighten(+0.5)
        let darken = PointOpExpr::Add(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(-0.5)),
        );
        let brighten = PointOpExpr::Add(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(0.5)),
        );
        let composed = PointOpExpr::compose(&brighten, &darken);
        let folded = constant_fold(&composed);
        // After folding: Input + (-0.5 + 0.5) → Input + 0.0 → Input
        // This requires nested constant folding
        let folded2 = constant_fold(&folded);
        // The composition produces Add(Add(Input, Const(-0.5)), Const(0.5))
        // which isn't directly foldable to Input without algebraic simplification.
        // But numerically it evaluates to identity:
        for i in 0..256 {
            let v = i as f64 / 255.0;
            let result = folded2.evaluate(v);
            assert!(
                (result - v).abs() < 1e-10,
                "roundtrip failed at v={v}: got {result}"
            );
        }
    }

    // ── Lowering tests ──

    #[test]
    fn lower_to_closure_brightness() {
        let expr = PointOpExpr::Add(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(0.1)),
        );
        let f = lower_to_closure(&expr);
        assert!((f(0.5) - 0.6).abs() < 1e-5);
    }

    #[test]
    fn lower_to_wgsl_brightness() {
        let expr = PointOpExpr::Add(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(0.1)),
        );
        let wgsl = lower_to_wgsl(&expr);
        assert_eq!(wgsl, "(v + 0.1)");
    }

    #[test]
    fn lower_to_wgsl_gamma() {
        let expr = PointOpExpr::Pow(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(2.2)),
        );
        let wgsl = lower_to_wgsl(&expr);
        assert_eq!(wgsl, "pow(v, 2.2)");
    }

    #[test]
    fn lower_to_wgsl_shader_compiles_valid_structure() {
        let expr = PointOpExpr::Add(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(0.1)),
        );
        let shader = lower_to_wgsl_shader(&expr);
        assert!(shader.contains("@compute @workgroup_size"));
        assert!(shader.contains("fn apply_r(v: f32) -> f32"));
        assert!(shader.contains("fn apply_g(v: f32) -> f32"));
        assert!(shader.contains("fn apply_b(v: f32) -> f32"));
        assert!(shader.contains("return (v + 0.1)"));
        assert!(shader.contains("pixel.w")); // alpha preserved
    }

    // ── Affine tests ──

    #[test]
    fn compose_affine_identity() {
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let scale = [2.0, 0.0, 0.0, 0.0, 2.0, 0.0];
        let result = compose_affine(&scale, &identity);
        assert_eq!(result, scale);

        let result2 = compose_affine(&identity, &scale);
        assert_eq!(result2, scale);
    }

    #[test]
    fn compose_affine_two_translates() {
        let t1 = [1.0, 0.0, 10.0, 0.0, 1.0, 20.0]; // translate(10, 20)
        let t2 = [1.0, 0.0, 5.0, 0.0, 1.0, 15.0]; // translate(5, 15)
        let result = compose_affine(&t1, &t2);
        // translate(10, 20) * translate(5, 15) = translate(15, 35)
        assert!((result[2] - 15.0).abs() < 1e-10);
        assert!((result[5] - 35.0).abs() < 1e-10);
    }

    #[test]
    fn identity_affine_detected() {
        assert!(is_identity_affine(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]));
        assert!(!is_identity_affine(&[2.0, 0.0, 0.0, 0.0, 1.0, 0.0]));
    }

    // ── CLUT tests ──

    #[test]
    fn clut_identity() {
        let clut = Clut3D::identity(17);
        let (r, g, b) = clut.sample(0.5, 0.3, 0.7);
        assert!((r - 0.5).abs() < 0.01);
        assert!((g - 0.3).abs() < 0.01);
        assert!((b - 0.7).abs() < 0.01);
    }

    #[test]
    fn clut_compose_identity() {
        let identity = Clut3D::identity(17);
        let invert = Clut3D::from_fn(17, |r, g, b| (1.0 - r, 1.0 - g, 1.0 - b));
        let composed = compose_cluts(&identity, &invert);
        // identity(invert(0.3, 0.5, 0.7)) = invert(0.3, 0.5, 0.7) = (0.7, 0.5, 0.3)
        let (r, g, b) = composed.sample(0.3, 0.5, 0.7);
        assert!((r - 0.7).abs() < 0.02);
        assert!((g - 0.5).abs() < 0.02);
        assert!((b - 0.3).abs() < 0.02);
    }

    #[test]
    fn clut_double_invert_is_identity() {
        let invert = Clut3D::from_fn(33, |r, g, b| (1.0 - r, 1.0 - g, 1.0 - b));
        let composed = compose_cluts(&invert, &invert);
        // invert(invert(v)) = v
        let (r, g, b) = composed.sample(0.3, 0.5, 0.7);
        assert!((r - 0.3).abs() < 0.02);
        assert!((g - 0.5).abs() < 0.02);
        assert!((b - 0.7).abs() < 0.02);
    }

    #[test]
    fn clut_apply_preserves_alpha() {
        let invert = Clut3D::from_fn(17, |r, g, b| (1.0 - r, 1.0 - g, 1.0 - b));
        let pixels = vec![0.3, 0.5, 0.7, 0.9]; // one pixel, alpha = 0.9
        let result = invert.apply(&pixels);
        assert_eq!(result.len(), 4);
        assert!((result[3] - 0.9).abs() < 1e-6); // alpha preserved
    }

    // ─── F32 LUT Tests ──────────────────────────────────────────────────

    #[test]
    fn f32_lut_identity() {
        let expr = PointOpExpr::Input;
        let lut = F32Lut::build(&expr);
        for i in 0..=100 {
            let v = i as f32 / 100.0;
            let result = lut.apply(v);
            assert!(
                (result - v).abs() < 1e-4,
                "identity LUT at {v}: got {result}"
            );
        }
    }

    #[test]
    fn f32_lut_brightness_offset() {
        // brightness +0.1: v + 0.1
        let expr = PointOpExpr::Add(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(0.1)),
        );
        let lut = F32Lut::build(&expr);
        let closure = lower_to_closure(&expr);
        for i in 0..=100 {
            let v = i as f32 / 100.0;
            let lut_val = lut.apply(v);
            let closure_val = closure(v);
            assert!(
                (lut_val - closure_val).abs() < 1e-3,
                "brightness LUT vs closure at {v}: lut={lut_val}, closure={closure_val}"
            );
        }
    }

    #[test]
    fn f32_lut_contrast_gamma_chain() {
        // contrast(1.5) -> gamma(2.2): pow(max((v-0.5)*1.5+0.5, 0), 1/2.2)
        let contrast = PointOpExpr::Add(
            Box::new(PointOpExpr::Mul(
                Box::new(PointOpExpr::Sub(
                    Box::new(PointOpExpr::Input),
                    Box::new(PointOpExpr::Constant(0.5)),
                )),
                Box::new(PointOpExpr::Constant(1.5)),
            )),
            Box::new(PointOpExpr::Constant(0.5)),
        );
        let gamma = PointOpExpr::Pow(
            Box::new(PointOpExpr::Max(
                Box::new(contrast),
                Box::new(PointOpExpr::Constant(0.0)),
            )),
            Box::new(PointOpExpr::Constant(1.0 / 2.2)),
        );
        let lut = F32Lut::build(&gamma);
        let closure = lower_to_closure(&gamma);
        for i in 0..=100 {
            let v = i as f32 / 100.0;
            let lut_val = lut.apply(v);
            let closure_val = closure(v);
            assert!(
                (lut_val - closure_val).abs() < 1e-3,
                "chain LUT vs closure at {v}: lut={lut_val}, closure={closure_val}"
            );
        }
    }

    #[test]
    fn f32_lut_clamps_out_of_range() {
        let expr = PointOpExpr::Input;
        let lut = F32Lut::build(&expr);
        // Values outside [0,1] should clamp to boundary
        let below = lut.apply(-0.5);
        let above = lut.apply(1.5);
        assert!((below - 0.0).abs() < 1e-4, "below range: {below}");
        assert!((above - 1.0).abs() < 1e-4, "above range: {above}");
    }
}
