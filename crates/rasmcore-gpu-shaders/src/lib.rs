//! Shared WGSL compute shader fragments and composition API.
//!
//! Provides reusable shader functions (pixel pack/unpack, bilinear sampling)
//! as string constants, plus composition helpers to build complete shaders
//! from fragments and filter-specific body code.
//!
//! # Convention
//!
//! All shaders follow the standard binding layout:
//! - `@group(0) @binding(0)` — input storage buffer (read-only)
//! - `@group(0) @binding(1)` — output storage buffer (read-write)
//! - `@group(0) @binding(2)` — uniform params
//! - `@group(0) @binding(3+)` — extra storage buffers
//!
//! Params struct must have `width: u32` and `height: u32` as the first
//! two fields for `sample_bilinear` compatibility.

/// Pixel pack/unpack functions for RGBA8 as packed u32.
///
/// Provides `fn unpack(pixel: u32) -> vec4<f32>` and
/// `fn pack(color: vec4<f32>) -> u32`.
pub const PIXEL_OPS: &str = include_str!("wgsl/pixel_ops.wgsl");

/// Bilinear sampling from the input buffer.
///
/// Provides `fn sample_bilinear(fx: f32, fy: f32) -> vec4<f32>`.
///
/// **Requires:** `PIXEL_OPS` must be included first (uses `unpack`).
/// **Requires:** Params struct with `width` and `height` as first two fields.
/// **Requires:** `@group(0) @binding(0) var<storage, read> input: array<u32>`.
pub const SAMPLE_BILINEAR: &str = include_str!("wgsl/sample_bilinear.wgsl");

/// Identity pack/unpack for f32 vec4 buffers.
///
/// Provides `fn unpack_f32(pixel: vec4<f32>) -> vec4<f32>` and
/// `fn pack_f32(color: vec4<f32>) -> vec4<f32>`.
///
/// Used with `array<vec4<f32>>` storage buffers — no conversion needed.
pub const PIXEL_OPS_F32: &str = include_str!("wgsl/pixel_ops_f32.wgsl");

/// Bilinear sampling from f32 vec4 input buffer.
///
/// Provides `fn sample_bilinear_f32(fx: f32, fy: f32) -> vec4<f32>`.
///
/// **Requires:** Params struct with `width` and `height` as first two fields.
/// **Requires:** `@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>`.
pub const SAMPLE_BILINEAR_F32: &str = include_str!("wgsl/sample_bilinear_f32.wgsl");

/// Compose a complete shader from library fragments and a filter body.
///
/// Fragments are prepended in order, followed by the body. The body should
/// contain the `Params` struct, binding declarations, and entry point function.
///
/// ```rust
/// let shader = rasmcore_gpu_shaders::compose(
///     &[rasmcore_gpu_shaders::PIXEL_OPS],
///     "struct Params { ... }\n@compute fn main(...) { ... }",
/// );
/// ```
pub fn compose(fragments: &[&str], body: &str) -> String {
    let total_len: usize = fragments.iter().map(|f| f.len() + 1).sum::<usize>() + body.len();
    let mut src = String::with_capacity(total_len);
    for fragment in fragments {
        src.push_str(fragment);
        src.push('\n');
    }
    src.push_str(body);
    src
}

/// Compose a shader that needs pixel pack/unpack only.
///
/// Use for filters that operate per-pixel without sampling neighbors
/// (blur, sharpen via kernel, bilateral, median, point ops).
pub fn with_pixel_ops(body: &str) -> String {
    compose(&[PIXEL_OPS], body)
}

/// Compose a shader that needs pixel ops + bilinear sampling.
///
/// Use for distortion filters, motion blur, affine transforms — anything
/// that samples the input at non-integer coordinates.
pub fn with_sampling(body: &str) -> String {
    compose(&[PIXEL_OPS, SAMPLE_BILINEAR], body)
}

/// Compose an f32-format shader that needs identity pack/unpack only.
///
/// Use for filters operating on `array<vec4<f32>>` storage buffers.
/// The shader body should use `unpack_f32()`/`pack_f32()` and declare
/// `input: array<vec4<f32>>` / `output: array<vec4<f32>>`.
pub fn with_pixel_ops_f32(body: &str) -> String {
    compose(&[PIXEL_OPS_F32], body)
}

/// Compose an f32-format shader that needs identity ops + bilinear sampling.
///
/// Use for distortion/transform filters operating on f32 buffers.
/// The shader body should use `sample_bilinear_f32()` and `pack_f32()`.
pub fn with_sampling_f32(body: &str) -> String {
    compose(&[PIXEL_OPS_F32, SAMPLE_BILINEAR_F32], body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compose_concatenates_fragments_and_body() {
        let result = compose(&["// frag1", "// frag2"], "// body");
        assert_eq!(result, "// frag1\n// frag2\n// body");
    }

    #[test]
    fn with_pixel_ops_includes_pack_unpack() {
        let shader = with_pixel_ops("struct Params { width: u32 }");
        assert!(shader.contains("fn unpack("));
        assert!(shader.contains("fn pack("));
        assert!(shader.contains("struct Params"));
    }

    #[test]
    fn with_sampling_includes_all() {
        let shader = with_sampling("struct Params { width: u32, height: u32 }");
        assert!(shader.contains("fn unpack("));
        assert!(shader.contains("fn pack("));
        assert!(shader.contains("fn sample_bilinear("));
        assert!(shader.contains("struct Params"));
    }

    #[test]
    fn pixel_ops_fragment_is_valid() {
        assert!(PIXEL_OPS.contains("fn unpack(pixel: u32) -> vec4<f32>"));
        assert!(PIXEL_OPS.contains("fn pack(color: vec4<f32>) -> u32"));
    }

    #[test]
    fn sample_bilinear_fragment_is_valid() {
        assert!(SAMPLE_BILINEAR.contains("fn sample_bilinear(fx: f32, fy: f32) -> vec4<f32>"));
    }

    #[test]
    fn pixel_ops_f32_fragment_is_valid() {
        assert!(PIXEL_OPS_F32.contains("fn unpack_f32(pixel: vec4<f32>) -> vec4<f32>"));
        assert!(PIXEL_OPS_F32.contains("fn pack_f32(color: vec4<f32>) -> vec4<f32>"));
    }

    #[test]
    fn sample_bilinear_f32_fragment_is_valid() {
        assert!(SAMPLE_BILINEAR_F32.contains("fn sample_bilinear_f32(fx: f32, fy: f32) -> vec4<f32>"));
    }

    #[test]
    fn with_pixel_ops_f32_includes_identity() {
        let shader = with_pixel_ops_f32("struct Params { width: u32 }");
        assert!(shader.contains("fn unpack_f32("));
        assert!(shader.contains("fn pack_f32("));
        assert!(shader.contains("struct Params"));
    }

    #[test]
    fn with_sampling_f32_includes_all() {
        let shader = with_sampling_f32("struct Params { width: u32, height: u32 }");
        assert!(shader.contains("fn unpack_f32("));
        assert!(shader.contains("fn pack_f32("));
        assert!(shader.contains("fn sample_bilinear_f32("));
        assert!(shader.contains("struct Params"));
    }
}
