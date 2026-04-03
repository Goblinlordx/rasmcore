//! Declarative macros for GPU filter registration.
//!
//! These macros generate `impl GpuFilter` blocks from shader source + param
//! layout declarations. Filters embed their WGSL source via these macros
//! rather than writing manual trait implementations.
//!
//! # Single-pass filter
//!
//! ```ignore
//! gpu_filter!(MyFilter, shader: MY_SHADER_WGSL, workgroup: [16, 16, 1],
//!     params(self, w, h) => [w, h, self.radius, 0u32]);
//! ```
//!
//! # Multi-pass filter (e.g., separable blur, reduction + apply)
//!
//! ```ignore
//! gpu_filter_multipass!(MyFilter, shader: MY_APPLY_WGSL, workgroup: [256, 1, 1],
//!     params(self, w, h) => [w, h, self.strength, 0u32],
//!     passes(self, w, h) => {
//!         let reduction = GpuReduction::histogram_256(256);
//!         let passes = reduction.build_passes(w, h);
//!         let apply = GpuShader::new(MY_APPLY_WGSL.to_string(), "main", [256, 1, 1], self.params(w, h))
//!             .with_reduction_buffers(vec![reduction.read_buffer(&passes)]);
//!         vec![passes.pass1, passes.pass2, apply]
//!     });
//! ```

/// Generate `impl GpuFilter` for a single-pass GPU filter.
///
/// `shader` — WGSL constant reference (e.g., `crate::gpu_shaders::spatial::GAUSSIAN_BLUR_H`)
/// `workgroup` — `[x, y, z]` workgroup dimensions
/// `params` — list of expressions serialized as little-endian bytes (auto-padded to 16-byte alignment)
#[macro_export]
macro_rules! gpu_filter {
    (
        $filter_type:ty,
        shader: $shader:expr,
        workgroup: [$wx:expr, $wy:expr, $wz:expr],
        params($self_:ident, $w:ident, $h:ident) => [$($param:expr),* $(,)?]
    ) => {
        impl $crate::ops::GpuFilter for $filter_type {
            fn shader_body(&self) -> &str { $shader }
            fn workgroup_size(&self) -> [u32; 3] { [$wx, $wy, $wz] }
            fn params(&self, $w: u32, $h: u32) -> Vec<u8> {
                let $self_ = self;
                let mut _buf = Vec::new();
                $(
                    _buf.extend_from_slice(&$param.to_le_bytes());
                )*
                // Pad to 16-byte alignment (WGSL uniform requirement)
                while _buf.len() % 16 != 0 {
                    _buf.extend_from_slice(&0u32.to_le_bytes());
                }
                _buf
            }
        }
    };
}

/// Generate `impl GpuFilter` for a multi-pass GPU filter with custom `gpu_shaders()`.
///
/// Same as `gpu_filter!` but adds a `passes` block that returns `Vec<GpuShader>`.
/// The `shader`/`workgroup`/`params` define the default single-pass (used by `gpu_shader()`),
/// while `passes` overrides `gpu_shaders()` for the actual multi-pass chain.
#[macro_export]
macro_rules! gpu_filter_multipass {
    (
        $filter_type:ty,
        shader: $shader:expr,
        workgroup: [$wx:expr, $wy:expr, $wz:expr],
        params($self_:ident, $w:ident, $h:ident) => [$($param:expr),* $(,)?],
        passes($self2:ident, $w2:ident, $h2:ident) => $passes_body:expr
    ) => {
        impl $crate::ops::GpuFilter for $filter_type {
            fn shader_body(&self) -> &str { $shader }
            fn workgroup_size(&self) -> [u32; 3] { [$wx, $wy, $wz] }
            fn params(&self, $w: u32, $h: u32) -> Vec<u8> {
                let $self_ = self;
                let mut _buf = Vec::new();
                $(
                    _buf.extend_from_slice(&$param.to_le_bytes());
                )*
                while _buf.len() % 16 != 0 {
                    _buf.extend_from_slice(&0u32.to_le_bytes());
                }
                _buf
            }
            fn gpu_shaders(&self, $w2: u32, $h2: u32) -> Vec<$crate::node::GpuShader> {
                let $self2 = self;
                $passes_body
            }
        }
    };
}

/// Generate `impl GpuFilter` for a filter with only custom `gpu_shaders()` (no default single-pass).
///
/// Use when the multi-pass shader chain is the only execution path and there's no
/// meaningful single-pass representation.
#[macro_export]
macro_rules! gpu_filter_passes_only {
    (
        $filter_type:ty,
        passes($self_:ident, $w:ident, $h:ident) => $passes_body:expr
    ) => {
        impl $crate::ops::GpuFilter for $filter_type {
            fn shader_body(&self) -> &str { "" }
            fn workgroup_size(&self) -> [u32; 3] { [256, 1, 1] }
            fn params(&self, _w: u32, _h: u32) -> Vec<u8> { vec![] }
            fn gpu_shaders(&self, $w: u32, $h: u32) -> Vec<$crate::node::GpuShader> {
                let $self_ = self;
                $passes_body
            }
        }
    };
}
