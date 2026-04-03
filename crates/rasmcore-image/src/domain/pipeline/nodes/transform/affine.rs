//! Affine composition trait, helpers, and composed affine node.

use crate::domain::error::ImageError;
use crate::domain::pipeline::graph::{AccessPattern, ImageNode};
use crate::domain::transform;
use crate::domain::types::*;
use rasmcore_pipeline::{GpuCapable, GpuOp, Rect};

use std::sync::LazyLock;

static AFFINE_SHADER_F32: LazyLock<String> = LazyLock::new(|| {
    rasmcore_gpu_shaders::with_pixel_ops_f32(include_str!("../../../../shaders/affine_resample_f32.wgsl"))
});

/// Trait for transform nodes that can be expressed as a 2x3 affine matrix.
///
/// Implementors return a forward mapping matrix [a, b, tx, c, d, ty] where:
///   x' = a*x + b*y + tx
///   y' = c*x + d*y + ty
///
/// The pipeline optimizer composes consecutive AffineOp nodes into a single
/// matrix and resamples once, eliminating multi-pass interpolation artifacts.
pub trait AffineOp {
    /// Return the 2x3 affine matrix and output dimensions for this transform.
    fn to_affine(&self) -> ([f64; 6], u32, u32);
}

/// Compose two 2x3 affine matrices: result = outer * inner.
///
/// Treats each as a 3x3 homogeneous matrix with bottom row [0, 0, 1].
/// The composed matrix applies `inner` first, then `outer`.
pub fn compose_affine(outer: &[f64; 6], inner: &[f64; 6]) -> [f64; 6] {
    let [a1, b1, tx1, c1, d1, ty1] = *outer;
    let [a2, b2, tx2, c2, d2, ty2] = *inner;
    [
        a1 * a2 + b1 * c2,
        a1 * b2 + b1 * d2,
        a1 * tx2 + b1 * ty2 + tx1,
        c1 * a2 + d1 * c2,
        c1 * b2 + d1 * d2,
        c1 * tx2 + d1 * ty2 + ty1,
    ]
}

/// Compute bounding box dimensions after applying a 2x3 affine matrix
/// to a rectangle of given width/height originating at (0,0).
pub fn affine_output_dims(matrix: &[f64; 6], src_w: u32, src_h: u32) -> (u32, u32) {
    let [a, b, tx, c, d, ty] = *matrix;
    let w = src_w as f64;
    let h = src_h as f64;

    // Transform the four corners
    let corners = [
        (tx, ty),                                 // (0,0)
        (a * w + tx, c * w + ty),                 // (w,0)
        (b * h + tx, d * h + ty),                 // (0,h)
        (a * w + b * h + tx, c * w + d * h + ty), // (w,h)
    ];

    let min_x = corners.iter().map(|c| c.0).fold(f64::INFINITY, f64::min);
    let max_x = corners
        .iter()
        .map(|c| c.0)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = corners.iter().map(|c| c.1).fold(f64::INFINITY, f64::min);
    let max_y = corners
        .iter()
        .map(|c| c.1)
        .fold(f64::NEG_INFINITY, f64::max);

    let out_w = (max_x - min_x).round().max(1.0) as u32;
    let out_h = (max_y - min_y).round().max(1.0) as u32;
    (out_w, out_h)
}

/// Pipeline node that applies a single composed affine transform.
///
/// Created by the affine fusion optimizer when it detects a chain of
/// AffineOp nodes. Replaces the entire chain with one resample pass.
pub struct ComposedAffineNode {
    /// Upstream of the first node in the original chain.
    upstream: u32,
    /// Source image info (from the chain root's upstream).
    source_info: ImageInfo,
    /// The composed 2x3 affine matrix.
    matrix: [f64; 6],
    /// Output dimensions after the composed transform.
    out_width: u32,
    out_height: u32,
}

impl ComposedAffineNode {
    pub fn new(
        upstream: u32,
        source_info: ImageInfo,
        matrix: [f64; 6],
        out_width: u32,
        out_height: u32,
    ) -> Self {
        Self {
            upstream,
            source_info,
            matrix,
            out_width,
            out_height,
        }
    }
}

impl ImageNode for ComposedAffineNode {
    fn info(&self) -> ImageInfo {
        ImageInfo {
            width: self.out_width,
            height: self.out_height,
            ..self.source_info.clone()
        }
    }

    fn compute_region(
        &self,
        _request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        let full_src = Rect::new(0, 0, self.source_info.width, self.source_info.height);
        let src_pixels = upstream_fn(self.upstream, full_src)?;
        // Single resample with the composed affine matrix
        let bg = vec![0u8; 4]; // transparent/black background
        let result = transform::affine(
            &src_pixels,
            &self.source_info,
            &self.matrix,
            self.out_width,
            self.out_height,
            &bg,
        )?;
        Ok(result.pixels)
    }

    fn input_rect(&self, _output: Rect, _bounds_w: u32, _bounds_h: u32) -> Rect {
        Rect::new(0, 0, self.source_info.width, self.source_info.height)
    }

    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::RandomAccess
    }

    fn upstream_id(&self) -> Option<u32> {
        Some(self.upstream)
    }
}

impl GpuCapable for ComposedAffineNode {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<GpuOp>> {
        self.gpu_ops_with_format(width, height, rasmcore_pipeline::BufferFormat::F32Vec4)
    }

    fn gpu_ops_with_format(&self, _width: u32, _height: u32, buffer_format: rasmcore_pipeline::BufferFormat) -> Option<Vec<GpuOp>> {
        use rasmcore_pipeline::BufferFormat;

        // V2: F32Vec4 only
        if buffer_format != BufferFormat::F32Vec4 {
            return None;
        }

        // Compute inverse of the 2x3 affine matrix
        let [a, b, tx, c, d, ty] = self.matrix;
        let det = a * d - b * c;
        if det.abs() < 1e-10 {
            return None; // Degenerate matrix
        }
        let inv_det = 1.0 / det;
        let inv_a = (d * inv_det) as f32;
        let inv_b = (-b * inv_det) as f32;
        let inv_tx = ((b * ty - d * tx) * inv_det) as f32;
        let inv_c = (-c * inv_det) as f32;
        let inv_d = (a * inv_det) as f32;
        let inv_ty = ((c * tx - a * ty) * inv_det) as f32;

        let mut params = Vec::with_capacity(48);
        params.extend_from_slice(&self.source_info.width.to_le_bytes());
        params.extend_from_slice(&self.source_info.height.to_le_bytes());
        params.extend_from_slice(&self.out_width.to_le_bytes());
        params.extend_from_slice(&self.out_height.to_le_bytes());
        params.extend_from_slice(&inv_a.to_le_bytes());
        params.extend_from_slice(&inv_b.to_le_bytes());
        params.extend_from_slice(&inv_tx.to_le_bytes());
        params.extend_from_slice(&inv_c.to_le_bytes());
        params.extend_from_slice(&inv_d.to_le_bytes());
        params.extend_from_slice(&inv_ty.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes()); // _pad1
        params.extend_from_slice(&0u32.to_le_bytes()); // _pad2

        let shader = AFFINE_SHADER_F32.clone();

        Some(vec![GpuOp::Compute {
            shader,
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
            buffer_format,
        }])
    }
}
