//! Transform nodes — wrap existing domain transform operations.
//!
//! Nodes that represent affine transforms implement [`AffineOp`], enabling
//! the pipeline optimizer to compose consecutive transforms into a single
//! resample pass (better quality + fewer passes).

use crate::domain::error::ImageError;
use crate::domain::metadata::ExifOrientation;
use crate::domain::pipeline::graph::{AccessPattern, ImageNode};
use crate::domain::transform;
use crate::domain::types::*;
use rasmcore_pipeline::Rect;

// ─── Affine Composition ────────────────────────────────────────────────────

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

    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::RandomAccess
    }

    fn upstream_id(&self) -> Option<u32> {
        Some(self.upstream)
    }
}

/// Resize node.
pub struct ResizeNode {
    upstream: u32,
    target_width: u32,
    target_height: u32,
    filter: ResizeFilter,
    source_info: ImageInfo,
}

impl ResizeNode {
    pub fn new(
        upstream: u32,
        source_info: ImageInfo,
        width: u32,
        height: u32,
        filter: ResizeFilter,
    ) -> Self {
        Self {
            upstream,
            target_width: width,
            target_height: height,
            filter,
            source_info,
        }
    }
}

impl AffineOp for ResizeNode {
    fn to_affine(&self) -> ([f64; 6], u32, u32) {
        let sx = self.target_width as f64 / self.source_info.width as f64;
        let sy = self.target_height as f64 / self.source_info.height as f64;
        (
            [sx, 0.0, 0.0, 0.0, sy, 0.0],
            self.target_width,
            self.target_height,
        )
    }
}

impl ImageNode for ResizeNode {
    fn info(&self) -> ImageInfo {
        ImageInfo {
            width: self.target_width,
            height: self.target_height,
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
        let result = transform::resize(
            &src_pixels,
            &self.source_info,
            self.target_width,
            self.target_height,
            self.filter,
        )?;
        Ok(result.pixels)
    }

    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::LocalNeighborhood
    }

    fn upstream_id(&self) -> Option<u32> {
        Some(self.upstream)
    }

    fn as_affine_op(&self) -> Option<([f64; 6], u32, u32)> {
        Some(self.to_affine())
    }
}

/// Crop node.
pub struct CropNode {
    upstream: u32,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    source_info: ImageInfo,
}

impl CropNode {
    pub fn new(
        upstream: u32,
        source_info: ImageInfo,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            upstream,
            x,
            y,
            width,
            height,
            source_info,
        }
    }
}

impl AffineOp for CropNode {
    fn to_affine(&self) -> ([f64; 6], u32, u32) {
        // Crop = translation by (-x, -y), output is crop dimensions
        (
            [1.0, 0.0, -(self.x as f64), 0.0, 1.0, -(self.y as f64)],
            self.width,
            self.height,
        )
    }
}

impl ImageNode for CropNode {
    fn info(&self) -> ImageInfo {
        ImageInfo {
            width: self.width,
            height: self.height,
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
        let result = transform::crop(
            &src_pixels,
            &self.source_info,
            self.x,
            self.y,
            self.width,
            self.height,
        )?;
        Ok(result.pixels)
    }

    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::Sequential
    }

    fn upstream_id(&self) -> Option<u32> {
        Some(self.upstream)
    }

    fn as_affine_op(&self) -> Option<([f64; 6], u32, u32)> {
        Some(self.to_affine())
    }
}

/// Rotate node.
pub struct RotateNode {
    upstream: u32,
    rotation: Rotation,
    source_info: ImageInfo,
}

impl RotateNode {
    pub fn new(upstream: u32, source_info: ImageInfo, rotation: Rotation) -> Self {
        Self {
            upstream,
            rotation,
            source_info,
        }
    }
}

impl AffineOp for RotateNode {
    fn to_affine(&self) -> ([f64; 6], u32, u32) {
        let w = self.source_info.width as f64;
        let h = self.source_info.height as f64;
        let (mat, ow, oh) = match self.rotation {
            Rotation::R90 => (
                [0.0, -1.0, h, 1.0, 0.0, 0.0],
                self.source_info.height,
                self.source_info.width,
            ),
            Rotation::R180 => (
                [-1.0, 0.0, w, 0.0, -1.0, h],
                self.source_info.width,
                self.source_info.height,
            ),
            Rotation::R270 => (
                [0.0, 1.0, 0.0, -1.0, 0.0, w],
                self.source_info.height,
                self.source_info.width,
            ),
        };
        (mat, ow, oh)
    }
}

impl ImageNode for RotateNode {
    fn info(&self) -> ImageInfo {
        let (w, h) = match self.rotation {
            Rotation::R90 | Rotation::R270 => (self.source_info.height, self.source_info.width),
            Rotation::R180 => (self.source_info.width, self.source_info.height),
        };
        ImageInfo {
            width: w,
            height: h,
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
        let result = transform::rotate(&src_pixels, &self.source_info, self.rotation)?;
        Ok(result.pixels)
    }

    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::RandomAccess
    }

    fn upstream_id(&self) -> Option<u32> {
        Some(self.upstream)
    }

    fn as_affine_op(&self) -> Option<([f64; 6], u32, u32)> {
        Some(self.to_affine())
    }
}

/// Flip node.
pub struct FlipNode {
    upstream: u32,
    direction: FlipDirection,
    source_info: ImageInfo,
}

impl FlipNode {
    pub fn new(upstream: u32, source_info: ImageInfo, direction: FlipDirection) -> Self {
        Self {
            upstream,
            direction,
            source_info,
        }
    }
}

impl AffineOp for FlipNode {
    fn to_affine(&self) -> ([f64; 6], u32, u32) {
        let w = self.source_info.width as f64;
        let h = self.source_info.height as f64;
        let mat = match self.direction {
            FlipDirection::Horizontal => [-1.0, 0.0, w, 0.0, 1.0, 0.0],
            FlipDirection::Vertical => [1.0, 0.0, 0.0, 0.0, -1.0, h],
        };
        (mat, self.source_info.width, self.source_info.height)
    }
}

impl ImageNode for FlipNode {
    fn info(&self) -> ImageInfo {
        self.source_info.clone()
    }

    fn compute_region(
        &self,
        _request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        let full_src = Rect::new(0, 0, self.source_info.width, self.source_info.height);
        let src_pixels = upstream_fn(self.upstream, full_src)?;
        let result = transform::flip(&src_pixels, &self.source_info, self.direction)?;
        Ok(result.pixels)
    }

    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::RandomAccess
    }

    fn upstream_id(&self) -> Option<u32> {
        Some(self.upstream)
    }

    fn as_affine_op(&self) -> Option<([f64; 6], u32, u32)> {
        Some(self.to_affine())
    }
}

/// Auto-orient node — applies EXIF orientation transform.
pub struct AutoOrientNode {
    upstream: u32,
    orientation: ExifOrientation,
    source_info: ImageInfo,
}

impl AutoOrientNode {
    pub fn new(upstream: u32, source_info: ImageInfo, orientation: ExifOrientation) -> Self {
        Self {
            upstream,
            orientation,
            source_info,
        }
    }
}

impl ImageNode for AutoOrientNode {
    fn info(&self) -> ImageInfo {
        let (w, h) = match self.orientation {
            ExifOrientation::Normal
            | ExifOrientation::FlipHorizontal
            | ExifOrientation::Rotate180
            | ExifOrientation::FlipVertical => (self.source_info.width, self.source_info.height),
            ExifOrientation::Transpose
            | ExifOrientation::Rotate90
            | ExifOrientation::Transverse
            | ExifOrientation::Rotate270 => (self.source_info.height, self.source_info.width),
        };
        ImageInfo {
            width: w,
            height: h,
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
        let result = transform::auto_orient(&src_pixels, &self.source_info, self.orientation)?;
        Ok(result.pixels)
    }

    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::RandomAccess
    }
}
