//! Filter nodes — wrap existing domain filter operations.

use crate::domain::error::ImageError;
use crate::domain::filters;
use crate::domain::pipeline::graph::{AccessPattern, ImageNode};
use crate::domain::types::*;
use rasmcore_pipeline::{Overlap, Rect};

macro_rules! simple_filter_node {
    ($name:ident, $param_type:ty, $fn_name:ident, $overlap_val:expr, $doc:expr) => {
        #[doc = $doc]
        pub struct $name {
            upstream: u32,
            param: $param_type,
            source_info: ImageInfo,
        }

        impl $name {
            pub fn new(upstream: u32, source_info: ImageInfo, param: $param_type) -> Self {
                Self {
                    upstream,
                    param,
                    source_info,
                }
            }
        }

        impl ImageNode for $name {
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
                filters::$fn_name(&src_pixels, &self.source_info, self.param)
            }

            fn overlap(&self) -> Overlap {
                $overlap_val
            }
            fn access_pattern(&self) -> AccessPattern {
                AccessPattern::LocalNeighborhood
            }
        }
    };
}

simple_filter_node!(
    BlurNode,
    f32,
    blur,
    Overlap::uniform(10),
    "Gaussian blur node."
);
simple_filter_node!(
    SharpenNode,
    f32,
    sharpen,
    Overlap::uniform(2),
    "Sharpen node."
);
simple_filter_node!(
    BrightnessNode,
    f32,
    brightness,
    Overlap::zero(),
    "Brightness adjustment node."
);
simple_filter_node!(
    ContrastNode,
    f32,
    contrast,
    Overlap::zero(),
    "Contrast adjustment node."
);

// ─── Convolution filter nodes ───────────────────────────────────────────────

/// General convolution node with custom kernel.
pub struct ConvolveNode {
    upstream: u32,
    kernel: Vec<f32>,
    kw: usize,
    kh: usize,
    divisor: f32,
    source_info: ImageInfo,
}

impl ConvolveNode {
    pub fn new(
        upstream: u32,
        source_info: ImageInfo,
        kernel: Vec<f32>,
        kw: usize,
        kh: usize,
        divisor: f32,
    ) -> Self {
        Self {
            upstream,
            kernel,
            kw,
            kh,
            divisor,
            source_info,
        }
    }
}

impl ImageNode for ConvolveNode {
    fn info(&self) -> ImageInfo {
        self.source_info.clone()
    }
    fn compute_region(
        &self,
        _request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        let full = Rect::new(0, 0, self.source_info.width, self.source_info.height);
        let src = upstream_fn(self.upstream, full)?;
        filters::convolve(
            &src,
            &self.source_info,
            &self.kernel,
            self.kw,
            self.kh,
            self.divisor,
        )
    }
    fn overlap(&self) -> Overlap {
        Overlap::uniform((self.kw.max(self.kh) / 2) as u32)
    }
    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::LocalNeighborhood
    }
}

/// Median filter node.
pub struct MedianNode {
    upstream: u32,
    radius: u32,
    source_info: ImageInfo,
}

impl MedianNode {
    pub fn new(upstream: u32, source_info: ImageInfo, radius: u32) -> Self {
        Self {
            upstream,
            radius,
            source_info,
        }
    }
}

impl ImageNode for MedianNode {
    fn info(&self) -> ImageInfo {
        self.source_info.clone()
    }
    fn compute_region(
        &self,
        _request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        let full = Rect::new(0, 0, self.source_info.width, self.source_info.height);
        let src = upstream_fn(self.upstream, full)?;
        filters::median(&src, &self.source_info, self.radius)
    }
    fn overlap(&self) -> Overlap {
        Overlap::uniform(self.radius)
    }
    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::LocalNeighborhood
    }
}

/// Sobel edge detection node. Output is single-channel grayscale.
pub struct SobelNode {
    upstream: u32,
    source_info: ImageInfo,
}

impl SobelNode {
    pub fn new(upstream: u32, source_info: ImageInfo) -> Self {
        Self {
            upstream,
            source_info,
        }
    }
}

impl ImageNode for SobelNode {
    fn info(&self) -> ImageInfo {
        ImageInfo {
            format: PixelFormat::Gray8,
            ..self.source_info.clone()
        }
    }
    fn compute_region(
        &self,
        _request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        let full = Rect::new(0, 0, self.source_info.width, self.source_info.height);
        let src = upstream_fn(self.upstream, full)?;
        filters::sobel(&src, &self.source_info)
    }
    fn overlap(&self) -> Overlap {
        Overlap::uniform(1)
    }
    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::LocalNeighborhood
    }
}

/// Canny edge detection node. Output is single-channel binary (0/255).
pub struct CannyNode {
    upstream: u32,
    low_threshold: f32,
    high_threshold: f32,
    source_info: ImageInfo,
}

impl CannyNode {
    pub fn new(upstream: u32, source_info: ImageInfo, low: f32, high: f32) -> Self {
        Self {
            upstream,
            low_threshold: low,
            high_threshold: high,
            source_info,
        }
    }
}

impl ImageNode for CannyNode {
    fn info(&self) -> ImageInfo {
        ImageInfo {
            format: PixelFormat::Gray8,
            ..self.source_info.clone()
        }
    }
    fn compute_region(
        &self,
        _request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        let full = Rect::new(0, 0, self.source_info.width, self.source_info.height);
        let src = upstream_fn(self.upstream, full)?;
        filters::canny(
            &src,
            &self.source_info,
            self.low_threshold,
            self.high_threshold,
        )
    }
    fn overlap(&self) -> Overlap {
        Overlap::uniform(3)
    }
    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::LocalNeighborhood
    }
}

// ─── Point-op nodes (LUT-fusible) ───────────────────────────────────────────

/// A pipeline node for any composable LUT-based point operation.
///
/// At execution, builds the LUT from the stored `PointOp` and applies it in one pass.
/// The pipeline optimizer can detect consecutive `PointOpNode`s and replace them
/// with a single `FusedLutNode` that applies one pre-composed LUT.
pub struct PointOpNode {
    upstream: u32,
    op: crate::domain::point_ops::PointOp,
    source_info: ImageInfo,
}

impl PointOpNode {
    pub fn new(
        upstream: u32,
        source_info: ImageInfo,
        op: crate::domain::point_ops::PointOp,
    ) -> Self {
        Self {
            upstream,
            op,
            source_info,
        }
    }
}

impl ImageNode for PointOpNode {
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
        let lut = crate::domain::point_ops::build_lut(&self.op);
        crate::domain::point_ops::apply_lut(&src_pixels, &self.source_info, &lut)
    }

    fn overlap(&self) -> Overlap {
        Overlap::zero()
    }
    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::Sequential
    }
}

/// A pipeline node holding a pre-composed LUT from fused point operations.
///
/// Created by the pipeline optimizer when it detects consecutive `PointOpNode`s.
/// Applies one LUT pass regardless of how many ops were fused.
pub struct FusedLutNode {
    upstream: u32,
    lut: [u8; 256],
    source_info: ImageInfo,
}

impl FusedLutNode {
    pub fn new(upstream: u32, source_info: ImageInfo, lut: [u8; 256]) -> Self {
        Self {
            upstream,
            lut,
            source_info,
        }
    }

    /// Create a fused node from multiple point operations.
    pub fn from_ops(
        upstream: u32,
        source_info: ImageInfo,
        ops: &[crate::domain::point_ops::PointOp],
    ) -> Self {
        use crate::domain::point_ops::{build_lut, compose_luts};
        let mut lut: [u8; 256] = std::array::from_fn(|i| i as u8); // identity
        for op in ops {
            let op_lut = build_lut(op);
            lut = compose_luts(&lut, &op_lut);
        }
        Self::new(upstream, source_info, lut)
    }
}

impl ImageNode for FusedLutNode {
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
        crate::domain::point_ops::apply_lut(&src_pixels, &self.source_info, &self.lut)
    }

    fn overlap(&self) -> Overlap {
        Overlap::zero()
    }
    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::Sequential
    }
}

// ─── 3D color LUT nodes (3D-fusible) ────────────────────────────────────────

/// A pipeline node for a 3D-fusible color operation (hue rotate, saturate, etc.).
///
/// At execution, builds a 3D CLUT from the stored `ColorOp` and applies it.
/// The pipeline optimizer can detect consecutive `ColorOpNode`s and replace them
/// with a single `FusedClutNode`.
pub struct ColorOpNode {
    upstream: u32,
    op: crate::domain::color_lut::ColorOp,
    source_info: ImageInfo,
}

impl ColorOpNode {
    pub fn new(
        upstream: u32,
        source_info: ImageInfo,
        op: crate::domain::color_lut::ColorOp,
    ) -> Self {
        Self {
            upstream,
            op,
            source_info,
        }
    }
}

impl ImageNode for ColorOpNode {
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
        let clut = self.op.to_clut(crate::domain::color_lut::DEFAULT_GRID_SIZE);
        clut.apply(&src_pixels, &self.source_info)
    }

    fn overlap(&self) -> Overlap {
        Overlap::zero()
    }
    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::Sequential
    }
}

/// A pipeline node holding a pre-composed 3D CLUT from fused color operations.
///
/// Created by the pipeline optimizer when it detects consecutive 3D-fusible nodes.
/// Applies one tetrahedral interpolation pass regardless of how many ops were fused.
/// Can also absorb adjacent 1D point ops as pre/post-curves.
pub struct FusedClutNode {
    upstream: u32,
    clut: crate::domain::color_lut::ColorLut3D,
    source_info: ImageInfo,
}

impl FusedClutNode {
    pub fn new(
        upstream: u32,
        source_info: ImageInfo,
        clut: crate::domain::color_lut::ColorLut3D,
    ) -> Self {
        Self {
            upstream,
            clut,
            source_info,
        }
    }

    /// Create from multiple 3D color operations, optionally with 1D pre/post-curves.
    pub fn from_ops(
        upstream: u32,
        source_info: ImageInfo,
        pre_1d: Option<&[u8; 256]>,
        color_ops: &[crate::domain::color_lut::ColorOp],
        post_1d: Option<&[u8; 256]>,
    ) -> Self {
        use crate::domain::color_lut::*;
        let grid = DEFAULT_GRID_SIZE;

        // Build composed 3D CLUT from all color ops
        let mut clut = ColorLut3D::identity(grid);
        for op in color_ops {
            let op_clut = op.to_clut(grid);
            clut = compose_cluts(&clut, &op_clut);
        }

        // Absorb 1D pre-curves
        if let Some(pre) = pre_1d {
            clut = absorb_1d_pre(pre, &clut);
        }

        // Absorb 1D post-curves
        if let Some(post) = post_1d {
            clut = absorb_1d_post(&clut, post);
        }

        Self::new(upstream, source_info, clut)
    }
}

impl ImageNode for FusedClutNode {
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
        self.clut.apply(&src_pixels, &self.source_info)
    }

    fn overlap(&self) -> Overlap {
        Overlap::zero()
    }
    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::Sequential
    }
}

/// Grayscale node — changes pixel format to Gray8.
pub struct GrayscaleNode {
    upstream: u32,
    source_info: ImageInfo,
}

impl GrayscaleNode {
    pub fn new(upstream: u32, source_info: ImageInfo) -> Self {
        Self {
            upstream,
            source_info,
        }
    }
}

impl ImageNode for GrayscaleNode {
    fn info(&self) -> ImageInfo {
        ImageInfo {
            format: PixelFormat::Gray8,
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
        let result = filters::grayscale(&src_pixels, &self.source_info)?;
        Ok(result.pixels)
    }

    fn overlap(&self) -> Overlap {
        Overlap::zero()
    }
    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::Sequential
    }
}

// ─── Histogram-based nodes (whole-image analysis + LUT application) ─────────

macro_rules! histogram_node {
    ($name:ident, $fn_name:ident, $doc:expr) => {
        #[doc = $doc]
        pub struct $name {
            upstream: u32,
            source_info: ImageInfo,
        }

        impl $name {
            pub fn new(upstream: u32, source_info: ImageInfo) -> Self {
                Self {
                    upstream,
                    source_info,
                }
            }
        }

        impl ImageNode for $name {
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
                crate::domain::histogram::$fn_name(&src_pixels, &self.source_info)
            }

            fn overlap(&self) -> Overlap {
                // Needs entire image for histogram computation
                Overlap::uniform(u32::MAX)
            }
            fn access_pattern(&self) -> AccessPattern {
                AccessPattern::RandomAccess
            }
        }
    };
}

histogram_node!(EqualizeNode, equalize, "Histogram equalization node.");
histogram_node!(NormalizeNode, normalize, "Histogram normalization node.");
histogram_node!(
    AutoLevelNode,
    auto_level,
    "Auto-level (min/max stretch) node."
);

/// Contrast stretch node with configurable black/white point percentiles.
pub struct ContrastStretchNode {
    upstream: u32,
    source_info: ImageInfo,
    black_pct: f64,
    white_pct: f64,
}

impl ContrastStretchNode {
    pub fn new(upstream: u32, source_info: ImageInfo, black_pct: f64, white_pct: f64) -> Self {
        Self {
            upstream,
            source_info,
            black_pct,
            white_pct,
        }
    }
}

impl ImageNode for ContrastStretchNode {
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
        crate::domain::histogram::contrast_stretch(
            &src_pixels,
            &self.source_info,
            self.black_pct,
            self.white_pct,
        )
    }

    fn overlap(&self) -> Overlap {
        Overlap::uniform(u32::MAX)
    }
    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::RandomAccess
    }
}
