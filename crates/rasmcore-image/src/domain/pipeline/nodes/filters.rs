//! Filter nodes — wrap existing domain filter operations.

use crate::domain::error::ImageError;
use crate::domain::filters;
use crate::domain::pipeline::graph::{AccessPattern, ImageNode, bytes_per_pixel, crop_region};
use crate::domain::types::*;
use rasmcore_pipeline::{Overlap, Rect};

/// Tiled execution helper: expand request by overlap, fetch upstream, apply filter, crop back.
///
/// This is the standard pattern for all local-neighborhood filter nodes.
/// For Overlap::zero() nodes, no expansion or cropping occurs.
fn tiled_filter<F>(
    request: Rect,
    overlap: &Overlap,
    source_info: &ImageInfo,
    upstream: u32,
    upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    apply: F,
) -> Result<Vec<u8>, ImageError>
where
    F: FnOnce(&[u8], &ImageInfo) -> Result<Vec<u8>, ImageError>,
{
    let bpp = bytes_per_pixel(source_info.format);
    let upstream_rect = request.expand(overlap, source_info.width, source_info.height);
    let src_pixels = upstream_fn(upstream, upstream_rect)?;

    let region_info = ImageInfo {
        width: upstream_rect.width,
        height: upstream_rect.height,
        ..*source_info
    };

    let filtered = apply(&src_pixels, &region_info)?;

    if upstream_rect == request {
        Ok(filtered)
    } else {
        let sub = Rect::new(
            request.x - upstream_rect.x,
            request.y - upstream_rect.y,
            request.width,
            request.height,
        );
        let out_rect = Rect::new(0, 0, upstream_rect.width, upstream_rect.height);
        Ok(crop_region(&filtered, out_rect, sub, bpp))
    }
}

/// Build ImageInfo for a tile region (same format/color_space, tile dimensions).
#[inline]
fn tile_info(request: Rect, source: &ImageInfo) -> ImageInfo {
    ImageInfo {
        width: request.width,
        height: request.height,
        ..*source
    }
}

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
                request: Rect,
                upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
            ) -> Result<Vec<u8>, ImageError> {
                let overlap = self.overlap();
                let bounds_w = self.source_info.width;
                let bounds_h = self.source_info.height;
                let bpp = bytes_per_pixel(self.source_info.format);

                // Expand request by overlap, clamped to image bounds
                let upstream_rect = request.expand(&overlap, bounds_w, bounds_h);

                let src_pixels = upstream_fn(self.upstream, upstream_rect)?;

                // Build info for the upstream region (may be larger than request)
                let region_info = ImageInfo {
                    width: upstream_rect.width,
                    height: upstream_rect.height,
                    ..self.source_info
                };

                // Apply filter to the fetched region
                let filtered = filters::$fn_name(&src_pixels, &region_info, self.param)?;

                // Crop back to the originally requested rect (remove overlap padding)
                if upstream_rect == request {
                    Ok(filtered)
                } else {
                    // The request rect relative to the upstream_rect origin
                    let sub = Rect::new(
                        request.x - upstream_rect.x,
                        request.y - upstream_rect.y,
                        request.width,
                        request.height,
                    );
                    let out_rect = Rect::new(0, 0, upstream_rect.width, upstream_rect.height);
                    Ok(crop_region(&filtered, out_rect, sub, bpp))
                }
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
        request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        let kernel = &self.kernel;
        let kw = self.kw;
        let kh = self.kh;
        let divisor = self.divisor;
        tiled_filter(
            request,
            &self.overlap(),
            &self.source_info,
            self.upstream,
            upstream_fn,
            |px, info| filters::convolve(px, info, kernel, kw, kh, divisor),
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
        request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        let radius = self.radius;
        tiled_filter(
            request,
            &self.overlap(),
            &self.source_info,
            self.upstream,
            upstream_fn,
            |px, info| filters::median(px, info, radius),
        )
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
        request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        let overlap = self.overlap();
        let bpp = bytes_per_pixel(self.source_info.format);
        let ur = request.expand(&overlap, self.source_info.width, self.source_info.height);
        let src = upstream_fn(self.upstream, ur)?;
        {
            let ri = ImageInfo {
                width: ur.width,
                height: ur.height,
                ..self.source_info
            };
            let f = filters::sobel(&src, &ri)?;
            if ur == request {
                Ok(f)
            } else {
                let sub = Rect::new(
                    request.x - ur.x,
                    request.y - ur.y,
                    request.width,
                    request.height,
                );
                Ok(crop_region(
                    &f,
                    Rect::new(0, 0, ur.width, ur.height),
                    sub,
                    1,
                ))
            }
        }
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
        request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        // tiled: use request directly
        let src = upstream_fn(self.upstream, request)?;
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
        request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        use crate::domain::point_ops;
        // tiled: use request directly
        let src_pixels = upstream_fn(self.upstream, request)?;
        // Auto-dispatch: 16-bit formats use 65536-entry LUT, 8-bit use 256-entry
        if matches!(
            self.source_info.format,
            PixelFormat::Rgb16 | PixelFormat::Rgba16 | PixelFormat::Gray16
        ) {
            let lut = point_ops::build_lut_u16(&self.op);
            point_ops::apply_lut_u16(&src_pixels, &self.source_info, &lut)
        } else {
            let lut = point_ops::build_lut(&self.op);
            point_ops::apply_lut(&src_pixels, &self.source_info, &lut)
        }
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
/// Supports both 8-bit (256-entry) and 16-bit (65536-entry) LUTs.
pub struct FusedLutNode {
    upstream: u32,
    ops: Vec<crate::domain::point_ops::PointOp>,
    source_info: ImageInfo,
}

impl FusedLutNode {
    /// Create a fused node from a pre-composed 8-bit LUT (legacy API).
    pub fn new(upstream: u32, source_info: ImageInfo, _lut: [u8; 256]) -> Self {
        // For backward compat: store as identity ops (the LUT is rebuilt at compute time)
        Self {
            upstream,
            ops: Vec::new(),
            source_info,
        }
    }

    /// Create a fused node from multiple point operations.
    pub fn from_ops(
        upstream: u32,
        source_info: ImageInfo,
        ops: &[crate::domain::point_ops::PointOp],
    ) -> Self {
        Self {
            upstream,
            ops: ops.to_vec(),
            source_info,
        }
    }
}

impl ImageNode for FusedLutNode {
    fn info(&self) -> ImageInfo {
        self.source_info.clone()
    }

    fn compute_region(
        &self,
        request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        use crate::domain::point_ops;
        // tiled: use request directly
        let src_pixels = upstream_fn(self.upstream, request)?;

        if matches!(
            self.source_info.format,
            PixelFormat::Rgb16 | PixelFormat::Rgba16 | PixelFormat::Gray16
        ) {
            // 16-bit path: build and compose 65536-entry LUTs
            let mut lut: Vec<u16> = (0..65536).map(|i| i as u16).collect();
            for op in &self.ops {
                let op_lut = point_ops::build_lut_u16(op);
                lut = point_ops::compose_luts_u16(&lut, &op_lut);
            }
            point_ops::apply_lut_u16(&src_pixels, &self.source_info, &lut)
        } else {
            // 8-bit path: build and compose 256-entry LUTs
            let mut lut: [u8; 256] = std::array::from_fn(|i| i as u8);
            for op in &self.ops {
                let op_lut = point_ops::build_lut(op);
                lut = point_ops::compose_luts(&lut, &op_lut);
            }
            point_ops::apply_lut(&src_pixels, &self.source_info, &lut)
        }
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
        request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        // tiled: use request directly
        let src_pixels = upstream_fn(self.upstream, request)?;
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
        request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        // tiled: use request directly
        let src_pixels = upstream_fn(self.upstream, request)?;
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
        request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        // tiled: use request directly
        let src_pixels = upstream_fn(self.upstream, request)?;
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
                request: Rect,
                upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
            ) -> Result<Vec<u8>, ImageError> {
                // tiled: use request directly
                let src_pixels = upstream_fn(self.upstream, request)?;
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

// ─── Bit Depth Conversion Node ──────────────────────────────────────────────

/// Pipeline node for explicit bit depth conversion (8↔16).
///
/// Allows users to control where in the pipeline bit depth changes happen.
/// - Insert early for speed (process at 8-bit after 16-bit decode)
/// - Insert late for precision (process at 16-bit, downconvert before encode)
/// - NOOP when input already matches target depth (optimizer can remove)
pub struct BitDepthNode {
    upstream: u32,
    target_depth: u8, // 8 or 16
    source_info: ImageInfo,
}

impl BitDepthNode {
    pub fn new(upstream: u32, source_info: ImageInfo, target_depth: u8) -> Self {
        Self {
            upstream,
            target_depth,
            source_info,
        }
    }

    /// Returns true if this node is a NOOP (input already matches target depth).
    pub fn is_noop(&self) -> bool {
        let is_src_16 = matches!(
            self.source_info.format,
            PixelFormat::Rgb16 | PixelFormat::Rgba16 | PixelFormat::Gray16
        );
        (self.target_depth == 16 && is_src_16) || (self.target_depth == 8 && !is_src_16)
    }

    fn target_format(&self) -> PixelFormat {
        if self.target_depth == 16 {
            match self.source_info.format {
                PixelFormat::Rgb8 => PixelFormat::Rgb16,
                PixelFormat::Rgba8 => PixelFormat::Rgba16,
                PixelFormat::Gray8 => PixelFormat::Gray16,
                other => other, // already 16-bit or unsupported
            }
        } else {
            match self.source_info.format {
                PixelFormat::Rgb16 => PixelFormat::Rgb8,
                PixelFormat::Rgba16 => PixelFormat::Rgba8,
                PixelFormat::Gray16 => PixelFormat::Gray8,
                other => other, // already 8-bit
            }
        }
    }
}

impl ImageNode for BitDepthNode {
    fn info(&self) -> ImageInfo {
        ImageInfo {
            format: self.target_format(),
            ..self.source_info.clone()
        }
    }

    fn compute_region(
        &self,
        request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        // tiled: use request directly
        let src_pixels = upstream_fn(self.upstream, request)?;

        if self.is_noop() {
            return Ok(src_pixels);
        }

        let target = self.target_format();
        let result =
            crate::domain::transform::convert_format(&src_pixels, &self.source_info, target)?;
        Ok(result.pixels)
    }

    fn overlap(&self) -> Overlap {
        Overlap::zero()
    }
    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::Sequential
    }
}

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
        request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        // tiled: use request directly
        let src_pixels = upstream_fn(self.upstream, request)?;
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

// ─── Smart Crop Node ────────────────────────────────────────────────────────

/// Smart crop node — content-aware crop using entropy or attention strategy.
pub struct SmartCropNode {
    upstream: u32,
    source_info: ImageInfo,
    target_w: u32,
    target_h: u32,
    strategy: crate::domain::smart_crop::SmartCropStrategy,
}

impl SmartCropNode {
    pub fn new(
        upstream: u32,
        source_info: ImageInfo,
        target_w: u32,
        target_h: u32,
        strategy: crate::domain::smart_crop::SmartCropStrategy,
    ) -> Self {
        Self {
            upstream,
            source_info,
            target_w,
            target_h,
            strategy,
        }
    }
}

impl ImageNode for SmartCropNode {
    fn info(&self) -> ImageInfo {
        ImageInfo {
            width: self.target_w,
            height: self.target_h,
            ..self.source_info.clone()
        }
    }

    fn compute_region(
        &self,
        request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        // tiled: use request directly
        let src_pixels = upstream_fn(self.upstream, request)?;
        let result = crate::domain::smart_crop::smart_crop(
            &src_pixels,
            &self.source_info,
            self.target_w,
            self.target_h,
            self.strategy,
        )?;
        Ok(result.pixels)
    }

    fn overlap(&self) -> Overlap {
        Overlap::uniform(u32::MAX) // needs full image for analysis
    }
    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::RandomAccess
    }
}
