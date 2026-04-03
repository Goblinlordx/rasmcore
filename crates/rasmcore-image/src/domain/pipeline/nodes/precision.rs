//! Precision conversion nodes for the pipeline.
//!
//! PromoteNode converts u8/u16 pixels to f32 (inserted after source in HP mode).
//! DemoteNode converts f32 pixels back to u8/u16 (inserted before encode sink).

use crate::domain::error::ImageError;
use crate::domain::pipeline::graph::{AccessPattern, ImageNode};
use crate::domain::transform::convert_format;
use crate::domain::types::{ImageInfo, PixelFormat};
use rasmcore_pipeline::Rect;
use rasmcore_pipeline::gpu::{BufferFormat, GpuCapable, GpuOp};
use std::sync::LazyLock;

/// Promotes pixel data from any format to f32.
/// Inserted automatically after source nodes.
pub struct PromoteNode {
    upstream: u32,
    source_info: ImageInfo,
    target_format: PixelFormat,
}

impl PromoteNode {
    /// Promote to Rgba32f — the canonical pipeline format.
    /// All formats (Gray, Rgb, Rgba, u8/u16/f16/f32) become Rgba32f.
    /// Gray → R=G=B=luma, A=1.0. Rgb → A=1.0. Already-Rgba32f is a no-op.
    pub fn new(upstream: u32, source_info: ImageInfo) -> Self {
        Self {
            upstream,
            source_info,
            target_format: PixelFormat::Rgba32f,
        }
    }

    /// Promote to a specific target format (for non-standard use cases).
    pub fn with_target(upstream: u32, source_info: ImageInfo, target_format: PixelFormat) -> Self {
        Self {
            upstream,
            source_info,
            target_format,
        }
    }
}

impl ImageNode for PromoteNode {
    fn info(&self) -> ImageInfo {
        ImageInfo {
            format: self.target_format,
            ..self.source_info.clone()
        }
    }

    fn compute_region(
        &self,
        request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        let src_pixels = upstream_fn(self.upstream, request)?;
        // Build a temporary ImageInfo with the region dimensions for conversion
        let region_info = ImageInfo {
            width: request.width,
            height: request.height,
            format: self.source_info.format,
            color_space: self.source_info.color_space,
        };
        let result = convert_format(&src_pixels, &region_info, self.target_format)?;
        Ok(result.pixels)
    }

    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::Sequential
    }

    fn upstream_id(&self) -> Option<u32> {
        Some(self.upstream)
    }
}

static PROMOTE_F32_SHADER: LazyLock<String> = LazyLock::new(|| {
    include_str!("../../../shaders/promote_f32.wgsl").to_string()
});

impl GpuCapable for PromoteNode {
    fn gpu_ops_with_format(
        &self,
        width: u32,
        height: u32,
        _buffer_format: BufferFormat,
    ) -> Option<Vec<GpuOp>> {
        // Only promote Rgba8 → Rgba32f on GPU (most common path)
        if self.source_info.format != PixelFormat::Rgba8 || self.target_format != PixelFormat::Rgba32f {
            return None;
        }

        let mut params = Vec::with_capacity(8);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());

        // The u8 source pixels go as extra_buffers[0] (binding 3).
        // The graph walker populates this from upstream u8 data.
        // Ping-pong input (binding 0) is unused; output (binding 1) is f32.
        Some(vec![GpuOp::Compute {
            shader: PROMOTE_F32_SHADER.clone(),
            entry_point: "main",
            workgroup_size: [256, 1, 1],
            params,
            extra_buffers: vec![], // populated by graph walker at dispatch time
            buffer_format: BufferFormat::F32Vec4, // output format
        }])
    }

    fn input_buffer_format(&self) -> Option<BufferFormat> {
        // Upstream data is u8 packed (4 bytes/pixel), not f32 (16 bytes/pixel)
        if self.source_info.format == PixelFormat::Rgba8 {
            Some(BufferFormat::U32Packed)
        } else {
            None
        }
    }
}

/// Demotes pixel data from f32 to u8 format.
/// Inserted automatically before encode sinks when outputting to 8-bit formats.
pub struct DemoteNode {
    upstream: u32,
    source_info: ImageInfo,
    target_format: PixelFormat,
}

impl DemoteNode {
    pub fn new(upstream: u32, source_info: ImageInfo, target_format: PixelFormat) -> Self {
        Self {
            upstream,
            source_info,
            target_format,
        }
    }
}

impl ImageNode for DemoteNode {
    fn info(&self) -> ImageInfo {
        ImageInfo {
            format: self.target_format,
            ..self.source_info.clone()
        }
    }

    fn compute_region(
        &self,
        request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        let src_pixels = upstream_fn(self.upstream, request)?;
        let region_info = ImageInfo {
            width: request.width,
            height: request.height,
            format: self.source_info.format,
            color_space: self.source_info.color_space,
        };
        let result = convert_format(&src_pixels, &region_info, self.target_format)?;
        Ok(result.pixels)
    }

    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::Sequential
    }

    fn upstream_id(&self) -> Option<u32> {
        Some(self.upstream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::ColorSpace;

    fn make_info(format: PixelFormat) -> ImageInfo {
        ImageInfo {
            width: 2,
            height: 2,
            format,
            color_space: ColorSpace::Srgb,
        }
    }

    #[test]
    fn promote_rgba8_to_rgba32f() {
        let info = make_info(PixelFormat::Rgba8);
        let node = PromoteNode::new(0, info);
        assert_eq!(node.info().format, PixelFormat::Rgba32f);
    }

    #[test]
    fn promote_rgb8_to_rgba32f() {
        // f32 pipeline: everything promotes to Rgba32f
        let info = make_info(PixelFormat::Rgb8);
        let node = PromoteNode::new(0, info);
        assert_eq!(node.info().format, PixelFormat::Rgba32f);
    }

    #[test]
    fn promote_gray8_to_rgba32f() {
        // f32 pipeline: even grayscale promotes to Rgba32f (R=G=B=luma, A=1)
        let info = make_info(PixelFormat::Gray8);
        let node = PromoteNode::new(0, info);
        assert_eq!(node.info().format, PixelFormat::Rgba32f);
    }

    #[test]
    fn promote_computes_f32_pixels() {
        let info = make_info(PixelFormat::Rgba8);
        let node = PromoteNode::new(0, info);
        // 2x2 RGBA8: all pixels = [128, 64, 255, 200]
        let input = vec![128u8, 64, 255, 200, 128, 64, 255, 200, 128, 64, 255, 200, 128, 64, 255, 200];
        let request = Rect::new(0, 0, 2, 2);
        let mut upstream = |_id: u32, _rect: Rect| -> Result<Vec<u8>, ImageError> {
            Ok(input.clone())
        };
        let result = node.compute_region(request, &mut upstream).unwrap();
        // 2x2 Rgba32f = 4 pixels * 16 bytes = 64 bytes
        assert_eq!(result.len(), 64);
        // Check first pixel R channel
        let r = f32::from_le_bytes([result[0], result[1], result[2], result[3]]);
        assert!((r - 128.0 / 255.0).abs() < 0.01);
    }

    #[test]
    fn demote_rgba32f_to_rgba8() {
        let info = make_info(PixelFormat::Rgba32f);
        let node = DemoteNode::new(0, info, PixelFormat::Rgba8);
        assert_eq!(node.info().format, PixelFormat::Rgba8);
    }

    #[test]
    fn promote_demote_round_trip() {
        let info_8 = make_info(PixelFormat::Rgba8);
        let promote = PromoteNode::new(0, info_8.clone());

        let input = vec![100u8, 150, 200, 255, 50, 75, 125, 128, 0, 0, 0, 255, 255, 255, 255, 255];
        let request = Rect::new(0, 0, 2, 2);

        // Promote
        let mut upstream_promote = |_id: u32, _rect: Rect| -> Result<Vec<u8>, ImageError> {
            Ok(input.clone())
        };
        let promoted = promote.compute_region(request, &mut upstream_promote).unwrap();

        // Demote
        let info_f32 = make_info(PixelFormat::Rgba32f);
        let demote = DemoteNode::new(0, info_f32, PixelFormat::Rgba8);
        let mut upstream_demote = |_id: u32, _rect: Rect| -> Result<Vec<u8>, ImageError> {
            Ok(promoted.clone())
        };
        let result = demote.compute_region(request, &mut upstream_demote).unwrap();

        assert_eq!(input, result);
    }
}
