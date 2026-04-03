//! Filter: pixelate (category: effect)
//!
//! Block-grid mosaic: tiles a fixed grid from (0,0), averages each cell, fills
//! with the mean color. Edge cells are truncated to image bounds (not padded).
//! This matches the behavior of Photoshop Mosaic, GIMP/GEGL pixelize, and
//! FFmpeg pixelize. Resize-based pixelation (OpenCV, ImageMagick -scale) uses
//! proportional mapping instead, producing visually uniform blocks but is a
//! different operation.

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::{CpuFilter, GpuFilter};


#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "pixelate", category = "effect", reference = "block mosaic pixelation")]
pub struct PixelateParams {
    #[param(min = 1, max = 128, step = 1, default = 8)]
    pub block_size: u32,
}

impl CpuFilter for PixelateParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    let block_size = self.block_size;

    validate_format(info.format)?;

    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |px, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(px.to_vec());
            self.compute(r, &mut u, i8)
        });
    }

    let bs = block_size.max(1) as usize;
    let w = info.width as usize;
    let h = info.height as usize;
    let ch = channels(info.format);
    let mut out = vec![0u8; pixels.len()];

    let mut by = 0;
    while by < h {
        let bh = bs.min(h - by);
        let mut bx = 0;
        while bx < w {
            let bw = bs.min(w - bx);
            let count = bw * bh;

            // Accumulate channel sums
            let mut sums = [0u32; 4]; // max 4 channels
            for row in by..(by + bh) {
                for col in bx..(bx + bw) {
                    let off = (row * w + col) * ch;
                    for c in 0..ch {
                        sums[c] += pixels[off + c] as u32;
                    }
                }
            }

            // Compute averages
            let mut avg = [0u8; 4];
            for c in 0..ch {
                avg[c] = ((sums[c] + count as u32 / 2) / count as u32) as u8;
            }

            // Fill block with average
            for row in by..(by + bh) {
                for col in bx..(bx + bw) {
                    let off = (row * w + col) * ch;
                    out[off..off + ch].copy_from_slice(&avg[..ch]);
                }
            }

            bx += bs;
        }
        by += bs;
    }

    Ok(out)
}
}

impl GpuFilter for PixelateParams {
    fn gpu_ops(&self, _width: u32, _height: u32) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        None
    }

    fn gpu_ops_with_format(
        &self,
        width: u32,
        height: u32,
        buffer_format: rasmcore_pipeline::gpu::BufferFormat,
    ) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        if buffer_format != rasmcore_pipeline::gpu::BufferFormat::F32Vec4 {
            return None;
        }
        use rasmcore_pipeline::gpu::GpuOp;
        use std::sync::LazyLock;
        static SHADER: LazyLock<String> = LazyLock::new(|| {
            include_str!("../../../shaders/pixelate_f32.wgsl").to_string()
        });
        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.block_size.max(1).to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes()); // _pad
        Some(vec![GpuOp::Compute {
            shader: SHADER.clone(),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
            buffer_format: rasmcore_pipeline::gpu::BufferFormat::F32Vec4,
        }])
    }
}

