//! Filter: oil_paint (category: effect)

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::{CpuFilter, GpuFilter};

/// Oil painting effect: for each pixel, find the most frequent intensity
/// in the neighborhood and output that pixel's color.

#[derive(rasmcore_macros::Filter, Clone)]
/// Oil paint effect — neighborhood mode filter
#[filter(name = "oil_paint", category = "effect", reference = "Kuwahara-variant oil painting simulation")]
pub struct OilPaintParams {
    /// Radius of the neighborhood (1-10)
    #[param(min = 1, max = 10, step = 1, default = 3)]
    pub radius: u32,
}

impl CpuFilter for OilPaintParams {
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
    let radius = self.radius;

    validate_format(info.format)?;
    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            self.compute(r, &mut u, i8)
        });
    }

    let w = info.width as usize;
    let h = info.height as usize;
    let ch = channels(info.format);
    let r = radius as usize;

    // Use 256 intensity bins (one per level) to match ImageMagick's -paint
    // behavior. Coarser binning (e.g. 20 bins) trades accuracy for memory
    // but diverges from the reference.
    const BINS: usize = 256;
    let mut out = vec![0u8; pixels.len()];

    for y in 0..h {
        for x in 0..w {
            let mut count = [0u32; BINS];
            let mut sum_r = [0u32; BINS];
            let mut sum_g = [0u32; BINS];
            let mut sum_b = [0u32; BINS];

            let y0 = y.saturating_sub(r);
            let y1 = (y + r + 1).min(h);
            let x0 = x.saturating_sub(r);
            let x1 = (x + r + 1).min(w);

            for ny in y0..y1 {
                for nx in x0..x1 {
                    let idx = (ny * w + nx) * ch;
                    let intensity = if ch >= 3 {
                        // BT.601 luminance
                        ((pixels[idx] as u32 * 77
                            + pixels[idx + 1] as u32 * 150
                            + pixels[idx + 2] as u32 * 29
                            + 128)
                            >> 8) as usize
                    } else {
                        pixels[idx] as usize
                    };
                    count[intensity] += 1;
                    if ch >= 3 {
                        sum_r[intensity] += pixels[idx] as u32;
                        sum_g[intensity] += pixels[idx + 1] as u32;
                        sum_b[intensity] += pixels[idx + 2] as u32;
                    } else {
                        sum_r[intensity] += pixels[idx] as u32;
                    }
                }
            }

            // Find the bin with the highest count (mode)
            let mut max_bin = 0;
            let mut max_count = 0;
            for (i, &c) in count.iter().enumerate() {
                if c > max_count {
                    max_count = c;
                    max_bin = i;
                }
            }

            let oidx = (y * w + x) * ch;
            if max_count > 0 {
                if ch >= 3 {
                    out[oidx] = (sum_r[max_bin] / max_count) as u8;
                    out[oidx + 1] = (sum_g[max_bin] / max_count) as u8;
                    out[oidx + 2] = (sum_b[max_bin] / max_count) as u8;
                    if ch == 4 {
                        out[oidx + 3] = pixels[oidx + 3]; // preserve alpha
                    }
                } else {
                    out[oidx] = (sum_r[max_bin] / max_count) as u8;
                }
            } else {
                out[oidx..oidx + ch].copy_from_slice(&pixels[oidx..oidx + ch]);
            }
        }
    }

    Ok(out)
}
}

impl GpuFilter for OilPaintParams {
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
            include_str!("../../../shaders/oil_paint_f32.wgsl").to_string()
        });
        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.radius.max(1).to_le_bytes());
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

