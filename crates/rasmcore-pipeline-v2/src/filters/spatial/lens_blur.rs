use crate::node::PipelineError;
use crate::ops::{Filter, GpuFilter};

use super::convolve::Convolve;
use super::{make_disc_kernel, make_polygon_kernel};
use crate::gpu_shaders::spatial;

/// Lens blur — polygon/disc-shaped bokeh kernel convolution.
///
/// Simulates camera lens defocus with configurable aperture blade count.
/// `blade_count=0` produces a circular disc; `blade_count>=3` produces a polygon.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "lens_blur", category = "spatial", cost = "O(n * r^2)")]
pub struct LensBlur {
    #[param(min = 0, max = 50, default = 5)]
    pub radius: u32,
    /// Aperture blade count. 0 = disc, 5-8 = polygon.
    #[param(min = 0, max = 12, default = 0)]
    pub blade_count: u32,
    /// Blade rotation in degrees.
    #[param(min = 0.0, max = 360.0, default = 0.0)]
    pub rotation: f32,
}

impl Filter for LensBlur {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        if self.radius == 0 {
            return Ok(input.to_vec());
        }
        let kernel = if self.blade_count < 3 {
            make_disc_kernel(self.radius)
        } else {
            make_polygon_kernel(self.radius, self.blade_count, self.rotation)
        };
        let ksize = (self.radius * 2 + 1) as usize;
        let divisor: f32 = kernel.iter().sum();
        let conv = Convolve {
            kernel,
            kernel_width: ksize as u32,
            kernel_height: ksize as u32,
            divisor: divisor.max(1e-6),
        };
        conv.compute(input, width, height)
    }

    fn tile_overlap(&self) -> u32 {
        self.radius
    }
}

// ── LensBlur GPU (kernel convolution via extra_buffer) ──────────────────────

impl GpuFilter for LensBlur {
    fn shader_body(&self) -> &str { spatial::LENS_BLUR }
    fn workgroup_size(&self) -> [u32; 3] { [16, 16, 1] }
    fn params(&self, w: u32, h: u32) -> Vec<u8> {
        let ksize = self.radius * 2 + 1;
        let kernel = if self.blade_count < 3 {
            make_disc_kernel(self.radius)
        } else {
            make_polygon_kernel(self.radius, self.blade_count, self.rotation)
        };
        let divisor: f32 = kernel.iter().sum::<f32>().max(1e-6);
        let mut buf = Vec::with_capacity(32);
        buf.extend_from_slice(&w.to_le_bytes());
        buf.extend_from_slice(&h.to_le_bytes());
        buf.extend_from_slice(&ksize.to_le_bytes());
        buf.extend_from_slice(&ksize.to_le_bytes());
        buf.extend_from_slice(&(1.0f32 / divisor).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }
    fn extra_buffers(&self) -> Vec<Vec<u8>> {
        let kernel = if self.blade_count < 3 {
            make_disc_kernel(self.radius)
        } else {
            make_polygon_kernel(self.radius, self.blade_count, self.rotation)
        };
        let mut kb = Vec::with_capacity(kernel.len() * 4);
        for &w in &kernel {
            kb.extend_from_slice(&w.to_le_bytes());
        }
        vec![kb]
    }
}
