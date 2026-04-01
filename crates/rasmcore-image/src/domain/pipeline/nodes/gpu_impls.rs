//! GpuCapable implementations for auto-generated filter pipeline nodes.
//!
//! Each filter node that has a corresponding WGSL shader gets a GpuCapable
//! impl here. The impl packs filter params into the uniform buffer layout
//! matching the shader's `Params` struct (little-endian, 4-byte aligned).

use super::filters::*;
use crate::domain::types::PixelFormat;
use rasmcore_pipeline::{GpuCapable, GpuOp};

/// Helper: only support RGBA8 on GPU (the shader packs/unpacks u32 as RGBA).
fn is_rgba8(node_info: &crate::domain::types::ImageInfo) -> bool {
    node_info.format == PixelFormat::Rgba8
}

// ─── Spatial Filters ─────────────────────────────────────────────────────────

impl GpuCapable for BlurNode {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<GpuOp>> {
        if !is_rgba8(&self.source_info) || self.config.radius <= 0.0 {
            return None;
        }
        let sigma = self.config.radius;
        let kernel_radius = (sigma * 3.0).ceil() as u32;
        if kernel_radius > 32 {
            return None; // Too large for GPU kernel buffer
        }

        // Build 1D Gaussian kernel weights
        let ksize = 2 * kernel_radius + 1;
        let mut weights = Vec::with_capacity(ksize as usize);
        let mut sum = 0.0f32;
        for i in 0..ksize {
            let x = i as f32 - kernel_radius as f32;
            let w = (-0.5 * (x / sigma) * (x / sigma)).exp();
            weights.push(w);
            sum += w;
        }
        let inv_sum = 1.0 / sum;
        let mut kernel_buf = Vec::with_capacity(ksize as usize * 4);
        for w in &weights {
            kernel_buf.extend_from_slice(&(w * inv_sum).to_le_bytes());
        }

        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&kernel_radius.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes()); // _pad

        let shader = include_str!("../../../shaders/gaussian_blur.wgsl");

        Some(vec![
            GpuOp {
                shader,
                entry_point: "blur_h",
                workgroup_size: [256, 1, 1],
                params: params.clone(),
                extra_buffers: vec![kernel_buf.clone()],
            },
            GpuOp {
                shader,
                entry_point: "blur_v",
                workgroup_size: [1, 256, 1],
                params,
                extra_buffers: vec![kernel_buf],
            },
        ])
    }
}

impl GpuCapable for SharpenNode {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<GpuOp>> {
        if !is_rgba8(&self.source_info) {
            return None;
        }

        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.config.amount.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes()); // _pad

        Some(vec![GpuOp {
            shader: include_str!("../../../shaders/sharpen.wgsl"),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
        }])
    }
}

impl GpuCapable for BilateralNode {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<GpuOp>> {
        if !is_rgba8(&self.source_info) {
            return None;
        }
        let radius = self.config.diameter / 2;

        let mut params = Vec::with_capacity(32);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&radius.to_le_bytes());
        params.extend_from_slice(&self.config.sigma_space.to_le_bytes());
        params.extend_from_slice(&self.config.sigma_color.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes()); // _pad1
        params.extend_from_slice(&0u32.to_le_bytes()); // _pad2
        params.extend_from_slice(&0u32.to_le_bytes()); // _pad3

        Some(vec![GpuOp {
            shader: include_str!("../../../shaders/bilateral.wgsl"),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
        }])
    }
}

impl GpuCapable for GuidedFilterNode {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<GpuOp>> {
        if !is_rgba8(&self.source_info) {
            return None;
        }

        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.config.radius.to_le_bytes());
        params.extend_from_slice(&self.config.epsilon.to_le_bytes());

        Some(vec![GpuOp {
            shader: include_str!("../../../shaders/guided_filter.wgsl"),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
        }])
    }
}

impl GpuCapable for MedianNode {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<GpuOp>> {
        if !is_rgba8(&self.source_info) || self.config.radius > 3 {
            return None; // GPU median limited to radius <= 3 (7x7)
        }

        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.config.radius.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes()); // _pad

        Some(vec![GpuOp {
            shader: include_str!("../../../shaders/median.wgsl"),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
        }])
    }
}

impl GpuCapable for SpinBlurNode {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<GpuOp>> {
        if !is_rgba8(&self.source_info) {
            return None;
        }
        let samples = ((self.config.angle.abs() * 180.0 / std::f32::consts::PI).ceil() as u32)
            .clamp(8, 128);

        let mut params = Vec::with_capacity(32);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.config.center_x.to_le_bytes());
        params.extend_from_slice(&self.config.center_y.to_le_bytes());
        params.extend_from_slice(&self.config.angle.to_le_bytes());
        params.extend_from_slice(&samples.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes()); // _pad1
        params.extend_from_slice(&0u32.to_le_bytes()); // _pad2

        Some(vec![GpuOp {
            shader: include_str!("../../../shaders/spin_blur.wgsl"),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
        }])
    }
}

impl GpuCapable for MotionBlurNode {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<GpuOp>> {
        if !is_rgba8(&self.source_info) || self.config.length == 0 {
            return None;
        }
        let angle_rad = self.config.angle_degrees * std::f32::consts::PI / 180.0;

        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.config.length.to_le_bytes());
        params.extend_from_slice(&angle_rad.to_le_bytes());

        Some(vec![GpuOp {
            shader: include_str!("../../../shaders/motion_blur.wgsl"),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
        }])
    }
}

impl GpuCapable for ZoomBlurNode {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<GpuOp>> {
        if !is_rgba8(&self.source_info) || self.config.factor == 0.0 {
            return None;
        }
        let samples = ((self.config.factor.abs() * 64.0).ceil() as u32).clamp(8, 128);

        let mut params = Vec::with_capacity(32);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.config.center_x.to_le_bytes());
        params.extend_from_slice(&self.config.center_y.to_le_bytes());
        params.extend_from_slice(&self.config.factor.to_le_bytes());
        params.extend_from_slice(&samples.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes()); // _pad1
        params.extend_from_slice(&0u32.to_le_bytes()); // _pad2

        Some(vec![GpuOp {
            shader: include_str!("../../../shaders/zoom_blur.wgsl"),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
        }])
    }
}

// ─── Distortion Filters ──────────────────────────────────────────────────────

impl GpuCapable for SpherizeNode {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<GpuOp>> {
        if !is_rgba8(&self.source_info) {
            return None;
        }

        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.config.amount.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes()); // _pad

        Some(vec![GpuOp {
            shader: include_str!("../../../shaders/spherize.wgsl"),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
        }])
    }
}

impl GpuCapable for SwirlNode {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<GpuOp>> {
        if !is_rgba8(&self.source_info) {
            return None;
        }

        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.config.angle.to_le_bytes());
        params.extend_from_slice(&self.config.radius.to_le_bytes());

        Some(vec![GpuOp {
            shader: include_str!("../../../shaders/swirl.wgsl"),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
        }])
    }
}

impl GpuCapable for BarrelNode {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<GpuOp>> {
        if !is_rgba8(&self.source_info) {
            return None;
        }

        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.config.k1.to_le_bytes());
        params.extend_from_slice(&self.config.k2.to_le_bytes());

        Some(vec![GpuOp {
            shader: include_str!("../../../shaders/barrel.wgsl"),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
        }])
    }
}

impl GpuCapable for RippleNode {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<GpuOp>> {
        if !is_rgba8(&self.source_info) {
            return None;
        }

        let mut params = Vec::with_capacity(32);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.config.amplitude.to_le_bytes());
        params.extend_from_slice(&self.config.wavelength.to_le_bytes());
        params.extend_from_slice(&self.config.center_x.to_le_bytes());
        params.extend_from_slice(&self.config.center_y.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes()); // _pad1
        params.extend_from_slice(&0u32.to_le_bytes()); // _pad2

        Some(vec![GpuOp {
            shader: include_str!("../../../shaders/ripple.wgsl"),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
        }])
    }
}

impl GpuCapable for WaveNode {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<GpuOp>> {
        if !is_rgba8(&self.source_info) {
            return None;
        }

        let mut params = Vec::with_capacity(32);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.config.amplitude.to_le_bytes());
        params.extend_from_slice(&self.config.wavelength.to_le_bytes());
        params.extend_from_slice(&self.config.vertical.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes()); // _pad1
        params.extend_from_slice(&0u32.to_le_bytes()); // _pad2
        params.extend_from_slice(&0u32.to_le_bytes()); // _pad3

        Some(vec![GpuOp {
            shader: include_str!("../../../shaders/wave.wgsl"),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
        }])
    }
}

impl GpuCapable for PolarNode {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<GpuOp>> {
        if !is_rgba8(&self.source_info) {
            return None;
        }

        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes()); // _pad1
        params.extend_from_slice(&0u32.to_le_bytes()); // _pad2

        Some(vec![GpuOp {
            shader: include_str!("../../../shaders/polar.wgsl"),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
        }])
    }
}

impl GpuCapable for DepolarNode {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<GpuOp>> {
        if !is_rgba8(&self.source_info) {
            return None;
        }

        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes()); // _pad1
        params.extend_from_slice(&0u32.to_le_bytes()); // _pad2

        Some(vec![GpuOp {
            shader: include_str!("../../../shaders/depolar.wgsl"),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: vec![],
        }])
    }
}
