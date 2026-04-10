//! LUT application filters — ApplyCubeLut and ApplyHaldLut with GPU support.

use crate::fusion::Clut3D;
use crate::node::PipelineError;
use crate::ops::{Filter, GpuFilter};

use super::super::color::ClutOp;

#[allow(unused_imports)]
use crate::registry::{
    OperationCapabilities, OperationKind, OperationRegistration, ParamDescriptor, ParamType,
};

// ─── ApplyCubeLut ─────────────────────────────────────────────────────────

/// Apply a .cube format 3D LUT.
///
/// The Clut3D is pre-built from parsed .cube data. This filter wraps it
/// as a standard Filter + ClutOp for pipeline integration.
#[derive(Clone)]
pub struct ApplyCubeLut {
    pub clut: Clut3D,
}

impl Filter for ApplyCubeLut {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        Ok(self.clut.apply(input))
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }
}

impl ClutOp for ApplyCubeLut {
    fn build_clut(&self) -> Clut3D {
        self.clut.clone()
    }
}

impl GpuFilter for ApplyCubeLut {
    fn shader_body(&self) -> &str {
        crate::gpu_shaders::grading::LUT_3D_APPLY
    }

    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        clut_gpu_params(&self.clut, width, height)
    }

    fn extra_buffers(&self) -> Vec<Vec<u8>> {
        vec![clut_to_f32_bytes(&self.clut)]
    }
}

// Apply Cube LUT registration
inventory::submit! { &OperationRegistration { name: "apply_cube_lut", display_name: "Apply .cube LUT", category: "grading",
    kind: OperationKind::Filter, params: &[], doc_path: "", cost: "O(n)",
    capabilities: OperationCapabilities { gpu: true, analytic: false, affine: false, clut: true },
} }

// ─── ApplyHaldLut ─────────────────────────────────────────────────────────

/// Apply a Hald CLUT image as a 3D LUT.
///
/// The Clut3D is pre-built from parsed Hald image data.
#[derive(Clone)]
pub struct ApplyHaldLut {
    pub clut: Clut3D,
}

impl Filter for ApplyHaldLut {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        Ok(self.clut.apply(input))
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }
}

impl ClutOp for ApplyHaldLut {
    fn build_clut(&self) -> Clut3D {
        self.clut.clone()
    }
}

impl GpuFilter for ApplyHaldLut {
    fn shader_body(&self) -> &str {
        crate::gpu_shaders::grading::LUT_3D_APPLY
    }

    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        clut_gpu_params(&self.clut, width, height)
    }

    fn extra_buffers(&self) -> Vec<Vec<u8>> {
        vec![clut_to_f32_bytes(&self.clut)]
    }
}

// Apply Hald LUT registration
inventory::submit! { &OperationRegistration { name: "apply_hald_lut", display_name: "Apply Hald CLUT", category: "grading",
    kind: OperationKind::Filter, params: &[], doc_path: "", cost: "O(n)",
    capabilities: OperationCapabilities { gpu: true, analytic: false, affine: false, clut: true },
} }

// ─── Helpers ──────────────────────────────────────────────────────────────

/// Serialize CLUT data to f32 little-endian bytes for GPU extra_buffer.
pub(crate) fn clut_to_f32_bytes(clut: &Clut3D) -> Vec<u8> {
    let mut buf = Vec::with_capacity(clut.data.len() * 4);
    for &v in &clut.data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

/// Build GPU uniform params for CLUT shader.
pub(crate) fn clut_gpu_params(clut: &Clut3D, width: u32, height: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(16);
    buf.extend_from_slice(&width.to_le_bytes());
    buf.extend_from_slice(&height.to_le_bytes());
    buf.extend_from_slice(&clut.grid_size.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // _pad
    buf
}
