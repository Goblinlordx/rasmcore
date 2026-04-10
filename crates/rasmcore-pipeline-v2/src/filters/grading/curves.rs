//! Per-channel tone curves — master, red, green, blue.

use crate::fusion::Clut3D;
use crate::node::PipelineError;
use crate::ops::Filter;

use super::super::color::ClutOp;

use crate::filter_node::FilterNode;
#[allow(unused_imports)]
use crate::registry::{
    FilterFactoryRegistration, OperationCapabilities, OperationKind, OperationRegistration,
    ParamDescriptor, ParamMap, ParamType,
};

use super::{CURVE_PARAMS, build_curve_lut_f32, curve_from_params};

// ─── Curves Master ────────────────────────────────────────────────────────

/// Per-channel tone curves — master (same curve for R, G, B).
#[derive(Clone)]
pub struct CurvesMaster {
    /// Control points as (x, y) pairs in [0,1].
    pub points: Vec<(f32, f32)>,
}

impl Filter for CurvesMaster {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let lut = build_curve_lut_f32(&self.points, 4096);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] = lut[(pixel[0] * 4095.0).round().clamp(0.0, 4095.0) as usize];
            pixel[1] = lut[(pixel[1] * 4095.0).round().clamp(0.0, 4095.0) as usize];
            pixel[2] = lut[(pixel[2] * 4095.0).round().clamp(0.0, 4095.0) as usize];
        }
        Ok(out)
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }

    fn preferred_color_space(&self) -> Option<crate::color_space::ColorSpace> {
        Some(crate::color_space::ColorSpace::AcesCct)
    }
}

impl ClutOp for CurvesMaster {
    fn build_clut(&self) -> Clut3D {
        let lut = build_curve_lut_f32(&self.points, 4096);
        Clut3D::from_fn(33, move |r, g, b| {
            let ri = (r * 4095.0).round().clamp(0.0, 4095.0) as usize;
            let gi = (g * 4095.0).round().clamp(0.0, 4095.0) as usize;
            let bi = (b * 4095.0).round().clamp(0.0, 4095.0) as usize;
            (lut[ri], lut[gi], lut[bi])
        })
    }
}

// Curves Master registration
inventory::submit! { &FilterFactoryRegistration { name: "curves_master",
    display_name: "Curves (Master)", category: "grading", params: &CURVE_PARAMS, doc_path: "", cost: "O(n)",
    factory: |upstream, info, params| {
        let pts = curve_from_params(params.get_f32("shadows"), params.get_f32("midtones"), params.get_f32("highlights"));
        Box::new(FilterNode::point_op(upstream, info, CurvesMaster { points: pts }))
    },
} }
inventory::submit! { &OperationRegistration { name: "curves_master", display_name: "Curves (Master)", category: "grading",
    kind: OperationKind::Filter, params: &CURVE_PARAMS, doc_path: "", cost: "O(n)",
    capabilities: OperationCapabilities { gpu: false, analytic: false, affine: false, clut: true },
} }

// ─── Curves Red ───────────────────────────────────────────────────────────

/// Per-channel tone curve — red channel only.
#[derive(Clone)]
pub struct CurvesRed {
    pub points: Vec<(f32, f32)>,
}

impl Filter for CurvesRed {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let lut = build_curve_lut_f32(&self.points, 4096);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let idx = (pixel[0] * 4095.0).round().clamp(0.0, 4095.0) as usize;
            pixel[0] = lut[idx];
        }
        Ok(out)
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }

    fn preferred_color_space(&self) -> Option<crate::color_space::ColorSpace> {
        Some(crate::color_space::ColorSpace::AcesCct)
    }
}

impl ClutOp for CurvesRed {
    fn build_clut(&self) -> Clut3D {
        let lut = build_curve_lut_f32(&self.points, 4096);
        Clut3D::from_fn(33, move |r, g, b| {
            let ri = (r * 4095.0).round().clamp(0.0, 4095.0) as usize;
            (lut[ri], g, b)
        })
    }
}

// Curves Red registration
inventory::submit! { &FilterFactoryRegistration { name: "curves_red",
    display_name: "Curves (Red)", category: "grading", params: &CURVE_PARAMS, doc_path: "", cost: "O(n)",
    factory: |upstream, info, params| {
        let pts = curve_from_params(params.get_f32("shadows"), params.get_f32("midtones"), params.get_f32("highlights"));
        Box::new(FilterNode::point_op(upstream, info, CurvesRed { points: pts }))
    },
} }
inventory::submit! { &OperationRegistration { name: "curves_red", display_name: "Curves (Red)", category: "grading",
    kind: OperationKind::Filter, params: &CURVE_PARAMS, doc_path: "", cost: "O(n)",
    capabilities: OperationCapabilities { gpu: false, analytic: false, affine: false, clut: true },
} }

// ─── Curves Green ─────────────────────────────────────────────────────────

/// Per-channel tone curve — green channel only.
#[derive(Clone)]
pub struct CurvesGreen {
    pub points: Vec<(f32, f32)>,
}

impl Filter for CurvesGreen {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let lut = build_curve_lut_f32(&self.points, 4096);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let idx = (pixel[1] * 4095.0).round().clamp(0.0, 4095.0) as usize;
            pixel[1] = lut[idx];
        }
        Ok(out)
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }

    fn preferred_color_space(&self) -> Option<crate::color_space::ColorSpace> {
        Some(crate::color_space::ColorSpace::AcesCct)
    }
}

impl ClutOp for CurvesGreen {
    fn build_clut(&self) -> Clut3D {
        let lut = build_curve_lut_f32(&self.points, 4096);
        Clut3D::from_fn(33, move |r, g, b| {
            let gi = (g * 4095.0).round().clamp(0.0, 4095.0) as usize;
            (r, lut[gi], b)
        })
    }
}

// Curves Green registration
inventory::submit! { &FilterFactoryRegistration { name: "curves_green",
    display_name: "Curves (Green)", category: "grading", params: &CURVE_PARAMS, doc_path: "", cost: "O(n)",
    factory: |upstream, info, params| {
        let pts = curve_from_params(params.get_f32("shadows"), params.get_f32("midtones"), params.get_f32("highlights"));
        Box::new(FilterNode::point_op(upstream, info, CurvesGreen { points: pts }))
    },
} }
inventory::submit! { &OperationRegistration { name: "curves_green", display_name: "Curves (Green)", category: "grading",
    kind: OperationKind::Filter, params: &CURVE_PARAMS, doc_path: "", cost: "O(n)",
    capabilities: OperationCapabilities { gpu: false, analytic: false, affine: false, clut: true },
} }

// ─── Curves Blue ──────────────────────────────────────────────────────────

/// Per-channel tone curve — blue channel only.
#[derive(Clone)]
pub struct CurvesBlue {
    pub points: Vec<(f32, f32)>,
}

impl Filter for CurvesBlue {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let lut = build_curve_lut_f32(&self.points, 4096);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let idx = (pixel[2] * 4095.0).round().clamp(0.0, 4095.0) as usize;
            pixel[2] = lut[idx];
        }
        Ok(out)
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }

    fn preferred_color_space(&self) -> Option<crate::color_space::ColorSpace> {
        Some(crate::color_space::ColorSpace::AcesCct)
    }
}

impl ClutOp for CurvesBlue {
    fn build_clut(&self) -> Clut3D {
        let lut = build_curve_lut_f32(&self.points, 4096);
        Clut3D::from_fn(33, move |r, g, b| {
            let bi = (b * 4095.0).round().clamp(0.0, 4095.0) as usize;
            (r, g, lut[bi])
        })
    }
}

// Curves Blue registration
inventory::submit! { &FilterFactoryRegistration { name: "curves_blue",
    display_name: "Curves (Blue)", category: "grading", params: &CURVE_PARAMS, doc_path: "", cost: "O(n)",
    factory: |upstream, info, params| {
        let pts = curve_from_params(params.get_f32("shadows"), params.get_f32("midtones"), params.get_f32("highlights"));
        Box::new(FilterNode::point_op(upstream, info, CurvesBlue { points: pts }))
    },
} }
inventory::submit! { &OperationRegistration { name: "curves_blue", display_name: "Curves (Blue)", category: "grading",
    kind: OperationKind::Filter, params: &CURVE_PARAMS, doc_path: "", cost: "O(n)",
    capabilities: OperationCapabilities { gpu: false, analytic: false, affine: false, clut: true },
} }
