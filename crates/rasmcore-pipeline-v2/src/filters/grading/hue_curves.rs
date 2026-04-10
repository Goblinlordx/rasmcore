//! Hue-based curves — HueVsSat, HueVsLum, LumVsSat, SatVsSat.

use crate::fusion::Clut3D;
use crate::node::PipelineError;
use crate::ops::Filter;

use super::super::color::ClutOp;
use super::super::helpers::{hsl_to_rgb, rgb_to_hsl};

use crate::filter_node::FilterNode;
#[allow(unused_imports)]
use crate::registry::{
    FilterFactoryRegistration, OperationCapabilities, OperationKind, OperationRegistration,
    ParamDescriptor, ParamMap, ParamType,
};

use super::{
    HUE_CURVE_PARAMS, NORM_CURVE_PARAMS, build_curve_lut_f32, curve_from_params,
    hue_curve_from_params,
};

// ─── Hue vs Saturation ───────────────────────────────────────────────────

/// Hue vs Saturation — adjust saturation based on hue position.
#[derive(Clone)]
pub struct HueVsSat {
    /// Control points (x in [0,1] mapping to hue 0-360, y in [0,1]).
    /// y=0.5 is neutral, y>0.5 boosts, y<0.5 reduces.
    pub points: Vec<(f32, f32)>,
}

impl Filter for HueVsSat {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let lut = build_curve_lut_f32(&self.points, 360);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (h, s, l) = rgb_to_hsl(pixel[0], pixel[1], pixel[2]);
            let idx = (h.round() as usize).min(359);
            let mult = lut[idx] * 2.0;
            let (r, g, b) = hsl_to_rgb(h, (s * mult).clamp(0.0, 1.0), l);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
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

impl ClutOp for HueVsSat {
    fn build_clut(&self) -> Clut3D {
        let lut = build_curve_lut_f32(&self.points, 360);
        Clut3D::from_fn(33, move |r, g, b| {
            let (h, s, l) = rgb_to_hsl(r, g, b);
            let idx = (h.round() as usize).min(359);
            let mult = lut[idx] * 2.0;
            hsl_to_rgb(h, (s * mult).clamp(0.0, 1.0), l)
        })
    }
}

// Hue vs Saturation registration
inventory::submit! { &FilterFactoryRegistration { name: "hue_vs_sat",
    display_name: "Hue vs Sat", category: "grading", params: &HUE_CURVE_PARAMS, doc_path: "", cost: "O(n)",
    factory: |upstream, info, params| {
        let pts = hue_curve_from_params(params.get_f32("center"), params.get_f32("amount"), params.get_f32("width"));
        Box::new(FilterNode::point_op(upstream, info, HueVsSat { points: pts }))
    },
} }
inventory::submit! { &OperationRegistration { name: "hue_vs_sat", display_name: "Hue vs Sat", category: "grading",
    kind: OperationKind::Filter, params: &HUE_CURVE_PARAMS, doc_path: "", cost: "O(n)",
    capabilities: OperationCapabilities { gpu: false, analytic: false, affine: false, clut: true },
} }

// ─── Hue vs Luminance ────────────────────────────────────────────────────

/// Hue vs Luminance — adjust luminance based on hue position.
#[derive(Clone)]
pub struct HueVsLum {
    pub points: Vec<(f32, f32)>,
}

impl Filter for HueVsLum {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let lut = build_curve_lut_f32(&self.points, 360);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (h, s, l) = rgb_to_hsl(pixel[0], pixel[1], pixel[2]);
            let idx = (h.round() as usize).min(359);
            let offset = (lut[idx] - 0.5) * 2.0;
            let (r, g, b) = hsl_to_rgb(h, s, (l + offset).clamp(0.0, 1.0));
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
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

impl ClutOp for HueVsLum {
    fn build_clut(&self) -> Clut3D {
        let lut = build_curve_lut_f32(&self.points, 360);
        Clut3D::from_fn(33, move |r, g, b| {
            let (h, s, l) = rgb_to_hsl(r, g, b);
            let idx = (h.round() as usize).min(359);
            let offset = (lut[idx] - 0.5) * 2.0;
            hsl_to_rgb(h, s, (l + offset).clamp(0.0, 1.0))
        })
    }
}

// Hue vs Luminance registration
inventory::submit! { &FilterFactoryRegistration { name: "hue_vs_lum",
    display_name: "Hue vs Lum", category: "grading", params: &HUE_CURVE_PARAMS, doc_path: "", cost: "O(n)",
    factory: |upstream, info, params| {
        let pts = hue_curve_from_params(params.get_f32("center"), params.get_f32("amount"), params.get_f32("width"));
        Box::new(FilterNode::point_op(upstream, info, HueVsLum { points: pts }))
    },
} }
inventory::submit! { &OperationRegistration { name: "hue_vs_lum", display_name: "Hue vs Lum", category: "grading",
    kind: OperationKind::Filter, params: &HUE_CURVE_PARAMS, doc_path: "", cost: "O(n)",
    capabilities: OperationCapabilities { gpu: false, analytic: false, affine: false, clut: true },
} }

// ─── Lum vs Saturation ───────────────────────────────────────────────────

/// Luminance vs Saturation — adjust saturation based on luminance.
#[derive(Clone)]
pub struct LumVsSat {
    pub points: Vec<(f32, f32)>,
}

impl Filter for LumVsSat {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let lut = build_curve_lut_f32(&self.points, 256);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (h, s, l) = rgb_to_hsl(pixel[0], pixel[1], pixel[2]);
            let idx = (l * 255.0).round().clamp(0.0, 255.0) as usize;
            let mult = lut[idx] * 2.0;
            let (r, g, b) = hsl_to_rgb(h, (s * mult).clamp(0.0, 1.0), l);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
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

impl ClutOp for LumVsSat {
    fn build_clut(&self) -> Clut3D {
        let lut = build_curve_lut_f32(&self.points, 256);
        Clut3D::from_fn(33, move |r, g, b| {
            let (h, s, l) = rgb_to_hsl(r, g, b);
            let idx = (l * 255.0).round().clamp(0.0, 255.0) as usize;
            let mult = lut[idx] * 2.0;
            hsl_to_rgb(h, (s * mult).clamp(0.0, 1.0), l)
        })
    }
}

// Lum vs Saturation registration
inventory::submit! { &FilterFactoryRegistration { name: "lum_vs_sat",
    display_name: "Lum vs Sat", category: "grading", params: &NORM_CURVE_PARAMS, doc_path: "", cost: "O(n)",
    factory: |upstream, info, params| {
        let pts = curve_from_params(params.get_f32("shadows"), params.get_f32("midtones"), params.get_f32("highlights"));
        Box::new(FilterNode::point_op(upstream, info, LumVsSat { points: pts }))
    },
} }
inventory::submit! { &OperationRegistration { name: "lum_vs_sat", display_name: "Lum vs Sat", category: "grading",
    kind: OperationKind::Filter, params: &NORM_CURVE_PARAMS, doc_path: "", cost: "O(n)",
    capabilities: OperationCapabilities { gpu: false, analytic: false, affine: false, clut: true },
} }

// ─── Sat vs Saturation ───────────────────────────────────────────────────

/// Saturation vs Saturation — remap saturation based on current saturation.
#[derive(Clone)]
pub struct SatVsSat {
    pub points: Vec<(f32, f32)>,
}

impl Filter for SatVsSat {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let lut = build_curve_lut_f32(&self.points, 256);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (h, s, l) = rgb_to_hsl(pixel[0], pixel[1], pixel[2]);
            let idx = (s * 255.0).round().clamp(0.0, 255.0) as usize;
            let mult = lut[idx] * 2.0;
            let (r, g, b) = hsl_to_rgb(h, (s * mult).clamp(0.0, 1.0), l);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
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

impl ClutOp for SatVsSat {
    fn build_clut(&self) -> Clut3D {
        let lut = build_curve_lut_f32(&self.points, 256);
        Clut3D::from_fn(33, move |r, g, b| {
            let (h, s, l) = rgb_to_hsl(r, g, b);
            let idx = (s * 255.0).round().clamp(0.0, 255.0) as usize;
            let mult = lut[idx] * 2.0;
            hsl_to_rgb(h, (s * mult).clamp(0.0, 1.0), l)
        })
    }
}

// Sat vs Saturation registration
inventory::submit! { &FilterFactoryRegistration { name: "sat_vs_sat",
    display_name: "Sat vs Sat", category: "grading", params: &NORM_CURVE_PARAMS, doc_path: "", cost: "O(n)",
    factory: |upstream, info, params| {
        let pts = curve_from_params(params.get_f32("shadows"), params.get_f32("midtones"), params.get_f32("highlights"));
        Box::new(FilterNode::point_op(upstream, info, SatVsSat { points: pts }))
    },
} }
inventory::submit! { &OperationRegistration { name: "sat_vs_sat", display_name: "Sat vs Sat", category: "grading",
    kind: OperationKind::Filter, params: &NORM_CURVE_PARAMS, doc_path: "", cost: "O(n)",
    capabilities: OperationCapabilities { gpu: false, analytic: false, affine: false, clut: true },
} }
