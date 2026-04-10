//! ASC Color Decision List filter.

use crate::fusion::Clut3D;
use crate::node::PipelineError;
use crate::ops::Filter;

use super::super::color::ClutOp;
use super::super::helpers::luminance;

use crate::filter_node::FilterNode;
#[allow(unused_imports)]
use crate::registry::{
    FilterFactoryRegistration, OperationCapabilities, OperationKind, OperationRegistration,
    ParamDescriptor, ParamMap, ParamType,
};

use super::ASC_CDL_PARAMS;

/// ASC Color Decision List — per-channel slope, offset, power with optional saturation.
///
/// Formula: `out = clamp01((in * slope + offset) ^ power)`
/// Optional saturation adjustment via Rec. 709 luma.
#[derive(Clone)]
pub struct AscCdl {
    pub slope: [f32; 3],
    pub offset: [f32; 3],
    pub power: [f32; 3],
    /// Overall saturation (1.0 = unchanged).
    pub saturation: f32,
}

impl Filter for AscCdl {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (r, g, b) = asc_cdl_pixel(pixel[0], pixel[1], pixel[2], self);
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

fn asc_cdl_pixel(r: f32, g: f32, b: f32, cdl: &AscCdl) -> (f32, f32, f32) {
    let mut or = ((r * cdl.slope[0] + cdl.offset[0]).max(0.0)).powf(cdl.power[0]);
    let mut og = ((g * cdl.slope[1] + cdl.offset[1]).max(0.0)).powf(cdl.power[1]);
    let mut ob = ((b * cdl.slope[2] + cdl.offset[2]).max(0.0)).powf(cdl.power[2]);
    if cdl.saturation != 1.0 {
        let luma = luminance(or, og, ob);
        or = luma + (or - luma) * cdl.saturation;
        og = luma + (og - luma) * cdl.saturation;
        ob = luma + (ob - luma) * cdl.saturation;
    }
    (or, og, ob)
}

impl ClutOp for AscCdl {
    fn build_clut(&self) -> Clut3D {
        let cdl = self.clone();
        Clut3D::from_fn(33, move |r, g, b| asc_cdl_pixel(r, g, b, &cdl))
    }
}

// ASC CDL registration
inventory::submit! { &FilterFactoryRegistration { name: "asc_cdl",
    display_name: "ASC CDL", category: "grading", params: &ASC_CDL_PARAMS, doc_path: "", cost: "O(n)",
    factory: |upstream, info, params| {
        Box::new(FilterNode::point_op(upstream, info, AscCdl {
            slope: [params.get_f32("slope_r"), params.get_f32("slope_g"), params.get_f32("slope_b")],
            offset: [params.get_f32("offset_r"), params.get_f32("offset_g"), params.get_f32("offset_b")],
            power: [params.get_f32("power_r"), params.get_f32("power_g"), params.get_f32("power_b")],
            saturation: params.get_f32("saturation"),
        }))
    },
} }
inventory::submit! { &OperationRegistration { name: "asc_cdl", display_name: "ASC CDL", category: "grading",
    kind: OperationKind::Filter, params: &ASC_CDL_PARAMS, doc_path: "", cost: "O(n)",
    capabilities: OperationCapabilities { gpu: false, analytic: false, affine: false, clut: true },
} }
