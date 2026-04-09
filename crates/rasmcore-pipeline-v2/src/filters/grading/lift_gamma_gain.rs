//! Lift/Gamma/Gain 3-way color corrector.

use crate::fusion::Clut3D;
use crate::node::PipelineError;
use crate::ops::Filter;

use super::super::color::ClutOp;

use crate::filter_node::FilterNode;
#[allow(unused_imports)]
use crate::registry::{
    FilterFactoryRegistration, OperationRegistration, OperationKind,
    OperationCapabilities, ParamDescriptor, ParamMap, ParamType,
};

use super::LIFT_GAMMA_GAIN_PARAMS;

/// 3-way color corrector — DaVinci Resolve style lift/gamma/gain per channel.
///
/// Formula: `out = gain * (input + lift * (1 - input)) ^ (1/gamma)`
#[derive(Clone)]
pub struct LiftGammaGain {
    pub lift: [f32; 3],
    pub gamma: [f32; 3],
    pub gain: [f32; 3],
}

impl Filter for LiftGammaGain {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (r, g, b) = lgg_pixel(pixel[0], pixel[1], pixel[2], self);
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

fn lgg_channel(val: f32, lift: f32, gamma: f32, gain: f32) -> f32 {
    let lifted = val + lift * (1.0 - val);
    let gammaed = if gamma > 0.0 && lifted > 0.0 {
        lifted.powf(1.0 / gamma)
    } else {
        0.0
    };
    gain * gammaed
}

fn lgg_pixel(r: f32, g: f32, b: f32, lgg: &LiftGammaGain) -> (f32, f32, f32) {
    (
        lgg_channel(r, lgg.lift[0], lgg.gamma[0], lgg.gain[0]),
        lgg_channel(g, lgg.lift[1], lgg.gamma[1], lgg.gain[1]),
        lgg_channel(b, lgg.lift[2], lgg.gamma[2], lgg.gain[2]),
    )
}

impl ClutOp for LiftGammaGain {
    fn build_clut(&self) -> Clut3D {
        let lgg = self.clone();
        Clut3D::from_fn(33, move |r, g, b| lgg_pixel(r, g, b, &lgg))
    }
}

// Lift/Gamma/Gain registration
inventory::submit! { &FilterFactoryRegistration { name: "lift_gamma_gain",
    display_name: "Lift/Gamma/Gain", category: "grading", params: &LIFT_GAMMA_GAIN_PARAMS, doc_path: "", cost: "O(n)",
    factory: |upstream, info, params| {
        Box::new(FilterNode::point_op(upstream, info, LiftGammaGain {
            lift: [params.get_f32("lift_r"), params.get_f32("lift_g"), params.get_f32("lift_b")],
            gamma: [params.get_f32("gamma_r"), params.get_f32("gamma_g"), params.get_f32("gamma_b")],
            gain: [params.get_f32("gain_r"), params.get_f32("gain_g"), params.get_f32("gain_b")],
        }))
    },
} }
inventory::submit! { &OperationRegistration { name: "lift_gamma_gain", display_name: "Lift/Gamma/Gain", category: "grading",
    kind: OperationKind::Filter, params: &LIFT_GAMMA_GAIN_PARAMS, doc_path: "", cost: "O(n)",
    capabilities: OperationCapabilities { gpu: false, analytic: false, affine: false, clut: true },
} }
