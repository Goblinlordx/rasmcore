//! Split toning — tint shadows and highlights with different colors.

use crate::fusion::Clut3D;
use crate::node::PipelineError;
use crate::ops::Filter;

use super::super::color::ClutOp;
use super::super::helpers::luminance;

use crate::filter_node::FilterNode;
#[allow(unused_imports)]
use crate::registry::{
    FilterFactoryRegistration, OperationRegistration, OperationKind,
    OperationCapabilities, ParamDescriptor, ParamMap, ParamType,
};

use super::{hsl_to_rgb_simple, SPLIT_TONING_PARAMS};

/// Split toning — tint shadows and highlights with different colors.
#[derive(Clone)]
pub struct SplitToning {
    pub shadow_color: [f32; 3],
    pub highlight_color: [f32; 3],
    /// Balance: -1.0 (all shadow) to +1.0 (all highlight).
    pub balance: f32,
    /// Strength: 0.0 (none) to 1.0 (full).
    pub strength: f32,
}

impl Filter for SplitToning {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (r, g, b) = split_toning_pixel(pixel[0], pixel[1], pixel[2], self);
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

fn split_toning_pixel(r: f32, g: f32, b: f32, st: &SplitToning) -> (f32, f32, f32) {
    let luma = luminance(r, g, b);
    let midpoint = 0.5 + st.balance * 0.5;
    let shadow_w = (1.0 - luma / midpoint.max(0.001)).clamp(0.0, 1.0) * st.strength;
    let highlight_w = ((luma - midpoint) / (1.0 - midpoint).max(0.001)).clamp(0.0, 1.0) * st.strength;
    let or = r + (st.shadow_color[0] - r) * shadow_w + (st.highlight_color[0] - r) * highlight_w;
    let og = g + (st.shadow_color[1] - g) * shadow_w + (st.highlight_color[1] - g) * highlight_w;
    let ob = b + (st.shadow_color[2] - b) * shadow_w + (st.highlight_color[2] - b) * highlight_w;
    (or, og, ob)
}

impl ClutOp for SplitToning {
    fn build_clut(&self) -> Clut3D {
        let st = self.clone();
        Clut3D::from_fn(33, move |r, g, b| split_toning_pixel(r, g, b, &st))
    }
}

// Split Toning registration
inventory::submit! { &FilterFactoryRegistration { name: "split_toning",
    display_name: "Split Toning", category: "grading", params: &SPLIT_TONING_PARAMS, doc_path: "", cost: "O(n)",
    factory: |upstream, info, params| {
        let sh = hsl_to_rgb_simple(params.get_f32("shadow_hue"), 0.7, 0.3);
        let hh = hsl_to_rgb_simple(params.get_f32("highlight_hue"), 0.7, 0.7);
        Box::new(FilterNode::point_op(upstream, info, SplitToning {
            shadow_color: sh, highlight_color: hh,
            balance: 0.5, strength: (params.get_f32("shadow_strength") + params.get_f32("highlight_strength")) * 0.5,
        }))
    },
} }
inventory::submit! { &OperationRegistration { name: "split_toning", display_name: "Split Toning", category: "grading",
    kind: OperationKind::Filter, params: &SPLIT_TONING_PARAMS, doc_path: "", cost: "O(n)",
    capabilities: OperationCapabilities { gpu: false, analytic: false, affine: false, clut: true },
} }
