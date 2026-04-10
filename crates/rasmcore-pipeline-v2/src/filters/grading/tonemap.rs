//! Tone mapping filters — Reinhard, Drago, Filmic/ACES.

use crate::fusion::Clut3D;
use crate::node::PipelineError;
use crate::ops::Filter;

use super::super::color::ClutOp;

/// Reinhard global tone mapping: `out = v / (1 + v)`.
/// Maps HDR [0, inf) to [0, 1).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "tonemap_reinhard", category = "grading", cost = "O(n)")]
pub struct TonemapReinhard;

impl Filter for TonemapReinhard {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] = pixel[0] / (1.0 + pixel[0]);
            pixel[1] = pixel[1] / (1.0 + pixel[1]);
            pixel[2] = pixel[2] / (1.0 + pixel[2]);
        }
        Ok(out)
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }
}

impl ClutOp for TonemapReinhard {
    fn build_clut(&self) -> Clut3D {
        Clut3D::from_fn(33, |r, g, b| (r / (1.0 + r), g / (1.0 + g), b / (1.0 + b)))
    }
}

/// Drago logarithmic tone mapping.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "tonemap_drago", category = "grading", cost = "O(n)")]
pub struct TonemapDrago {
    /// Maximum luminance in scene (default 1.0 for SDR).
    #[param(min = 0.0, max = 100.0, default = 1.0)]
    pub l_max: f32,
    /// Bias parameter (0.7-0.9, default 0.85).
    #[param(min = 0.5, max = 1.0, default = 0.85)]
    pub bias: f32,
}

impl Filter for TonemapDrago {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let log_max = (1.0 + self.l_max).ln();
        let bias_pow = (self.bias.ln() / 0.5f32.ln()).max(0.01);
        let mut out = input.to_vec();
        let drago = |v: f32| -> f32 {
            if v <= 0.0 {
                0.0
            } else {
                ((1.0 + v).ln() / log_max).powf(1.0 / bias_pow)
            }
        };
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] = drago(pixel[0]);
            pixel[1] = drago(pixel[1]);
            pixel[2] = drago(pixel[2]);
        }
        Ok(out)
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }
}

impl ClutOp for TonemapDrago {
    fn build_clut(&self) -> Clut3D {
        let log_max = (1.0 + self.l_max).ln();
        let bias_pow = (self.bias.ln() / 0.5f32.ln()).max(0.01);
        Clut3D::from_fn(33, move |r, g, b| {
            let drago = |v: f32| -> f32 {
                if v <= 0.0 {
                    0.0
                } else {
                    ((1.0 + v).ln() / log_max).powf(1.0 / bias_pow)
                }
            };
            (drago(r), drago(g), drago(b))
        })
    }
}

/// Filmic/ACES tone mapping (Narkowicz 2015 approximation).
///
/// Formula: `out = (x*(a*x+b)) / (x*(c*x+d) + e)`
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "tonemap_filmic", category = "grading", cost = "O(n)")]
pub struct TonemapFilmic {
    #[param(min = 0.0, max = 10.0, default = 2.51)]
    pub a: f32,
    #[param(min = 0.0, max = 1.0, default = 0.03)]
    pub b: f32,
    #[param(min = 0.0, max = 10.0, default = 2.43)]
    pub c: f32,
    #[param(min = 0.0, max = 2.0, default = 0.59)]
    pub d: f32,
    #[param(min = 0.0, max = 1.0, default = 0.14)]
    pub e: f32,
}

impl Filter for TonemapFilmic {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        let (a, b, c, d, e) = (self.a, self.b, self.c, self.d, self.e);
        let filmic = |x: f32| -> f32 { x * (a * x + b) / (x * (c * x + d) + e) };
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] = filmic(pixel[0]);
            pixel[1] = filmic(pixel[1]);
            pixel[2] = filmic(pixel[2]);
        }
        Ok(out)
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }
}

impl ClutOp for TonemapFilmic {
    fn build_clut(&self) -> Clut3D {
        let (a, b, c, d, e) = (self.a, self.b, self.c, self.d, self.e);
        Clut3D::from_fn(33, move |r, g, bi| {
            let filmic = |x: f32| -> f32 {
                let num = x * (a * x + b);
                let den = x * (c * x + d) + e;
                num / den
            };
            (filmic(r), filmic(g), filmic(bi))
        })
    }
}
