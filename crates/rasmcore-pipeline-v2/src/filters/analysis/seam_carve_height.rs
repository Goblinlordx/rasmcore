//! SeamCarveHeight filter.

use crate::node::PipelineError;
use crate::ops::Filter;

use super::SeamCarveWidth;

/// Content-aware height reduction via seam carving.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "seam_carve_height", category = "transform")]
pub struct SeamCarveHeight {
    #[param(min = 1, max = 500, step = 1, default = 50)]
    pub seams: u32,
}

impl Filter for SeamCarveHeight {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        // Transpose → seam_carve_width → transpose back
        let mut transposed = vec![0.0f32; input.len()];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let si = (y * width as usize + x) * 4;
                let di = (x * height as usize + y) * 4;
                transposed[di..di + 4].copy_from_slice(&input[si..si + 4]);
            }
        }
        let scw = SeamCarveWidth { seams: self.seams };
        let carved = scw.compute(&transposed, height, width)?;
        // Transpose back
        let mut out = vec![0.0f32; (width * height * 4) as usize];
        for y in 0..width as usize {
            for x in 0..height as usize {
                let si = (y * height as usize + x) * 4;
                let di = (x * width as usize + y) * 4;
                out[di..di + 4].copy_from_slice(&carved[si..si + 4]);
            }
        }
        Ok(out)
    }
}
