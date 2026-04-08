//! SmartCrop filter.

use crate::node::{PipelineError};
use crate::ops::Filter;


/// Smart crop — crop to most salient region based on edge energy.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "smart_crop", category = "transform")]
pub struct SmartCrop {
    #[param(min = 0.1, max = 1.0, step = 0.01, default = 0.75)]
    pub ratio: f32,
}

impl Filter for SmartCrop {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize; let h = height as usize;
        let cw = ((w as f32 * self.ratio) as usize).max(1).min(w);
        let ch = ((h as f32 * self.ratio) as usize).max(1).min(h);
        if cw >= w && ch >= h { return Ok(input.to_vec()); }

        // Find region with maximum gradient energy (analysis pass)
        let mut best_x = 0; let mut best_y = 0; let mut best_energy = f32::NEG_INFINITY;
        let step = 4.max(cw / 10);
        for sy in (0..=h.saturating_sub(ch)).step_by(step) {
            for sx in (0..=w.saturating_sub(cw)).step_by(step) {
                let mut energy = 0.0f32;
                for y in (sy..sy+ch).step_by(4) {
                    for x in (sx..sx+cw).step_by(4) {
                        if x + 1 < w {
                            let i = (y * w + x) * 4;
                            let i1 = (y * w + x + 1) * 4;
                            let gx = (input[i] - input[i1]).abs();
                            energy += gx;
                        }
                    }
                }
                if energy > best_energy { best_energy = energy; best_x = sx; best_y = sy; }
            }
        }

        // Extract crop — output is smaller than input
        let mut out = vec![0.0f32; cw * ch * 4];
        for y in 0..ch {
            for x in 0..cw {
                let si = ((best_y + y) * w + best_x + x) * 4;
                let di = (y * cw + x) * 4;
                out[di..di+4].copy_from_slice(&input[si..si+4]);
            }
        }
        Ok(out)
    }

    // ## GPU status: CPU-only
    //
    // Smart crop cannot run on GPU in the current architecture because it
    // requires an **analysis→render phase separation** that doesn't exist yet:
    //
    // 1. Analysis phase: scan full image energy to determine crop window position
    // 2. Render phase: extract the crop region using the discovered position
    //
    // The position (a spatial result — an x,y rect) isn't known until after
    // the analysis completes. Our GPU reduction infrastructure produces scalar
    // results (sums, histograms) but not spatial decisions like "best rect."
    //
    // Professional GPU pipelines (Resolve, Nuke, Flame) solve this with
    // two-phase graph evaluation:
    //   Phase 1 (backwards): ROI negotiation — each node declares input region needed
    //   Phase 2 (forwards): pixel execution at determined buffer sizes
    //
    // For smart_crop, the ROI depends on content analysis, which breaks the
    // static ROI model. The solution is an explicit analysis pass that runs
    // before the render pass, producing metadata (crop rect) that configures
    // the render pass.
    //
    // This same pattern is needed for:
    //   - Content-aware resize (seam selection depends on energy analysis)
    //   - Auto white balance (correction depends on scene statistics)
    //   - Any operation where output geometry depends on input content
    //
    // Tracked as future architectural work: analysis→render pipeline phases.
}
