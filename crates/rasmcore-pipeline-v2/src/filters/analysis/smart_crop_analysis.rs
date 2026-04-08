//! SmartCropAnalysis filter.

/// Extract the crop analysis logic from SmartCrop for reuse.
pub fn smart_crop_find_rect(input: &[f32], width: u32, height: u32, ratio: f32) -> crate::rect::Rect {
    let w = width as usize;
    let h = height as usize;
    let cw = ((w as f32 * ratio) as usize).max(1).min(w);
    let ch = ((h as f32 * ratio) as usize).max(1).min(h);
    if cw >= w && ch >= h {
        return crate::rect::Rect::new(0, 0, width, height);
    }

    let mut best_x = 0usize;
    let mut best_y = 0usize;
    let mut best_energy = f32::NEG_INFINITY;
    let step = 4.max(cw / 10);
    for sy in (0..=h.saturating_sub(ch)).step_by(step) {
        for sx in (0..=w.saturating_sub(cw)).step_by(step) {
            let mut energy = 0.0f32;
            for y in (sy..sy + ch).step_by(4) {
                for x in (sx..sx + cw).step_by(4) {
                    if x + 1 < w {
                        let i = (y * w + x) * 4;
                        let i1 = (y * w + x + 1) * 4;
                        let gx = (input[i] - input[i1]).abs();
                        energy += gx;
                    }
                }
            }
            if energy > best_energy {
                best_energy = energy;
                best_x = sx;
                best_y = sy;
            }
        }
    }

    crate::rect::Rect::new(best_x as u32, best_y as u32, cw as u32, ch as u32)
}

/// SmartCropAnalysis — standalone analysis node that determines the optimal crop rect.
///
/// Separates "determine what to crop" from "do the crop". Produces an
/// AnalysisResult::Rect that downstream nodes can use to configure cropping.
pub struct SmartCropAnalysis {
    upstream: u32,
    info: crate::node::NodeInfo,
    ratio: f32,
}

impl SmartCropAnalysis {
    pub fn new(upstream: u32, info: crate::node::NodeInfo, ratio: f32) -> Self {
        Self { upstream, info, ratio }
    }
}

impl crate::node::Node for SmartCropAnalysis {
    fn info(&self) -> crate::node::NodeInfo {
        self.info.clone()
    }

    fn compute(
        &self,
        request: crate::rect::Rect,
        upstream: &mut dyn crate::node::Upstream,
    ) -> Result<Vec<f32>, crate::node::PipelineError> {
        // Passthrough — analysis doesn't modify pixels
        upstream.request(self.upstream, request)
    }

    fn upstream_ids(&self) -> Vec<u32> {
        vec![self.upstream]
    }
}

impl crate::staged::AnalysisNode for SmartCropAnalysis {
    fn result_type(&self) -> crate::staged::AnalysisResultType {
        crate::staged::AnalysisResultType::Rect
    }

    fn analyze(
        &self,
        input: &[f32],
        width: u32,
        height: u32,
    ) -> Result<crate::staged::AnalysisResult, crate::node::PipelineError> {
        let rect = smart_crop_find_rect(input, width, height, self.ratio);
        Ok(crate::staged::AnalysisResult::Rect(rect))
    }
}
