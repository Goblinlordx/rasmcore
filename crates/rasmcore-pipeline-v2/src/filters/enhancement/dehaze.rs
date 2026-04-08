use crate::node::PipelineError;
use crate::ops::Filter;

/// Dehaze — dark channel prior dehazing.
///
/// 1. Dark channel: local min over RGB in patch
/// 2. Atmospheric light: brightest 0.1% of dark channel
/// 3. Transmission: `t(x) = 1 - omega * dark(I/A)`
/// 4. Recover: `J = (I - A) / max(t, t_min) + A`
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "dehaze", category = "enhancement", cost = "O(n * r^2)")]
pub struct Dehaze {
    #[param(min = 1, max = 50, default = 7)]
    pub patch_radius: u32,
    #[param(min = 0.0, max = 1.0, default = 0.95)]
    pub omega: f32,
    #[param(min = 0.0, max = 1.0, default = 0.1)]
    pub t_min: f32,
}

impl Filter for Dehaze {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let r = self.patch_radius as usize;
        let n = w * h;

        // Step 1: Dark channel — min over RGB in local patch
        let mut dark_channel = vec![0.0f32; n];
        for y in 0..h {
            for x in 0..w {
                let mut min_val = f32::MAX;
                let y0 = y.saturating_sub(r);
                let y1 = (y + r + 1).min(h);
                let x0 = x.saturating_sub(r);
                let x1 = (x + r + 1).min(w);
                for py in y0..y1 {
                    for px in x0..x1 {
                        let idx = (py * w + px) * 4;
                        let channel_min = input[idx].min(input[idx + 1]).min(input[idx + 2]);
                        min_val = min_val.min(channel_min);
                    }
                }
                dark_channel[y * w + x] = min_val;
            }
        }

        // Step 2: Atmospheric light — average of top 0.1% brightest dark channel pixels
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_unstable_by(|&a, &b| {
            dark_channel[b]
                .partial_cmp(&dark_channel[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top_count = (n as f32 * 0.001).ceil() as usize;
        let top_count = top_count.max(1);
        let mut a_light = [0.0f32; 3];
        for &idx in &indices[..top_count.min(n)] {
            let pixel_idx = idx * 4;
            a_light[0] += input[pixel_idx];
            a_light[1] += input[pixel_idx + 1];
            a_light[2] += input[pixel_idx + 2];
        }
        let inv_count = 1.0 / top_count as f32;
        a_light[0] *= inv_count;
        a_light[1] *= inv_count;
        a_light[2] *= inv_count;

        // Step 3: Transmission estimate
        let mut transmission = vec![0.0f32; n];
        for y in 0..h {
            for x in 0..w {
                let mut min_val = f32::MAX;
                let y0 = y.saturating_sub(r);
                let y1 = (y + r + 1).min(h);
                let x0 = x.saturating_sub(r);
                let x1 = (x + r + 1).min(w);
                for py in y0..y1 {
                    for px in x0..x1 {
                        let idx = (py * w + px) * 4;
                        let nr = input[idx] / a_light[0].max(1e-10);
                        let ng = input[idx + 1] / a_light[1].max(1e-10);
                        let nb = input[idx + 2] / a_light[2].max(1e-10);
                        min_val = min_val.min(nr.min(ng).min(nb));
                    }
                }
                transmission[y * w + x] = 1.0 - self.omega * min_val;
            }
        }

        // Step 4: Recover scene
        let mut out = input.to_vec();
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 4;
                let t = transmission[y * w + x].max(self.t_min);
                let inv_t = 1.0 / t;
                for c in 0..3 {
                    out[idx + c] = (input[idx + c] - a_light[c]) * inv_t + a_light[c];
                }
            }
        }

        Ok(out)
    }
}

// ── Dehaze GPU (2-pass: dark channel + apply) ───────────────────────────

use crate::gpu_shaders::analysis;
use crate::node::GpuShader;

gpu_filter_multipass!(Dehaze,
    shader: analysis::DEHAZE_APPLY,
    workgroup: [256, 1, 1],
    params(self_, w, h) => [
        w, h,
        // Atmospheric light estimated on CPU as fallback (GPU path estimates via dark channel max)
        1.0f32, 1.0f32, 1.0f32, // atmosphere RGB
        self_.omega,
        0u32, 0u32
    ],
    passes(self2, w2, h2) => {
        let mut dark_params = Vec::with_capacity(16);
        dark_params.extend_from_slice(&w2.to_le_bytes());
        dark_params.extend_from_slice(&h2.to_le_bytes());
        dark_params.extend_from_slice(&self2.patch_radius.to_le_bytes());
        dark_params.extend_from_slice(&0u32.to_le_bytes());

        let pass1 = GpuShader::new(
            analysis::DEHAZE_DARK_CHANNEL.to_string(), "main", [16, 16, 1], dark_params,
        );
        let pass2 = GpuShader::new(
            analysis::DEHAZE_APPLY.to_string(), "main", [256, 1, 1], self2.params(w2, h2),
        );
        vec![pass1, pass2]
    }
);
