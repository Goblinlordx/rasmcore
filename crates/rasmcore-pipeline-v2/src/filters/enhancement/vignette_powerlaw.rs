use crate::node::PipelineError;
use crate::ops::{Filter, GpuFilter};

/// Power-law vignette — simple radial falloff.
///
/// `factor = 1.0 - strength * (dist / max_dist) ^ falloff`
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "vignette_powerlaw", category = "enhancement", cost = "O(n)")]
pub struct VignettePowerlaw {
    #[param(min = 0.0, max = 1.0, default = 0.5)]
    pub strength: f32,
    #[param(min = 0.5, max = 5.0, default = 2.0)]
    pub falloff: f32,
}

impl Filter for VignettePowerlaw {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let cx = w as f32 / 2.0;
        let cy = h as f32 / 2.0;
        let max_dist = (cx * cx + cy * cy).sqrt();
        if max_dist < 1e-10 {
            return Ok(input.to_vec());
        }
        let inv_max = 1.0 / max_dist;

        let mut out = input.to_vec();
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt() * inv_max;
                let factor = (1.0 - self.strength * dist.powf(self.falloff)).max(0.0);
                let idx = (y * w + x) * 4;
                out[idx] *= factor;
                out[idx + 1] *= factor;
                out[idx + 2] *= factor;
            }
        }

        Ok(out)
    }
}

// ── VignettePowerlaw GPU ────────────────────────────────────────────────

const VIGNETTE_POWERLAW_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  strength: f32,
  falloff: f32,
  inv_max_dist: f32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let idx = gid.y * params.width + gid.x;
  let pixel = load_pixel(idx);
  let cx = f32(params.width) / 2.0;
  let cy = f32(params.height) / 2.0;
  let dx = f32(gid.x) - cx;
  let dy = f32(gid.y) - cy;
  let dist = sqrt(dx * dx + dy * dy) * params.inv_max_dist;
  let factor = max(1.0 - params.strength * pow(dist, params.falloff), 0.0);
  store_pixel(idx, vec4<f32>(pixel.x * factor, pixel.y * factor, pixel.z * factor, pixel.w));
}
"#;

impl GpuFilter for VignettePowerlaw {
    fn shader_body(&self) -> &str { VIGNETTE_POWERLAW_WGSL }
    fn workgroup_size(&self) -> [u32; 3] { [16, 16, 1] }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let cx = width as f32 / 2.0;
        let cy = height as f32 / 2.0;
        let max_dist = (cx * cx + cy * cy).sqrt().max(1.0);
        let mut buf = Vec::with_capacity(32);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&self.strength.to_le_bytes());
        buf.extend_from_slice(&self.falloff.to_le_bytes());
        buf.extend_from_slice(&(1.0f32 / max_dist).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }
}
