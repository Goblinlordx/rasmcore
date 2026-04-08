use crate::node::PipelineError;
use crate::ops::{Filter, GpuFilter};

/// White balance via gray world assumption (automatic).
/// Computes per-channel means and normalizes so all channels have equal mean.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "white_balance_gray_world", category = "color", cost = "O(n)")]
pub struct WhiteBalanceGrayWorld;

impl Filter for WhiteBalanceGrayWorld {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let pixel_count = input.len() / 4;
        if pixel_count == 0 {
            return Ok(input.to_vec());
        }
        let (mut sum_r, mut sum_g, mut sum_b) = (0.0f64, 0.0f64, 0.0f64);
        for pixel in input.chunks_exact(4) {
            sum_r += pixel[0] as f64;
            sum_g += pixel[1] as f64;
            sum_b += pixel[2] as f64;
        }
        let n = pixel_count as f64;
        let avg_r = sum_r / n;
        let avg_g = sum_g / n;
        let avg_b = sum_b / n;
        let avg_all = (avg_r + avg_g + avg_b) / 3.0;
        let sr = if avg_r > 1e-10 { (avg_all / avg_r) as f32 } else { 1.0 };
        let sg = if avg_g > 1e-10 { (avg_all / avg_g) as f32 } else { 1.0 };
        let sb = if avg_b > 1e-10 { (avg_all / avg_b) as f32 } else { 1.0 };
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] *= sr;
            pixel[1] *= sg;
            pixel[2] *= sb;
        }
        Ok(out)
    }
}

// ── WhiteBalanceGrayWorld GPU (3-pass zero-atomic via GpuReduction) ─────────

/// Pass 3 apply shader: reads channel sums from reduction buffer, applies scales.
const WB_GRAY_WORLD_APPLY_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  _pad0: u32,
  _pad1: u32,
}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> reduction: array<vec4<f32>>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }

  let sums = reduction[0]; // vec4(sum_r, sum_g, sum_b, count)
  let avg_r = sums.x / max(sums.w, 1.0);
  let avg_g = sums.y / max(sums.w, 1.0);
  let avg_b = sums.z / max(sums.w, 1.0);
  let avg_all = (avg_r + avg_g + avg_b) / 3.0;

  let scale_r = select(avg_all / avg_r, 1.0, avg_r < 0.000001);
  let scale_g = select(avg_all / avg_g, 1.0, avg_g < 0.000001);
  let scale_b = select(avg_all / avg_b, 1.0, avg_b < 0.000001);

  let pixel = input[idx];
  output[idx] = vec4<f32>(
    pixel.x * scale_r,
    pixel.y * scale_g,
    pixel.z * scale_b,
    pixel.w,
  );
}
"#;

impl GpuFilter for WhiteBalanceGrayWorld {
    fn shader_body(&self) -> &str {
        WB_GRAY_WORLD_APPLY_WGSL
    }
    fn workgroup_size(&self) -> [u32; 3] { [256, 1, 1] }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }
    fn gpu_shaders(&self, width: u32, height: u32) -> Vec<crate::node::GpuShader> {
        use crate::gpu_shaders::reduction::GpuReduction;

        let reduction = GpuReduction::channel_sum(256);
        let passes = reduction.build_passes(width, height);

        let pass3 = crate::node::GpuShader::new(
            WB_GRAY_WORLD_APPLY_WGSL.to_string(),
            "main",
            [256, 1, 1],
            self.params(width, height),
        )
        .with_reduction_buffers(vec![reduction.read_buffer(&passes)]);

        vec![passes.pass1, passes.pass2, pass3]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gray_world_equalizes_channels() {
        // Image with blue cast: R low, G medium, B high
        let input = vec![
            0.2, 0.4, 0.8, 1.0, 0.3, 0.5, 0.9, 1.0, 0.1, 0.3, 0.7, 1.0, 0.2, 0.4, 0.8, 1.0,
        ];
        let f = WhiteBalanceGrayWorld;
        let out = f.compute(&input, 2, 2).unwrap();
        // After gray world, channel means should be closer
        let avg_r: f32 = (0..4).map(|i| out[i * 4]).sum::<f32>() / 4.0;
        let avg_g: f32 = (0..4).map(|i| out[i * 4 + 1]).sum::<f32>() / 4.0;
        let avg_b: f32 = (0..4).map(|i| out[i * 4 + 2]).sum::<f32>() / 4.0;
        let spread = (avg_r - avg_g).abs().max((avg_g - avg_b).abs());
        assert!(spread < 0.01, "Gray world should equalize channels, spread={spread}");
    }
}
