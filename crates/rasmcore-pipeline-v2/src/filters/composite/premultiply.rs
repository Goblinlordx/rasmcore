use crate::node::PipelineError;
use crate::ops::Filter;

/// Premultiply alpha — multiply RGB channels by the alpha channel.
///
/// `out.rgb = in.rgb * in.a`
/// `out.a = in.a`
///
/// Required before alpha compositing for correct blending math.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "premultiply", category = "composite", cost = "O(n)")]
pub struct Premultiply;

impl Filter for Premultiply {
    fn compute(&self, input: &[f32], _w: u32, _h: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        for px in out.chunks_exact_mut(4) {
            let a = px[3];
            px[0] *= a;
            px[1] *= a;
            px[2] *= a;
        }
        Ok(out)
    }

    fn gpu_shader_body(&self) -> Option<&'static str> {
        Some(PREMULTIPLY_WGSL)
    }

    fn gpu_params(&self, width: u32, height: u32) -> Option<Vec<u8>> {
        let mut buf = Vec::with_capacity(8);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        Some(buf)
    }
}

const PREMULTIPLY_WGSL: &str = r#"
struct Params { width: u32, height: u32, }
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    let idx = gid.y * params.width + gid.x;
    let p = load_pixel(idx);
    store_pixel(idx, vec4(p.rgb * p.a, p.a));
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::composite::unpremultiply::Unpremultiply;

    fn pixel(r: f32, g: f32, b: f32, a: f32) -> Vec<f32> {
        vec![r, g, b, a]
    }

    #[test]
    fn premultiply_basic() {
        let input = pixel(0.8, 0.6, 0.4, 0.5);
        let out = Premultiply.compute(&input, 1, 1).unwrap();
        assert!((out[0] - 0.4).abs() < 1e-6);
        assert!((out[1] - 0.3).abs() < 1e-6);
        assert!((out[2] - 0.2).abs() < 1e-6);
        assert!((out[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn premultiply_unpremultiply_roundtrip() {
        let input = pixel(0.7, 0.3, 0.9, 0.6);
        let pre = Premultiply.compute(&input, 1, 1).unwrap();
        let back = Unpremultiply.compute(&pre, 1, 1).unwrap();
        for i in 0..4 {
            assert!((back[i] - input[i]).abs() < 1e-5, "channel {i}: {} vs {}", back[i], input[i]);
        }
    }
}
