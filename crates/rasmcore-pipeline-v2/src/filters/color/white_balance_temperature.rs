use crate::fusion::Clut3D;
use crate::node::PipelineError;
use crate::ops::{Filter, GpuFilter};

use super::ClutOp;

/// White balance via color temperature (Kelvin).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "white_balance_temperature", category = "color", cost = "O(n)")]
pub struct WhiteBalanceTemperature {
    /// Temperature in Kelvin (2000-12000). 6500 = daylight neutral.
    #[param(min = 2000.0, max = 12000.0, default = 6500.0)]
    pub temperature: f32,
    /// Green-magenta tint (-1 to 1).
    #[param(min = -1.0, max = 1.0, default = 0.0)]
    pub tint: f32,
}

impl Filter for WhiteBalanceTemperature {
    fn compute(&self, input: &[f32], _width: u32, _height: u32) -> Result<Vec<f32>, PipelineError> {
        let m = wb_adaptation_matrix(self.temperature, self.tint);
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            let (r, g, b) = crate::color_math::apply_3x3(&m, pixel[0], pixel[1], pixel[2]);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
        }
        Ok(out)
    }

    fn fusion_clut(&self) -> Option<crate::fusion::Clut3D> {
        Some(ClutOp::build_clut(self))
    }
}

/// Compute the CAT16 chromatic adaptation matrix for white balance.
///
/// Photography convention: temperature = source illuminant of the scene.
/// The adaptation maps from the source illuminant to D65 (display white).
/// Setting 8000K means "scene was under blue sky" → warm up to D65.
/// Setting 3200K means "scene was under tungsten" → cool down to D65.
fn wb_adaptation_matrix(temperature: f32, tint: f32) -> [f32; 9] {
    use crate::color_math::{cat16_adaptation_matrix, cie_d_illuminant_xy, tint_shift_xy, CIE_D65_XY};
    let source = cie_d_illuminant_xy(temperature);
    let source = tint_shift_xy(source, tint);
    cat16_adaptation_matrix(source, CIE_D65_XY)
}

impl ClutOp for WhiteBalanceTemperature {
    fn build_clut(&self) -> Clut3D {
        let m = wb_adaptation_matrix(self.temperature, self.tint);
        Clut3D::from_fn(33, move |r, g, b| crate::color_math::apply_3x3(&m, r, g, b))
    }
}

/// GPU shader for CAT16 white balance — 3x3 matrix multiply.
const WB_CAT16_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  m0: f32, m1: f32, m2: f32,
  m3: f32, m4: f32, m5: f32,
  m6: f32, m7: f32, m8: f32,
  _pad: u32,
}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }
  let idx = y * params.width + x;
  let px = input[idx];
  output[idx] = vec4<f32>(
    params.m0 * px.x + params.m1 * px.y + params.m2 * px.z,
    params.m3 * px.x + params.m4 * px.y + params.m5 * px.z,
    params.m6 * px.x + params.m7 * px.y + params.m8 * px.z,
    px.w,
  );
}
"#;

impl GpuFilter for WhiteBalanceTemperature {
    fn shader_body(&self) -> &str {
        WB_CAT16_WGSL
    }
    fn workgroup_size(&self) -> [u32; 3] {
        [16, 16, 1]
    }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let m = wb_adaptation_matrix(self.temperature, self.tint);
        let mut buf = Vec::with_capacity(48);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        for v in &m {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf.extend_from_slice(&0u32.to_le_bytes()); // _pad
        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_pixel(r: f32, g: f32, b: f32) -> Vec<f32> {
        vec![r, g, b, 1.0]
    }

    fn assert_rgb_close(actual: &[f32], expected: (f32, f32, f32), tol: f32, label: &str) {
        assert!(
            (actual[0] - expected.0).abs() < tol
                && (actual[1] - expected.1).abs() < tol
                && (actual[2] - expected.2).abs() < tol,
            "{label}: expected ({:.4}, {:.4}, {:.4}), got ({:.4}, {:.4}, {:.4})",
            expected.0,
            expected.1,
            expected.2,
            actual[0],
            actual[1],
            actual[2]
        );
    }

    #[test]
    fn wb_temp_neutral_is_identity() {
        let input = test_pixel(0.5, 0.5, 0.5);
        let f = WhiteBalanceTemperature {
            temperature: 6500.0,
            tint: 0.0,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        // 6500K ≠ D65 (D65 CCT ≈ 6504K). Close but not exact identity.
        assert_rgb_close(&out, (0.5, 0.5, 0.5), 0.002, "wb 6500K ≈ neutral");
    }

    #[test]
    fn wb_warm_increases_red() {
        let input = test_pixel(0.5, 0.5, 0.5);
        let f = WhiteBalanceTemperature {
            temperature: 8000.0,
            tint: 0.0,
        };
        let out = f.compute(&input, 1, 1).unwrap();
        assert!(out[0] > 0.5, "Warm WB should increase red");
        assert!(out[2] < 0.5, "Warm WB should decrease blue");
    }
}
