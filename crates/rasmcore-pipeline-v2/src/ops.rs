//! Operation traits — the f32-only API that all operations implement.
//!
//! Each trait defines a single code path. No format dispatch. No u8/u16/f32
//! branching. Every trait operates on `&[f32]` (RGBA, 4 channels per pixel).
//!
//! GPU support is opt-in: implement `GpuFilter` alongside `Filter` to provide
//! a WGSL shader body. The pipeline auto-composes it with io_f32 bindings.

use crate::node::{GpuShader, NodeInfo, PipelineError};
use crate::rect::Rect;

/// Image filter — the core processing trait.
///
/// All pixel data is `&[f32]`: RGBA, 4 floats per pixel, interleaved.
/// `input` has `width * height * 4` elements.
///
/// Returns processed pixel data with the same dimensions.
pub trait Filter {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError>;
}

/// GPU filter — provides a WGSL compute shader for GPU execution.
///
/// The pipeline auto-composes `shader_body()` with the io_f32 I/O fragment:
/// - `load_pixel(idx: u32) -> vec4<f32>` reads from input
/// - `store_pixel(idx: u32, color: vec4<f32>)` writes to output
///
/// The shader body should declare `@group(0) @binding(2) var<uniform> params: Params`
/// and the entry point function. Input/output bindings are added by the pipeline.
pub trait GpuFilter {
    /// WGSL shader body (without input/output bindings).
    fn shader_body(&self) -> &str;

    /// Entry point function name in the shader.
    fn entry_point(&self) -> &'static str {
        "main"
    }

    /// Workgroup size (x, y, z).
    fn workgroup_size(&self) -> [u32; 3] {
        [16, 16, 1]
    }

    /// Serialized uniform parameters (little-endian, 4-byte aligned).
    fn params(&self, width: u32, height: u32) -> Vec<u8>;

    /// Optional extra storage buffers (kernel weights, LUT data, etc.).
    fn extra_buffers(&self) -> Vec<Vec<u8>> {
        vec![]
    }

    /// Build a complete GpuShader from this filter's configuration.
    fn gpu_shader(&self, width: u32, height: u32) -> GpuShader {
        GpuShader {
            body: self.shader_body().to_string(),
            entry_point: self.entry_point(),
            workgroup_size: self.workgroup_size(),
            params: self.params(width, height),
            extra_buffers: self.extra_buffers(),
        }
    }
}

/// Image decoder — converts encoded bytes to f32 pixel data.
///
/// Always outputs f32 RGBA. Format-specific decoding (JPEG, PNG, EXR)
/// happens internally. The pipeline receives uniform f32 data.
pub trait Decoder {
    /// Decode image data to f32 RGBA pixels.
    ///
    /// Returns pixel data + metadata. Pixels are f32 RGBA, 4 channels,
    /// in the color space appropriate for the format (sRGB for JPEG/PNG,
    /// Linear for EXR/HDR — the promote/linearize stage handles conversion).
    fn decode(&self, data: &[u8]) -> Result<DecodedImage, PipelineError>;

    /// Check if this decoder can handle the given data (magic byte detection).
    fn can_decode(&self, data: &[u8]) -> bool;

    /// File extensions this decoder handles.
    fn extensions(&self) -> &[&str];
}

/// Image encoder — converts f32 pixel data to encoded bytes.
///
/// Always receives f32 RGBA input. Format-specific encoding (quantization,
/// gamma, compression) happens internally.
pub trait Encoder {
    /// Encode f32 RGBA pixels to the target format.
    ///
    /// The encoder handles view transform (Linear→sRGB) and quantization
    /// (f32→u8) internally based on the output format requirements.
    fn encode(
        &self,
        pixels: &[f32],
        width: u32,
        height: u32,
        quality: Option<u8>,
    ) -> Result<Vec<u8>, PipelineError>;

    /// MIME type for this format.
    fn mime_type(&self) -> &str;

    /// File extensions this encoder handles.
    fn extensions(&self) -> &[&str];
}

/// Decoded image result from a decoder.
#[derive(Debug, Clone)]
pub struct DecodedImage {
    /// f32 RGBA pixel data (width * height * 4 elements).
    pub pixels: Vec<f32>,
    /// Image metadata.
    pub info: NodeInfo,
    /// Optional ICC profile data.
    pub icc_profile: Option<Vec<u8>>,
}

/// Spatial transform — changes image dimensions.
///
/// Crop, resize, rotate, flip, pad, etc. The transform declares its
/// output dimensions and provides an inverse coordinate mapping for
/// demand-driven tile execution.
pub trait Transform {
    /// Compute output dimensions given input dimensions.
    fn output_dimensions(&self, input_width: u32, input_height: u32) -> (u32, u32);

    /// Compute the transform output for a requested region.
    ///
    /// `input` contains the upstream pixels for the region returned by
    /// `input_rect()`. The transform maps output coordinates to source
    /// coordinates and resamples.
    fn compute(
        &self,
        input: &[f32],
        input_width: u32,
        input_height: u32,
        output_rect: Rect,
    ) -> Result<Vec<f32>, PipelineError>;

    /// Compute the input region needed for a given output region.
    ///
    /// For crop: offsets the output rect into source space.
    /// For resize: inverse-scales the output rect.
    /// For rotate: computes the bounding box of the inverse rotation.
    fn input_rect(&self, output: Rect, input_width: u32, input_height: u32) -> Rect;
}

/// Analytic point operation — provides a symbolic expression for fusion.
///
/// Operations that implement this can be composed into a single expression
/// tree by the fusion optimizer, avoiding intermediate buffers.
pub trait AnalyticOp {
    /// Return the symbolic expression for this point operation.
    ///
    /// The expression is per-channel: `f(v) -> v'` where v is a single
    /// channel value. The optimizer composes multiple expressions via
    /// substitution: `f(g(v))`.
    fn expression(&self) -> PointOpExpr;
}

/// Symbolic expression tree for point operation fusion.
///
/// Represents a per-channel mathematical function that can be composed,
/// optimized (constant folding, identity elimination), and lowered to
/// different backends (f32 closure, u8 LUT, WGSL shader source).
#[derive(Debug, Clone)]
pub enum PointOpExpr {
    /// The input channel value.
    Input,
    /// A constant f32 value.
    Constant(f32),
    /// Addition: left + right.
    Add(Box<PointOpExpr>, Box<PointOpExpr>),
    /// Subtraction: left - right.
    Sub(Box<PointOpExpr>, Box<PointOpExpr>),
    /// Multiplication: left * right.
    Mul(Box<PointOpExpr>, Box<PointOpExpr>),
    /// Division: left / right.
    Div(Box<PointOpExpr>, Box<PointOpExpr>),
    /// Power: base ^ exponent.
    Pow(Box<PointOpExpr>, Box<PointOpExpr>),
    /// Clamp: clamp(value, min, max).
    Clamp(Box<PointOpExpr>, f32, f32),
    /// Floor: floor(value).
    Floor(Box<PointOpExpr>),
    /// Maximum: max(a, b).
    Max(Box<PointOpExpr>, Box<PointOpExpr>),
    /// Minimum: min(a, b).
    Min(Box<PointOpExpr>, Box<PointOpExpr>),
}

impl PointOpExpr {
    /// Compose two expressions: substitute `Input` in `outer` with `inner`.
    ///
    /// Result: outer(inner(v))
    pub fn compose(outer: &PointOpExpr, inner: &PointOpExpr) -> PointOpExpr {
        match outer {
            PointOpExpr::Input => inner.clone(),
            PointOpExpr::Constant(v) => PointOpExpr::Constant(*v),
            PointOpExpr::Add(a, b) => PointOpExpr::Add(
                Box::new(Self::compose(a, inner)),
                Box::new(Self::compose(b, inner)),
            ),
            PointOpExpr::Sub(a, b) => PointOpExpr::Sub(
                Box::new(Self::compose(a, inner)),
                Box::new(Self::compose(b, inner)),
            ),
            PointOpExpr::Mul(a, b) => PointOpExpr::Mul(
                Box::new(Self::compose(a, inner)),
                Box::new(Self::compose(b, inner)),
            ),
            PointOpExpr::Div(a, b) => PointOpExpr::Div(
                Box::new(Self::compose(a, inner)),
                Box::new(Self::compose(b, inner)),
            ),
            PointOpExpr::Pow(a, b) => PointOpExpr::Pow(
                Box::new(Self::compose(a, inner)),
                Box::new(Self::compose(b, inner)),
            ),
            PointOpExpr::Clamp(v, min, max) => {
                PointOpExpr::Clamp(Box::new(Self::compose(v, inner)), *min, *max)
            }
            PointOpExpr::Floor(v) => PointOpExpr::Floor(Box::new(Self::compose(v, inner))),
            PointOpExpr::Max(a, b) => PointOpExpr::Max(
                Box::new(Self::compose(a, inner)),
                Box::new(Self::compose(b, inner)),
            ),
            PointOpExpr::Min(a, b) => PointOpExpr::Min(
                Box::new(Self::compose(a, inner)),
                Box::new(Self::compose(b, inner)),
            ),
        }
    }

    /// Evaluate the expression for a given input value (f64 precision).
    pub fn evaluate(&self, v: f64) -> f64 {
        match self {
            PointOpExpr::Input => v,
            PointOpExpr::Constant(c) => *c as f64,
            PointOpExpr::Add(a, b) => a.evaluate(v) + b.evaluate(v),
            PointOpExpr::Sub(a, b) => a.evaluate(v) - b.evaluate(v),
            PointOpExpr::Mul(a, b) => a.evaluate(v) * b.evaluate(v),
            PointOpExpr::Div(a, b) => {
                let d = b.evaluate(v);
                if d.abs() < 1e-30 { 0.0 } else { a.evaluate(v) / d }
            }
            PointOpExpr::Pow(a, b) => a.evaluate(v).powf(b.evaluate(v)),
            PointOpExpr::Clamp(x, min, max) => x.evaluate(v).clamp(*min as f64, *max as f64),
            PointOpExpr::Floor(x) => x.evaluate(v).floor(),
            PointOpExpr::Max(a, b) => a.evaluate(v).max(b.evaluate(v)),
            PointOpExpr::Min(a, b) => a.evaluate(v).min(b.evaluate(v)),
        }
    }

    /// Bake the expression to a 256-entry u8 LUT (for u8 output boundary).
    pub fn bake_to_lut(&self) -> [u8; 256] {
        let mut lut = [0u8; 256];
        for i in 0..256 {
            let v = i as f64 / 255.0;
            let result = self.evaluate(v);
            lut[i] = (result * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        }
        lut
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brightness_expr() {
        // brightness(+0.1): v + 0.1
        let expr = PointOpExpr::Add(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(0.1)),
        );
        assert!((expr.evaluate(0.5) - 0.6).abs() < 1e-6);
    }

    #[test]
    fn compose_brightness_then_contrast() {
        // brightness(+0.1): v + 0.1
        let bright = PointOpExpr::Add(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(0.1)),
        );
        // contrast(1.5): (v - 0.5) * 1.5 + 0.5
        let contrast = PointOpExpr::Add(
            Box::new(PointOpExpr::Mul(
                Box::new(PointOpExpr::Sub(
                    Box::new(PointOpExpr::Input),
                    Box::new(PointOpExpr::Constant(0.5)),
                )),
                Box::new(PointOpExpr::Constant(1.5)),
            )),
            Box::new(PointOpExpr::Constant(0.5)),
        );

        // Compose: contrast(brightness(v))
        let composed = PointOpExpr::compose(&contrast, &bright);
        // For v=0.5: brightness gives 0.6, contrast gives (0.6-0.5)*1.5+0.5 = 0.65
        assert!((composed.evaluate(0.5) - 0.65).abs() < 1e-6);
    }

    #[test]
    fn darken_brighten_roundtrip() {
        // darken(-0.5) then brighten(+0.5) should be identity
        let darken = PointOpExpr::Add(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(-0.5)),
        );
        let brighten = PointOpExpr::Add(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(0.5)),
        );

        let composed = PointOpExpr::compose(&brighten, &darken);
        // v + (-0.5) + 0.5 = v (algebraically identity)
        for i in 0..256 {
            let v = i as f64 / 255.0;
            let result = composed.evaluate(v);
            assert!(
                (result - v).abs() < 1e-10,
                "roundtrip failed at v={v}: got {result}"
            );
        }
    }

    #[test]
    fn hdr_values_preserved() {
        // HDR: v=5.0 through brightness(+1.0) → 6.0 (no clamping)
        let bright = PointOpExpr::Add(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(1.0)),
        );
        assert!((bright.evaluate(5.0) - 6.0).abs() < 1e-10);
    }

    #[test]
    fn bake_to_lut_identity() {
        let identity = PointOpExpr::Input;
        let lut = identity.bake_to_lut();
        for i in 0..256 {
            assert_eq!(lut[i], i as u8);
        }
    }

    #[test]
    fn bake_to_lut_invert() {
        let invert = PointOpExpr::Sub(
            Box::new(PointOpExpr::Constant(1.0)),
            Box::new(PointOpExpr::Input),
        );
        let lut = invert.bake_to_lut();
        assert_eq!(lut[0], 255);
        assert_eq!(lut[255], 0);
        assert_eq!(lut[128], 127);
    }

    #[test]
    fn gamma_expr() {
        // gamma(2.2): v ^ (1/2.2)
        let gamma = PointOpExpr::Pow(
            Box::new(PointOpExpr::Input),
            Box::new(PointOpExpr::Constant(1.0 / 2.2)),
        );
        // 0.5 ^ (1/2.2) ≈ 0.7297
        assert!((gamma.evaluate(0.5) - 0.7297).abs() < 0.001);
    }
}
