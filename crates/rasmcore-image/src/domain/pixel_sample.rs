//! Generic pixel sample abstraction for multi-precision pipeline support.
//!
//! The `PixelSample` trait abstracts over u8, u16, and f32 sample types,
//! enabling filters to be written once and monomorphized for each precision.
//! For f32, `to_f32`/`from_f32` are identity functions optimized away by LLVM.

/// A single channel value in a pixel buffer.
///
/// Implementors provide conversion to/from f32 normalized [0.0, 1.0] range.
/// The trait is designed for zero-cost abstraction: for `f32`, `to_f32` and
/// `from_f32` are identity operations that LLVM eliminates entirely.
pub trait PixelSample: Copy + Send + Sync + 'static + PartialOrd {
    /// Size of this sample in bytes (1 for u8, 2 for u16, 4 for f32).
    const BYTE_SIZE: usize;

    /// Maximum representable value as f32 (255.0 for u8, 65535.0 for u16, 1.0 for f32).
    const MAX_VALUE: f32;

    /// Convert this sample to f32 in normalized [0.0, 1.0] range.
    fn to_f32(self) -> f32;

    /// Convert an f32 in normalized [0.0, 1.0] range to this sample type.
    fn from_f32(v: f32) -> Self;

    /// Read samples from a little-endian byte buffer.
    fn from_bytes(bytes: &[u8]) -> Vec<Self>;

    /// Write samples to a little-endian byte buffer.
    fn to_bytes(samples: &[Self]) -> Vec<u8>;
}

impl PixelSample for u8 {
    const BYTE_SIZE: usize = 1;
    const MAX_VALUE: f32 = 255.0;

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32 / 255.0
    }

    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8
    }

    fn from_bytes(bytes: &[u8]) -> Vec<Self> {
        bytes.to_vec()
    }

    fn to_bytes(samples: &[Self]) -> Vec<u8> {
        samples.to_vec()
    }
}

impl PixelSample for u16 {
    const BYTE_SIZE: usize = 2;
    const MAX_VALUE: f32 = 65535.0;

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32 / 65535.0
    }

    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        (v * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16
    }

    fn from_bytes(bytes: &[u8]) -> Vec<Self> {
        bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    }

    fn to_bytes(samples: &[Self]) -> Vec<u8> {
        samples.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
}

impl PixelSample for f32 {
    const BYTE_SIZE: usize = 4;
    const MAX_VALUE: f32 = 1.0;

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }

    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        v
    }

    fn from_bytes(bytes: &[u8]) -> Vec<Self> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn to_bytes(samples: &[Self]) -> Vec<u8> {
        samples.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
}

/// Pipeline precision mode.
///
/// Controls whether the pipeline operates in standard 8-bit or
/// high-precision 32-bit float mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PipelinePrecision {
    /// Standard 8-bit processing (default). Zero overhead, backward compatible.
    #[default]
    Standard,
    /// High-precision 32-bit float processing. 4x memory per pixel.
    /// Eliminates banding, enables HDR workflows, preserves precision
    /// across long filter chains.
    HighPrecision,
}

/// Convert a byte buffer between sample types.
///
/// Reads `src` as samples of type `S`, converts each to f32, then
/// converts to type `D` and writes back to bytes.
pub fn convert_samples<S: PixelSample, D: PixelSample>(src_bytes: &[u8]) -> Vec<u8> {
    let src_samples = S::from_bytes(src_bytes);
    let dst_samples: Vec<D> = src_samples
        .iter()
        .map(|&s| D::from_f32(s.to_f32()))
        .collect();
    D::to_bytes(&dst_samples)
}

/// Determine the corresponding 8-bit format for a given f32 format.
/// Returns `None` if the format is not f32.
pub fn f32_to_standard_format(format: super::types::PixelFormat) -> Option<super::types::PixelFormat> {
    use super::types::PixelFormat;
    match format {
        PixelFormat::Rgba32f => Some(PixelFormat::Rgba8),
        PixelFormat::Rgb32f => Some(PixelFormat::Rgb8),
        PixelFormat::Gray32f => Some(PixelFormat::Gray8),
        _ => None,
    }
}

/// Determine the corresponding f32 format for a given 8-bit format.
/// Returns `None` if the format is not 8-bit RGB/RGBA/Gray.
pub fn standard_to_f32_format(format: super::types::PixelFormat) -> Option<super::types::PixelFormat> {
    use super::types::PixelFormat;
    match format {
        PixelFormat::Rgba8 => Some(PixelFormat::Rgba32f),
        PixelFormat::Rgb8 => Some(PixelFormat::Rgb32f),
        PixelFormat::Gray8 => Some(PixelFormat::Gray32f),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u8_round_trip() {
        for v in [0u8, 1, 127, 128, 254, 255] {
            let f = v.to_f32();
            let back = u8::from_f32(f);
            assert_eq!(v, back, "u8 round-trip failed for {v}");
        }
    }

    #[test]
    fn u16_round_trip() {
        for v in [0u16, 1, 32767, 32768, 65534, 65535] {
            let f = v.to_f32();
            let back = u16::from_f32(f);
            assert_eq!(v, back, "u16 round-trip failed for {v}");
        }
    }

    #[test]
    fn f32_identity() {
        for v in [0.0f32, 0.5, 1.0, 0.001, 0.999] {
            assert_eq!(v, f32::to_f32(v));
            assert_eq!(v, f32::from_f32(v));
        }
    }

    #[test]
    fn u8_to_f32_range() {
        assert_eq!(0u8.to_f32(), 0.0);
        assert_eq!(255u8.to_f32(), 1.0);
    }

    #[test]
    fn u16_to_f32_range() {
        assert_eq!(0u16.to_f32(), 0.0);
        assert_eq!(65535u16.to_f32(), 1.0);
    }

    #[test]
    fn f32_max_value() {
        assert_eq!(u8::MAX_VALUE, 255.0);
        assert_eq!(u16::MAX_VALUE, 65535.0);
        assert_eq!(f32::MAX_VALUE, 1.0);
    }

    #[test]
    fn u8_from_bytes_round_trip() {
        let data = vec![10u8, 20, 30, 40];
        let samples = u8::from_bytes(&data);
        assert_eq!(samples, data);
        let back = u8::to_bytes(&samples);
        assert_eq!(back, data);
    }

    #[test]
    fn u16_from_bytes_round_trip() {
        let values = [256u16, 512, 1024, 65535];
        let bytes = u16::to_bytes(&values);
        let back = u16::from_bytes(&bytes);
        assert_eq!(back, values);
    }

    #[test]
    fn f32_from_bytes_round_trip() {
        let values = [0.0f32, 0.25, 0.5, 1.0];
        let bytes = f32::to_bytes(&values);
        let back = f32::from_bytes(&bytes);
        assert_eq!(back, values);
    }

    #[test]
    fn convert_u8_to_f32() {
        let src = vec![0u8, 128, 255];
        let dst_bytes = convert_samples::<u8, f32>(&src);
        let dst = f32::from_bytes(&dst_bytes);
        assert_eq!(dst.len(), 3);
        assert!((dst[0] - 0.0).abs() < 1e-6);
        assert!((dst[1] - 128.0 / 255.0).abs() < 1e-6);
        assert!((dst[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn convert_f32_to_u8() {
        let src_vals = [0.0f32, 0.5, 1.0];
        let src_bytes = f32::to_bytes(&src_vals);
        let dst_bytes = convert_samples::<f32, u8>(&src_bytes);
        assert_eq!(dst_bytes, vec![0u8, 128, 255]);
    }

    #[test]
    fn convert_u8_to_u16() {
        let src = vec![0u8, 128, 255];
        let dst_bytes = convert_samples::<u8, u16>(&src);
        let dst = u16::from_bytes(&dst_bytes);
        assert_eq!(dst[0], 0);
        // 128/255 * 65535 + 0.5 ≈ 32896
        assert_eq!(dst[1], 32896);
        assert_eq!(dst[2], 65535);
    }

    #[test]
    fn pipeline_precision_default() {
        assert_eq!(PipelinePrecision::default(), PipelinePrecision::Standard);
    }
}
