//! Proof-of-concept: invert filter using derive(Filter) pattern.
//!
//! This demonstrates the new unified registration architecture.
//! The old invert filter (in point_ops) continues to work alongside this.

use crate::domain::error::ImageError;
use crate::domain::filter_traits::{CpuFilter, PointOp};
use crate::domain::types::ImageInfo;
use rasmcore_pipeline::Rect;

/// Invert colors (negative) — new-style derive(Filter) registration.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "invert_v2", category = "adjustment", reference = "color inversion (derive(Filter) PoC)")]
pub struct InvertV2 {
    /// Strength of inversion (1.0 = full invert, 0.0 = no change)
    #[param(min = 0.0, max = 1.0, step = 0.1, default = 1.0)]
    pub strength: f32,
}

impl CpuFilter for InvertV2 {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
        let pixels = upstream(request)?;
        let channels = crate::domain::types::bytes_per_pixel(info.format) as usize;
        let has_alpha = channels == 4;

        let mut output = pixels.clone();
        for i in 0..output.len() {
            if has_alpha && i % 4 == 3 {
                continue; // preserve alpha
            }
            let orig = output[i];
            let inverted = 255 - orig;
            // Blend between original and inverted based on strength
            output[i] = ((orig as f32 * (1.0 - self.strength) + inverted as f32 * self.strength) + 0.5) as u8;
        }

        Ok(output)
    }
}

impl PointOp for InvertV2 {
    fn build_lut(&self) -> [u8; 256] {
        let mut lut = [0u8; 256];
        for (i, entry) in lut.iter_mut().enumerate() {
            let orig = i as f32;
            let inverted = 255.0 - orig;
            *entry = ((orig * (1.0 - self.strength) + inverted * self.strength) + 0.5) as u8;
        }
        lut
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invert_v2_default_is_full_invert() {
        let filter = InvertV2::default();
        assert_eq!(filter.strength, 1.0);

        let lut = filter.build_lut();
        assert_eq!(lut[0], 255);
        assert_eq!(lut[255], 0);
        assert_eq!(lut[128], 127);
    }

    #[test]
    fn invert_v2_zero_strength_is_identity() {
        let filter = InvertV2 { strength: 0.0 };
        let lut = filter.build_lut();
        for (i, &v) in lut.iter().enumerate() {
            assert_eq!(v, i as u8);
        }
    }

    #[test]
    fn invert_v2_param_descriptors() {
        let descriptors = InvertV2::param_descriptors();
        assert_eq!(descriptors.len(), 1);
        assert_eq!(descriptors[0].name, "strength");
        assert_eq!(descriptors[0].param_type, "f32");
        assert_eq!(descriptors[0].default_val, "1.0");
    }

    #[test]
    fn invert_v2_registered() {
        // Verify it appears in the filter registry
        let filters = crate::domain::filter_registry::registered_filters();
        let found = filters.iter().any(|f| f.name == "invert_v2");
        assert!(found, "invert_v2 should be in the filter registry");
    }

    #[test]
    fn invert_v2_cpu_filter_works() {
        let filter = InvertV2 { strength: 1.0 };
        let pixels = vec![0u8, 128, 255]; // RGB
        let info = ImageInfo {
            width: 1,
            height: 1,
            format: crate::domain::types::PixelFormat::Rgb8,
            color_space: crate::domain::types::ColorSpace::Srgb,
        };

        let result = filter.compute(
            Rect::new(0, 0, 1, 1),
            &mut move |_| Ok(pixels.clone()),
            &info,
        ).unwrap();

        assert_eq!(result[0], 255); // 0 → 255
        assert_eq!(result[1], 127); // 128 → 127
        assert_eq!(result[2], 0);   // 255 → 0
    }
}
