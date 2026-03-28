//! VP8 rate control — quality-to-quantizer mapping.
//!
//! Maps user-facing quality (1-100) to VP8 encoder parameters
//! (QP index, filter strength, filter type) matching libwebp behavior.

use crate::filter::FilterType;
use crate::quant;

/// Complete encoder parameters derived from quality setting.
#[derive(Debug, Clone)]
pub struct EncodeParams {
    /// Luma quantizer index (0-127).
    pub qp_y: u8,
    /// Chroma quantizer index (0-127).
    pub qp_uv: u8,
    /// Loop filter strength (0-63).
    pub filter_level: u8,
    /// Loop filter sharpness (0-7).
    pub filter_sharpness: u8,
    /// Loop filter type.
    pub filter_type: FilterType,
}

/// Map user-facing quality (1-100) to complete VP8 encoder parameters.
///
/// This follows the libwebp mapping:
/// - Quality → QP via `quant::quality_to_qp`
/// - QP → filter strength via strength selection heuristic
/// - UV QP is slightly lower than Y QP (better chroma quality)
/// - Filter type is Normal for quality < 80, Simple for >= 80
pub fn quality_to_params(quality: u8) -> EncodeParams {
    let quality = quality.clamp(1, 100);
    let qp_y = quant::quality_to_qp(quality);

    // UV quantizer: slightly better quality than luma (matches libwebp behavior)
    // Offset by -4 QP steps, clamped to valid range
    let qp_uv = qp_y.saturating_sub(4);

    // Filter strength: proportional to QP
    // Higher QP = more quantization noise = need stronger filtering
    // Formula approximates libwebp's SetupFilterStrength
    let filter_level = qp_to_filter_level(qp_y);

    // Sharpness: higher quality → less sharpness adjustment
    let filter_sharpness = if quality >= 80 {
        0
    } else if quality >= 50 {
        1
    } else {
        2
    };

    // Filter type: use normal (stronger) at lower quality,
    // simple (faster) at higher quality where artifacts are minimal
    let filter_type = if quality >= 80 {
        FilterType::Simple
    } else {
        FilterType::Normal
    };

    EncodeParams {
        qp_y,
        qp_uv,
        filter_level,
        filter_sharpness,
        filter_type,
    }
}

/// Map QP index to loop filter strength.
///
/// Approximates libwebp's filter strength selection:
/// - QP 0 → filter_level 0 (no filtering needed for lossless-like quality)
/// - QP 127 → filter_level 63 (maximum filtering for heavy quantization)
fn qp_to_filter_level(qp: u8) -> u8 {
    if qp == 0 {
        return 0;
    }
    // Linear mapping: filter_level ≈ qp / 2, clamped to 0-63
    // This is a good approximation of libwebp's heuristic
    (qp as u32).div_ceil(2).min(63) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_100_gives_lowest_qp() {
        let params = quality_to_params(100);
        assert_eq!(params.qp_y, 0);
        assert_eq!(params.qp_uv, 0);
        assert_eq!(params.filter_level, 0);
    }

    #[test]
    fn quality_1_gives_highest_qp() {
        let params = quality_to_params(1);
        assert_eq!(params.qp_y, 127);
        assert!(params.filter_level > 30, "should have strong filtering");
    }

    #[test]
    fn quality_75_reasonable_params() {
        let params = quality_to_params(75);
        // QP should be moderate (roughly 25-40)
        assert!(params.qp_y > 10 && params.qp_y < 60, "qp_y={}", params.qp_y);
        // UV should be better than Y
        assert!(params.qp_uv <= params.qp_y);
        // Filter should be moderate
        assert!(params.filter_level > 0);
    }

    #[test]
    fn uv_qp_never_exceeds_y_qp() {
        for q in 1..=100 {
            let params = quality_to_params(q);
            assert!(
                params.qp_uv <= params.qp_y,
                "q={q}: uv={} > y={}",
                params.qp_uv,
                params.qp_y
            );
        }
    }

    #[test]
    fn filter_level_monotonic_with_qp() {
        let mut prev_level = 0u8;
        for qp in 0..=127u8 {
            let level = qp_to_filter_level(qp);
            assert!(level >= prev_level, "filter level should increase with QP");
            prev_level = level;
        }
    }

    #[test]
    fn high_quality_uses_simple_filter() {
        let params = quality_to_params(90);
        assert_eq!(params.filter_type, FilterType::Simple);
    }

    #[test]
    fn low_quality_uses_normal_filter() {
        let params = quality_to_params(50);
        assert_eq!(params.filter_type, FilterType::Normal);
    }

    #[test]
    fn quality_clamped_to_valid_range() {
        let p0 = quality_to_params(0); // clamped to 1
        let p1 = quality_to_params(1);
        assert_eq!(p0.qp_y, p1.qp_y);

        let p200 = quality_to_params(200); // clamped to 100
        let p100 = quality_to_params(100);
        assert_eq!(p200.qp_y, p100.qp_y);
    }
}
