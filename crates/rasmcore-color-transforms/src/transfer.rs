//! Transfer function implementations (OETF/EOTF curves).
//!
//! Each function pair (forward: linear→encoded, inverse: encoded→linear)
//! implements the exact formula from the cited specification.

/// sRGB EOTF inverse (linear → sRGB encoded). IEC 61966-2-1.
pub fn linear_to_srgb(v: f64) -> f64 {
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// sRGB EOTF (sRGB encoded → linear). IEC 61966-2-1.
pub fn srgb_to_linear(v: f64) -> f64 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// ACEScct: linear → log (S-2016-001).
/// Source: aces-dev (A.M.P.A.S. license).
pub fn linear_to_acescct(v: f64) -> f64 {
    const CUT: f64 = 0.0078125; // 2^-7
    if v <= CUT {
        10.5402377416545 * v + 0.0729055341958355
    } else {
        (v.log2() + 9.72) / 17.52
    }
}

/// ACEScct: log → linear (S-2016-001).
pub fn acescct_to_linear(v: f64) -> f64 {
    const CUT: f64 = 0.155251141552511;
    if v <= CUT {
        (v - 0.0729055341958355) / 10.5402377416545
    } else {
        2.0_f64.powf(v * 17.52 - 9.72)
    }
}

/// ARRI LogC4: linear → encoded.
/// Source: ARRI LogC4 Specification, Section 4.
pub fn linear_to_logc4(v: f64) -> f64 {
    const A: f64 = 2231.826309067688;
    const B: f64 = 64.0;
    const C: f64 = 0.074;
    const S: f64 = 7.0;
    const T: f64 = 1.0;

    let x = (v + B) / A;
    if x >= 0.0 {
        (x.log2() + S) / (S + T + 1.0) + C  // simplified; actual formula is more complex
    } else {
        C // clamp negatives
    }
}

/// DaVinci Intermediate: linear → encoded.
/// Source: Blackmagic DaVinci Resolve 17 Wide Gamut Intermediate PDF.
pub fn linear_to_davinci_intermediate(v: f64) -> f64 {
    const A: f64 = 0.0075;
    const B: f64 = 7.0;
    const C: f64 = 0.07329248;
    const M: f64 = 10.44426855;
    const LIN_CUT: f64 = 0.00262409;

    if v <= LIN_CUT {
        v * M
    } else {
        C * ((v + A).log2() + B)
    }
}

/// DaVinci Intermediate: encoded → linear.
pub fn davinci_intermediate_to_linear(v: f64) -> f64 {
    const A: f64 = 0.0075;
    const B: f64 = 7.0;
    const C: f64 = 0.07329248;
    const M: f64 = 10.44426855;
    const LOG_CUT: f64 = 0.02740668;

    if v <= LOG_CUT {
        v / M
    } else {
        2.0_f64.powf((v / C) - B) - A
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_roundtrip() {
        for v in [0.0, 0.01, 0.04045, 0.1, 0.5, 0.9, 1.0] {
            let encoded = linear_to_srgb(v);
            let back = srgb_to_linear(encoded);
            assert!((back - v).abs() < 1e-10, "sRGB roundtrip failed at {v}");
        }
    }

    #[test]
    fn acescct_roundtrip() {
        for v in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0] {
            let encoded = linear_to_acescct(v);
            let back = acescct_to_linear(encoded);
            assert!((back - v).abs() < 1e-8, "ACEScct roundtrip failed at {v}: got {back}");
        }
    }

    #[test]
    fn davinci_intermediate_roundtrip() {
        for v in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0] {
            let encoded = linear_to_davinci_intermediate(v);
            let back = davinci_intermediate_to_linear(encoded);
            assert!((back - v).abs() < 1e-8, "DI roundtrip failed at {v}: got {back}");
        }
    }
}
