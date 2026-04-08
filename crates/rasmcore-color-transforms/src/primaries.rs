//! CIE xy chromaticity coordinates for standard color spaces.
//!
//! Sources cited per space. All values are mathematical constants
//! derived from CIE colorimetry standards or vendor specifications.

use crate::Primaries;

/// D65 white point (CIE standard illuminant).
pub const D65: (f64, f64) = (0.3127, 0.3290);

/// D60 white point (ACES, SMPTE ST 2065-1).
pub const D60: (f64, f64) = (0.32168, 0.33767);

// ─── Open Standards ────────────────────────────────────────────────────────

/// Rec.709 / sRGB primaries (ITU-R BT.709, IEC 61966-2-1).
pub const REC709: Primaries = Primaries {
    red: (0.6400, 0.3300),
    green: (0.3000, 0.6000),
    blue: (0.1500, 0.0600),
    white: D65,
};

/// Rec.2020 primaries (ITU-R BT.2020).
pub const REC2020: Primaries = Primaries {
    red: (0.7080, 0.2920),
    green: (0.1700, 0.7970),
    blue: (0.1310, 0.0460),
    white: D65,
};

/// Display P3 primaries (DCI-P3 chromaticities with D65 white).
pub const DISPLAY_P3: Primaries = Primaries {
    red: (0.6800, 0.3200),
    green: (0.2650, 0.6900),
    blue: (0.1500, 0.0600),
    white: D65,
};

// ─── ACES (A.M.P.A.S. license — see licenses/ACES.txt) ───────────────────

/// ACES AP0 primaries (SMPTE ST 2065-1, ACES2065-1).
pub const AP0: Primaries = Primaries {
    red: (0.7347, 0.2653),
    green: (0.0000, 1.0000),
    blue: (0.0001, -0.0770),
    white: D60,
};

/// ACES AP1 primaries (S-2014-004, ACEScg/ACEScct/ACEScc).
pub const AP1: Primaries = Primaries {
    red: (0.7130, 0.2930),
    green: (0.1650, 0.8300),
    blue: (0.1280, 0.0440),
    white: D60,
};

// ─── ARRI (see licenses/ARRI.txt) ─────────────────────────────────────────

/// ARRI Wide Gamut 4 primaries.
/// Source: ARRI LogC4 Specification PDF, Section 4.
pub const ARRI_WIDE_GAMUT_4: Primaries = Primaries {
    red: (0.7347, 0.2653),
    green: (0.1424, 0.8576),
    blue: (0.0991, -0.0308),
    white: D65,
};

// ─── Blackmagic (see licenses/BLACKMAGIC.txt) ─────────────────────────────

/// DaVinci Wide Gamut primaries.
/// Source: Blackmagic DaVinci Resolve 17 Wide Gamut Intermediate PDF.
pub const DAVINCI_WIDE_GAMUT: Primaries = Primaries {
    red: (0.8000, 0.3130),
    green: (0.1682, 0.9877),
    blue: (0.0790, -0.1155),
    white: D65,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d65_white_is_standard() {
        assert!((D65.0 - 0.3127).abs() < 1e-4);
        assert!((D65.1 - 0.3290).abs() < 1e-4);
    }

    #[test]
    fn rec709_primaries_are_standard() {
        assert!((REC709.red.0 - 0.64).abs() < 1e-4);
        assert!((REC709.green.1 - 0.60).abs() < 1e-4);
    }
}
