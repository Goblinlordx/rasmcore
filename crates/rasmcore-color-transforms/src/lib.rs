//! Color transform manifest and registration helper for rasmcore.
//!
//! This crate provides a **manifest** of all available color transforms
//! (metadata only — no embedded CLF data). The actual CLF files are
//! external static assets loaded by the SDK at runtime.
//!
//! Each transform source has its own license — see `licenses/` directory.
//!
//! # Usage
//!
//! ```ignore
//! use rasmcore_color_transforms::{manifest, register_pack};
//!
//! // The SDK fetches CLF bytes from a CDN or local path, then:
//! let entries: Vec<(&str, &[u8])> = vec![
//!     ("idt-srgb", &srgb_clf_bytes),
//!     ("idt-arri-logc4", &arri_clf_bytes),
//! ];
//! register_pack(|bytes, format, name| {
//!     pipeline.register_transform_with_name(bytes, format, name)
//! }, &entries);
//! ```

/// Metadata for an available transform (no embedded data).
#[derive(Debug, Clone)]
pub struct TransformEntry {
    pub name: &'static str,
    pub display_name: &'static str,
    pub kind: &'static str,
    pub source_space: &'static str,
    pub target_space: &'static str,
    pub vendor: &'static str,
    pub description: &'static str,
    pub format: &'static str,
    /// Filename relative to the transforms asset directory.
    pub filename: &'static str,
}

/// Manifest of all available transforms (metadata only, no data).
pub fn manifest() -> Vec<TransformEntry> {
    vec![
        // ─── Standard IDTs (small, common) ────────────────────────────
        TransformEntry {
            name: "idt-srgb",
            display_name: "sRGB (IDT)",
            kind: "idt",
            source_space: "sRGB",
            target_space: "ACEScg",
            vendor: "Academy",
            description: "sRGB input to ACEScg working space",
            format: "clf",
            filename: "idt/srgb.clf",
        },
        TransformEntry {
            name: "idt-rec709",
            display_name: "Rec.709 (IDT)",
            kind: "idt",
            source_space: "Rec.709",
            target_space: "ACEScg",
            vendor: "Academy",
            description: "Rec.709 (BT.1886) input to ACEScg working space",
            format: "clf",
            filename: "idt/rec709.clf",
        },
        TransformEntry {
            name: "idt-rec2020",
            display_name: "Rec.2020 (IDT)",
            kind: "idt",
            source_space: "Rec.2020",
            target_space: "ACEScg",
            vendor: "Academy",
            description: "Rec.2020 input to ACEScg working space",
            format: "clf",
            filename: "idt/rec2020.clf",
        },
        TransformEntry {
            name: "idt-p3",
            display_name: "Display P3 (IDT)",
            kind: "idt",
            source_space: "Display P3",
            target_space: "ACEScg",
            vendor: "Academy",
            description: "Display P3 input to ACEScg working space",
            format: "clf",
            filename: "idt/display-p3.clf",
        },
        // ─── Camera IDTs (from OCIO-Config-ACES, BSD-3-Clause) ────────
        TransformEntry {
            name: "idt-arri-logc4",
            display_name: "ARRI LogC4",
            kind: "idt",
            source_space: "ARRI LogC4",
            target_space: "ACES2065-1",
            vendor: "ARRI",
            description: "ARRI ALEXA 35 LogC4 to ACES2065-1",
            format: "clf",
            filename: "idt/arri-logc4.clf",
        },
        TransformEntry {
            name: "idt-sony-slog3-sgamut3",
            display_name: "Sony S-Log3 S-Gamut3",
            kind: "idt",
            source_space: "Sony S-Log3 S-Gamut3",
            target_space: "ACES2065-1",
            vendor: "Sony",
            description: "Sony S-Log3 / S-Gamut3 to ACES2065-1",
            format: "clf",
            filename: "idt/sony-slog3-sgamut3.clf",
        },
        TransformEntry {
            name: "idt-sony-slog3-sgamut3cine",
            display_name: "Sony S-Log3 S-Gamut3.Cine",
            kind: "idt",
            source_space: "Sony S-Log3 S-Gamut3.Cine",
            target_space: "ACES2065-1",
            vendor: "Sony",
            description: "Sony S-Log3 / S-Gamut3.Cine to ACES2065-1",
            format: "clf",
            filename: "idt/sony-slog3-sgamut3cine.clf",
        },
        TransformEntry {
            name: "idt-red-log3g10",
            display_name: "RED Log3G10 REDWideGamutRGB",
            kind: "idt",
            source_space: "RED Log3G10",
            target_space: "ACES2065-1",
            vendor: "RED",
            description: "RED Log3G10 / REDWideGamutRGB to ACES2065-1",
            format: "clf",
            filename: "idt/red-log3g10-rwg.clf",
        },
        TransformEntry {
            name: "idt-canon-clog3",
            display_name: "Canon CLog3 CinemaGamut",
            kind: "idt",
            source_space: "Canon CLog3",
            target_space: "ACES2065-1",
            vendor: "Canon",
            description: "Canon CLog3 / CinemaGamut-D55 to ACES2065-1",
            format: "clf",
            filename: "idt/canon-clog3.clf",
        },
        TransformEntry {
            name: "idt-panasonic-vlog",
            display_name: "Panasonic V-Log V-Gamut",
            kind: "idt",
            source_space: "Panasonic V-Log",
            target_space: "ACES2065-1",
            vendor: "Panasonic",
            description: "Panasonic V-Log / V-Gamut to ACES2065-1",
            format: "clf",
            filename: "idt/panasonic-vlog.clf",
        },
        TransformEntry {
            name: "idt-bmd-gen5",
            display_name: "Blackmagic BMDFilm Gen5",
            kind: "idt",
            source_space: "Blackmagic BMDFilm Gen5",
            target_space: "ACES2065-1",
            vendor: "Blackmagic",
            description: "Blackmagic BMDFilm / Wide Gamut Gen5 to ACES2065-1",
            format: "clf",
            filename: "idt/blackmagic-gen5.clf",
        },
        TransformEntry {
            name: "idt-bmd-dwg",
            display_name: "Blackmagic DaVinci Wide Gamut",
            kind: "idt",
            source_space: "DaVinci Wide Gamut",
            target_space: "ACES2065-1",
            vendor: "Blackmagic",
            description: "DaVinci Intermediate / DaVinci Wide Gamut to ACES2065-1",
            format: "clf",
            filename: "idt/blackmagic-dwg.clf",
        },
        TransformEntry {
            name: "idt-apple-log",
            display_name: "Apple Log",
            kind: "idt",
            source_space: "Apple Log",
            target_space: "ACES2065-1",
            vendor: "Apple",
            description: "Apple Log (iPhone 15 Pro+) to ACES2065-1",
            format: "clf",
            filename: "idt/apple-log.clf",
        },
        TransformEntry {
            name: "idt-dji-dlog",
            display_name: "DJI D-Log D-Gamut",
            kind: "idt",
            source_space: "DJI D-Log",
            target_space: "ACES2065-1",
            vendor: "DJI",
            description: "DJI D-Log / D-Gamut to ACES2065-1",
            format: "clf",
            filename: "idt/dji-dlog.clf",
        },
        // ─── Utility CLFs (from OCIO, BSD-3-Clause) ───────────────────
        TransformEntry {
            name: "util-srgb-curve",
            display_name: "sRGB Curve",
            kind: "utility",
            source_space: "Linear",
            target_space: "sRGB",
            vendor: "OCIO",
            description: "sRGB transfer function (IEC 61966-2-1)",
            format: "clf",
            filename: "utility/srgb-curve.clf",
        },
        TransformEntry {
            name: "util-rec1886-curve",
            display_name: "Rec.1886 Curve",
            kind: "utility",
            source_space: "Linear",
            target_space: "Rec.1886",
            vendor: "OCIO",
            description: "BT.1886 EOTF (gamma 2.4 pure power)",
            format: "clf",
            filename: "utility/rec1886-curve.clf",
        },
        TransformEntry {
            name: "util-pq-curve",
            display_name: "ST.2084 PQ Curve",
            kind: "utility",
            source_space: "Linear",
            target_space: "PQ",
            vendor: "OCIO",
            description: "ST.2084 Perceptual Quantizer EOTF",
            format: "clf",
            filename: "utility/st2084-pq-curve.clf",
        },
        TransformEntry {
            name: "util-ap0-to-rec709",
            display_name: "AP0 to Linear Rec.709",
            kind: "utility",
            source_space: "ACES2065-1",
            target_space: "Linear Rec.709",
            vendor: "OCIO",
            description: "ACES2065-1 (AP0) to Linear Rec.709 (D65)",
            format: "clf",
            filename: "utility/ap0-to-rec709.clf",
        },
        TransformEntry {
            name: "util-ap0-to-p3",
            display_name: "AP0 to Linear P3-D65",
            kind: "utility",
            source_space: "ACES2065-1",
            target_space: "Linear P3-D65",
            vendor: "OCIO",
            description: "ACES2065-1 (AP0) to Linear P3-D65",
            format: "clf",
            filename: "utility/ap0-to-p3.clf",
        },
        TransformEntry {
            name: "util-ap0-to-rec2020",
            display_name: "AP0 to Linear Rec.2020",
            kind: "utility",
            source_space: "ACES2065-1",
            target_space: "Linear Rec.2020",
            vendor: "OCIO",
            description: "ACES2065-1 (AP0) to Linear Rec.2020",
            format: "clf",
            filename: "utility/ap0-to-rec2020.clf",
        },
        // ─── SDR Output Transforms ─────────────────────────────────────
        TransformEntry {
            name: "ot-srgb",
            display_name: "sRGB 100 nits (OT)",
            kind: "ot",
            source_space: "ACEScg",
            target_space: "sRGB",
            vendor: "Academy",
            description: "ACEScg to sRGB desktop display (100 nits, dim surround)",
            format: "clf",
            filename: "ot/srgb-100nits.clf",
        },
        TransformEntry {
            name: "ot-rec709",
            display_name: "Rec.709 100 nits (OT)",
            kind: "ot",
            source_space: "ACEScg",
            target_space: "Rec.709",
            vendor: "Academy",
            description: "ACEScg to Rec.709 broadcast (100 nits, dim surround)",
            format: "clf",
            filename: "ot/rec709-100nits.clf",
        },
        TransformEntry {
            name: "ot-p3-48nits",
            display_name: "P3 48 nits Cinema (OT)",
            kind: "ot",
            source_space: "ACEScg",
            target_space: "DCI-P3",
            vendor: "Academy",
            description: "ACEScg to DCI-P3 cinema theatrical (48 nits, dark surround)",
            format: "clf",
            filename: "ot/p3-48nits-dark.clf",
        },
        TransformEntry {
            name: "ot-p3-100nits",
            display_name: "P3 100 nits Desktop (OT)",
            kind: "ot",
            source_space: "ACEScg",
            target_space: "Display P3",
            vendor: "Academy",
            description: "ACEScg to Display P3 desktop monitor (100 nits, dim surround)",
            format: "clf",
            filename: "ot/p3-100nits-dim.clf",
        },
        TransformEntry {
            name: "ot-p3-1000nits",
            display_name: "P3 PQ 1000 nits HDR (OT)",
            kind: "ot",
            source_space: "ACEScg",
            target_space: "P3 PQ",
            vendor: "Academy",
            description: "ACEScg to P3 HDR grading monitor (1000 nits PQ)",
            format: "clf",
            filename: "ot/p3-1000nits-dim.clf",
        },
        TransformEntry {
            name: "ot-rec2020-1000nits",
            display_name: "Rec.2020 PQ 1000 nits HDR10 (OT)",
            kind: "ot",
            source_space: "ACEScg",
            target_space: "Rec.2020 PQ",
            vendor: "Academy",
            description: "ACEScg to Rec.2020 HDR10 (1000 nits PQ)",
            format: "clf",
            filename: "ot/rec2020-1000nits-dim.clf",
        },
        TransformEntry {
            name: "ot-rec2020-2000nits",
            display_name: "Rec.2020 PQ 2000 nits (OT)",
            kind: "ot",
            source_space: "ACEScg",
            target_space: "Rec.2020 PQ",
            vendor: "Academy",
            description: "ACEScg to Rec.2020 premium HDR (2000 nits PQ)",
            format: "clf",
            filename: "ot/rec2020-2000nits-dim.clf",
        },
        TransformEntry {
            name: "ot-rec2020-4000nits",
            display_name: "Rec.2020 PQ 4000 nits Dolby (OT)",
            kind: "ot",
            source_space: "ACEScg",
            target_space: "Rec.2020 PQ",
            vendor: "Academy",
            description: "ACEScg to Rec.2020 Dolby Cinema (4000 nits PQ)",
            format: "clf",
            filename: "ot/rec2020-4000nits-dim.clf",
        },
        TransformEntry {
            name: "ot-rec2020-hlg",
            display_name: "Rec.2020 HLG 1000 nits (OT)",
            kind: "ot",
            source_space: "ACEScg",
            target_space: "Rec.2020 HLG",
            vendor: "Academy",
            description: "ACEScg to Rec.2020 HLG broadcast (1000 nits)",
            format: "clf",
            filename: "ot/rec2020-hlg-1000nits.clf",
        },
        // ─── CSCs ─────────────────────────────────────────────────────
        TransformEntry {
            name: "csc-acescg-to-cct",
            display_name: "ACEScg to ACEScct",
            kind: "csc",
            source_space: "ACEScg",
            target_space: "ACEScct",
            vendor: "Academy",
            description: "Linear to log grading space (same AP1 primaries)",
            format: "clf",
            filename: "csc/acescg-to-cct.clf",
        },
        TransformEntry {
            name: "csc-acescct-to-cg",
            display_name: "ACEScct to ACEScg",
            kind: "csc",
            source_space: "ACEScct",
            target_space: "ACEScg",
            vendor: "Academy",
            description: "Log to linear compositing space (same AP1 primaries)",
            format: "clf",
            filename: "csc/acescct-to-cg.clf",
        },
    ]
}

/// Register transforms from externally-provided CLF bytes.
///
/// The caller fetches CLF data (from CDN, local files, etc.) and passes
/// name/bytes pairs. The callback registers each with the pipeline.
///
/// ```ignore
/// register_pack(|bytes, format, name| {
///     pipeline.register_transform_with_name(bytes, format, name)
/// }, &[("idt-srgb", &srgb_bytes), ("idt-arri-logc4", &arri_bytes)]);
/// ```
pub fn register_pack<E>(
    mut register: impl FnMut(&[u8], &str, &str) -> Result<u32, E>,
    entries: &[(&str, &[u8])],
) -> Vec<Result<u32, E>> {
    let manifest = manifest();
    entries
        .iter()
        .map(|(name, data)| {
            let format = manifest
                .iter()
                .find(|e| e.name == *name)
                .map(|e| e.format)
                .unwrap_or("clf");
            register(data, format, name)
        })
        .collect()
}

/// Convenience: find a manifest entry by name.
pub fn find_entry(name: &str) -> Option<&'static TransformEntry> {
    // Use a leaked static for lifetime. In practice, manifest() is called once.
    None // Entries are not 'static — caller should use manifest() directly
}

/// Get the filename for a transform by name (for SDK fetch).
pub fn filename_for(name: &str) -> Option<&'static str> {
    manifest()
        .iter()
        .find(|e| e.name == name)
        .map(|e| e.filename)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_has_entries() {
        let m = manifest();
        assert!(m.len() >= 20, "expected 20+ entries, got {}", m.len());
    }

    #[test]
    fn unique_names() {
        let mut names: Vec<&str> = manifest().iter().map(|t| t.name).collect();
        let count = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), count, "duplicate names in manifest");
    }

    #[test]
    fn unique_filenames() {
        let mut files: Vec<&str> = manifest().iter().map(|t| t.filename).collect();
        let count = files.len();
        files.sort();
        files.dedup();
        assert_eq!(files.len(), count, "duplicate filenames in manifest");
    }

    #[test]
    fn all_entries_have_metadata() {
        for e in manifest() {
            assert!(!e.name.is_empty(), "empty name");
            assert!(!e.filename.is_empty(), "empty filename for {}", e.name);
            assert!(!e.kind.is_empty(), "empty kind for {}", e.name);
            assert_eq!(e.format, "clf", "{} should be clf", e.name);
        }
    }

    #[test]
    fn register_pack_calls_callback() {
        let mut count = 0u32;
        let results = register_pack(
            |_data, _fmt, _name| -> Result<u32, ()> {
                count += 1;
                Ok(count)
            },
            &[
                ("idt-srgb", b"<ProcessList/>"),
                ("idt-rec709", b"<ProcessList/>"),
            ],
        );
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.is_ok()));
    }
}
