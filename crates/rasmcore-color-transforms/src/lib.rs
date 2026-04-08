//! Built-in color transform definitions for rasmcore.
//!
//! This crate contains CLF (Common LUT Format) files for standard color
//! transforms, embedded via `include_bytes!()`. It has zero pipeline
//! dependencies — it just provides the raw bytes and a registration helper.
//!
//! Each transform source has its own license — see `licenses/` directory.
//!
//! # Usage
//!
//! ```ignore
//! use rasmcore_color_transforms::register_builtins;
//!
//! register_builtins(|bytes, format, name| {
//!     pipeline.register_transform_with_name(bytes, format, name)
//! });
//! ```

/// Metadata for a built-in transform.
#[derive(Debug, Clone)]
pub struct BuiltinTransform {
    pub name: &'static str,
    pub display_name: &'static str,
    pub kind: &'static str,
    pub source_space: &'static str,
    pub target_space: &'static str,
    pub vendor: &'static str,
    pub description: &'static str,
    pub format: &'static str,
    pub data: &'static [u8],
}

/// All built-in transforms with their embedded CLF data.
pub fn builtins() -> Vec<BuiltinTransform> {
    vec![
        BuiltinTransform {
            name: "idt-srgb", display_name: "sRGB (IDT)", kind: "idt",
            source_space: "sRGB", target_space: "ACEScg", vendor: "Academy",
            description: "sRGB input to ACEScg working space",
            format: "clf", data: include_bytes!("../transforms/idt/srgb.clf"),
        },
        BuiltinTransform {
            name: "idt-rec709", display_name: "Rec.709 (IDT)", kind: "idt",
            source_space: "Rec.709", target_space: "ACEScg", vendor: "Academy",
            description: "Rec.709 (BT.1886) input to ACEScg working space",
            format: "clf", data: include_bytes!("../transforms/idt/rec709.clf"),
        },
        BuiltinTransform {
            name: "idt-rec2020", display_name: "Rec.2020 (IDT)", kind: "idt",
            source_space: "Rec.2020", target_space: "ACEScg", vendor: "Academy",
            description: "Rec.2020 input to ACEScg working space",
            format: "clf", data: include_bytes!("../transforms/idt/rec2020.clf"),
        },
        BuiltinTransform {
            name: "idt-p3", display_name: "Display P3 (IDT)", kind: "idt",
            source_space: "Display P3", target_space: "ACEScg", vendor: "Academy",
            description: "Display P3 input to ACEScg working space",
            format: "clf", data: include_bytes!("../transforms/idt/display-p3.clf"),
        },

        // ─── SDR Output Transforms ─────────────────────────────────────
        BuiltinTransform {
            name: "ot-srgb", display_name: "sRGB 100 nits (OT)", kind: "ot",
            source_space: "ACEScg", target_space: "sRGB", vendor: "Academy",
            description: "ACEScg to sRGB desktop display (100 nits, dim surround)",
            format: "clf", data: include_bytes!("../transforms/ot/srgb-100nits.clf"),
        },
        BuiltinTransform {
            name: "ot-rec709", display_name: "Rec.709 100 nits (OT)", kind: "ot",
            source_space: "ACEScg", target_space: "Rec.709", vendor: "Academy",
            description: "ACEScg to Rec.709 broadcast (100 nits, dim surround)",
            format: "clf", data: include_bytes!("../transforms/ot/rec709-100nits.clf"),
        },
        BuiltinTransform {
            name: "ot-p3-48nits", display_name: "P3 48 nits Cinema (OT)", kind: "ot",
            source_space: "ACEScg", target_space: "DCI-P3", vendor: "Academy",
            description: "ACEScg to DCI-P3 cinema theatrical (48 nits, dark surround)",
            format: "clf", data: include_bytes!("../transforms/ot/p3-48nits-dark.clf"),
        },
        BuiltinTransform {
            name: "ot-p3-100nits", display_name: "P3 100 nits Desktop (OT)", kind: "ot",
            source_space: "ACEScg", target_space: "Display P3", vendor: "Academy",
            description: "ACEScg to Display P3 desktop monitor (100 nits, dim surround)",
            format: "clf", data: include_bytes!("../transforms/ot/p3-100nits-dim.clf"),
        },

        // ─── HDR Output Transforms ─────────────────────────────────────
        // Note: These are simplified CSCs without the ACES RRT tonescale.
        // PQ/HLG encoding is placeholder until proper implementation.
        BuiltinTransform {
            name: "ot-p3-1000nits", display_name: "P3 PQ 1000 nits HDR (OT)", kind: "ot",
            source_space: "ACEScg", target_space: "P3 PQ", vendor: "Academy",
            description: "ACEScg to P3 HDR grading monitor (1000 nits PQ, placeholder)",
            format: "clf", data: include_bytes!("../transforms/ot/p3-1000nits-dim.clf"),
        },
        BuiltinTransform {
            name: "ot-rec2020-1000nits", display_name: "Rec.2020 PQ 1000 nits HDR10 (OT)", kind: "ot",
            source_space: "ACEScg", target_space: "Rec.2020 PQ", vendor: "Academy",
            description: "ACEScg to Rec.2020 HDR10 (1000 nits PQ, placeholder)",
            format: "clf", data: include_bytes!("../transforms/ot/rec2020-1000nits-dim.clf"),
        },
        BuiltinTransform {
            name: "ot-rec2020-2000nits", display_name: "Rec.2020 PQ 2000 nits (OT)", kind: "ot",
            source_space: "ACEScg", target_space: "Rec.2020 PQ", vendor: "Academy",
            description: "ACEScg to Rec.2020 premium HDR (2000 nits PQ, placeholder)",
            format: "clf", data: include_bytes!("../transforms/ot/rec2020-2000nits-dim.clf"),
        },
        BuiltinTransform {
            name: "ot-rec2020-4000nits", display_name: "Rec.2020 PQ 4000 nits Dolby (OT)", kind: "ot",
            source_space: "ACEScg", target_space: "Rec.2020 PQ", vendor: "Academy",
            description: "ACEScg to Rec.2020 Dolby Cinema (4000 nits PQ, placeholder)",
            format: "clf", data: include_bytes!("../transforms/ot/rec2020-4000nits-dim.clf"),
        },
        BuiltinTransform {
            name: "ot-rec2020-hlg", display_name: "Rec.2020 HLG 1000 nits (OT)", kind: "ot",
            source_space: "ACEScg", target_space: "Rec.2020 HLG", vendor: "Academy",
            description: "ACEScg to Rec.2020 HLG broadcast (1000 nits, placeholder)",
            format: "clf", data: include_bytes!("../transforms/ot/rec2020-hlg-1000nits.clf"),
        },
        BuiltinTransform {
            name: "csc-acescg-to-cct", display_name: "ACEScg to ACEScct", kind: "csc",
            source_space: "ACEScg", target_space: "ACEScct", vendor: "Academy",
            description: "Linear to log grading space (same AP1 primaries)",
            format: "clf", data: include_bytes!("../transforms/csc/acescg-to-cct.clf"),
        },
        BuiltinTransform {
            name: "csc-acescct-to-cg", display_name: "ACEScct to ACEScg", kind: "csc",
            source_space: "ACEScct", target_space: "ACEScg", vendor: "Academy",
            description: "Log to linear compositing space (same AP1 primaries)",
            format: "clf", data: include_bytes!("../transforms/csc/acescct-to-cg.clf"),
        },
    ]
}

/// Register all built-in transforms using the provided callback.
///
/// The callback is called once per built-in with (data, format, name).
/// Same code path as user-imported CLF files — no special cases.
pub fn register_builtins<E>(
    mut register: impl FnMut(&[u8], &str, &str) -> Result<u32, E>,
) -> Vec<Result<u32, E>> {
    builtins().iter().map(|t| register(t.data, t.format, t.name)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_builtins_have_clf_data() {
        for t in builtins() {
            assert!(!t.data.is_empty(), "{} has no data", t.name);
            let text = std::str::from_utf8(t.data).expect("should be UTF-8");
            assert!(text.contains("<ProcessList"), "{} missing ProcessList", t.name);
        }
    }

    #[test]
    fn unique_names() {
        let mut names: Vec<&str> = builtins().iter().map(|t| t.name).collect();
        let count = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), count);
    }

    #[test]
    fn register_callback_invoked_per_builtin() {
        let mut count = 0u32;
        let results = register_builtins(|_data, _fmt, _name| -> Result<u32, ()> {
            count += 1;
            Ok(count)
        });
        assert_eq!(results.len(), builtins().len());
        assert!(results.iter().all(|r| r.is_ok()));
    }
}
