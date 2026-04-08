//! Built-in transform presets — the catalog of transforms available
//! via get-transform().

use crate::TransformPreset;

/// All built-in transform presets.
pub fn all() -> Vec<TransformPreset> {
    vec![
        // ─── IDTs (source → ACES) ─────────────────────────────────────
        TransformPreset {
            name: "idt-srgb", display_name: "sRGB (IDT)", kind: "idt",
            source: "srgb", target: "acescg", vendor: "Academy",
            description: "sRGB input to ACEScg working space",
        },
        TransformPreset {
            name: "idt-rec709", display_name: "Rec.709 (IDT)", kind: "idt",
            source: "rec709", target: "acescg", vendor: "Academy",
            description: "Rec.709 input to ACEScg working space",
        },
        TransformPreset {
            name: "idt-rec2020", display_name: "Rec.2020 (IDT)", kind: "idt",
            source: "rec2020", target: "acescg", vendor: "Academy",
            description: "Rec.2020 input to ACEScg working space",
        },
        TransformPreset {
            name: "idt-p3", display_name: "Display P3 (IDT)", kind: "idt",
            source: "display-p3", target: "acescg", vendor: "Academy",
            description: "Display P3 input to ACEScg working space",
        },
        TransformPreset {
            name: "idt-arri-logc4", display_name: "ARRI LogC4 / AWG4 (IDT)", kind: "idt",
            source: "arri-logc4-awg4", target: "acescg", vendor: "ARRI",
            description: "ARRI ALEXA 35 LogC4 to ACEScg",
        },
        TransformPreset {
            name: "idt-davinci-wg", display_name: "DaVinci Wide Gamut (IDT)", kind: "idt",
            source: "davinci-wg-di", target: "acescg", vendor: "Blackmagic",
            description: "DaVinci Wide Gamut / Intermediate to ACEScg",
        },

        // ─── Output Transforms (ACES → display) ──────────────────────
        TransformPreset {
            name: "ot-srgb", display_name: "sRGB 100 nits (OT)", kind: "ot",
            source: "acescg", target: "srgb", vendor: "Academy",
            description: "ACEScg to sRGB display output",
        },
        TransformPreset {
            name: "ot-rec709", display_name: "Rec.709 100 nits (OT)", kind: "ot",
            source: "acescg", target: "rec709", vendor: "Academy",
            description: "ACEScg to Rec.709 display output",
        },
        TransformPreset {
            name: "ot-rec2020", display_name: "Rec.2020 100 nits (OT)", kind: "ot",
            source: "acescg", target: "rec2020", vendor: "Academy",
            description: "ACEScg to Rec.2020 display output",
        },
        TransformPreset {
            name: "ot-p3", display_name: "Display P3 100 nits (OT)", kind: "ot",
            source: "acescg", target: "display-p3", vendor: "Academy",
            description: "ACEScg to Display P3 output",
        },

        // ─── CSCs (working space conversions) ─────────────────────────
        TransformPreset {
            name: "csc-acescg-to-cct", display_name: "ACEScg to ACEScct", kind: "csc",
            source: "acescg", target: "acescct", vendor: "Academy",
            description: "Linear to log grading space (same AP1 primaries)",
        },
        TransformPreset {
            name: "csc-acescct-to-cg", display_name: "ACEScct to ACEScg", kind: "csc",
            source: "acescct", target: "acescg", vendor: "Academy",
            description: "Log to linear compositing space (same AP1 primaries)",
        },

        // ─── Utility ──────────────────────────────────────────────────
        TransformPreset {
            name: "identity", display_name: "Identity (passthrough)", kind: "lmt",
            source: "any", target: "any", vendor: "rasmcore",
            description: "No-op transform for testing",
        },
    ]
}

/// Find a preset by name.
pub fn find(name: &str) -> Option<TransformPreset> {
    all().into_iter().find(|p| p.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_presets_have_unique_names() {
        let presets = all();
        let mut names: Vec<&str> = presets.iter().map(|p| p.name).collect();
        let count = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), count, "duplicate preset names");
    }

    #[test]
    fn find_standard_presets() {
        assert!(find("idt-srgb").is_some());
        assert!(find("ot-rec709").is_some());
        assert!(find("idt-arri-logc4").is_some());
        assert!(find("idt-davinci-wg").is_some());
        assert!(find("nonexistent").is_none());
    }

    #[test]
    fn preset_count() {
        assert!(all().len() >= 13, "should have at least 13 presets");
    }
}
