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
            name: "ot-srgb", display_name: "sRGB 100 nits (OT)", kind: "ot",
            source_space: "ACEScg", target_space: "sRGB", vendor: "Academy",
            description: "ACEScg to sRGB display (simplified, no RRT tonemap)",
            format: "clf", data: include_bytes!("../transforms/ot/srgb-100nits.clf"),
        },
        BuiltinTransform {
            name: "ot-rec709", display_name: "Rec.709 100 nits (OT)", kind: "ot",
            source_space: "ACEScg", target_space: "Rec.709", vendor: "Academy",
            description: "ACEScg to Rec.709 display (simplified, no RRT tonemap)",
            format: "clf", data: include_bytes!("../transforms/ot/rec709-100nits.clf"),
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
