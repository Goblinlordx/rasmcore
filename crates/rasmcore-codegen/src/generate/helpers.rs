//! Helper functions for code generation: type mapping, name conversion.

/// Convert snake_case to PascalCase.
pub fn to_pascal_case(snake: &str) -> String {
    snake
        .split('_')
        .map(|seg| {
            let mut c = seg.chars();
            match c.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().collect::<String>() + c.as_str(),
            }
        })
        .collect()
}

/// Convert Rust name to WIT kebab-case.
/// Handles both snake_case (foo_bar → foo-bar) and PascalCase (ColorRgb → color-rgb).
pub fn to_wit_name(rust_name: &str) -> String {
    let clean = rust_name.trim_start_matches('_');
    // If already snake_case, convert underscores to hyphens
    if clean.contains('_') {
        return clean.replace('_', "-");
    }
    // PascalCase → kebab-case (e.g., ColorRgb → color-rgb)
    let mut result = String::new();
    let bytes = clean.as_bytes();
    for (i, ch) in clean.chars().enumerate() {
        if ch.is_uppercase() && i > 0 {
            let prev_upper = bytes.get(i - 1).is_some_and(|b| b.is_ascii_uppercase());
            let next_lower = bytes.get(i + 1).is_some_and(|b| b.is_ascii_lowercase());
            if !prev_upper || next_lower {
                result.push('-');
            }
        }
        result.push(ch.to_ascii_lowercase());
    }
    result
}

/// Map Rust type to WIT type.
pub fn to_wit_type(rust_type: &str) -> String {
    match rust_type {
        "f32" => "f32".to_string(),
        "f64" => "f64".to_string(),
        "u8" => "u8".to_string(),
        "u16" => "u16".to_string(),
        "u32" => "u32".to_string(),
        "u64" => "u64".to_string(),
        "i32" => "s32".to_string(),
        "i64" => "s64".to_string(),
        "bool" => "bool".to_string(),
        "&[f32]" => "list<f32>".to_string(),
        "&[f64]" => "list<f64>".to_string(),
        "&[u8]" => "list<u8>".to_string(),
        "&[u32]" => "list<u32>".to_string(),
        "&[Point2D]" => "list<point2d>".to_string(),
        "String" | "&str" => "string".to_string(),
        other => other.to_string(),
    }
}

/// Convert Rust domain type to owned type (for node struct fields).
/// Strips leading `&` for references (e.g., `&SpinBlurParams` → `SpinBlurParams`).
pub fn to_owned_type(rust_type: &str) -> &str {
    match rust_type {
        "&[f32]" => "Vec<f32>",
        "&[f64]" => "Vec<f64>",
        "&[u8]" => "Vec<u8>",
        "&[u32]" => "Vec<u32>",
        "&[Point2D]" => "Vec<Point2D>",
        "&str" => "String",
        other if other.starts_with('&') => &other[1..],
        other => other,
    }
}

/// Convert Rust domain type to WIT binding type (for trait signatures).
/// Strips leading `&` for references (owned at WIT boundary).
pub fn to_binding_type(rust_type: &str) -> &str {
    match rust_type {
        "&[f32]" => "Vec<f32>",
        "&[f64]" => "Vec<f64>",
        "&[u8]" => "Vec<u8>",
        "&[u32]" => "Vec<u32>",
        "&[Point2D]" => "Vec<Point2D>",
        "&str" => "String",
        other if other.starts_with('&') => &other[1..],
        other => other,
    }
}

/// Convert a Rust domain type to a fully-qualified path for use in the
/// WASM pipeline adapter macro (which lacks `use domain::filters::*`).
/// Types that live in `crate::domain::param_types` (not `crate::domain::filters`).
const PARAM_TYPES: &[&str] = &["ColorRgba", "ColorRgb", "Point2D"];

pub fn to_qualified_binding_type(rust_type: &str) -> String {
    match rust_type {
        "&[f32]" => "Vec<f32>".to_string(),
        "&[f64]" => "Vec<f64>".to_string(),
        "&[u8]" => "Vec<u8>".to_string(),
        "&[u32]" => "Vec<u32>".to_string(),
        "&[Point2D]" => "Vec<crate::domain::param_types::Point2D>".to_string(),
        "&str" => "String".to_string(),
        "f32" | "f64" | "u8" | "u16" | "u32" | "u64" | "i32" | "i64" | "bool" | "String" => {
            rust_type.to_string()
        }
        other => {
            let (prefix, inner) = if let Some(stripped) = other.strip_prefix('&') {
                ("&", stripped)
            } else {
                ("", other)
            };
            let module = if PARAM_TYPES.contains(&inner) {
                "crate::domain::param_types"
            } else {
                "crate::domain::filters"
            };
            format!("{prefix}{module}::{inner}")
        }
    }
}

/// Default range values for a given Rust type (min, max, step, default).
pub fn default_range_for_type(ty: &str) -> (&str, &str, &str, &str) {
    match ty {
        "f32" => ("0.0", "1.0", "0.01", "0.0"),
        "u32" => ("0", "1000", "1", "0"),
        "bool" => ("null", "null", "null", "false"),
        _ => ("null", "null", "null", "null"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pascal_case() {
        assert_eq!(to_pascal_case("blur"), "Blur");
        assert_eq!(to_pascal_case("zoom_blur"), "ZoomBlur");
        assert_eq!(to_pascal_case("asc_cdl"), "AscCdl");
    }

    #[test]
    fn wit_names() {
        assert_eq!(to_wit_name("blur"), "blur");
        assert_eq!(to_wit_name("zoom_blur"), "zoom-blur");
        assert_eq!(to_wit_name("_reserved"), "reserved");
    }

    #[test]
    fn wit_types() {
        assert_eq!(to_wit_type("f32"), "f32");
        assert_eq!(to_wit_type("i32"), "s32");
        assert_eq!(to_wit_type("String"), "string");
        assert_eq!(to_wit_type("&[u8]"), "list<u8>");
    }
}
