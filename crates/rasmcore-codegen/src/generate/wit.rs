//! Generate WIT filter declarations from structured data.

use crate::types::FilterReg;

use super::helpers::{to_wit_name, to_wit_type};

/// Generate WIT declarations for all filters.
pub fn generate(filters: &[FilterReg]) -> String {
    let mut output = String::new();
    for f in filters {
        let wit_params = f
            .params
            .iter()
            .map(|(n, t)| format!("{}: {}", to_wit_name(n), to_wit_type(t)))
            .collect::<Vec<_>>()
            .join(", ");

        let sig = if wit_params.is_empty() {
            format!(
                "    {}: func(pixels: buffer, info: image-info) -> result<buffer, rasmcore-error>;",
                to_wit_name(&f.name)
            )
        } else {
            format!(
                "    {}: func(pixels: buffer, info: image-info, {}) -> result<buffer, rasmcore-error>;",
                to_wit_name(&f.name),
                wit_params
            )
        };
        output.push_str(&sig);
        output.push('\n');
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_wit_basic() {
        let filters = vec![FilterReg {
            name: "zoom_blur".to_string(),
            category: "spatial".to_string(),
            group: String::new(),
            variant: String::new(),
            reference: String::new(),
            overlap: "zero".to_string(),
            fn_name: "zoom_blur".to_string(),
            params: vec![
                ("center_x".to_string(), "f32".to_string()),
                ("factor".to_string(), "f32".to_string()),
            ],
        }];
        let wit = generate(&filters);
        assert!(wit.contains("zoom-blur: func(pixels: buffer, info: image-info, center-x: f32, factor: f32)"));
    }

    #[test]
    fn generate_wit_no_params() {
        let filters = vec![FilterReg {
            name: "grayscale".to_string(),
            category: "color".to_string(),
            group: String::new(),
            variant: String::new(),
            reference: String::new(),
            overlap: "zero".to_string(),
            fn_name: "grayscale".to_string(),
            params: vec![],
        }];
        let wit = generate(&filters);
        assert!(wit.contains("grayscale: func(pixels: buffer, info: image-info)"));
        assert!(!wit.contains(", )"));
    }
}
