//! Generate WIT filter declarations from structured data.

use crate::types::FilterReg;

use super::helpers::{to_wit_name, to_wit_type};

/// Generate WIT declarations for all filters.
///
/// When a filter has `config_struct`, generates a WIT record type and
/// uses it in the function signature. Otherwise, uses individual params.
pub fn generate(filters: &[FilterReg]) -> String {
    let mut records = String::new();
    let mut funcs = String::new();

    for f in filters {
        if f.config_struct.is_some() {
            // Generate record type
            let record_name = format!("{}-config", to_wit_name(&f.name));
            records.push_str(&format!("    record {record_name} {{\n"));
            for (n, t) in &f.params {
                records.push_str(&format!(
                    "        {}: {},\n",
                    to_wit_name(n),
                    to_wit_type(t)
                ));
            }
            records.push_str("    }\n\n");

            // Generate func with config param
            funcs.push_str(&format!(
                "    {}: func(pixels: buffer, info: image-info, config: {record_name}) -> result<buffer, rasmcore-error>;\n",
                to_wit_name(&f.name)
            ));
        } else {
            // Individual params (current behavior)
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
            funcs.push_str(&sig);
            funcs.push('\n');
        }
    }

    // Records first, then functions
    let mut output = String::new();
    if !records.is_empty() {
        output.push_str(&records);
    }
    output.push_str(&funcs);
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
            config_struct: None,
            point_op: false,
        }];
        let wit = generate(&filters);
        assert!(wit.contains(
            "zoom-blur: func(pixels: buffer, info: image-info, center-x: f32, factor: f32)"
        ));
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
            config_struct: None,
            point_op: false,
        }];
        let wit = generate(&filters);
        assert!(wit.contains("grayscale: func(pixels: buffer, info: image-info)"));
        assert!(!wit.contains(", )"));
    }

    #[test]
    fn generate_wit_config_struct() {
        let filters = vec![FilterReg {
            name: "blur".to_string(),
            category: "spatial".to_string(),
            group: String::new(),
            variant: String::new(),
            reference: String::new(),
            overlap: "zero".to_string(),
            fn_name: "blur".to_string(),
            params: vec![("radius".to_string(), "f32".to_string())],
            config_struct: Some("BlurParams".to_string()),
        }];
        let wit = generate(&filters);
        assert!(
            wit.contains("record blur-config {"),
            "should generate record: {wit}"
        );
        assert!(
            wit.contains("radius: f32,"),
            "should have field in record: {wit}"
        );
        assert!(
            wit.contains("config: blur-config"),
            "should use config param: {wit}"
        );
    }
}
