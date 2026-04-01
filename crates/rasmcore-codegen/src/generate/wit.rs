//! Generate WIT filter declarations from structured data.

use crate::types::FilterReg;

use super::helpers::{to_wit_name, to_wit_type};

/// Generate WIT declarations for all filters (records + stateless functions).
///
/// When a filter has `config_struct`, generates a WIT record type from the
/// struct's fields (looked up in `param_structs`) and uses it in the function
/// signature. Also generates records for nested ConfigParams types.
pub fn generate(
    filters: &[FilterReg],
    param_structs: &std::collections::HashMap<String, Vec<crate::types::ParamField>>,
) -> String {
    let mut records = String::new();
    let mut funcs = String::new();
    let mut emitted_records = std::collections::HashSet::new();

    // Helper: emit a record and any nested ConfigParams records it references
    fn emit_record(
        struct_name: &str,
        wit_record_name: &str,
        param_structs: &std::collections::HashMap<String, Vec<crate::types::ParamField>>,
        records: &mut String,
        emitted: &mut std::collections::HashSet<String>,
    ) {
        if emitted.contains(wit_record_name) {
            return;
        }
        emitted.insert(wit_record_name.to_string());

        if let Some(fields) = param_structs.get(struct_name) {
            // Emit nested records first
            for field in fields {
                if param_structs.contains_key(&field.param_type) {
                    let nested_wit = to_wit_name(&field.param_type);
                    emit_record(&field.param_type, &nested_wit, param_structs, records, emitted);
                }
            }

            records.push_str(&format!("    record {wit_record_name} {{\n"));
            for field in fields {
                let wit_type = if param_structs.contains_key(&field.param_type) {
                    to_wit_name(&field.param_type)
                } else {
                    to_wit_type(&field.param_type)
                };
                records.push_str(&format!(
                    "        {}: {},\n",
                    to_wit_name(&field.name),
                    wit_type
                ));
            }
            records.push_str("    }\n\n");
        }
    }

    for f in filters {
        if let Some(config_name) = &f.config_struct {
            let record_name = format!("{}-config", to_wit_name(&f.name));
            emit_record(config_name, &record_name, param_structs, &mut records, &mut emitted_records);

            // Generate func — extra params first, then config
            let extra_params: Vec<String> = f.params.iter()
                .filter(|(_n, t)| !(t.starts_with('&') && t.ends_with("Params")))
                .map(|(n, t)| format!("{}: {}", to_wit_name(n), to_wit_type(t)))
                .collect();
            let mut parts = vec!["pixels: buffer".to_string(), "info: image-info".to_string()];
            parts.extend(extra_params);
            parts.push(format!("config: {record_name}"));

            funcs.push_str(&format!(
                "    {}: func({}) -> result<buffer, rasmcore-error>;\n",
                to_wit_name(&f.name),
                parts.join(", ")
            ));
        } else {
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

    let mut output = String::new();
    if !records.is_empty() {
        output.push_str(&records);
    }
    output.push_str(&funcs);
    output
}

/// Generate WIT for filters only (backward-compatible wrapper).
pub fn generate_filters_only(filters: &[FilterReg]) -> String {
    generate(filters, &std::collections::HashMap::new())
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
            rect_request: true,
            fn_name: "zoom_blur".to_string(),
            params: vec![
                ("center_x".to_string(), "f32".to_string()),
                ("factor".to_string(), "f32".to_string()),
            ],
            config_struct: None,
            point_op: false,
            color_op: false,
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
            rect_request: true,
            fn_name: "grayscale".to_string(),
            params: vec![],
            config_struct: None,
            point_op: false,
            color_op: false,
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
            rect_request: true,
            fn_name: "blur".to_string(),
            params: vec![("radius".to_string(), "f32".to_string())],
            config_struct: Some("BlurParams".to_string()),
            point_op: false,
            color_op: false,
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
