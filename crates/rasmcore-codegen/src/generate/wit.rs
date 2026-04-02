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
        let has_config_param = f.params.iter().any(|(_n, t)| t.starts_with('&') && t.ends_with("Params"));
        // derive(Filter) filters have config_struct but no entries in params vec.
        // Only true when the config struct has actual fields in param_structs.
        let has_derive_config = f.config_struct.as_ref()
            .filter(|_| f.params.is_empty())
            .and_then(|name| param_structs.get(name.as_str()))
            .map_or(false, |fields| !fields.is_empty());

        // Get config struct field names to avoid duplicating fields
        let config_field_names: std::collections::HashSet<String> = if has_config_param {
            if let Some(config_name) = &f.config_struct {
                param_structs.get(config_name.as_str())
                    .map(|fields| fields.iter().map(|f| f.name.trim_start_matches('_').to_string()).collect())
                    .unwrap_or_default()
            } else {
                std::collections::HashSet::new()
            }
        } else {
            std::collections::HashSet::new()
        };

        // Extra params that are NOT already fields in the ConfigParams struct
        let extra_params: Vec<&(String, String)> = f.params.iter()
            .filter(|(n, t)| {
                !(config_field_names.contains(n.trim_start_matches('_')) || t.starts_with('&') && t.ends_with("Params"))
            })
            .collect();
        let record_name = format!("{}-config", to_wit_name(&f.name));

        // Emit config record — old-style (has_config_param) or derive-style (has_derive_config)
        if has_config_param || has_derive_config {
            if let Some(config_name) = &f.config_struct {
                emit_record(config_name, &record_name, param_structs, &mut records, &mut emitted_records);
            }
        }

        // If there are extra params (string/list), add them as fields in the config record.
        // For filters with no ConfigParams struct, generate a synthetic record from extra params only.
        if !extra_params.is_empty() {
            if !has_config_param && !emitted_records.contains(&record_name) {
                // Synthetic record from extra params only
                records.push_str(&format!("    record {} {{\n", record_name));
                for (n, t) in &extra_params {
                    let wn = to_wit_name(n.trim_start_matches('_'));
                    records.push_str(&format!("        {}: {},\n", wn, to_wit_type(t)));
                }
                records.push_str("    }\n\n");
                emitted_records.insert(record_name.clone());
            } else if has_config_param && emitted_records.contains(&record_name) {
                // Append extra param fields to already-emitted record
                // Find the closing "    }\n\n" of the record and insert before it
                let close_marker = format!("    record {} {{", record_name);
                if let Some(pos) = records.find(&close_marker) {
                    // Find the closing brace after this record
                    let after = &records[pos..];
                    if let Some(close_pos) = after.find("    }\n\n") {
                        let insert_at = pos + close_pos;
                        let extra_fields: String = extra_params.iter()
                            .map(|(n, t)| {
                                let wn = to_wit_name(n.trim_start_matches('_'));
                                format!("        {}: {},\n", wn, to_wit_type(t))
                            })
                            .collect();
                        records.insert_str(insert_at, &extra_fields);
                    }
                }
            }
        }

        // Generate func signature — always (pixels, info, config) or (pixels, info) for zero-param
        let needs_config = !f.params.is_empty() || has_derive_config;
        if needs_config {
            funcs.push_str(&format!(
                "    {}: func(pixels: buffer, info: image-info, config: {}) -> result<buffer, rasmcore-error>;\n",
                to_wit_name(&f.name),
                record_name
            ));
        } else {
            funcs.push_str(&format!(
                "    {}: func(pixels: buffer, info: image-info) -> result<buffer, rasmcore-error>;\n",
                to_wit_name(&f.name)
            ));
        }
    }

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
            rect_request: true,
            fn_name: "zoom_blur".to_string(),
            params: vec![
                ("center_x".to_string(), "f32".to_string()),
                ("factor".to_string(), "f32".to_string()),
            ],
            config_struct: None,
            point_op: false,
            color_op: false,
            gpu: false,
            derive_style: false,
        }];
        let empty = std::collections::HashMap::new();
        let wit = generate(&filters, &empty);
        // Extra params without config_struct produce a synthetic config record
        assert!(
            wit.contains("record zoom-blur-config {"),
            "should generate synthetic config record: {wit}"
        );
        assert!(
            wit.contains("config: zoom-blur-config"),
            "should use config param: {wit}"
        );
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
            gpu: false,
            derive_style: false,
        }];
        let empty = std::collections::HashMap::new();
        let wit = generate(&filters, &empty);
        assert!(wit.contains("grayscale: func(pixels: buffer, info: image-info)"));
        assert!(!wit.contains(", )"));
    }

    #[test]
    fn generate_wit_config_struct() {
        use crate::types::ParamField;
        let filters = vec![FilterReg {
            name: "blur".to_string(),
            category: "spatial".to_string(),
            group: String::new(),
            variant: String::new(),
            reference: String::new(),
            rect_request: true,
            fn_name: "blur".to_string(),
            params: vec![("_config".to_string(), "&BlurParams".to_string())],
            config_struct: Some("BlurParams".to_string()),
            point_op: false,
            color_op: false,
            gpu: false,
            derive_style: false,
        }];
        let mut param_structs = std::collections::HashMap::new();
        param_structs.insert("BlurParams".to_string(), vec![
            ParamField { name: "radius".to_string(), param_type: "f32".to_string(), default_val: String::new(), min: String::new(), max: String::new(), step: String::new(), label: String::new(), hint: String::new(), options: Vec::new() },
        ]);
        let wit = generate(&filters, &param_structs);
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

    #[test]
    fn generate_wit_derive_style_config() {
        use crate::types::ParamField;
        // derive(Filter) filters have config_struct but empty params vec
        let filters = vec![FilterReg {
            name: "wave".to_string(),
            category: "distortion".to_string(),
            group: String::new(),
            variant: String::new(),
            reference: String::new(),
            rect_request: true,
            fn_name: "wave".to_string(),
            params: vec![],
            config_struct: Some("WaveParams".to_string()),
            point_op: false,
            color_op: false,
            gpu: false,
            derive_style: true,
        }];
        let mut param_structs = std::collections::HashMap::new();
        param_structs.insert("WaveParams".to_string(), vec![
            ParamField { name: "amplitude".to_string(), param_type: "f32".to_string(), default_val: String::new(), min: String::new(), max: String::new(), step: String::new(), label: String::new(), hint: String::new(), options: Vec::new() },
            ParamField { name: "wavelength".to_string(), param_type: "f32".to_string(), default_val: String::new(), min: String::new(), max: String::new(), step: String::new(), label: String::new(), hint: String::new(), options: Vec::new() },
        ]);
        let wit = generate(&filters, &param_structs);
        assert!(
            wit.contains("record wave-config {"),
            "derive(Filter) should generate config record: {wit}"
        );
        assert!(
            wit.contains("amplitude: f32,"),
            "should have amplitude field: {wit}"
        );
        assert!(
            wit.contains("config: wave-config"),
            "derive(Filter) should use config param in func signature: {wit}"
        );
        assert!(
            !wit.contains("func(pixels: buffer, info: image-info) -> result"),
            "derive(Filter) with config should NOT generate zero-param signature: {wit}"
        );
    }
}
