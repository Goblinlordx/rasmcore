//! Generate pipeline write adapter methods and WIT declarations from encoder configs.

use crate::parse::encoders::EncoderInfo;
use super::helpers::to_pascal_case;

/// Generate WIT write config records for all encoders.
pub fn generate_wit_configs(encoders: &[EncoderInfo]) -> String {
    let mut wit = String::new();
    for enc in encoders {
        if enc.fields.is_empty() {
            wit.push_str(&format!(
                "    record {}-write-config {{\n        reserved: option<bool>,\n    }}\n\n",
                enc.format.replace('_', "-")
            ));
        } else {
            wit.push_str(&format!("    record {}-write-config {{\n", enc.format.replace('_', "-")));
            for field in &enc.fields {
                if field.is_enum {
                    let wit_enum = field.rust_type.replace('_', "-").to_lowercase();
                    wit.push_str(&format!("        {}: option<{}>,\n", field.name.replace('_', "-"), wit_enum));
                } else {
                    let wit_type = match field.rust_type.as_str() {
                        "u8" => "u8", "u16" => "u16", "u32" => "u32",
                        "bool" => "bool", "f32" => "f32", _ => "u32",
                    };
                    wit.push_str(&format!("        {}: option<{}>,\n", field.name.replace('_', "-"), wit_type));
                }
            }
            wit.push_str("    }\n\n");
        }
    }
    wit
}

/// Generate WIT enum declarations for encoder config enum fields.
pub fn generate_wit_enums(encoders: &[EncoderInfo]) -> String {
    let mut wit = String::new();
    let mut seen = std::collections::HashSet::new();
    for enc in encoders {
        for field in &enc.fields {
            if field.is_enum && !seen.contains(&field.rust_type) {
                seen.insert(field.rust_type.clone());
                let wit_name = field.rust_type.replace('_', "-").to_lowercase();
                let variants: Vec<String> = field.enum_variants.iter().map(|v| {
                    let mut kebab = String::new();
                    for (i, ch) in v.chars().enumerate() {
                        if ch.is_uppercase() && i > 0 { kebab.push('-'); }
                        kebab.push(ch.to_lowercase().next().unwrap());
                    }
                    kebab
                }).collect();
                wit.push_str(&format!("    enum {} {{ {} }}\n\n", wit_name, variants.join(", ")));
            }
        }
    }
    wit
}

/// Generate WIT write method declarations — uniform (source, config) signature.
pub fn generate_wit_write_methods(encoders: &[EncoderInfo]) -> String {
    let mut wit = String::new();
    for enc in encoders {
        let wit_name = enc.format.replace('_', "-");
        let config_type = format!("{}-write-config", wit_name);
        wit.push_str(&format!(
            "        write-{}: func(source: node-id, config: {}) -> result<buffer, rasmcore-error>;\n",
            wit_name, config_type
        ));
    }
    wit
}

/// Generate Rust adapter write methods — metadata from pipeline chain state.
pub fn generate_adapter_methods(encoders: &[EncoderInfo]) -> String {
    let mut code = String::new();
    code.push_str("// Auto-generated pipeline write adapter methods.\n");
    code.push_str("// All write methods take (source, config). Metadata comes from pipeline chain state.\n\n");

    for enc in encoders {
        let method_name = format!("write_{}", enc.format);
        let config_pascal = to_pascal_case(&format!("{}_write_config", enc.format));

        // Uniform signature: (source, config)
        if enc.fields.is_empty() {
            code.push_str(&format!(
                "    fn {method_name}(&self, source: NodeId, _config: pipeline::{config_pascal}) -> Result<Vec<u8>, RasmcoreError> {{\n"
            ));
        } else {
            code.push_str(&format!(
                "    fn {method_name}(&self, source: NodeId, config: pipeline::{config_pascal}) -> Result<Vec<u8>, RasmcoreError> {{\n"
            ));
        }

        // Build domain config if there are fields
        if !enc.fields.is_empty() {
            let module = &enc.module;
            let config_struct = &enc.config_struct;
            code.push_str(&format!("        let cfg = domain::encoder::{module}::{config_struct} {{\n"));
            for field in &enc.fields {
                if field.is_enum {
                    code.push_str(&format!(
                        "            {}: to_domain_{}_pipeline(config.{}),\n",
                        field.name, field.rust_type.to_lowercase(), field.name
                    ));
                } else {
                    let default = if field.default_val.is_empty() {
                        match field.rust_type.as_str() {
                            "u8" => "0", "u16" => "0", "bool" => "false", _ => "0",
                        }
                    } else {
                        &field.default_val
                    };
                    code.push_str(&format!(
                        "            {}: config.{}.unwrap_or({}),\n",
                        field.name, field.name, default
                    ));
                }
            }
            code.push_str("        };\n");

            if enc.sink_takes_metadata {
                code.push_str("        let domain_meta = self.resolve_metadata();\n");
                code.push_str(&format!(
                    "        let result = sink::{method_name}(&mut self.graph.borrow_mut(), source, &cfg, domain_meta.as_ref()).map_err(to_wit_error);\n"
                ));
            } else {
                code.push_str(&format!(
                    "        let result = sink::{method_name}(&mut self.graph.borrow_mut(), source, &cfg).map_err(to_wit_error);\n"
                ));
            }
        } else {
            if enc.sink_takes_metadata {
                code.push_str("        let domain_meta = self.resolve_metadata();\n");
                code.push_str(&format!(
                    "        let result = sink::{method_name}(&mut self.graph.borrow_mut(), source, domain_meta.as_ref()).map_err(to_wit_error);\n"
                ));
            } else {
                code.push_str(&format!(
                    "        let result = sink::{method_name}(&mut self.graph.borrow_mut(), source).map_err(to_wit_error);\n"
                ));
            }
        }

        code.push_str("        self.finalize_cache();\n");
        code.push_str("        result\n");
        code.push_str("    }\n\n");
    }
    code
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::encoders::{EncoderField, EncoderInfo};

    fn jpeg_encoder() -> EncoderInfo {
        EncoderInfo {
            format: "jpeg".into(), config_struct: "JpegEncodeConfig".into(),
            fields: vec![
                EncoderField { name: "quality".into(), rust_type: "u8".into(), is_enum: false, enum_variants: vec![], default_val: "85".into() },
                EncoderField { name: "progressive".into(), rust_type: "bool".into(), is_enum: false, enum_variants: vec![], default_val: "false".into() },
            ],
            encode_fn: "encode_pixels".into(), module: "jpeg".into(), sink_takes_metadata: true,
        }
    }

    fn bmp_encoder() -> EncoderInfo {
        EncoderInfo {
            format: "bmp".into(), config_struct: "BmpEncodeConfig".into(),
            fields: vec![], encode_fn: "encode_pixels".into(), module: "bmp".into(), sink_takes_metadata: false,
        }
    }

    #[test]
    fn wit_config_simple() {
        let wit = generate_wit_configs(&[bmp_encoder()]);
        assert!(wit.contains("record bmp-write-config {"));
    }

    #[test]
    fn wit_config_complex() {
        let wit = generate_wit_configs(&[jpeg_encoder()]);
        assert!(wit.contains("quality: option<u8>"));
        assert!(wit.contains("progressive: option<bool>"));
    }

    #[test]
    fn wit_write_methods_uniform() {
        let wit = generate_wit_write_methods(&[jpeg_encoder(), bmp_encoder()]);
        assert!(wit.contains("write-jpeg: func(source: node-id, config: jpeg-write-config)"));
        assert!(wit.contains("write-bmp: func(source: node-id, config: bmp-write-config)"));
        assert!(!wit.contains("metadata"));
    }

    #[test]
    fn adapter_simple() {
        let code = generate_adapter_methods(&[bmp_encoder()]);
        assert!(code.contains("fn write_bmp("));
        assert!(code.contains("sink::write_bmp("));
        assert!(code.contains("finalize_cache"));
    }

    #[test]
    fn adapter_with_metadata_sink() {
        let code = generate_adapter_methods(&[jpeg_encoder()]);
        assert!(code.contains("config.quality.unwrap_or(85)"));
        assert!(code.contains("resolve_metadata"));
        assert!(code.contains("finalize_cache"));
    }
}
