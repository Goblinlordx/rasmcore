//! Generate pipeline write adapter methods and WIT declarations from encoder configs.

use super::helpers::to_pascal_case;

/// Convert PascalCase to snake_case (e.g., TiffCompression → tiff_compression).
fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() && i > 0 {
            result.push('_');
        }
        result.push(ch.to_ascii_lowercase());
    }
    result
}
use crate::parse::encoders::EncoderInfo;

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
            wit.push_str(&format!(
                "    record {}-write-config {{\n",
                enc.format.replace('_', "-")
            ));
            for field in &enc.fields {
                if field.is_enum {
                    let wit_enum = field.rust_type.replace('_', "-").to_lowercase();
                    wit.push_str(&format!(
                        "        {}: option<{}>,\n",
                        field.name.replace('_', "-"),
                        wit_enum
                    ));
                } else {
                    let wit_type = match field.rust_type.as_str() {
                        "u8" => "u8",
                        "u16" => "u16",
                        "u32" => "u32",
                        "bool" => "bool",
                        "f32" => "f32",
                        _ => "u32",
                    };
                    wit.push_str(&format!(
                        "        {}: option<{}>,\n",
                        field.name.replace('_', "-"),
                        wit_type
                    ));
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
                let variants: Vec<String> = field
                    .enum_variants
                    .iter()
                    .map(|v| {
                        let mut kebab = String::new();
                        for (i, ch) in v.chars().enumerate() {
                            if ch.is_uppercase() && i > 0 {
                                kebab.push('-');
                            }
                            kebab.push(ch.to_lowercase().next().unwrap());
                        }
                        kebab
                    })
                    .collect();
                wit.push_str(&format!(
                    "    enum {} {{ {} }}\n\n",
                    wit_name,
                    variants.join(", ")
                ));
            }
        }
    }
    wit
}

/// Generate WIT write method declarations — metadata parameter when supported.
pub fn generate_wit_write_methods(encoders: &[EncoderInfo]) -> String {
    let mut wit = String::new();
    for enc in encoders {
        let wit_name = enc.format.replace('_', "-");
        let config_type = format!("{}-write-config", wit_name);
        if enc.sink_takes_metadata {
            wit.push_str(&format!(
                "        write-{}: func(source: node-id, config: {}, metadata: option<metadata-set>) -> result<buffer, rasmcore-error>;\n",
                wit_name, config_type
            ));
        } else {
            wit.push_str(&format!(
                "        write-{}: func(source: node-id, config: {}) -> result<buffer, rasmcore-error>;\n",
                wit_name, config_type
            ));
        }
    }
    wit
}

/// Generate Rust adapter write methods matching the current WIT API.
///
/// Formats that support metadata get a `metadata: Option<pipeline::MetadataSet>`
/// parameter, converted via `super::to_domain_metadata_set`.
///
/// Output is a `macro_rules!` that is included at module level and invoked
/// inside the `impl GuestImagePipeline` block (same pattern as filter adapter).
pub fn generate_adapter_methods(encoders: &[EncoderInfo]) -> String {
    let mut code = String::new();
    code.push_str("// Auto-generated pipeline write adapter methods.\n");
    code.push_str("// Do not edit — regenerate by changing encoder configs and rebuilding.\n\n");
    code.push_str("macro_rules! generated_pipeline_write_methods {\n    () => {\n");

    for enc in encoders {
        let method_name = format!("write_{}", enc.format);
        let config_pascal = to_pascal_case(&format!("{}_write_config", enc.format));

        // Build signature — metadata parameter when sink supports it
        let config_prefix = if enc.fields.is_empty() { "_" } else { "" };
        if enc.sink_takes_metadata {
            code.push_str(&format!(
                "    fn {method_name}(&self, source: NodeId, {config_prefix}config: pipeline::{config_pascal}, metadata: Option<pipeline::MetadataSet>) -> Result<Vec<u8>, RasmcoreError> {{\n"
            ));
        } else {
            code.push_str(&format!(
                "    fn {method_name}(&self, source: NodeId, {config_prefix}config: pipeline::{config_pascal}) -> Result<Vec<u8>, RasmcoreError> {{\n"
            ));
        }

        // Build domain config if there are fields
        if !enc.fields.is_empty() {
            let module = &enc.module;
            let config_struct = &enc.config_struct;
            code.push_str(&format!(
                "        let cfg = domain::encoder::{module}::{config_struct} {{\n"
            ));
            for field in &enc.fields {
                if field.is_enum {
                    let snake = to_snake_case(&field.rust_type);
                    code.push_str(&format!(
                        "            {}: to_domain_{}_pipeline(config.{}),\n",
                        field.name, snake, field.name
                    ));
                } else {
                    let default = if field.default_val.is_empty() {
                        match field.rust_type.as_str() {
                            "u8" => "0",
                            "u16" => "0",
                            "bool" => "false",
                            _ => "0",
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
                code.push_str("        let domain_meta = metadata.as_ref().map(super::to_domain_metadata_set);\n");
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
                code.push_str("        let domain_meta = metadata.as_ref().map(super::to_domain_metadata_set);\n");
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
    code.push_str("    } // end macro\n}\n");
    code
}

/// Generate WIT encode config records for the stateless encoder interface.
/// Same configs as pipeline write but named `<format>-encode-config` (not write).
pub fn generate_wit_encode_configs(encoders: &[EncoderInfo]) -> String {
    // Reuse the write config generation — same records, different naming
    generate_wit_configs(encoders)
        .replace("-write-config", "-encode-config")
}

/// Generate WIT encode method declarations for the stateless encoder interface.
pub fn generate_wit_encode_methods(encoders: &[EncoderInfo]) -> String {
    let mut wit = String::new();
    for enc in encoders {
        let wit_name = enc.format.replace('_', "-");
        let config_type = format!("{}-encode-config", wit_name);
        wit.push_str(&format!(
            "    encode-{}: func(pixels: buffer, info: image-info, config: {}) -> result<buffer, rasmcore-error>;\n",
            wit_name, config_type
        ));
    }
    wit
}

/// Generate the stateless encoder adapter — per-format encode_xxx methods.
///
/// Each method converts WIT config → domain config and calls the encoder.
/// Output is a `macro_rules!` included at module level and invoked inside
/// `impl encoder::Guest`.
pub fn generate_stateless_encoder_adapter(encoders: &[EncoderInfo]) -> String {
    let mut code = String::new();
    code.push_str("// Auto-generated stateless encoder adapter methods.\n");
    code.push_str("// Do not edit — regenerate by changing encoder configs and rebuilding.\n\n");
    code.push_str("macro_rules! generated_encoder_methods {\n    () => {\n");

    for enc in encoders {
        let method_name = format!("encode_{}", enc.format);
        let config_pascal = to_pascal_case(&format!("{}_encode_config", enc.format));
        let format = &enc.format;

        if enc.fields.is_empty() {
            // No config fields — delegate to generic encode with None quality
            code.push_str(&format!(
                "    fn {method_name}(pixels: Vec<u8>, info: types::ImageInfo, _config: encoder::{config_pascal}) -> Result<Vec<u8>, RasmcoreError> {{\n"
            ));
            code.push_str("        let domain_info = to_domain_image_info(&info);\n");
            code.push_str(&format!(
                "        domain::encoder::encode(&pixels, &domain_info, \"{format}\", None).map_err(to_wit_error)\n"
            ));
        } else {
            // Has config fields — extract quality if present, delegate to generic encode
            let quality_field = enc.fields.iter().find(|f| f.name == "quality");
            code.push_str(&format!(
                "    fn {method_name}(pixels: Vec<u8>, info: types::ImageInfo, config: encoder::{config_pascal}) -> Result<Vec<u8>, RasmcoreError> {{\n"
            ));
            code.push_str("        let domain_info = to_domain_image_info(&info);\n");
            if let Some(qf) = quality_field {
                let default = if qf.default_val.is_empty() { "0" } else { &qf.default_val };
                code.push_str(&format!(
                    "        let quality = Some(config.quality.unwrap_or({default}));\n"
                ));
                code.push_str(&format!(
                    "        domain::encoder::encode(&pixels, &domain_info, \"{format}\", quality).map_err(to_wit_error)\n"
                ));
            } else {
                code.push_str(&format!(
                    "        domain::encoder::encode(&pixels, &domain_info, \"{format}\", None).map_err(to_wit_error)\n"
                ));
            }
        }

        code.push_str("    }\n\n");
    }
    code.push_str("    } // end macro\n}\n");
    code
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::encoders::{EncoderField, EncoderInfo};

    fn jpeg_encoder() -> EncoderInfo {
        EncoderInfo {
            format: "jpeg".into(),
            config_struct: "JpegEncodeConfig".into(),
            fields: vec![
                EncoderField {
                    name: "quality".into(),
                    rust_type: "u8".into(),
                    is_enum: false,
                    enum_variants: vec![],
                    default_val: "85".into(),
                },
                EncoderField {
                    name: "progressive".into(),
                    rust_type: "bool".into(),
                    is_enum: false,
                    enum_variants: vec![],
                    default_val: "false".into(),
                },
            ],
            encode_fn: "encode_pixels".into(),
            module: "jpeg".into(),
            sink_takes_metadata: true,
        }
    }

    fn bmp_encoder() -> EncoderInfo {
        EncoderInfo {
            format: "bmp".into(),
            config_struct: "BmpEncodeConfig".into(),
            fields: vec![],
            encode_fn: "encode_pixels".into(),
            module: "bmp".into(),
            sink_takes_metadata: false,
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
    fn wit_write_methods_with_metadata() {
        let wit = generate_wit_write_methods(&[jpeg_encoder(), bmp_encoder()]);
        assert!(wit.contains("write-jpeg: func(source: node-id, config: jpeg-write-config, metadata: option<metadata-set>)"));
        assert!(wit.contains("write-bmp: func(source: node-id, config: bmp-write-config)"));
        // bmp has no metadata, jpeg does
        assert!(!wit.contains("write-bmp") || !wit.contains("bmp-write-config, metadata"));
    }

    #[test]
    fn adapter_simple() {
        let code = generate_adapter_methods(&[bmp_encoder()]);
        assert!(code.contains("fn write_bmp("));
        assert!(code.contains("sink::write_bmp("));
        assert!(code.contains("finalize_cache"));
        assert!(!code.contains("metadata"));
    }

    #[test]
    fn adapter_with_metadata_param() {
        let code = generate_adapter_methods(&[jpeg_encoder()]);
        assert!(code.contains("config.quality.unwrap_or(85)"));
        assert!(code.contains("metadata: Option<pipeline::MetadataSet>"));
        assert!(code.contains("metadata.as_ref().map(super::to_domain_metadata_set)"));
        assert!(code.contains("finalize_cache"));
    }
}

/// Generate the encode() dispatch macro — replaces the hardcoded match in encoder/mod.rs.
///
/// Each encoder gets a match arm: "format" => { let config = Config::default(); module::encode_fn(pixels, info, &config) }
/// For encoders with a quality field, the quality param is applied.
pub fn generate_encode_dispatch(encoders: &[EncoderInfo]) -> String {
    let mut code = String::new();
    code.push_str("// Auto-generated encode dispatch — replaces hardcoded match in encoder/mod.rs.\n");
    code.push_str("// Do not edit — regenerate by changing encoder configs and rebuilding.\n\n");
    code.push_str("macro_rules! generated_encode_dispatch {\n");
    code.push_str("    ($pixels:expr, $info:expr, $format:expr, $quality:expr) => {\n");
    code.push_str("        match $format {\n");

    for enc in encoders {
        let module = &enc.module;
        let config_struct = &enc.config_struct;
        let format = &enc.format;

        // Determine encode function: use parsed name, or derive from module
        let encode_fn = if !enc.encode_fn.is_empty() {
            enc.encode_fn.clone()
        } else {
            // Convention: try encode_pixels, encode, or encode_<format>
            "encode_pixels".to_string()
        };

        // Check if config has a quality field
        let has_quality = enc.fields.iter().any(|f| f.name == "quality");
        let has_real_config = !enc.fields.is_empty();

        // Generate aliases (e.g., "jpeg" | "jpg")
        let aliases: Vec<&str> = match format.as_str() {
            "jpeg" => vec!["jpeg", "jpg"],
            "heic" => vec!["heic", "heif"],
            "tiff" => vec!["tiff", "tif"],
            "pnm" => vec!["pnm", "ppm", "pgm", "pbm"],
            "exr" => vec!["exr", "openexr"],
            "jp2" => vec!["jp2", "j2k", "jpeg2000"],
            "fits" => vec!["fits", "fit"],
            _ => vec![format.as_str()],
        };
        let pattern = aliases.iter().map(|a| format!("\"{}\"", a)).collect::<Vec<_>>().join(" | ");

        if has_real_config {
            code.push_str(&format!("            {pattern} => {{\n"));
            if has_quality {
                let has_extra_fields = enc.fields.iter().any(|f| f.name != "quality");
                let struct_tail = if has_extra_fields { ", ..Default::default()" } else { "" };
                code.push_str(&format!(
                    "                let config = {module}::{config_struct} {{ quality: $quality.unwrap_or({module}::{config_struct}::default().quality){struct_tail} }};\n"
                ));
            } else {
                code.push_str(&format!(
                    "                let config = {module}::{config_struct}::default();\n"
                ));
            }
            code.push_str(&format!(
                "                {module}::{encode_fn}($pixels, $info, &config)\n"
            ));
            code.push_str("            }\n");
        } else {
            // Zero-field config (unit struct like IcoEncodeConfig) — pass as &Config
            code.push_str(&format!("            {pattern} => {{\n"));
            code.push_str(&format!(
                "                {module}::{encode_fn}($pixels, $info, &{module}::{config_struct})\n"
            ));
            code.push_str("            }\n");
        }
    }

    code.push_str("            other => Err(crate::domain::error::ImageError::UnsupportedFormat(\n");
    code.push_str("                [\"encode format '\", other, \"' not supported\"].concat()\n");
    code.push_str("            ))\n");
    code.push_str("        }\n");
    code.push_str("    };\n");
    code.push_str("}\n");
    code
}

