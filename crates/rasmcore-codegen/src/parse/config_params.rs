//! Parse `#[derive(ConfigParams)]` structs and their `#[param(...)]` field attributes.

use crate::types::{ParamField, ParsedStruct};
use std::collections::HashMap;

use super::param_attr::ParamAttr;

/// Extract all ConfigParams structs from a parsed file.
/// Returns a map of struct_name → Vec<ParamField> (with nested types resolved).
pub fn extract_config_params(file: &syn::File) -> HashMap<String, Vec<ParamField>> {
    let raw = extract_raw(file);
    let mut result = HashMap::new();
    for (name, parsed) in &raw {
        let fields = resolve_nested_fields(&parsed.fields, &parsed.config_hint, &raw);
        result.insert(name.clone(), fields);
    }
    result
}

/// Extract raw ConfigParams structs without resolving nested types.
fn extract_raw(file: &syn::File) -> HashMap<String, ParsedStruct> {
    let mut result = HashMap::new();
    for item in &file.items {
        if let syn::Item::Struct(s) = item {
            let has_config_params = s.attrs.iter().any(|attr| {
                if let syn::Meta::List(ml) = &attr.meta {
                    ml.tokens.to_string().contains("ConfigParams")
                } else {
                    false
                }
            });
            if !has_config_params {
                continue;
            }

            let struct_name = s.ident.to_string();

            // Extract #[config_hint("...")] if present
            let config_hint = s
                .attrs
                .iter()
                .filter_map(|attr| {
                    if attr.path().is_ident("config_hint") {
                        if let syn::Meta::List(ml) = &attr.meta {
                            let t = ml.tokens.to_string();
                            return Some(t.trim().trim_matches('"').to_string());
                        }
                    }
                    None
                })
                .next()
                .unwrap_or_default();

            let mut fields = Vec::new();
            if let syn::Fields::Named(named) = &s.fields {
                for field in &named.named {
                    let fname = field
                        .ident
                        .as_ref()
                        .map(|i| i.to_string())
                        .unwrap_or_default();
                    let ftype = super::type_to_string(&field.ty);

                    // Extract doc comment as label
                    let label = field
                        .attrs
                        .iter()
                        .filter_map(|a| {
                            if let syn::Meta::NameValue(nv) = &a.meta {
                                if nv.path.is_ident("doc") {
                                    if let syn::Expr::Lit(lit) = &nv.value {
                                        if let syn::Lit::Str(s) = &lit.lit {
                                            return Some(s.value().trim().to_string());
                                        }
                                    }
                                }
                            }
                            None
                        })
                        .next()
                        .unwrap_or_default();

                    // Extract #[param(...)] via syn parser
                    let mut min = "null".to_string();
                    let mut max = "null".to_string();
                    let mut step = "null".to_string();
                    let mut default_val = String::new();
                    let mut hint = String::new();
                    let mut options = Vec::new();

                    for attr in &field.attrs {
                        if attr.path().is_ident("param") {
                            if let Ok(param_attr) = attr.parse_args::<ParamAttr>() {
                                if let Some(v) = &param_attr.min {
                                    min = v.to_json_string();
                                }
                                if let Some(v) = &param_attr.max {
                                    max = v.to_json_string();
                                }
                                if let Some(v) = &param_attr.step {
                                    step = v.to_json_string();
                                }
                                if let Some(v) = &param_attr.default {
                                    default_val = v.to_json_string();
                                }
                                if let Some(h) = &param_attr.hint {
                                    hint = h.clone();
                                }
                                if let Some(o) = &param_attr.options {
                                    options = super::param_attr::parse_options_string(o);
                                }
                            }
                        }
                    }

                    if default_val.is_empty() {
                        default_val = default_for_type(&ftype);
                    }

                    fields.push(ParamField {
                        name: fname,
                        param_type: ftype,
                        min,
                        max,
                        step,
                        default_val,
                        label,
                        hint,
                        options,
                    });
                }
            }

            if !fields.is_empty() {
                result.insert(
                    struct_name,
                    ParsedStruct {
                        fields,
                        config_hint,
                    },
                );
            }
        }
    }
    result
}

/// Resolve nested struct fields by inlining ConfigParams sub-structs.
fn resolve_nested_fields(
    fields: &[ParamField],
    _struct_hint: &str,
    all_structs: &HashMap<String, ParsedStruct>,
) -> Vec<ParamField> {
    let mut result = Vec::new();
    for field in fields {
        let type_name = field
            .param_type
            .rsplit("::")
            .next()
            .unwrap_or(&field.param_type);
        if let Some(nested) = all_structs
            .get(&field.param_type)
            .or_else(|| all_structs.get(type_name))
        {
            let hint = if field.hint.is_empty() {
                &nested.config_hint
            } else {
                &field.hint
            };
            for nf in &nested.fields {
                result.push(ParamField {
                    name: format!("{}.{}", field.name, nf.name),
                    param_type: nf.param_type.clone(),
                    min: nf.min.clone(),
                    max: nf.max.clone(),
                    step: nf.step.clone(),
                    default_val: nf.default_val.clone(),
                    label: nf.label.clone(),
                    hint: if hint.is_empty() {
                        nf.hint.clone()
                    } else {
                        hint.to_string()
                    },
                    options: nf.options.clone(),
                });
            }
        } else {
            result.push(field.clone());
        }
    }
    result
}

fn default_for_type(ty: &str) -> String {
    match ty {
        "f32" | "f64" => "0.0".to_string(),
        "u8" | "u16" | "u32" | "u64" | "i8" | "i16" | "i32" | "i64" => "0".to_string(),
        "bool" => "false".to_string(),
        _ => "null".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_config_params() {
        let source = r#"
            #[derive(ConfigParams)]
            pub struct BlurParams {
                /// Blur radius in pixels
                #[param(min = 0.0, max = 100.0, step = 0.5, default = 3.0)]
                pub radius: f32,
            }
        "#;
        let file = syn::parse_file(source).unwrap();
        let structs = extract_config_params(&file);
        let fields = structs.get("BlurParams").unwrap();
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].name, "radius");
        assert_eq!(fields[0].param_type, "f32");
        assert_eq!(fields[0].min, "0");
        assert_eq!(fields[0].max, "100");
        assert_eq!(fields[0].step, "0.5");
        assert_eq!(fields[0].default_val, "3");
        assert_eq!(fields[0].label, "Blur radius in pixels");
    }

    #[test]
    fn parse_multiple_fields() {
        let source = r#"
            #[derive(ConfigParams)]
            pub struct MixerParams {
                #[param(min = -2.0, max = 2.0, step = 0.01, default = 1.0)]
                pub rr: f32,
                #[param(min = -2.0, max = 2.0, step = 0.01, default = 0.0)]
                pub rg: f32,
            }
        "#;
        let file = syn::parse_file(source).unwrap();
        let structs = extract_config_params(&file);
        let fields = structs.get("MixerParams").unwrap();
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].name, "rr");
        assert_eq!(fields[0].default_val, "1");
        assert_eq!(fields[1].name, "rg");
        assert_eq!(fields[1].default_val, "0");
    }

    #[test]
    fn parse_with_hint() {
        let source = r#"
            #[derive(ConfigParams)]
            pub struct ColorParams {
                #[param(min = 0, max = 255, step = 1, default = 128, hint = "rc.color_rgb")]
                pub r: u8,
            }
        "#;
        let file = syn::parse_file(source).unwrap();
        let structs = extract_config_params(&file);
        let fields = structs.get("ColorParams").unwrap();
        assert_eq!(fields[0].hint, "rc.color_rgb");
    }

    #[test]
    fn parse_no_param_attr_uses_defaults() {
        let source = r#"
            #[derive(ConfigParams)]
            pub struct SimpleParams {
                pub value: f32,
            }
        "#;
        let file = syn::parse_file(source).unwrap();
        let structs = extract_config_params(&file);
        let fields = structs.get("SimpleParams").unwrap();
        assert_eq!(fields[0].default_val, "0.0");
        assert_eq!(fields[0].min, "null");
    }

    #[test]
    fn ignores_non_config_params_structs() {
        let source = r#"
            #[derive(Debug, Clone)]
            pub struct NotParams {
                pub x: u32,
            }
        "#;
        let file = syn::parse_file(source).unwrap();
        let structs = extract_config_params(&file);
        assert!(structs.is_empty());
    }

    #[test]
    fn parse_nested_config_params() {
        let source = r#"
            #[derive(ConfigParams)]
            #[config_hint("rc.color_rgb")]
            pub struct ColorRgb {
                #[param(min = 0, max = 255, step = 1, default = 128)]
                pub r: u8,
                #[param(min = 0, max = 255, step = 1, default = 128)]
                pub g: u8,
                #[param(min = 0, max = 255, step = 1, default = 128)]
                pub b: u8,
            }

            #[derive(ConfigParams)]
            pub struct ColorizeParams {
                pub target: ColorRgb,
                #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
                pub amount: f32,
            }
        "#;
        let file = syn::parse_file(source).unwrap();
        let structs = extract_config_params(&file);
        let fields = structs.get("ColorizeParams").unwrap();
        // ColorRgb should be inlined as target.r, target.g, target.b
        assert_eq!(fields.len(), 4); // 3 from ColorRgb + 1 amount
        assert_eq!(fields[0].name, "target.r");
        assert_eq!(fields[0].hint, "rc.color_rgb");
        assert_eq!(fields[3].name, "amount");
    }

    #[test]
    fn parse_param_with_options() {
        let source = r#"
            #[derive(ConfigParams)]
            pub struct ColorizeParams {
                /// Colorize method
                #[param(default = "w3c", hint = "rc.enum", options = "w3c:PS/W3C standard|lab:CIELAB perceptual")]
                pub method: String,
            }
        "#;
        let file = syn::parse_file(source).unwrap();
        let structs = extract_config_params(&file);
        let fields = structs.get("ColorizeParams").unwrap();
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].name, "method");
        assert_eq!(fields[0].hint, "rc.enum");
        assert_eq!(fields[0].options.len(), 2);
        assert_eq!(fields[0].options[0].0, "w3c");
        assert_eq!(fields[0].options[0].1, "PS/W3C standard");
        assert_eq!(fields[0].options[1].0, "lab");
        assert_eq!(fields[0].options[1].1, "CIELAB perceptual");
    }

    #[test]
    fn parse_param_without_options_has_empty_vec() {
        let source = r#"
            #[derive(ConfigParams)]
            pub struct BlurParams {
                #[param(min = 0.0, max = 100.0, default = 3.0)]
                pub radius: f32,
            }
        "#;
        let file = syn::parse_file(source).unwrap();
        let structs = extract_config_params(&file);
        let fields = structs.get("BlurParams").unwrap();
        assert!(fields[0].options.is_empty());
    }
}
