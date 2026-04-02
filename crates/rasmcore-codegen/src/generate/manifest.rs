//! Generate param-manifest.json from structured data.

use crate::types::{CodegenData, ParamField};
use serde_json::{json, Value};

use super::helpers::{default_range_for_type, to_pascal_case};

/// Parse a string value into a serde_json::Value (number, bool, array, or string).
fn parse_json_value(s: &str) -> Value {
    if s == "null" || s.is_empty() {
        return Value::Null;
    }
    if let Ok(v) = serde_json::from_str::<Value>(s) {
        return v;
    }
    Value::String(s.to_string())
}

/// Generate param-manifest.json content from codegen data.
pub fn generate(data: &CodegenData) -> String {
    let mut filter_entries: Vec<Value> = Vec::new();

    for f in &data.filters {
        let struct_name = format!("{}Params", to_pascal_case(&f.name));
        let params_struct = data.param_structs.get(&struct_name);

        let params_json: Vec<Value> = if let Some(fields) = params_struct {
            fields.iter().map(field_to_json).collect()
        } else if !f.params.is_empty() {
            f.params
                .iter()
                .map(|(pname, ptype)| {
                    let (min, max, step, def) = default_range_for_type(ptype);
                    json!({
                        "name": pname,
                        "type": ptype,
                        "min": parse_json_value(min),
                        "max": parse_json_value(max),
                        "step": parse_json_value(step),
                        "default": parse_json_value(def),
                        "label": "",
                        "hint": "",
                    })
                })
                .collect()
        } else {
            vec![]
        };

        filter_entries.push(json!({
            "name": f.name,
            "category": f.category,
            "group": f.group,
            "variant": f.variant,
            "reference": f.reference,
            "params": params_json,
        }));
    }

    let gen_entries: Vec<Value> = data
        .generators
        .iter()
        .map(|g| {
            json!({
                "name": g.name,
                "category": g.category,
                "group": g.group,
                "variant": g.variant,
                "reference": g.reference,
                "kind": "generator",
            })
        })
        .collect();

    let comp_entries: Vec<Value> = data
        .compositors
        .iter()
        .map(|c| {
            json!({
                "name": c.name,
                "category": c.category,
                "group": c.group,
                "variant": c.variant,
                "reference": c.reference,
                "kind": "compositor",
            })
        })
        .collect();

    let map_entries: Vec<Value> = data
        .mappers
        .iter()
        .map(|m| {
            let struct_name = format!("{}Params", to_pascal_case(&m.name));
            let params_struct = data.param_structs.get(&struct_name);

            let params_json: Vec<Value> = if let Some(fields) = params_struct {
                fields.iter().map(field_to_json).collect()
            } else if !m.params.is_empty() {
                m.params
                    .iter()
                    .map(|(pname, ptype)| {
                        let (min, max, step, def) = default_range_for_type(ptype);
                        json!({
                            "name": pname,
                            "type": ptype,
                            "min": parse_json_value(min),
                            "max": parse_json_value(max),
                            "step": parse_json_value(step),
                            "default": parse_json_value(def),
                            "label": "",
                            "hint": "",
                        })
                    })
                    .collect()
            } else {
                vec![]
            };

            json!({
                "name": m.name,
                "category": m.category,
                "group": m.group,
                "variant": m.variant,
                "reference": m.reference,
                "kind": "mapper",
                "params": params_json,
            })
        })
        .collect();

    // ─── Transforms ───
    let transform_entries: Vec<Value> = data
        .transforms
        .iter()
        .map(|t| {
            let params_json: Vec<Value> = t
                .params
                .iter()
                .map(|(pname, ptype)| {
                    let (min, max, step, def) = default_range_for_type(ptype);
                    json!({
                        "name": pname,
                        "type": ptype,
                        "min": parse_json_value(min),
                        "max": parse_json_value(max),
                        "step": parse_json_value(step),
                        "default": parse_json_value(def),
                        "label": "",
                        "hint": "",
                    })
                })
                .collect();

            json!({
                "name": t.name,
                "params": params_json,
                "multi_input": t.multi_input,
            })
        })
        .collect();

    // ─── Encoders ───
    let encoder_entries: Vec<Value> = data
        .encoders
        .iter()
        .map(|e| {
            let params_json: Vec<Value> = e
                .fields
                .iter()
                .map(|f| {
                    let mut entry = json!({
                        "name": f.name,
                        "type": f.rust_type,
                        "default": parse_json_value(&f.default_val),
                    });
                    if f.is_enum {
                        entry["enum_variants"] = json!(f.enum_variants);
                    }
                    entry
                })
                .collect();

            json!({
                "format": e.format,
                "params": params_json,
                "takes_metadata": e.sink_takes_metadata,
            })
        })
        .collect();

    let manifest = json!({
        "filters": filter_entries,
        "generators": gen_entries,
        "compositors": comp_entries,
        "mappers": map_entries,
        "transforms": transform_entries,
        "encoders": encoder_entries,
    });

    serde_json::to_string_pretty(&manifest).unwrap()
}

fn field_to_json(field: &ParamField) -> Value {
    json!({
        "name": field.name,
        "type": field.param_type,
        "min": parse_json_value(&field.min),
        "max": parse_json_value(&field.max),
        "step": parse_json_value(&field.step),
        "default": parse_json_value(&field.default_val),
        "label": field.label,
        "hint": field.hint,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CodegenData, FilterReg};
    use crate::parse::transforms::TransformReg;
    use crate::parse::encoders::EncoderInfo;

    #[test]
    fn generate_manifest_basic() {
        let data = CodegenData {
            filters: vec![FilterReg {
                name: "blur".to_string(),
                category: "spatial".to_string(),
                group: String::new(),
                variant: String::new(),
                reference: String::new(),
                fn_name: "blur".to_string(),
                params: vec![("radius".to_string(), "f32".to_string())],
                config_struct: None,
                point_op: false,
                color_op: false,
                rect_request: true,
            }],
            generators: vec![],
            compositors: vec![],
            mappers: vec![],
            param_structs: std::collections::HashMap::new(),
            transforms: vec![],
            encoders: vec![],
        };

        let json_str = generate(&data);
        let parsed: Value = serde_json::from_str(&json_str).unwrap();

        assert_eq!(parsed["filters"].as_array().unwrap().len(), 1);
        assert_eq!(parsed["filters"][0]["name"], "blur");
        assert_eq!(parsed["filters"][0]["params"][0]["name"], "radius");
        assert_eq!(parsed["filters"][0]["params"][0]["type"], "f32");
    }

    #[test]
    fn generate_manifest_valid_json() {
        let data = CodegenData {
            filters: vec![],
            generators: vec![],
            compositors: vec![],
            mappers: vec![],
            param_structs: std::collections::HashMap::new(),
            transforms: vec![],
            encoders: vec![],
        };
        let json_str = generate(&data);
        let parsed: Value = serde_json::from_str(&json_str).unwrap();
        assert!(parsed["filters"].as_array().unwrap().is_empty());
        assert!(parsed["transforms"].as_array().unwrap().is_empty());
        assert!(parsed["encoders"].as_array().unwrap().is_empty());
    }
}
