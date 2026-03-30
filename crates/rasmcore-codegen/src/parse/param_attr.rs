//! Fully typed syn parser for `#[param(min = 0.0, max = 1.0, ...)]` attributes.

use crate::types::LitValue;
use syn::parse::{Parse, ParseStream};
use syn::{Ident, Lit, Token};

/// Parsed `#[param(...)]` attribute with all fields.
#[derive(Debug, Clone, Default)]
pub struct ParamAttr {
    pub min: Option<LitValue>,
    pub max: Option<LitValue>,
    pub step: Option<LitValue>,
    pub default: Option<LitValue>,
    pub hint: Option<String>,
}

/// A single key = value pair inside #[param(...)].
struct ParamKV {
    key: String,
    value: LitValue,
}

impl Parse for ParamKV {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let key: Ident = input.parse()?;
        input.parse::<Token![=]>()?;

        // Handle negative numbers: if we see a `-`, parse it as part of the literal
        let neg = input.peek(Token![-]);
        if neg {
            input.parse::<Token![-]>()?;
        }

        let value = if input.peek(Lit) {
            let lit: Lit = input.parse()?;
            match lit {
                Lit::Float(f) => {
                    let v: f64 = f.base10_parse()?;
                    LitValue::Float(if neg { -v } else { v })
                }
                Lit::Int(i) => {
                    let v: i64 = i.base10_parse()?;
                    LitValue::Int(if neg { -v } else { v })
                }
                Lit::Bool(b) => LitValue::Bool(b.value),
                Lit::Str(s) => LitValue::Str(s.value()),
                _ => LitValue::Null,
            }
        } else {
            // Try parsing as an identifier (e.g., `true`, `false`, or a path)
            let ident: Ident = input.parse()?;
            match ident.to_string().as_str() {
                "true" => LitValue::Bool(true),
                "false" => LitValue::Bool(false),
                _ => LitValue::Str(ident.to_string()),
            }
        };

        Ok(ParamKV {
            key: key.to_string(),
            value,
        })
    }
}

impl Parse for ParamAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut attr = ParamAttr::default();

        while !input.is_empty() {
            let kv: ParamKV = input.parse()?;
            match kv.key.as_str() {
                "min" => attr.min = Some(kv.value),
                "max" => attr.max = Some(kv.value),
                "step" => attr.step = Some(kv.value),
                "default" => attr.default = Some(kv.value),
                "hint" => {
                    if let LitValue::Str(s) = kv.value {
                        attr.hint = Some(s);
                    }
                }
                _ => {} // ignore unknown keys
            }

            // Consume trailing comma if present
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(attr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_param(tokens: &str) -> ParamAttr {
        syn::parse_str::<ParamAttr>(tokens).unwrap()
    }

    #[test]
    fn parse_full_param() {
        let attr = parse_param(r#"min = 0.0, max = 1.0, step = 0.01, default = 0.5, hint = "rc.angle_deg""#);
        assert_eq!(attr.min, Some(LitValue::Float(0.0)));
        assert_eq!(attr.max, Some(LitValue::Float(1.0)));
        assert_eq!(attr.step, Some(LitValue::Float(0.01)));
        assert_eq!(attr.default, Some(LitValue::Float(0.5)));
        assert_eq!(attr.hint.as_deref(), Some("rc.angle_deg"));
    }

    #[test]
    fn parse_int_params() {
        let attr = parse_param("min = 0, max = 255, step = 1, default = 128");
        assert_eq!(attr.min, Some(LitValue::Int(0)));
        assert_eq!(attr.max, Some(LitValue::Int(255)));
        assert_eq!(attr.step, Some(LitValue::Int(1)));
        assert_eq!(attr.default, Some(LitValue::Int(128)));
    }

    #[test]
    fn parse_negative_values() {
        let attr = parse_param("min = -100.0, max = 100.0, default = 0.0");
        assert_eq!(attr.min, Some(LitValue::Float(-100.0)));
        assert_eq!(attr.max, Some(LitValue::Float(100.0)));
    }

    #[test]
    fn parse_bool_default() {
        let attr = parse_param("default = true");
        assert_eq!(attr.default, Some(LitValue::Bool(true)));
    }

    #[test]
    fn parse_string_default() {
        let attr = parse_param(r#"default = "0.0:000000,1.0:FFFFFF""#);
        assert_eq!(
            attr.default,
            Some(LitValue::Str("0.0:000000,1.0:FFFFFF".to_string()))
        );
    }

    #[test]
    fn parse_hint_only() {
        let attr = parse_param(r#"hint = "rc.color_rgb""#);
        assert_eq!(attr.hint.as_deref(), Some("rc.color_rgb"));
        assert_eq!(attr.min, None);
    }

    #[test]
    fn parse_empty() {
        let attr = parse_param("");
        assert_eq!(attr.min, None);
        assert_eq!(attr.max, None);
    }

    #[test]
    fn parse_trailing_comma() {
        let attr = parse_param("min = 0, max = 255,");
        assert_eq!(attr.min, Some(LitValue::Int(0)));
        assert_eq!(attr.max, Some(LitValue::Int(255)));
    }
}
