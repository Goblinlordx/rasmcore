//! Parse encoder config structs from syn AST for pipeline write method generation.

use std::collections::HashMap;

/// A parsed encoder with its config struct metadata.
#[derive(Debug, Clone)]
pub struct EncoderInfo {
    /// Format name (e.g., "jpeg", "png", "tiff")
    pub format: String,
    /// Config struct name (e.g., "JpegEncodeConfig")
    pub config_struct: String,
    /// Config fields with types and defaults
    pub fields: Vec<EncoderField>,
    /// The encoder function name
    pub encode_fn: String,
    /// The module path (e.g., "jpeg", "png")
    pub module: String,
    /// Whether the sink function accepts metadata (detected from signature)
    pub sink_takes_metadata: bool,
}

/// A field in an encoder config struct.
#[derive(Debug, Clone)]
pub struct EncoderField {
    pub name: String,
    pub rust_type: String,
    pub is_enum: bool,
    pub enum_variants: Vec<String>,
    pub default_val: String,
}

/// Extract encoder config structs from a parsed source file.
pub fn extract_encoder_configs(file: &syn::File) -> Vec<EncoderInfo> {
    let mut configs = Vec::new();
    let mut enums: HashMap<String, Vec<String>> = HashMap::new();

    for item in &file.items {
        if let syn::Item::Enum(e) = item {
            let variants: Vec<String> = e.variants.iter().map(|v| v.ident.to_string()).collect();
            enums.insert(e.ident.to_string(), variants);
        }
    }

    for item in &file.items {
        if let syn::Item::Struct(s) = item {
            let name = s.ident.to_string();
            if !name.ends_with("EncodeConfig") {
                continue;
            }
            let format = name
                .strip_suffix("EncodeConfig")
                .unwrap_or(&name)
                .to_lowercase();
            let mut fields = Vec::new();

            if let syn::Fields::Named(named) = &s.fields {
                for field in &named.named {
                    let fname = field
                        .ident
                        .as_ref()
                        .map(|i| i.to_string())
                        .unwrap_or_default();
                    let ftype = super::type_to_string(&field.ty);
                    let is_enum = enums.contains_key(&ftype);
                    let enum_variants = enums.get(&ftype).cloned().unwrap_or_default();
                    fields.push(EncoderField {
                        name: fname,
                        rust_type: ftype,
                        is_enum,
                        enum_variants,
                        default_val: String::new(),
                    });
                }
            }

            configs.push(EncoderInfo {
                format: format.clone(),
                config_struct: name,
                fields,
                encode_fn: String::new(),
                module: format,
                sink_takes_metadata: false,
            });
        }
    }
    configs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_jpeg_config() {
        let source = r#"
            pub struct JpegEncodeConfig {
                pub quality: u8,
                pub progressive: bool,
                pub turbo: bool,
            }
        "#;
        let file = syn::parse_file(source).unwrap();
        let configs = extract_encoder_configs(&file);
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].format, "jpeg");
        assert_eq!(configs[0].fields.len(), 3);
    }

    #[test]
    fn parse_config_with_enum() {
        let source = r#"
            pub enum TiffCompression { None, Lzw, Deflate, PackBits }
            pub struct TiffEncodeConfig { pub compression: TiffCompression }
        "#;
        let file = syn::parse_file(source).unwrap();
        let configs = extract_encoder_configs(&file);
        assert_eq!(configs.len(), 1);
        assert!(configs[0].fields[0].is_enum);
        assert_eq!(configs[0].fields[0].enum_variants.len(), 4);
    }

    #[test]
    fn parse_unit_config() {
        let source = "pub struct BmpEncodeConfig;";
        let file = syn::parse_file(source).unwrap();
        let configs = extract_encoder_configs(&file);
        assert_eq!(configs.len(), 1);
        assert!(configs[0].fields.is_empty());
    }

    #[test]
    fn ignores_non_encode_structs() {
        let source = "pub struct ImageInfo { pub width: u32 }";
        let file = syn::parse_file(source).unwrap();
        assert!(extract_encoder_configs(&file).is_empty());
    }
}
