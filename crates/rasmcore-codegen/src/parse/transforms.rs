//! Parse `#[register_transform(...)]` annotations from syn AST.
//!
//! Scans for `impl ImageNode for FooNode` blocks annotated with
//! `#[register_transform(name = "...")]`. Then finds the corresponding
//! struct's `new()` constructor to extract parameter types.

use std::collections::HashMap;

/// A parsed transform registration.
#[derive(Debug, Clone)]
pub struct TransformReg {
    /// Transform name (e.g., "resize", "crop")
    pub name: String,
    /// Node struct name (e.g., "ResizeNode")
    pub node_type: String,
    /// Module import path for the node (e.g., "transform", "color", "composite")
    pub node_module: String,
    /// Constructor parameters, excluding `upstream: u32` and `source_info: ImageInfo`.
    /// Each entry is (param_name, rust_type).
    pub params: Vec<(String, String)>,
    /// Whether the constructor returns Result (fallible).
    pub fallible: bool,
    /// Whether this is a multi-input node (e.g., composite with fg + bg).
    pub multi_input: bool,
}

/// An enum definition found in the source.
#[derive(Debug, Clone)]
pub struct EnumDef {
    pub name: String,
    pub variants: Vec<String>,
    /// Domain module path (e.g., "types", "metadata")
    pub domain_module: String,
}

/// Extract all transform registrations and enum definitions from a parsed source file.
/// `module_name` is the import path for nodes in this file (e.g., "transform", "color").
pub fn extract_transforms(
    file: &syn::File,
    enums: &HashMap<String, EnumDef>,
    module_name: &str,
) -> Vec<TransformReg> {
    // Collect struct constructors: struct_name -> params from impl block's new() method
    let mut constructors: HashMap<String, (Vec<(String, String)>, bool, bool)> = HashMap::new();
    for item in &file.items {
        if let syn::Item::Impl(impl_block) = item {
            // Only plain impl blocks (not trait impls)
            if impl_block.trait_.is_some() {
                continue;
            }
            let struct_name = type_name(&impl_block.self_ty);
            for item in &impl_block.items {
                if let syn::ImplItem::Fn(method) = item {
                    if method.sig.ident == "new" {
                        let (params, fallible, multi_input) =
                            extract_constructor_params(&method.sig, &enums);
                        constructors
                            .insert(struct_name.clone(), (params, fallible, multi_input));
                    }
                }
            }
        }
    }

    // Find #[register_transform] on impl ImageNode blocks
    let mut transforms = Vec::new();
    for item in &file.items {
        if let syn::Item::Impl(impl_block) = item {
            if let Some(mut reg) =
                extract_transform_reg(impl_block, &constructors)
            {
                reg.node_module = module_name.to_string();
                transforms.push(reg);
            }
        }
    }
    transforms
}

/// Collect all enum definitions from a source file.
/// `domain_module` is the module path (e.g., "types", "metadata").
pub fn extract_enums(file: &syn::File, domain_module: &str) -> HashMap<String, EnumDef> {
    let mut enums = HashMap::new();
    for item in &file.items {
        if let syn::Item::Enum(e) = item {
            let variants: Vec<String> = e.variants.iter().map(|v| v.ident.to_string()).collect();
            enums.insert(
                e.ident.to_string(),
                EnumDef {
                    name: e.ident.to_string(),
                    variants,
                    domain_module: domain_module.to_string(),
                },
            );
        }
    }
    enums
}

fn extract_transform_reg(
    impl_block: &syn::ItemImpl,
    constructors: &HashMap<String, (Vec<(String, String)>, bool, bool)>,
) -> Option<TransformReg> {
    // Must be a trait impl (impl ImageNode for ...)
    impl_block.trait_.as_ref()?;

    // Check for #[register_transform(...)] attribute
    for attr in &impl_block.attrs {
        let path = attr.path();
        let is_register = path.is_ident("register_transform")
            || path
                .segments
                .last()
                .map(|s| s.ident == "register_transform")
                .unwrap_or(false);
        if !is_register {
            continue;
        }

        let tokens = match &attr.meta {
            syn::Meta::List(ml) => ml.tokens.to_string(),
            _ => continue,
        };

        let name = extract_kv(&tokens, "name")?;
        let node_type = type_name(&impl_block.self_ty);

        // Look up the constructor params
        let (params, fallible, multi_input) = constructors
            .get(&node_type)
            .cloned()
            .unwrap_or_default();

        return Some(TransformReg {
            name,
            node_type,
            node_module: String::new(), // set by caller
            params,
            fallible,
            multi_input,
        });
    }
    None
}

/// Extract constructor parameters, skipping infrastructure params.
/// Returns (params, fallible, multi_input).
fn extract_constructor_params(
    sig: &syn::Signature,
    _enums: &HashMap<String, EnumDef>,
) -> (Vec<(String, String)>, bool, bool) {
    let mut params = Vec::new();
    let mut fallible = false;
    let mut upstream_count = 0u32;

    // Check return type for Result
    if let syn::ReturnType::Type(_, ty) = &sig.output {
        let ret_str = quote::quote!(#ty).to_string();
        if ret_str.contains("Result") {
            fallible = true;
        }
    }

    for input in &sig.inputs {
        if let syn::FnArg::Typed(pat_type) = input {
            let name = quote::quote!(#pat_type).to_string();
            // Skip infrastructure params — count upstream occurrences
            if name.contains("upstream") || name.contains("fg_upstream") || name.contains("bg_upstream") {
                upstream_count += 1;
                continue;
            }
            if name.contains("source_info")
                || name.contains("ImageInfo")
                || name.contains("self")
                || name.contains("fg_info")
                || name.contains("bg_info")
            {
                continue;
            }

            let param_name = if let syn::Pat::Ident(pi) = &*pat_type.pat {
                pi.ident.to_string()
            } else {
                continue;
            };

            let param_type = super::type_to_string(&pat_type.ty);

            // Skip the first u32 param if it looks like an upstream ID
            // (already filtered by name above)
            params.push((param_name, param_type));
        }
    }

    let multi_input = upstream_count > 1;
    (params, fallible, multi_input)
}

fn extract_kv(tokens: &str, key: &str) -> Option<String> {
    let pattern = format!("{key} = \"");
    let start = tokens.find(&pattern)?;
    let after = &tokens[start + pattern.len()..];
    let end = after.find('"')?;
    Some(after[..end].to_string())
}

fn type_name(ty: &syn::Type) -> String {
    if let syn::Type::Path(tp) = ty {
        tp.path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .unwrap_or_default()
    } else {
        String::new()
    }
}
