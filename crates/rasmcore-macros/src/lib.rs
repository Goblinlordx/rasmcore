//! Rasmcore proc macros for auto-registration of filters, encoders, and decoders.
//!
//! # Filter Registration
//!
//! ```ignore
//! #[register_filter(name = "blur", category = "spatial")]
//! pub fn blur(pixels: &[u8], info: &ImageInfo, params: BlurParams) -> Result<Vec<u8>, ImageError> {
//!     // implementation
//! }
//! ```
//!
//! # Config Struct Params
//!
//! ```ignore
//! #[derive(ConfigParams)]
//! pub struct BlurParams {
//!     /// Blur radius in pixels
//!     #[param(min = 0.0, max = 100.0, step = 0.5, default = 3.0)]
//!     pub radius: f32,
//! }
//! ```

use proc_macro::TokenStream;
use quote::{quote, format_ident};
use syn::{parse_macro_input, ItemFn, ItemStruct, LitStr, Ident, Token, Fields, Meta, Expr, Lit};
use syn::parse::{Parse, ParseStream};

// ─── Filter Registration ────────────────────────────────────────────────────

struct RegisterFilterArgs {
    name: String,
    category: String,
}

impl Parse for RegisterFilterArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut name = String::new();
        let mut category = String::new();

        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            let _: Token![=] = input.parse()?;

            match ident.to_string().as_str() {
                "name" => { let lit: LitStr = input.parse()?; name = lit.value(); }
                "category" => { let lit: LitStr = input.parse()?; category = lit.value(); }
                other => return Err(syn::Error::new(ident.span(), format!("unknown attribute: {other}"))),
            }

            if input.peek(Token![,]) { let _: Token![,] = input.parse()?; }
        }

        if name.is_empty() {
            return Err(syn::Error::new(proc_macro2::Span::call_site(), "missing required `name` attribute"));
        }

        Ok(RegisterFilterArgs { name, category })
    }
}

#[proc_macro_attribute]
pub fn register_filter(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as RegisterFilterArgs);
    let input_fn = parse_macro_input!(item as ItemFn);

    let fn_name = &input_fn.sig.ident;
    let filter_name = &args.name;
    let filter_category = &args.category;
    let reg_ident = format_ident!("__RASMCORE_FILTER_{}", fn_name.to_string().to_uppercase());
    let params = &input_fn.sig.inputs;
    let param_count = params.len().saturating_sub(2);

    let expanded = quote! {
        #input_fn

        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        pub static #reg_ident: ::rasmcore_image::domain::filter_registry::StaticFilterRegistration =
            ::rasmcore_image::domain::filter_registry::StaticFilterRegistration {
                name: #filter_name,
                category: #filter_category,
                param_count: #param_count,
                fn_name: stringify!(#fn_name),
                module_path: module_path!(),
            };
        inventory::submit!(&#reg_ident);
    };

    TokenStream::from(expanded)
}

// ─── Encoder Registration ───────────────────────────────────────────────────

struct RegisterEncoderArgs { name: String, format: String, mime: String }

impl Parse for RegisterEncoderArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut name = String::new();
        let mut format = String::new();
        let mut mime = String::new();
        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            let _: Token![=] = input.parse()?;
            let lit: LitStr = input.parse()?;
            match ident.to_string().as_str() {
                "name" => name = lit.value(),
                "format" => format = lit.value(),
                "mime" => mime = lit.value(),
                other => return Err(syn::Error::new(ident.span(), format!("unknown: {other}"))),
            }
            if input.peek(Token![,]) { let _: Token![,] = input.parse()?; }
        }
        Ok(RegisterEncoderArgs { name, format, mime })
    }
}

#[proc_macro_attribute]
pub fn register_encoder(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as RegisterEncoderArgs);
    let input_fn = parse_macro_input!(item as ItemFn);
    let fn_name = &input_fn.sig.ident;
    let name = &args.name;
    let format = &args.format;
    let mime = &args.mime;
    let reg_ident = format_ident!("__RASMCORE_ENCODER_{}", fn_name.to_string().to_uppercase());
    let expanded = quote! {
        #input_fn
        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        pub static #reg_ident: ::rasmcore_image::domain::encoder::StaticEncoderRegistration =
            ::rasmcore_image::domain::encoder::StaticEncoderRegistration {
                name: #name, format: #format, mime: #mime, fn_name: stringify!(#fn_name),
            };
        inventory::submit!(&#reg_ident);
    };
    TokenStream::from(expanded)
}

// ─── Decoder Registration ───────────────────────────────────────────────────

struct RegisterDecoderArgs { name: String, formats: String }

impl Parse for RegisterDecoderArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut name = String::new();
        let mut formats = String::new();
        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            let _: Token![=] = input.parse()?;
            let lit: LitStr = input.parse()?;
            match ident.to_string().as_str() {
                "name" => name = lit.value(),
                "formats" => formats = lit.value(),
                other => return Err(syn::Error::new(ident.span(), format!("unknown: {other}"))),
            }
            if input.peek(Token![,]) { let _: Token![,] = input.parse()?; }
        }
        Ok(RegisterDecoderArgs { name, formats })
    }
}

#[proc_macro_attribute]
pub fn register_decoder(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as RegisterDecoderArgs);
    let input_fn = parse_macro_input!(item as ItemFn);
    let fn_name = &input_fn.sig.ident;
    let name = &args.name;
    let formats = &args.formats;
    let reg_ident = format_ident!("__RASMCORE_DECODER_{}", fn_name.to_string().to_uppercase());
    let expanded = quote! {
        #input_fn
        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        pub static #reg_ident: ::rasmcore_image::domain::decoder::StaticDecoderRegistration =
            ::rasmcore_image::domain::decoder::StaticDecoderRegistration {
                name: #name, formats: #formats, fn_name: stringify!(#fn_name),
            };
        inventory::submit!(&#reg_ident);
    };
    TokenStream::from(expanded)
}

// ─── ConfigParams Derive Macro ──────────────────────────────────────────────

/// Derive param metadata from config structs. Works for filter params,
/// encoder configs, or any operation config.
///
/// Generates:
/// - `param_descriptors()` → `Vec<ParamDescriptorJson>` with metadata
/// - `Default` impl using `#[param(default = ...)]` values
///
/// # Example
///
/// ```ignore
/// #[derive(ConfigParams)]
/// pub struct BlurParams {
///     /// Blur radius in pixels
///     #[param(min = 0.0, max = 100.0, step = 0.5, default = 3.0)]
///     pub radius: f32,
/// }
///
/// #[derive(ConfigParams)]
/// pub struct JpegConfig {
///     /// JPEG quality (1-100)
///     #[param(min = 1, max = 100, default = 85)]
///     pub quality: u8,
///     /// Enable progressive encoding
///     #[param(default = false)]
///     pub progressive: bool,
/// }
/// ```
#[proc_macro_derive(ConfigParams, attributes(param))]
pub fn derive_config_params(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as ItemStruct);
    let struct_name = &input.ident;

    let fields = match &input.fields {
        Fields::Named(f) => &f.named,
        _ => panic!("ConfigParams only supports named struct fields"),
    };

    let mut descriptor_entries = Vec::new();
    let mut default_entries = Vec::new();

    for field in fields {
        let field_name = field.ident.as_ref().unwrap();
        let field_name_str = field_name.to_string();
        let field_type = &field.ty;
        let field_type_str = quote!(#field_type).to_string().replace(' ', "");

        // Extract doc comment as label
        let label = field.attrs.iter()
            .filter(|a| a.path().is_ident("doc"))
            .filter_map(|a| {
                if let Meta::NameValue(nv) = &a.meta
                    && let Expr::Lit(lit) = &nv.value
                        && let Lit::Str(s) = &lit.lit {
                            return Some(s.value().trim().to_string());
                        }
                None
            })
            .next()
            .unwrap_or_default();

        // Parse #[param(...)] attributes
        let mut min_s = String::from("null");
        let mut max_s = String::from("null");
        let mut step_s = String::from("null");
        let mut default_s = String::new();
        let mut hint_s = String::new();

        for attr in &field.attrs {
            if !attr.path().is_ident("param") { continue; }
            let _ = attr.parse_nested_meta(|meta| {
                let key = meta.path.get_ident().unwrap().to_string();
                let value = meta.value()?;
                let lit: Lit = value.parse()?;
                let val = match &lit {
                    Lit::Float(f) => f.base10_digits().to_string(),
                    Lit::Int(i) => i.base10_digits().to_string(),
                    Lit::Str(s) => s.value(),
                    Lit::Bool(b) => b.value.to_string(),
                    _ => String::new(),
                };
                match key.as_str() {
                    "min" => min_s = val,
                    "max" => max_s = val,
                    "step" => step_s = val,
                    "default" => default_s = val,
                    "hint" => hint_s = val,
                    _ => {}
                }
                Ok(())
            });
        }

        // Default expression
        let default_expr = if !default_s.is_empty() {
            if default_s.contains('.') {
                let v: f64 = default_s.parse().unwrap_or(0.0);
                let lit = proc_macro2::Literal::f64_unsuffixed(v);
                quote! { #lit as #field_type }
            } else if default_s == "true" {
                quote! { true }
            } else if default_s == "false" {
                quote! { false }
            } else if default_s.chars().next().is_some_and(|c| c.is_ascii_digit() || c == '-') {
                let v: i64 = default_s.parse().unwrap_or(0);
                let lit = proc_macro2::Literal::i64_unsuffixed(v);
                quote! { #lit as #field_type }
            } else {
                quote! { Default::default() }
            }
        } else {
            quote! { Default::default() }
        };

        default_entries.push(quote! { #field_name: #default_expr });

        descriptor_entries.push(quote! {
            ::rasmcore_image::domain::filter_registry::ParamDescriptorJson {
                name: #field_name_str.to_string(),
                param_type: #field_type_str.to_string(),
                min: #min_s.to_string(),
                max: #max_s.to_string(),
                step: #step_s.to_string(),
                default_val: #default_s.to_string(),
                label: #label.to_string(),
                hint: #hint_s.to_string(),
            }
        });
    }

    let expanded = quote! {
        impl #struct_name {
            /// Get parameter descriptors for manifest generation.
            pub fn param_descriptors() -> ::std::vec::Vec<::rasmcore_image::domain::filter_registry::ParamDescriptorJson> {
                ::std::vec![#(#descriptor_entries),*]
            }
        }

        impl ::std::default::Default for #struct_name {
            fn default() -> Self {
                Self { #(#default_entries),* }
            }
        }
    };

    TokenStream::from(expanded)
}
