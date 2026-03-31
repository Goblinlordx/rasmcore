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
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{Expr, Fields, Ident, ItemFn, ItemStruct, Lit, LitStr, Meta, Token, parse_macro_input};

// ─── Filter Registration ────────────────────────────────────────────────────

struct RegisterFilterArgs {
    name: String,
    category: String,
    group: String,
    variant: String,
    reference: String,
}

impl Parse for RegisterFilterArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut name = String::new();
        let mut category = String::new();
        let mut group = String::new();
        let mut variant = String::new();
        let mut reference = String::new();

        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            let _: Token![=] = input.parse()?;

            match ident.to_string().as_str() {
                "name" => {
                    let lit: LitStr = input.parse()?;
                    name = lit.value();
                }
                "category" => {
                    let lit: LitStr = input.parse()?;
                    category = lit.value();
                }
                "group" => {
                    let lit: LitStr = input.parse()?;
                    group = lit.value();
                }
                "variant" => {
                    let lit: LitStr = input.parse()?;
                    variant = lit.value();
                }
                "reference" => {
                    let lit: LitStr = input.parse()?;
                    reference = lit.value();
                }
                "overlap" => {
                    // Overlap is parsed by build.rs for pipeline node generation.
                    // The proc macro just skips it.
                    let _lit: LitStr = input.parse()?;
                }
                "output_format" => {
                    // Output format is parsed by build.rs for mapper pipeline node generation.
                    // The proc macro just skips it.
                    let _lit: LitStr = input.parse()?;
                }
                "point_op" => {
                    // Point op flag is parsed by build.rs for LUT fusion.
                    // The proc macro just skips it.
                    let _lit: LitStr = input.parse()?;
                }
                "color_op" => {
                    // Color op flag is parsed by build.rs for 3D CLUT fusion.
                    // The proc macro just skips it.
                    let _lit: LitStr = input.parse()?;
                }
                other => {
                    return Err(syn::Error::new(
                        ident.span(),
                        format!("unknown attribute: {other}"),
                    ));
                }
            }

            if input.peek(Token![,]) {
                let _: Token![,] = input.parse()?;
            }
        }

        if name.is_empty() {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "missing required `name` attribute",
            ));
        }

        Ok(RegisterFilterArgs {
            name,
            category,
            group,
            variant,
            reference,
        })
    }
}

#[proc_macro_attribute]
pub fn register_filter(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as RegisterFilterArgs);
    let input_fn = parse_macro_input!(item as ItemFn);

    let fn_name = &input_fn.sig.ident;
    let filter_name = &args.name;
    let filter_category = &args.category;
    let filter_group = &args.group;
    let filter_variant = &args.variant;
    let filter_reference = &args.reference;
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
                group: #filter_group,
                variant: #filter_variant,
                reference: #filter_reference,
                param_count: #param_count,
                fn_name: stringify!(#fn_name),
                module_path: module_path!(),
            };
        inventory::submit!(&#reg_ident);
    };

    TokenStream::from(expanded)
}

// ─── Generator Registration ────────────────────────────────────────────────

/// Register a generator function (procedural image source — no pixel input).
///
/// Generators create images from config parameters (e.g., noise, gradients).
/// The function signature takes only config params (no pixels/info) and returns
/// pixel data.
#[proc_macro_attribute]
pub fn register_generator(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as RegisterFilterArgs);
    let input_fn = parse_macro_input!(item as ItemFn);

    let fn_name = &input_fn.sig.ident;
    let gen_name = &args.name;
    let gen_category = &args.category;
    let gen_group = &args.group;
    let gen_variant = &args.variant;
    let gen_reference = &args.reference;
    let reg_ident = format_ident!(
        "__RASMCORE_GENERATOR_{}",
        fn_name.to_string().to_uppercase()
    );
    let params = &input_fn.sig.inputs;
    let param_count = params.len();

    let expanded = quote! {
        #input_fn

        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        pub static #reg_ident: ::rasmcore_image::domain::filter_registry::StaticGeneratorRegistration =
            ::rasmcore_image::domain::filter_registry::StaticGeneratorRegistration {
                name: #gen_name,
                category: #gen_category,
                group: #gen_group,
                variant: #gen_variant,
                reference: #gen_reference,
                param_count: #param_count,
                fn_name: stringify!(#fn_name),
                module_path: module_path!(),
            };
        inventory::submit!(&#reg_ident);
    };

    TokenStream::from(expanded)
}

// ─── Compositor Registration ───────────────────────────────────────────────

/// Register a compositor function (multi-input blending/composition).
///
/// Compositors take two or more image inputs and produce a single output.
/// The function signature includes multiple pixel/info pairs.
#[proc_macro_attribute]
pub fn register_compositor(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as RegisterFilterArgs);
    let input_fn = parse_macro_input!(item as ItemFn);

    let fn_name = &input_fn.sig.ident;
    let comp_name = &args.name;
    let comp_category = &args.category;
    let comp_group = &args.group;
    let comp_variant = &args.variant;
    let comp_reference = &args.reference;
    let reg_ident = format_ident!(
        "__RASMCORE_COMPOSITOR_{}",
        fn_name.to_string().to_uppercase()
    );
    let params = &input_fn.sig.inputs;
    let param_count = params.len();

    let expanded = quote! {
        #input_fn

        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        pub static #reg_ident: ::rasmcore_image::domain::filter_registry::StaticCompositorRegistration =
            ::rasmcore_image::domain::filter_registry::StaticCompositorRegistration {
                name: #comp_name,
                category: #comp_category,
                group: #comp_group,
                variant: #comp_variant,
                reference: #comp_reference,
                param_count: #param_count,
                fn_name: stringify!(#fn_name),
                module_path: module_path!(),
            };
        inventory::submit!(&#reg_ident);
    };

    TokenStream::from(expanded)
}

// ─── Mapper Registration ───────────────────────────────────────────────────

/// Register a mapper function (format-changing operation).
///
/// Mappers take a single image input and produce an output with potentially
/// different pixel format (e.g., RGB8 → Gray8, RGBA8 → RGB8).
#[proc_macro_attribute]
pub fn register_mapper(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as RegisterFilterArgs);
    let input_fn = parse_macro_input!(item as ItemFn);

    let fn_name = &input_fn.sig.ident;
    let map_name = &args.name;
    let map_category = &args.category;
    let map_group = &args.group;
    let map_variant = &args.variant;
    let map_reference = &args.reference;
    let reg_ident = format_ident!("__RASMCORE_MAPPER_{}", fn_name.to_string().to_uppercase());
    let params = &input_fn.sig.inputs;
    let param_count = params.len().saturating_sub(2); // subtract pixels + info

    let expanded = quote! {
        #input_fn

        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        pub static #reg_ident: ::rasmcore_image::domain::filter_registry::StaticMapperRegistration =
            ::rasmcore_image::domain::filter_registry::StaticMapperRegistration {
                name: #map_name,
                category: #map_category,
                group: #map_group,
                variant: #map_variant,
                reference: #map_reference,
                param_count: #param_count,
                fn_name: stringify!(#fn_name),
                module_path: module_path!(),
            };
        inventory::submit!(&#reg_ident);
    };

    TokenStream::from(expanded)
}

// ─── Encoder Registration ───────────────────────────────────────────────────

struct RegisterEncoderArgs {
    name: String,
    format: String,
    mime: String,
    extensions: Vec<String>,
}

impl Parse for RegisterEncoderArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut name = String::new();
        let mut format = String::new();
        let mut mime = String::new();
        let mut extensions = Vec::new();
        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            let _: Token![=] = input.parse()?;
            let lit: LitStr = input.parse()?;
            match ident.to_string().as_str() {
                "name" => name = lit.value(),
                "format" => format = lit.value(),
                "mime" => mime = lit.value(),
                "extensions" => {
                    extensions = lit
                        .value()
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .collect();
                }
                other => return Err(syn::Error::new(ident.span(), format!("unknown: {other}"))),
            }
            if input.peek(Token![,]) {
                let _: Token![,] = input.parse()?;
            }
        }
        Ok(RegisterEncoderArgs {
            name,
            format,
            mime,
            extensions,
        })
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
    let ext_strs: Vec<_> = args.extensions.iter().map(|e| quote! { #e }).collect();
    let reg_ident = format_ident!("__RASMCORE_ENCODER_{}", fn_name.to_string().to_uppercase());
    let expanded = quote! {
        #input_fn
        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        pub static #reg_ident: ::rasmcore_image::domain::encoder::StaticEncoderRegistration =
            ::rasmcore_image::domain::encoder::StaticEncoderRegistration {
                name: #name,
                format: #format,
                mime: #mime,
                extensions: &[#(#ext_strs),*],
                fn_name: stringify!(#fn_name),
            };
        inventory::submit!(&#reg_ident);
    };
    TokenStream::from(expanded)
}

// ─── Decoder Registration ───────────────────────────────────────────────────

struct RegisterDecoderArgs {
    name: String,
    formats: String,
}

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
            if input.peek(Token![,]) {
                let _: Token![,] = input.parse()?;
            }
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
/// - `config_hint()` → `&'static str` (from `#[config_hint("...")]`, or `""`)
/// - `Default` impl using `#[param(default = ...)]` values
///
/// # Type-level hints
///
/// Add `#[config_hint("rc.color_rgba")]` on the struct to define a type-level
/// hint. When this struct is used as a field in another ConfigParams, the hint
/// propagates automatically to all flattened params.
///
/// # Nested structs
///
/// Fields whose type is another ConfigParams struct are auto-flattened:
/// their descriptors are prefixed with the field name (e.g., `color.r`)
/// and the nested type's `config_hint()` propagates unless the field has
/// an explicit `#[param(hint = "...")]` override.
///
/// # Example
///
/// ```ignore
/// #[derive(ConfigParams)]
/// #[config_hint("rc.color_rgba")]
/// pub struct ColorRgba {
///     #[param(min = 0, max = 255, step = 1, default = 255)]
///     pub r: u8,
///     // ...
/// }
///
/// #[derive(ConfigParams)]
/// pub struct DrawLineParams {
///     pub color: ColorRgba,  // auto-flattened, hint inherited
///     #[param(min = 0.5, max = 100.0, step = 0.5, default = 2.0)]
///     pub width: f32,
/// }
/// ```
#[proc_macro_derive(ConfigParams, attributes(param, config_hint))]
pub fn derive_config_params(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as ItemStruct);
    let struct_name = &input.ident;

    // Parse struct-level #[config_hint("...")] attribute
    let struct_hint = input
        .attrs
        .iter()
        .filter(|a| a.path().is_ident("config_hint"))
        .filter_map(|a| {
            // Parse #[config_hint("rc.color_rgba")]
            let tokens: proc_macro2::TokenStream = a.meta.require_list().ok()?.tokens.clone();
            let lit: LitStr = syn::parse2(tokens).ok()?;
            Some(lit.value())
        })
        .next()
        .unwrap_or_default();

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

        // Check if this is a primitive/known type or a nested ConfigParams struct
        let is_primitive = is_primitive_param_type(&field_type_str);

        // Extract doc comment as label
        let label = field
            .attrs
            .iter()
            .filter(|a| a.path().is_ident("doc"))
            .filter_map(|a| {
                if let Meta::NameValue(nv) = &a.meta
                    && let Expr::Lit(lit) = &nv.value
                    && let Lit::Str(s) = &lit.lit
                {
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
            if !attr.path().is_ident("param") {
                continue;
            }
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

        if !is_primitive {
            // Nested ConfigParams struct — flatten its descriptors with prefix
            // The hint comes from: field-level override > nested type's config_hint
            let hint_override = hint_s.clone();
            descriptor_entries.push(quote! {{
                let nested = #field_type::param_descriptors();
                let type_hint = #field_type::config_hint();
                let hint = if #hint_override.is_empty() {
                    type_hint.to_string()
                } else {
                    #hint_override.to_string()
                };
                for mut d in nested {
                    d.name = format!("{}.{}", #field_name_str, d.name);
                    if !hint.is_empty() {
                        d.hint = hint.clone();
                    }
                    __descriptors.push(d);
                }
            }});

            // Default: use nested type's Default impl
            default_entries.push(quote! { #field_name: Default::default() });
        } else {
            // Primitive field — standard descriptor
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
                } else if default_s
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_digit() || c == '-')
                {
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
                __descriptors.push(::rasmcore_image::domain::filter_registry::ParamDescriptorJson {
                    name: #field_name_str.to_string(),
                    param_type: #field_type_str.to_string(),
                    min: #min_s.to_string(),
                    max: #max_s.to_string(),
                    step: #step_s.to_string(),
                    default_val: #default_s.to_string(),
                    label: #label.to_string(),
                    hint: #hint_s.to_string(),
                });
            });
        }
    }

    let expanded = quote! {
        impl #struct_name {
            /// Get parameter descriptors for manifest generation.
            pub fn param_descriptors() -> ::std::vec::Vec<::rasmcore_image::domain::filter_registry::ParamDescriptorJson> {
                let mut __descriptors = ::std::vec::Vec::new();
                #(#descriptor_entries)*
                __descriptors
            }

            /// Type-level UI hint (e.g., `"rc.color_rgba"`). Empty string if none.
            pub fn config_hint() -> &'static str {
                #struct_hint
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

/// Check if a type string represents a primitive param type (not a nested ConfigParams struct).
fn is_primitive_param_type(ty: &str) -> bool {
    matches!(
        ty,
        "f32"
            | "f64"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "i8"
            | "i16"
            | "i32"
            | "i64"
            | "bool"
            | "&str"
            | "String"
            | "[u8;3]"
            | "[u8;4]"
    )
}
