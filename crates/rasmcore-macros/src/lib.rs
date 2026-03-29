//! Rasmcore proc macros for auto-registration of filters, encoders, and decoders.
//!
//! # Filter Registration
//!
//! ```ignore
//! #[register_filter(name = "blur", category = "spatial")]
//! pub fn blur(pixels: &[u8], info: &ImageInfo, radius: f32) -> Result<Vec<u8>, ImageError> {
//!     // implementation
//! }
//! ```
//!
//! The macro keeps the original function unchanged and generates:
//! 1. A `FilterRegistration` const with metadata (name, category, function pointer)
//! 2. An `inventory::submit!` call for cross-crate collection
//!
//! The adapter layer uses `inventory::iter::<FilterRegistration>` to auto-generate
//! dispatch without manual wiring.

use proc_macro::TokenStream;
use quote::{quote, format_ident};
use syn::{parse_macro_input, ItemFn, LitStr, Ident, Token};
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
                "name" => {
                    let lit: LitStr = input.parse()?;
                    name = lit.value();
                }
                "category" => {
                    let lit: LitStr = input.parse()?;
                    category = lit.value();
                }
                other => {
                    return Err(syn::Error::new(ident.span(), format!("unknown attribute: {other}")));
                }
            }

            if input.peek(Token![,]) {
                let _: Token![,] = input.parse()?;
            }
        }

        if name.is_empty() {
            return Err(syn::Error::new(proc_macro2::Span::call_site(), "missing required `name` attribute"));
        }

        Ok(RegisterFilterArgs { name, category })
    }
}

/// Register a function as a rasmcore filter.
///
/// The macro generates a `FilterRegistration` that is collected via `inventory`
/// at link time across all crates. The adapter layer iterates these registrations
/// to build dispatch tables automatically.
///
/// # Attributes
///
/// - `name` (required): The filter name as exposed in WIT/SDK (e.g., "blur")
/// - `category` (required): Filter category for grouping (e.g., "spatial", "color", "edge")
///
/// # Example
///
/// ```ignore
/// #[register_filter(name = "blur", category = "spatial")]
/// pub fn blur(pixels: &[u8], info: &ImageInfo, radius: f32) -> Result<Vec<u8>, ImageError> {
///     // ...
/// }
/// ```
#[proc_macro_attribute]
pub fn register_filter(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as RegisterFilterArgs);
    let input_fn = parse_macro_input!(item as ItemFn);

    let fn_name = &input_fn.sig.ident;
    let filter_name = &args.name;
    let filter_category = &args.category;

    // Generate unique identifier for the registration
    let reg_ident = format_ident!("__RASMCORE_FILTER_{}", fn_name.to_string().to_uppercase());

    // Extract parameter info from function signature (skip pixels and info)
    let params = &input_fn.sig.inputs;
    let param_count = params.len().saturating_sub(2); // subtract pixels + info

    let expanded = quote! {
        // Original function — unchanged
        #input_fn

        // Filter registration metadata
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

        // Submit to inventory for cross-crate collection
        inventory::submit!(&#reg_ident);
    };

    TokenStream::from(expanded)
}

// ─── Encoder Registration ───────────────────────────────────────────────────

struct RegisterEncoderArgs {
    name: String,
    format: String,
    mime: String,
}

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

/// Register a function as a rasmcore encoder.
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
                name: #name,
                format: #format,
                mime: #mime,
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

            if input.peek(Token![,]) { let _: Token![,] = input.parse()?; }
        }

        Ok(RegisterDecoderArgs { name, formats })
    }
}

/// Register a function as a rasmcore decoder.
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
                name: #name,
                formats: #formats,
                fn_name: stringify!(#fn_name),
            };
        inventory::submit!(&#reg_ident);
    };
    TokenStream::from(expanded)
}
