//! Dump V2 operation registry as JSON for SDK generation and docs.
#[allow(unused_imports)]
use rasmcore_pipeline_v2::filters as _f;
#[allow(unused_imports)]
use rasmcore_codecs_v2 as _c;

use rasmcore_pipeline_v2 as v2;

fn param_type_str(t: v2::ParamType) -> &'static str {
    match t {
        v2::ParamType::F32 => "f32", v2::ParamType::F64 => "f64",
        v2::ParamType::U32 => "u32", v2::ParamType::I32 => "i32",
        v2::ParamType::Bool => "bool", v2::ParamType::String => "string",
        v2::ParamType::Rect => "rect",
        v2::ParamType::NodeRef => "node_ref",
        v2::ParamType::FontRef => "font_ref",
        v2::ParamType::LutRef => "lut_ref",
    }
}

fn opt_f64(v: Option<f64>) -> String {
    v.map(|v| v.to_string()).unwrap_or("null".into())
}

fn opt_str(v: Option<&str>) -> String {
    v.map(|s| format!("\"{}\"", s)).unwrap_or("null".into())
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

fn main() {
    let filters = v2::registry::registered_filter_registrations();
    let encoders = v2::registered_encoders();
    let decoders = v2::registered_decoders();

    print!("{{\"filters\":[");
    for (i, f) in filters.iter().enumerate() {
        if i > 0 { print!(","); }
        print!("{{\"name\":\"{}\",\"displayName\":\"{}\",\"category\":\"{}\",\"docPath\":\"{}\",\"cost\":\"{}\",\"params\":[",
            f.name, f.display_name, f.category, json_escape(f.doc_path), json_escape(f.cost));
        for (j, p) in f.params.iter().enumerate() {
            if j > 0 { print!(","); }
            print!("{{\"name\":\"{}\",\"type\":\"{}\",\"min\":{},\"max\":{},\"step\":{},\"default\":{},\"hint\":{},\"description\":\"{}\"}}",
                p.name, param_type_str(p.value_type),
                opt_f64(p.min), opt_f64(p.max), opt_f64(p.step), opt_f64(p.default),
                opt_str(p.hint), json_escape(p.description));
        }
        print!("]}}");
    }
    print!("],\"encoders\":[");
    for (i, e) in encoders.iter().enumerate() {
        if i > 0 { print!(","); }
        print!("{{\"name\":\"{}\",\"displayName\":\"{}\",\"mime\":\"{}\",\"docPath\":\"{}\",\"extensions\":[",
            e.name, e.display_name, e.mime, json_escape(e.doc_path));
        for (j, ext) in e.extensions.iter().enumerate() {
            if j > 0 { print!(","); }
            print!("\"{}\"", ext);
        }
        print!("]}}");
    }
    print!("],\"decoders\":[");
    for (i, d) in decoders.iter().enumerate() {
        if i > 0 { print!(","); }
        print!("{{\"name\":\"{}\",\"displayName\":\"{}\",\"docPath\":\"{}\",\"extensions\":[",
            d.name, d.display_name, json_escape(d.doc_path));
        for (j, ext) in d.extensions.iter().enumerate() {
            if j > 0 { print!(","); }
            print!("\"{}\"", ext);
        }
        print!("]}}");
    }
    println!("]}}");
}
