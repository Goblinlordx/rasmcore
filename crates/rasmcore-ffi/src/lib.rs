#![allow(private_interfaces)]

//! rasmcore-ffi — C shared library wrapping the rasmcore pipeline API.
//!
//! Produces a `.so` / `.dylib` / `.dll` with a flat C ABI. Every `extern "C"`
//! function is wrapped in `catch_unwind` so panics never cross the FFI boundary.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::panic;
use std::rc::Rc;

// ---------------------------------------------------------------------------
// Thread-local error storage
// ---------------------------------------------------------------------------

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap());
}

fn set_error(msg: &str) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() =
            CString::new(msg).unwrap_or_else(|_| CString::new("error with null byte").unwrap());
    });
}

fn catch_err<F, T>(default: T, f: F) -> T
where
    F: FnOnce() -> Result<T, String>,
{
    match panic::catch_unwind(panic::AssertUnwindSafe(f)) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            set_error(&e);
            default
        }
        Err(_) => {
            set_error("panic in rasmcore");
            default
        }
    }
}

/// Return a pointer to the last error message (NUL-terminated, UTF-8).
///
/// The pointer is valid until the next call to any `rasmcore_*` function on
/// the same thread.
#[no_mangle]
pub extern "C" fn rasmcore_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

// ---------------------------------------------------------------------------
// Opaque pipeline handle
// ---------------------------------------------------------------------------

struct PipelineState {
    graph: rasmcore_image::domain::pipeline::graph::NodeGraph,
    layer_cache: Option<Rc<RefCell<rasmcore_pipeline::LayerCache>>>,
}

/// Create a new pipeline with `cache_budget_mb` megabytes of spatial cache.
#[no_mangle]
pub extern "C" fn rasmcore_pipeline_new(cache_budget_mb: u32) -> *mut PipelineState {
    catch_err(std::ptr::null_mut(), || {
        let budget = cache_budget_mb as usize * 1024 * 1024;
        let state = PipelineState {
            graph: rasmcore_image::domain::pipeline::graph::NodeGraph::new(budget),
            layer_cache: None,
        };
        Ok(Box::into_raw(Box::new(state)))
    })
}

/// Free a pipeline handle. Safe to call with NULL.
#[no_mangle]
pub extern "C" fn rasmcore_pipeline_free(pipe: *mut PipelineState) {
    if !pipe.is_null() {
        unsafe {
            drop(Box::from_raw(pipe));
        }
    }
}

// ---------------------------------------------------------------------------
// Layer cache (cross-pipeline persistence)
// ---------------------------------------------------------------------------

type LayerCacheHandle = Rc<RefCell<rasmcore_pipeline::LayerCache>>;

/// Create a shared layer cache with `budget_mb` megabytes.
#[no_mangle]
pub extern "C" fn rasmcore_layer_cache_new(budget_mb: u32) -> *mut LayerCacheHandle {
    catch_err(std::ptr::null_mut(), || {
        let budget = budget_mb as usize * 1024 * 1024;
        let lc = Rc::new(RefCell::new(rasmcore_pipeline::LayerCache::new(budget)));
        Ok(Box::into_raw(Box::new(lc)))
    })
}

/// Free a layer cache handle. Safe to call with NULL.
#[no_mangle]
pub extern "C" fn rasmcore_layer_cache_free(cache: *mut LayerCacheHandle) {
    if !cache.is_null() {
        unsafe {
            drop(Box::from_raw(cache));
        }
    }
}

/// Attach a shared layer cache to a pipeline. The pipeline's node graph is
/// recreated with the cache, using a 4 MB default spatial cache budget.
#[no_mangle]
pub extern "C" fn rasmcore_pipeline_set_cache(
    pipe: *mut PipelineState,
    cache: *mut LayerCacheHandle,
) {
    if pipe.is_null() || cache.is_null() {
        return;
    }
    let pipe = unsafe { &mut *pipe };
    let cache = unsafe { &*cache };
    pipe.layer_cache = Some(Rc::clone(cache));
    pipe.graph = rasmcore_image::domain::pipeline::graph::NodeGraph::with_layer_cache(
        4 * 1024 * 1024,
        Rc::clone(cache),
    );
}

// ---------------------------------------------------------------------------
// Source reading
// ---------------------------------------------------------------------------

/// Decode an image from an in-memory buffer and return a node id.
///
/// Returns `UINT32_MAX` on error — call `rasmcore_last_error()` for details.
#[no_mangle]
pub extern "C" fn rasmcore_read(pipe: *mut PipelineState, data: *const u8, len: usize) -> u32 {
    catch_err(u32::MAX, || {
        if pipe.is_null() || data.is_null() {
            return Err("null pointer".into());
        }
        let pipe = unsafe { &mut *pipe };
        let bytes = unsafe { std::slice::from_raw_parts(data, len) };
        let source_hash = rasmcore_pipeline::compute_source_hash(bytes);
        let node =
            rasmcore_image::domain::pipeline::nodes::source::SourceNode::new(bytes.to_vec())
                .map_err(|e| e.to_string())?;
        Ok(pipe.graph.add_node_with_hash(Box::new(node), source_hash))
    })
}

/// Decode an image from a file path and return a node id.
///
/// Returns `UINT32_MAX` on error.
#[no_mangle]
pub extern "C" fn rasmcore_read_file(pipe: *mut PipelineState, path: *const c_char) -> u32 {
    catch_err(u32::MAX, || {
        if pipe.is_null() || path.is_null() {
            return Err("null pointer".into());
        }
        let pipe = unsafe { &mut *pipe };
        let path = unsafe { CStr::from_ptr(path) }
            .to_str()
            .map_err(|e| e.to_string())?;
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        let source_hash = rasmcore_pipeline::compute_source_hash(&bytes);
        let node = rasmcore_image::domain::pipeline::nodes::source::SourceNode::new(bytes)
            .map_err(|e| e.to_string())?;
        Ok(pipe.graph.add_node_with_hash(Box::new(node), source_hash))
    })
}

// ---------------------------------------------------------------------------
// Filter dispatch
// ---------------------------------------------------------------------------

/// Apply a filter to a source node, returning the new node id.
///
/// `name` is the filter name (e.g. "blur", "brightness", "sepia").
/// `params_json` is a JSON object whose keys are parameter names and values
/// are strings, e.g. `{"radius": "5", "sigma": "1.5"}`. Pass NULL or `"{}"`
/// for defaults.
///
/// Returns `UINT32_MAX` on error.
#[no_mangle]
pub extern "C" fn rasmcore_filter(
    pipe: *mut PipelineState,
    source: u32,
    name: *const c_char,
    params_json: *const c_char,
) -> u32 {
    catch_err(u32::MAX, || {
        if pipe.is_null() || name.is_null() {
            return Err("null pointer".into());
        }
        let pipe = unsafe { &mut *pipe };
        let name = unsafe { CStr::from_ptr(name) }
            .to_str()
            .map_err(|e| e.to_string())?;

        // Parse JSON params into HashMap<String, String>
        let params: HashMap<String, String> = if params_json.is_null() {
            HashMap::new()
        } else {
            let json_str = unsafe { CStr::from_ptr(params_json) }
                .to_str()
                .map_err(|e| e.to_string())?;
            if json_str.is_empty() || json_str == "{}" {
                HashMap::new()
            } else {
                // Parse JSON object — values are coerced to strings
                let val: serde_json::Value =
                    serde_json::from_str(json_str).map_err(|e| e.to_string())?;
                match val {
                    serde_json::Value::Object(map) => map
                        .into_iter()
                        .map(|(k, v)| {
                            let s = match &v {
                                serde_json::Value::String(s) => s.clone(),
                                other => other.to_string(),
                            };
                            (k, s)
                        })
                        .collect(),
                    _ => return Err("params_json must be a JSON object".into()),
                }
            }
        };

        let info = pipe.graph.node_info(source).map_err(|e| e.to_string())?;
        let (node, gpu) =
            rasmcore_image::domain::pipeline::dispatch::dispatch_filter(name, source, info, &params)
                .map_err(|e| e.to_string())?;
        let id = pipe.graph.add_node(node);
        if let Some(gpu_node) = gpu {
            pipe.graph.register_gpu(id, gpu_node);
        }
        Ok(id)
    })
}

// ---------------------------------------------------------------------------
// Write / encode output
// ---------------------------------------------------------------------------

/// Encode the result of `node` into `format` (e.g. "png", "jpeg", "webp").
///
/// `quality` is 1-100 for lossy formats; pass 0 for default.
/// On success, returns a heap-allocated buffer and writes its length to
/// `*out_len`. Free the buffer with `rasmcore_buffer_free()`.
///
/// Returns NULL on error.
#[no_mangle]
pub extern "C" fn rasmcore_write(
    pipe: *mut PipelineState,
    node: u32,
    format: *const c_char,
    quality: u32,
    out_len: *mut usize,
) -> *mut u8 {
    catch_err(std::ptr::null_mut(), || {
        if pipe.is_null() || format.is_null() {
            return Err("null pointer".into());
        }
        let pipe = unsafe { &mut *pipe };
        let format = unsafe { CStr::from_ptr(format) }
            .to_str()
            .map_err(|e| e.to_string())?;
        let quality = if quality > 0 {
            Some(quality as u8)
        } else {
            None
        };

        let output = rasmcore_image::domain::pipeline::nodes::sink::write(
            &mut pipe.graph,
            node,
            format,
            quality,
            None,
        )
        .map_err(|e| e.to_string())?;

        pipe.graph.finalize_layer_cache();

        if !out_len.is_null() {
            unsafe {
                *out_len = output.len();
            }
        }
        let mut buf = output.into_boxed_slice();
        let ptr = buf.as_mut_ptr();
        std::mem::forget(buf);
        Ok(ptr)
    })
}

/// Encode and write the result of `node` to a file.
///
/// Format is inferred from the file extension. `quality` is 1-100; pass 0
/// for default. Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn rasmcore_write_file(
    pipe: *mut PipelineState,
    node: u32,
    path: *const c_char,
    quality: u32,
) -> i32 {
    catch_err(-1, || {
        if pipe.is_null() || path.is_null() {
            return Err("null pointer".into());
        }
        let pipe = unsafe { &mut *pipe };
        let path_str = unsafe { CStr::from_ptr(path) }
            .to_str()
            .map_err(|e| e.to_string())?;
        let format = std::path::Path::new(path_str)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("png");
        let quality = if quality > 0 {
            Some(quality as u8)
        } else {
            None
        };

        let output = rasmcore_image::domain::pipeline::nodes::sink::write(
            &mut pipe.graph,
            node,
            format,
            quality,
            None,
        )
        .map_err(|e| e.to_string())?;

        pipe.graph.finalize_layer_cache();
        std::fs::write(path_str, &output).map_err(|e| e.to_string())?;
        Ok(0)
    })
}

/// Free a buffer previously returned by `rasmcore_write()`.
#[no_mangle]
pub extern "C" fn rasmcore_buffer_free(buf: *mut u8, len: usize) {
    if !buf.is_null() && len > 0 {
        unsafe {
            drop(Vec::from_raw_parts(buf, len, len));
        }
    }
}

// ---------------------------------------------------------------------------
// Info
// ---------------------------------------------------------------------------

/// Return a JSON string describing the image at `node`.
///
/// The pointer is valid until the next `rasmcore_*` call on the same thread.
/// Returns NULL on error.
#[no_mangle]
pub extern "C" fn rasmcore_node_info_json(pipe: *mut PipelineState, node: u32) -> *const c_char {
    catch_err(std::ptr::null(), || {
        if pipe.is_null() {
            return Err("null pointer".into());
        }
        let pipe = unsafe { &*pipe };
        let info = pipe.graph.node_info(node).map_err(|e| e.to_string())?;
        let json = format!(
            r#"{{"width":{},"height":{},"format":"{:?}","colorSpace":"{:?}"}}"#,
            info.width, info.height, info.format, info.color_space
        );
        let cstr = CString::new(json).map_err(|e| e.to_string())?;
        // Store in thread-local to keep the pointer alive until next call
        LAST_ERROR.with(|e| {
            *e.borrow_mut() = cstr;
        });
        Ok(LAST_ERROR.with(|e| e.borrow().as_ptr()))
    })
}
