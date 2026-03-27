//! WASM integration test harness for rasmcore components.
//!
//! Loads compiled WASM components via wasmtime and validates them against
//! ImageMagick reference outputs.

use std::path::{Path, PathBuf};

use wasmtime::component::{Component, Linker};
use wasmtime::{Config, Engine, Store};
use wasmtime_wasi::p2::{IoView, WasiCtx, WasiCtxBuilder, WasiView};

wasmtime::component::bindgen!({
    path: "../../wit/image",
    world: "image-processor",
});

/// WASI host state for the component store.
pub struct HostState {
    ctx: WasiCtx,
    table: wasmtime::component::ResourceTable,
}

impl IoView for HostState {
    fn table(&mut self) -> &mut wasmtime::component::ResourceTable {
        &mut self.table
    }
}

impl WasiView for HostState {
    fn ctx(&mut self) -> &mut WasiCtx {
        &mut self.ctx
    }
}

/// Instantiate the rasmcore-image component and return (store, bindings).
pub fn instantiate_image_component() -> (Store<HostState>, ImageProcessor) {
    let mut config = Config::new();
    config.wasm_component_model(true);
    let engine = Engine::new(&config).expect("failed to create wasmtime engine");

    let wasm_path = find_component_wasm();
    let component =
        Component::from_file(&engine, &wasm_path).expect("failed to load WASM component");

    let mut linker = Linker::<HostState>::new(&engine);
    wasmtime_wasi::p2::add_to_linker_sync(&mut linker).expect("failed to add WASI to linker");

    let wasi_ctx = WasiCtxBuilder::new().build();
    let state = HostState {
        ctx: wasi_ctx,
        table: wasmtime::component::ResourceTable::new(),
    };
    let mut store = Store::new(&engine, state);

    let bindings = ImageProcessor::instantiate(&mut store, &component, &linker)
        .expect("failed to instantiate component");

    (store, bindings)
}

/// Locate the built rasmcore_image.wasm component.
fn find_component_wasm() -> PathBuf {
    let project_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let candidates = [
        project_root.join("target/wasm32-wasip1/debug/rasmcore_image.wasm"),
        project_root.join("target/wasm32-wasip1/release/rasmcore_image.wasm"),
        project_root.join("target/wasm32-wasip2/debug/rasmcore_image.wasm"),
        project_root.join("target/wasm32-wasip2/release/rasmcore_image.wasm"),
    ];
    for path in &candidates {
        if path.exists() {
            return path.clone();
        }
    }
    panic!(
        "rasmcore_image.wasm not found. Run `cargo component build -p rasmcore-image` first.\nSearched: {:?}",
        candidates
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
    );
}

// ─── Fixture utilities ───

/// Root directory for test fixtures.
pub fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/generated")
}

/// Load a fixture input image.
pub fn load_fixture(name: &str) -> Vec<u8> {
    let path = fixtures_dir().join("inputs").join(name);
    std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "Failed to read fixture {}: {}. Run tests/fixtures/generate.sh first.",
            path.display(),
            e
        )
    })
}

/// Load an ImageMagick reference output.
pub fn load_reference(name: &str) -> Vec<u8> {
    let path = fixtures_dir().join("reference").join(name);
    std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "Failed to read reference {}: {}. Run tests/fixtures/generate.sh first.",
            path.display(),
            e
        )
    })
}

// ─── Comparison utilities ───

/// Mean absolute error between two pixel buffers.
pub fn mean_absolute_error(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "pixel buffer length mismatch");
    if a.is_empty() {
        return 0.0;
    }
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .sum();
    sum / a.len() as f64
}

/// Peak signal-to-noise ratio between two pixel buffers.
pub fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mse: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff = x as f64 - y as f64;
            diff * diff
        })
        .sum::<f64>()
        / a.len() as f64;
    if mse == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0_f64 * 255.0 / mse).log10()
}
