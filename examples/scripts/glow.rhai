//! name: glow
//! category: effect
//! strategy: composite
//! param radius: f32 = 5.0 [0.1, 50.0]
//! param intensity: f32 = 0.5 [0.0, 1.0]

// Composite strategy: builds a lazy pipeline DAG with multi-input blend.
// 'input' is a NodeRef — passing it as a config value to blend creates
// a second upstream connection. The engine resolves the full graph with
// pipeline optimizations (tiling, caching).

fn graph(input, params) {
    let blurred = input.apply("blur", #{ radius: params.radius * 3.0 });
    blurred.apply("blend", #{ source: input, mode: "screen", opacity: params.intensity })
}
