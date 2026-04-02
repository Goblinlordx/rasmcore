//! name: vintage_film
//! category: grading
//! strategy: composite
//! param grain: f32 = 0.3 [0.0, 1.0]
//! param vignette_str: f32 = 0.4 [0.0, 1.0]
//! param warmth: f32 = 0.2 [0.0, 1.0]

// Composite strategy: builds a lazy pipeline graph via .apply() chains.
// Filter names are resolved at graph build time against built-in + script registries.
// The composed graph gets full pipeline optimizations (LUT fusion, GPU dispatch, tiling).

fn graph(input, params) {
    input
        .apply("contrast", #{ amount: 0.15 })
        .apply("warm_tint", #{ warmth: params.warmth })
        .apply("film_grain", #{ amount: params.grain, size: 1.5 })
        .apply("vignette", #{ strength: params.vignette_str })
}
