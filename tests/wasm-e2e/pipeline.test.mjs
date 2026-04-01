/**
 * WASM pipeline end-to-end tests.
 *
 * Tests the full path: JS → WIT bindings → WASM → Rust pipeline → encode → JS.
 * Runs via Node.js (no browser needed).
 *
 * Run with:
 *   npm run demo:build && node --test tests/wasm-e2e/pipeline.test.mjs
 */

import { describe, it, before } from "node:test";
import assert from "node:assert/strict";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { performance } from "node:perf_hooks";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = join(__dirname, "../..");
const SDK_PATH = join(PROJECT_ROOT, "demo/sdk/rasmcore-image.js");

const sdk = await import(SDK_PATH);
const { pipeline, encoder } = sdk;
const Pipeline = pipeline.ImagePipeline;
const LayerCache = pipeline.LayerCache;

// ─── Test image ──────────────────────────────────────────────────────────────

let testPng;

function createTestPng() {
  const w = 64, h = 64;
  const pixels = new Uint8Array(w * h * 3);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 3;
      pixels[i]     = Math.floor(x * 255 / w);
      pixels[i + 1] = Math.floor(y * 255 / h);
      pixels[i + 2] = 128;
    }
  }
  return encoder.encode(pixels, {
    width: w, height: h, format: "rgb8", colorSpace: "srgb",
  }, "png", undefined);
}

function pngValid(buf) {
  return buf && buf.length > 8 && buf[0] === 0x89 && buf[1] === 0x50;
}

function buffersEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

// ─── Basic operations ────────────────────────────────────────────────────────

describe("Pipeline basic operations", () => {
  before(() => { testPng = createTestPng(); });

  it("creates valid test PNG", () => {
    assert.ok(pngValid(testPng), "should produce valid PNG");
    assert.ok(testPng.length > 100);
  });

  it("read → write roundtrip", () => {
    const p = new Pipeline();
    const out = p.writePng(p.read(testPng), {}, undefined);
    assert.ok(pngValid(out));
  });

  it("brightness", () => {
    const p = new Pipeline();
    const src = p.read(testPng);
    const out = p.writePng(p.brightness(src, { amount: 0.2 }), {}, undefined);
    assert.ok(pngValid(out));
    assert.ok(!buffersEqual(out, p.writePng(src, {}, undefined)));
  });

  it("invert", () => {
    const p = new Pipeline();
    const src = p.read(testPng);
    const out = p.writePng(p.invert(src), {}, undefined);
    assert.ok(pngValid(out));
  });

  it("sepia", () => {
    const p = new Pipeline();
    const out = p.writePng(p.sepia(p.read(testPng), { intensity: 0.8 }), {}, undefined);
    assert.ok(pngValid(out));
  });

  it("hueRotate", () => {
    const p = new Pipeline();
    const out = p.writePng(p.hueRotate(p.read(testPng), { degrees: 90 }), {}, undefined);
    assert.ok(pngValid(out));
  });

  it("blur", () => {
    const p = new Pipeline();
    const out = p.writePng(p.blur(p.read(testPng), { radius: 2.0 }), {}, undefined);
    assert.ok(pngValid(out));
  });
});

// ─── Chaining ────────────────────────────────────────────────────────────────

describe("Pipeline chaining", () => {
  before(() => { testPng = createTestPng(); });

  it("brightness → contrast", () => {
    const p = new Pipeline();
    const s = p.read(testPng);
    const out = p.writePng(p.contrast(p.brightness(s, { amount: 0.1 }), { amount: 0.2 }), {}, undefined);
    assert.ok(pngValid(out));
  });

  it("hueRotate → sepia → saturate", () => {
    const p = new Pipeline();
    const s = p.read(testPng);
    const h = p.hueRotate(s, { degrees: 45 });
    const se = p.sepia(h, { intensity: 0.3 });
    const out = p.writePng(p.saturate(se, { factor: 1.5 }), {}, undefined);
    assert.ok(pngValid(out));
  });

  it("blur → brightness", () => {
    const p = new Pipeline();
    const s = p.read(testPng);
    const out = p.writePng(p.brightness(p.blur(s, { radius: 1.0 }), { amount: 0.2 }), {}, undefined);
    assert.ok(pngValid(out));
  });

  it("invert → invert = identity (LUT fusion)", () => {
    const p1 = new Pipeline();
    const s1 = p1.read(testPng);
    const doubled = p1.writePng(p1.invert(p1.invert(s1)), {}, undefined);

    const p2 = new Pipeline();
    const srcOnly = p2.writePng(p2.read(testPng), {}, undefined);

    assert.ok(buffersEqual(doubled, srcOnly), "double invert should equal source");
  });
});

// ─── Layer cache ─────────────────────────────────────────────────────────────

describe("Layer cache across pipelines", () => {
  before(() => { testPng = createTestPng(); });

  it("identical pipeline reuse — byte-identical output", () => {
    const lc = new LayerCache(64);

    const run = () => {
      const p = new Pipeline();
      p.setLayerCache(lc);
      const s = p.read(testPng);
      return p.writePng(p.brightness(s, { amount: 0.2 }), {}, undefined);
    };

    const t0 = performance.now();
    const out1 = run();
    const time1 = performance.now() - t0;

    const t1 = performance.now();
    const out2 = run();
    const time2 = performance.now() - t1;

    assert.ok(buffersEqual(out1, out2), "same pipeline must produce identical output");
    console.log(`  run1=${time1.toFixed(1)}ms, run2=${time2.toFixed(1)}ms`);
  });

  it("add node — output changes, prefix reused", () => {
    const lc = new LayerCache(64);

    const p1 = new Pipeline();
    p1.setLayerCache(lc);
    const s1 = p1.read(testPng);
    const out1 = p1.writePng(p1.brightness(s1, { amount: 0.2 }), {}, undefined);

    const p2 = new Pipeline();
    p2.setLayerCache(lc);
    const s2 = p2.read(testPng);
    const b2 = p2.brightness(s2, { amount: 0.2 });
    const out2 = p2.writePng(p2.contrast(b2, { amount: 0.3 }), {}, undefined);

    assert.ok(!buffersEqual(out1, out2), "adding contrast should change output");
  });

  it("remove last node — matches fresh uncached", () => {
    const lc = new LayerCache(64);

    // Run with 2 ops
    const p1 = new Pipeline();
    p1.setLayerCache(lc);
    const s1 = p1.read(testPng);
    p1.writePng(p1.contrast(p1.brightness(s1, { amount: 0.2 }), { amount: 0.3 }), {}, undefined);

    // Run with 1 op (removed contrast), using same cache
    const p2 = new Pipeline();
    p2.setLayerCache(lc);
    const s2 = p2.read(testPng);
    const out2 = p2.writePng(p2.brightness(s2, { amount: 0.2 }), {}, undefined);

    // Fresh uncached reference
    const p3 = new Pipeline();
    const s3 = p3.read(testPng);
    const outFresh = p3.writePng(p3.brightness(s3, { amount: 0.2 }), {}, undefined);

    assert.ok(buffersEqual(out2, outFresh), "cached after removal must match fresh");
  });

  it("change params — output differs", () => {
    const lc = new LayerCache(64);

    const run = (amount) => {
      const p = new Pipeline();
      p.setLayerCache(lc);
      const s = p.read(testPng);
      return p.writePng(p.brightness(s, { amount }), {}, undefined);
    };

    const out1 = run(0.2);
    const out2 = run(0.5);
    assert.ok(!buffersEqual(out1, out2), "different params must produce different output");
  });

  it("fused chain — cached matches uncached", () => {
    const pNc = new Pipeline();
    const sNc = pNc.read(testPng);
    const outNc = pNc.writePng(pNc.contrast(pNc.brightness(sNc, { amount: 0.1 }), { amount: 0.2 }), {}, undefined);

    const lc = new LayerCache(64);
    const p1 = new Pipeline();
    p1.setLayerCache(lc);
    const s1 = p1.read(testPng);
    const outCached = p1.writePng(p1.contrast(p1.brightness(s1, { amount: 0.1 }), { amount: 0.2 }), {}, undefined);

    assert.ok(buffersEqual(outNc, outCached), "fused cached must match uncached");
  });

  it("cache stats reflect usage", () => {
    const lc = new LayerCache(64);

    const run = () => {
      const p = new Pipeline();
      p.setLayerCache(lc);
      const s = p.read(testPng);
      return p.writePng(p.brightness(s, { amount: 0.2 }), {}, undefined);
    };

    run();
    const stats1 = lc.stats();
    assert.ok(Number(stats1.entries) > 0, "entries should be > 0");

    run();
    const stats2 = lc.stats();
    assert.ok(Number(stats2.hits) > Number(stats1.hits), "hits should increase");
    console.log(`  entries=${stats2.entries}, hits=${stats2.hits}`);
  });
});
