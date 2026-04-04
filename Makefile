# rasmcore project Makefile
# Delegates to subdirectories — no Node.js deps at root.

.PHONY: docs web-ui sdk build-all web-ui-build web-ui-serve web-ui-check

# ─── Docs ────────────────────────────────────────────────────────────────────

## Build documentation site (render examples + Next.js static export)
docs:
	cargo run --bin render_examples -p rasmcore-v2-wasm --release 2>/dev/null
	cargo run --bin dump_registry -p rasmcore-v2-wasm 2>/dev/null > /tmp/v2_registry_docs.json
	cp -r /tmp/docs-examples/* docs/site/public/assets/examples/ 2>/dev/null || true
	cd docs/site && npm install --silent && npm run build

## Start docs dev server
docs-dev:
	cd docs/site && npm install --silent && npm run dev

# ─── Web UI ──────────────────────────────────────────────────────────────────

## Build the web-ui package (V2 WASM + SDK + Vite)
web-ui: web-ui-build

web-ui-build:
	./scripts/build-v2-webui.sh

## Start the web-ui dev server
web-ui-serve:
	./scripts/web-ui-serve.sh

## Run TypeScript type check + ESLint on web-ui
web-ui-check:
	cd web-ui && npm install --silent && npx tsc --noEmit && npx eslint src/ --ext .ts

# ─── SDK ─────────────────────────────────────────────────────────────────────

## Generate V2 SDK (WASM component + jco transpile + fluent SDK)
sdk:
	./scripts/generate-v2-sdk.sh
	node scripts/generate-v2-fluent-sdk.mjs

# ─── All ─────────────────────────────────────────────────────────────────────

## Build everything
build-all: sdk web-ui docs
