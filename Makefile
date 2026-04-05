# rasmcore project Makefile
# Delegates to subdirectories — no Node.js deps at root.

.PHONY: docs docs-serve docs-dev docs-clean \
        web-ui web-ui-build web-ui-serve web-ui-check \
        sdk build-all

# ─── Sentinel files for incremental builds ──────────────────────────────────

DOCS_SITE         := docs/site
DOCS_NODE_MODULES := $(DOCS_SITE)/node_modules/.package-lock.json
DOCS_REGISTRY     := /tmp/v2_registry_docs.json
DOCS_EXAMPLES     := /tmp/docs-examples/reference.png
DOCS_SDK          := $(DOCS_SITE)/public/sdk/v2/rasmcore-v2-image.js
DOCS_OUT          := docs/out/index.html

# ─── Docs ────────────────────────────────────────────────────────────────────

## Install docs deps (only if node_modules is stale)
$(DOCS_NODE_MODULES): $(DOCS_SITE)/package.json
	cd $(DOCS_SITE) && npm install --silent

## Dump registry JSON (rebuild if any pipeline-v2 or codecs-v2 source changed)
V2_SOURCES := $(shell find crates/rasmcore-pipeline-v2/src -name '*.rs') \
              $(shell find crates/rasmcore-codecs-v2/src -name '*.rs' 2>/dev/null)
$(DOCS_REGISTRY): $(V2_SOURCES)
	@rm -f $@
	cargo run --bin dump_registry -p rasmcore-v2-wasm > $@
	@test -s $@ || (echo "ERROR: dump_registry produced empty output" && rm -f $@ && exit 1)

## Render example images (rebuild if any V2 source changed)
$(DOCS_EXAMPLES): $(V2_SOURCES)
	cargo run --bin render_examples -p rasmcore-v2-wasm --release

## Copy SDK for live playground
$(DOCS_SDK): sdk/typescript/v2-generated/rasmcore-v2-image.js $(DOCS_NODE_MODULES)
	mkdir -p $(DOCS_SITE)/public/sdk/v2
	cp sdk/typescript/v2-generated/rasmcore-v2-image.js $(DOCS_SITE)/public/sdk/v2/
	cp sdk/typescript/v2-generated/rasmcore-v2-image.d.ts $(DOCS_SITE)/public/sdk/v2/ 2>/dev/null || true
	cp sdk/typescript/v2-generated/*.wasm $(DOCS_SITE)/public/sdk/v2/
	cp -r sdk/typescript/v2-generated/interfaces $(DOCS_SITE)/public/sdk/v2/ 2>/dev/null || true
	mkdir -p $(DOCS_SITE)/public/sdk/v2/preview2-shim
	cp $(DOCS_SITE)/node_modules/@bytecodealliance/preview2-shim/lib/browser/*.js $(DOCS_SITE)/public/sdk/v2/preview2-shim/ 2>/dev/null || true
	@# Rewrite bare @bytecodealliance/preview2-shim/* to absolute paths for Web Worker compat
	sed -i '' "s|from '@bytecodealliance/preview2-shim/|from '/sdk/v2/preview2-shim/|g" $(DOCS_SITE)/public/sdk/v2/rasmcore-v2-image.js

## Copy example images to public
.PHONY: docs-copy-examples
docs-copy-examples: $(DOCS_EXAMPLES)
	mkdir -p $(DOCS_SITE)/public/assets/examples
	cp /tmp/docs-examples/*.png $(DOCS_SITE)/public/assets/examples/ 2>/dev/null || true

## Build documentation site (incremental — only rebuilds what changed)
docs: $(DOCS_NODE_MODULES) $(DOCS_REGISTRY) docs-copy-examples $(DOCS_SDK)
	@rm -rf docs/out  # clean stale output from old distDir config
	cd $(DOCS_SITE) && npm run build

## Build and serve docs site
docs-serve: docs
	@echo ""
	@echo "Serving docs at http://localhost:4000"
	cd $(DOCS_SITE) && npx serve out -l 4000

## Start docs dev server (hot reload, no static export)
docs-dev: $(DOCS_NODE_MODULES) $(DOCS_REGISTRY) docs-copy-examples $(DOCS_SDK)
	cd $(DOCS_SITE) && npm run dev

## Clean docs build artifacts (including cached registry/examples)
docs-clean:
	rm -rf $(DOCS_SITE)/.next $(DOCS_SITE)/out $(DOCS_SITE)/node_modules docs/out
	rm -f $(DOCS_SITE)/next-env.d.ts
	rm -f $(DOCS_REGISTRY)
	rm -rf /tmp/docs-examples

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
