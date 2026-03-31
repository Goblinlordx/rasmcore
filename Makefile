# rasmcore project Makefile
# Provides simple targets for web-ui and common operations.

.PHONY: web-ui-build web-ui-serve web-ui-check

## Build the web-ui package (TypeScript + Vite)
web-ui-build:
	./scripts/web-ui-build.sh

## Start the web-ui dev server
web-ui-serve:
	./scripts/web-ui-serve.sh

## Run TypeScript type check + ESLint on web-ui
web-ui-check:
	cd web-ui && npm install --silent && npx tsc --noEmit && npx eslint src/ --ext .ts
