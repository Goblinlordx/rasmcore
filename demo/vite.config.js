import { defineConfig } from 'vite';

export default defineConfig({
  // Serve .wasm files with correct MIME type
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  // Allow importing from sdk/ directory
  resolve: {
    alias: {
      '@sdk': '/sdk',
    },
  },
  // Ensure .wasm files are served correctly
  assetsInclude: ['**/*.wasm'],
});
