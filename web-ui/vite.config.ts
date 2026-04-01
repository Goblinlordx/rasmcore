import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  resolve: {
    alias: {
      '@sdk': '/sdk',
    },
  },
  assetsInclude: ['**/*.wasm'],
  define: {
    __SDK_PATH__: JSON.stringify(process.env.VITE_SDK_PATH || './sdk'),
  },
});
