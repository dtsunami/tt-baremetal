import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

// Dev server proxies API + WS to the FastAPI backend (bhtop-web on :8000).
// `npm run build` emits static files to dist/, which FastAPI serves in production.
export default defineConfig({
  plugins: [svelte()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
      '/ws': { target: 'ws://localhost:8000', ws: true },
    },
  },
  build: { outDir: 'dist', emptyOutDir: true },
})
