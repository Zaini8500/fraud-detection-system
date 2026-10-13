import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// In local dev, /api is proxied to the FastAPI backend (uvicorn on :8000).
// In production on Vercel, /api is served by the Python serverless functions
// in api/, so no proxy is needed -- same-origin requests just work.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        // FastAPI serves /predict directly; strip the /api prefix locally.
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
});
