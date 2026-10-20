import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const backend = `http://127.0.0.1:${process.env.AI_STUDIO_PORT || 8765}`

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    proxy: {
      '/api': { target: backend },
      '/ws': { target: backend, ws: true },
    },
  },
  build: { chunkSizeWarningLimit: 2000 },
})
