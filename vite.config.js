import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import studioBackend from './server/studioBackend.js'

// https://vite.dev/config/
export default defineConfig({
  // studioBackend serves /api/* (Ollama models, toolkit install, live job streams)
  plugins: [react(), studioBackend()],
})
