import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The daemon owns everything; the app is a thin client of localhost:4310.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { '/v1': 'http://127.0.0.1:4310' },
  },
})
