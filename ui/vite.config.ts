/// <reference types="vitest/config" />
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Интерфейс открывается в pywebview из локальных файлов, поэтому пути относительные.
export default defineConfig({
  base: './',
  plugins: [react(), tailwindcss()],
  build: { outDir: 'dist', emptyOutDir: true, target: 'es2022' },
  test: { environment: 'jsdom', include: ['src/**/*.test.{ts,tsx}'] },
})
