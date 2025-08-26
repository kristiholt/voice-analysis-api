import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5000,
    allowedHosts: [
      '8aaa31d6-e602-4c2b-8fe1-d8abf2a761d4-00-4jcm7e7278pt.kirk.replit.dev',
      '.replit.dev',
      'localhost'
    ],
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:80',
        changeOrigin: true,
        secure: false
      },
      '/health': {
        target: 'http://127.0.0.1:80',
        changeOrigin: true,
        secure: false
      },
      '/analyze': {
        target: 'http://127.0.0.1:80',
        changeOrigin: true,
        secure: false
      },
      '/v1': {
        target: 'http://127.0.0.1:80',
        changeOrigin: true,
        secure: false
      }
    }
  },
  build: {
    // Exclude large files from build
    rollupOptions: {
      external: ['data/**/*'],
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom']
        }
      }
    },
    // Optimize build for faster processing
    target: 'esnext',
    minify: 'esbuild',
    // Increase chunk size warning limit
    chunkSizeWarningLimit: 1600
  },
  resolve: {
    alias: {
      '@': '/src',
    },
  },
})