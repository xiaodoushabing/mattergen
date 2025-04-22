import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import tailwindcss from '@tailwindcss/vite'

// // https://vite.dev/config/
// export default defineConfig({
//   plugins: [react()],
// })

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: '0.0.0.0', // this makes Vite accessible from outside the container
    port: 5173,      // optional, default is 5173

    // setup proxy only for development
    proxy: {        // configure proxy to backend if backend runs on different port
      '/api': {
        target: 'http://localhost:8000', // your backend server
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})

